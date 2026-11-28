"""Independent preparation and active Chat, with small files and deterministic barriers."""

import ast
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.app.preparation_admission import PreparationAdmission
from ruinedfooocus.core.preparation import PreparationRequest, PreparationService
from ruinedfooocus.features.chat.model_preparation import (
    ChatModelSession,
    ModelTransfer,
    capture_model,
    validate_message,
)

ROOT = Path(__file__).resolve().parents[1]


def callback(name, scope):
    tree = ast.parse((ROOT / "ruinedfooocus/features/chat/ui.py").read_text(encoding="utf-8"))
    create = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_chat"
    )
    node = next(
        node for node in ast.walk(create) if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), "chat/ui.py", "exec"), scope)
    return scope[name]


class Manager:
    def __init__(self, root):
        self.root = root
        self.DOWNLOADABLE_FILES = {
            "llm/A.gguf": {
                "filename": "A.gguf",
                "path": "path_llm",
                "url": "https://fixture/A",
                "size": 7,
            },
        }

    def get_model_folders(self, kind):
        return [self.root]

    def preparation_request(self, key, *, destination=None):
        info = self.DOWNLOADABLE_FILES[key]
        return PreparationRequest(
            info["url"], destination or self.root / info["filename"], info["size"]
        )


class ChatPreparationTests(unittest.TestCase):
    def test_snapshot_retains_paths_settings_and_text_does_not_fetch_projector(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager = Manager(Path(temporary))
            values = {"llama_mmproj": "missing-projector.gguf", "llm_sampling": {"temperature": 0}}
            captured = capture_model("A.gguf", values, manager)
            manager.root = manager.root / "changed"
            values["llm_sampling"]["temperature"] = 1
            self.assertEqual(captured.model, (Path(temporary) / "A.gguf").resolve())
            self.assertEqual(captured.settings["llm_sampling"], {"temperature": 0})
            self.assertIsNone(captured.projector)
            self.assertEqual(len(captured.artifacts), 1)

    def test_blocked_download_does_not_activate_or_reject_active_chat_and_stale_completion_is_fenced(
        self,
    ):
        entered, release = threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def transfer(request, staging, progress, cancelled):
                entered.set()
                if not release.wait(5):
                    raise AssertionError("Barrier was not released")
                progress(7, 0, 0)
                path = staging / "complete"
                path.write_bytes(b"weights")
                return path

            service = PreparationService(transport=transfer, staging_root=root / "staging")
            try:
                manager = Manager(root)

                class Session(ChatModelSession):
                    def start(self, request):
                        return super().start(request, service=service)

                session = Session()
                session.active = str(root / "B.gguf")
                scope = {
                    "capture_model": capture_model,
                    "settings": SimpleNamespace(default_settings={}),
                    "path_manager": manager,
                    "local_models": lambda: {},
                    "time": SimpleNamespace(sleep=lambda _: release.wait(0.01)),
                    "preparation_status": "preparation",
                    "llama_download": "download",
                    "CancelledError": CancelledError,
                    "describe_error": str,
                    "action_updates": lambda value, session=None: {"actions": value},
                    "gr": SimpleNamespace(update=lambda **values: values),
                }
                callback("download_update", scope)
                download = callback("download_model", scope)
                updates = []
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(lambda: updates.extend(download("A.gguf", session)))
                    self.assertTrue(entered.wait(5))
                    # This invokes the actual submission validator with candidate A
                    # still selected; the ready active model owns the conversation.
                    get_text = callback("llama_get_text", {"validate_message": validate_message})
                    self.assertEqual(
                        get_text("Reply using B", False, session.active, "A.gguf"),
                        ("Reply using B", False),
                    )
                    self.assertEqual(session.selected, "A.gguf")
                    sent = []
                    response_scope = {
                        "worker": SimpleNamespace(
                            add_task=lambda data: sent.append(data) or 1,
                            task_result=lambda _: ("results", "answer from B"),
                        ),
                        "image_controls": {},
                        "display_history": lambda history, _: history,
                        "llama_chat": "chat",
                        "llama_history": "history",
                        "llama_model_status": "status",
                    }
                    respond = callback("llama_respond", response_scope)
                    reply = list(
                        respond(
                            "Reply using B",
                            "",
                            "[]",
                            [],
                            session.active,
                            False,
                            False,
                            None,
                            False,
                            False,
                            "New image",
                            None,
                            0.25,
                            False,
                            "standard",
                        )
                    )
                    self.assertEqual(sent[0]["model"], session.active)
                    self.assertEqual(reply[-1]["history"][-1]["content"], "answer from B")
                    session.select("B.gguf")
                    release.set()
                    future.result(timeout=5)
                self.assertEqual(session.active, str(root / "B.gguf"))
                self.assertFalse(any("Downloaded:" in str(update) for update in updates))
                self.assertEqual(updates[-1]["actions"], "B.gguf")
                self.assertTrue((root / "A.gguf").is_file())
            finally:
                release.set()
                service.close()

    def test_cancel_and_hash_failure_preserve_active_and_no_partial_final(self):
        for failure in (False, True):
            with self.subTest(hash_failure=failure), tempfile.TemporaryDirectory() as temporary:
                entered, release = threading.Event(), threading.Event()
                root = Path(temporary)

                def transport(request, staging, progress, cancelled):
                    entered.set()
                    release.wait(5)
                    path = staging / "complete"
                    path.write_bytes(b"weights")
                    return path

                service = PreparationService(transport=transport, staging_root=root / "staging")
                session = ChatModelSession()
                session.active = "B.gguf"
                try:
                    captured = capture_model("A.gguf", {}, Manager(root))
                    if failure:
                        captured = type(captured)(
                            captured.model,
                            (
                                PreparationRequest(
                                    "https://fixture/A", captured.model, sha256="0" * 64
                                ),
                            ),
                            "{}",
                        )
                    _, transfer = session.start(captured, service=service)
                    self.assertTrue(entered.wait(5))
                    if not failure:
                        session.cancel()
                    release.set()
                    with self.assertRaises((CancelledError, IOError)):
                        transfer.result()
                    self.assertEqual(validate_message("Still reply", session.active), "Still reply")
                    self.assertEqual(session.active, "B.gguf")
                finally:
                    release.set()
                    service.close()
                self.assertFalse((root / "A.gguf").exists())

    def test_existing_active_artifact_can_be_verified_with_its_read_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "B.gguf"
            model.write_bytes(b"weights")
            service = PreparationService(staging_root=root / "staging")
            try:
                request = capture_model(str(model), {}, Manager(root))
                with service.artifact_lease(model):
                    transfer = ModelTransfer(request, service=service)
                    self.assertTrue(transfer.done())
                    self.assertEqual(transfer.result(), request)
            finally:
                service.close()

    def test_worker_reuse_checks_the_captured_settings_after_live_settings_change(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/chat/llama_pipeline.py").read_text(encoding="utf-8")
        )
        owner_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pipeline"
        )
        owner_class.body = [
            node
            for node in owner_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("current_model_settings", "process")
        ]
        live = {"llm_n_ctx": 8192}
        scope = {"settings": SimpleNamespace(default_settings=live)}
        exec(
            compile(ast.Module(body=[owner_class], type_ignores=[]), "llama_pipeline.py", "exec"),
            scope,
        )
        owner = scope["pipeline"]()
        owner.runtime_name = "llama.cpp"
        owner.runtime_label = "ROCm"
        owner.fallback_reason = "Vulkan: no visible GPU"
        owner.llm = object()
        owner.loaded_paths = ()
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "B.gguf"
            model.write_bytes(b"weights")
            request = capture_model(str(model), {"llm_n_ctx": 16384}, Manager(model.parent))
            owner.model_settings = owner.current_model_settings(str(model))
            owner.load_base_model = Mock()
            result = owner.process(
                {"load_only": True, "model": str(model), "_chat_preparation": request}
            )
        self.assertTrue(result["ready"])
        self.assertEqual(result["runtime_label"], "ROCm")
        self.assertEqual(result["runtime_name"], "llama.cpp")
        self.assertEqual(result["fallback_reason"], "Vulkan: no visible GPU")
        owner.load_base_model.assert_called_once()
        self.assertEqual(
            owner.load_base_model.call_args.kwargs["prepared"].settings["llm_n_ctx"], 16384
        )
        owner.load_base_model.reset_mock()
        owner.llm = SimpleNamespace(
            sampling_source={"status": "not_found"}, sampling_state=lambda: {}
        )
        owner.model_settings = owner.current_model_settings(str(model), values=request.settings)
        with patch(
            "ruinedfooocus.features.chat.sampling_sources.cached_sampling_source",
            return_value={"status": "reviewed_variant", "profile_id": "language:variant"},
        ):
            owner.process({"load_only": True, "model": str(model), "_chat_preparation": request})
        owner.load_base_model.assert_called_once()

    def test_download_progress_known_and_unknown_lengths(self):
        fn = callback(
            "download_update",
            {
                "gr": SimpleNamespace(update=lambda **values: values),
                "llama_download": "bar",
                "preparation_status": "status",
                "action_updates": lambda *args: {},
            },
        )
        self.assertIn('value="25.0"', fn((25, 100, 1))["bar"]["value"])
        self.assertNotIn('value="', fn((25, 0, 1))["bar"]["value"])

    def test_replacement_releases_only_previous_artifact_leases(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/chat/llama_pipeline.py").read_text(encoding="utf-8")
        )
        owner_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pipeline"
        )
        node = next(
            node
            for node in owner_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "_load_prepared_model"
        )
        scope = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "llama_pipeline.py", "exec"), scope)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("A.gguf", "B.gguf"):
                (root / name).write_bytes(b"weights")
            service = PreparationService(staging_root=root / "staging")
            owner = SimpleNamespace(llm=None, loaded_paths=())

            def load(model, progress=None, *, prepared=None, **options):
                owner.llm = object()
                owner.loaded_paths = (str(prepared.model),)

            owner.load_base_model = load
            try:
                with patch("ruinedfooocus.core.preparation.preparation_service", service):
                    first = capture_model(str(root / "A.gguf"), {}, Manager(root))
                    second = capture_model(str(root / "B.gguf"), {}, Manager(root))
                    scope["_load_prepared_model"](owner, prepared=first)
                    key = service._key(first.model)
                    self.assertEqual(service._artifacts[key].readers, 1)
                    # Reusing A transfers its lease rather than acquiring another
                    # reader; changing to B releases A without releasing B.
                    scope["_load_prepared_model"](owner, prepared=first)
                    self.assertEqual(service._artifacts[key].readers, 1)
                    scope["_load_prepared_model"](owner, prepared=second)
                    self.assertNotIn(key, service._artifacts)
                    self.assertEqual(service._artifacts[service._key(second.model)].readers, 1)
            finally:
                for lease in owner._artifact_leases.values():
                    lease.close()
                service.close()

    def test_preparation_failure_never_enters_compute_and_ready_work_is_independent(self):
        entered, release, ready = threading.Event(), threading.Event(), threading.Event()
        admission = PreparationAdmission(max_workers=2, max_jobs=2)
        compute = Mock()
        errors = []

        def blocked():
            entered.set()
            release.wait(5)
            raise IOError("fixture failure")

        try:
            failed = admission.submit(blocked, compute, errors.append)
            self.assertTrue(entered.wait(5))
            second = admission.submit(lambda: "ready B", lambda value: ready.set(), errors.append)
            self.assertTrue(ready.wait(5))
            second.result(timeout=5)
            release.set()
            failed.result(timeout=5)
            compute.assert_not_called()
            self.assertEqual(str(errors[0]), "fixture failure")
        finally:
            release.set()
            admission.close()


if __name__ == "__main__":
    unittest.main()
