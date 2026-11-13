"""RAG file preparation stays outside the existing txtai compute boundary."""

from concurrent.futures import CancelledError
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from huggingface_hub.errors import LocalEntryNotFoundError

from ruinedfooocus.core.preparation import PreparationRequest
from ruinedfooocus.features.chat import rag_preparation as rag
from ruinedfooocus.app import resource_lifecycle as life
from ruinedfooocus.app.preparation_admission import PreparationAdmission
from ruinedfooocus.features.chat.model_preparation import (
    capture_chat_request,
    prepare_chat_request,
    ModelPreparation,
)
from ui_test_helpers import function_from


class RagPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        snapshot = patch.object(
            rag, "snapshot_download", side_effect=LocalEntryNotFoundError("not cached")
        )
        self.snapshot = snapshot.start()
        self.addCleanup(snapshot.stop)

    def capture(self, **kwargs):
        return rag.capture_rag([["text", "local document"]], cache_root=self.root / "rag", **kwargs)

    def complete(self, prepared):
        artifacts = []
        for request in prepared.artifacts:
            request.destination.parent.mkdir(parents=True, exist_ok=True)
            payload = request.destination.name.encode()
            request.destination.write_bytes(payload)
            artifacts.append(
                PreparationRequest(
                    request.url,
                    request.destination,
                    len(payload),
                    hashlib.sha256(payload).hexdigest(),
                )
            )
        return replace(prepared, artifacts=tuple(artifacts))

    def test_empty_context_does_not_resolve_or_prepare_a_model(self):
        self.assertIsNone(rag.capture_rag([]))
        self.snapshot.assert_not_called()
        self.assertIsNone(rag.prepare_rag(None))

    def test_capture_pins_only_the_native_transformer_closure_and_copies_policy(self):
        prepared = self.capture(cpu=True, offline=True)
        self.assertEqual(len(prepared.artifacts), 8)
        self.assertTrue(all(rag.MODEL_REVISION in request.url for request in prepared.artifacts))
        self.assertFalse(
            any(
                "onnx" in request.url or "openvino" in request.url for request in prepared.artifacts
            )
        )
        weight = next(
            request
            for request in prepared.artifacts
            if request.destination.name == "model.safetensors"
        )
        self.assertEqual(weight.expected_size, 90868376)
        self.assertEqual(len(weight.sha256), 64)
        values = prepared.settings
        values["gpu"] = True
        self.assertFalse(prepared.settings["gpu"])
        self.snapshot.assert_called_once_with(
            rag.MODEL_ID, revision=rag.MODEL_REVISION, local_files_only=True
        )

    def test_complete_pinned_hub_snapshot_is_reused_without_copying_weights(self):
        cached = self.root / "hub-snapshot"
        for name, _, _ in rag.MODEL_FILES:
            path = cached / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        self.snapshot.side_effect = None
        self.snapshot.return_value = str(cached)
        prepared = self.capture()
        self.assertEqual(prepared.model, cached.resolve())
        self.assertTrue(prepared.present)
        self.assertFalse((self.root / "rag").exists())

    def test_offline_missing_file_refuses_before_transfer_or_runtime(self):
        prepared = self.capture(offline=True)
        with patch.object(
            rag, "ModelTransfer", side_effect=AssertionError("No acquisition offline")
        ):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                rag.prepare_rag(prepared)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            rag.runtime_options(prepared)

    def test_complete_offline_model_uses_existing_validation_and_native_local_options(self):
        prepared = self.complete(self.capture(cpu=True, offline=True))
        with patch.dict("sys.modules", {"txtai": None}):
            self.assertIs(rag.prepare_rag(prepared), prepared)
            options = rag.runtime_options(prepared)
        self.assertTrue(Path(options["path"]).is_absolute())
        self.assertFalse(options["gpu"])
        self.assertNotIn("device", options)
        self.assertNotIn(
            "local_files_only", {key: value for key, value in options.items() if key != "vectors"}
        )
        self.assertEqual(options["vectors"], {"local_files_only": True, "trust_remote_code": False})
        prepared.artifacts[0].destination.unlink()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            rag.runtime_options(prepared)

    def test_unchanged_cached_model_is_not_reread_on_each_request(self):
        prepared = self.complete(self.capture(offline=True))
        rag.prepare_rag(prepared)
        actual_open = open

        def no_payload_read(path, mode="r", *args, **kwargs):
            if mode == "rb" and Path(path) in {item.destination for item in prepared.artifacts}:
                raise AssertionError("Unchanged model payload was reread")
            return actual_open(path, mode, *args, **kwargs)

        with patch("builtins.open", side_effect=no_payload_read):
            self.assertIs(rag.prepare_rag(prepared), prepared)
        damaged = prepared.artifacts[0].destination
        original = damaged.read_bytes()
        damaged.write_bytes(b"x" * len(original))
        with self.assertRaisesRegex(IOError, "SHA256"):
            rag.prepare_rag(prepared)

    def test_native_tokenizer_file_resolution_stays_local_even_without_a_flag(self):
        location = Path(importlib.util.find_spec("transformers").origin).parent / "utils/hub.py"
        forbidden = Mock(side_effect=AssertionError("Unexpected Hub request"))
        resolver = function_from(
            location,
            "cached_files",
            {
                "os": os,
                "Path": Path,
                "is_offline_mode": lambda: False,
                "logger": Mock(),
                "hf_hub_download": forbidden,
                "snapshot_download": forbidden,
            },
        )
        prepared = self.complete(self.capture())
        self.assertEqual(
            resolver(str(prepared.model), ["tokenizer.json", "tokenizer_config.json"]),
            [str(prepared.model / "tokenizer.json"), str(prepared.model / "tokenizer_config.json")],
        )
        self.assertIsNone(
            resolver(
                str(prepared.model),
                ["missing-optional.json"],
                _raise_exceptions_for_missing_entries=False,
            )
        )
        forbidden.assert_not_called()

    def test_cancel_detaches_the_existing_transfer_subscription(self):
        transfer = SimpleNamespace(
            done=lambda: False, progress=(0, 0, 0), cancel=Mock(), result=Mock()
        )
        cancelled = SimpleNamespace(is_set=lambda: False, wait=lambda timeout: True)
        with patch.object(rag, "ModelTransfer", return_value=transfer):
            with self.assertRaises(CancelledError):
                rag.prepare_rag(self.capture(), cancelled=cancelled)
        transfer.cancel.assert_called_once()
        transfer.result.assert_not_called()

    def test_chat_capture_and_preparation_keep_rag_policy_before_compute(self):
        values = {"cpu_everywhere": False}
        chat = ModelPreparation(self.root / "chat.gguf", (), json.dumps(values))
        args = SimpleNamespace(
            cpu=True, directml=None, offline=True, _compute_status={"cli": ["--cpu"]}
        )
        captured = capture_chat_request(
            {
                "model": str(chat.model),
                "embed": json.dumps([["text", "local context"]]),
                "_chat_preparation": chat,
            },
            values,
            SimpleNamespace(),
            {},
            launch_args=args,
        )
        prepared = captured["_rag_preparation"]
        args.cpu, args.offline, values["cpu_everywhere"] = False, False, True
        self.assertEqual(prepared.settings, {"gpu": False, "offline": True})
        with patch.object(rag, "prepare_rag", return_value=prepared) as prepare:
            self.assertIs(prepare_chat_request(captured), captured)
        prepare.assert_called_once_with(prepared, progress=None, cancelled=None)

    def test_offline_context_url_is_not_downloaded_when_the_embedding_model_is_cached(self):
        prepared = self.complete(self.capture(offline=True))
        context = PreparationRequest(
            "https://fixture.invalid/context", self.root / "missing-context.txt"
        )
        with patch.object(
            rag, "prepare_rag", side_effect=AssertionError("No runtime or acquisition")
        ):
            with self.assertRaisesRegex(RuntimeError, "context files are missing"):
                prepare_chat_request(
                    {"_rag_preparation": prepared, "_context_preparation": (context,)}
                )

    def test_skip_package_install_does_not_disable_explicit_model_preparation(self):
        chat = ModelPreparation(self.root / "chat.gguf", (), "{}")
        args = SimpleNamespace(
            cpu=False, directml=None, offline=False, skip_dependency_install=True
        )
        with patch.dict(os.environ, {"RF_OFFLINE": "0"}):
            captured = capture_chat_request(
                {
                    "model": str(chat.model),
                    "embed": json.dumps([["text", "context"]]),
                    "_chat_preparation": chat,
                },
                {},
                SimpleNamespace(),
                {},
                launch_args=args,
            )
        self.assertFalse(captured["_rag_preparation"].settings["offline"])

    def test_same_path_context_and_model_mutations_invalidate_the_cached_index(self):
        prepared = self.complete(self.capture())
        document = self.root / "context.txt"
        document.write_text("first context", encoding="utf-8")
        sources = [["url", "https://fixture.invalid/context"]]
        paths = {sources[0][1]: document}
        original = rag.index_identity(sources, prepared, paths)
        with patch(
            "builtins.open", side_effect=AssertionError("Identity must use file stats only")
        ):
            self.assertEqual(rag.index_identity(sources, prepared, paths), original)
        document.write_text("changed context at same path", encoding="utf-8")
        changed_context = rag.index_identity(sources, prepared, paths)
        self.assertNotEqual(changed_context, original)
        prepared.artifacts[0].destination.write_bytes(b"changed config")
        self.assertNotEqual(rag.index_identity(sources, prepared, paths), changed_context)

    def test_index_factory_is_owned_and_failed_or_cancelled_replacement_retains_old_index(self):
        index = function_from(
            "ruinedfooocus/features/chat/llama_pipeline.py",
            "index_sources",
            {"Path": Path, "re": re},
        )
        unload = function_from(
            "ruinedfooocus/features/chat/llama_pipeline.py",
            "unload",
            {
                "NativeServer": type("Native", (), {}),
                "ProfiledChatClient": type("Profiled", (), {}),
                "gc": SimpleNamespace(collect=lambda: None),
            },
        )
        prepared = self.complete(self.capture(cpu=True, offline=True))
        for outcome in ("success", "load_failure", "index_failure", "cancel"):
            with self.subTest(outcome=outcome):
                item = life.JobRegistry().create(owner=("test", "RAG"))
                previous = SimpleNamespace(close=Mock())
                owner = SimpleNamespace(embeddings=previous, embeddings_hash="old", llm=None)
                candidate = SimpleNamespace(close=Mock())
                seen = []

                def factory(**options):
                    self.assertIs(life.require_compute_owner(), item)
                    self.assertFalse(options["gpu"])
                    self.assertEqual(options["path"], str(prepared.model))
                    if outcome == "load_failure":
                        raise RuntimeError("inert model load failure")
                    return candidate

                def build(documents):
                    self.assertIs(life.require_compute_owner(), item)
                    seen.extend(documents)
                    if outcome == "index_failure":
                        raise RuntimeError("inert index failure")
                    if outcome == "cancel":
                        item.cancelled.set()

                candidate.index = build
                native = ModuleType("txtai")
                native.Embeddings = factory
                with (
                    patch.dict(sys.modules, {"txtai": native}),
                    patch.object(life, "admission", life.FairAdmission()),
                    life.job_scope(item),
                    life.admission.lease(item),
                ):
                    if outcome == "success":
                        index(owner, [["text", "alpha\n\nbeta"]], prepared_model=prepared)
                        self.assertIs(owner.embeddings, candidate)
                        self.assertEqual(seen, ["alpha", "beta"])
                        previous.close.assert_called_once()
                        unload(owner)
                        candidate.close.assert_called_once()
                        self.assertIsNone(owner.embeddings)
                    else:
                        with self.assertRaises((RuntimeError, CancelledError)):
                            index(owner, [["text", "alpha\n\nbeta"]], prepared_model=prepared)
                        self.assertIs(owner.embeddings, previous)
                        self.assertEqual(owner.embeddings_hash, "old")
                        previous.close.assert_not_called()
                        if outcome != "load_failure":
                            candidate.close.assert_called_once()

    def test_actual_worker_keeps_rag_preparation_outside_compute_and_ready_queue(self):
        from ruinedfooocus.app import preparation_admission
        from ruinedfooocus.features.chat import model_preparation, llama_models

        entered, release = threading.Event(), threading.Event()
        jobs = life.JobRegistry()
        prepared = self.capture()
        start = Mock()
        from ruinedfooocus.core.task_errors import describe_error

        namespace = {
            "settings": SimpleNamespace(default_settings={}),
            "shared": SimpleNamespace(path_manager=None),
            "registry": jobs,
            "api_jobs": {},
            "start_worker": start,
            "job_scope": life.job_scope,
            "current_job": life.current_job,
            "describe_error": describe_error,
            "add_result": Mock(),
        }
        for helper in ("validate_editor_data", "discard_editor_result", "add_task"):
            function_from("ruinedfooocus/app/async_worker.py", helper, namespace)
        enqueue = namespace["add_task"]
        admission = PreparationAdmission(max_workers=1, max_jobs=1)

        def prepare(*args, **kwargs):
            item = life.current_job()
            self.assertIsNotNone(item)
            self.assertIs(item, jobs.jobs[item.task_id])
            self.assertFalse(item.started)
            with self.assertRaises(life.OwnershipError):
                life.require_compute_owner()
            entered.set()
            self.assertTrue(release.wait(3))
            return prepared

        try:
            with (
                patch.object(preparation_admission, "preparation_admission", admission),
                patch.object(
                    model_preparation,
                    "capture_chat_request",
                    side_effect=lambda data, *args: dict(data, _rag_preparation=prepared),
                ),
                patch.object(llama_models, "local_models", return_value={}),
                patch.object(rag, "prepare_rag", side_effect=prepare),
            ):
                other = jobs.create(owner=("test", "active-B"))
                with life.admission.lease(other):
                    task_id = enqueue({"task_type": "llama"})
                    self.assertTrue(entered.wait(2))
                    self.assertFalse(jobs.ready)
                    start.assert_not_called()
                    self.assertFalse(jobs.jobs[task_id].started)
                    release.set()
                    jobs.jobs[task_id].preparation.result(2)
                    self.assertFalse(jobs.jobs[task_id].started)
                queued, _ = jobs.take(timeout=0)
                self.assertEqual(queued.task_id, task_id)
                start.assert_called_once()
        finally:
            release.set()
            admission.close()

    def test_unowned_or_unprepared_context_cannot_start_a_native_index(self):
        index = function_from(
            "ruinedfooocus/features/chat/llama_pipeline.py",
            "index_sources",
            {"Path": Path, "re": re},
        )
        prepared = self.complete(self.capture())
        owner = SimpleNamespace(embeddings=None, embeddings_hash="")
        with self.assertRaises(life.OwnershipError):
            index(owner, [["text", "context"]], prepared_model=prepared)
        with self.assertRaisesRegex(ValueError, "preparation is missing"):
            index(owner, [["text", "context"]])
        with self.assertRaisesRegex(ValueError, "Prepare URL context"):
            index(owner, [["url", "https://fixture.invalid/context"]], prepared_model=prepared)


if __name__ == "__main__":
    unittest.main()
