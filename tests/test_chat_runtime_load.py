"""Pipeline load identity and initialized capability callers without native work."""

import ast
import sys
from contextlib import nullcontext
from concurrent.futures import CancelledError
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import ruinedfooocus.app as app_modules
from ruinedfooocus.backends.llama.runtime_events import RuntimeLoad
from ruinedfooocus.features.chat.model_preparation import (
    initialized_runtime_status,
    runtime_selection_key,
)
from ruinedfooocus.features.chat.model_profiles import ProfiledChatClient
from ruinedfooocus.features.models.model_capabilities import RuntimeIdentity
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.core.preparation import PreparationRequest, PreparationService

ROOT = Path(__file__).resolve().parents[1]


def owner():
    tree = ast.parse(
        (ROOT / "ruinedfooocus/features/chat/llama_pipeline.py").read_text(encoding="utf-8")
    )
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pipeline"
    )
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in ("load_base_model", "runtime_status")
    ]
    scope = {"CancelledError": CancelledError}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "pipeline-load", "exec"), scope)
    value = scope["pipeline"]()
    value.load_state = None
    value.runtime_snapshot = None
    value.unload = Mock(side_effect=lambda: setattr(value, "runtime_snapshot", None))
    return value


class ChatRuntimeLoadTests(unittest.TestCase):
    def test_inline_inspection_load_receives_owned_cancellation_before_completion(self):
        from ruinedfooocus.app import api_runtime

        item = SimpleNamespace(owner=("ui", "fixture"))
        request = SimpleNamespace(
            owner=item.owner,
            call_id="inspect-1",
            session_id="session",
            revision=1,
            source_id="source",
        )
        prepared = api_runtime.PreparedImageTool(
            request, {}, "original", True, inspection_path="fixture.png"
        )
        cancelled = threading.Event()

        def check():
            if cancelled.is_set():
                raise CancelledError()

        def load(*args, **options):
            self.assertIs(options["check_cancelled"], check)
            self.assertEqual(options["load_id"], "chat-inspect-inspect-1")
            cancelled.set()
            options["check_cancelled"]()

        pipeline = SimpleNamespace(load_base_model=Mock(side_effect=load), complete_text=Mock())
        worker = SimpleNamespace(chat_stage=lambda data: nullcontext())
        with (
            patch.dict("sys.modules", {"ruinedfooocus.app.async_worker": worker}),
            patch.object(app_modules, "async_worker", worker, create=True),
            patch.object(api_runtime, "_check_image_tool_runtime"),
            patch("ruinedfooocus.features.gallery.editing_workflow.editor_source_record"),
        ):
            with self.assertRaises(CancelledError):
                api_runtime.execute_chat_image_tool(
                    request,
                    gen_data={"_job": item},
                    pipeline=pipeline,
                    prepared=prepared,
                    check_cancelled=check,
                )
        pipeline.complete_text.assert_not_called()

    def test_api_review_and_restore_forward_the_bound_job_checker(self):
        tree = ast.parse((ROOT / "ruinedfooocus/app/api_runtime.py").read_text(encoding="utf-8"))
        review = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "review"
        )
        run = next(
            node for node in review.body if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        for cancel_stage in (None, "review", "restore"):
            with self.subTest(cancel_stage=cancel_stage):
                job = SimpleNamespace(cancelled=threading.Event())
                current = SimpleNamespace(
                    llm=object(),
                    model_settings=("original",),
                    vision_projector=None,
                    _prepared_request=object(),
                    unload=Mock(),
                )
                reviewer = SimpleNamespace(
                    vision_projector="projector",
                    unload=Mock(),
                    complete_text=Mock(return_value="review"),
                )

                def check(data):
                    if data["_api_job"].cancelled.is_set():
                        raise CancelledError()

                def loading(stage, *args, **kwargs):
                    if stage == cancel_stage:
                        job.cancelled.set()
                    kwargs["check_cancelled"]()

                current.load_base_model = Mock(
                    side_effect=lambda *args, **kwargs: loading("restore", *args, **kwargs)
                )
                reviewer.load_base_model = Mock(
                    side_effect=lambda *args, **kwargs: loading("review", *args, **kwargs)
                )
                scope = {"selected": {"path": "reviewer"}, "prepared": object(), "messages": []}
                exec(
                    compile(ast.Module(body=[run], type_ignores=[]), "api-review-load", "exec"),
                    scope,
                )
                shared = SimpleNamespace(state={"pipeline": current})
                worker = SimpleNamespace(check_interrupt=check)
                with (
                    patch.dict(
                        "sys.modules",
                        {
                            "ruinedfooocus.app.state": shared,
                            "ruinedfooocus.features.chat.llama_pipeline": SimpleNamespace(
                                pipeline=lambda: reviewer
                            ),
                            "ruinedfooocus.app.async_worker": worker,
                        },
                    ),
                    patch.object(app_modules, "state", shared, create=True),
                    patch.object(
                        sys.modules["ruinedfooocus.app"], "async_worker", worker, create=True
                    ),
                ):
                    if cancel_stage is not None:
                        with self.assertRaises(CancelledError):
                            scope["run"](job, 42)
                    else:
                        self.assertEqual(scope["run"](job, 42), "review")
                self.assertEqual(
                    reviewer.load_base_model.call_args.kwargs["load_id"], "api-review-42"
                )
                reviewer.unload.assert_called_once()
                if cancel_stage == "review":
                    current.load_base_model.assert_not_called()
                else:
                    self.assertEqual(
                        current.load_base_model.call_args.kwargs["load_id"], "api-review-42-restore"
                    )

    def test_prompt_helper_forwards_load_cancellation_and_never_returns_fallback_on_stop(self):
        import re

        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/chat/llama_pipeline.py").read_text(encoding="utf-8")
        )
        method = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_llama"
        )
        method.decorator_list = []
        helper = SimpleNamespace(load_base_model=Mock(side_effect=CancelledError()), unload=Mock())
        scope = {
            "re": re,
            "pipeline": lambda: helper,
            "model_management": SimpleNamespace(
                InterruptProcessingException=type("Interrupted", (Exception,), {})
            ),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), "prompt-helper", "exec"), scope)
        check = Mock()
        prepared = SimpleNamespace(model=Path("fixture.gguf"))
        with patch("ruinedfooocus.app.resource_lifecycle.service_pipeline", return_value=None):
            with self.assertRaises(CancelledError):
                scope["run_llama"](
                    "unused",
                    "prompt",
                    instruction="Rewrite",
                    prepared=prepared,
                    check_cancelled=check,
                )
        self.assertIs(helper.load_base_model.call_args.kwargs["check_cancelled"], check)
        helper.unload.assert_called_once()

    def test_waiting_artifact_reader_checks_cancellation_without_releasing_writer(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/chat/llama_pipeline.py").read_text(encoding="utf-8")
        )
        cls = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pipeline"
        )
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "_load_prepared_model"
        )
        scope = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "pipeline-lease", "exec"), scope)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.gguf"
            path.write_bytes(b"fixture")
            service = PreparationService(staging_root=Path(temporary) / "staging")
            owner = SimpleNamespace(llm=None, load_base_model=Mock())
            prepared = SimpleNamespace(artifacts=(PreparationRequest("", path),))
            check = Mock(side_effect=[None, CancelledError()])
            try:
                with (
                    service.artifact_lease(path, write=True),
                    patch("ruinedfooocus.core.preparation.preparation_service", service),
                ):
                    with self.assertRaises(CancelledError):
                        scope["_load_prepared_model"](
                            owner, prepared=prepared, check_cancelled=check
                        )
                    self.assertTrue(service._artifacts[service._key(path)].writer)
                owner.load_base_model.assert_not_called()
            finally:
                service.close()

    def test_ready_publication_is_copied_and_bound_to_one_load_id(self):
        pipeline = owner()
        events = []

        def initialize(*args, runtime_load, **kwargs):
            runtime_load.emit(phase="loading_model")
            pipeline.runtime_snapshot = {
                "load_id": runtime_load.load_id,
                "model_identity": {"path": "A"},
            }

        pipeline._initialize_model = initialize
        pipeline.load_base_model("A", load_id="request-A", runtime_progress=events.append)
        self.assertEqual([event["phase"] for event in events], ["loading_model", "ready"])
        self.assertEqual(events[-1]["result"]["load_id"], "request-A")
        status = pipeline.runtime_status()
        status["initialized"]["model_identity"]["path"] = "mutated"
        self.assertEqual(pipeline.runtime_snapshot["model_identity"]["path"], "A")

    def test_cancelled_health_completion_clears_only_the_matching_owner(self):
        pipeline = owner()
        events = []
        pipeline._initialize_model = Mock(side_effect=CancelledError())
        with self.assertRaises(CancelledError):
            pipeline.load_base_model("A", load_id="A", runtime_progress=events.append)
        pipeline.unload.assert_called_once()
        self.assertEqual(events[-1]["phase"], "cancelled")
        pipeline.unload.reset_mock()

        def replaced(*args, runtime_load, **kwargs):
            runtime_load.cancel()
            pipeline._runtime_load = RuntimeLoad("newer")
            pipeline.runtime_snapshot = {"load_id": "newer"}
            runtime_load.check()

        pipeline._initialize_model = replaced
        with self.assertRaises(CancelledError):
            pipeline.load_base_model("old", load_id="old", runtime_progress=events.append)
        pipeline.unload.assert_not_called()
        self.assertEqual(pipeline.runtime_snapshot["load_id"], "newer")
        self.assertFalse(any(event["load_id"] == "old" for event in events))

    def test_prevalidation_failure_preserves_valid_initialized_model_and_safe_error(self):
        pipeline = owner()
        pipeline.runtime_snapshot = {"load_id": "old-valid"}
        pipeline._initialize_model = Mock(
            side_effect=ValueError("private API-key must not enter event")
        )
        events = []
        with self.assertRaises(ValueError):
            pipeline.load_base_model("bad", runtime_progress=events.append)
        pipeline.unload.assert_not_called()
        self.assertEqual(pipeline.runtime_snapshot["load_id"], "old-valid")
        self.assertEqual(events[-1]["phase"], "failed")
        self.assertNotIn("private", str(events))

    def test_same_path_projector_replacement_invalidates_explicit_and_resolved_auto(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, projector = (Path(temporary) / name for name in ("model.gguf", "projector.gguf"))
            model.write_bytes(b"model")
            projector.write_bytes(b"first")
            values = {"llama_mmproj": str(projector)}
            explicit = runtime_selection_key(model, values, vision=True)
            automatic = runtime_selection_key(
                model, {"llama_mmproj": "Auto"}, vision=True, projector=projector
            )
            projector.write_bytes(b"new projector content")
            self.assertNotEqual(explicit, runtime_selection_key(model, values, vision=True))
            self.assertNotEqual(
                automatic,
                runtime_selection_key(
                    model, {"llama_mmproj": "Auto"}, vision=True, projector=projector
                ),
            )

    def test_terminal_resident_state_is_passive_and_none_after_service_cleanup(self):
        pipeline = owner()
        pipeline.llm = object()
        pipeline.loaded_paths = ("A",)
        pipeline.runtime_snapshot = {"load_id": "A"}
        with patch("ruinedfooocus.app.resource_lifecycle.service_pipeline", return_value=pipeline):
            self.assertEqual(initialized_runtime_status()["active_model"], "A")
        with patch("ruinedfooocus.app.resource_lifecycle.service_pipeline", return_value=None):
            self.assertIsNone(initialized_runtime_status()["runtime_snapshot"])

    def test_actual_initialized_unknown_runtime_cannot_inherit_configured_pin_capabilities(self):
        artifact = load_registry().artifact(
            "existing:llm/Qwen3.5-9B-ultra-uncensored-heretic-v2-Q4_K_M.gguf"
        )
        template = "{{ tools }} tool_calls {% if enable_thinking %}think{% endif %}"
        for build, supported in (
            ("b12110", True),
            ("b12327", True),
            ("b12366", True),
            ("b99999", False),
            (None, False),
        ):
            adapter = ProfiledChatClient(
                Mock(),
                artifact,
                {},
                initialized=True,
                template=template,
                resources=(("fixture.gguf", 1, 2, 3),),
                initialized_runtime=RuntimeIdentity(
                    "llama.cpp", build, origin="initialized fixture"
                ),
            )
            self.assertEqual(adapter.image_tool_capability["available"], supported)
            self.assertEqual(
                adapter.reasoning_capability["control"] == "enable_thinking", supported
            )


if __name__ == "__main__":
    unittest.main()
