"""Real tool DTOs, worker continuations and private HTTP transport; inert compute."""

from contextlib import contextmanager
from concurrent.futures import CancelledError
import base64
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from ruinedfooocus.app import api_runtime
from ruinedfooocus.app import resource_lifecycle as life
from ruinedfooocus.app.preparation_admission import PreparationAdmission
from ruinedfooocus.features.chat.chat_tools import (
    ImageToolContext,
    ImageToolContinuation,
    dispatch_image_tool,
    public_image_tool_result,
    validate_image_tool_call,
)
from ruinedfooocus.features.generation.image_preparation import ImageReadiness, ImageRequirement
from ruinedfooocus.features.gallery import editor_session
from ruinedfooocus.integrations.openai_api import create_apps
from ui_test_helpers import function_from

WORKER = "ruinedfooocus/app/async_worker.py"
CAPABILITY = {"available": True, "binding": "initialized-template-identity"}


def request(settings=None, *, owner=("ui", "service"), intent="run"):
    context = ImageToolContext(owner, intent, settings or {})
    return validate_image_tool_call(
        {
            "id": "call_actual_1",
            "type": "function",
            "function": {
                "name": "generate_image",
                "arguments": json.dumps({"prompt": "Exactly two blue cups"}),
            },
        },
        context,
        CAPABILITY,
    )


class ChatImageContinuationTests(unittest.TestCase):
    def setUp(self):
        self.registry, self.admission = life.JobRegistry(), life.FairAdmission()
        self.item = self.registry.create(owner=("ui", "service"))
        self.registry.enqueue(self.item, {"task_type": "llama", "task_id": self.item.task_id})
        started, _ = self.registry.take(timeout=0)
        self.assertIs(started, self.item)
        self.pool = PreparationAdmission(max_workers=1, max_jobs=2)
        self.addCleanup(self.pool.close)
        self.enterContext(patch.object(life, "registry", self.registry))
        self.enterContext(patch.object(life, "admission", self.admission))
        self.enterContext(
            patch("ruinedfooocus.app.preparation_admission.preparation_admission", self.pool)
        )
        self.scope = dict(
            registry=self.registry,
            current_job=life.current_job,
            start_worker=Mock(),
            describe_error=str,
            CancelledError=CancelledError,
            api_jobs={},
            discard_editor_result=lambda *a, **k: None,
        )
        for name in ("add_result", "publish_image_tool_error", "defer_chat_image_tool"):
            function_from(WORKER, name, self.scope)

    def test_one_turn_can_schedule_only_one_preparation_and_mutation(self):
        continuation = ImageToolContinuation(
            request(), object(), lambda value: "rendered", "chat.gguf", False
        )
        prepare = Mock(return_value="prepared")
        with (
            patch.object(api_runtime, "prepare_chat_image_tool", prepare),
            life.job_scope(self.item),
        ):
            data = {"task_id": self.item.task_id, "_job": self.item}
            self.scope["defer_chat_image_tool"](self.item, data, continuation)
            pending = self.item.preparation
            pending.result(timeout=3)
            with self.assertRaisesRegex(ValueError, "Only one"):
                self.scope["defer_chat_image_tool"](self.item, data, continuation)
        prepare.assert_called_once()
        self.assertEqual(len(self.registry.ready), 1)
        self.assertEqual(self.registry.ready[0][0].identity, self.item.identity)

    def test_preparation_leaves_chat_compute_and_requeues_the_same_owner_and_job(self):
        observed = []
        continuation = ImageToolContinuation(
            request(), object(), lambda value: "rendered", "chat.gguf", False
        )

        def prepare(*args, **kwargs):
            observed.append((dict(self.admission.held), life.current_job()))
            return "prepared"

        with (
            patch.object(api_runtime, "prepare_chat_image_tool", side_effect=prepare),
            life.job_scope(self.item),
        ):
            with life.service_scope("chat", {}):
                self.assertIn("compute:shared", self.admission.held)
            self.scope["defer_chat_image_tool"](
                self.item, {"task_id": self.item.task_id, "_job": self.item}, continuation
            )
            self.item.preparation.result(timeout=3)
        selected, data = self.registry.take(timeout=0)
        self.assertIs(selected, self.item)
        self.assertEqual(observed, [({}, self.item)])
        self.assertEqual(data["task_type"], "chat_image_tool")
        self.assertEqual(data["task_id"], self.item.task_id)
        self.assertFalse(self.item.terminal)
        self.assertFalse(self.item.finished)

    def test_cancel_during_preparation_never_reenqueues_or_publishes_a_completion(self):
        entered, release = threading.Event(), threading.Event()
        continuation = ImageToolContinuation(
            request(), object(), lambda value: "rendered", "chat.gguf", False
        )

        def prepare(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("Fixture release was not received")
            kwargs["check_cancelled"]()

        with (
            patch.object(api_runtime, "prepare_chat_image_tool", side_effect=prepare),
            life.job_scope(self.item),
        ):
            self.scope["defer_chat_image_tool"](
                self.item, {"task_id": self.item.task_id, "_job": self.item}, continuation
            )
            pending = self.item.preparation
            self.assertTrue(entered.wait(2))
            self.registry.cancel(self.item.task_id, owner=self.item.owner)
            release.set()
            pending.result(timeout=3)
        self.assertEqual(list(self.registry.ready), [])
        terminal = list(self.item.events)
        self.assertEqual([kind for kind, _ in terminal], ["image_tool_result", "results"])
        self.assertEqual(terminal[0][1]["status"], "cancelled")
        self.assertEqual(terminal[0][1]["call_id"], continuation.request.call_id)
        self.assertTrue(self.item.finished)

    def test_resume_publishes_owned_history_and_one_terminal_in_the_same_job(self):
        publish = Mock(return_value={"snapshot": {"session_id": "owned_session", "revision": 1}})
        self.scope.update(
            check_interrupt=lambda data: self.item.check(), publish_editor_result=publish
        )
        resume = function_from(WORKER, "resume_chat_image_tool", self.scope)
        action = request()

        def render(value):
            publish.assert_not_called()
            self.assertFalse(self.item.terminal)
            self.assertEqual(list(self.item.events), [])
            return "rendered image"

        continuation = ImageToolContinuation(action, object(), render, "chat.gguf", False)
        value = dict(status="completed", call_id=action.call_id, images=["internal-result.png"])
        data = dict(
            task_id=self.item.task_id,
            _job=self.item,
            _image_tool_continuation=continuation,
            _prepared_image_tool=object(),
        )
        with (
            patch.object(api_runtime, "execute_chat_image_tool", return_value=value) as execute,
            life.job_scope(self.item),
        ):
            resume(data)
        execute.assert_called_once()
        self.assertIs(execute.call_args.kwargs["gen_data"]["_job"], self.item)
        publish.assert_called_once_with(self.item)
        self.assertEqual([kind for kind, _ in self.item.events], ["image_tool_result", "results"])
        terminal, result = list(self.item.events)
        self.assertEqual(terminal[1]["call_id"], action.call_id)
        self.assertEqual(terminal[1]["editor"]["session_id"], "owned_session")
        self.assertNotIn("images", terminal[1])
        self.assertEqual(result[1], "rendered image")
        self.assertTrue(self.item.terminal)

    def test_cancel_after_render_cannot_publish_completed_history_or_tool_result(self):
        publish = Mock()
        self.scope.update(
            check_interrupt=lambda data: self.item.check(), publish_editor_result=publish
        )
        resume = function_from(WORKER, "resume_chat_image_tool", self.scope)
        action = request()
        entered, release = threading.Event(), threading.Event()
        failures = []

        def cancel_at_render(value):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("Render barrier was not released")
            return "discarded rendering"

        continuation = ImageToolContinuation(action, object(), cancel_at_render, "chat.gguf", False)
        data = dict(
            task_id=self.item.task_id,
            _job=self.item,
            _image_tool_continuation=continuation,
            _prepared_image_tool=object(),
        )

        def run():
            try:
                with life.job_scope(self.item):
                    resume(data)
            except Exception as error:
                failures.append(error)

        with (
            patch.object(
                api_runtime,
                "execute_chat_image_tool",
                return_value=dict(
                    status="completed", call_id=action.call_id, images=["result.png"]
                ),
            ),
            life.job_scope(self.item),
        ):
            thread = threading.Thread(target=run)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                self.registry.cancel(self.item.task_id, owner=self.item.owner)
            finally:
                release.set()
                thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(failures), 1)
            self.assertIsInstance(failures[0], CancelledError)
            self.scope["publish_image_tool_error"](
                self.item, continuation, CancelledError("stopped")
            )
            self.scope["add_result"](self.item.task_id, "results", {"error": "stopped"})
            self.scope["publish_image_tool_error"](
                self.item, continuation, CancelledError("duplicate")
            )
        publish.assert_not_called()
        self.assertEqual([kind for kind, _ in self.item.events], ["image_tool_result", "results"])
        self.assertEqual(self.item.events[0][1]["status"], "cancelled")


class ChatImageServiceTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.checkpoint = self.folder / "checkpoint.safetensors"
        header = b'{"conditioner.embedders.0.x":{},"first_stage_model.x":{}}'
        self.checkpoint.write_bytes(len(header).to_bytes(8, "little") + header)
        self.registry = life.JobRegistry()
        self.item = self.registry.create(owner=("ui", "service"))
        self.enterContext(patch.object(life, "registry", self.registry))
        self.enterContext(life.job_scope(self.item))
        self.store = editor_session.EditorSessionStore(self.folder / "sessions")
        self.enterContext(patch.object(editor_session, "default_store", return_value=self.store))
        models = SimpleNamespace(
            offline=True,
            get_models_by_path=lambda *a, **k: {"baseModel": "SDXL"},
            get_model_base=lambda row: row["baseModel"],
            get_file=lambda *a: self.checkpoint,
        )
        manager = SimpleNamespace(
            paths={},
            DOWNLOADABLE_FILES={},
            get_abspath=Path,
            get_model_folders=lambda category: [self.folder],
            model_paths={"temp_outputs_path": self.folder},
        )
        self.shared = SimpleNamespace(
            models=models,
            path_manager=manager,
            state={},
            settings=SimpleNamespace(default_settings={"image_engine": "native"}),
            resolution_settings=SimpleNamespace(
                CUSTOM_RESOLUTION="Custom...", aspect_ratios={"square": (64, 64)}
            ),
            performance_settings=SimpleNamespace(apply=lambda data, family: dict(data)),
        )
        self.enterContext(patch.dict(sys.modules, {"ruinedfooocus.app.state": self.shared}))
        self.enterContext(
            patch.object(sys.modules["ruinedfooocus.app"], "state", self.shared, create=True)
        )
        scope = dict(
            shared=self.shared,
            settings=self.shared.settings,
            current_job=life.current_job,
            control_tools=SimpleNamespace(get_settings=lambda data: data.get("controlnet") or {}),
            json=json,
            check_interrupt=lambda data: self.item.check(),
        )
        for name in (
            "validate_editor_data",
            "capture_image_data",
            "prepare_image_data",
            "stage_editor_original",
            "publish_editor_result",
            "discard_editor_result",
        ):
            function_from(WORKER, name, scope)
        self.scope = scope
        self.compute = Mock()
        self.worker = SimpleNamespace(
            **{
                name: scope[name]
                for name in (
                    "capture_image_data",
                    "prepare_image_data",
                    "stage_editor_original",
                    "publish_editor_result",
                    "discard_editor_result",
                )
            }
        )
        self.worker.check_interrupt = lambda data: self.item.check()

        @contextmanager
        def handoff(*args):
            self.item.check()
            yield

        self.worker.image_handoff = handoff
        self.worker._process = self.compute
        self.enterContext(patch.dict(sys.modules, {"ruinedfooocus.app.async_worker": self.worker}))
        self.pipeline = SimpleNamespace(
            image_tool_capability=lambda: CAPABILITY,
            vision_projector=None,
            _prepared_request=SimpleNamespace(model="chat.gguf"),
        )
        self.settings = dict(
            base_model_name=self.checkpoint.name,
            prompt="old request",
            negative="No text or crop",
            loras=[],
            performance_selection="Own settings",
            custom_steps=9,
            cfg=4.5,
            sampler_name="euler",
            scheduler="normal",
            clip_skip=1,
            image_engine="native",
            aspect_ratios_selection="Custom...",
            custom_width=64,
            custom_height=64,
            image_number=1,
            controlnet={},
            automatic_upscale="Off",
        )

    def test_preview_has_a_real_plan_and_readiness_without_preparation_or_compute(self):
        action = request(self.settings, intent="preview")
        with patch.object(
            self.worker, "prepare_image_data", side_effect=AssertionError("Preview prepared files")
        ):
            prepared = api_runtime.prepare_chat_image_tool(
                action, gen_data={"_job": self.item}, pipeline=self.pipeline
            )
            result = api_runtime.execute_chat_image_tool(
                action, gen_data={"_job": self.item}, pipeline=self.pipeline, prepared=prepared
            )
        self.assertEqual(result["status"], "preview")
        self.assertEqual(result["readiness"]["state"], "ready")
        self.assertIn("_effective_image_plan", prepared.data)
        self.compute.assert_not_called()
        self.assertEqual(list((self.folder / "sessions").glob("*.json")), [])

    def test_preview_public_readiness_preserves_missing_roles_without_paths_or_preparation(self):
        readiness = ImageReadiness(
            "missing",
            (
                ImageRequirement(
                    "vae",
                    "selected-vae.safetensors",
                    self.folder / "private" / "selected-vae.safetensors",
                    "missing",
                    "trusted catalogue",
                    "publisher sha256",
                    1234,
                    "Run prepares the selected required role.",
                ),
            ),
            1234,
        )
        action = request(self.settings, intent="preview")
        with (
            patch(
                "ruinedfooocus.features.generation.image_preparation.image_readiness",
                return_value=readiness,
            ),
            patch.object(
                self.worker,
                "prepare_image_data",
                side_effect=AssertionError("Preview prepared files"),
            ),
        ):
            prepared = api_runtime.prepare_chat_image_tool(
                action, gen_data={"_job": self.item}, pipeline=self.pipeline
            )
            result = api_runtime.execute_chat_image_tool(
                action, gen_data={"_job": self.item}, pipeline=self.pipeline, prepared=prepared
            )
        public = public_image_tool_result(result)
        self.assertEqual(public["status"], "preview")
        self.assertEqual(public["readiness"]["state"], "missing")
        self.assertEqual(public["readiness"]["missing_bytes"], 1234)
        role = public["readiness"]["requirements"][0]
        self.assertEqual(
            (role["role"], role["source"], role["integrity"]),
            ("vae", "trusted catalogue", "publisher sha256"),
        )
        self.assertEqual(role["reason"], "Run prepares the selected required role.")
        self.assertNotIn("destination", role)
        self.assertNotIn(str(self.folder), json.dumps(public))
        self.compute.assert_not_called()

    def test_invalid_calls_reject_at_the_actual_service_before_capture_prepare_or_submission(self):
        for function in (
            {"name": "generate_image", "arguments": '{"prompt":"cups","path":"C:/private.png"}'},
            {"name": "generate_image", "arguments": '{"prompt":'},
            {"name": "shell", "arguments": "{}"},
        ):
            payload = {
                "intent": "run",
                "call": {"id": "call_invalid", "type": "function", "function": function},
            }
            with (
                self.subTest(function=function),
                patch.object(life, "service_pipeline", return_value=self.pipeline),
                patch.object(api_runtime, "image_defaults", self.settings),
                patch.object(api_runtime, "submit") as submit,
                patch.object(self.worker, "capture_image_data") as capture,
                patch.object(self.worker, "prepare_image_data") as prepare,
            ):
                with self.assertRaises(ValueError):
                    api_runtime.image_tool(payload, owner=self.item.owner)
                capture.assert_not_called()
                prepare.assert_not_called()
                submit.assert_not_called()

    def editor_action(self, snapshot, *, owner=None):
        context = ImageToolContext(
            owner or self.item.owner,
            "run",
            self.settings,
            session_id=snapshot.session_id,
            source_id=snapshot.current.artifact_id,
            revision=snapshot.revision,
            edit_operations=("edit",),
        )
        return validate_image_tool_call(
            {
                "id": "call_edit",
                "type": "function",
                "function": {
                    "name": "edit_image",
                    "arguments": json.dumps(
                        {
                            "prompt": "Change only the background",
                            "source_id": snapshot.current.artifact_id,
                            "operation": "edit",
                        }
                    ),
                },
            },
            context,
            CAPABILITY,
        )

    def test_foreign_and_stale_selected_sources_reject_before_preparation(self):
        source = self.folder / "source.png"
        Image.new("RGB", (64, 64), "white").save(source)
        snapshot = self.store.start(source, owner=self.item.owner)
        foreign = self.registry.create(owner=("ui", "foreign"))
        action = self.editor_action(snapshot, owner=foreign.owner)
        with (
            patch.object(self.worker, "capture_image_data") as capture,
            patch.object(self.worker, "prepare_image_data") as prepare,
        ):
            with self.assertRaises(PermissionError):
                api_runtime.prepare_chat_image_tool(
                    action, gen_data={"_job": foreign}, pipeline=self.pipeline
                )
            capture.assert_not_called()
            prepare.assert_not_called()

        stale = self.editor_action(snapshot)
        ticket = self.store.capture(
            snapshot.session_id,
            snapshot.revision,
            "processing",
            source_id=snapshot.current.artifact_id,
            owner=self.item.owner,
        )
        output = self.folder / "new-current.png"
        Image.new("RGB", (64, 64), "blue").save(output)
        current = self.store.complete(
            ticket, output, execution={"kind": "processing"}, owner=self.item.owner
        )
        with (
            patch.object(self.worker, "capture_image_data") as capture,
            patch.object(self.worker, "prepare_image_data") as prepare,
        ):
            with self.assertRaises(CancelledError):
                api_runtime.prepare_chat_image_tool(
                    stale, gen_data={"_job": self.item}, pipeline=self.pipeline
                )
            capture.assert_not_called()
            prepare.assert_not_called()
        after = self.store.snapshot(snapshot.session_id, owner=self.item.owner)
        self.assertEqual(after.current.artifact_id, current.current.artifact_id)
        self.assertEqual(len(after.steps), 1)

    def test_runtime_binding_change_and_foreign_owner_reject_before_capture(self):
        action = request(self.settings)
        with patch.object(self.worker, "capture_image_data") as capture:
            self.pipeline.image_tool_capability = lambda: dict(CAPABILITY, binding="new-template")
            with self.assertRaisesRegex(ValueError, "template changed"):
                api_runtime.prepare_chat_image_tool(
                    action, gen_data={"_job": self.item}, pipeline=self.pipeline
                )
            capture.assert_not_called()
            other = life.JobRegistry().create(owner=("ui", "other"))
            with self.assertRaises(PermissionError):
                api_runtime.prepare_chat_image_tool(
                    action, gen_data={"_job": other}, pipeline=self.pipeline
                )
            capture.assert_not_called()

    def test_runtime_changes_after_preparation_reject_before_preview_inspection_or_image_handoff(
        self,
    ):
        source = self.folder / "source.png"
        Image.new("RGB", (64, 64), "white").save(source)
        snapshot = self.store.start(source, owner=self.item.owner)
        self.pipeline.vision_projector = object()
        self.compute.return_value = [str(source)]
        for name, intent in (
            ("generate_image", "run"),
            ("generate_image", "preview"),
            ("inspect_image", "run"),
            ("inspect_image", "preview"),
        ):
            for change in ({"binding": "new-template"}, {"available": False}):
                with self.subTest(name=name, intent=intent, change=change):
                    self.pipeline.image_tool_capability = Mock(return_value=dict(CAPABILITY))
                    if name == "inspect_image":
                        context = ImageToolContext(
                            self.item.owner,
                            intent,
                            self.settings,
                            session_id=snapshot.session_id,
                            revision=snapshot.revision,
                            source_id=snapshot.current.artifact_id,
                            inspect_available=True,
                        )
                        action = validate_image_tool_call(
                            {
                                "id": "call_inspect",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(
                                        {
                                            "source_id": snapshot.current.artifact_id,
                                            "question": "What is visible?",
                                        }
                                    ),
                                },
                            },
                            context,
                            CAPABILITY,
                        )
                    else:
                        action = request(self.settings, intent=intent)
                    prepared = api_runtime.prepare_chat_image_tool(
                        action, gen_data={"_job": self.item}, pipeline=self.pipeline
                    )
                    state_before = dict(self.item.state)
                    editor_before = self.store.snapshot(
                        snapshot.session_id, owner=self.item.owner
                    ).to_dict(public=False)
                    self.pipeline.image_tool_capability.return_value = dict(CAPABILITY, **change)
                    with (
                        patch.object(self.worker, "image_handoff") as handoff,
                        patch.object(self.worker, "chat_stage", create=True) as chat,
                        patch.object(self.worker, "stage_editor_original") as stage,
                        patch.object(self.worker, "publish_editor_result") as publish,
                        patch.object(self.pipeline, "load_base_model", create=True) as load,
                        patch.object(self.pipeline, "complete_text", create=True) as complete,
                    ):
                        with self.assertRaisesRegex(ValueError, "template changed"):
                            api_runtime.execute_chat_image_tool(
                                action,
                                gen_data={"_job": self.item},
                                pipeline=self.pipeline,
                                prepared=prepared,
                            )
                        handoff.assert_not_called()
                        chat.assert_not_called()
                        stage.assert_not_called()
                        publish.assert_not_called()
                        load.assert_not_called()
                        complete.assert_not_called()
                    self.compute.assert_not_called()
                    self.assertEqual(self.item.state, state_before)
                    self.assertEqual(
                        self.store.snapshot(snapshot.session_id, owner=self.item.owner).to_dict(
                            public=False
                        ),
                        editor_before,
                    )

    def test_run_preserves_captured_settings_and_stages_an_owned_result_until_terminal(self):
        action = request(self.settings)
        self.settings.update(negative="Changed later", custom_steps=77)
        prepared = api_runtime.prepare_chat_image_tool(
            action, gen_data={"_job": self.item, "model": "chat.gguf"}, pipeline=self.pipeline
        )
        self.assertEqual(prepared.data["negative"], "No text or crop")
        self.assertEqual(prepared.data["custom_steps"], 9)
        output = self.folder / "generated.png"
        Image.new("RGB", (64, 64), "blue").save(output)
        self.compute.return_value = [str(output)]
        result = api_runtime.execute_chat_image_tool(
            action, gen_data={"_job": self.item}, pipeline=self.pipeline, prepared=prepared
        )
        self.assertEqual(result["images"], [str(output)])
        self.assertIn("_editor_original", self.item.state)
        self.assertEqual(list((self.folder / "sessions").glob("*.json")), [])
        operation = lambda job, task_id: None
        operation.owner = self.item.owner
        job = api_runtime.Job(operation)
        self.item.external = job
        self.assertTrue(job._terminal("result", result))
        kind, published = job.events.get_nowait()
        self.assertEqual(kind, "result")
        self.assertIn("editor", published)
        self.assertEqual(published["editor"]["current"]["frame"]["width"], 64)
        self.assertNotIn(str(self.folder), json.dumps(published["editor"]))
        self.assertFalse(job._terminal("result", result))

    def test_restoration_failure_keeps_generated_artifact_out_of_completed_history(self):
        action = request(self.settings)
        prepared = api_runtime.prepare_chat_image_tool(
            action, gen_data={"_job": self.item}, pipeline=self.pipeline
        )
        output = self.folder / "unpublished.png"
        Image.new("RGB", (64, 64), "blue").save(output)
        self.compute.return_value = [str(output)]

        @contextmanager
        def failed_restoration(*args):
            yield
            raise RuntimeError("Chat restoration failed")

        self.worker.image_handoff = failed_restoration
        terminal = dispatch_image_tool(
            action,
            lambda value: api_runtime.execute_chat_image_tool(
                value,
                gen_data={"_job": self.item},
                pipeline=self.pipeline,
                prepared=prepared,
            ),
        )
        self.assertEqual((terminal["status"], terminal["call_id"]), ("error", action.call_id))
        self.assertIn("restoration failed", terminal["error"]["message"])
        self.assertNotIn("_editor_original", self.item.state)
        self.assertEqual(list((self.folder / "sessions").glob("*.json")), [])

    def test_cancel_after_compute_cannot_stage_a_completed_original(self):
        action = request(self.settings)
        prepared = api_runtime.prepare_chat_image_tool(
            action, gen_data={"_job": self.item}, pipeline=self.pipeline
        )
        output = self.folder / "cancelled.png"
        Image.new("RGB", (64, 64), "blue").save(output)

        def compute(data):
            self.item.cancelled.set()
            return [str(output)]

        self.worker._process = compute
        terminal = dispatch_image_tool(
            action,
            lambda value: api_runtime.execute_chat_image_tool(
                value,
                gen_data={"_job": self.item},
                pipeline=self.pipeline,
                prepared=prepared,
            ),
        )
        self.assertEqual(terminal["status"], "cancelled")
        self.assertNotIn("_editor_original", self.item.state)
        self.assertEqual(list((self.folder / "sessions").glob("*.json")), [])


class RFImageToolHTTPTests(unittest.TestCase):
    def setUp(self):
        self.backend = SimpleNamespace(
            image_tool=Mock(), open_editor_session=Mock(), editor_session_state=Mock()
        )
        api, _ = create_apps(self.backend, "fixture-key")
        app = FastAPI()
        app.mount("/v1", api)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.headers = {"Authorization": "Bearer fixture-key"}

    def job(self, value):
        events = queue.Queue()
        events.put(("result", value))
        return SimpleNamespace(events=events, cancelled=threading.Event())

    def payload(self, **options):
        value = dict(
            call={
                "id": "call_1",
                "type": "function",
                "function": {"name": "generate_image", "arguments": '{"prompt":"two cups"}'},
            },
            intent="preview",
        )
        value.update(options)
        return value

    def test_auth_strict_authority_and_server_owned_identity(self):
        response = self.client.post("/v1/rf/image-tools", json=self.payload())
        self.assertEqual(response.status_code, 401)
        for extra in (
            {"owner": ["ui", "foreign"]},
            {"settings": {"base_model_name": "arbitrary"}},
            {"intent": "execute-shell"},
        ):
            response = self.client.post(
                "/v1/rf/image-tools", headers=self.headers, json=self.payload(**extra)
            )
            self.assertEqual(response.status_code, 400)
        self.backend.image_tool.assert_not_called()
        self.backend.image_tool.return_value = self.job(
            {"status": "preview", "call_id": "call_1", "images": []}
        )
        response = self.client.post("/v1/rf/image-tools", headers=self.headers, json=self.payload())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "preview")
        self.assertEqual(
            self.backend.image_tool.call_args.kwargs["owner"][:2], ("api", "image-tools")
        )

    def test_json_and_sse_publish_one_typed_terminal_without_internal_paths(self):
        value = {
            "status": "completed",
            "call_id": "call_1",
            "images": [],
            "text": "actual observation",
        }
        for stream in (False, True):
            completed = self.job(value)
            self.backend.image_tool.return_value = completed
            response = self.client.post(
                "/v1/rf/image-tools", headers=self.headers, json=self.payload(stream=stream)
            )
            self.assertEqual(response.status_code, 200)
            self.assertNotIn('"images"', response.text)
            if stream:
                rows = [line for line in response.text.splitlines() if line.startswith("data: ")]
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[-1], "data: [DONE]")
                self.assertEqual(json.loads(rows[0][6:])["call_id"], "call_1")
            else:
                self.assertEqual(response.json()["text"], "actual observation")
            self.assertTrue(completed.cancelled.is_set())

    def test_preview_readiness_survives_json_and_sse_without_local_paths(self):
        readiness = {
            "state": "missing",
            "missing_bytes": 1234,
            "requirements": [
                {
                    "role": "vae",
                    "name": "selected-vae.safetensors",
                    "status": "missing",
                    "source": "trusted catalogue",
                    "integrity": "publisher sha256",
                    "reason": "Run prepares the selected required role.",
                    "source_path": "C:/private/model-cache/selected-vae.safetensors",
                }
            ],
        }
        for stream in (False, True):
            value = {"status": "preview", "call_id": "call_1", "images": [], "readiness": readiness}
            completed = self.job(value)
            self.backend.image_tool.return_value = completed
            response = self.client.post(
                "/v1/rf/image-tools", headers=self.headers, json=self.payload(stream=stream)
            )
            self.assertEqual(response.status_code, 200)
            if stream:
                rows = [line for line in response.text.splitlines() if line.startswith("data: ")]
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[-1], "data: [DONE]")
                public = json.loads(rows[0][6:])
            else:
                public = response.json()
            self.assertEqual(public["call_id"], "call_1")
            self.assertEqual(public["readiness"]["missing_bytes"], 1234)
            requirement = public["readiness"]["requirements"][0]
            self.assertEqual((requirement["role"], requirement["status"]), ("vae", "missing"))
            self.assertEqual(requirement["reason"], "Run prepares the selected required role.")
            self.assertNotIn("source_path", requirement)
            self.assertNotIn("C:/private", response.text)
            self.assertNotIn("images", public)
            self.assertTrue(completed.cancelled.is_set())


if __name__ == "__main__":
    unittest.main()
