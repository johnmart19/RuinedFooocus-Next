"""Production lifecycle contracts using barriers and fake devices, never inference."""

import ast
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.app import api_runtime
from ruinedfooocus.app.preparation_admission import PreparationAdmission
from ruinedfooocus.app import resource_lifecycle as life
from ruinedfooocus.backends.onnx import acquisition
from ruinedfooocus.features.image_processing import auxiliary_resources as auxiliary
from ruinedfooocus.features.models.component_profiles import checkpoint_identity
from ruinedfooocus.features.models.lora_sources import capture_request as capture_loras
from test_powerup_family_controls import functions as powerup_functions

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def generation_callback(scope):
    tool_ui, _ = powerup_functions()
    scope = {
        **scope,
        "ui_controlnet": SimpleNamespace(
            tool_kind=tool_ui["tool_kind"], family_for_checkpoint=lambda _: "SDXL"
        ),
        "settings": {},
    }
    shared = scope.setdefault("shared", SimpleNamespace(state={}))
    shared.models = SimpleNamespace()
    shared.settings = SimpleNamespace(default_settings={})
    scope["generate_request"] = function("ruinedfooocus/ui/app.py", "generate_request", scope)
    # These callers exercise submission/stream/admission ownership. The UI's
    # generate_clicked adapter now captures an Editor version separately.
    return lambda *values: scope["generate_request"](
        dict(zip(scope["state"]["ctrls_name"], values))
    )


class JobChannelTests(unittest.TestCase):
    def test_two_owners_cannot_poll_or_cancel_each_others_jobs(self):
        jobs = life.JobRegistry()
        a, b = jobs.create(owner=("ui", "a")), jobs.create(owner=("ui", "b"))
        jobs.publish(a.task_id, "preview", "private A")
        jobs.publish(b.task_id, "results", "private B")
        for action in (
            lambda: jobs.result(b.task_id, owner=a.owner),
            lambda: jobs.cancel(b.task_id, owner=a.owner),
        ):
            with self.assertRaises(PermissionError):
                action()
        self.assertFalse(b.cancelled.is_set())
        self.assertEqual(jobs.result(a.task_id, owner=a.owner), ("preview", "private A"))
        self.assertEqual(jobs.result(b.task_id, owner=b.owner), ("results", "private B"))

    def test_no_context_stop_is_noop_and_session_stop_is_scoped(self):
        jobs = life.JobRegistry()
        a, b = jobs.create(owner=("ui", "a")), jobs.create(owner=("ui", "b"))
        with patch.object(life, "current_owner", return_value=None):
            self.assertFalse(jobs.cancel())
        with patch.object(life, "current_owner", return_value=a.owner):
            self.assertTrue(jobs.cancel())
        self.assertTrue(a.cancelled.is_set())
        self.assertFalse(b.cancelled.is_set())

    def test_channel_coalesces_progress_and_terminal_cannot_be_overwritten(self):
        jobs = life.JobRegistry(channel_capacity=3)
        item = jobs.create(owner=("ui", "a"))
        for index in range(100):
            jobs.publish(item.task_id, "preview", index)
        jobs.publish(item.task_id, "error", "one failure")
        jobs.publish(item.task_id, "results", [])
        self.assertFalse(jobs.publish(item.task_id, "preview", "late"))
        self.assertEqual(
            list(item.events), [("preview", 99), ("error", "one failure"), ("results", [])]
        )

    def test_default_stop_prefers_owners_active_job_over_later_queued_job(self):
        jobs = life.JobRegistry()
        active = jobs.create(owner=("ui", "a"))
        jobs.enqueue(active, {})
        jobs.take(timeout=0)
        queued = jobs.create(owner=active.owner)
        other = jobs.create(owner=("ui", "b"))
        jobs.cancel(owner=active.owner)
        self.assertTrue(active.cancelled.is_set())
        self.assertFalse(queued.cancelled.is_set())
        self.assertFalse(other.cancelled.is_set())

    def test_cancelled_queue_entry_and_finished_capacity_are_released_once(self):
        jobs = life.JobRegistry(capacity=1, retained=2)
        item = jobs.create(owner=("ui", "a"))
        jobs.enqueue(item, {"payload": "A"})
        with self.assertRaisesRegex(RuntimeError, "full"):
            jobs.create()
        callback = Mock()
        item.on_finish = callback
        jobs.cancel(item.task_id, owner=item.owner)
        self.assertIsNone(jobs.take(timeout=0))
        self.assertFalse(jobs.finish(item))
        callback.assert_called_once()
        self.assertIsNotNone(jobs.create())

    def test_worker_death_fails_ready_jobs_and_releases_api_terminal_once(self):
        jobs = life.JobRegistry()
        item = jobs.create(owner=("ui", "a"))
        external = Mock()
        item.external = external
        jobs.enqueue(item, {})
        jobs.fail_ready("worker died")
        external.fail.assert_called_once_with("worker died")
        self.assertTrue(item.finished)
        self.assertEqual(
            jobs.result(item.task_id, owner=item.owner), ("results", {"error": "worker died"})
        )

    def test_preview_and_interrupt_state_are_job_local(self):
        state = life.RuntimeState({"pipeline": "process-wide service cache"})
        jobs = life.JobRegistry()
        a, b = jobs.create(owner=("ui", "a")), jobs.create(owner=("ui", "b"))
        with life.job_scope(a):
            state.update(interrupted=True, preview_total=7, last_image="A.png")
        with life.job_scope(b):
            state.update(interrupted=False, preview_total=1, last_image="B.png")
            self.assertEqual(state["last_image"], "B.png")
        with life.job_scope(a):
            self.assertTrue(state["interrupted"])
            self.assertEqual(state["preview_total"], 7)
            self.assertEqual(state["last_image"], "A.png")


class AdmissionTests(unittest.TestCase):
    def test_conflicting_waiters_are_fifo_and_other_fake_device_proceeds(self):
        admission, jobs = life.FairAdmission(), life.JobRegistry()
        a, b, c, d = [jobs.create() for _ in range(4)]
        first, second, independent = threading.Event(), threading.Event(), threading.Event()
        order = []

        def waiting(item, event):
            with admission.lease(item, ("device:A",), on_wait=event.set):
                order.append(item.task_id)

        with ThreadPoolExecutor(max_workers=3) as executor:
            with admission.lease(a, ("device:A",)):
                fb = executor.submit(waiting, b, first)
                self.assertTrue(first.wait(2))
                fc = executor.submit(waiting, c, second)
                self.assertTrue(second.wait(2))

                def other():
                    with admission.lease(d, ("device:B",)):
                        independent.set()

                fd = executor.submit(other)
                self.assertTrue(independent.wait(2))
            fb.result(2), fc.result(2), fd.result(2)
        self.assertEqual(order, [b.task_id, c.task_id])
        self.assertFalse(admission.held)

    def test_compound_stage_reenters_before_waiter_that_needs_its_parent_lease(self):
        admission, jobs = life.FairAdmission(), life.JobRegistry()
        a, b = jobs.create(), jobs.create()
        waiting = threading.Event()

        def other():
            with admission.lease(b, ("compute:shared", "service:image"), on_wait=waiting.set):
                return True

        with ThreadPoolExecutor(max_workers=1) as executor:
            with admission.lease(a, ("compute:shared", "service:chat")):
                future = executor.submit(other)
                self.assertTrue(waiting.wait(2))
                with admission.lease(a, ("compute:shared", "service:image")):
                    self.assertEqual(admission.held["compute:shared"][1], 2)
            self.assertTrue(future.result(2))
        self.assertFalse(admission.held)

    def test_waiting_cancellation_releases_ticket_without_touching_active_owner(self):
        admission, jobs = life.FairAdmission(), life.JobRegistry()
        a, b = jobs.create(), jobs.create()
        waiting = threading.Event()

        def other():
            with admission.lease(b, ("device:A",), on_wait=waiting.set):
                raise AssertionError("cancelled waiter ran")

        with ThreadPoolExecutor(max_workers=1) as executor:
            with admission.lease(a, ("device:A",)):
                future = executor.submit(other)
                self.assertTrue(waiting.wait(2))
                b.cancelled.set()
                with self.assertRaises(CancelledError):
                    future.result(2)
                self.assertEqual(admission.held["device:A"][0], a.identity.token)
        self.assertFalse(admission.waiters)

    def test_production_service_transition_releases_only_inactive_owner_and_restores_parent(self):
        jobs = life.JobRegistry()
        item = jobs.create()
        state = life.RuntimeState({"pipeline": None})
        chat, image = Mock(), Mock()
        with (
            patch.object(life, "admission", life.FairAdmission()),
            patch.object(life, "_PIPELINES", {}),
            patch.object(life, "_RESIDENT", None),
            patch.object(life, "_release_native") as release_native,
            patch.object(auxiliary, "release_except"),
        ):
            with life.job_scope(item), life.service_scope("chat", state):
                state["pipeline"] = chat
                life.remember_pipeline(chat)
                with life.service_scope("image", state):
                    chat.unload.assert_called_once()
                    state["pipeline"] = image
                    life.remember_pipeline(image)
                self.assertIs(state["pipeline"], chat)
                release_native.assert_called_once()
                self.assertIs(life.service_pipeline("image"), image)
            self.assertFalse(life.admission.held)

    def test_unmanaged_native_release_is_refused_without_eviction(self):
        with patch.object(life, "_release_native") as release:
            with self.assertRaises(life.OwnershipError):
                life.release_native_for_external()
            release.assert_not_called()


class CapacityTests(unittest.TestCase):
    def test_cancelled_queued_preparation_returns_slot_without_running(self):
        admission = PreparationAdmission(max_workers=1, max_jobs=2)
        entered, release = threading.Event(), threading.Event()
        failed = Mock()

        def blocked():
            entered.set()
            self.assertTrue(release.wait(3))

        try:
            active = admission.submit(blocked, lambda value: None, failed)
            self.assertTrue(entered.wait(2))
            queued = admission.submit(
                lambda: self.fail("cancelled preparation ran"), Mock(), failed
            )
            self.assertTrue(queued.cancel())
            replacement = admission.submit(lambda: "replacement", Mock(), failed)
            release.set()
            active.result(2), replacement.result(2)
            self.assertEqual(failed.call_count, 1)
            self.assertIsInstance(failed.call_args.args[0], CancelledError)
        finally:
            release.set()
            admission.close()

    def test_api_pending_cancel_and_late_run_release_capacity_exactly_once(self):
        slots = threading.BoundedSemaphore(1)
        with patch.object(api_runtime, "slots", slots):
            operation = Mock()
            job = api_runtime.Job(operation)
            self.assertFalse(slots.acquire(False))
            job.cancelled.set()
            job.cancelled.set()
            job.run(1)
            operation.assert_not_called()
            self.assertTrue(slots.acquire(False))
            self.assertFalse(slots.acquire(False))
            slots.release()

    def test_api_active_cancel_keeps_capacity_until_operation_unwinds(self):
        slots = threading.BoundedSemaphore(1)
        entered, release = threading.Event(), threading.Event()

        def run(job, task_id):
            entered.set()
            self.assertTrue(release.wait(3))
            return "must not publish cancelled result"

        with (
            patch.object(api_runtime, "slots", slots),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            job = api_runtime.Job(run)
            future = executor.submit(job.run, 1)
            self.assertTrue(entered.wait(2))
            job.cancelled.set()
            self.assertFalse(slots.acquire(False))
            release.set()
            future.result(2)
            events = []
            while not job.events.empty():
                events.append(job.events.get_nowait())
            self.assertFalse(any(kind == "result" for kind, value in events))
            self.assertEqual(events, [("error", "Request cancelled.")])
            self.assertTrue(slots.acquire(False))
            slots.release()

    def test_active_auxiliary_call_cannot_be_closed_by_bulk_transition(self):
        entered, release = threading.Event(), threading.Event()

        class Resource:
            resource_group = "rembg"
            close = Mock()

            @auxiliary.using
            def predict(self):
                entered.set()
                self.assert_release = release.wait(3)

        resource = Resource()
        auxiliary.register(resource)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(resource.predict)
            self.assertTrue(entered.wait(2))
            with self.assertRaisesRegex(RuntimeError, "in use"):
                auxiliary.release_except()
            resource.close.assert_not_called()
            release.set()
            future.result(2)
        auxiliary.release_except()
        resource.close.assert_called_once()


class CallerTests(unittest.TestCase):
    def test_main_stop_targets_its_bound_task_in_both_chat_queue_orders(self):
        for main_active in (True, False):
            with self.subTest(main_active=main_active):
                jobs = life.JobRegistry()
                first = jobs.create(owner=("ui", "a"))
                jobs.enqueue(first, {})
                jobs.take(timeout=0)
                second = jobs.create(owner=first.owner)
                other = jobs.create(owner=("ui", "b"))
                main, chat = (first, second) if main_active else (second, first)
                stop = function(
                    "ruinedfooocus/ui/app.py",
                    "stop_clicked",
                    {
                        "worker": SimpleNamespace(
                            interrupt_processing=lambda task_id: jobs.cancel(
                                task_id, owner=main.owner
                            )
                        )
                    },
                )
                stop(None)
                self.assertFalse(any(item.cancelled.is_set() for item in (main, chat, other)))
                stop(main.task_id)
                self.assertTrue(main.cancelled.is_set())
                self.assertFalse(chat.cancelled.is_set())
                self.assertFalse(other.cancelled.is_set())

    def test_generation_yields_bound_id_clears_completion_and_cancels_disconnect(self):
        for disconnected in (False, True):
            with self.subTest(disconnected=disconnected):
                worker = SimpleNamespace(
                    task_result=lambda task_id: ("results", []), interrupt_processing=Mock()
                )
                callback = generation_callback(
                    {
                        "state": {"ctrls_name": ["image_number", "preset_selection"]},
                        "generation_task": "owned-task",
                        "worker": worker,
                        "shared": SimpleNamespace(state={}),
                        "append_work": lambda data: 42,
                        "update_clicked": lambda: {},
                        "update_results": lambda product: {},
                        "time": SimpleNamespace(sleep=lambda seconds: None),
                    },
                )
                stream = callback(1, None)
                self.assertIsNone(next(stream)["owned-task"])
                self.assertEqual(next(stream), {"owned-task": 42})
                if disconnected:
                    stream.close()
                    worker.interrupt_processing.assert_called_once_with(42)
                else:
                    self.assertIsNone(next(stream)["owned-task"])
                    self.assertEqual(list(stream), [])
                    worker.interrupt_processing.assert_not_called()

    def test_submission_failure_restores_main_controls_without_canceling_other_work(self):
        warning, cancel = Mock(), Mock()
        callback = generation_callback(
            {
                "state": {"ctrls_name": ["image_number", "preset_selection"]},
                "generation_task": "owned-task",
                "worker": SimpleNamespace(interrupt_processing=cancel),
                "append_work": Mock(side_effect=RuntimeError("Compute queue is full")),
                "update_clicked": lambda: {"running": True},
                "update_results": lambda product: {"running": False},
                "gr": SimpleNamespace(Warning=warning),
            },
        )
        updates = list(callback(1, None))
        self.assertEqual(updates[-1], {"running": False, "owned-task": None})
        warning.assert_called_once()
        cancel.assert_not_called()

    def test_inline_handoff_restores_routine_failure_but_defers_cancel_or_runtime_failure(self):
        for outcome in ("success", "routine", "cancel", "runtime"):
            with self.subTest(outcome=outcome):
                item = life.JobRegistry().create(owner=("ui", "a"))
                chat = SimpleNamespace(unload=Mock(), load_base_model=Mock())
                shared = SimpleNamespace(state={})

                @contextmanager
                def stage(data):
                    with life.job_scope(item):
                        yield

                handoff = contextmanager(
                    function(
                        "ruinedfooocus/app/async_worker.py",
                        "image_handoff",
                        {"chat_stage": stage, "current_job": life.current_job, "shared": shared},
                    )
                )
                data = {"history": [{"role": "user", "content": "keep this"}]}
                try:
                    with handoff(data, chat, "original", False):
                        chat.unload.assert_called_once()
                        if outcome == "cancel":
                            item.cancelled.set()
                        if outcome == "routine":
                            raise ValueError("unsupported request")
                        if outcome == "runtime":
                            raise RuntimeError("device failed")
                except (ValueError, RuntimeError):
                    pass
                self.assertEqual(data["history"], [{"role": "user", "content": "keep this"}])
                self.assertIs(shared.state["pipeline"], chat)
                if outcome in ("cancel", "runtime"):
                    chat.load_base_model.assert_not_called()
                    self.assertEqual(data["_chat_restoration_pending"], "original")
                else:
                    chat.load_base_model.assert_called_once_with("original", vision=False)

    def test_successful_stream_preserves_full_chunk_queue_and_one_result(self):
        slots = threading.BoundedSemaphore(1)

        def run(job, task_id):
            for index in range(64):
                job.emit(index)
            return "complete"

        with patch.object(api_runtime, "slots", slots):
            job = api_runtime.Job(run)
            job.run(1)
        self.assertEqual(
            [job.events.get_nowait() for _ in range(65)],
            [("chunk", index) for index in range(64)] + [("result", "complete")],
        )
        self.assertFalse(job._terminal("result", "duplicate"))
        self.assertTrue(slots.acquire(False))

    def test_ui_review_cancellation_or_runtime_failure_defers_original_reload(self):
        for cancellation, error in ((True, None), (False, RuntimeError("device failed"))):
            with self.subTest(cancellation=cancellation):
                item = life.JobRegistry().create(owner=("ui", "a"))
                reviewer = SimpleNamespace(vision_projector=None, llm=object(), unload=Mock())

                def complete(*args, **kwargs):
                    if cancellation:
                        item.cancelled.set()
                    if error:
                        raise error
                    return "review"

                reviewer.load_base_model = Mock()
                reviewer.complete_text = complete
                inspect = function(
                    "ruinedfooocus/features/chat/llama_pipeline.py",
                    "inspect_image",
                    {
                        "worker": SimpleNamespace(
                            current_job=lambda: item, check_interrupt=lambda data: None
                        )
                    },
                )
                data = {"task_id": 1}
                if error:
                    with self.assertRaises(RuntimeError):
                        inspect(reviewer, [], "reviewer", "original", data)
                else:
                    inspect(reviewer, [], "reviewer", "original", data)
                self.assertEqual(reviewer.load_base_model.call_args.args, ("reviewer",))
                reviewer.load_base_model.assert_called_once()
                reviewer.unload.assert_called_once()
                self.assertTrue(data["_chat_restoration_pending"])

    def test_frozen_performance_recipe_survives_a_barrier_and_live_override(self):
        recipes = {"steps": 12}
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        checkpoint = root / "fixture.safetensors"
        header = b'{"conditioner.embedders.0.x":{},"first_stage_model.x":{}}'
        checkpoint.write_bytes(len(header).to_bytes(8, "little") + header)
        shared = SimpleNamespace(
            models=SimpleNamespace(
                get_models_by_path=lambda *a, **k: {},
                get_model_base=lambda info: "SDXL",
                get_file=lambda *a: checkpoint,
            ),
            performance_settings=SimpleNamespace(
                apply=lambda data, family: dict(data, custom_steps=recipes["steps"])
            ),
            path_manager=SimpleNamespace(
                paths={},
                DOWNLOADABLE_FILES={},
                get_abspath=Path,
                get_model_folders=lambda category: [root],
            ),
            resolution_settings=object(),
        )
        capture = function(
            "ruinedfooocus/app/async_worker.py",
            "capture_image_data",
            {
                "shared": shared,
                "settings": SimpleNamespace(default_settings={}),
                "control_tools": SimpleNamespace(get_settings=lambda data: {}),
            },
        )

        def snapshot_components(data, *args):
            # Retain the production typed prompt/LoRA and checkpoint identity
            # contract while this fixture excludes component model discovery.
            return capture_loras(
                dict(data, _image_component_scope=checkpoint_identity(checkpoint)), shared.models
            )

        with (
            patch(
                "ruinedfooocus.features.models.component_profiles.snapshot_request",
                side_effect=snapshot_components,
            ),
            patch(
                "ruinedfooocus.features.generation.sdcpp_pipeline.capture_image_request",
                side_effect=lambda data, *a, **k: data,
            ),
            patch(
                "ruinedfooocus.features.generation.image_preparation.capture_image_assets",
                side_effect=lambda data, *a, **k: data,
            ),
        ):
            queued = capture(
                {
                    "base_model_name": "A",
                    "performance_selection": "Model Defaults",
                    "prompt": "captured scene",
                }
            )
        entered, release = threading.Event(), threading.Event()
        observed = []

        class ReachedPipeline(Exception):
            pass

        def selected(data):
            observed.append(data["custom_steps"])
            raise ReachedPipeline()

        process = function(
            "ruinedfooocus/app/async_worker.py",
            "_process",
            {
                "check_interrupt": lambda data: None,
                "shared": shared,
                "process_metadata": lambda data: data,
                "control_tools": SimpleNamespace(
                    get_settings=lambda data: data.get("controlnet") or {}
                ),
                "pipelines": SimpleNamespace(update=selected),
            },
        )

        def compute():
            entered.set()
            self.assertTrue(release.wait(3))
            with self.assertRaises(ReachedPipeline):
                process(queued)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(compute)
            self.assertTrue(entered.wait(2))
            recipes["steps"] = 99
            release.set()
            future.result(2)
        self.assertEqual(observed, [12])

    def test_cancellable_acquisition_propagates_env_cwd_and_other_spawn_kwargs(self):
        process, job = Mock(returncode=0), Mock()
        process.poll.return_value = 0
        with patch(
            "ruinedfooocus.core.child_process.start_process", return_value=(process, job)
        ) as start:
            acquisition.run(
                ["inert"],
                check_cancelled=lambda: None,
                timeout=1,
                capture_output=True,
                env={"CHILD_ONLY": "yes"},
                cwd="owned",
                text=True,
                encoding="utf-8",
            )
        self.assertEqual(start.call_args.kwargs["env"], {"CHILD_ONLY": "yes"})
        self.assertEqual(start.call_args.kwargs["cwd"], "owned")
        self.assertTrue(start.call_args.kwargs["text"])
        self.assertEqual(start.call_args.kwargs["encoding"], "utf-8")
        job.close.assert_called_once()

    def test_logout_cancels_only_jobs_bound_to_that_valid_server_login(self):
        from fastapi import FastAPI, Request
        from starlette.responses import RedirectResponse
        from starlette.testclient import TestClient
        from ruinedfooocus.integrations.web_access import PrivateResponses, authenticated_owner
        from starlette.requests import Request as RawRequest

        app = FastAPI()
        app.cookie_id, app.tokens = "fixture", {"login-A": "same-user", "login-B": "same-user"}

        @app.get("/logout")
        def logout(request: Request):
            app.tokens.pop(request.cookies.get("access-token-fixture"), None)
            return RedirectResponse("/", status_code=302)

        def owner(token):
            scope = {
                "type": "http",
                "app": app,
                "headers": [(b"cookie", f"access-token-fixture={token}".encode())],
            }
            return authenticated_owner(RawRequest(scope))

        jobs = life.JobRegistry()
        a, b = (
            jobs.create(owner=owner("login-A") + ("session-A",)),
            jobs.create(owner=owner("login-B") + ("session-B",)),
        )
        app.add_middleware(PrivateResponses)
        with patch.object(life, "registry", jobs), TestClient(app) as client:
            response = client.get(
                "/logout", cookies={"access-token-fixture": "login-A"}, follow_redirects=False
            )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(a.cancelled.is_set())
        self.assertFalse(b.cancelled.is_set())


if __name__ == "__main__":
    unittest.main()
