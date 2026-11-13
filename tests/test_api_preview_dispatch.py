"""API passive tools bypass an inert blocked compute worker, never its ownership."""

import copy
import json
import queue
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.app import api_activity, api_runtime as api, resource_lifecycle as life
from ruinedfooocus.app.preparation_admission import PreparationAdmission
from ruinedfooocus.features.chat.chat_tools import (
    ChatPermissionSession,
    ImageToolContext,
    validate_image_tool_call,
)
from ui_test_helpers import function_from


OWNER = ("api", "preview-fixture")
CAPABILITY = {"available": True, "binding": "captured-template"}


class ApiPreviewDispatchTests(unittest.TestCase):
    def setUp(self):
        self.registry, self.admission = life.JobRegistry(), life.FairAdmission()
        self.pool = PreparationAdmission(max_workers=1, max_jobs=3)
        self.addCleanup(self.pool.close)
        self.activity = api_activity.ApiActivity()
        self.api_slots = threading.BoundedSemaphore(8)
        self.enterContext(patch.object(life, "registry", self.registry))
        self.enterContext(patch.object(life, "admission", self.admission))
        self.enterContext(patch.object(life, "_RESIDENT", None))
        self.enterContext(patch.object(life, "_PIPELINES", {}))
        self.enterContext(patch.object(api, "slots", self.api_slots))
        self.enterContext(patch.object(api_activity, "activity", self.activity))
        self.enterContext(
            patch("ruinedfooocus.app.preparation_admission.preparation_admission", self.pool)
        )
        self.release_native = self.enterContext(patch.object(life, "_release_native"))
        self.release_service = self.enterContext(patch.object(life, "_release_service"))
        self.pipeline = SimpleNamespace(
            image_tool_capability=lambda: dict(CAPABILITY),
            vision_projector=None,
            _prepared_request=SimpleNamespace(model="unchanged-chat.gguf"),
        )
        self.enterContext(patch.object(life, "service_pipeline", return_value=self.pipeline))
        self.settings = {"prompt": "old draft", "negative": "keep exclusions", "image_number": 1}
        self.enterContext(patch.object(api, "image_defaults", self.settings))
        self.enterContext(patch.object(api, "_image_runtime_snapshot", return_value=None))
        self.shared = SimpleNamespace(state={})
        self.enterContext(patch.dict(sys.modules, {"ruinedfooocus.app.state": self.shared}))
        self.enterContext(
            patch.object(sys.modules["ruinedfooocus.app"], "state", self.shared, create=True)
        )
        self.captured = []

        def capture(data):
            item = life.current_job()
            self.assertEqual(item.owner, OWNER)
            with self.assertRaises(life.OwnershipError):
                life.require_compute_owner()
            self.captured.append(copy.deepcopy(data))
            return dict(data)

        self.worker = SimpleNamespace(
            capture_image_data=Mock(side_effect=capture),
            prepare_image_data=Mock(side_effect=AssertionError("Preview acquired image assets")),
            _process=Mock(side_effect=AssertionError("Preview invoked compute")),
        )
        self.worker.add_task = Mock(side_effect=self.enqueue_compute)
        self.enterContext(patch.dict(sys.modules, {"ruinedfooocus.app.async_worker": self.worker}))
        self.handler = function_from(
            "ruinedfooocus/app/async_worker.py",
            "handler",
            {"service_scope": life.service_scope, "shared": self.shared},
        )

    def payload(self, *, intent="preview", prompt="Exactly two blue cups", proposal=False):
        return {
            "intent": intent,
            "update_create_prompt": proposal,
            "call": {
                "id": "call_preview_1",
                "type": "function",
                "function": {
                    "name": "propose_create_prompt" if proposal else "generate_image",
                    "arguments": json.dumps({"prompt": prompt}),
                },
            },
        }

    def enqueue_compute(self, data):
        job = data["_api_job"]
        item = self.registry.create(owner=job.owner, cancelled=job.cancelled)
        item.external, job.task_id = job, item.task_id
        self.registry.enqueue(item, dict(data, task_id=item.task_id, _job=item))
        return item.task_id

    def finish_pool(self):
        self.pool.submit_work(lambda: None).result(timeout=3)

    def assert_released_once(self, slots=None, count=8):
        slots = slots or self.api_slots
        for _ in range(count):
            self.assertTrue(slots.acquire(blocking=False))
        self.assertFalse(slots.acquire(blocking=False))
        for _ in range(count):
            slots.release()

    def occupy_pool(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def block():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Fixture pool barrier timed out")

        future = self.pool.submit_work(block)
        self.assertTrue(entered.wait(2))
        return release, future

    def test_preview_and_literal_proposal_finish_before_blocked_compute_is_released(self):
        entered, release, second_entered = threading.Event(), threading.Event(), threading.Event()
        stop, failures = threading.Event(), []

        def heavy(job, task_id):
            self.assertIs(life.require_compute_owner(), life.current_job())
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Fixture compute barrier timed out")
            return "heavy finished"

        def second(job, task_id):
            self.assertIs(life.require_compute_owner(), life.current_job())
            second_entered.set()
            return "second finished"

        heavy.owner, heavy.resource_group = ("api", "heavy"), "image"
        second.owner, second.resource_group = ("api", "second"), "image"

        def serial_worker():
            while not stop.is_set():
                selected = self.registry.take(timeout=0.02)
                if selected is None:
                    continue
                item, data = selected
                try:
                    with life.job_scope(item):
                        self.handler(data)
                except BaseException as error:
                    failures.append(error)
                finally:
                    self.registry.finish(item)

        heavy_job = api.submit(heavy)
        thread = threading.Thread(target=serial_worker)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            queued = api.submit(second)
            compute_item = self.registry.get(heavy_job.task_id, owner=heavy.owner)
            preview = api.image_tool(self.payload(), owner=OWNER)
            value = api.wait_for_result(preview, timeout=2)
            proposal = api.image_tool(self.payload(intent="text", proposal=True), owner=OWNER)
            prompt = api.wait_for_result(proposal, timeout=2)
            self.finish_pool()
            self.assertEqual((value["status"], value["call_id"]), ("preview", "call_preview_1"))
            self.assertEqual(value["readiness"]["state"], "ready")
            self.assertEqual(prompt["prompt_proposal"]["prompt"], "Exactly two blue cups")
            self.assertEqual(self.captured[0]["prompt"], "Exactly two blue cups")
            self.assertFalse(release.is_set())
            self.assertFalse(second_entered.is_set())
            self.assertEqual(self.admission.held["compute:shared"][0], compute_item.identity.token)
            self.assertEqual(len(self.registry.ready), 1)
            self.assertTrue(self.registry.get(preview.task_id, owner=OWNER).finished)
            self.assertTrue(self.registry.get(proposal.task_id, owner=OWNER).finished)
            self.assertEqual(self.worker.add_task.call_count, 2)
            self.release_native.assert_not_called()
            self.release_service.assert_not_called()
            release.set()
            self.assertEqual(api.wait_for_result(heavy_job, timeout=2), "heavy finished")
            self.assertEqual(api.wait_for_result(queued, timeout=2), "second finished")
            self.assertTrue(second_entered.is_set())
        finally:
            release.set()
            stop.set()
            thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assert_released_once()

    def test_queued_cancel_keeps_owner_fence_and_releases_both_capacities(self):
        release, blocker = self.occupy_pool()
        job = api.image_tool(self.payload(), owner=OWNER)
        item = self.registry.get(job.task_id, owner=OWNER)
        with self.assertRaises(PermissionError):
            self.registry.cancel(job.task_id, owner=("api", "foreign"))
        job.cancelled.set()
        self.assertEqual(job.events.get(timeout=1), ("error", "Request cancelled."))
        self.assertTrue(item.finished)
        release.set()
        blocker.result(timeout=3)
        self.finish_pool()
        self.worker.capture_image_data.assert_not_called()
        self.assertTrue(job.events.empty())
        self.assertEqual(self.activity.snapshot()["counts"], {"cancelled": 1})
        self.assert_released_once()

    def test_running_cancel_acknowledges_once_and_holds_capacity_until_callback_returns(self):
        slots = threading.BoundedSemaphore(1)
        self.enterContext(patch.object(api, "slots", slots))
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def capture(data):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Fixture preview barrier timed out")
            return dict(data)

        self.worker.capture_image_data.side_effect = capture
        job = api.image_tool(self.payload(), owner=OWNER)
        self.assertTrue(entered.wait(2))
        self.assertTrue(job.running)
        job.cancelled.set()
        self.assertEqual(job.events.get(timeout=1), ("error", "Request cancelled."))
        self.assertFalse(slots.acquire(blocking=False))
        self.assertEqual(self.activity.snapshot()["counts"], {"running": 1})
        release.set()
        self.finish_pool()
        self.assertTrue(job.events.empty())
        self.assertTrue(self.registry.get(job.task_id, owner=OWNER).finished)
        self.assertEqual(self.activity.snapshot()["counts"], {"cancelled": 1})
        self.assert_released_once(slots, count=1)

    def test_capture_failure_finishes_registry_and_api_error_status_without_cleanup(self):
        for error, kind in (
            (ValueError("invalid capture"), "invalid_request"),
            (RuntimeError("failed capture"), "error"),
        ):
            with self.subTest(kind=kind):
                self.worker.capture_image_data.side_effect = error
                job = api.image_tool(self.payload(), owner=OWNER)
                self.assertEqual(job.events.get(timeout=2), (kind, str(error)))
                self.finish_pool()
                self.assertTrue(self.registry.get(job.task_id, owner=OWNER).finished)
                self.assertTrue(job.events.empty())
        self.assertEqual(self.activity.snapshot()["counts"], {"failed": 2})
        self.release_native.assert_not_called()
        self.release_service.assert_not_called()
        self.assert_released_once()

    def test_pool_overflow_releases_api_and_registry_capacity(self):
        release, blocker = self.occupy_pool()
        waiting = [self.pool.submit_work(lambda: None) for _ in range(2)]
        parent = self.registry.create(owner=("api", "other-context"))
        parent.state["_editor_original"] = "unrelated staged result"
        self.worker.publish_editor_result = Mock(
            side_effect=AssertionError("Touched parent result")
        )
        with life.job_scope(parent):
            with self.assertRaisesRegex(RuntimeError, "Preparation queue is full"):
                api.image_tool(self.payload(), owner=OWNER)
        self.assertTrue(
            all(item.finished for item in self.registry.jobs.values() if item is not parent)
        )
        self.assertFalse(parent.finished)
        self.worker.publish_editor_result.assert_not_called()
        self.assertEqual(self.activity.snapshot()["counts"], {"failed": 1})
        self.assert_released_once()
        release.set()
        blocker.result(timeout=3)
        for future in waiting:
            future.result(timeout=3)

    def test_captured_request_isolated_from_later_defaults_and_runtime_change_is_rejected(self):
        release, blocker = self.occupy_pool()
        job = api.image_tool(self.payload(prompt="fresh request only"), owner=OWNER)
        self.settings.update(prompt="unrelated later draft", negative="later exclusion")
        release.set()
        blocker.result(timeout=3)
        self.assertEqual(api.wait_for_result(job, timeout=2)["status"], "preview")
        self.assertEqual(self.captured[0]["prompt"], "fresh request only")
        self.assertEqual(self.captured[0]["negative"], "keep exclusions")
        release, blocker = self.occupy_pool()
        changed = api.image_tool(self.payload(), owner=OWNER)
        self.pipeline.image_tool_capability = lambda: dict(CAPABILITY, binding="later-template")
        release.set()
        blocker.result(timeout=3)
        kind, error = changed.events.get(timeout=2)
        self.assertEqual(kind, "invalid_request")
        self.assertIn("template changed", error)
        self.finish_pool()
        self.assertEqual(self.worker.capture_image_data.call_count, 1)
        self.assert_released_once()

    def test_run_and_arbitrary_noncompute_flags_still_use_serial_worker(self):
        job = api.image_tool(self.payload(intent="run"), owner=OWNER)
        self.assertTrue(hasattr(job.operation, "prepare"))
        self.assertTrue(getattr(job.operation, "compute_required", True))
        operation = Mock(return_value="not a whitelisted tool")
        operation.owner, operation.compute_required = OWNER, False
        other = api.submit(operation)
        self.assertEqual(self.worker.add_task.call_count, 2)
        self.assertEqual(len(self.registry.ready), 2)
        self.assertTrue(job.events.empty())
        context = ImageToolContext(OWNER, "run", {})
        request = validate_image_tool_call(self.payload(intent="run")["call"], context, CAPABILITY)
        with self.assertRaisesRegex(ValueError, "Only passive"):
            api._submit_passive_image_tool(operation, request)
        for pending in (job, other):
            pending.cancelled.set()
        self.worker.capture_image_data.assert_not_called()
        self.worker.prepare_image_data.assert_not_called()
        self.assert_released_once()

    def test_queued_prompt_copy_rechecks_revoked_permission_before_publication(self):
        permissions = ChatPermissionSession()
        policy = permissions.capture(OWNER, update_create_prompt=True)
        context = ImageToolContext(OWNER, "text", {}, permissions=policy)
        release, blocker = self.occupy_pool()
        with patch.object(api, "capture_image_tool_context", return_value=context):
            job = api.image_tool(self.payload(intent="text", proposal=True), owner=OWNER)
        permissions.revoke(OWNER)
        release.set()
        blocker.result(timeout=3)
        kind, error = job.events.get(timeout=2)
        self.assertEqual(kind, "invalid_request")
        self.assertIn("permissions were revoked", error)
        self.finish_pool()
        self.assertTrue(self.registry.get(job.task_id, owner=OWNER).finished)
        self.assertTrue(job.events.empty())
        self.assertEqual(self.activity.snapshot()["counts"], {"failed": 1})
        self.worker.capture_image_data.assert_not_called()
        self.assert_released_once()


if __name__ == "__main__":
    unittest.main()
