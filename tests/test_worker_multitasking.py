"""Concurrent requests keep their own state and are never left waiting forever."""

import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ruinedfooocus.app import resource_lifecycle as life
from ruinedfooocus.app.preparation_admission import PreparationAdmission
from ui_test_helpers import function_from

WORKER = "ruinedfooocus/app/async_worker.py"


class PromptBatchStateTests(unittest.TestCase):
    """The browser callback and the worker write the same per-owner preview state."""

    def setUp(self):
        token = life._REQUEST_OWNER.set(("ui", "", "local", "tab"))
        self.addCleanup(life._REQUEST_OWNER.reset, token)
        self.state = life.RuntimeState({"pipeline": None})
        self.jobs = life.JobRegistry()
        self.results = Mock()
        previews = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.scope = dict(
            shared=SimpleNamespace(state=self.state),
            control_tools=SimpleNamespace(get_settings=lambda data: {}),
            add_result=self.results,
            Path=Path,
            _preview_path=lambda gen_data: previews / "preview.jpg",
        )
        function_from(WORKER, "reset_preview", self.scope)
        self.process = function_from(WORKER, "process", self.scope)

    def run_job(self, data):
        item = self.jobs.create()
        # generate_request clears the flag right after add_task, before the worker
        # thread has touched this job's own state.
        self.state["interrupted"] = False
        with life.job_scope(item):
            self.process(dict(data, task_id=item.task_id))
        return item

    def test_every_prompt_of_a_list_runs_when_the_browser_reset_its_flag_first(self):
        self.scope["_process"] = Mock(side_effect=lambda data: [data["prompt"] + ".png"])
        self.run_job(dict(prompt=["first", "second"], image_total=2, show_preview=False))
        self.assertEqual(self.scope["_process"].call_count, 2)
        self.results.assert_called_once_with(1, "results", ["first.png", "second.png"])

    def test_a_second_job_of_the_same_tab_does_not_inherit_the_previous_stop(self):
        self.scope["_process"] = Mock(side_effect=lambda data: [data["prompt"] + ".png"])
        stopped = self.jobs.create()
        with life.job_scope(stopped):
            self.state["interrupted"] = True
        item = self.run_job(dict(prompt=["first", "second"], image_total=2, show_preview=False))
        self.assertEqual(self.scope["_process"].call_count, 2)
        self.assertFalse(item.state["interrupted"])

    def test_generate_forever_repeats_until_its_own_job_is_stopped(self):
        def generate(data):
            if generate.count:
                self.state["interrupted"] = True
                return []
            generate.count += 1
            return ["first.png"]

        generate.count = 0
        self.scope["_process"] = Mock(side_effect=generate)
        self.run_job(dict(prompt="forever", image_total=1, generate_forever=True))
        self.assertEqual(self.scope["_process"].call_count, 2)
        self.results.assert_called_once_with(1, "results", ["first.png"])


class SubmissionFailureTests(unittest.TestCase):
    def test_a_submission_that_fails_after_registration_returns_its_queue_slot(self):
        from ruinedfooocus.features.generation import image_preparation as assets

        class Unreachable:
            reachable = False

            def is_file(self):
                if not self.reachable:
                    raise PermissionError(13, "The model drive is not accessible")
                return True

        destination = Unreachable()
        artifact = SimpleNamespace(
            signature="known", request=SimpleNamespace(destination=destination)
        )
        captured = {assets.KEY: assets.ImageAssets((artifact,), offline=True)}
        jobs = life.JobRegistry(capacity=1)
        scope = dict(
            registry=jobs,
            shared=SimpleNamespace(),
            settings=SimpleNamespace(),
            api_jobs={},
            job_scope=life.job_scope,
            start_worker=Mock(),
            capture_image_data=lambda data: dict(data, **captured),
            add_result=jobs.publish,
            describe_error=str,
        )
        submit = function_from(WORKER, "add_task", scope)
        api_job = SimpleNamespace(
            owner=("api", "client"), cancelled=threading.Event(), operation=lambda: None
        )
        with self.assertRaises(PermissionError):
            submit({"task_type": "process", "_api_job": api_job})
        self.assertEqual(scope["api_jobs"], {})
        self.assertTrue(all(item.finished for item in jobs.jobs.values()))
        with self.assertRaises(PermissionError):
            submit({"task_type": "process"})
        destination.reachable = True
        # The only compute slot is free again once the drive is back.
        task_id = submit({"task_type": "process"})
        item, _ = jobs.take(0)
        self.assertEqual(item.task_id, task_id)
        scope["start_worker"].assert_called_once_with()


class WorkerRestartTests(unittest.TestCase):
    def test_a_job_admitted_while_the_worker_is_stopping_gets_a_new_worker(self):
        proceed, admitted, spawned, runs = threading.Event(), [], [], []

        class Registry(life.JobRegistry):
            def fail_ready(registry, message):
                super().fail_ready(message)
                if not admitted:
                    # A request arrives after the stopping worker has failed its
                    # queue and before its thread has ended.
                    admitted.append(registry.create(owner=("ui", "late")))
                    registry.enqueue(admitted[0], {})
                    scope["start_worker"]()
                    spawned.append(scope["worker_thread"])

        jobs = Registry()

        def worker():
            runs.append(threading.current_thread())
            if len(runs) == 1:
                self.assertTrue(proceed.wait(5))
                raise SystemExit
            item, _ = jobs.take(timeout=0)
            jobs.publish(item.task_id, "results", "served")
            jobs.finish(item)

        scope = dict(
            threading=threading,
            task_lock=threading.Lock(),
            worker_thread=None,
            worker=worker,
            registry=jobs,
            api_jobs={},
            settings=SimpleNamespace(default_settings={}),
            _active_job=None,
            _startup_preloaded=True,
        )
        function_from(WORKER, "_worker_entry", scope)
        function_from(WORKER, "start_worker", scope)
        scope["start_worker"]()
        first = scope["worker_thread"]
        proceed.set()
        first.join(5)
        self.assertFalse(first.is_alive())
        self.assertEqual(len(spawned), 1)
        self.assertIsNot(spawned[0], first)
        spawned[0].join(5)
        self.assertFalse(spawned[0].is_alive())
        self.assertTrue(admitted[0].finished)
        self.assertEqual(list(admitted[0].events), [("results", "served")])


class AdmissionShutdownTests(unittest.TestCase):
    def test_close_signals_cancellation_without_holding_its_own_lock(self):
        # An API job's cancellation takes the job registry lock, and the registry
        # cancels a queued future (whose completion takes this pool's lock) while
        # holding its own. Signalling under the pool's lock would close that cycle.
        pool = PreparationAdmission(max_workers=1, max_jobs=2)
        entered, release, free = threading.Event(), threading.Event(), []

        class Signal(threading.Event):
            def set(signal):
                def probe():
                    acquired = pool._lock.acquire(blocking=False)
                    if acquired:
                        pool._lock.release()
                    free.append(acquired)

                observer = threading.Thread(target=probe)
                observer.start()
                observer.join()
                super().set()

        def work():
            entered.set()
            release.wait(5)

        try:
            pool.submit_work(work, cancelled=Signal())
            self.assertTrue(entered.wait(5))
            pool.close(wait=False)
        finally:
            release.set()
            pool.executor.shutdown(wait=True)
        self.assertEqual(free, [True])


class FenceCapacityTests(unittest.TestCase):
    def test_a_cancelled_request_of_a_departed_tab_does_not_hold_a_slot_forever(self):
        fences = life.LatestUIRequests(capacity=2)
        fences.begin(("ui", "left"), "library-view", 1)
        kept = fences.begin(("ui", "open"), "library-view", 1)
        self.assertTrue(fences.cancel(("ui", "left"), "library-view"))
        newest = fences.begin(("ui", "new"), "library-view", 1)
        self.assertTrue(fences.current(kept))
        self.assertTrue(fences.current(newest))
        with self.assertRaises(life.CancelledError):
            fences.peek(("ui", "left"), "library-view")
        # Requests that can still publish are never dropped to make room.
        with self.assertRaisesRegex(RuntimeError, "full"):
            fences.begin(("ui", "another"), "library-view", 1)


if __name__ == "__main__":
    unittest.main()
