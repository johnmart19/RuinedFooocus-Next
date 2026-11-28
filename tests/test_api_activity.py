"""Actual API Job state projection with inert work and strict private-data exclusion."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr

from ruinedfooocus.app import api_activity, api_runtime
from ruinedfooocus.ui.api_activity import activity_view, create_api_activity_controls


class ApiActivityTests(unittest.TestCase):
    def test_running_cancel_is_timed_until_the_actual_callback_releases_its_job(self):
        now = [0.0]
        log = api_activity.ApiActivity(clock=lambda: now[0])
        observations = []

        def operation(job, task_id):
            now[0] = 2.0
            job.cancelled.set()
            observations.append(log.snapshot()["counts"])
            now[0] = 5.0  # Inert callback cleanup finishes after cancellation acknowledgement.

        operation.resource_group = "chat"
        with patch.object(api_activity, "activity", log):
            job = api_runtime.Job(operation)
            now[0] = 1.0
            job.run(1)
        self.assertEqual(observations, [{"running": 1}])
        row = log.snapshot()["rows"][0]
        self.assertEqual(row["state"], "cancelled")
        self.assertEqual((row["wait_ms"], row["run_ms"], row["total_ms"]), (1000.0, 4000.0, 5000.0))
        self.assertTrue(job.released)

    def test_bounded_retention_known_states_and_observed_durations(self):
        now = [10.0]
        log = api_activity.ApiActivity(capacity=2, retention_seconds=5, clock=lambda: now[0])
        token = log.begin("chat")
        now[0] = 12.0
        log.started(token)
        now[0] = 13.0
        log.finished(token, "completed")
        log.finished(token, "cancelled")
        snapshot = log.snapshot()
        self.assertEqual(
            snapshot["rows"],
            [
                {
                    "kind": "chat",
                    "state": "completed",
                    "wait_ms": 2000.0,
                    "run_ms": 1000.0,
                    "total_ms": 3000.0,
                }
            ],
        )
        self.assertEqual(snapshot["counts"], {"completed": 1})
        log.begin("image")
        log.begin("untrusted private model path")
        self.assertEqual(log.snapshot()["retained"], 2)
        self.assertEqual(log.snapshot()["rows"][0]["kind"], "other")
        now[0] = 19.0
        self.assertEqual(log.snapshot(), {"retained": 0, "counts": {}, "rows": []})

    def test_real_job_lifecycle_excludes_bodies_errors_paths_keys_and_ui_owned_work(self):
        log = api_activity.ApiActivity()
        marker = "private-prompt-document-filename-api-key-native-path"

        def complete(job, task_id):
            return {"content": marker, "path": marker}

        complete.resource_group = "chat"

        def fail(job, task_id):
            raise ValueError(marker)

        fail.resource_group = marker
        with patch.object(api_activity, "activity", log):
            done = api_runtime.Job(complete)
            self.assertEqual(log.snapshot()["counts"], {"waiting": 1})
            done.run(1)
            done.cancelled.set()  # Post-response cleanup must not relabel success.
            broken = api_runtime.Job(fail)
            broken.run(2)
            cancelled = api_runtime.Job(complete)
            cancelled.cancelled.set()
            with patch(
                "ruinedfooocus.app.resource_lifecycle.current_owner",
                return_value=("ui", "owner", "tab"),
            ):
                ui = api_runtime.Job(complete)
                ui.run(3)
        snapshot = log.snapshot()
        self.assertEqual(snapshot["counts"], {"completed": 1, "failed": 1, "cancelled": 1})
        self.assertNotIn(marker, str(snapshot))
        self.assertNotIn("owner", str(snapshot))
        for row in snapshot["rows"]:
            self.assertEqual(set(row), {"kind", "state", "wait_ms", "run_ms", "total_ms"})
        self.assertTrue(all(job.released for job in (done, broken, cancelled, ui)))

    def test_real_running_job_and_capacity_never_capture_the_operation(self):
        log = api_activity.ApiActivity()
        observations = []

        def operation(job, task_id):
            observations.append(log.snapshot()["counts"])
            return None

        operation.resource_group = "image"
        with patch.object(api_activity, "activity", log):
            api_runtime.Job(operation).run(1)
        self.assertEqual(observations, [{"running": 1}])
        for _ in range(200):
            token = log.begin("chat")
            log.finished(token, "completed")
        self.assertEqual(log.snapshot()["retained"], 128)
        self.assertEqual(len(log.snapshot()["rows"]), 20)
        self.assertTrue(
            all(
                not callable(value)
                for entry in log.entries.values()
                for value in vars(entry).values()
            )
        )

    def test_owner_view_projects_only_summary_rows_and_disabled_api_reads_nothing(self):
        log = api_activity.ApiActivity()
        log.finished(log.begin("chat"), "cancelled")
        shared = SimpleNamespace(
            server_app=SimpleNamespace(state=SimpleNamespace(rf_openai_api_requires_key=True))
        )
        with (
            patch("ruinedfooocus.app.state", shared, create=True),
            patch.object(api_activity, "activity", log),
        ):
            summary, rows = activity_view("en")
        self.assertIn("cancelled 1", summary)
        self.assertEqual(rows[0][:2], ["Chat", "Cancelled"])
        shared.server_app.state.rf_openai_api_requires_key = None
        with (
            patch("ruinedfooocus.app.state", shared, create=True),
            patch.object(log, "snapshot", side_effect=AssertionError("disabled API")),
            patch.object(api_activity, "activity", log),
        ):
            summary, rows = activity_view("en")
        self.assertEqual(rows, [])
        self.assertIn("unavailable", summary)

    def test_controls_are_private_nonqueued_reads_without_a_background_timer(self):
        with gr.Blocks() as app:
            with gr.Tab("Connections") as tab:
                create_api_activity_controls(tab)
        configuration = app.get_config_file()
        self.assertEqual(len(configuration["dependencies"]), 2)
        self.assertTrue(
            all(
                row["api_visibility"] == "private" and not row["queue"]
                for row in configuration["dependencies"]
            )
        )
        self.assertFalse(any(row["type"] == "timer" for row in configuration["components"]))


if __name__ == "__main__":
    unittest.main()
