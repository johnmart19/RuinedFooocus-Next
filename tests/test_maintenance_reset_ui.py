"""Real Maintenance components require preview/confirmation and report partial failure."""

import asyncio
import copy
from concurrent.futures import CancelledError, Future
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState

from ruinedfooocus.core.maintenance_reset import ResetPlan, Target
from ruinedfooocus.ui import storage_reset as ui
from ruinedfooocus.ui.gradio_frontend import normalize_hidden_progress
from ruinedfooocus.ui.user_errors import apply as apply_user_errors


class MaintenanceResetUiTests(unittest.TestCase):
    def test_application_configured_blocks_validate_with_immediate_maintenance_callbacks(self):
        gradio_root = gr.Blocks(analytics_enabled=False).queue(api_open=True)
        self.addCleanup(gradio_root.close)
        with gr.Blocks(analytics_enabled=False) as app:
            ui.create_storage_reset(object(), object())
        self.addCleanup(app.close)
        # These are the actual launch_app configuration owners, before launch
        # performs this native validation. No HTTP server needs to start.
        normalize_hidden_progress(app)
        apply_user_errors(app, gradio_root)
        app.validate_queue_settings()
        self.assertTrue(all(not function.queue for function in app.fns.values()))

    def test_construction_is_lazy_actions_are_private_and_cancellation_is_immediate(self):
        with (
            patch.object(ui, "preview_reset", side_effect=AssertionError("No scan on navigation")),
            patch.object(ui, "execute_reset", side_effect=AssertionError("No reset on navigation")),
            gr.Blocks() as app,
        ):
            ui.create_storage_reset(object(), object())
        functions = {function.fn.__name__: function for function in app.fns.values() if function.fn}
        self.assertTrue(
            {"preview_cache", "preview_outputs", "preview_factory", "apply_reset", "check_folders"}
            <= functions.keys()
        )
        for name in (
            "preview_cache",
            "preview_outputs",
            "preview_factory",
            "apply_reset",
            "check_folders",
        ):
            self.assertEqual(functions[name].api_visibility, "private")
            self.assertEqual(functions[name].concurrency_id, "cache_maintenance")
            self.assertFalse(functions[name].queue)
        control = ui.ResetSession()
        self.assertIsNot(copy.deepcopy(control).cancelled, control.cancelled)
        functions["cancel_reset"].fn(control)
        self.assertTrue(control.cancelled.is_set())
        self.assertFalse(functions["cancel_reset"].queue)
        plan = ResetPlan("cache-outputs", None, (), (), "private")
        self.assertFalse(functions["confirm"].fn(plan, False)["interactive"])
        self.assertTrue(functions["confirm"].fn(plan, True)["interactive"])

    def test_preview_rows_total_each_folder_instead_of_listing_every_file(self):
        import os

        cache = os.path.join("cache", "library-thumbnails")
        targets = (
            Target(os.path.join(cache, "a.webp"), "library-thumbnails", None, 1024),
            Target(os.path.join(cache, "b.webp"), "library-thumbnails", None, 2048),
            Target(os.path.join("cache", "images.db"), "library-index", None, 99, 12),
            Target(os.path.join("outputs", "day", "one.png"), "output-file", None, 5 << 20),
            Target(os.path.join("outputs", "a"), "output-directory", None),
            Target(os.path.join("outputs", "a", "b"), "output-directory", None),
        )
        plan = ResetPlan("cache-outputs", None, targets, (), "private")
        self.assertEqual(
            ui.target_rows(plan, "en"),
            [
                [cache, "Library thumbnails", 2, "3.0 KiB", 0],
                [os.path.join("cache", "images.db"), "Library index", 0, "0 B", 12],
                [
                    os.path.join("outputs", "day", "one.png"),
                    "Original output files",
                    1,
                    "5.0 MiB",
                    0,
                ],
                [os.path.join("outputs", "a"), "Subfolders (2)", 0, "0 B", 0],
            ],
        )

    def test_failed_or_cancelled_partial_results_never_say_completed(self):
        base = {
            "complete": False,
            "cancelled": False,
            "removed": 1,
            "retained": [],
            "errors": ["private error"],
        }
        self.assertTrue(ui.result_text(base).startswith("Failed"))
        self.assertTrue(ui.result_text({**base, "cancelled": True}).startswith("Cancelled"))
        self.assertNotIn("Completed", ui.result_text(base))

    def test_cancel_dismisses_each_preview_and_checked_confirmation_through_native_events(self):
        with gr.Blocks(analytics_enabled=False) as app:
            ui.create_storage_reset(object(), object())
        self.addCleanup(app.close)
        functions = {value.fn.__name__: key for key, value in app.fns.items() if value.fn}
        apply = app.fns[functions["apply_reset"]]
        plan_id = apply.inputs[0]._id
        target = Target("private-preview-only.png", "output-file", None, 123)

        async def check(action, callback):
            state = SessionState(app)
            plan = ResetPlan(action, "current", (target,), ("private-retained",), "private")
            with patch.object(ui, "preview_reset", return_value=plan):
                preview = await app.process_api(functions[callback], [None], state=state)
            self.assertIsInstance(state[plan_id], ResetPlan)
            self.assertTrue(preview["data"][1]["data"])
            confirmed = await app.process_api(functions["confirm"], [None, True], state=state)
            self.assertTrue(confirmed["data"][0]["interactive"])
            cancelled = await app.process_api(functions["cancel_reset"], [None], state=state)
            self.assertIsNone(state[plan_id])
            self.assertEqual(cancelled["data"][1]["data"], [])
            self.assertEqual(cancelled["data"][2], "")
            self.assertFalse(cancelled["data"][3]["value"])
            self.assertFalse(cancelled["data"][3]["visible"])
            self.assertFalse(cancelled["data"][4]["interactive"])
            self.assertEqual(cancelled["data"][5], "Cancelled")
            # The preview table, retention and confirmation hide again.
            self.assertFalse(cancelled["data"][6]["visible"])
            self.assertTrue(preview["data"][6]["visible"])
            # A stale client-side checked value cannot resurrect the state plan.
            with patch.object(ui, "execute_reset", wraps=ui.execute_reset) as execute:
                rejected = await app.process_api(
                    functions["apply_reset"], [None, True, None], state=state
                )
            self.assertIsNone(execute.call_args.args[0])
            self.assertIn("Preview the maintenance action first", rejected["data"][3])
            self.assertFalse(rejected["data"][4]["visible"])

        for action, callback in (
            ("cache", "preview_cache"),
            ("cache-outputs", "preview_outputs"),
            ("factory-profile", "preview_factory"),
        ):
            with self.subTest(action=action):
                asyncio.run(check(action, callback))

    def test_cancelled_late_previews_cannot_restore_plan_through_native_events(self):
        with gr.Blocks(analytics_enabled=False) as app:
            ui.create_storage_reset(object(), object())
        self.addCleanup(app.close)
        functions = {value.fn.__name__: key for key, value in app.fns.items() if value.fn}
        plan_id = app.fns[functions["apply_reset"]].inputs[0]._id
        pending = []

        def delayed(work):
            future = Future()
            pending.append(future)
            return future

        async def check(action, callback):
            state = SessionState(app)
            with patch.object(ui.preparation_admission, "submit_work", side_effect=delayed):
                preview = asyncio.create_task(
                    app.process_api(functions[callback], [None], state=state)
                )
                await asyncio.sleep(0)
                await app.process_api(functions["cancel_reset"], [None], state=state)
                pending.pop().set_result(ResetPlan(action, None, (), (), "late-preview"))
                late = await preview
            self.assertIsNone(state[plan_id])
            self.assertTrue(all(value == gr.skip() for value in late["data"][1:]))

        for action, callback in (
            ("cache", "preview_cache"),
            ("cache-outputs", "preview_outputs"),
            ("factory-profile", "preview_factory"),
        ):
            with self.subTest(action=action):
                asyncio.run(check(action, callback))

    def test_cancel_reaches_work_waiting_in_preparation_and_older_previews_cannot_publish(self):
        with gr.Blocks() as app:
            ui.create_storage_reset(object(), object())
        functions = {function.fn.__name__: function for function in app.fns.values() if function.fn}
        futures = []

        def queued(work):
            future = Future()
            futures.append((future, work))
            return future

        async def cancelled_action():
            control = ui.ResetSession()
            plan = ResetPlan("cache-outputs", None, (), (), "private")
            task = asyncio.create_task(functions["apply_reset"].fn(plan, True, control))
            await asyncio.sleep(0)
            functions["cancel_reset"].fn(control)
            future, work = futures.pop()
            with self.assertRaises(CancelledError):
                work()
            future.set_exception(CancelledError("Maintenance was cancelled"))
            result = await task
            self.assertIn("Cancelled", result[3])
            self.assertNotIn("Completed", result[3])

        def check_cancelled(*args, cancelled, **kwargs):
            if cancelled.is_set():
                raise CancelledError("Maintenance was cancelled")
            raise AssertionError("Cancellation did not reach pending work")

        async def replaced_preview():
            control = ui.ResetSession()
            old = asyncio.create_task(functions["preview_outputs"].fn(control))
            await asyncio.sleep(0)
            new = asyncio.create_task(functions["preview_cache"].fn(control))
            await asyncio.sleep(0)
            first, second = futures
            first[0].set_result(ResetPlan("cache-outputs", None, (), (), "old"))
            second[0].set_result(ResetPlan("cache", None, (), (), "new"))
            self.assertEqual(await old, (gr.skip(),) * 7)
            self.assertEqual((await new)[0].action, "cache")

        with (
            patch.object(ui.preparation_admission, "submit_work", side_effect=queued),
            patch.object(ui, "execute_reset", side_effect=check_cancelled),
        ):
            asyncio.run(cancelled_action())
            asyncio.run(replaced_preview())


if __name__ == "__main__":
    unittest.main()
