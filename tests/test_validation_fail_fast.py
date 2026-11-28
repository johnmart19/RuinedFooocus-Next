"""A failed adapter or cleanup cannot dispatch another native model load."""

from contextlib import nullcontext
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import validate_comfy_workflows as validator


class ValidationFailFastTests(unittest.TestCase):
    def execute(self, app_error=None, cleanup_error=None, native_error=None):
        app = Mock(return_value={"settings": {"seed": 1}}, side_effect=app_error)
        native = Mock(return_value={"settings": {"seed": 1}}, side_effect=native_error)
        memory = SimpleNamespace()
        torch = SimpleNamespace(inference_mode=nullcontext)
        with (
            patch.dict(
                sys.modules,
                {
                    "torch": torch,
                    "comfy": SimpleNamespace(model_management=memory),
                    "comfy.model_management": memory,
                },
            ),
            patch.object(validator, "initialize"),
            patch.object(validator, "revision", return_value="fixed-source"),
            patch.object(validator, "provenance", return_value={}),
            patch.object(validator, "load_entrypoint", side_effect=[app, native]) as load,
            patch.object(
                validator,
                "cleanup_runtime",
                return_value={"completed": True},
                side_effect=cleanup_error,
            ) as cleanup,
        ):
            report = validator.run({side: {"entrypoint": side} for side in ("app", "native")})
        return report, app, native, load, cleanup

    def test_app_device_error_stops_before_native_load(self):
        report, app, native, load, cleanup = self.execute(app_error=RuntimeError("HIP device lost"))
        app.assert_called_once()
        native.assert_not_called()
        self.assertEqual(load.call_count, 1)
        self.assertEqual(cleanup.call_count, 1)
        self.assertEqual(report["aborted_after"], "app")
        self.assertIn("HIP device lost", report["runs"]["app"]["error"])
        self.assertFalse(report["comparison"]["passed"])

    def test_cleanup_error_stops_before_native_load_and_is_not_retried(self):
        report, _, native, load, cleanup = self.execute(
            cleanup_error=RuntimeError("cleanup failed")
        )
        native.assert_not_called()
        self.assertEqual(load.call_count, 1)
        self.assertEqual(cleanup.call_count, 1)
        self.assertEqual(report["cleanup_errors"][0]["side"], "app")
        self.assertFalse(report["comparison"]["passed"])

    def test_ordinary_adapter_error_also_stops(self):
        report, _, native, load, _ = self.execute(app_error=ValueError("invalid source"))
        native.assert_not_called()
        self.assertEqual(load.call_count, 1)
        self.assertEqual(report["aborted_after"], "app")

    def test_successful_pair_keeps_comparison_and_final_cleanup(self):
        report, app, native, load, cleanup = self.execute()
        app.assert_called_once()
        native.assert_called_once()
        self.assertEqual(load.call_count, 2)
        self.assertEqual(cleanup.call_count, 3)
        self.assertNotIn("aborted_after", report)
        self.assertTrue(report["comparison"]["passed"])

    def test_native_error_fails_without_retry(self):
        report, _, native, load, cleanup = self.execute(native_error=RuntimeError("native failed"))
        native.assert_called_once()
        self.assertEqual(load.call_count, 2)
        self.assertEqual(cleanup.call_count, 2)
        self.assertEqual(report["aborted_after"], "native")
        self.assertFalse(report["comparison"]["passed"])

    def test_final_cleanup_failure_cannot_leave_comparison_passed(self):
        report, _, _, _, cleanup = self.execute(
            cleanup_error=[{}, {}, RuntimeError("shutdown failed")]
        )
        self.assertEqual(cleanup.call_count, 3)
        self.assertEqual(report["cleanup_errors"][0]["side"], "shutdown")
        self.assertFalse(report["comparison"]["passed"])
