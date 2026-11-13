"""Utility device selection stays lazy, explicit, precise and recoverable."""

from concurrent.futures import CancelledError
import os
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from ruinedfooocus.features.image_processing import utility_device as policy


class UtilityDevicePolicyTests(unittest.TestCase):
    def setUp(self):
        self.native = NS(
            get_torch_device=Mock(return_value=torch.device("cuda:0")),
            text_encoder_device=Mock(return_value=torch.device("cuda:0")),
            text_encoder_offload_device=Mock(return_value=torch.device("cpu")),
            intermediate_device=Mock(return_value=torch.device("cpu")),
            intermediate_dtype=Mock(side_effect=AssertionError("Pixel work stays float32")),
        )
        self.launch = NS(cpu=False)
        self.enterContext(patch.object(sys, "argv", ["utility-test"]))
        self.enterContext(patch.dict(os.environ, {"TORCH_PLATFORM": ""}))
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.arguments": NS(args=self.launch),
                    "comfy.cli_args": NS(args=NS(cpu=False)),
                    "comfy": NS(model_management=self.native),
                },
            )
        )
        for name in ("is_available", "current_device", "device_count", "init"):
            self.enterContext(
                patch.object(torch.cuda, name, side_effect=AssertionError("No GPU probe"))
            )
        self.warning = self.enterContext(patch.object(policy, "warn_cpu_fallback"))

    def run_job(self, operation=lambda selected: str(selected.device), **kwargs):
        return policy.run_utility("fixture", operation, check_cancelled=lambda: None, **kwargs)

    def test_import_is_passive_without_torch_comfy_or_application_arguments(self):
        script = """
import sys
from ruinedfooocus.features.image_processing import utility_device
assert 'torch' not in sys.modules
assert 'comfy.model_management' not in sys.modules
assert 'ruinedfooocus.app.arguments' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)

    def test_explicit_cpu_flag_environment_and_applied_profile_never_resolve_gpu(self):
        cases = [
            dict(argv=["fixture", "--cpu"]),
            dict(environment="cpu"),
            dict(args=NS(cpu=True)),
            dict(args=NS(cpu=True, _compute_status={"cli": ["--cpu"], "backend": {}})),
            dict(args=NS(cpu=True, _compute_status={"backend": {"cpu_applied": True}})),
        ]
        for case in cases:
            with (
                self.subTest(case=case),
                patch.object(sys, "argv", case.get("argv", ["fixture"])),
                patch.dict(os.environ, {"TORCH_PLATFORM": case.get("environment", "")}),
                patch.dict(
                    sys.modules,
                    {"ruinedfooocus.app.arguments": NS(args=case.get("args", NS(cpu=False)))},
                ),
            ):
                result = self.run_job()
                self.assertEqual(result.value, "cpu")
                self.assertEqual(result.metadata()["requested"], "cpu")
                self.assertFalse(result.metadata()["fallback"])
        self.native.get_torch_device.assert_not_called()
        self.warning.assert_not_called()

    def test_auto_uses_native_selected_device_and_float32_pixel_precision(self):
        # CUDA and ROCm share Torch's cuda device type; it is not proof of one runtime.
        for device in ("cuda:0", "cuda:1", "xpu:0", "mps", "privateuseone:0"):
            with self.subTest(device=device):
                self.native.get_torch_device.return_value = torch.device(device)
                result = self.run_job()
                self.assertEqual(result.value, device)
                self.assertEqual(result.policy.intermediate_dtype, torch.float32)
                self.assertEqual(result.metadata()["backend"], "native-torch")
                self.assertEqual(result.metadata()["intermediate_device"], "cpu")
                self.assertFalse(result.metadata()["fallback"])
        self.native.intermediate_dtype.assert_not_called()
        self.warning.assert_not_called()

    def test_model_placement_keeps_native_gpu_and_generic_pixel_callers_stay_independent(self):
        result = self.run_job(native_model_placement=True)
        self.assertEqual(result.value, "cuda:0")
        self.native.text_encoder_device.assert_called_once()
        self.native.text_encoder_device.reset_mock()
        self.run_job()
        self.native.text_encoder_device.assert_not_called()
        self.warning.assert_not_called()

    def test_native_model_cpu_precision_and_capacity_are_intentional_not_fallback(self):
        self.native.text_encoder_device.return_value = torch.device("cpu")
        for state, forced, reason in (
            ("LOW_VRAM", False, "LOW_VRAM memory policy"),
            ("NO_VRAM", False, "NO_VRAM memory policy"),
            ("NORMAL_VRAM", True, "forced FP32"),
            ("NORMAL_VRAM", False, "precision/capacity"),
        ):
            with self.subTest(state=state, force_fp32=forced):
                self.native.vram_state = NS(name=state)
                self.native.FORCE_FP32 = forced
                result = self.run_job(native_model_placement=True)
                self.assertEqual(result.value, "cpu")
                self.assertEqual(result.metadata()["requested"], "auto")
                self.assertFalse(result.metadata()["fallback"])
                self.assertIn(reason, result.metadata()["cpu_only_reason"])
        self.warning.assert_not_called()

    def test_explicit_model_cpu_never_discovers_native_model_devices(self):
        with patch.object(sys, "argv", ["fixture", "--cpu"]):
            result = self.run_job(native_model_placement=True)
        self.assertEqual(result.value, "cpu")
        self.assertEqual(result.metadata()["requested"], "cpu")
        self.native.get_torch_device.assert_not_called()
        self.native.text_encoder_device.assert_not_called()
        self.warning.assert_not_called()

    def test_native_model_discovery_and_accelerator_failure_retain_actual_reason(self):
        callback = Mock(side_effect=[RuntimeError("HIP error: device lost"), "cpu result"])
        result = self.run_job(callback, native_model_placement=True)
        self.assertEqual(
            [str(call.args[0].device) for call in callback.call_args_list], ["cuda:0", "cpu"]
        )
        self.assertEqual(result.value, "cpu result")
        self.assertIn("HIP error: device lost", result.metadata()["fallback_reason"])
        self.warning.assert_called_once()

    def test_gpu_only_model_offload_rejects_cpu_retry_before_cpu_constructor_or_warning(self):
        self.native.text_encoder_offload_device.return_value = torch.device("cuda:0")
        callback = Mock(side_effect=RuntimeError("CUDA unsupported device"))
        with self.assertRaisesRegex(RuntimeError, "Disable GPU-only and use Auto.*--cpu"):
            self.run_job(callback, native_model_placement=True)
        callback.assert_called_once()
        self.warning.assert_not_called()

    def test_applied_cli_gpu_choice_precedes_lower_priority_cpu_environment(self):
        self.launch._compute_status = {"cli": ["--directml"], "backend": {"cpu_applied": False}}
        with patch.dict(os.environ, {"TORCH_PLATFORM": "cpu"}):
            result = self.run_job()
        self.assertEqual(result.value, "cuda:0")
        self.assertEqual(result.metadata()["requested"], "auto")

    def test_automatic_cpu_startup_keeps_auto_origin_and_reports_missing_accelerator(self):
        self.launch.cpu = True
        self.launch._compute_status = {
            "backend": {"cpu_applied": False, "saved_image_backend": "auto"}
        }
        self.native.get_torch_device.return_value = torch.device("cpu")
        result = self.run_job()
        self.assertEqual(result.metadata()["requested"], "auto")
        self.assertTrue(result.metadata()["fallback"])
        self.assertIn("native PyTorch runtime selected CPU", result.metadata()["fallback_reason"])
        self.warning.assert_called_once()

    def test_native_resolution_failure_keeps_actual_reason_and_runs_cpu(self):
        self.native.get_torch_device.side_effect = RuntimeError("HIP error: unavailable device")
        result = self.run_job()
        self.assertEqual(result.value, "cpu")
        self.assertIn("HIP error: unavailable device", result.metadata()["fallback_reason"])
        self.warning.assert_called_once()

    def test_host_only_capability_neither_probes_nor_claims_device_failure(self):
        result = self.run_job(cpu_only_reason="Pinned operator uses Pillow")
        self.assertEqual(result.value, "cpu")
        self.assertEqual(result.metadata()["cpu_only_reason"], "Pinned operator uses Pillow")
        self.assertFalse(result.metadata()["fallback"])
        self.native.get_torch_device.assert_not_called()
        self.warning.assert_not_called()

    def test_automatic_cpu_policy_reports_terminal_reason_but_explicit_cpu_stays_silent(self):
        with patch.object(policy.logging, "warning") as message:
            result = self.run_job(cpu_only_reason="Pinned operator uses Pillow")
            message.assert_called_once_with(
                "%s is running on CPU: %s.", "fixture", "Pinned operator uses Pillow"
            )
            self.assertFalse(result.metadata()["fallback"])
            message.reset_mock()
            with patch.object(sys, "argv", ["fixture", "--cpu"]):
                explicit = self.run_job(cpu_only_reason="Pinned operator uses Pillow")
            message.assert_not_called()
            self.assertEqual(explicit.metadata()["requested"], "cpu")
        self.warning.assert_not_called()

    def test_unsupported_gpu_callback_restarts_once_with_fresh_cpu_policy(self):
        calls = []

        def operation(selected):
            calls.append(str(selected.device))
            if selected.device.type != "cpu":
                raise NotImplementedError("fixture operator lacks selected backend")
            return b"fresh cpu pixels"

        result = self.run_job(operation)
        self.assertEqual(calls, ["cuda:0", "cpu"])
        self.assertEqual(result.value, b"fresh cpu pixels")
        self.assertEqual(result.metadata()["backend"], "cpu")
        self.assertIn("NotImplementedError", result.metadata()["fallback_reason"])
        self.warning.assert_called_once()

    def test_only_device_runtime_failures_recover(self):
        for error in (
            RuntimeError("CUDA error: invalid device function"),
            OSError("DirectML device lost"),
        ):
            with self.subTest(error=error):
                callback = Mock(side_effect=[error, "recovered"])
                self.assertEqual(self.run_job(callback).value, "recovered")
                self.assertEqual(callback.call_count, 2)

    def test_source_validation_disk_errors_and_cpu_failure_are_not_retried(self):
        errors = (
            ValueError("source changed"),
            RuntimeError("invalid source dimensions"),
            OSError("image decode failed"),
            FileNotFoundError(2, "missing image", "gpu-mask.png"),
            PermissionError(13, "denied", "cuda-image.png"),
        )
        for error in errors:
            with self.subTest(error=error):
                callback = Mock(side_effect=error)
                with self.assertRaises(type(error)):
                    self.run_job(callback)
                callback.assert_called_once()
        self.warning.assert_not_called()
        with patch.object(sys, "argv", ["fixture", "--cpu"]):
            callback = Mock(side_effect=RuntimeError("out of memory"))
            with self.assertRaises(RuntimeError):
                self.run_job(callback)
            callback.assert_called_once()

    def test_cancel_before_selection_or_gpu_retry_never_runs_another_attempt(self):
        callback = Mock()
        with self.assertRaises(CancelledError):
            policy.run_utility(
                "fixture", callback, check_cancelled=Mock(side_effect=CancelledError)
            )
        self.native.get_torch_device.assert_not_called()
        callback.assert_not_called()
        callback = Mock(side_effect=NotImplementedError("unavailable GPU operator"))
        cancel = Mock(side_effect=[None, None, CancelledError()])
        with self.assertRaises(CancelledError):
            policy.run_utility("fixture", callback, check_cancelled=cancel)
        callback.assert_called_once()
        self.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
