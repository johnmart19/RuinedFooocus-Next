"""Color caller dispatch and injected policy; no actual accelerator execution."""

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from ruinedfooocus.features.image_processing import color_native as native
from ruinedfooocus.features.image_processing import image_color_tools as colors
from ruinedfooocus.features.image_processing import utility_device


class ImageColorDevicePolicyTests(unittest.TestCase):
    def capture(self, operation, **values):
        source = Image.new("RGBA", (32, 24), (25, 50, 75, 64))
        values.update(input_image=source, color_operation=operation)
        if operation in ("blend", "transfer"):
            values["color_reference"] = Image.new("RGB", source.size, (75, 100, 125))
        if operation == "sdr-to-hdr":
            values.update(
                output_format="exr", output_bit_depth="32-bit float", output_color_space="HDR"
            )
        return colors.capture_assets(values, None)[0]

    def test_capture_and_worker_transport_do_not_select_a_device(self):
        with patch.object(
            utility_device,
            "run_utility",
            side_effect=AssertionError("device selection during capture"),
        ):
            data = self.capture("morphology")
            worker = colors.worker_data(data)
        self.assertIs(worker[colors.KEY], data[colors.KEY])

    def test_actual_native_color_jobs_use_shared_policy_and_report_cpu_execution(self):
        original = utility_device.run_utility
        for operation in ("blend", "transfer", "morphology", "sdr-to-hdr"):
            with self.subTest(operation=operation):
                data = self.capture(operation)
                calls = []

                def run(feature, callback, *, check_cancelled, cpu_only_reason=None):
                    calls.append((feature, cpu_only_reason))
                    return original(
                        feature,
                        callback,
                        check_cancelled=check_cancelled,
                        cpu_only_reason=cpu_only_reason,
                    )

                with patch.object(utility_device, "run_utility", side_effect=run):
                    result = colors.process(None, data, None, lambda: None)[0]
                self.assertEqual(calls, [(colors.NODES[operation], None)])
                self.assertEqual(np.asarray(result).shape, (24, 32, 4))
                compute = data["image_utility_source"]["compute"]
                self.assertEqual(compute["device"], "cpu")
                self.assertEqual(compute["backend"], "cpu")
                self.assertFalse(compute["fallback"])
                self.assertEqual(compute["cpu_only_reason"], "")

    def test_pillow_native_operations_have_truthful_host_capability_metadata(self):
        for operation in ("text-overlay", "quantize"):
            with self.subTest(operation=operation):
                data = self.capture(operation, color_text="A", color_text_font_size=25)
                colors.process(None, data, None, lambda: None)
                compute = data["image_utility_source"]["compute"]
                self.assertEqual(compute["device"], "cpu")
                self.assertTrue(compute["cpu_only_reason"])
                self.assertFalse(compute["fallback"])
                self.assertEqual(compute["fallback_reason"], "")

    def test_bridge_injects_configured_compute_and_intermediate_policy_without_probe(self):
        # Constructing a device descriptor does not initialize/probe an accelerator.
        policy = utility_device.UtilityPolicy(
            torch.device("cuda:2"), torch.device("cuda:1"), torch.float32, requested="auto"
        )
        signatures = native.source_signatures()
        for options, name in (
            (dict(transfer=True), "ColorTransfer"),
            (dict(finishing="morphology"), "Morphology"),
            (dict(finishing="sdr-to-hdr"), "ImageSDRToHDR"),
        ):
            with self.subTest(name=name):
                ops = native.operations(signatures, lambda: None, policy=policy, **options)
                namespace = getattr(ops, name).execute.__func__.__globals__
                management = namespace["comfy"].model_management
                self.assertEqual(management.get_torch_device(), policy.device)
                self.assertEqual(management.intermediate_device(), policy.intermediate_device)
                self.assertEqual(management.intermediate_dtype(), torch.float32)

    def test_float32_intermediate_keeps_source_alpha_precision(self):
        source = np.full((3, 5, 4), 0.25, dtype=np.float32)
        source[..., 3] = 0.1234567
        data = colors.capture_assets(
            dict(
                input_image=source,
                color_operation="sdr-to-hdr",
                output_format="exr",
                output_bit_depth="32-bit float",
                output_color_space="HDR",
            ),
            None,
        )[0]
        policy = utility_device.UtilityPolicy(
            torch.device("cpu"), torch.device("cpu"), torch.float32, requested="cpu"
        )

        class Execution:
            def __init__(self, value):
                self.value = value

            def metadata(self):
                return {"device": "cpu", "intermediate_dtype": "torch.float32"}

        def run(feature, callback, **kwargs):
            return Execution(callback(policy))

        with patch.object(utility_device, "run_utility", side_effect=run):
            result = colors.process(None, data, None, lambda: None)[0]
        np.testing.assert_array_equal(result[..., 3], source[..., 3])
        self.assertEqual(
            data["image_utility_source"]["compute"]["intermediate_dtype"], "torch.float32"
        )

    def test_unsupported_accelerator_attempt_recovers_through_real_cpu_native_operation(self):
        data = self.capture("blend")
        source_identity = data[colors.KEY].source.source_sha256
        selected = utility_device.UtilityPolicy(
            torch.device("cuda:2"), torch.device("cpu"), torch.float32
        )
        original = native.operations
        attempts = []

        def operations(signatures, cancelled, *, policy, **options):
            attempts.append(str(policy.device))
            if policy.device.type != "cpu":
                # Simulate an unsupported native kernel before any allocation or
                # device initialization. Recovery still executes the real CPU body.
                raise NotImplementedError("selected accelerator lacks the native image kernel")
            return original(signatures, cancelled, policy=policy, **options)

        with (
            patch.object(utility_device, "resolve_policy", return_value=selected),
            patch.object(native, "operations", side_effect=operations),
            patch.object(utility_device, "warn_cpu_fallback") as warning,
        ):
            result = colors.process(None, data, None, lambda: None)[0]
        self.assertEqual(attempts, ["cuda:2", "cpu"])
        self.assertEqual(np.asarray(result).shape, (24, 32, 4))
        compute = data["image_utility_source"]["compute"]
        self.assertEqual(compute["device"], "cpu")
        self.assertTrue(compute["fallback"])
        self.assertIn("lacks the native image kernel", compute["fallback_reason"])
        warning.assert_called_once()
        self.assertEqual(data[colors.KEY].source.source_sha256, source_identity)

    def test_cpu_native_failure_is_reported_without_a_second_attempt_or_gpu_claim(self):
        data = self.capture("morphology")
        selected = utility_device.UtilityPolicy(
            torch.device("cpu"), torch.device("cpu"), torch.float32, requested="cpu"
        )
        with (
            patch.object(utility_device, "resolve_policy", return_value=selected),
            patch.object(
                native, "operations", side_effect=NotImplementedError("CPU kernel unavailable")
            ) as execute,
            patch.object(utility_device, "warn_cpu_fallback") as warning,
        ):
            with self.assertRaisesRegex(NotImplementedError, "CPU kernel unavailable"):
                colors.process(None, data, None, lambda: None)
        execute.assert_called_once()
        warning.assert_not_called()
        self.assertNotIn("image_utility_source", data)


if __name__ == "__main__":
    unittest.main()
