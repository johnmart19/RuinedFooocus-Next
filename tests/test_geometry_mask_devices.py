"""Configured device routing and real native CPU retry without GPU allocation."""

from concurrent.futures import CancelledError
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from ruinedfooocus.features.image_processing import image_geometry_tools as geometry
from ruinedfooocus.features.image_processing import image_mask_tools as masks
from ruinedfooocus.features.image_processing import utility_device


class GeometryMaskDeviceTests(unittest.TestCase):
    def geometry_request(self):
        source = np.arange(3 * 4 * 4, dtype=np.uint16).reshape(3, 4, 4) * 1037
        selection = np.zeros((3, 4), dtype=np.uint8)
        selection[1:, 1:3] = 255
        data = geometry.capture_assets(
            dict(
                input_image=source,
                mask_image=selection,
                transform_operation="crop-to-mask",
                output_bit_depth="16-bit",
            ),
            None,
        )[0]
        return data, source[1:, 1:3].astype(np.float32) / 65535

    def mask_request(self, operation="threshold"):
        selection = np.array([[0, 0.25, 0.75], [0.5, 1, 0]], dtype=np.float32)
        data = masks.capture_assets(
            dict(
                input_image=Image.new("RGB", (3, 2)),
                mask_image=selection,
                mask_operation=operation,
                mask_threshold=0.5,
                mask_expand=1,
                output_bit_depth="16-bit",
            ),
            None,
        )[0]
        return data, np.repeat((selection > 0.5)[..., None], 3, axis=-1).astype(np.float32)

    def gpu_policy(self):
        # Constructing a device descriptor does not enumerate or initialize hardware.
        return utility_device.UtilityPolicy(
            torch.device("cuda:2"), torch.device("cpu"), torch.float32
        )

    def test_capture_never_selects_a_device(self):
        with patch.object(utility_device, "resolve_policy", side_effect=AssertionError("capture")):
            self.geometry_request()
            self.mask_request()

    def test_geometry_retries_real_native_cpu_from_frozen_precision_source(self):
        data, expected = self.geometry_request()
        captured = data[geometry.KEY]
        actual = geometry._process_pixels
        attempts = []

        def attempt(request, policy, check_cancelled):
            attempts.append(str(policy.device))
            self.assertIs(request, captured)
            if policy.device.type != "cpu":
                # Fail before any GPU tensor allocation; the CPU retry is real native math.
                data.update(input_image=object(), mask_image=object(), transform_operation="flip")
                raise NotImplementedError("owned synthetic GPU operation failure")
            return actual(request, policy, check_cancelled)

        with (
            patch.object(
                utility_device, "resolve_policy", return_value=self.gpu_policy()
            ) as select,
            patch.object(utility_device, "warn_cpu_fallback") as warn,
            patch.object(geometry, "_process_pixels", side_effect=attempt),
        ):
            result = geometry.process(None, data, None, lambda: None)[0]
        select.assert_called_once_with(cpu_only_reason=None)
        warn.assert_called_once()
        self.assertEqual(attempts, ["cuda:2", "cpu"])
        np.testing.assert_array_equal(result, expected)
        compute = data["image_utility_source"]["compute"]
        self.assertEqual(compute["device"], "cpu")
        self.assertTrue(compute["fallback"])
        self.assertIn("owned synthetic GPU", compute["fallback_reason"])
        self.assertEqual(geometry.output_metadata(data)["image_utility_source"]["compute"], compute)

    def test_mask_retries_real_native_cpu_without_changing_selection_semantics(self):
        data, expected = self.mask_request()
        actual = masks._process_pixels
        attempts = []

        def attempt(request, policy, check_cancelled):
            attempts.append(str(policy.device))
            if policy.device.type != "cpu":
                raise NotImplementedError("owned synthetic native device failure")
            return actual(request, policy, check_cancelled)

        with (
            patch.object(utility_device, "resolve_policy", return_value=self.gpu_policy()),
            patch.object(utility_device, "warn_cpu_fallback"),
            patch.object(masks, "_process_pixels", side_effect=attempt),
        ):
            result = masks.process(None, data, None, lambda: None)[0]
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(attempts, ["cuda:2", "cpu"])
        self.assertTrue(data["image_utility_source"]["compute"]["fallback"])

    def test_native_scipy_grow_reports_host_only_capability(self):
        data, _ = self.mask_request("grow")
        with patch.object(
            utility_device, "resolve_policy", wraps=utility_device.resolve_policy
        ) as select:
            result = masks.process(None, data, None, lambda: None)[0]
        select.assert_called_once_with(cpu_only_reason="Pinned GrowMask uses SciPy and NumPy")
        self.assertEqual(result.shape, (2, 3, 3))
        compute = data["image_utility_source"]["compute"]
        self.assertEqual(compute["device"], "cpu")
        self.assertFalse(compute["fallback"])
        self.assertIn("SciPy", compute["cpu_only_reason"])

    def test_cancellation_after_device_failure_prevents_cpu_retry_and_metadata(self):
        data, _ = self.geometry_request()
        cancelled = False

        def fail(*args):
            nonlocal cancelled
            cancelled = True
            raise NotImplementedError("owned synthetic device failure")

        def check():
            if cancelled:
                raise CancelledError()

        with (
            patch.object(utility_device, "resolve_policy", return_value=self.gpu_policy()),
            patch.object(utility_device, "warn_cpu_fallback") as warn,
            patch.object(geometry, "_process_pixels", side_effect=fail) as process,
        ):
            with self.assertRaises(CancelledError):
                geometry.process(None, data, None, check)
        self.assertEqual(process.call_count, 1)
        warn.assert_not_called()
        self.assertNotIn("image_utility_source", data)

    def test_cancellation_after_real_native_pixels_prevents_publication(self):
        for module, capture in ((geometry, self.geometry_request), (masks, self.mask_request)):
            with self.subTest(kind=module.KINDS[0]):
                data, _ = capture()
                completed = False
                actual = module._process_pixels

                def finish(*args):
                    nonlocal completed
                    value = actual(*args)
                    completed = True
                    return value

                def check():
                    if completed:
                        raise CancelledError()

                with patch.object(module, "_process_pixels", side_effect=finish):
                    with self.assertRaises(CancelledError):
                        module.process(None, data, None, check)
                self.assertTrue(completed)
                self.assertNotIn("image_utility_source", data)


if __name__ == "__main__":
    unittest.main()
