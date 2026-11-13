"""Package policy remains import-free and separates legacy, modern and AMD."""

import importlib.metadata
import unittest
from unittest.mock import patch

from ruinedfooocus.packaging import cuda_selection
from ruinedfooocus.packaging.runtime_bundles import BUNDLES, bundle_for, python_bundle_supported


class RuntimeBundleTests(unittest.TestCase):
    def test_wheel_admission_checks_only_required_image_companions(self):
        with patch.object(cuda_selection, "wheel_available", return_value=True) as available:
            self.assertTrue(cuda_selection.cuda_wheels_available("cu132"))
        self.assertEqual(
            [call.args[1] for call in available.call_args_list], ["torch", "torchvision"]
        )

    def test_new_python_branch_is_not_qualified_merely_by_one_wheel(self):
        self.assertTrue(python_bundle_supported("rocm10.1", (3, 12)))
        self.assertFalse(python_bundle_supported("rocm10.1", (3, 10)))
        self.assertFalse(python_bundle_supported("cpu", (3, 15)))
        self.assertFalse(python_bundle_supported("cpu", (3, 12), "pypy"))
        self.assertIsNone(python_bundle_supported("unknown", (3, 12)))

    def test_modern_and_legacy_are_distinct_and_amd_keeps_vendor_companions(self):
        self.assertEqual(bundle_for("cpu", (3, 12)).torch, "2.15.0")
        self.assertEqual(bundle_for("cpu", (3, 10)).torch, "2.14.1")
        self.assertEqual(bundle_for("xpu", (3, 12)).torch, "2.15.0")
        self.assertFalse(python_bundle_supported("xpu", (3, 10)))
        self.assertEqual(bundle_for("cu132", (3, 12)).torch, "2.15.0")
        self.assertEqual(bundle_for("cu132", (3, 10)).torch, "2.14.1")
        self.assertEqual(BUNDLES["cu126"].torch, "2.14.1")
        self.assertIn("a0+rocm10.1.0", BUNDLES["rocm10.1"].vision)
        with self.assertRaises(TypeError):
            BUNDLES["cpu"] = BUNDLES["cu132"]

    def test_old_cards_and_new_cards_select_different_candidates(self):
        self.assertEqual(cuda_selection.cuda_candidates(6.1, (13, 2)), ["cu126", "cu124"])
        self.assertEqual(cuda_selection.cuda_candidates(8.6, (13, 2))[0], "cu132")
        self.assertEqual(cuda_selection.cuda_candidates(8.6, (13, 4))[0], "cu134")
        self.assertFalse(cuda_selection.cuda_candidates(3.5, (12, 6)))

    def test_stable_cuda134_does_not_implicitly_select_nightlies(self):
        self.assertEqual(
            cuda_selection.cuda_index("cu134"), "https://download.pytorch.org/whl/cu134"
        )
        self.assertEqual(
            cuda_selection.cuda_index("cu134", nightly=True),
            "https://download.pytorch.org/whl/nightly/cu134",
        )
        with patch.object(cuda_selection, "wheel_available", return_value=True) as available:
            self.assertTrue(cuda_selection.cuda_wheels_available("cu134"))
        self.assertEqual([call.args[2] for call in available.call_args_list], ["2.15.0", "0.29.2"])
        self.assertFalse(python_bundle_supported("cu134", (3, 10)))

    def test_explicit_legacy_validates_all_devices_driver_and_wheel_tags(self):
        with (
            patch.object(cuda_selection, "nvidia_devices", return_value=((6.1, 8.6), (13, 2))),
            patch.object(
                cuda_selection,
                "cuda_wheels_available",
                side_effect=lambda runtime: runtime == "cu126",
            ),
        ):
            self.assertEqual(cuda_selection.select_cuda_legacy("Windows"), "cu126")
        for capabilities, driver in [
            ((6.1,), (12, 3)),
            ((12.1,), (13, 2)),
            ((), None),
            ((0.0,), (13, 2)),
        ]:
            with (
                self.subTest(capabilities=capabilities),
                patch.object(cuda_selection, "nvidia_devices", return_value=(capabilities, driver)),
                patch.object(cuda_selection, "cuda_wheels_available", return_value=True),
            ):
                with self.assertRaises(RuntimeError):
                    cuda_selection.select_cuda_legacy("Linux")

    def test_offline_legacy_uses_installed_compatible_bundle_without_index_access(self):
        with (
            patch.object(cuda_selection, "nvidia_devices", return_value=((6.1,), (12, 6))),
            patch.object(
                cuda_selection,
                "cuda_wheels_available",
                side_effect=AssertionError("Network forbidden"),
            ),
            patch.object(importlib.metadata, "version", return_value="2.14.1+cu126"),
        ):
            self.assertEqual(cuda_selection.select_cuda_legacy("Linux", install=False), "cu126")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                cuda_selection.select_cuda_legacy("Linux", "cu124", install=False)

    def test_legacy_rejects_conflicts_and_unmanaged_incompatible_packages(self):
        for runtime in ("rocm10.1", "cpu", "cu132"):
            with self.subTest(runtime=runtime), self.assertRaisesRegex(RuntimeError, "conflicts"):
                cuda_selection.select_cuda_legacy("Windows", runtime)
        for version in ("2.15.0+cu132", "2.15.0+cu126", "2.14.1+cpu", "2.14.0+rocm10.1.0"):
            with (
                self.subTest(version=version),
                patch.object(importlib.metadata, "version", return_value=version),
                self.assertRaisesRegex(RuntimeError, "outside"),
            ):
                cuda_selection.select_cuda_legacy("Windows", install=False)


if __name__ == "__main__":
    unittest.main()
