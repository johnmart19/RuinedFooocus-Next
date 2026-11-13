"""Intel Arc / Core Ultra GPUs get pinned XPU PyTorch and the SYCL llama.cpp build."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ruinedfooocus.backends.llama import llama_installer
from ruinedfooocus.packaging import backend_preferences, hardware_inventory, runtime_support
from ruinedfooocus.packaging.gpu_installer import torch_install_commands
from ruinedfooocus.packaging.runtime_bundles import bundle_for

INTEL = {"vendor": "intel", "name": "Arc B580", "pci_id": "8086:e20b"}
AMD = {"vendor": "amd", "name": "Radeon RX 7900 XTX", "pci_id": "1002:744c"}


class XpuTorch(unittest.TestCase):
    def test_installed_xpu_build_is_recognised(self):
        self.assertEqual(hardware_inventory.installed_torch_kind("2.14.1+xpu"), "xpu")

    def test_install_is_pinned_to_the_official_xpu_index(self):
        if sys.version_info[:2] < (3, 11):
            self.skipTest("XPU wheels need Python 3.11+")
        bundle = bundle_for("xpu")
        (command,) = torch_install_commands(None, "xpu", "Windows")
        self.assertIn(f"torch=={bundle.torch}", command)
        self.assertIn(f"torchvision=={bundle.vision}", command)
        self.assertEqual(command[-1], "https://download.pytorch.org/whl/xpu")

    def test_saved_xpu_switch_exports_the_platform(self):
        args = SimpleNamespace(directml=None, rocm10=False, cuda_nightly=False, legacy=False)
        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "ruinedfooocus.packaging.startup_options.saved_startup_policy",
                return_value={"image_backend": "xpu"},
            ),
            patch.object(backend_preferences, "installed_torch_kind", return_value="cpu"),
        ):
            os.environ.pop("TORCH_PLATFORM", None)
            decision = backend_preferences.apply_saved_image_backend(
                args, "Windows", reinstall=True
            )
            self.assertEqual(decision["action"], "apply")
            self.assertEqual(os.environ["TORCH_PLATFORM"], "xpu")


class MixedVendorDetection(unittest.TestCase):
    def detect(self, *vendors):
        gpus = [SimpleNamespace(vendor_id=vendor) for vendor in vendors]
        detection = SimpleNamespace(
            get_torch_platform=lambda _: (_ for _ in ()).throw(NotImplementedError("multiple"))
        )
        return runtime_support._platform_for(SimpleNamespace(platform_detection=detection), gpus)

    def test_amd_dgpu_with_core_ultra_igpu_leaves_rocm_planning_to_amd(self):
        self.assertEqual(self.detect("1002", "8086"), "cpu")

    def test_intel_only_conflict_uses_xpu(self):
        # A vendor label cannot turn an unrecognized device into XPU support.
        self.assertEqual(self.detect("8086", "8086"), "cpu")

    def test_mixed_vendors_try_a_qualified_intel_group_before_cpu(self):
        gpus = [SimpleNamespace(vendor_id="1002"), SimpleNamespace(vendor_id="8086")]

        def platform_for(group):
            if len(group) > 1:
                raise NotImplementedError("multiple vendors")
            return "xpu" if group[0].vendor_id == "8086" else "cpu"

        detection = SimpleNamespace(get_torch_platform=platform_for)
        self.assertEqual(
            runtime_support._platform_for(SimpleNamespace(platform_detection=detection), gpus),
            "xpu",
        )


class SyclChat(unittest.TestCase):
    def test_sycl_is_offered_only_for_intel_gpus(self):
        with patch("ruinedfooocus.backends.llama.llama_build.built_server", return_value=None):
            intel = hardware_inventory.llm_backend_choices([INTEL], "Windows", "AMD64")
            amd = hardware_inventory.llm_backend_choices([AMD], "Windows", "AMD64")
        self.assertIn("SYCL", intel)
        self.assertNotIn("SYCL", amd)

    def test_sycl_packages_match_official_release_assets(self):
        self.assertEqual(
            llama_installer.package_name("Windows", "AMD64", True, backend="SYCL"),
            f"llama-{llama_installer.VERSION}-bin-win-sycl-x64.zip",
        )
        with patch("ruinedfooocus.backends.llama.llama_build.built_server", return_value=None):
            linux = llama_installer.package_name("Linux", "x86_64", True, backend="SYCL")
        self.assertEqual(linux, f"llama-{llama_installer.VERSION}-bin-ubuntu-sycl-fp16-x64.tar.gz")


if __name__ == "__main__":
    unittest.main()
