"""Backend choices follow detected hardware; nothing probes or initialises a device."""

import unittest
from unittest.mock import patch

from ruinedfooocus.packaging import hardware_inventory as inventory

NVIDIA = {"vendor": "nvidia", "name": "GeForce RTX 4090", "pci_id": "10de:2684"}
AMD = {"vendor": "amd", "name": "Radeon RX 7900 XTX", "pci_id": "1002:744c"}
INTEL = {"vendor": "intel", "name": "Arc A770", "pci_id": "8086:56a0"}


def image(gpus, system="Windows", **installed):
    return inventory.image_backend_choices(gpus, system, installed, "AMD64")


def llm(gpus, system="Windows", machine="AMD64"):
    with patch("ruinedfooocus.backends.llama.llama_build.built_server", return_value=None):
        return inventory.llm_backend_choices(gpus, system, machine)


class ImageChoices(unittest.TestCase):
    def test_choices_follow_the_installed_vendors(self):
        self.assertEqual(image([NVIDIA]), ["auto", "cuda", "cpu"])
        self.assertEqual(image([AMD]), ["auto", "rocm", "cpu"])
        self.assertEqual(image([INTEL]), ["auto", "xpu", "cpu"])
        self.assertEqual(image([]), ["auto", "cpu"])
        self.assertEqual(image([AMD, NVIDIA]), ["auto", "cuda", "rocm", "cpu"])

    def test_platforms_without_vendor_wheels_only_offer_cpu(self):
        self.assertEqual(image([AMD], "Darwin"), ["auto", "cpu"])
        self.assertEqual(
            inventory.image_backend_choices([AMD], "Windows", {}, "arm64"), ["auto", "cpu"]
        )
        self.assertEqual(image([AMD], "Linux"), ["auto", "rocm", "cpu"])

    def test_an_installed_build_keeps_its_choice_even_when_pci_hides_the_gpu(self):
        self.assertIn("rocm", image([], "Linux", kind="rocm"))
        self.assertIn("cuda", image([], "Linux", kind="cuda"))

    def test_directml_only_appears_when_already_installed(self):
        self.assertNotIn("directml", image([AMD]))
        self.assertIn("directml", image([AMD], directml=True))


class LlmChoices(unittest.TestCase):
    def test_cuda_and_rocm_labels_need_matching_hardware(self):
        self.assertEqual(llm([NVIDIA]), ["Auto", "CPU", "Vulkan", "CUDA 12.4", "CUDA 13.4"])
        self.assertEqual(llm([AMD]), ["Auto", "CPU", "Vulkan", "ROCm"])
        self.assertEqual(llm([INTEL]), ["Auto", "CPU", "Vulkan", "SYCL"])
        self.assertEqual(llm([]), ["Auto", "CPU"])
        self.assertEqual(
            llm([AMD, NVIDIA]), ["Auto", "CPU", "Vulkan", "ROCm", "CUDA 12.4", "CUDA 13.4"]
        )

    def test_linux_and_wsl_use_the_linux_package_names(self):
        self.assertEqual(
            llm([NVIDIA], "Linux", "x86_64"), ["Auto", "CPU", "Vulkan", "CUDA 12.8", "CUDA 13.4"]
        )
        self.assertEqual(llm([AMD], "Linux", "x86_64"), ["Auto", "CPU", "Vulkan", "ROCm"])

    def test_a_finished_local_cuda_build_is_offered_only_with_nvidia(self):
        with patch(
            "ruinedfooocus.backends.llama.llama_build.built_server", return_value="/cache/server"
        ):
            nvidia = inventory.llm_backend_choices([NVIDIA], "Linux", "x86_64")
            amd = inventory.llm_backend_choices([AMD], "Linux", "x86_64")
        self.assertIn("CUDA (local build)", nvidia)
        self.assertNotIn("CUDA (local build)", amd)


class BackgroundChoicesAndStatus(unittest.TestCase):
    def test_windows_ml_is_windows_only(self):
        self.assertEqual(
            inventory.onnx_backend_choices([], "Windows", "AMD64"), ["Auto", "CPU", "Windows GPU"]
        )
        self.assertEqual(inventory.onnx_backend_choices([], "Linux", "x86_64"), ["Auto", "CPU"])
        self.assertEqual(inventory.onnx_backend_choices([], "Darwin", "arm64"), ["Auto", "CPU"])

    def test_torch_builds_are_classified_from_the_version_string(self):
        for version, kind in (
            ("2.14.0+rocm10.1.0", "rocm"),
            ("2.9.1+cu130", "cuda"),
            ("2.9.1+cpu", "cpu"),
            ("2.9.1", "other"),
        ):
            self.assertEqual(inventory.installed_torch_kind(version), kind)

    def test_gpu_detection_is_cached_and_survives_a_failed_scan(self):
        inventory._gpus = None
        with (
            patch("torchruntime.device_db.get_gpus", side_effect=OSError("no lspci")),
            patch("ruinedfooocus.packaging.cuda_selection.nvidia_devices", return_value=((), None)),
        ):
            self.assertEqual(inventory.detect_gpus(refresh=True), [])
        inventory._gpus = [AMD]
        returned = inventory.detect_gpus()
        returned[0]["name"] = "changed"
        self.assertEqual(inventory.detect_gpus()[0]["name"], "Radeon RX 7900 XTX")
        inventory._gpus = None

    def test_passive_inventory_never_runs_nvidia_smi(self):
        inventory._gpus = None
        with (
            patch("torchruntime.device_db.get_gpus", return_value=[]),
            patch(
                "ruinedfooocus.packaging.cuda_selection.nvidia_devices",
                side_effect=AssertionError("active device probes are forbidden"),
            ) as probe,
            patch("platform.system", return_value="Linux"),
            patch.object(inventory, "is_wsl", return_value=False),
        ):
            gpus = inventory.detect_gpus(refresh=True)
        self.assertEqual(gpus, [])
        probe.assert_not_called()
        inventory._gpus = None

    def test_duplicate_product_ids_are_preserved_as_separate_adapter_ids(self):
        adapters = inventory.identify_adapters([NVIDIA, NVIDIA])
        self.assertEqual(len(adapters), 2)
        self.assertNotEqual(adapters[0]["adapter_id"], adapters[1]["adapter_id"])
        self.assertFalse(adapters[0]["identity_stable"])
        stable = inventory.identify_adapters(
            [NVIDIA | {"hardware_id": "bus:01"}, NVIDIA | {"hardware_id": "bus:02"}]
        )
        self.assertTrue(all(gpu["identity_stable"] for gpu in stable))

    def test_inventory_keeps_detection_failure_distinct_from_absence(self):
        with patch.object(inventory, "is_wsl", return_value=False):
            with patch("torchruntime.device_db.get_gpus", side_effect=OSError("unreadable")):
                self.assertEqual(inventory.gpu_inventory(refresh=True)["state"], "unknown")
            with patch("torchruntime.device_db.get_gpus", return_value=[]):
                self.assertEqual(inventory.gpu_inventory(refresh=True)["state"], "absent")
        inventory._gpus = None

    def test_component_status_reads_metadata_without_importing_torch(self):
        import sys

        had_torch = "torch" in sys.modules
        status = inventory.component_status("Linux")
        self.assertEqual(set(status), {"image", "llm", "background"})
        self.assertEqual("torch" in sys.modules, had_torch)


if __name__ == "__main__":
    unittest.main()
