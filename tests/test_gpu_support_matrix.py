"""Per-GPU and per-OS choices for every component; nothing installs, probes or loads."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ruinedfooocus.backends.onnx import onnx_installer
from ruinedfooocus.packaging import gpu_installer
from ruinedfooocus.packaging import hardware_inventory as inventory
from ruinedfooocus.packaging import support_matrix

NVIDIA = {"vendor": "nvidia", "name": "GeForce RTX 4090", "pci_id": "10de:2684"}
AMD = {"vendor": "amd", "name": "Radeon RX 7900 XTX", "pci_id": "1002:744c"}
INTEL = {"vendor": "intel", "name": "Arc A770", "pci_id": "8086:56a0"}
WSL_RELEASE = "6.6.87.2-microsoft-standard-WSL2"
HOST_JSON = json.dumps(
    [
        {"Name": "AMD Radeon RX 7900 XTX", "PNPDeviceID": "PCI\\VEN_1002&DEV_744C&SUBSYS_1"},
        {"Name": "Microsoft Basic Display Adapter", "PNPDeviceID": "ROOT\\BasicDisplay\\0000"},
        {"Name": "Virtual", "PNPDeviceID": "PCI\\VEN_1414&DEV_008E"},
    ]
)


class OperatingSystems(unittest.TestCase):
    def test_wsl_is_its_own_install_target(self):
        self.assertEqual(inventory.os_kind("Linux", WSL_RELEASE), "WSL")
        self.assertEqual(inventory.os_kind("Linux", "6.8.0-45-generic"), "Linux")
        self.assertEqual(inventory.os_kind("Windows", "11"), "Windows")

    def test_host_gpus_are_read_from_the_windows_query(self):
        adapters = inventory.parse_host_gpus(HOST_JSON)
        self.assertEqual(
            [(gpu["vendor"], gpu["name"], gpu["pci_id"]) for gpu in adapters],
            [("amd", "AMD Radeon RX 7900 XTX", "1002:744c")],
        )
        self.assertEqual(adapters[0]["location"], "host")
        self.assertIsNone(adapters[0]["guest_available"])
        single = json.dumps({"Name": "Arc B580", "PNPDeviceID": "PCI\\VEN_8086&DEV_E20B"})
        self.assertEqual(inventory.parse_host_gpus(single)[0]["vendor"], "intel")
        self.assertEqual(inventory.parse_host_gpus("not json"), [])

    def test_wsl_detection_replaces_the_virtual_adapter_with_host_gpus(self):
        class Virtual:
            vendor_id, device_id, device_name = "1414", "008e", "Microsoft Basic Render Driver"

        inventory._gpus = None
        with (
            patch("torchruntime.device_db.get_gpus", return_value=[Virtual()]),
            patch.object(inventory, "is_wsl", return_value=True),
            patch.object(inventory, "wsl_host_gpus", return_value=[AMD]),
            patch("ruinedfooocus.packaging.cuda_selection.nvidia_devices", return_value=((), None)),
        ):
            gpus = inventory.detect_gpus(refresh=True)
        inventory._gpus = None
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0]["pci_id"], AMD["pci_id"])
        self.assertEqual(gpus[0]["location"], "host")
        self.assertIsNone(gpus[0]["guest_available"])

    def test_wsl_torch_platform_does_not_promote_host_ids_to_guest_capability(self):
        from ruinedfooocus.packaging import runtime_support

        class Torchruntime:
            class device_db:
                @staticmethod
                def get_device_infos(ids):
                    return [("looked-up", ids)]

        with (
            patch.object(inventory, "is_wsl", return_value=True),
            patch.object(inventory, "wsl_host_gpus", return_value=[INTEL]),
        ):
            self.assertEqual(
                runtime_support.wsl_device_infos(Torchruntime, []),
                [],
            )
        with patch.object(inventory, "is_wsl", return_value=False):
            self.assertEqual(runtime_support.wsl_device_infos(Torchruntime, []), [])


class ComponentChoices(unittest.TestCase):
    def test_onnx_runtimes_follow_vendor_and_os(self):
        def choices(gpus, system="Windows", machine="AMD64", wsl=False):
            with patch.object(inventory, "is_wsl", return_value=wsl):
                return inventory.onnx_backend_choices(gpus, system, machine)

        self.assertEqual(choices([NVIDIA]), ["Auto", "CPU", "CUDA", "Windows GPU"])
        self.assertEqual(choices([INTEL], "Linux"), ["Auto", "CPU", "OpenVINO"])
        self.assertEqual(choices([AMD], "Linux"), ["Auto", "CPU", "MIGraphX"])
        # AMD does not support MIGraphX on WSL, and there is no Windows MIGraphX build.
        self.assertEqual(choices([AMD], "Linux", wsl=True), ["Auto", "CPU"])
        self.assertEqual(choices([AMD]), ["Auto", "CPU", "Windows GPU"])
        self.assertEqual(choices([NVIDIA], "Windows", "arm64"), ["Auto", "CPU", "Windows GPU"])

    def test_xllamacpp_backends_follow_vendor_and_os(self):
        self.assertEqual(
            inventory.xllamacpp_backend_choices([NVIDIA], "Windows", "AMD64"),
            ["Auto", "CPU", "Vulkan", "CUDA 12.8", "CUDA 13.2"],
        )
        self.assertEqual(
            inventory.xllamacpp_backend_choices([AMD], "Linux", "x86_64", "2.14.1+cpu"),
            ["Auto", "CPU", "Vulkan", "ROCm 7.2"],
        )
        # The ROCm 7.2 wheel crashes in the same process as ROCm 10 PyTorch.
        self.assertEqual(
            inventory.xllamacpp_backend_choices([AMD], "Linux", "x86_64", "2.14.0+rocm10.1.0"),
            ["Auto", "CPU", "Vulkan"],
        )
        self.assertEqual(
            inventory.xllamacpp_backend_choices([AMD], "Windows", "AMD64"),
            ["Auto", "CPU", "Vulkan"],
        )
        self.assertEqual(
            inventory.xllamacpp_backend_choices([], "Linux", "x86_64"), ["Auto", "CPU"]
        )

    def test_explicit_xllamacpp_backend_picks_its_index_without_silent_fallback(self):
        root = "https://xorbitsai.github.io/xllamacpp/whl"
        self.assertEqual(
            gpu_installer.xllamacpp_index("cpu", "Linux", choice="ROCm 7.2"),
            f"{root}/rocm-7.2.4",
        )
        self.assertEqual(
            gpu_installer.xllamacpp_index("cu130", "Windows", choice="CPU"),
            "https://pypi.org/simple",
        )
        with self.assertRaisesRegex(RuntimeError, "Linux"):
            gpu_installer.xllamacpp_index("cpu", "Windows", choice="ROCm 7.2")
        with self.assertRaisesRegex(RuntimeError, "ROCm 10"):
            gpu_installer.xllamacpp_index("rocm10.1", "Linux", choice="ROCm 7.2")
        self.assertTrue(gpu_installer.xllamacpp_index("rocm10.1", "Linux").endswith("/vulkan"))
        with (
            patch("ruinedfooocus.packaging.cuda_selection.wheel_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "cu132"),
        ):
            gpu_installer.xllamacpp_index("cu130", "Linux", "2026.10.11870", choice="CUDA 13.2")

    def test_installed_xllamacpp_backend_comes_from_the_install_record(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "xllamacpp-backend.json"
            marker.write_text(json.dumps({"index": "https://x/whl/vulkan"}), encoding="utf-8")
            with (
                patch("importlib.metadata.version", return_value="2026.10.11870"),
                patch("ruinedfooocus.core.paths.data_path", return_value=marker),
            ):
                self.assertEqual(
                    inventory.xllamacpp_installed(),
                    {"version": "2026.10.11870", "backend": "Vulkan"},
                )


class IsolatedOnnxRuntimes(unittest.TestCase):
    def test_cuda_build_follows_the_installed_pytorch_cuda_major(self):
        self.assertEqual(onnx_installer.native_runtime("CUDA", "2.11.0+cu128").version, "1.29.0")
        self.assertEqual(
            onnx_installer.native_runtime("CUDA", "2.11.0+cu128").index, onnx_installer.CUDA12_FEED
        )
        cuda13 = onnx_installer.native_runtime("CUDA", "2.14.1+cu130")
        self.assertEqual((cuda13.version, cuda13.extras), ("1.32.0", ()))
        # Without CUDA PyTorch the CUDA/cuDNN libraries come from the package extras.
        cpu = onnx_installer.native_runtime("CUDA", "2.14.0+rocm10.1.0")
        self.assertEqual(cpu.extras, ("cuda", "cudnn"))

    def test_plan_is_isolated_pinned_and_os_checked(self):
        with (
            patch.object(onnx_installer.platform, "system", return_value="Linux"),
            patch.object(onnx_installer.platform, "machine", return_value="x86_64"),
        ):
            with self.assertRaisesRegex(RuntimeError, "exact native Linux target"):
                onnx_installer.native_install_plan("MIGraphX", "folder")
        self.assertEqual(
            onnx_installer.NATIVE_RUNTIMES["MIGraphX"].package, "onnxruntime-ep-migraphx"
        )
        with (
            patch.object(onnx_installer.platform, "system", return_value="Windows"),
            patch.object(onnx_installer.platform, "machine", return_value="AMD64"),
        ):
            with self.assertRaisesRegex(RuntimeError, "not published"):
                onnx_installer.native_install_plan("MIGraphX", "folder")
            openvino = onnx_installer.native_install_plan("OpenVINO", "folder")
        self.assertIn("openvino==2025.4.1", openvino)

    def test_missing_requirements_include_requested_extras_only(self):
        with tempfile.TemporaryDirectory() as folder:
            info = Path(folder) / "onnxruntime_gpu-1.31.1.dist-info"
            info.mkdir()
            (info / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: onnxruntime-gpu\nVersion: 1.31.1\n"
                "Requires-Dist: rf-test-missing-lib>=1\n"
                'Requires-Dist: rf-test-cuda-lib==13.0; extra == "cuda"\n',
                encoding="utf-8",
            )
            self.assertEqual(
                onnx_installer.missing_requirements(folder), ["rf-test-missing-lib>=1"]
            )
            self.assertEqual(
                onnx_installer.missing_requirements(folder, ("cuda",)),
                ["rf-test-cuda-lib==13.0", "rf-test-missing-lib>=1"],
            )


class SupportTable(unittest.TestCase):
    def rows(self, gpus, system, machine="AMD64", release="11", **installed):
        with patch.object(inventory, "is_wsl", return_value="microsoft" in release.lower()):
            with patch("ruinedfooocus.backends.llama.llama_build.built_server", return_value=None):
                return {
                    (row["component"], row["option"]): row
                    for row in support_matrix.support_rows(
                        gpus, system, machine, installed, release
                    )
                }

    def test_amd_on_windows_lists_rocm_and_hides_vendor_options_with_reasons(self):
        rows = self.rows([AMD], "Windows", kind="rocm", torch="2.14.0+rocm10.1.0")
        self.assertEqual(
            rows[("ComfyUI (PyTorch)", "ROCm (AMD)")]["status"], "Installed (unverified)"
        )
        self.assertEqual(
            rows[("ComfyUI (PyTorch)", "CUDA (NVIDIA)")]["note"], "Needs an NVIDIA GPU"
        )
        self.assertEqual(rows[("xllamacpp", "ROCm 7.2")]["note"], "Linux and WSL only")
        self.assertEqual(rows[("ONNX Runtime", "MIGraphX (AMD)")]["note"], "Native Linux only")
        self.assertEqual(rows[("Windows ML", "AMDGPU")]["status"], "Candidate")
        self.assertEqual(rows[("Windows ML", "OpenVINO")]["note"], "Needs an Intel GPU")

    def test_wsl_lists_no_windows_ml_and_no_migraphx(self):
        with patch.object(inventory, "rocm10_torch", return_value=False):
            rows = self.rows([AMD], "Linux", "x86_64", WSL_RELEASE)
        self.assertEqual(rows[("Windows ML", "AMDGPU")]["note"], "Windows only")
        self.assertEqual(rows[("ONNX Runtime", "MIGraphX (AMD)")]["status"], "Not available")
        self.assertEqual(rows[("xllamacpp", "ROCm 7.2")]["status"], "Candidate")
        self.assertIn(support_matrix.WSL_VULKAN_NOTE, rows[("xllamacpp", "Vulkan")]["note"])

    def test_rocm10_pytorch_blocks_the_rocm7_xllamacpp_wheel(self):
        with patch.object(inventory, "rocm10_torch", return_value=True):
            rows = self.rows([AMD], "Linux", "x86_64", WSL_RELEASE)
        self.assertEqual(rows[("xllamacpp", "ROCm 7.2")]["note"], "Conflicts with ROCm 10 PyTorch")
        self.assertEqual(rows[("llama.cpp", "ROCm")]["status"], "Candidate")

    def test_resolved_rows_render_exact_shared_evidence_and_adapter_identity(self):
        decision = dict(
            backend="Vulkan",
            operation="llama",
            adapter_id="guest:gpu-2",
            state="unknown",
            active=False,
            eligible=False,
            usable=False,
            qualified=False,
            reasons=("Guest runtime visibility is unknown",),
        )
        rows = support_matrix.support_rows(decisions=[decision])
        self.assertEqual(rows[0]["status"], "Unknown")
        self.assertEqual(rows[0]["adapter_id"], "guest:gpu-2")
        self.assertIn(decision["reasons"][0], rows[0]["note"])


class SavedValues(unittest.TestCase):
    def test_new_backend_values_are_valid_settings(self):
        from ruinedfooocus.core.settings_schema import validate_known

        validate_known({"image_backend": "xpu", "xllamacpp_backend": "ROCm 7.2"})
        validate_known({"rembg_backend": "OpenVINO"})
        with self.assertRaises(ValueError):
            validate_known({"xllamacpp_backend": "ROCm 6.4"})

    def test_windows_ml_providers_follow_the_detected_vendor(self):
        from ruinedfooocus.backends.onnx.windows_ml import CATALOG_PROVIDERS

        names = support_matrix.windows_ml_provider_names
        self.assertEqual(
            names(CATALOG_PROVIDERS, [INTEL], machine="AMD64"), ["OpenVINOExecutionProvider"]
        )
        self.assertIn(
            "QNNExecutionProvider",
            names(CATALOG_PROVIDERS, [INTEL], "QNNExecutionProvider", machine="AMD64"),
        )
        self.assertIn("QNNExecutionProvider", names(CATALOG_PROVIDERS, [], machine="ARM64"))


if __name__ == "__main__":
    unittest.main()
