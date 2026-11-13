"""WinML installation and selection contracts without loading a GPU provider."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.backends.onnx import windows_ml as winml
from ruinedfooocus.backends.onnx import onnx_runtime as runtime
from ruinedfooocus.core.settings_validation import validate_settings


class WindowsMLTests(unittest.TestCase):
    def test_old_explicit_windows_backend_preserves_intent_without_rewriting_saved_values(self):
        import onnxruntime as ort

        for values, legacy in (
            ({"rembg_backend": "Windows GPU"}, True),
            ({"rembg_backend": "Auto"}, False),
            ({"rembg_backend": "Windows GPU", "rembg_provider": "Auto"}, False),
        ):
            original = dict(values)
            with (
                self.subTest(values=values),
                patch.dict(
                    "sys.modules",
                    {
                        "ruinedfooocus.app.state": SimpleNamespace(
                            settings=SimpleNamespace(default_settings=values)
                        ),
                        "ruinedfooocus.app.arguments": SimpleNamespace(
                            args=SimpleNamespace(cpu=False, directml=None)
                        ),
                    },
                ),
                patch.object(ort, "get_available_providers", return_value=["CPUExecutionProvider"]),
                patch.object(runtime.platform, "system", return_value="Windows"),
                patch.object(runtime, "runtime_available", return_value=True),
                patch.object(runtime, "create_mask_session") as cpu,
            ):
                session = runtime.new_session()
                if legacy:
                    self.assertEqual(session.provider, "DirectML")
                    cpu.assert_not_called()
                else:
                    cpu.assert_called_once_with(providers=["CPUExecutionProvider"])
                self.assertEqual(values, original)

    def test_acquisition_uses_requested_device_type_and_only_certified_candidates(self):
        from ruinedfooocus.backends.onnx.winml_catalog import select_provider

        gpu, npu, multi, uncertified = [
            SimpleNamespace(name=name, certification=certification)
            for name, certification in (
                ("MIGraphXExecutionProvider", "certified"),
                ("VitisAIExecutionProvider", "certified"),
                ("OpenVINOExecutionProvider", "certified"),
                ("QNNExecutionProvider", "uncertified"),
            )
        ]
        providers = [gpu, multi, uncertified, npu]
        self.assertIs(select_provider(providers, "certified", device_type="GPU"), gpu)
        self.assertIs(select_provider(providers, "certified", device_type="NPU"), npu)
        self.assertIs(select_provider(providers, "certified", device_type="CPU"), multi)
        with self.assertRaisesRegex(RuntimeError, "NPU provider"):
            select_provider([gpu, uncertified], "certified", device_type="NPU")
        with patch.object(winml.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "does not target NPU"):
                winml.install_provider("MIGraphXExecutionProvider", "NPU")
        run.assert_not_called()

    def test_uncertified_error_lists_the_actual_local_catalogue(self):
        from ruinedfooocus.backends.onnx.winml_catalog import select_provider

        providers = [
            SimpleNamespace(name="AMDGPUExecutionProvider", certification=2),
            SimpleNamespace(name="MIGraphXExecutionProvider", certification=1),
        ]
        with self.assertRaisesRegex(
            RuntimeError, r"AMDGPU \(not certified\).*MIGraphX \(certified\)"
        ):
            select_provider(iter(providers), 1, "AMDGPUExecutionProvider", "GPU")
        self.assertEqual(
            select_provider(providers, 1, "Auto", "GPU").name, "MIGraphXExecutionProvider"
        )

    def test_installer_surfaces_the_catalogue_bridge_diagnostic(self):
        import subprocess

        failure = subprocess.CalledProcessError(
            1,
            ["bridge"],
            stderr="Traceback...\nRuntimeError: AMDGPU not certified; MIGraphX certified",
        )
        with (
            patch(
                "ruinedfooocus.packaging.startup_options.skip_dependency_install",
                return_value=False,
            ),
            patch.object(winml, "install_plan", return_value=["pip"]),
            patch(
                "ruinedfooocus.backends.onnx.onnx_installer.worker_command", return_value=["bridge"]
            ),
            patch.object(
                winml.subprocess, "run", side_effect=[SimpleNamespace(returncode=0), failure]
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "MIGraphX certified"):
                winml.install_provider("MIGraphXExecutionProvider", "GPU")

    def test_builtin_cpu_selection_needs_no_vendor_download_or_probe(self):
        with (
            patch.dict("os.environ", {"RF_OFFLINE": "1"}),
            patch.object(
                winml.subprocess, "run", side_effect=AssertionError("No subprocess on CPU")
            ),
        ):
            self.assertIn("already available", winml.install_provider(device_type="CPU"))
            self.assertEqual(winml.discover_devices(device_type="CPU"), [])

    def test_all_vendor_providers_select_only_the_requested_cpu_gpu_or_npu_class(self):
        self.assertEqual(
            set(winml.CATALOG_PROVIDERS),
            {
                "AMDGPUExecutionProvider",
                "MIGraphXExecutionProvider",
                "NvTensorRtRtxExecutionProvider",
                "OpenVINOExecutionProvider",
                "QNNExecutionProvider",
                "VitisAIExecutionProvider",
            },
        )
        for provider in winml.CATALOG_PROVIDERS:
            devices = [
                SimpleNamespace(
                    ep_name=provider,
                    device=SimpleNamespace(type=kind, vendor_id=1, device_id=0, metadata={}),
                )
                for kind in ("CPU", "GPU", "NPU")
            ]
            ort = SimpleNamespace(
                get_ep_devices=Mock(return_value=devices),
                OrtHardwareDeviceType=SimpleNamespace(CPU="CPU", GPU="GPU", NPU="NPU"),
            )
            self.assertEqual(len({winml.device_identity(device) for device in devices}), 3)
            for kind, device in zip(("CPU", "GPU", "NPU"), devices):
                self.assertIs(
                    winml.select_device(ort, {"name": provider}, device_type=kind), device
                )
            ort.get_ep_devices.return_value = devices[:1]
            with self.assertRaisesRegex(RuntimeError, "compatible NPU"):
                winml.select_device(ort, {"name": provider}, device_type="NPU")

    def test_directml_is_never_selected_automatically(self):
        import onnxruntime as ort

        with (
            patch.dict(
                "sys.modules",
                {
                    "ruinedfooocus.app.arguments": SimpleNamespace(
                        args=SimpleNamespace(cpu=False, directml=None)
                    )
                },
            ),
            patch.object(ort, "get_available_providers", return_value=["CPUExecutionProvider"]),
            patch.object(runtime, "runtime_available", return_value=True),
            patch.object(runtime.platform, "system", return_value="Windows"),
            patch.object(runtime, "create_mask_session") as factory,
        ):
            runtime.new_session(preference="Auto", provider="Auto")
            factory.assert_called_once_with(providers=["CPUExecutionProvider"])

    def test_metadata_does_not_import_or_probe_runtime_and_rejects_changed_library(self):
        with tempfile.TemporaryDirectory() as folder:
            library = Path(folder) / "provider.dll"
            library.write_bytes(b"trusted fixture")
            path = Path(folder) / "provider.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "MIGraphXExecutionProvider",
                        "library_path": str(library),
                        "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    }
                )
            )
            with (
                patch.object(winml, "manifest_path", return_value=path),
                patch.object(winml.platform, "system", return_value="Windows"),
                patch.object(
                    winml, "register_provider", side_effect=AssertionError("No device probe")
                ),
            ):
                self.assertIsNotNone(winml.installed_provider("MIGraphXExecutionProvider"))
                library.write_bytes(b"changed fixture")
                self.assertIsNone(winml.installed_provider("MIGraphXExecutionProvider"))

    def test_catalogue_bridge_is_isolated_and_offline_never_installs(self):
        with (
            patch.object(winml.platform, "system", return_value="Windows"),
            patch.object(winml.platform, "machine", return_value="AMD64"),
            patch.object(
                winml.sys,
                "getwindowsversion",
                return_value=SimpleNamespace(build=26100),
                create=True,
            ),
        ):
            command = winml.install_plan("private-target")
        self.assertNotIn("--no-deps", command)
        self.assertIn("--target", command)
        self.assertIn("private-target", command)
        self.assertIn("onnxruntime-windowsml==1.25.2.202605110140", command)
        with (
            patch.dict("os.environ", {"RF_OFFLINE": "1"}),
            patch.object(winml.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                winml.install_provider()
        run.assert_not_called()

    def test_provider_device_identity_is_not_a_torch_or_dml_index(self):
        def device(number):
            return SimpleNamespace(
                ep_name="MIGraphXExecutionProvider",
                hardware_device=SimpleNamespace(
                    type="GPU", vendor_id=4098, device_id=number, metadata={"LUID": str(number)}
                ),
            )

        first, second = device(8), device(2)
        ort = SimpleNamespace(
            get_ep_devices=Mock(return_value=[first, second]),
            OrtHardwareDeviceType=SimpleNamespace(GPU="GPU"),
        )
        record = {"name": first.ep_name}
        identity = winml.device_identity(second)
        self.assertEqual(len(identity), 24)
        self.assertIs(winml.select_device(ort, record, identity), second)
        ort.get_ep_devices.return_value.reverse()
        self.assertIs(winml.select_device(ort, record, identity), second)
        with self.assertRaisesRegex(RuntimeError, "missing or ambiguous"):
            winml.select_device(ort, record, "2")
        ort.get_ep_devices.return_value = [second, second]
        with self.assertRaisesRegex(RuntimeError, "missing or ambiguous"):
            winml.select_device(ort, record, identity)

    def test_explicit_winml_and_cpu_override_never_inherit_directml_device_number(self):
        args = SimpleNamespace(cpu=False, directml=None)
        state = SimpleNamespace(args=args)
        with (
            patch.dict("sys.modules", {"ruinedfooocus.app.arguments": state}),
            patch.object(
                winml, "installed_provider", return_value={"name": "MIGraphXExecutionProvider"}
            ),
            patch.object(runtime, "create_mask_session") as cpu,
        ):
            session = runtime.new_session(
                preference="Windows GPU", provider="Windows ML", ep_device="a" * 24
            )
            self.assertIsInstance(session, runtime.IsolatedMaskSession)
            self.assertIsNone(session.device_id)
            self.assertEqual(session.ep_device, "a" * 24)
            cpu.assert_not_called()
            args.cpu = True
            runtime.new_session(preference="Windows GPU", provider="Windows ML")
            cpu.assert_called_once_with(providers=["CPUExecutionProvider"])

    def test_new_settings_validate_without_migrating_existing_backend(self):
        result = validate_settings(
            {
                "rembg_backend": "Windows GPU",
                "rembg_provider": "Windows ML",
                "rembg_winml_device": "a" * 24,
            }
        )
        self.assertEqual(result["rembg_backend"], "Windows GPU")
        for values in ({"rembg_provider": "WinML renamed DirectML"}, {"rembg_winml_device": "0"}):
            with self.assertRaises(ValueError):
                validate_settings(values)
