"""Exact AMD routes and plugin boundaries, with no GPU imports or installs."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import io
import zipfile

from ruinedfooocus.backends.onnx import (
    acquisition,
    onnx_installer,
    plugins,
    provider_recipes as recipes,
    windows_ml,
)
from ruinedfooocus.packaging.runtime_support import RuntimeArtifact, RuntimeTarget


class ProviderRecipeTests(unittest.TestCase):
    def test_new_linux_plugin_cannot_enable_auto_with_forged_or_stale_qualified_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "onnxruntime-ep-migraphx-1.0.0+rocm10.1.0"
            root.mkdir()
            for evidence in (
                {"qualified": True},
                {
                    "qualified": True,
                    "cache_identity": "stale",
                    "contract": {"model_sha256": "derived", "driver": "old"},
                },
            ):
                (root / "verified.json").write_text(json.dumps({"workload_evidence": evidence}))
                with (
                    patch.object(onnx_installer, "native_runtime_available", return_value=True),
                    patch.object(onnx_installer, "native_runtime_path", return_value=root),
                ):
                    self.assertFalse(onnx_installer.native_runtime_auto_eligible("MIGraphX"))
            with (
                patch.object(onnx_installer, "native_runtime_available", return_value=True),
                patch.object(
                    onnx_installer,
                    "native_runtime_path",
                    return_value=Path(directory) / "onnxruntime-migraphx-1.27.1",
                ),
            ):
                for name in ("MIGraphX", "CUDA", "OpenVINO"):
                    self.assertTrue(onnx_installer.native_runtime_auto_eligible(name))

    def test_explicit_acquisition_produces_target_hashed_native_wheels_and_cleans_its_stage(self):
        wheels, report = {}, []
        for package in recipes.MIGRAPHX_PACKAGES:
            name, version = package.split("==")
            filename = f"{name.replace('-', '_')}-{version}-cp312-cp312-manylinux_2_28_x86_64.whl"
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("native/provider.so.1", b"native fixture only")
            wheels[filename] = buffer.getvalue()
            report.append(
                dict(
                    metadata=dict(name=name, version=version),
                    download_info=dict(
                        url=recipes.MIGRAPHX_INDEX + filename,
                        archive_info=dict(
                            hashes=dict(sha256=hashlib.sha256(wheels[filename]).hexdigest())
                        ),
                    ),
                )
            )

        def run(command, **kwargs):
            if "--report" in command:
                Path(command[command.index("--report") + 1]).write_text(
                    json.dumps({"install": report})
                )
            elif "--dest" in command:
                destination = Path(command[command.index("--dest") + 1])
                destination.mkdir()
                for filename, payload in wheels.items():
                    (destination / filename).write_bytes(payload)
            else:
                self.fail("Unexpected acquisition command")
            return SimpleNamespace(returncode=0)

        adapter = dict(self.adapter, vendor="amd", **{"rocm10.1_driver_supported": None})
        adapter.pop("onnx_device_id")
        with (
            patch.object(acquisition, "current_target", return_value=self.target),
            patch.object(acquisition, "linux_amd_adapters", return_value=[adapter]),
            patch.object(acquisition, "run", side_effect=run),
        ):
            with acquisition.prepare_migraphx() as prepared:
                self.assertEqual(prepared.target, self.target)
                self.assertTrue(prepared.evidence["native_payload"])
                self.assertNotIn("onnx_device_id", prepared.adapter)
                paths = list(prepared.evidence["artifact_paths"].values())
                self.assertTrue(all(Path(path).is_file() for path in paths))
        self.assertTrue(all(not Path(path).exists() for path in paths))

    def test_passive_linux_topology_uses_physical_pci_and_reported_gfx_not_ordinal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node = root / "class/kfd/kfd/topology/nodes/4"
            node.mkdir(parents=True)
            (node / "properties").write_text("gfx_target_version 110000\ndrm_render_minor 128\n")
            device = root / "class/drm/renderD128/device"
            device.mkdir(parents=True)
            (device / "vendor").write_text("0x1002")
            (device / "uevent").write_text("PCI_SLOT_NAME=0000:03:00.0\n")
            adapters = acquisition.linux_amd_adapters(root)
        self.assertEqual(adapters[0]["gfx"], "gfx1100")
        self.assertEqual(adapters[0]["adapter_id"], "native:pci:0000:03:00.0")
        self.assertNotIn("onnx_device_id", adapters[0])

    def setUp(self):
        self.target = RuntimeTarget(
            "Linux",
            "x86_64",
            "3.12.10",
            "cp312",
            frozenset(("cp312-cp312-manylinux_2_28_x86_64", "py3-none-manylinux_2_28_x86_64")),
        )
        self.adapter = dict(
            adapter_id="amd-pci",
            vendor="AMD",
            kind="gpu",
            location="native",
            gfx="gfx1100",
            onnx_device_id="a" * 24,
            **{"rocm10.1_driver_supported": True},
        )
        self.artifacts = tuple(
            RuntimeArtifact(
                name=name,
                version=version,
                filename=f"{name.replace('-', '_')}-{version}-cp312-cp312-manylinux_2_28_x86_64.whl",
                available=True,
                source=recipes.MIGRAPHX_INDEX,
                sha256="1" * 64,
            )
            for name, version in (p.split("==") for p in recipes.MIGRAPHX_PACKAGES)
        )
        self.evidence = dict(artifacts=self.artifacts, closure_complete=True, native_payload=True)

    def test_native_linux_plugin_recipe_is_exact_and_never_installs_integrated_legacy_wheel(self):
        decision = recipes.migraphx_decision(self.target, self.adapter, **self.evidence)
        self.assertTrue(decision["eligible"], decision)
        self.assertFalse(decision["auto_eligible"])
        with (
            patch.object(onnx_installer.platform, "system", return_value="Linux"),
            patch.object(onnx_installer.platform, "machine", return_value="x86_64"),
        ):
            command = onnx_installer.native_install_plan(
                "MIGraphX",
                target="private",
                target_info=self.target,
                adapter=self.adapter,
                evidence=self.evidence,
            )
        for package in recipes.MIGRAPHX_PACKAGES:
            name, version = package.split("==")
            self.assertTrue(
                any(
                    part.startswith(name + " @ ") and version in part and "#sha256=" in part
                    for part in command
                )
            )
        self.assertFalse(any("onnxruntime-migraphx" in part for part in command))
        self.assertIn(recipes.MIGRAPHX_LIBS_INDEX, command)

    def test_wrong_os_wsl_python_gfx_vendor_and_placeholder_are_rejected(self):
        for target, adapter, evidence in (
            (replace(self.target, system="Windows"), self.adapter, self.evidence),
            (replace(self.target, environment="wsl"), self.adapter, self.evidence),
            (
                replace(self.target, python_version="3.14.0", python_abi="cp314"),
                self.adapter,
                self.evidence,
            ),
            (self.target, dict(self.adapter, gfx="gfx9999"), self.evidence),
            (self.target, dict(self.adapter, vendor="NVIDIA"), self.evidence),
            (self.target, self.adapter, dict(self.evidence, native_payload=False)),
            (
                self.target,
                self.adapter,
                dict(
                    self.evidence,
                    artifacts=(replace(self.artifacts[0], version="0.0.0"), *self.artifacts[1:]),
                ),
            ),
        ):
            with self.subTest(target=target, adapter=adapter):
                self.assertFalse(recipes.migraphx_decision(target, adapter, **evidence)["eligible"])

    def test_unknown_closure_and_hash_cannot_be_offered_for_installation(self):
        self.assertFalse(recipes.migraphx_decision(self.target, self.adapter)["eligible"])
        artifacts = (replace(self.artifacts[0], sha256=""), *self.artifacts[1:])
        decision = recipes.migraphx_decision(
            self.target, self.adapter, **dict(self.evidence, artifacts=artifacts)
        )
        self.assertIn("SHA-256", "; ".join(decision["reasons"]))
        with (
            patch.object(onnx_installer.platform, "system", return_value="Linux"),
            patch.object(onnx_installer.platform, "machine", return_value="x86_64"),
        ):
            with self.assertRaisesRegex(RuntimeError, "exact native Linux target"):
                onnx_installer.native_install_plan("MIGraphX")

    def test_hip_cp312_gfx1100_has_precise_gate_and_is_never_auto(self):
        decision = recipes.hip_decision(self.target, self.adapter)
        self.assertFalse(decision["eligible"])
        self.assertFalse(decision["auto_eligible"])
        self.assertIn("cp314", "; ".join(decision["reasons"]))
        self.assertIn("gfx1100", "; ".join(decision["reasons"]))

    def test_plugin_validates_apis_alias_native_payload_and_record_before_registration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            library = root / "provider.so"
            library.write_bytes(b"native fixture only")
            amd = SimpleNamespace(
                ep_name="MIGraphXExecutionProvider", device=SimpleNamespace(vendor_id=0x1002)
            )
            nvidia = SimpleNamespace(
                ep_name="CUDAExecutionProvider", device=SimpleNamespace(vendor_id=0x10DE)
            )
            module = SimpleNamespace(
                get_ep_names=lambda: ["MIGraphXExecutionProvider"],
                get_library_paths=lambda: [str(library)],
            )
            ort = SimpleNamespace(
                register_execution_provider_library=Mock(), get_ep_devices=lambda: [nvidia, amd]
            )
            with patch.object(plugins.importlib, "import_module", return_value=module):
                devices, records = plugins.register_migraphx(ort, root)
                self.assertEqual(devices, [amd])
                plugins.register_migraphx(ort, root, records)
                self.assertEqual(ort.register_execution_provider_library.call_count, 2)
                library.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    plugins.register_migraphx(ort, root, records)
                self.assertEqual(ort.register_execution_provider_library.call_count, 2)
            for module, reason in (
                (SimpleNamespace(), "placeholder"),
                (
                    SimpleNamespace(
                        get_ep_names=lambda: ["hipgpu"], get_library_paths=lambda: [str(library)]
                    ),
                    "alias",
                ),
                (
                    SimpleNamespace(
                        get_ep_names=lambda: ["MIGraphXExecutionProvider"],
                        get_library_paths=lambda: ["missing.so"],
                    ),
                    "library",
                ),
            ):
                with (
                    patch.object(plugins.importlib, "import_module", return_value=module),
                    self.assertRaisesRegex(RuntimeError, reason),
                ):
                    plugins.register_migraphx(ort, root)
            with self.assertRaisesRegex(RuntimeError, "registration/device API"):
                plugins.register_migraphx(SimpleNamespace(), root)

    def test_stale_first_winml_record_does_not_hide_valid_second_or_probe_a_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            library = root / "provider.dll"
            library.write_bytes(b"fixture")
            record = dict(
                name="OpenVINOExecutionProvider",
                library_path=str(library),
                sha256=hashlib.sha256(b"fixture").hexdigest(),
            )
            document = {
                "providers": {
                    "MIGraphXExecutionProvider": dict(
                        record, name="MIGraphXExecutionProvider", sha256="0" * 64
                    ),
                    record["name"]: record,
                }
            }
            manifest = root / "provider.json"
            manifest.write_text(json.dumps(document))
            with (
                patch.object(windows_ml, "manifest_path", return_value=manifest),
                patch.object(windows_ml.platform, "system", return_value="Windows"),
                patch.object(
                    windows_ml,
                    "register_provider",
                    side_effect=AssertionError("No passive registration"),
                ),
            ):
                self.assertEqual(windows_ml.installed_providers(), [record])
                self.assertIsNone(windows_ml.installed_provider())

    def test_registration_failure_and_missing_device_try_later_candidate(self):
        records = [
            {"name": name, "workload_evidence": {"contract": {"device": "a" * 24}}}
            for name in ("MIGraphXExecutionProvider", "OpenVINOExecutionProvider")
        ]
        handle = Mock()
        for failure in (RuntimeError("stale plugin"), OSError("missing DLL")):
            with (
                patch.object(windows_ml, "installed_providers", return_value=records),
                patch.object(windows_ml, "matching_record", return_value=True),
                patch.object(windows_ml, "device_identity", return_value="a" * 24),
                patch.object(windows_ml, "hardware", return_value=SimpleNamespace(type="GPU")),
                patch.object(
                    windows_ml, "register_provider", side_effect=[failure, (None, handle)]
                ),
                patch.object(windows_ml, "select_device", return_value="device"),
            ):
                record, device, handles = windows_ml.select_installed_device(Mock())
            self.assertEqual(record, records[1])
            self.assertEqual(device, "device")
        with (
            patch.object(windows_ml, "installed_providers", return_value=records),
            patch.object(windows_ml, "matching_record", return_value=True),
            patch.object(windows_ml, "device_identity", return_value="a" * 24),
            patch.object(windows_ml, "hardware", return_value=SimpleNamespace(type="GPU")),
            patch.object(windows_ml, "register_provider", return_value=(None, handle)),
            patch.object(
                windows_ml, "select_device", side_effect=[RuntimeError("no GPU"), "device"]
            ),
        ):
            self.assertEqual(windows_ml.select_installed_device(Mock())[0], records[1])
        handle.close.assert_called_once()

    def test_winml_dll_directory_lives_until_session_owner_releases_it(self):
        import ctypes

        handle = Mock()
        ort = SimpleNamespace(register_execution_provider_library=Mock(), get_ep_devices=Mock())
        record = dict(
            name="MIGraphXExecutionProvider", library_path=str(Path("fixture.dll").resolve())
        )
        with (
            patch.object(ctypes, "WinDLL", return_value="cpp-runtime", create=True),
            patch.object(windows_ml.os, "add_dll_directory", return_value=handle, create=True),
        ):
            handles = windows_ml.register_provider(ort, record)
            self.assertEqual(handles, ("cpp-runtime", handle))
            handle.close.assert_not_called()
            ort.register_execution_provider_library.side_effect = RuntimeError(
                "registration failed"
            )
            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                windows_ml.register_provider(ort, record)
        handle.close.assert_called_once_with()

    def test_amdgpu_bridge_gate_prevents_any_download_and_preserves_directml(self):
        with patch.object(windows_ml.subprocess, "run") as execute:
            with self.assertRaisesRegex(RuntimeError, "2.5.96"):
                windows_ml.install_provider("AMDGPUExecutionProvider", "GPU")
        execute.assert_not_called()
        self.assertTrue(windows_ml.bridge_compatible("MIGraphXExecutionProvider"))


if __name__ == "__main__":
    unittest.main()
