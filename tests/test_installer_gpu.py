"""CPU-only installer plans; no downloads or installation."""

import unittest
import json
import os
from pathlib import Path
import tempfile
import hashlib
import base64
from types import SimpleNamespace
from unittest.mock import patch
import ruinedfooocus.packaging.gpu_installer as gpu_installer


class TorchPackagePlanTests(unittest.TestCase):
    def test_payload_damage_is_detected_even_when_record_metadata_is_unchanged(self):
        class File:
            def __init__(self, path, name, digest=None):
                self.path, self.name = path, name
                self.hash = SimpleNamespace(mode="sha256", value=digest) if digest else None

            def locate(self):
                return self.path

            def __str__(self):
                return self.name

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary, record = root / "backend.dll", root / "RECORD"
            binary.write_bytes(b"original")
            record.write_text("same RECORD bytes")
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(binary.read_bytes()).digest())
                .rstrip(b"=")
                .decode()
            )
            distribution = SimpleNamespace(
                version="1",
                files=[
                    File(binary, "backend.dll", digest),
                    File(record, "xllamacpp-1.dist-info/RECORD"),
                ],
            )
            with patch.object(
                gpu_installer.importlib.metadata, "distribution", return_value=distribution
            ):
                _, original = gpu_installer._xllamacpp_installation()
                binary.write_bytes(b"damaged")
                with self.assertRaisesRegex(OSError, "integrity mismatch"):
                    gpu_installer._xllamacpp_installation()
                self.assertEqual(
                    original["record_sha256"], hashlib.sha256(record.read_bytes()).hexdigest()
                )

    def test_old_backend_markers_do_not_leave_ghost_package_metadata_after_upgrade(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stale = root / "xllamacpp-1.dist-info"
            current = root / "xllamacpp-2.dist-info"
            for directory in (stale, current):
                directory.mkdir()
                (directory / "ruinedfooocus-backend.json").write_text("{}", encoding="utf-8")
            metadata = current / "METADATA"
            original = b"Name: xllamacpp\nVersion: 2\n"
            metadata.write_bytes(original)
            with patch.object(gpu_installer.sysconfig, "get_path", return_value=folder):
                gpu_installer.cleanup_xllamacpp_markers()
            self.assertFalse(stale.exists())
            self.assertEqual(metadata.read_bytes(), original)
            self.assertFalse((current / "ruinedfooocus-backend.json").exists())

    def test_cpu_plan_uses_the_verified_patch_bundle(self):
        with patch.object(gpu_installer.sys, "version_info", (3, 12)):
            commands = gpu_installer.torch_install_commands(None, "cpu", "Windows")
        self.assertEqual(commands[0][:2], ["torch==2.15.0", "torchvision==0.29.2"])
        self.assertEqual(commands[0][-2:], ["--index-url", "https://download.pytorch.org/whl/cpu"])
        with patch.object(gpu_installer.sys, "version_info", (3, 10)):
            legacy = gpu_installer.torch_install_commands(None, "cpu", "Windows")
        self.assertEqual(legacy[0][:2], ["torch==2.14.1", "torchvision==0.29.1"])

    def test_cuda_channels_require_only_image_torch_companions_without_execution(self):
        with (
            patch(
                "ruinedfooocus.packaging.cuda_selection.cuda_wheels_available", return_value=True
            ),
            patch(
                "ruinedfooocus.packaging.cuda_selection.nvidia_devices",
                return_value=((8.6,), (13, 4)),
            ),
            patch.object(gpu_installer.subprocess, "run") as execute,
        ):
            for channel in ("cu126", "cu130", "cu132", "cu134"):
                with self.subTest(channel=channel):
                    commands = gpu_installer.torch_install_commands(None, channel, "Windows")
                    self.assertEqual(
                        commands[0][:2],
                        [
                            f"torch=={'2.15.0' if channel in ('cu132', 'cu134') else '2.14.1'}+{channel}",
                            f"torchvision=={'0.29.2' if channel in ('cu132', 'cu134') else '0.29.1'}+{channel}",
                        ],
                    )
                    self.assertEqual(len(commands), 1)
                    self.assertNotIn("--pre", commands[0])
                    self.assertNotIn("nightly", " ".join(commands[0]))
                    self.assertFalse(any("torchaudio" in value for value in commands[0]))
            execute.assert_not_called()

    def test_cpu_xpu_and_nightly_plans_do_not_add_audio_packages(self):
        with (
            patch(
                "ruinedfooocus.packaging.cuda_selection.cuda_wheels_available", return_value=True
            ),
            patch("ruinedfooocus.packaging.cuda_selection.validate_cuda_runtime"),
            patch.object(gpu_installer.subprocess, "run") as execute,
        ):
            for runtime, nightly in (("cpu", False), ("xpu", False), ("cu132", True)):
                with self.subTest(runtime=runtime):
                    commands = gpu_installer.torch_install_commands(
                        None, runtime, "Windows", nightly
                    )
                    self.assertFalse(
                        any("torchaudio" in value for command in commands for value in command)
                    )
            execute.assert_not_called()

    def test_other_gpu_runtimes_use_vulkan_llm_wheels(self):
        for torch_platform, os_platform in [
            ("xpu", "Windows"),
            ("rocm6.2", "Linux"),
            ("cu118", "Linux"),
        ]:
            with self.subTest(torch_platform=torch_platform):
                self.assertTrue(
                    gpu_installer.xllamacpp_index(torch_platform, os_platform).endswith("/vulkan")
                )
        self.assertTrue(gpu_installer.xllamacpp_index("cpu", "Linux").endswith("/vulkan"))
        self.assertEqual(gpu_installer.xllamacpp_index("mps", "Darwin"), "https://pypi.org/simple")

    def test_runtime_selection_uses_the_same_named_profile_as_the_app_without_writing(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            from ruinedfooocus.core.paths import settings_path

            self.assertEqual(gpu_installer.selected_llm_runtime(), "llama.cpp")
            path = settings_path("separate-profile")
            path.parent.mkdir(parents=True)
            original = json.dumps({"llm_runtime": "xllamacpp", "prompt": "keep me"}).encode()
            path.write_bytes(original)
            self.assertEqual(gpu_installer.selected_llm_runtime("separate-profile"), "xllamacpp")
            self.assertEqual(gpu_installer.selected_llm_runtime(), "llama.cpp")
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(settings_path().exists())

    def test_managed_packages_never_trigger_optional_runtime_installation_or_probes(self):
        with (
            patch(
                "ruinedfooocus.packaging.startup_options.skip_dependency_install", return_value=True
            ),
            patch(
                "ruinedfooocus.packaging.launch_util.is_installed", return_value=True
            ) as installed,
            patch.object(gpu_installer, "cached_xllamacpp_backend_matches", return_value=True),
        ):
            gpu_installer.ensure_xllamacpp("rocm10.1", "Windows")
            installed.assert_called_once_with("xllamacpp==2026.10.11870")
        with (
            patch(
                "ruinedfooocus.packaging.startup_options.skip_dependency_install", return_value=True
            ),
            patch("ruinedfooocus.packaging.launch_util.is_installed", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "select native llama.cpp"):
                gpu_installer.ensure_xllamacpp("cpu", "Windows")

    def test_offline_backend_mismatch_fails_without_network_or_device_probe(self):
        with (
            patch(
                "ruinedfooocus.packaging.startup_options.skip_dependency_install", return_value=True
            ),
            patch("ruinedfooocus.packaging.launch_util.is_installed", return_value=True),
            patch.object(gpu_installer, "cached_xllamacpp_backend_matches", return_value=False),
            patch(
                "ruinedfooocus.packaging.cuda_selection.wheel_available",
                side_effect=AssertionError("Network"),
            ),
            patch.object(
                gpu_installer, "has_xllamacpp_device", side_effect=AssertionError("Device probe")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "backend/interpreter identity"):
                gpu_installer.ensure_xllamacpp("cu132", "Windows")

    def test_saved_cpu_accepts_tracked_gpu_wheel_and_rejects_other_interpreter(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "marker.json"
            identity = {"version": "1", "record_sha256": "a", "target": "exact-abi"}
            marker.write_text(
                json.dumps(identity | {"index": "https://xorbitsai.github.io/xllamacpp/whl/vulkan"})
            )
            with patch.object(
                gpu_installer, "_xllamacpp_installation", return_value=(marker, identity)
            ):
                self.assertTrue(
                    gpu_installer.cached_xllamacpp_backend_matches(
                        "https://pypi.org/simple", cpu=True
                    )
                )
                self.assertFalse(
                    gpu_installer.cached_xllamacpp_backend_matches("https://pypi.org/simple")
                )
                marker.write_text(
                    json.dumps(
                        identity | {"target": "wrong-abi", "index": "https://pypi.org/simple"}
                    )
                )
                self.assertFalse(
                    gpu_installer.cached_xllamacpp_backend_matches(
                        "https://pypi.org/simple", cpu=True
                    )
                )
