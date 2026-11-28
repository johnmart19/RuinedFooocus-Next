"""CPU-only installer plans; no downloads or installation."""

import unittest
from unittest.mock import patch
import os
import sys
import ruinedfooocus.backends.llama.llama_installer as llama_installer


class NativePackagePlanTests(unittest.TestCase):
    def test_offline_native_sdk_acquisition_does_not_build_or_download_selector(self):
        from pathlib import Path
        import tempfile

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(llama_installer, "skip_dependency_install", return_value=True),
            patch("ruinedfooocus.packaging.rocm_installer.rocm_metadata_layer") as selector,
        ):
            with self.assertRaisesRegex(RuntimeError, "restart online"):
                llama_installer.ensure_rocm_libraries(Path(folder) / "sdk", "gfx1100")
            selector.assert_not_called()

    def test_invalid_active_sdk_pointer_shapes_preserve_legacy_directory(self):
        import json
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "rocm-10.0"
            base.mkdir()
            pointer = base.with_name(base.name + ".active.json")
            self.assertEqual(llama_installer.active_rocm_directory(base), base)
            for value in (
                "not-json",
                "[]",
                "null",
                "false",
                "123",
                json.dumps("path"),
                "{}",
                json.dumps({"directory": None}),
                json.dumps({"directory": []}),
                json.dumps({"directory": {}}),
                json.dumps({"directory": 123}),
            ):
                with self.subTest(pointer=value):
                    pointer.write_text(value, encoding="utf-8")
                    self.assertEqual(llama_installer.active_rocm_directory(base), base)
                    self.assertEqual(pointer.read_text(encoding="utf-8"), value)
            active = base.with_name(base.name + "-published")
            active.mkdir()
            pointer.write_text(json.dumps({"directory": str(active)}), encoding="utf-8")
            self.assertEqual(llama_installer.active_rocm_directory(base), active.resolve())

    def test_sdk_reinstall_stages_new_environment_and_failed_stage_preserves_pointer(self):
        import json
        from pathlib import Path
        import tempfile
        from dataclasses import replace
        from ruinedfooocus.packaging.package_plan import PackagePlan
        from ruinedfooocus.packaging.package_plan import PackageLayer

        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "rocm-10.0"
            base.mkdir()
            pointer = base.parent / (base.name + ".active.json")
            pointer.write_text(json.dumps({"directory": str(base)}))
            before = pointer.read_bytes()
            plan = PackagePlan("target", (), (), str(base), "native-rocm10.0")

            def planned(*args, **kwargs):
                return replace(plan, environment=str(kwargs["environment"]))

            with (
                patch.object(llama_installer, "skip_dependency_install", return_value=False),
                patch(
                    "ruinedfooocus.packaging.rocm_installer.rocm_metadata_layer",
                    return_value=PackageLayer(
                        "rocm-selector",
                        ("rocm @ file:///tmp/rocm-10.0.0-py3-none-any.whl",),
                        "https://stable.repo.amd.com/rocm/whl-next/",
                        "10.0.0",
                        "MIT",
                        "fixture-digest",
                    ),
                ) as selector,
                patch(
                    "ruinedfooocus.packaging.package_plan.resolve_packages", return_value=()
                ) as resolve,
                patch("ruinedfooocus.packaging.package_plan.plan_packages", side_effect=planned),
                patch(
                    "ruinedfooocus.packaging.package_plan.execute_plan",
                    side_effect=RuntimeError("stage failed"),
                ) as execute,
                patch.object(llama_installer.venv.EnvBuilder, "create"),
            ):
                with self.assertRaisesRegex(RuntimeError, "stage failed"):
                    llama_installer.ensure_rocm_libraries(base, "gfx1100", reinstall=True)
            selector.assert_called_once_with("10.0.0")
            layer = resolve.call_args.args[0][0]
            self.assertEqual(layer.identity, "native-llama-rocm")
            self.assertEqual(
                layer.requirements,
                ("rocm[libraries,device-gfx1100] @ file:///tmp/rocm-10.0.0-py3-none-any.whl",),
            )
            self.assertEqual(pointer.read_bytes(), before)
            self.assertNotEqual(Path(execute.call_args.args[0].environment), base)

    def test_native_rocm_target_does_not_depend_on_image_torch(self):
        with (
            patch.object(llama_installer.platform, "system", return_value="Windows"),
            patch(
                "ruinedfooocus.packaging.rocm_installer.detect_rocm_targets",
                return_value=["gfx1100"],
            ),
        ):
            self.assertEqual(llama_installer.rocm_target(True, windows=True), "gfx1100")

    def test_linux_auto_prefers_vulkan_without_importing_image_torch(self):
        self.assertIn(
            "ubuntu-vulkan-", llama_installer.package_name("Linux", "x86_64", True, rocm=True)
        )
        with (
            patch.object(llama_installer.platform, "system", return_value="Linux"),
            patch.object(
                llama_installer, "rocm_target", side_effect=AssertionError("Torch queried")
            ),
        ):
            self.assertEqual(
                llama_installer.runtime_environment(True, backend="Auto"), os.environ.copy()
            )

    def test_cached_native_identity_detects_tampering_offline(self):
        import json
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            binary = directory / (
                "llama-server.exe"
                if llama_installer.platform.system() == "Windows"
                else "llama-server"
            )
            binary.write_bytes(b"fixture")
            marker = directory / "rf-runtime.json"
            marker.write_text(json.dumps(llama_installer._runtime_manifest(directory, "package")))
            self.assertEqual(
                llama_installer._install_server(directory, "package"), binary.resolve()
            )
            binary.write_bytes(b"tampered")
            with (
                patch.dict(os.environ, RF_OFFLINE="1"),
                self.assertRaisesRegex(RuntimeError, "identity/integrity"),
            ):
                llama_installer._install_server(directory, "package")

    def test_offline_legacy_cache_can_adopt_exact_verified_bytes(self):
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            executable = directory / (
                "llama-server.exe"
                if llama_installer.platform.system() == "Windows"
                else "llama-server"
            )
            executable.write_bytes(b"publisher fixture")
            trusted = llama_installer._runtime_manifest(directory, "package") | {
                "archives": [
                    {"name": "package", "sha256": "a" * 64, "url": "https://publisher.test/package"}
                ]
            }
            with (
                patch.dict(os.environ, RF_OFFLINE="1"),
                patch.object(
                    llama_installer.requests, "get", side_effect=AssertionError("network")
                ),
            ):
                llama_installer.adopt_verified_cache(directory, "package", trusted)
                self.assertEqual(
                    llama_installer._install_server(directory, "package"), executable.resolve()
                )
                executable.write_bytes(b"different")
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    llama_installer.adopt_verified_cache(directory, "package", trusted)

    def test_build_skip_policy_prevents_device_queries_and_cache_writes(self):
        from ruinedfooocus.backends.llama import llama_build

        for argv, offline in [([], "1"), (["--offline"], ""), (["--skip-dependency-install"], "")]:
            with (
                self.subTest(argv=argv, offline=offline),
                patch.object(sys, "argv", ["test", *argv]),
                patch.dict(os.environ, RF_OFFLINE=offline),
                patch.object(llama_build, "build_capabilities") as probe,
                patch.object(llama_build, "build_root") as location,
                patch.object(llama_build, "_run") as execute,
            ):
                with self.assertRaisesRegex(RuntimeError, "installation is disabled"):
                    next(llama_build.build_cuda_runtime())
                probe.assert_not_called()
                location.assert_not_called()
                execute.assert_not_called()

    def test_offline_missing_runtime_fails_before_network_access(self):
        from pathlib import Path
        import tempfile

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, RF_OFFLINE="1", LLAMA_SERVER=""),
            patch.object(llama_installer, "package_directory", return_value=Path(folder)),
            patch.object(
                llama_installer.requests, "get", side_effect=AssertionError("Network accessed")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Native llama.cpp is missing"):
                llama_installer.server_path(False, backend="CPU")

    def test_optional_local_build_requires_its_actual_toolkit_driver(self):
        from ruinedfooocus.backends.llama import llama_build

        for capabilities, driver, expected in [
            ((8.6,), (13, 4), []),
            ((8.6,), (13, 5), ["86"]),
            ((6.1,), (13, 5), []),
        ]:
            with (
                self.subTest(capabilities=capabilities, driver=driver),
                patch.object(llama_build.platform, "system", return_value="Linux"),
                patch.object(llama_build.platform, "machine", return_value="x86_64"),
                patch(
                    "ruinedfooocus.packaging.cuda_selection.nvidia_devices",
                    return_value=(capabilities, driver),
                ),
            ):
                self.assertEqual(llama_build.build_capabilities(), expected)

    def test_retired_cuda_selection_uses_current_assets_and_still_validates_driver(self):
        self.assertEqual(
            llama_installer.package_name("Windows", "AMD64", True, backend="CUDA 13.2"),
            f"llama-{llama_installer.VERSION}-bin-win-cuda-13.5-x64.zip",
        )
        with patch(
            "ruinedfooocus.packaging.cuda_selection.nvidia_devices", return_value=((8.6,), (13, 2))
        ):
            with self.assertRaisesRegex(RuntimeError, "newer NVIDIA driver"):
                llama_installer.validate_cuda_backend(
                    llama_installer.canonical_backend("CUDA 13.2")
                )

    def test_supported_packages_match_published_asset_names(self):
        cases = [
            ("Windows", "AMD64", False, "Auto", "win-cpu-x64.zip"),
            ("Windows", "AMD64", True, "Vulkan", "win-vulkan-x64.zip"),
            ("Windows", "AMD64", True, "ROCm", "win-rocm-10.0-x64.zip"),
            ("Windows", "AMD64", True, "CUDA 12.4", "win-cuda-12.4-x64.zip"),
            ("Windows", "AMD64", True, "CUDA 13.4", "win-cuda-13.5-x64.zip"),
            ("Windows", "ARM64", False, "CPU", "win-cpu-arm64.zip"),
            ("Windows", "ARM64", True, "Auto", "win-vulkan-arm64.zip"),
            ("Windows", "ARM64", True, "CUDA 13.4", "win-cuda-13.5-arm64.zip"),
            ("Linux", "x86_64", True, "CUDA 12.8", "ubuntu-cuda-12.8-x64.tar.gz"),
            ("Linux", "x86_64", True, "CUDA 13.4", "ubuntu-cuda-13.5-x64.tar.gz"),
            ("Linux", "aarch64", True, "CUDA 13.4", "ubuntu-cuda-13.5-arm64.tar.gz"),
            ("Linux", "x86_64", False, "CPU", "ubuntu-x64.tar.gz"),
            ("Linux", "x86_64", True, "Vulkan", "ubuntu-vulkan-x64.tar.gz"),
            ("Linux", "x86_64", True, "ROCm", "ubuntu-rocm-10.0-x64.tar.gz"),
            ("Linux", "aarch64", True, "Vulkan", "ubuntu-vulkan-arm64.tar.gz"),
            ("Darwin", "arm64", True, "Auto", "macos-arm64.tar.gz"),
            ("Darwin", "x86_64", False, "CPU", "macos-x64.tar.gz"),
        ]
        with patch("ruinedfooocus.backends.llama.llama_build.built_server", return_value=None):
            for system, machine, gpu, backend, suffix in cases:
                with self.subTest(system=system, machine=machine, backend=backend):
                    self.assertEqual(
                        llama_installer.package_name(system, machine, gpu, backend=backend),
                        f"llama-{llama_installer.VERSION}-bin-{suffix}",
                    )

    def test_explicit_cuda_requires_driver_and_architecture(self):
        for capabilities, driver in [((8.6,), (13, 3)), ((6.1,), (13, 4)), ((), None)]:
            with (
                self.subTest(capabilities=capabilities, driver=driver),
                patch(
                    "ruinedfooocus.packaging.cuda_selection.nvidia_devices",
                    return_value=(capabilities, driver),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    llama_installer.validate_cuda_backend("CUDA 13.4")
        with patch(
            "ruinedfooocus.packaging.cuda_selection.nvidia_devices", return_value=((8.6,), (13, 5))
        ):
            llama_installer.validate_cuda_backend("CUDA 13.4")

    def test_custom_server_bypasses_binary_download(self):
        from pathlib import Path

        executable = Path(__file__).resolve()
        with (
            patch.dict(os.environ, {"LLAMA_SERVER": str(executable)}),
            patch.object(llama_installer.requests, "get") as download,
        ):
            self.assertEqual(llama_installer.server_path(), executable)
            download.assert_not_called()

    def test_linux_cuda_runtime_uses_bundled_cuda_libraries(self):
        from pathlib import Path
        import tempfile

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, {"LD_LIBRARY_PATH": "/opt/lib"}),
            patch.object(llama_installer.platform, "system", return_value="Linux"),
            patch.object(llama_installer.platform, "machine", return_value="x86_64"),
            patch.object(llama_installer, "package_directory", return_value=Path(folder)),
        ):
            libraries = Path(folder) / "cudart-llama-bin-ubuntu-cuda-13.4-x64"
            libraries.mkdir()
            (libraries / "libcudart.so.13").write_bytes(b"")
            environment = llama_installer.runtime_environment(True, backend="CUDA 13.4")
        self.assertEqual(environment["LD_LIBRARY_PATH"], f"{libraries}:/opt/lib")
