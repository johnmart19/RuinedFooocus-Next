"""CPU-only installer plans; no downloads or installation."""

import unittest
from unittest.mock import patch
import os
import ruinedfooocus.packaging.rocm_installer as rocm_installer


class RocmPackagePlanTests(unittest.TestCase):
    def test_clean_wsl_gets_target_from_host_tool_without_claiming_guest_visibility(self):
        from types import SimpleNamespace

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(rocm_installer.platform, "release", return_value="6.18-microsoft-WSL2"),
            patch.object(
                rocm_installer.shutil,
                "which",
                side_effect=[None, "/mnt/c/Windows/System32/clinfo.exe"],
            ),
            patch.object(
                rocm_installer.subprocess,
                "run",
                return_value=SimpleNamespace(
                    returncode=0, stdout="Device Type: CL_DEVICE_TYPE_GPU\nName: gfx1201\n"
                ),
            ) as execute,
        ):
            self.assertEqual(rocm_installer.detect_rocminfo_targets(), ["gfx1201"])
            self.assertEqual(execute.call_args.args[0], ["/mnt/c/Windows/System32/clinfo.exe"])

    def test_each_published_target_selects_its_own_extras_and_release_bundle(self):
        for system, targets in rocm_installer.SUPPORTED_TARGETS.items():
            for target in targets:
                with (
                    self.subTest(system=system, target=target),
                    patch.dict(os.environ, {}, clear=True),
                    patch.object(rocm_installer, "detect_rocm_targets", return_value=[target]),
                    patch.object(rocm_installer.platform, "machine", return_value="AMD64"),
                ):
                    plan = rocm_installer.automatic_rocm_plan(system, "cpu", force=True)
                    bundle = rocm_installer.bundle_for(plan.platform)
                    self.assertEqual(plan.args[0], f"torch[device-{target}]=={bundle.torch}")
                    self.assertEqual(plan.args[1], f"torchvision[device-{target}]=={bundle.vision}")
                    self.assertFalse(any("torchaudio" in arg for arg in plan.args))

    def test_metadata_selector_build_is_hash_pinned_without_installing_dependencies(self):
        import hashlib
        from pathlib import Path
        import tempfile

        payload = b"fixture metadata wheel"

        def build(command, **kwargs):
            self.assertIn("wheel", command)
            self.assertIn("--no-deps", command)
            self.assertIn("--no-build-isolation", command)
            self.assertIn("--no-binary=rocm", command)
            self.assertIn("--only-binary=:all:", command)
            self.assertEqual(
                command[-1],
                rocm_installer.ROCM_INDEX
                + "rocm/rocm-10.1.0.tar.gz#sha256="
                + rocm_installer._ROCM_METADATA_HASHES["10.1.0"],
            )
            folder = Path(command[command.index("--wheel-dir") + 1])
            (folder / "rocm-10.1.0-py3-none-any.whl").write_bytes(payload)

        with (
            tempfile.TemporaryDirectory() as folder,
            patch("ruinedfooocus.core.paths.data_path", return_value=Path(folder)),
            patch.object(rocm_installer.subprocess, "run", side_effect=build) as execute,
        ):
            result = rocm_installer.rocm_metadata_layer("10.1.0")
            wheel = (
                Path(folder) / hashlib.sha256(payload).hexdigest() / "rocm-10.1.0-py3-none-any.whl"
            )
            self.assertEqual(result.requirements, (f"rocm @ {wheel.as_uri()}",))
            self.assertEqual(result.digest, hashlib.sha256(payload).hexdigest())
            self.assertEqual(wheel.read_bytes(), payload)
            execute.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, "not pinned"):
                rocm_installer.rocm_metadata_layer("99.0.0")
            execute.assert_called_once()

    def test_old_amd_adapter_does_not_hide_a_supported_discrete_target(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(rocm_installer, "detect_rocm_targets", return_value=["gfx803", "gfx1100"]),
            patch.object(rocm_installer.platform, "machine", return_value="AMD64"),
        ):
            plan = rocm_installer.automatic_rocm_plan("Windows", "cpu", force=True)
        self.assertIn("device-gfx1100", plan.args[0])
        self.assertNotIn("device-gfx803", plan.args[0])

    def test_uncached_detection_helper_offline_never_installs_or_queries_devices(self):
        from pathlib import Path
        import tempfile

        with (
            tempfile.TemporaryDirectory() as folder,
            patch("ruinedfooocus.core.paths.data_path", return_value=Path(folder) / "helper"),
            patch(
                "ruinedfooocus.packaging.startup_options.skip_dependency_install", return_value=True
            ),
            patch.object(rocm_installer.subprocess, "run") as execute,
            patch.object(rocm_installer.venv.EnvBuilder, "create") as create,
        ):
            with self.assertRaisesRegex(RuntimeError, "not cached"):
                rocm_installer.isolated_gfx_detection()
            execute.assert_not_called()
            create.assert_not_called()

    def test_released_targets_use_only_platform_appropriate_device_extras(self):
        for target, os_platform in (
            ("gfx1153", "Windows"),
            ("gfx1153", "Linux"),
            ("gfx908", "Linux"),
            ("gfx90a", "Linux"),
            ("gfx950", "Linux"),
            ("gfx1201", "Windows"),
        ):
            with (
                self.subTest(target=target, platform=os_platform),
                patch.dict(os.environ, {}, clear=True),
                patch.object(rocm_installer, "detect_rocm_targets", return_value=[target]),
                patch.object(rocm_installer.platform, "machine", return_value="AMD64"),
            ):
                plan = rocm_installer.automatic_rocm_plan(os_platform, "cpu", force=True)
                self.assertIn(f"[device-{target}]", plan.args[0])
                self.assertIn(f"[device-{target}]", plan.args[1])
        for target, os_platform in (("gfx950", "Windows"), ("gfx1250", "Linux")):
            with (
                self.subTest(target=target, platform=os_platform),
                patch.dict(os.environ, {}, clear=True),
                patch.object(rocm_installer, "detect_rocm_targets", return_value=[target]),
                self.assertRaisesRegex(RuntimeError, "not validated"),
            ):
                rocm_installer.automatic_rocm_plan(os_platform, "cpu", force=True)

    def test_release_bundle_uses_validated_stable_versions_and_target_extra(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(rocm_installer, "detect_rocm_targets", return_value=["gfx1100"]),
            patch.object(rocm_installer.platform, "machine", return_value="AMD64"),
        ):
            plan = rocm_installer.automatic_rocm_plan(
                "Windows", rocm_installer.ROCM_PLATFORM, force=True
            )
        self.assertEqual(
            plan.args[:2],
            [
                "torch[device-gfx1100]==2.14.0+rocm10.1.0",
                "torchvision[device-gfx1100]==0.29.0a0+rocm10.1.0",
            ],
        )
        self.assertIn(rocm_installer.ROCM_INDEX, plan.args)

    def test_install_validates_native_vision_and_gpu_before_returning(self):
        from ruinedfooocus.packaging.package_plan import PackagePlan

        plan = PackagePlan("target", (), (), "environment", "rocm10.1")
        with (
            patch.object(rocm_installer.subprocess, "run") as run,
            patch("ruinedfooocus.packaging.package_plan.execute_plan") as execute,
        ):
            rocm_installer.install_rocm(plan)
        execute.assert_called_once_with(plan, plan.digest, force=False)
        self.assertEqual(run.call_count, 1)
        smoke = run.call_args_list[-1].args[0][-1]
        self.assertIn("import torch, torchvision;", smoke)
        self.assertNotIn("torchaudio", smoke)
        self.assertIn("torchvision.ops.nms", smoke)

    def test_raw_rocm_arguments_cannot_bypass_combined_package_resolution(self):
        with patch.object(rocm_installer.subprocess, "run") as execute:
            with self.assertRaisesRegex(RuntimeError, "frozen combined PackagePlan"):
                rocm_installer.install_rocm(["torch"])
            execute.assert_not_called()

    def test_explicit_other_backend_does_not_detect_or_install_amd(self):
        with (
            patch.dict(os.environ, {"TORCH_PLATFORM": "cu128"}),
            patch.object(rocm_installer, "detect_rocm_targets") as detect,
        ):
            self.assertIsNone(rocm_installer.automatic_rocm_plan("Windows", "cu128"))
            detect.assert_not_called()

    def test_unsupported_gpu_keeps_automatic_selection_but_explicit_request_fails(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(rocm_installer, "detect_rocm_targets", return_value=["gfx803"]),
            patch(
                "torchruntime.device_db.get_gpus",
                return_value=[type("Gpu", (), {"vendor_id": "1002"})()],
            ),
        ):
            self.assertIsNone(rocm_installer.automatic_rocm_plan("Windows", "directml"))
            self.assertIsNone(rocm_installer.automatic_rocm_plan("Linux", "rocm6.2"))
            with self.assertRaises(RuntimeError):
                rocm_installer.automatic_rocm_plan("Windows", "directml", force=True)
        with (
            patch.dict(os.environ, {"TORCH_PLATFORM": rocm_installer.ROCM_PLATFORM}),
            patch.object(rocm_installer, "detect_rocm_targets", return_value=["gfx803"]),
        ):
            with self.assertRaises(RuntimeError):
                rocm_installer.automatic_rocm_plan("Windows", rocm_installer.ROCM_PLATFORM)
