"""Focused contracts for the ComfyUI v0.46.3 integration update."""

import argparse
import ast
import enum
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.backends.comfy.comfy_args import (
    COMFY_DESTINATIONS,
    PerformanceFeature,
    add_comfy_arguments,
)
from ruinedfooocus.backends.comfy import comfy_compat
from ruinedfooocus.backends.comfy.image_compute import apply_compute
from ruinedfooocus.backends.comfy.native_step import end_native_step
from ruinedfooocus.backends.comfy.runtime_info import runtime_details
from ruinedfooocus.backends.comfy.sensenova_clip import load_sensenova_clip
from ruinedfooocus.features.generation.native_sampling import SCHEDULERS, scheduler_choices
from ruinedfooocus.features.models.model_identity import file_signature


ROOT = Path(__file__).resolve().parents[1]
COMFY_SOURCE = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))
COMFY_V0463_PIN = "cab5f4e13b83c58482cd514afb22ca925e117638"


class ComfyV0463PinAndParserTests(unittest.TestCase):
    def test_bootstrap_pins_comfyui_to_v0463(self):
        tree = ast.parse((ROOT / "ruinedfooocus/app/bootstrap.py").read_text(encoding="utf-8"))
        repos = ast.literal_eval(
            next(
                node.value
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "git_repos"
                    for target in node.targets
                )
            )
        )
        comfy_entry = next(row for row in repos if row["name"] == "ComfyUI")
        self.assertEqual(comfy_entry["hash"], COMFY_V0463_PIN)

    def test_checked_out_comfyui_matches_v0463_version_and_requirements(self):
        version_file = COMFY_SOURCE / "comfyui_version.py"
        if not version_file.is_file():
            self.skipTest("repositories/ComfyUI not checked out in this environment")
        version_tree = ast.parse(version_file.read_text(encoding="utf-8"))
        self.assertEqual(ast.literal_eval(version_tree.body[0].value), "0.46.3")
        reqs = (COMFY_SOURCE / "requirements.txt").read_text(encoding="utf-8").splitlines()
        self.assertIn("comfyui-frontend-package==1.59.6", reqs)
        self.assertIn("comfyui-workflow-templates==0.11.118", reqs)
        self.assertIn("comfyui-embedded-docs==0.5.17", reqs)
        self.assertIn("comfy-aimdo==0.6.0", reqs)
        self.assertIn("comfy-kitchen==0.2.41", reqs)

    def test_comfy_destinations_match_v0463_cli_args_without_directml(self):
        self.assertNotIn("directml", COMFY_DESTINATIONS)
        parser = argparse.ArgumentParser()
        add_comfy_arguments(parser)
        # Legacy --directml remains accepted by RuinedFooocus's parser for startup compatibility.
        parsed = parser.parse_args(["--directml"])
        self.assertEqual(parsed.directml, -1)

        cli_args_path = COMFY_SOURCE / "comfy/cli_args.py"
        if not cli_args_path.is_file():
            self.skipTest("repositories/ComfyUI not checked out in this environment")
        comfy = types.ModuleType("comfy")
        options = types.ModuleType("comfy.options")
        options.args_parsing = False
        comfy.options = options
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.options": options}):
            upstream = runpy.run_path(str(cli_args_path))
        backend = upstream["parser"]
        expected = {a.dest: a for a in backend._actions}
        self.assertNotIn("directml", expected)

        def normalized(value):
            return value.value if isinstance(value, enum.Enum) else value

        for action in parser._actions:
            if action.dest in ("help", "normalvram", "directml"):
                continue
            with self.subTest(dest=action.dest):
                self.assertIn(action.dest, COMFY_DESTINATIONS)
                other = expected[action.dest]
                for field in ("option_strings", "default", "const", "nargs", "required"):
                    self.assertEqual(
                        normalized(getattr(action, field)), normalized(getattr(other, field))
                    )
                if action.type is not None and issubclass(action.type, enum.Enum):
                    self.assertEqual(
                        [v.value for v in action.type], [v.value for v in other.type]
                    )
                else:
                    self.assertEqual(action.type, other.type)
                self.assertEqual(type(action), type(other))


class DirectMLRemovalAndCudaWarningTests(unittest.TestCase):
    def _memory_modules(self, *, has_directml=False):
        names = (
            "comfy",
            "comfy.cli_args",
            "comfy.model_management",
            "comfy.memory_management",
            "comfy.model_patcher",
            "comfy_aimdo",
            "comfy_aimdo.control",
        )
        modules = {name: types.ModuleType(name) for name in names}
        for name, mod in modules.items():
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(modules[parent], child, mod)
        cli_attrs = dict(
            reserve_vram=None,
            disable_nvml_pressure=False,
            vram_headroom=0.0,
            cpu=False,
            enable_dynamic_vram=False,
            disable_dynamic_vram=False,
            highvram=False,
            gpu_only=False,
            novram=False,
        )
        if has_directml:
            cli_attrs["directml"] = None
        cli_args = types.SimpleNamespace(**cli_attrs)
        modules["comfy.cli_args"].args = cli_args
        modules["comfy.cli_args"].enables_dynamic_vram = lambda: not cli_args.cpu
        modules["comfy_aimdo.control"].init = Mock(return_value=True)
        modules["comfy_aimdo.control"].init_devices = Mock(return_value=True)
        mgmt = modules["comfy.model_management"]
        mgmt.is_nvidia = Mock(return_value=True)
        mgmt.is_amd = Mock(return_value=False)
        mgmt.rocm_version = (0, 0)
        mgmt.torch_version_numeric = (2, 10)
        mgmt.get_all_torch_devices = Mock(return_value=[types.SimpleNamespace(index=0)])
        mgmt.cuda_version_warning = Mock()
        modules["comfy.memory_management"].aimdo_enabled = False
        modules["comfy.model_patcher"].CoreModelPatcher = object()
        modules["comfy.model_patcher"].ModelPatcherDynamic = object()
        return modules

    def test_initialize_comfy_memory_handles_missing_directml_attr_in_v046(self):
        modules = self._memory_modules(has_directml=False)
        self.assertFalse(hasattr(modules["comfy.cli_args"].args, "directml"))
        with patch.dict(sys.modules, modules):
            self.assertTrue(comfy_compat.initialize_comfy_memory())
            self.assertTrue(modules["comfy.memory_management"].aimdo_enabled)

    def test_initialize_backend_falls_back_to_cpu_when_directml_removed_in_v046(self):
        modules = self._memory_modules(has_directml=False)
        launch_args = types.SimpleNamespace(cpu=False, directml=0, offline=True)
        support = types.SimpleNamespace(
            inspect_installed_rocm=lambda p: None,
            inspect_installed_cuda=lambda p: None,
            select_torch_platform=lambda *a: "directml",
        )
        with (
            patch.dict(
                sys.modules,
                {
                    **modules,
                    "torchruntime": types.SimpleNamespace(),
                    "ruinedfooocus.packaging.runtime_support": support,
                },
            ),
            patch.object(comfy_compat, "_runtime_platform", None),
            patch.object(comfy_compat, "warn_cpu_fallback") as warn_fallback,
            patch.object(comfy_compat, "configure_comfy_runtime"),
            patch.object(comfy_compat, "configure_torch_allocator"),
            patch.object(comfy_compat, "configure_torch_compatibility"),
            patch.object(comfy_compat, "initialize_comfy_memory"),
            patch.object(comfy_compat, "unavailable_accelerator", return_value=None),
            patch(
                "ruinedfooocus.backends.comfy.cpu_startup.initialize_cpu_model_management"
            ) as init_cpu,
        ):
            platform_result = comfy_compat.initialize_backend(launch_args)
        self.assertEqual(platform_result[0], "cpu")
        self.assertTrue(modules["comfy.cli_args"].args.cpu)
        warn_fallback.assert_called_once()
        self.assertIn("DirectML", warn_fallback.call_args.args[1])
        init_cpu.assert_called_once()

    def test_initialize_backend_calls_cuda_version_warning_on_cuda(self):
        modules = self._memory_modules(has_directml=False)
        launch_args = types.SimpleNamespace(cpu=False, directml=None, offline=True)
        support = types.SimpleNamespace(
            inspect_installed_rocm=lambda p: None,
            inspect_installed_cuda=lambda p: None,
            select_torch_platform=lambda *a: "cu128",
        )
        with (
            patch.dict(
                sys.modules,
                {
                    **modules,
                    "torchruntime": types.SimpleNamespace(),
                    "ruinedfooocus.packaging.runtime_support": support,
                },
            ),
            patch.object(comfy_compat, "_runtime_platform", None),
            patch.object(comfy_compat, "configure_comfy_runtime"),
            patch.object(comfy_compat, "configure_torch_allocator"),
            patch.object(comfy_compat, "configure_torch_compatibility"),
            patch.object(comfy_compat, "initialize_comfy_memory"),
            patch.object(comfy_compat, "unavailable_accelerator", return_value=None),
        ):
            comfy_compat.initialize_backend(launch_args)
        modules["comfy.model_management"].cuda_version_warning.assert_called_once_with()

    def test_runtime_details_handles_missing_directml_attr(self):
        fake_torch = types.SimpleNamespace(
            __version__="2.10.0+cu130",
            version=types.SimpleNamespace(hip=None, cuda="13.0"),
        )
        device = types.SimpleNamespace(type="cuda")
        v046_args = types.SimpleNamespace(
            preview_method="auto",
            disable_async_offload=False,
            async_offload=None,
            cpu_vae=False,
            fast=set(),
        )
        memory = types.SimpleNamespace(aimdo_enabled=True)
        rows = dict(runtime_details(fake_torch, device, "RTX Test", v046_args, memory))
        self.assertEqual(rows["Active image backend"], "CUDA")
        self.assertEqual(rows["Dynamic VRAM active"], "Yes")


class NativeStepAndSenseNovaV046Tests(unittest.TestCase):
    def test_end_native_step_runs_upstream_cleanup_when_aimdo_enabled(self):
        modules = {
            "comfy": types.ModuleType("comfy"),
            "comfy.memory_management": types.ModuleType("comfy.memory_management"),
            "comfy.model_management": types.ModuleType("comfy.model_management"),
            "comfy.model_prefetch": types.ModuleType("comfy.model_prefetch"),
            "comfy_aimdo": types.ModuleType("comfy_aimdo"),
            "comfy_aimdo.model_vbar": types.ModuleType("comfy_aimdo.model_vbar"),
        }
        modules["comfy.memory_management"].aimdo_enabled = True
        modules["comfy.model_prefetch"].cleanup_prefetch_queues = Mock()
        modules["comfy.model_management"].reset_cast_buffers = Mock()
        modules["comfy_aimdo.model_vbar"].vbars_reset_watermark_limits = Mock()
        with patch.dict(sys.modules, modules):
            self.assertTrue(end_native_step())
        modules["comfy.model_prefetch"].cleanup_prefetch_queues.assert_called_once()
        modules["comfy.model_management"].reset_cast_buffers.assert_called_once()
        modules["comfy_aimdo.model_vbar"].vbars_reset_watermark_limits.assert_called_once()

    def test_sensenova_clip_reports_clear_error_when_v046_clip_target_is_none(self):
        class SenseNovaU15:
            def clip_target(self, state_dict={}):
                return None

        comfy_mod = types.SimpleNamespace(
            sd=types.SimpleNamespace(CLIP=Mock()),
            utils=types.SimpleNamespace(calculate_parameters=lambda sd: 0),
            model_management=types.SimpleNamespace(throw_exception_if_processing_interrupted=Mock()),
        )
        deps = {
            "comfy": comfy_mod,
            "comfy.sd": comfy_mod.sd,
            "comfy.utils": comfy_mod.utils,
            "comfy.model_management": comfy_mod.model_management,
        }
        with tempfile.TemporaryDirectory() as folder:
            ckpt = Path(folder) / "sensenova.safetensors"
            ckpt.write_bytes(b"fixture")
            with patch.dict(sys.modules, deps):
                with self.assertRaisesRegex(ValueError, "ComfyUI v0.46"):
                    load_sensenova_clip(SenseNovaU15(), file_signature(ckpt))
        comfy_mod.sd.CLIP.assert_not_called()

    def test_compute_preferences_and_native_schedulers_remain_wired(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--preview-method", default="auto")
        add_comfy_arguments(parser)
        args = parser.parse_args([])
        apply_compute(
            args,
            {
                "image_fast_mode": "FP16 accumulation",
                "image_attention": "Flash",
                "image_dynamic_vram": "Enabled",
                "image_async_offload": "2",
                "image_fast_disk": "Enabled",
                "image_compiler": "Disabled",
                "image_cuda_graphs": "Disabled",
            },
        )
        self.assertEqual(args.fast, [PerformanceFeature.Fp16Accumulation])
        self.assertTrue(args.use_flash_attention)
        self.assertTrue(args.enable_dynamic_vram)
        self.assertEqual(args.async_offload, 2)
        self.assertTrue(args.fast_disk)
        self.assertTrue(args.disable_comfy_compiler)
        self.assertTrue(args.disable_cuda_graphs)

        choices = scheduler_choices(["normal", "karras", "simple"])
        for native_sched in SCHEDULERS:
            self.assertIn(native_sched, choices)


if __name__ == "__main__":
    unittest.main()
