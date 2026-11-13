import os
import argparse
import enum
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.backends.comfy.comfy_compat import (
    configure_device_environment,
    initialize_comfy_memory,
)
import ruinedfooocus.backends.comfy.comfy_compat as comfy_compat
from ruinedfooocus.backends.comfy.comfy_args import add_comfy_arguments, COMFY_DESTINATIONS


class ComputeArgumentContractTests(unittest.TestCase):
    def test_independent_child_restores_only_proven_startup_masks(self):
        report = {"requested": {"CUDA_VISIBLE_DEVICES": "1"}, "inherited": {}}
        args = types.SimpleNamespace(_device_environment_status=report)
        environment = {"CUDA_VISIBLE_DEVICES": "1", "GGML_VK_VISIBLE_DEVICES": "0"}
        with patch.dict(
            sys.modules, {"ruinedfooocus.app.arguments": types.SimpleNamespace(args=args)}
        ):
            child = comfy_compat.child_device_environment(environment, service="fixture")
            self.assertNotIn("CUDA_VISIBLE_DEVICES", child)
            self.assertEqual(child["GGML_VK_VISIBLE_DEVICES"], "0")
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "1")
            report["inherited"]["CUDA_VISIBLE_DEVICES"] = "GPU-original"
            self.assertEqual(
                comfy_compat.child_device_environment(environment, service="fixture")[
                    "CUDA_VISIBLE_DEVICES"
                ],
                "GPU-original",
            )
            with self.assertRaisesRegex(RuntimeError, "changed after native startup"):
                comfy_compat.child_device_environment(
                    {"CUDA_VISIBLE_DEVICES": "2"}, service="fixture"
                )

    def test_compute_parser_matches_checked_out_backend(self):
        """Check every exposed compute action without importing Torch/backend code."""
        comfy = types.ModuleType("comfy")
        options = types.ModuleType("comfy.options")
        options.args_parsing = False
        comfy.options = options
        source = Path(__file__).resolve().parents[1] / "repositories/ComfyUI/comfy/cli_args.py"
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.options": options}):
            upstream = runpy.run_path(str(source))
        local = argparse.ArgumentParser()
        add_comfy_arguments(local)
        backend = upstream["parser"]
        expected = {a.dest: a for a in backend._actions}

        def normalized(value):
            if isinstance(value, enum.Enum):
                return value.value
            return value

        for action in local._actions:
            if action.dest in ("help", "normalvram"):
                continue
            with self.subTest(dest=action.dest):
                other = expected[action.dest]
                self.assertIn(action.dest, COMFY_DESTINATIONS)
                for field in ("option_strings", "default", "const", "nargs", "required"):
                    self.assertEqual(
                        normalized(getattr(action, field)), normalized(getattr(other, field))
                    )
                if action.type is not None and issubclass(action.type, enum.Enum):
                    self.assertEqual([v.value for v in action.type], [v.value for v in other.type])
                else:
                    self.assertEqual(action.type, other.type)
                self.assertEqual(type(action), type(other))
        destinations = set(COMFY_DESTINATIONS)

        def groups(parser):
            return {
                frozenset(a.dest for a in group._group_actions if a.dest in destinations)
                for group in parser._mutually_exclusive_groups
            } - {frozenset()}

        self.assertEqual(groups(local), groups(backend))

    def test_high_ram_and_compiler_implications_reach_backend(self):
        parser = argparse.ArgumentParser()
        add_comfy_arguments(parser)
        args = parser.parse_args(["--high-ram", "--disable-comfy-compiler", "--fast", "cublas_ops"])
        args.offline = False
        backend = types.SimpleNamespace()
        cli = types.ModuleType("comfy.cli_args")
        from ruinedfooocus.backends.comfy.comfy_args import PerformanceFeature

        cli.PerformanceFeature = PerformanceFeature
        cli.LatentPreviewMethod = types.SimpleNamespace(from_string=lambda value: value)
        with patch.dict(sys.modules, {"comfy.cli_args": cli}):
            comfy_compat.configure_comfy_runtime(args, backend)
        self.assertTrue(backend.high_ram)
        self.assertTrue(backend.cache_classic)
        self.assertTrue(backend.disable_cuda_graphs)
        self.assertEqual(backend.fast, {PerformanceFeature.CublasOps})


def device_args(**overrides):
    return types.SimpleNamespace(
        **(
            dict(
                cuda_device=None,
                default_device=0,
                gpu_device_id=None,
                oneapi_device_selector=None,
                deterministic=False,
            )
            | overrides
        )
    )


class DeviceSelectionTests(unittest.TestCase):
    def setUp(self):
        selected = patch.object(comfy_compat, "_native_mask_backend", return_value="cuda")
        selected.start()
        self.addCleanup(selected.stop)

    def test_cpu_selection_never_enumerates_or_rewrites_other_service_visibility(self):
        inherited = {"HIP_VISIBLE_DEVICES": "1", "CUDA_VISIBLE_DEVICES": "GPU-other"}
        with (
            patch.dict(os.environ, inherited, clear=True),
            patch.object(comfy_compat, "_physical_device_count") as probe,
        ):
            configure_device_environment(device_args(cpu=True, default_device=0))
            self.assertEqual(dict(os.environ), inherited)
            probe.assert_not_called()

    def test_invalid_memory_controls_fail_before_device_probe_or_environment_changes(self):
        for name in ("reserve_vram", "vram_headroom", "async_offload"):
            for value in (-1, float("nan"), float("inf")):
                with (
                    self.subTest(name=name, value=value),
                    patch.dict(os.environ, {}, clear=True),
                    patch.object(comfy_compat, "_physical_device_count") as probe,
                ):
                    with self.assertRaisesRegex(ValueError, name.replace("_", "-")):
                        configure_device_environment(device_args(**{name: value}))
                    probe.assert_not_called()
                    self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)
                    self.assertNotIn("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", os.environ)

    def test_zero_and_positive_memory_controls_are_valid(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_device_environment(
                device_args(default_device=None, reserve_vram=0, vram_headroom=4.0, async_offload=0)
            )

    def test_backend_startup_defaults_and_user_overrides(self):
        for existing in (
            {},
            {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "0", "MIMALLOC_PURGE_DELAY": "10"},
        ):
            with (
                patch.dict(os.environ, existing, clear=True),
                patch.object(comfy_compat.platform, "system", return_value="Windows"),
            ):
                configure_device_environment(device_args(default_device=None))
                self.assertEqual(
                    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"],
                    existing.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1"),
                )
                self.assertEqual(
                    os.environ["MIMALLOC_PURGE_DELAY"], existing.get("MIMALLOC_PURGE_DELAY", "0")
                )

    def test_single_nvidia_default_does_not_create_phantom_devices(self):
        from ruinedfooocus.packaging.cuda_selection import nvidia_devices

        responses = [
            '{"count":1,"backend":"cuda"}',
            "CUDA Version: 13.2",
            "GPU-test, 00000000:01:00.0, 8.6",
        ]
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "ruinedfooocus.packaging.cuda_selection.subprocess.check_output",
                side_effect=responses,
            ),
        ):
            configure_device_environment(device_args())
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(nvidia_devices(), ((8.6,), (13, 2)))

    def test_multi_gpu_reorders_only_real_devices(self):
        with (
            patch.dict(os.environ, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, clear=True),
            patch.object(comfy_compat, "_physical_device_count", return_value=3),
        ):
            configure_device_environment(device_args(default_device=1))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1,0,2")

    def test_ambiguous_nvidia_order_rejected_and_environment_restored(self):
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-old"}, clear=True),
            patch(
                "ruinedfooocus.backends.comfy.comfy_compat.subprocess.check_output",
                return_value='{"count":2,"backend":"cuda"}',
            ),
        ):
            with self.assertRaisesRegex(ValueError, "PCI_BUS_ID"):
                configure_device_environment(device_args())
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-old")

    def test_amd_count_probed_without_visibility(self):
        def probe(*args, **kwargs):
            self.assertEqual(os.environ["HIP_VISIBLE_DEVICES"], "1")
            self.assertNotIn("HIP_VISIBLE_DEVICES", kwargs["env"])
            self.assertNotIn("GPU_DEVICE_ORDINAL", kwargs["env"])
            self.assertEqual(kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"], "2")
            return '{"count":2,"backend":"hip"}'

        with (
            patch.dict(
                os.environ,
                {
                    "HIP_VISIBLE_DEVICES": "1",
                    "GPU_DEVICE_ORDINAL": "1",
                    "ASCEND_RT_VISIBLE_DEVICES": "2",
                },
                clear=True,
            ),
            patch(
                "ruinedfooocus.backends.comfy.comfy_compat.subprocess.check_output",
                side_effect=probe,
            ),
        ):
            self.assertEqual(comfy_compat._physical_device_count("hip"), 2)
            self.assertEqual(os.environ["HIP_VISIBLE_DEVICES"], "1")

    def test_cuda_count_preserves_other_vendors_in_child(self):
        with (
            patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "1"}, clear=True
            ),
            patch.object(
                comfy_compat.subprocess, "check_output", return_value='{"count":1,"backend":"cuda"}'
            ) as probe,
        ):
            self.assertEqual(comfy_compat._physical_device_count("cuda"), 1)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", probe.call_args.kwargs["env"])
            self.assertEqual(probe.call_args.kwargs["env"]["HIP_VISIBLE_DEVICES"], "1")

    def test_inherited_hip_ordinal_conflict_is_not_silently_overridden(self):
        with (
            patch.dict(os.environ, {"GPU_DEVICE_ORDINAL": "1"}, clear=True),
            patch.object(comfy_compat, "_native_mask_backend", return_value="hip"),
        ):
            with self.assertRaisesRegex(ValueError, "GPU_DEVICE_ORDINAL"):
                configure_device_environment(device_args(cuda_device="0"))
            self.assertEqual(dict(os.environ), {"GPU_DEVICE_ORDINAL": "1"})

    def test_unavailable_index_does_not_change_visibility(self):
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True),
            patch(
                "ruinedfooocus.backends.comfy.comfy_compat._physical_device_count", return_value=1
            ),
        ):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                configure_device_environment(device_args(default_device=1))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")

    def test_explicit_selection_skips_default_probe(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("ruinedfooocus.backends.comfy.comfy_compat._physical_device_count") as probe,
        ):
            configure_device_environment(device_args(cuda_device="2,0"))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,0")
            with self.assertRaisesRegex(ValueError, "mask conflict"):
                configure_device_environment(device_args(gpu_device_id=1))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,0")
            self.assertNotIn("HIP_VISIBLE_DEVICES", os.environ)
            self.assertNotIn("ASCEND_RT_VISIBLE_DEVICES", os.environ)
            probe.assert_not_called()

    def test_hip_selection_preserves_other_engine_masks_and_reports_identity(self):
        inherited = {"CUDA_VISIBLE_DEVICES": "GPU-other", "GGML_VK_VISIBLE_DEVICES": "1"}
        args = device_args(cuda_device="0", default_device=None)
        with (
            patch.dict(os.environ, inherited, clear=True),
            patch.object(comfy_compat, "_native_mask_backend", return_value="hip"),
        ):
            configure_device_environment(args)
            self.assertEqual(os.environ["HIP_VISIBLE_DEVICES"], "0")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-other")
            self.assertEqual(os.environ["GGML_VK_VISIBLE_DEVICES"], "1")
            self.assertEqual(args._device_environment_status["inherited"], inherited)
            self.assertEqual(
                args._device_environment_status["requested"], {"HIP_VISIBLE_DEVICES": "0"}
            )

    def test_inherited_rocr_conflict_and_initialized_runtime_refuse_all_changes(self):
        with (
            patch.dict(os.environ, {"ROCR_VISIBLE_DEVICES": "1"}, clear=True),
            patch.object(comfy_compat, "_native_mask_backend", return_value="hip"),
        ):
            with self.assertRaisesRegex(ValueError, "ROCR_VISIBLE_DEVICES"):
                configure_device_environment(device_args(cuda_device="0"))
            self.assertEqual(dict(os.environ), {"ROCR_VISIBLE_DEVICES": "1"})
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(comfy_compat, "_runtime_platform", ("cuda", "Windows")),
            patch.object(comfy_compat, "_physical_device_count") as count,
        ):
            with self.assertRaisesRegex(RuntimeError, "restart"):
                configure_device_environment(device_args(cuda_device="0"))
            count.assert_not_called()
            self.assertEqual(dict(os.environ), {})


class DynamicMemoryTests(unittest.TestCase):
    def setUp(self):
        names = (
            "comfy",
            "comfy.cli_args",
            "comfy.model_management",
            "comfy.memory_management",
            "comfy.model_patcher",
            "comfy_aimdo",
            "comfy_aimdo.control",
        )
        self.modules = {name: types.ModuleType(name) for name in names}
        for name, module in self.modules.items():
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(self.modules[parent], child, module)
        self.args = types.SimpleNamespace(
            reserve_vram=1.5,
            disable_nvml_pressure=False,
            vram_headroom=0.5,
            directml=None,
            cpu=False,
            enable_dynamic_vram=False,
            disable_dynamic_vram=False,
            highvram=False,
            gpu_only=False,
            novram=False,
        )
        cli = self.modules["comfy.cli_args"]
        cli.args = self.args
        cli.enables_dynamic_vram = lambda: (
            self.args.enable_dynamic_vram
            or not any(
                getattr(self.args, key)
                for key in ("disable_dynamic_vram", "highvram", "gpu_only", "novram", "cpu")
            )
        )
        self.control = self.modules["comfy_aimdo.control"]
        self.control.init = Mock()
        self.control.init_devices = Mock(return_value=True)
        self.management = self.modules["comfy.model_management"]
        self.management.is_nvidia = Mock(return_value=True)
        self.management.is_amd = Mock(return_value=False)
        self.management.rocm_version = (7, 15)
        self.management.torch_version_numeric = (2, 14)
        self.management.get_all_torch_devices = Mock(return_value=[types.SimpleNamespace(index=0)])
        self.memory = self.modules["comfy.memory_management"]
        self.memory.aimdo_enabled = False
        self.patcher = self.modules["comfy.model_patcher"]
        self.legacy = object()
        self.patcher.CoreModelPatcher = self.legacy
        self.patcher.ModelPatcherDynamic = object()
        self.scope = patch.dict(sys.modules, self.modules)
        self.scope.start()
        self.addCleanup(self.scope.stop)

    def test_forced_dynamic_never_initializes_controller_on_cpu_or_directml(self):
        self.args.enable_dynamic_vram = True
        for cpu, directml in ((True, None), (False, -1), (False, 0)):
            self.args.cpu, self.args.directml = cpu, directml
            self.assertFalse(initialize_comfy_memory())
        self.control.init.assert_not_called()

    def test_controller_failure_never_selects_dynamic_patcher(self):
        self.control.init.return_value = False
        with self.assertLogs(level="WARNING") as logs:
            self.assertFalse(initialize_comfy_memory())
        self.assertIn("DynamicVRAM controller unavailable", logs.output[0])
        self.control.init_devices.assert_not_called()
        self.assertIs(self.patcher.CoreModelPatcher, self.legacy)

    def test_backend_initializes_once_before_shared_state(self):
        order = []
        support = types.SimpleNamespace(
            inspect_installed_rocm=lambda platform: None,
            inspect_installed_cuda=lambda platform: None,
            select_torch_platform=lambda *args: "rocm10.1",
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(
                sys.modules,
                {
                    "torchruntime": types.SimpleNamespace(),
                    "ruinedfooocus.packaging.runtime_support": support,
                },
            ),
            patch.object(comfy_compat, "_runtime_platform", None),
            patch.object(comfy_compat, "unavailable_accelerator", return_value=None),
            patch.object(
                comfy_compat,
                "configure_torch_allocator",
                side_effect=lambda *args, **kwargs: order.append("allocator"),
            ),
            patch.object(
                comfy_compat,
                "configure_torch_compatibility",
                side_effect=lambda: order.append("compatibility"),
            ),
            patch.object(
                comfy_compat,
                "configure_comfy_runtime",
                side_effect=lambda *args: order.append("flags"),
            ),
            patch.object(
                comfy_compat, "initialize_comfy_memory", side_effect=lambda: order.append("memory")
            ),
        ):
            first = comfy_compat.initialize_backend(self.args)
            self.assertEqual(comfy_compat.initialize_backend(self.args), first)
            self.assertEqual(os.environ["OCL_SET_SVM_SIZE"], "4194304")
        self.assertEqual(order, ["flags", "allocator", "compatibility", "memory"])

    def test_nvidia_activation_applies_headroom_and_patcher(self):
        self.assertTrue(initialize_comfy_memory())
        self.control.init.assert_called_once_with(
            simple_vram_headroom=int(1.5 * 1024**3), nvml_pressure=True
        )
        self.control.init_devices.assert_called_once_with([(0, 512 * 1024**2)])
        self.assertTrue(self.memory.aimdo_enabled)
        self.assertIs(self.patcher.CoreModelPatcher, self.patcher.ModelPatcherDynamic)

    def test_modern_rocm_activates(self):
        self.management.is_nvidia.return_value = False
        self.management.is_amd.return_value = True
        self.assertTrue(initialize_comfy_memory())

    def test_unsupported_hardware_and_old_torch_skip_even_when_forced(self):
        for kind in ("old_rocm", "directml", "cpu", "old_torch", "other"):
            with self.subTest(kind=kind):
                self.args.enable_dynamic_vram = True
                self.args.cpu = kind == "cpu"
                self.args.directml = 0 if kind == "directml" else None
                self.management.is_nvidia.return_value = kind in ("directml", "cpu", "old_torch")
                self.management.is_amd.return_value = kind == "old_rocm"
                self.management.rocm_version = (7, 2)
                self.management.torch_version_numeric = (2, 6) if kind == "old_torch" else (2, 14)
                if kind in ("directml", "cpu"):
                    self.assertFalse(initialize_comfy_memory())
                else:
                    with self.assertLogs(level="WARNING") as logs:
                        self.assertFalse(initialize_comfy_memory())
                    self.assertIn("DynamicVRAM requires", logs.output[0])
                self.assertFalse(self.memory.aimdo_enabled)
                self.assertIs(self.patcher.CoreModelPatcher, self.legacy)
        self.control.init_devices.assert_not_called()

    def test_disabled_memory_modes_skip_initialization(self):
        for key in ("disable_dynamic_vram", "highvram", "gpu_only", "novram", "cpu"):
            with self.subTest(key=key):
                setattr(self.args, key, True)
                self.assertFalse(initialize_comfy_memory())
                setattr(self.args, key, False)
        self.control.init.assert_not_called()

    def test_activation_sets_controller_log_level_like_native_startup(self):
        self.modules["comfy.cli_args"].get_console_log_level = lambda outputs: (
            outputs[0][0] if outputs else "INFO"
        )
        for verbose, setter in (
            ([], "set_log_info"),
            ([("WARNING", None)], "set_log_warning"),
            ([("DETAIL", None)], "set_log_info"),
        ):  # DETAIL falls back when unsupported
            with self.subTest(verbose=verbose):
                for name in ("set_log_info", "set_log_warning"):
                    setattr(self.control, name, Mock())
                self.args.verbose = verbose
                self.assertTrue(initialize_comfy_memory())
                getattr(self.control, setter).assert_called_once_with()

    def test_device_failure_retains_legacy(self):
        for outcome in (False, RuntimeError("driver unavailable")):
            with self.subTest(outcome=outcome):
                if isinstance(outcome, Exception):
                    self.control.init_devices.side_effect = outcome
                else:
                    self.control.init_devices.return_value = outcome
                with self.assertLogs(level="WARNING") as logs:
                    self.assertFalse(initialize_comfy_memory())
                self.assertIn(
                    "driver unavailable"
                    if isinstance(outcome, Exception)
                    else "controller rejected PyTorch device indices 0",
                    logs.output[0],
                )
                self.assertFalse(self.memory.aimdo_enabled)
                self.assertIs(self.patcher.CoreModelPatcher, self.legacy)

    def test_no_visible_device_reports_selection_without_calling_controller(self):
        self.management.get_all_torch_devices.return_value = []
        with self.assertLogs(level="WARNING") as logs:
            self.assertFalse(initialize_comfy_memory())
        self.assertIn("no visible devices in PyTorch", logs.output[0])
        self.control.init_devices.assert_not_called()


class RocmAttentionTests(unittest.TestCase):
    def test_wide_heads_use_math_only_on_affected_rocm_devices(self):
        import torch
        import torch.nn.functional as functional
        from torch.nn.attention import SDPBackend

        backends, calls = [], []

        def original(*args, **kwargs):  # A plain function: Mock attributes would look installed.
            calls.append(kwargs)
            backends.append(
                torch.backends.cuda.math_sdp_enabled()
                and not torch.backends.cuda.flash_sdp_enabled()
            )

        wide = types.SimpleNamespace(shape=(1, 16, 77, 512), is_cuda=True, device=0)
        other = types.SimpleNamespace(shape=(1, 16, 77, 512), is_cuda=True, device=1)
        unknown = types.SimpleNamespace(shape=(1, 16, 77, 512), is_cuda=True, device=2)
        narrow = types.SimpleNamespace(shape=(1, 16, 77, 128), is_cuda=True, device=0)
        cpu = types.SimpleNamespace(shape=(1, 16, 77, 512), is_cuda=False, device="cpu")
        with patch.object(functional, "scaled_dot_product_attention", original):
            with patch.object(torch.version, "hip", None):
                self.assertFalse(comfy_compat.configure_rocm_attention())
                self.assertIs(functional.scaled_dot_product_attention, original)
            with (
                patch.object(torch.version, "hip", "7.16"),
                patch.object(
                    torch.cuda,
                    "get_device_properties",
                    side_effect=lambda device: types.SimpleNamespace(
                        gcnArchName={0: "gfx1100:sramecc-:xnack-", 1: "gfx1201", 2: ""}[device]
                    ),
                ) as properties,
            ):
                self.assertTrue(comfy_compat.configure_rocm_attention())
                self.assertFalse(comfy_compat.configure_rocm_attention())  # Installed once.
                for query in (wide, narrow, other, cpu, unknown, wide):
                    functional.scaled_dot_product_attention(query, query, query, scale=1.0)
                self.assertEqual(properties.call_count, 3)  # Cached per device; no CPU probe.
        self.assertEqual(backends, [True, False, False, False, True, True])
        self.assertEqual(calls, [{"scale": 1.0}] * 6)
        self.assertEqual(SDPBackend.MATH.name, "MATH")


class AimdoLogShutdownTests(unittest.TestCase):
    def setUp(self):
        import ctypes

        self.callback_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)
        self.callback = self.callback_type(lambda level, message: None)
        self.library = Mock()
        self.control = types.SimpleNamespace(
            lib=self.library,
            _LOG_CALLBACK=self.callback_type,
            _log_callback=self.callback,
            devctxs=[123],
        )
        self.callbacks = []
        self.state = patch.object(comfy_compat, "_aimdo_log_detach_registered", False)
        self.state.start()
        self.addCleanup(self.state.stop)
        self.registration = patch.object(
            comfy_compat.atexit, "register", side_effect=self.callbacks.append
        )
        self.registration.start()
        self.addCleanup(self.registration.stop)

    def test_keeps_callback_alive_and_detaches_only_logging_at_exit(self):
        import gc
        import weakref

        callback_ref = weakref.ref(self.callback)
        self.assertTrue(comfy_compat.register_aimdo_log_shutdown(self.control, "Windows", True))
        self.library.set_log_callback.assert_not_called()
        self.control.lib = None
        self.control._log_callback = None
        self.callback = None
        gc.collect()
        self.assertIsNotNone(callback_ref())
        self.callbacks[0]()
        self.library.set_log_callback.assert_called_once()
        self.assertFalse(bool(self.library.set_log_callback.call_args.args[0]))
        self.library.cleanup.assert_not_called()
        self.library.plat_cleanup.assert_not_called()
        self.assertEqual(self.control.devctxs, [123])

    def test_registers_once(self):
        self.assertTrue(comfy_compat.register_aimdo_log_shutdown(self.control, "Windows", True))
        self.assertFalse(comfy_compat.register_aimdo_log_shutdown(self.control, "Windows", True))
        self.assertEqual(len(self.callbacks), 1)

    def test_other_platforms_and_uninitialized_abi_are_untouched(self):
        for system, amd in (("Linux", True), ("Windows", False), ("Darwin", True)):
            with self.subTest(system=system, amd=amd):
                self.assertFalse(
                    comfy_compat.register_aimdo_log_shutdown(self.control, system, amd)
                )
        self.assertFalse(
            comfy_compat.register_aimdo_log_shutdown(
                types.SimpleNamespace(lib=None), "Windows", True
            )
        )
        self.assertFalse(
            comfy_compat.register_aimdo_log_shutdown(
                types.SimpleNamespace(lib=self.library), "Windows", True
            )
        )
        self.assertEqual(self.callbacks, [])

    def test_activation_registers_only_after_success(self):
        # Inspect call ordering through the existing isolated controller fixture.
        fixture = DynamicMemoryTests(methodName="test_modern_rocm_activates")
        fixture.setUp()
        try:
            fixture.management.is_nvidia.return_value = False
            fixture.management.is_amd.return_value = True
            order = []
            fixture.control.init_devices.side_effect = lambda devices: (
                order.append("devices") or True
            )
            with patch.object(
                comfy_compat,
                "register_aimdo_log_shutdown",
                side_effect=lambda *args: order.append("shutdown"),
            ):
                self.assertTrue(initialize_comfy_memory())
            self.assertEqual(order, ["devices", "shutdown"])
            order.clear()
            fixture.control.init_devices.side_effect = lambda devices: (
                order.append("devices") or False
            )
            with (
                patch.object(comfy_compat, "register_aimdo_log_shutdown") as register,
                self.assertLogs(level="WARNING"),
            ):
                self.assertFalse(initialize_comfy_memory())
            register.assert_not_called()
        finally:
            fixture.doCleanups()


if __name__ == "__main__":
    unittest.main()
