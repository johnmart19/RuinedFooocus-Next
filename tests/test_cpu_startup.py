"""Explicit CPU startup must not import/probe GPU detection; no real backends."""

import argparse
import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

import ruinedfooocus.backends.comfy.comfy_compat as comfy_compat
from ruinedfooocus.backends.comfy.comfy_args import add_comfy_arguments, PerformanceFeature


class CpuStartupTests(unittest.TestCase):
    def initialize(self, arguments, selected="rocm10.1", forbid_detection=False, unavailable=None):
        parser = argparse.ArgumentParser()
        add_comfy_arguments(parser)
        args = parser.parse_args(arguments)
        args.offline = True
        args.preview_method = "none"
        comfy = types.ModuleType("comfy")
        cli = types.ModuleType("comfy.cli_args")
        cli.args = types.SimpleNamespace()
        cli.PerformanceFeature = PerformanceFeature
        cli.LatentPreviewMethod = types.SimpleNamespace(from_string=lambda name: name)
        comfy.cli_args = cli
        support = types.SimpleNamespace(
            inspect_installed_rocm=Mock(return_value=None),
            inspect_installed_cuda=Mock(return_value=None),
            select_torch_platform=Mock(return_value=selected),
        )
        runtime = None if forbid_detection else types.ModuleType("torchruntime")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(
                sys.modules,
                {
                    "comfy": comfy,
                    "comfy.cli_args": cli,
                    "ruinedfooocus.packaging.runtime_support": support,
                    "torchruntime": runtime,
                },
            ),
            patch.object(comfy_compat, "_runtime_platform", None),
            patch.object(comfy_compat.platform, "system", return_value="Windows"),
            patch.object(comfy_compat, "configure_torch_allocator") as allocator,
            patch.object(comfy_compat, "unavailable_accelerator", return_value=unavailable),
            patch.object(comfy_compat, "configure_torch_compatibility") as compatibility,
            patch(
                "ruinedfooocus.backends.comfy.cpu_startup.initialize_cpu_model_management"
            ) as native_cpu,
            patch.object(comfy_compat, "initialize_comfy_memory") as memory,
        ):
            result = comfy_compat.initialize_backend(args)
            self.assertEqual(comfy_compat.initialize_backend(args), result)
            allocator.assert_called_once_with(
                "cpu" if forbid_detection else selected, "Windows", args, native=True
            )
            compatibility.assert_called_once()
            memory.assert_called_once()
            self.assertEqual(native_cpu.call_count, int(result[0] == "cpu"))
        return result, cli.args, support

    def test_cpu_skips_gpu_detection_even_with_an_installed_gpu_provider(self):
        for provider in ("rocm10.1", "cu130", "directml"):
            with self.subTest(provider=provider):
                result, backend, support = self.initialize(
                    ["--cpu"], provider, forbid_detection=True
                )
                self.assertEqual(result, ("cpu", "Windows"))
                self.assertTrue(backend.cpu)
                self.assertIsNone(backend.directml)
                for function in vars(support).values():
                    function.assert_not_called()

    def test_automatic_cuda_rocm_cpu_and_directml_selection_stays_native(self):
        for provider in ("rocm10.1", "cu130", "directml"):
            with self.subTest(provider=provider):
                result, backend, support = self.initialize([], provider)
                self.assertEqual(result, (provider, "Windows"))
                self.assertEqual(backend.cpu, provider == "cpu")
                self.assertEqual(backend.directml, -1 if provider == "directml" else None)
                support.select_torch_platform.assert_called_once()

    def test_missing_gpu_and_unavailable_installed_gpu_report_cpu_fallback(self):
        for selected, reason in (("cpu", None), ("rocm10.1", "GPU masked by environment")):
            with self.subTest(selected=selected), self.assertLogs(level="WARNING") as logs:
                result, backend, _ = self.initialize([], selected, unavailable=reason)
                self.assertEqual(result, ("cpu", "Windows"))
                self.assertTrue(backend.cpu)
                self.assertIn(reason or "no compatible PyTorch GPU backend", logs.output[0])
                self.assertIn("Using CPU fallback", logs.output[0])

    def test_explicit_directml_device_is_retained(self):
        _, backend, _ = self.initialize(["--directml", "0"])
        self.assertFalse(backend.cpu)
        self.assertEqual(backend.directml, 0)


if __name__ == "__main__":
    unittest.main()
