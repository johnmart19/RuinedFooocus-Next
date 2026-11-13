"""Run the actual pinned allocator policy with package/device metadata fixtures."""

import argparse
import ctypes
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.backends.comfy.comfy_args import add_comfy_arguments, PerformanceFeature
from ruinedfooocus.backends.comfy.comfy_compat import configure_torch_allocator
from ruinedfooocus.backends.comfy.image_compute import apply_compute, explicit_options


class NativeAllocatorTests(unittest.TestCase):
    def native(self, version, *, device="NVIDIA GeForce RTX 4090", fast=False):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "version.py").write_text(f"__version__ = {version!r}\n")
            original_spec = importlib.util.find_spec

            def find_spec(name, *args):
                return (
                    SimpleNamespace(submodule_search_locations=[folder])
                    if name == "torch"
                    else original_spec(name, *args)
                )

            cli = ModuleType("comfy.cli_args")
            cli.args = SimpleNamespace(
                cuda_malloc=False,
                disable_cuda_malloc=False,
                fast={PerformanceFeature.AutoTune} if fast else set(),
            )
            cli.PerformanceFeature = PerformanceFeature
            comfy = ModuleType("comfy")
            comfy.cli_args = cli

            def displays(_, index, pointer, flags):
                if index:
                    return False
                pointer._obj.DeviceString = device.encode()
                return True

            user32 = SimpleNamespace(EnumDisplayDevicesA=displays)
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.dict(sys.modules, {"comfy": comfy, "comfy.cli_args": cli}),
                patch.object(importlib.util, "find_spec", side_effect=find_spec),
                patch.object(ctypes, "windll", SimpleNamespace(user32=user32), create=True),
                patch(
                    "subprocess.check_output",
                    return_value=f"GPU 0: {device} (UUID: fixture)".encode(),
                ),
            ):
                configure_torch_allocator("cu132", "Windows")
                self.assertNotIn("PYTORCH_ALLOC_CONF", os.environ)  # Bootstrap defers.
                configure_torch_allocator("cu132", "Windows", native=True)
                return os.environ["PYTORCH_ALLOC_CONF"], cli.args.cuda_malloc

    def test_native_cuda_versions_blacklist_and_autotune_policy_are_reused(self):
        self.assertEqual(self.native("2.15.0+cu132"), ("backend:cudaMallocAsync", True))
        self.assertEqual(self.native("2.14.1+cu126"), ("backend:cudaMallocAsync", True))
        self.assertEqual(
            self.native("2.14.1+cu126", device="NVIDIA GeForce GTX 1650"),
            ("expandable_segments:True", False),
        )
        self.assertEqual(
            self.native("2.15.0+cu132", fast=True), ("expandable_segments:True", False)
        )

    def test_user_allocator_configuration_is_not_replaced_or_probed(self):
        for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            with (
                patch.dict(os.environ, {name: "backend:native,max_split_size_mb:64"}, clear=True),
                patch(
                    "runpy.run_path",
                    side_effect=AssertionError("Do not inspect native policy for explicit config"),
                ),
            ):
                configure_torch_allocator("cu132", "Windows", native=True)
                self.assertEqual(dict(os.environ), {name: "backend:native,max_split_size_mb:64"})

    def test_cpu_fallback_ignores_saved_allocator_but_preserves_explicit_cli(self):
        parser = argparse.ArgumentParser()
        add_comfy_arguments(parser)
        for argv, saved, expected in (
            ([], "Async", "expandable_segments:True"),
            ([], "Native", "expandable_segments:True"),
            (["--cuda-malloc"], "Native", "backend:cudaMallocAsync,expandable_segments:False"),
            (["--disable-cuda-malloc"], "Async", "backend:native,expandable_segments:False"),
        ):
            with self.subTest(argv=argv, saved=saved), patch.dict(os.environ, {}, clear=True):
                args = parser.parse_args(argv)
                apply_compute(args, {"image_allocator": saved}, explicit_options(parser, argv))
                configure_torch_allocator("cpu", "Windows", args)
                self.assertEqual(os.environ["PYTORCH_ALLOC_CONF"], expected)
                self.assertEqual(args.cuda_malloc, "--cuda-malloc" in argv)
                self.assertEqual(args.disable_cuda_malloc, "--disable-cuda-malloc" in argv)

    def test_explicit_cli_wins_and_wsl_rocm_protection_is_preserved(self):
        parser = argparse.ArgumentParser()
        add_comfy_arguments(parser)
        for flag, backend in (
            ("--cuda-malloc", "cudaMallocAsync"),
            ("--disable-cuda-malloc", "native"),
        ):
            args = parser.parse_args([flag])
            with patch.dict(
                os.environ,
                {
                    "PYTORCH_ALLOC_CONF": "backend:native,max_split_size_mb:64,expandable_segments:True"
                },
                clear=True,
            ):
                configure_torch_allocator("cu132", "Windows", args, native=True)
                self.assertEqual(
                    os.environ["PYTORCH_ALLOC_CONF"],
                    f"max_split_size_mb:64,backend:{backend},expandable_segments:False",
                )
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("platform.release", return_value="6.6-microsoft-WSL2"),
        ):
            configure_torch_allocator("rocm10.1", "Linux", native=True)
            self.assertEqual(os.environ["PYTORCH_ALLOC_CONF"], "expandable_segments:False")


if __name__ == "__main__":
    unittest.main()
