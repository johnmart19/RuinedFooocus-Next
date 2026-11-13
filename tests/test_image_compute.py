"""Saved compute preferences must preserve native auto policy and CLI priority."""

import argparse
import json
import runpy
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.backends.comfy.comfy_args import add_comfy_arguments
from ruinedfooocus.backends.comfy.image_compute import (
    apply_compute,
    explicit_options,
    load_compute,
    validate_compute,
)
from ruinedfooocus.core.settings_validation import validate_settings


class ComputePreferencesTests(unittest.TestCase):
    def test_bootstrap_retains_image_only_cpu_provenance_and_explicit_cpu_priority(self):
        from ruinedfooocus.packaging import backend_preferences as preferences

        parser = self.parser()
        values = {"image_backend": "cpu", "cpu_everywhere": False}
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict("os.environ", {}, clear=True),
            patch(
                "ruinedfooocus.backends.comfy.image_compute.data_path",
                side_effect=lambda *parts: Path(folder).joinpath(*parts),
            ),
            patch(
                "ruinedfooocus.packaging.startup_options.saved_startup_policy",
                return_value=values,
            ),
            patch.object(preferences, "installed_torch_kind", return_value="rocm"),
        ):
            root = Path(folder) / "settings"
            root.mkdir()
            (root / "settings.json").write_text(json.dumps(values))
            for argv in ([], ["--cpu"]):
                with self.subTest(argv=argv):
                    args = parser.parse_args(argv)
                    load_compute(args, parser, argv)
                    preferences.apply_saved_image_backend(args, "Windows", reinstall=False)
                    self.assertTrue(args.cpu)
                    self.assertEqual(preferences.force_cpu_for_auxiliary(args), bool(argv))
                    self.assertEqual(args._compute_status["backend"]["action"], "none")

    def test_advanced_preferences_preserve_cli_groups_and_cpu_dtype_policy(self):
        parser = self.parser()
        values = dict(
            image_unet_precision="FP16",
            image_text_precision="BF16",
            image_vae_precision="FP16",
            image_attention="Split",
            image_attention_upcast="Disabled",
            image_compiler="Disabled",
            image_cuda_graphs="Disabled",
        )
        argv = [
            "--fp8_e4m3fn-unet",
            "--fp32-text-enc",
            "--cpu-vae",
            "--use-pytorch-cross-attention",
            "--force-upcast-attention",
        ]
        args = parser.parse_args(argv)
        apply_compute(args, values, explicit_options(parser, argv))
        self.assertFalse(args.fp16_unet)
        self.assertFalse(args.bf16_text_enc)
        self.assertFalse(args.fp16_vae)
        self.assertFalse(args.use_split_cross_attention)
        self.assertFalse(args.dont_upcast_attention)
        self.assertTrue(args.disable_comfy_compiler)
        self.assertTrue(args.disable_cuda_graphs)
        args = parser.parse_args(["--cpu"])
        apply_compute(args, values, {"--cpu"})
        self.assertFalse(args.fp16_unet)
        self.assertFalse(args.bf16_text_enc)
        self.assertFalse(args.fp16_vae)
        self.assertTrue(args.use_split_cross_attention)

    def parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--settings")
        parser.add_argument("--preview-method", default="auto")
        add_comfy_arguments(parser)
        return parser

    def test_blank_and_auto_preserve_native_device_policy(self):
        parser = self.parser()
        args = parser.parse_args([])
        before = vars(args).copy()
        apply_compute(
            args,
            dict(
                image_dynamic_vram="Auto",
                image_async_offload="Auto",
                image_fast_disk="Auto",
                image_vae_device="Auto",
                image_reserve_vram=None,
                image_vram_headroom="",
                image_preview_method="auto",
            ),
        )
        self.assertEqual(vars(args), before)

    def test_cli_zero_abbreviations_and_opposing_flags_win(self):
        parser = self.parser()
        argv = [
            "--vram-he=0",
            "--reserve-vram",
            "0",
            "--disable-dynamic-vram",
            "--disable-async-offload",
            "--disable-fast-disk",
            "--preview-method=none",
        ]
        args = parser.parse_args(argv)
        before = vars(args).copy()
        apply_compute(
            args,
            dict(
                image_vram_headroom=2,
                image_reserve_vram=1,
                image_dynamic_vram="Enabled",
                image_async_offload="4",
                image_fast_disk="Enabled",
                image_preview_method="taesd",
            ),
            explicit_options(parser, argv),
        )
        self.assertEqual(vars(args), before)

    def test_preferences_reach_native_options_without_forcing_gpu_only(self):
        args = self.parser().parse_args([])
        apply_compute(
            args,
            dict(
                image_vram_headroom=2,
                image_reserve_vram=1,
                image_dynamic_vram="Disabled",
                image_async_offload="2",
                image_fast_disk="Enabled",
                image_vae_device="CPU",
                image_preview_method="latent2rgb",
            ),
        )
        self.assertEqual((args.vram_headroom, args.reserve_vram, args.async_offload), (2, 1, 2))
        self.assertTrue(args.disable_dynamic_vram)
        self.assertFalse(args.enable_dynamic_vram)
        self.assertTrue(args.fast_disk)
        self.assertTrue(args.cpu_vae)
        self.assertFalse(args.gpu_only)
        self.assertEqual(args.preview_method, "latent2rgb")

    def test_explicit_device_and_memory_modes_cannot_be_overridden(self):
        parser = self.parser()
        for flag in (
            "--cpu",
            "--directml",
            "--gpu-only",
            "--highvram",
            "--lowvram",
            "--novram",
            "--normalvram",
        ):
            args = parser.parse_args([flag])
            apply_compute(
                args,
                {"image_dynamic_vram": "Enabled", "image_vae_device": "CPU"},
                explicit_options(parser, [flag]),
            )
            self.assertFalse(args.enable_dynamic_vram)
            if flag == "--gpu-only":
                self.assertFalse(args.cpu_vae)

    def test_invalid_values_fail_before_settings_are_saved(self):
        for field in ("image_vram_headroom", "image_reserve_vram"):
            for invalid in (-1, True, float("nan"), float("inf"), "bad"):
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    validate_settings({field: invalid})
        for field in ("image_dynamic_vram", "image_async_offload", "image_preview_method"):
            with self.assertRaises(ValueError):
                validate_compute({field: "unknown"})

    def test_only_selected_settings_profile_is_loaded_before_backend_import(self):
        parser = self.parser()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("ruinedfooocus.backends.comfy.image_compute.data_path") as paths,
        ):
            # Patch only the settings root; no shared state or backend is imported.
            paths.side_effect = lambda *parts: Path(folder).joinpath(*parts)
            root = Path(folder) / "settings"
            (root / "portable").mkdir(parents=True)
            (root / "settings.json").write_text(json.dumps({"image_vram_headroom": 1}))
            (root / "portable/settings.json").write_text(json.dumps({"image_vram_headroom": 2}))
            for argv, expected in (
                ([], 1),
                (["--settings", "portable"], 2),
                (["--settings", "missing"], 0),
                (["--settings", "portable", "--vram-headroom=0"], 0),
            ):
                args = parser.parse_args(argv)
                load_compute(args, parser, argv)
                self.assertEqual(args.vram_headroom, expected)

    def test_application_parser_applies_preferences_at_startup(self):
        source = Path(__file__).resolve().parents[1] / "ruinedfooocus/app/arguments.py"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "settings"
            root.mkdir()
            (root / "settings.json").write_text(
                json.dumps(
                    {
                        "image_vram_headroom": 2,
                        "image_dynamic_vram": "Disabled",
                        "image_async_offload": "4",
                        "image_preview_method": "taesd",
                    }
                )
            )
            with (
                patch(
                    "ruinedfooocus.backends.comfy.image_compute.data_path",
                    side_effect=lambda *parts: Path(folder).joinpath(*parts),
                ),
                patch.object(
                    sys,
                    "argv",
                    [
                        "ruinedfooocus/app/arguments.py",
                        "--vram-headroom=0",
                        "--preview-method=none",
                    ],
                ),
            ):
                args = runpy.run_path(str(source))["args"]
            self.assertEqual(
                (args.vram_headroom, args.preview_method, args.async_offload), (0, "none", 4)
            )
            self.assertTrue(args.disable_dynamic_vram)


if __name__ == "__main__":
    unittest.main()
