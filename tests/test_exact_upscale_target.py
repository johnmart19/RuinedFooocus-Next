"""Actual target consumers retain source identity, native factor and replay intent."""

import ast
from collections import Counter
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
import ruinedfooocus.app

from ruinedfooocus.features.generation import automatic_upscale as upscale
from ruinedfooocus.features.generation.size_plan import KEY, SizePlan, capture_size_plan

ROOT = Path(__file__).resolve().parents[1]


class ExactUpscaleTargetTests(unittest.TestCase):
    def test_effective_stages_reports_generation_dimensions_when_upscale_is_off(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8"))
        callback = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "update_upscale_size"
        )
        scope = dict(
            ui_controlnet=SimpleNamespace(
                tool_kind=lambda *_: None, checkpoint_features=lambda _: {}
            ),
            resolution_settings=SimpleNamespace(aspect_ratios={"square": (1024, 1024)}),
            describe_size=upscale.describe_size,
            t=lambda text: text,
        )
        exec(
            compile(
                ast.Module(body=[callback], type_ignores=[]), "effective stages caller", "exec"
            ),
            scope,
        )
        self.assertEqual(
            scope["update_upscale_size"](
                "Off", "Generated image", "square", 1216, 832, None, size_mode="target"
            ),
            "1024 × 1024 pixels",
        )
        self.assertEqual(
            scope["update_upscale_size"](
                "Off", "Generated image", "Custom...", 613, 977, None, size_mode="target"
            ),
            "613 × 977 pixels",
        )
        upscaled = scope["update_upscale_size"](
            "2×", "Generated image", "Custom...", 1920, 1080, None, size_mode="target"
        )
        self.assertIn("960 × 540 content", upscaled)
        self.assertIn("1920 × 1080 target (2×)", upscaled)
        self.assertIn(
            "Drop an image",
            scope["update_upscale_size"]("Off", "Input image only", "square", 1024, 1024, None),
        )

    def test_all_registered_native_scales_are_available_without_filename_inference(self):
        self.assertEqual(
            Counter(value["native_scale"] for value in upscale.registered_upscalers().values()),
            {2: 9, 4: 27, 3: 2, 8: 2},
        )
        self.assertIsNone(upscale.filename_scale("4x-completely-unknown.pth"))
        for factor in (2, 4):
            choices = upscale.choices(factor)
            self.assertEqual(choices[0], "Off")
            for choice in choices[1:]:
                self.assertEqual(upscale.scale_for(choice), factor)
        self.assertNotIn("HAT-3x.pth", upscale.choices(2))
        self.assertIn("HAT-3x.pth", upscale.choices(advanced=True))

    def test_unknown_native_descriptor_cache_is_signature_bound_and_not_name_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "misleading-8x-name.pth"
            path.write_bytes(b"inert fixture")
            manager = SimpleNamespace(
                model_paths={"cache_path": root / "cache"},
                get_model_folders=lambda category: [root],
            )
            self.assertIsNone(upscale.native_metadata(path.name, manager))
            model = SimpleNamespace(scale=2, input_channels=3, output_channels=3)
            upscale.remember_native_metadata(path, model, manager)
            self.assertEqual(upscale.native_metadata(path.name, manager)["native_scale"], 2)
            path.write_bytes(b"replaced artifact with another native scale")
            self.assertIsNone(upscale.native_metadata(path.name, manager))

    def test_request_capture_freezes_target_without_overwriting_denoise_or_source(self):
        source = Image.new("RGBA", (37, 19), (30, 70, 100, 64))
        data = dict(
            size_mode="target",
            aspect_ratios_selection="Custom...",
            custom_width=1920,
            custom_height=1080,
            resolution_auto=False,
            automatic_upscale="2× General (RealESRGAN)",
            automatic_upscale_target="Input image only",
            input_image=source,
            inpaint_denoise=0.31,
        )
        captured = capture_size_plan(data, None, controls={"type": "upscale"})
        data["custom_width"] = 3840
        self.assertEqual(captured[KEY].target, (1920, 1080))
        self.assertEqual(captured[KEY].content, (960, 540))
        self.assertIs(captured["input_image"], source)
        self.assertEqual(captured["inpaint_denoise"], 0.31)
        protected = dict(
            data, automatic_upscale_target="Generated image", image_resize_mode="preserve"
        )
        self.assertNotIn(KEY, capture_size_plan(protected, None, controls={"type": "img2img"}))
        self.assertNotIn(
            KEY,
            capture_size_plan(dict(data, resolution_auto=True), None, controls={"type": "upscale"}),
        )

    def test_native_input_target_keeps_alpha_pattern_and_original_upload(self):
        source = Image.fromarray(
            np.array(
                [
                    [[255, 0, 0, 0], [0, 255, 0, 128], [0, 0, 255, 255]],
                    [[255, 0, 0, 0], [0, 255, 0, 128], [0, 0, 255, 255]],
                ],
                dtype=np.uint8,
            )
        )
        before = np.asarray(source).copy()
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/image_processing/upscale_pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        process = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "process"
        )
        namespace = dict(
            np=np,
            torch=torch,
            worker=Mock(),
            controlnet=SimpleNamespace(get_settings=lambda data: {"upscaler": "fixture"}),
            InterruptibleUpscaler=lambda model, data: model,
        )
        exec(
            compile(ast.Module(body=[process], type_ignores=[]), "native_target", "exec"), namespace
        )
        for factor in (2, 4):
            plan = SizePlan.create((12, 8), factor, alignment=1)
            owner = SimpleNamespace(load_upscaler_model=lambda *args: SimpleNamespace(scale=factor))
            native = SimpleNamespace(
                ImageUpscaleWithModel=SimpleNamespace(
                    execute=lambda model, pixels: (
                        pixels.repeat_interleave(factor, 1).repeat_interleave(factor, 2),
                    )
                )
            )
            data = dict(input_image=source, task_id=1, **{KEY: plan})
            with patch.dict("sys.modules", {"comfy_extras.nodes_upscale_model": native}):
                result = namespace["process"](owner, data)[0]
            self.assertEqual(result.shape, (8, 12, 4))
            self.assertEqual(int(result[..., 3].min()), 0)
            self.assertEqual(int(result[..., 3].max()), 255)
            self.assertEqual(data["upscale_source_size"], [3, 2])
            import hashlib

            self.assertEqual(
                data["upscale_source"]["pixels_sha256"],
                hashlib.sha256(before.tobytes()).hexdigest(),
            )
            self.assertEqual(data["size_plan"], plan.metadata())
            np.testing.assert_array_equal(np.asarray(source), before)

    def test_loaded_upscaler_factor_mismatch_fails_before_processing(self):
        owner = SimpleNamespace(
            automatic_upscale_pipeline=SimpleNamespace(
                load_upscaler_model=lambda *args: SimpleNamespace(scale=4)
            )
        )
        native = SimpleNamespace(ImageUpscaleWithModel=SimpleNamespace(execute=Mock()))
        with patch.dict(
            "sys.modules",
            {
                "ruinedfooocus.features.image_processing.upscale_pipeline": SimpleNamespace(
                    pipeline=Mock(), InterruptibleUpscaler=Mock()
                ),
                "ruinedfooocus.app.async_worker": Mock(),
                "comfy_extras.nodes_upscale_model": native,
            },
        ):
            with self.assertRaisesRegex(ValueError, "Unexpected scale"):
                upscale.upscale_decoded(
                    owner, torch.zeros(1, 2, 3, 3), dict(automatic_upscale="2×", task_id=1)
                )
        native.ImageUpscaleWithModel.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
