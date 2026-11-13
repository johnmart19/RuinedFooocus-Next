"""Native SD1/SD2 nine-channel detection and real native source/mask conditioning."""

from types import SimpleNamespace as NS
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from ruinedfooocus.features.image_editing.native_sd_inpaint import (
    is_native_sd_inpaint,
    native_sd_inpaint_conditioning,
    require_native_sd_inpaint_source,
    local_native_sd_inpaint,
)
from test_native_image_conditioning import native_mask_nodes
from test_qwen_image21_conditioning import native_class, native_function


def model(family, channels):
    return NS(model_config=type(family, (), {"unet_config": {"in_channels": channels}})())


class NativeSDInpaintTests(unittest.TestCase):
    def test_current_header_capability_and_public_masked_api_admit_nine_channels_only(self):
        from test_api_native_inpaint import NativeInpaintApiTests, inline

        fixture = NativeInpaintApiTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.selected["family"] = "SD 1.5"
        with patch(
            "ruinedfooocus.features.models.model_architecture.inspect_model_file",
            return_value=dict(native_class="SD15", native_config={"in_channels": 9}),
        ) as inspect:
            self.assertTrue(local_native_sd_inpaint(fixture.selected["path"]))
            response = fixture.json_edit(mask=inline(fixture.mask))
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(fixture.prepared[-1]["inpaint_toggle"])
            self.assertEqual(fixture.prepared[-1]["main_view"].size, fixture.size)
            inspect.return_value = dict(native_class="SD15", native_config={"in_channels": 4})
            response = fixture.json_edit(mask=inline(fixture.mask))
            self.assertEqual(response.status_code, 400)
            self.assertEqual(fixture.submit.call_count, 1)

    def test_production_masked_caller_selects_native_nine_channel_conditioning(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "ruinedfooocus/features/generation/image_pipeline.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        process = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "process"
        )
        branch = next(
            node
            for node in process.body
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "VAEEncodeForInpaint"
                for child in ast.walk(node)
            )
        )
        encode = lambda image: torch.cat(
            (
                torch.nn.functional.avg_pool2d(image.movedim(-1, 1), 8),
                torch.zeros((1, 1, image.shape[1] // 8, image.shape[2] // 8)),
            ),
            dim=1,
        )
        scope = dict(
            gen_data=dict(width=40, height=32, inpaint_toggle=True, inpaint_mask_grow=0),
            architecture="BaseModel",
            flux_inpaint=False,
            sd_inpaint=True,
            positive_conditioning=[["positive", {}]],
            negative_conditioning=[["negative", {}]],
            switched_prompt=[],
            inpaint_source=Image.new("RGB", (40, 32), (30, 60, 90)),
            inpaint_mask=Image.new("L", (40, 32), 255),
            guidance={"image_resize_mode": "preserve"},
            self=NS(
                xl_base_patched=NS(
                    unet=NS(model=model("SD15", 9)),
                    vae=NS(latent_channels=4, spacial_compression_encode=lambda: 8, encode=encode),
                )
            ),
        )
        with patch.dict("sys.modules", native_mask_nodes()):
            exec(
                compile(
                    ast.Module(body=[branch], type_ignores=[]),
                    "production native SD inpaint caller",
                    "exec",
                ),
                scope,
            )
        self.assertEqual(scope["latent"]["samples"].shape[1], 4)
        self.assertIn("concat_mask", scope["positive_conditioning"][0][1])
        self.assertEqual(scope["gen_data"]["native_inpaint_conditioning"]["native_config"], "SD15")

    def test_only_native_sd1_sd2_nine_channel_layouts_require_a_mask(self):
        for family in ("SD15", "SD20"):
            self.assertTrue(is_native_sd_inpaint(model(family, 9)))
            with self.assertRaisesRegex(ValueError, "requires Inpainting"):
                require_native_sd_inpaint_source(model(family, 9), False)
            require_native_sd_inpaint_source(model(family, 9), True)
        for family, channels in (
            ("SD15", 4),
            ("SD20", 4),
            ("SDXL", 9),
            ("SD15_instructpix2pix", 8),
        ):
            self.assertFalse(is_native_sd_inpaint(model(family, channels)))

    def test_native_node_conditions_both_scheduled_prompts_and_preserves_source_latent(self):
        pixels = torch.full((1, 32, 40, 3), 0.25)
        mask = torch.zeros((1, 32, 40))
        mask[:, 12:20, 15:23] = 1
        calls = []

        def encode(image):
            calls.append(image.clone())
            rgb = torch.nn.functional.avg_pool2d(image.movedim(-1, 1), 8)
            return torch.cat((rgb, torch.zeros_like(rgb[:, :1])), dim=1)

        positive = [["early", {"start_percent": 0}], ["late", {"start_percent": 0.5}]]
        negative = [["negative", {}]]
        with patch.dict("sys.modules", native_mask_nodes()):
            conditioned, unconditioned, latent = native_sd_inpaint_conditioning(
                model("SD20", 9),
                positive,
                negative,
                NS(encode=encode, latent_channels=4, spacial_compression_encode=lambda: 8),
                pixels,
                mask,
                0,
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(tuple(latent["samples"].shape), (1, 4, 4, 5))
        torch.testing.assert_close(
            latent["samples"][:, :3], torch.nn.functional.avg_pool2d(pixels.movedim(-1, 1), 8)
        )
        self.assertEqual(conditioned[1][1]["start_percent"], 0.5)
        self.assertNotIn("concat_mask", positive[0][1])
        for _, row in conditioned + unconditioned:
            self.assertIn("concat_latent_image", row)
            torch.testing.assert_close(row["concat_mask"], latent["noise_mask"])
        self.assertTrue((calls[0][mask.bool()] == 0.5).all())
        torch.testing.assert_close(calls[0][~mask.bool()], pixels[~mask.bool()])
        scope = dict(torch=torch)
        upscale = native_function("comfy/utils.py", "common_upscale", scope)
        repeat = native_function("comfy/utils.py", "resize_to_batch_size", scope)
        native = native_class(
            "comfy/model_base.py",
            "BaseModel",
            dict(torch=torch, utils=NS(common_upscale=upscale, resize_to_batch_size=repeat)),
            methods=("concat_cond",),
        )
        owner = native()
        owner.concat_keys = ("mask", "masked_image")
        owner.process_latent_in = lambda samples: samples
        concatenated = owner.concat_cond(**conditioned[0][1], noise=latent["samples"], device="cpu")
        self.assertEqual(tuple(concatenated.shape), (1, 5, 4, 5))
        self.assertEqual(torch.cat((latent["samples"], concatenated), dim=1).shape[1], 9)


if __name__ == "__main__":
    unittest.main()
