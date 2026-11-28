from ruinedfooocus.core.paths import asset_path, resource_path

"""Inpaint source/mask geometry through the pinned VAE node; CPU, no weights."""
import ast
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw
import torch

from ruinedfooocus.features.image_editing.image_geometry import (
    reference_plan,
    restore_reference_frame,
)

ROOT = Path(__file__).resolve().parents[1]


def native_node(class_name="VAEEncodeForInpaint", method_name="encode"):
    native = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))
    tree = ast.parse((native / "nodes.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method_name]
    scope = dict(asset_path=asset_path, resource_path=resource_path, torch=torch, math=math)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "native_inpaint", "exec"), scope)
    return scope[node.name]


class InpaintGeometryTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        process = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process"
        )
        branch = next(
            n
            for n in process.body
            if isinstance(n, ast.If)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "VAEEncodeForInpaint"
                for child in ast.walk(n)
            )
        )
        self.code = compile(
            ast.Module(body=[branch], type_ignores=[]), "production_inpaint_route", "exec"
        )
        self.vae = SimpleNamespace(
            spacial_compression_encode=lambda: 8,
            encode=Mock(
                side_effect=lambda pixels: torch.zeros(
                    (1, 4, pixels.shape[1] // 8, pixels.shape[2] // 8), device="cpu"
                )
            ),
        )

    def route(self, image, mask=None, mode="preserve", data=None, *, outpaint=False):
        if mask is None and not outpaint:
            mask = Image.new("L", image.size, 255)
        request = data if data is not None else dict(width=1152, height=896, inpaint_toggle=True)
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            np=np,
            torch=torch,
            gen_data=request,
            architecture="SDXL",
            guidance={"image_resize_mode": mode},
            inpaint_source=image,
            inpaint_mask=mask,
            reference_frame=None,
            VAEEncodeForInpaint=native_node(),
            flux_inpaint=False,
            sd_inpaint=False,
            self=SimpleNamespace(xl_base_patched=SimpleNamespace(vae=self.vae)),
        )
        exec(self.code, scope)
        return scope

    def test_canvas_extension_encodes_native_border_mask_and_preserves_unfeathered_source(self):
        from ruinedfooocus.features.image_editing.inpaint_utils import blend_outpaint

        image = np.arange(37 * 51 * 3, dtype=np.uint8).reshape(37, 51, 3)
        request = dict(
            width=256,
            height=256,
            inpaint_toggle=True,
            inpaint_mask_grow=0,
            outpaint_left=5,
            outpaint_top=7,
            outpaint_right=10,
            outpaint_feather=4,
        )
        native = native_node("ImagePadForOutpaint", "expand_image")
        with patch.dict(sys.modules, {"nodes": SimpleNamespace(ImagePadForOutpaint=native)}):
            result = self.route(Image.fromarray(image), mode="crop", data=request, outpaint=True)

        plan = result["reference_frame"]
        self.assertEqual((plan["width"], plan["height"], plan["mode"]), (66, 44, "outpaint"))
        self.assertEqual((request["width"], request["height"]), (80, 48))
        noise_mask = restore_reference_frame(result["latent"]["noise_mask"][:, 0, ..., None], plan)
        self.assertTrue(torch.all(noise_mask[0, :7] == 1))
        self.assertTrue(torch.all(noise_mask[0, 11:, 9:52] == 0))
        decoded = torch.zeros((1, 44, 66, 3))
        composite = blend_outpaint(decoded, *result["outpaint_composite"], plan)
        np.testing.assert_array_equal(
            np.clip(composite[0, 11:, 9:52].numpy() * 255, 0, 255).astype(np.uint8),
            image[4:, 4:47],
        )
        self.assertEqual(composite[0, 0].count_nonzero().item(), 0)
        self.assertLess(composite[0, 8, 9, 0], image[1, 4, 0] / 255)

    def test_canvas_extension_rejects_fractional_margins_and_oversized_canvas(self):
        from ruinedfooocus.features.image_editing.inpaint_utils import (
            outpaint_margins,
            prepare_outpaint,
        )

        with self.assertRaisesRegex(ValueError, "whole pixels"):
            outpaint_margins({"outpaint_right": 1.5})
        with (
            patch.dict(sys.modules, {"nodes": SimpleNamespace(ImagePadForOutpaint=Mock())}),
            self.assertRaisesRegex(ValueError, "4096"),
        ):
            prepare_outpaint(Image.new("RGB", (4096, 2)), {"outpaint_right": 1})

    def test_all_six_source_shapes_fit_the_selected_budget_without_cropped_edges(self):
        for size in (
            (3364, 7480),
            (928, 1664),
            (2758, 3270),
            (768, 1344),
            (1344, 896),
            (928, 1232),
        ):
            with self.subTest(size=size):
                result = self.route(Image.new("RGB", size))
                expected = reference_plan(size, (1152, 896))
                self.assertEqual(result["reference_frame"], expected)
                shape = result["latent"]["samples"].shape
                self.assertEqual(
                    (shape[-1] * 8, shape[-2] * 8),
                    (expected["canvas_width"], expected["canvas_height"]),
                )
                self.assertEqual(
                    tuple(result["latent"]["noise_mask"].shape[-2:]),
                    (expected["canvas_height"], expected["canvas_width"]),
                )
                self.assertEqual(result["gen_data"]["_image_requested_size"], (1152, 896))

    def test_small_non_grid_source_keeps_every_pixel_and_exact_final_size(self):
        image = np.arange(101 * 117 * 3, dtype=np.uint8).reshape(101, 117, 3)
        mask = Image.new("L", (117, 101))
        ImageDraw.Draw(mask).rectangle((40, 30, 60, 50), fill=255)
        result = self.route(Image.fromarray(image), mask)
        pixels = self.vae.encode.call_args.args[0]
        restored = restore_reference_frame(pixels, result["reference_frame"])[0]
        self.assertEqual(tuple(restored.shape), (101, 117, 3))
        unmasked = np.array(mask) == 0
        np.testing.assert_array_equal(
            (restored.numpy()[unmasked] * 255).round().astype(np.uint8), image[unmasked]
        )

    def test_source_and_mask_share_the_preserve_or_crop_transform(self):
        from ruinedfooocus.features.image_editing.inpaint_utils import prepare_inpaint

        mask = Image.new("L", (301, 701))
        ImageDraw.Draw(mask).ellipse((101, 301, 199, 399), fill=255)
        source = Image.merge("RGB", (mask, mask, mask))
        for mode in ("preserve", "pad", "crop"):
            pixels, alpha, plan = prepare_inpaint(source, mask, (192, 320), mode)
            restored = restore_reference_frame(pixels, plan)[0, ..., 0]
            cropped_alpha = alpha[
                0,
                plan["top"] : plan["top"] + plan["height"],
                plan["left"] : plan["left"] + plan["width"],
            ]
            if mode == "pad":
                content = torch.zeros_like(cropped_alpha, dtype=torch.bool)
                left, top = plan["content_left"], plan["content_top"]
                content[top : top + plan["content_height"], left : left + plan["content_width"]] = (
                    True
                )
                torch.testing.assert_close(
                    restored[content], cropped_alpha[content], rtol=0, atol=0
                )
                self.assertTrue((cropped_alpha[~content] == 0).all(), "Padding must stay unpainted")
                self.assertTrue(
                    (restored[~content] == 0.5).all(), "Image padding must stay neutral"
                )
            else:
                torch.testing.assert_close(restored, cropped_alpha, rtol=0, atol=0)
            y, x = torch.where(cropped_alpha > 0.5)
            self.assertLessEqual(abs(int(x.max() - x.min()) - int(y.max() - y.min())), 1)
            self.assertEqual(pixels.shape[1:3], alpha.shape[1:3])

    def test_repeated_requests_reuse_the_original_budget_instead_of_padded_metadata(self):
        request = dict(width=1152, height=896, inpaint_toggle=True)
        image = Image.new("RGB", (928, 1664))
        plans = []
        for _ in range(3):
            result = self.route(image, data=request)
            plans.append(result["reference_frame"])
            request.update(width=plans[-1]["width"], height=plans[-1]["height"])
        self.assertEqual(plans, [reference_plan(image.size, (1152, 896))] * 3)

    def test_invalid_inpaint_controls_fail_before_encoding(self):
        for field, value in (
            ("inpaint_denoise", 0),
            ("inpaint_denoise", 1.1),
            ("inpaint_mask_grow", -1),
            ("inpaint_mask_grow", 129),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    self.route(
                        Image.new("RGB", (32, 32)),
                        data=dict(width=64, height=64, inpaint_toggle=True, **{field: value}),
                    )
                self.vae.encode.assert_not_called()

    def test_inpaint_ignores_unused_img2img_uploads_without_erasing_them_or_control_guides(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        process = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process"
        )

        def assignment(name):
            return next(
                i
                for i, n in enumerate(process.body)
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
            )

        code = compile(
            ast.Module(
                body=process.body[assignment("controlnet") : assignment("device")], type_ignores=[]
            ),
            "inpaint_input_preflight",
            "exec",
        )
        model_type = type("SDXL", (), {})
        image = object()

        def prepare(control, inpaint=True, source=image):
            from ruinedfooocus.features.image_editing.native_conditioning import (
                require_instruct_edit,
            )
            from ruinedfooocus.features.image_editing.image_guidance import (
                normalize_reference_images,
            )
            from ruinedfooocus.features.image_editing.sd4x_conditioning import require_sd4x_source

            data = dict(
                inpaint_toggle=inpaint,
                input_image=source,
                reference_image_count=int(source is not None),
            )
            scope = dict(
                asset_path=asset_path,
                resource_path=resource_path,
                gen_data=data,
                input_image=source,
                image_edit=False,
                hidream_edit=False,
                architecture="SDXL",
                qwen_local_edit=False,
                normalize_reference_images=normalize_reference_images,
                guidance={"moss_image_edit_format": "Disabled"},
                require_instruct_edit=require_instruct_edit,
                require_sd4x_source=require_sd4x_source,
                pid_model=False,
                refiner_model=False,
                SDXL=model_type,
                self=SimpleNamespace(
                    xl_base_patched=SimpleNamespace(unet=SimpleNamespace(model=model_type()))
                ),
                control_tools=SimpleNamespace(
                    get_settings=lambda data: dict(control),
                    controlnet_models=dict.fromkeys(
                        ("img2img", "canny", "depth", "sketch", "recolour")
                    ),
                ),
            )
            exec(code, scope)
            self.assertIs(data["input_image"], source)
            return scope

        for control in ({}, {"type": "None"}, {"type": "img2img", "denoise": 0.25}):
            for source in (None, image):
                result = prepare(control, source=source)
                self.assertIsNone(result["input_image"])
                self.assertNotEqual(result["control_type"], "img2img")
                self.assertEqual(result["gen_data"]["reference_image_count"], 0)
        for kind in ("canny", "depth", "sketch", "recolour"):
            control = dict(type=kind, start=0, stop=1)
            self.assertIs(prepare(control)["input_image"], image)
            with self.assertRaisesRegex(ValueError, r"^Add an image in Image tools\.$"):
                prepare(control, source=None)
        result = prepare({"type": "img2img", "denoise": 0.25}, inpaint=False)
        self.assertIs(result["input_image"], image)
        self.assertEqual(result["control_type"], "img2img")


if __name__ == "__main__":
    unittest.main()
