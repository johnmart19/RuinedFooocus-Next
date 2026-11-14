"""Exercise the brush, native reference conditioning and output framing on CPU."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import torch

from ruinedfooocus.features.image_editing.image_geometry import restore_reference_frame
from ruinedfooocus.features.image_editing.image_guidance import normalize_reference_images
from ruinedfooocus.features.image_editing.inpaint_utils import source_and_mask
from ruinedfooocus.features.image_editing.qwen_image21_conditioning import QwenImage21References
from test_powerup_family_controls import functions
from test_qwen_image21_references import MODULE, clip_fixture, native_references, vae_fixture


class QwenImage21LocalEditTests(unittest.TestCase):
    def test_native_edit_reads_red_guide_and_keeps_unpainted_rgba_exact_in_mask_mode(self):
        raw = np.random.default_rng(42).integers(0, 256, (67, 81, 4), dtype=np.uint8)
        source = Image.fromarray(raw)
        layer = Image.new("RGBA", source.size)
        ImageDraw.Draw(layer).rectangle((11, 18, 35, 50), fill=(0, 0, 0, 255))
        layer.putpixel((36, 25), (0, 0, 0, 128))
        source, mask = source_and_mask(source, {"layers": [layer]}, preserve_alpha=True)
        extra = Image.new("RGBA", (32, 32), (0, 255, 0, 255))
        for mode in ("paint", "mask"):
            with self.subTest(mode=mode):
                vae, clip = vae_fixture(), clip_fixture()
                refs = QwenImage21References(
                    vae, source, [extra], (32, 32), "preserve", paint_mask=mask, edit_mode=mode
                )
                modules, upscale = native_references()
                prompt = "Make the area painted red in <image1> look like <image2>."
                with (
                    patch.dict("sys.modules", modules),
                    patch(MODULE + ".warm_patched_text_encoder"),
                ):
                    positive, _, latent = refs.conditioning(clip, prompt, "")
                self.assertEqual(positive[0][0], prompt)
                self.assertNotIn("noise_mask", latent)
                self.assertEqual(latent["samples"].shape, (1, 64, 6, 6))
                self.assertEqual(vae.encode.call_count, 2)
                upscale.assert_not_called()

                guide = restore_reference_frame(vae.encode.call_args_list[0].args[0], refs.frame)
                torch.testing.assert_close(guide[0, 20, 20], torch.tensor([1.0, 0.0, 0.0, 1.0]))
                vision = clip.tokenize.call_args_list[0].kwargs["images"]
                self.assertEqual(len(vision), 2)
                torch.testing.assert_close(vision[1][0, 0, 0], torch.tensor([0.0, 1.0, 0.0]))
                np.testing.assert_array_equal(
                    (guide[0].numpy() * 255).astype(np.uint8)[np.array(mask) == 0],
                    raw[np.array(mask) == 0],
                )

                generated = torch.zeros_like(refs.images["image_1"])
                generated[..., 2:] = 1
                framed = restore_reference_frame(refs.finish(generated), refs.frame)
                self.assertEqual(framed.shape, (1, 67, 81, 4))
                self.assertEqual(refs.frame["qwen_local_edit_mode"], mode)
                blue = torch.tensor([0.0, 0.0, 1.0, 1.0])
                torch.testing.assert_close(framed[0, 20, 20], blue)
                if mode == "paint":
                    self.assertIs(refs.finish(generated), generated)
                else:
                    np.testing.assert_array_equal(
                        (framed[0].numpy() * 255).astype(np.uint8)[np.array(mask) == 0],
                        raw[np.array(mask) == 0],
                    )
                    self.assertLess(framed[0, 25, 36, 3], 1)
                    self.assertGreater(framed[0, 25, 36, 3], raw[25, 36, 3] / 255)
        np.testing.assert_array_equal(np.asarray(source), raw)

    def test_pad_aligns_source_guide_and_composite_and_crop_rejects_lost_mask(self):
        source = Image.new("RGBA", (81, 27), (3, 21, 191, 80))
        mask = Image.new("L", source.size)
        ImageDraw.Draw(mask).rectangle((0, 0, 5, 7), fill=255)
        refs = QwenImage21References(
            vae_fixture(), source, [], (65, 65), "pad", paint_mask=mask, edit_mode="mask"
        )
        guide = restore_reference_frame(refs.images["image_1"], refs.frame)
        blended = restore_reference_frame(
            refs.finish(torch.ones_like(refs.images["image_1"])), refs.frame
        )
        top = refs.frame["content_top"]
        self.assertEqual(guide.shape, (1, 65, 65, 4))
        self.assertTrue((guide[0, top, 0] == torch.tensor([1, 0, 0, 1])).all())
        self.assertEqual(blended[0, top - 1, 0, 3], 0)
        self.assertEqual(blended[0, top, 0, 3], 1)
        with self.assertRaisesRegex(ValueError, "crops out the entire painted area"):
            QwenImage21References(
                vae_fixture(), source, [], (65, 65), "crop", paint_mask=mask, edit_mode="mask"
            )

    def test_qwen_ui_selects_main_editor_and_disallows_conflicting_image_tools(self):
        scope, models = functions()
        models.detect_checkpoint_base.return_value = "QwenImage21"
        scope["controlnet"].cn_options["Native image edit"] = {"type": "edit"}
        features = scope["checkpoint_features"]("qwen")
        self.assertTrue(features["qwen_local"])
        for selection in ("None", "Img2Img", "Native image edit"):
            upload, strength, toggle, options, _ = scope["input_context"](
                True, selection, "Img2Img", "qwen"
            )
            self.assertFalse(upload["visible"])
            self.assertEqual(strength["visible"], "hidden")
            self.assertTrue(toggle["visible"])
            self.assertTrue(options["visible"])
        for selection in ("Canny", "Remove background"):
            toggle = scope["input_context"](True, selection, "Img2Img", "qwen")[2]
            self.assertEqual(toggle, {"visible": False, "value": False})

    def test_pipeline_uses_main_and_native_edit_grid_instead_of_stale_tool_upload(self):
        pipeline = (
            Path(__file__).resolve().parents[1]
            / "ruinedfooocus/features/generation/image_pipeline.py"
        )
        process = next(
            node
            for node in ast.walk(ast.parse(pipeline.read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef) and node.name == "process"
        )

        def index(name):
            return next(
                i
                for i, node in enumerate(process.body)
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
            )

        preflight = process.body[index("qwen_local_edit") : index("checkpoint")]
        control_route = process.body[index("controlnet") : index("device")]
        # The Qwen reference test is an elif after the named-source branch.
        prepare_reference = next(
            node
            for node in ast.walk(process)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "qwen_reference_edit"
        )
        source = Image.new("RGBA", (81, 67), (12, 30, 70, 50))
        tool_upload = Image.new("RGBA", (44, 44))
        layer = Image.new("RGBA", source.size)
        ImageDraw.Draw(layer).rectangle((20, 21, 54, 49), fill=(0, 0, 0, 255))
        request = dict(
            inpaint_toggle=True,
            main_view=source,
            inpaint_view={"layers": [layer]},
            input_image=tool_upload,
            qwen_local_edit_mode="mask",
            outpaint_left=16,
            width=64,
            height=64,
        )
        scope = dict(
            gen_data=request,
            architecture="QwenImage21",
            SDXL=type("SDXL", (), {}),
            flux_inpaint=False,
            image_edit=False,
            self=SimpleNamespace(
                xl_base_patched=SimpleNamespace(
                    unet=SimpleNamespace(model=object()), vae=vae_fixture()
                )
            ),
            guidance={"image_resize_mode": "preserve", "moss_image_edit_format": "Disabled"},
            control_tools=SimpleNamespace(
                get_settings=lambda _: {"type": "img2img", "denoise": 0.4},
                controlnet_models={"img2img": None, "edit": None},
            ),
            require_instruct_edit=lambda *_: None,
            require_sd4x_source=lambda *_: None,
            normalize_reference_images=normalize_reference_images,
            pid_model=False,
            refiner_model=False,
        )
        nodes = [*preflight, *control_route, prepare_reference]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(pipeline), "exec"), scope)
        self.assertEqual(scope["control_type"], "edit")
        self.assertIs(request["input_image"], tool_upload)
        self.assertEqual(scope["input_image"].size, source.size)
        self.assertEqual(request["reference_image_count"], 1)
        self.assertEqual((request["width"], request["height"]), (96, 96))
        self.assertEqual(scope["reference_frame"]["qwen_local_edit_mode"], "mask")
