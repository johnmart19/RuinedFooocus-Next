"""Native masked conditioning and its real image-tool callers; no weights/devices."""

import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw
import scipy.ndimage
import torch

from ruinedfooocus.features.image_editing.inpaint_utils import prepare_inpaint, source_and_mask
from ruinedfooocus.features.image_editing.native_conditioning import (
    flux_inpaint_conditioning,
    is_flux_inpaint,
    require_inpaint_input,
)
from ruinedfooocus.features.models.image_components import inventory
from test_powerup_family_controls import functions

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))


def native_mask_nodes():
    """Execute only the pinned CPU methods, with no backend/module initialization."""
    scope = dict(torch=torch, np=np, scipy=scipy, IO=SimpleNamespace(NodeOutput=lambda x: (x,)))
    helpers = ast.parse((COMFY / "node_helpers.py").read_text())
    setter = next(
        node
        for node in helpers.body
        if isinstance(node, ast.FunctionDef) and node.name == "conditioning_set_values"
    )
    exec(compile(ast.Module(body=[setter], type_ignores=[]), "native_node_helpers", "exec"), scope)
    scope["node_helpers"] = SimpleNamespace(conditioning_set_values=scope[setter.name])
    for filename, name, method in (
        ("nodes.py", "InpaintModelConditioning", "encode"),
        ("comfy_extras/nodes_mask.py", "GrowMask", "execute"),
    ):
        tree = ast.parse((COMFY / filename).read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        node.bases = []
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method]
        node.body[0].returns = None
        exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), scope)
    return {
        "nodes": SimpleNamespace(InpaintModelConditioning=scope["InpaintModelConditioning"]),
        "comfy_extras.nodes_mask": SimpleNamespace(GrowMask=scope["GrowMask"]),
    }


def flux_model(channels=96, image_model="flux"):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            unet_config=dict(image_model=image_model, in_channels=channels)
        )
    )


class NativeImageConditioningTests(unittest.TestCase):
    def test_flux_inpaint_uses_existing_captured_flux_component_roles_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "unrelated-name.safetensors"
            header = json.dumps({"weight": {"shape": [1]}}).encode()
            model.write_bytes(len(header).to_bytes(8, "little") + header)
            paths = SimpleNamespace(
                paths={"path_clip": str(root / "clip"), "path_vae": str(root / "vae")},
                get_abspath=Path,
                DOWNLOADABLE_FILES={},
                download_file=Mock(),
            )
            with (
                patch.dict("sys.modules", {"comfy.model_detection": SimpleNamespace()}),
                patch(
                    "ruinedfooocus.features.models.model_architecture.inspect_tensor_model",
                    return_value=dict(
                        architecture="FluxInpaint", text_encoder_prefixes=(), vae_prefixes=()
                    ),
                ),
            ):
                state = inventory(model, "Unknown", {}, paths)
            self.assertEqual(state["architecture"], "Flux")
            self.assertEqual(
                [row["key"] for row in state["rows"]], ["clip_l", "clip_t5", "vae_flux"]
            )
            paths.download_file.assert_not_called()

    def pipeline_process(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        return next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "process"
        )

    def test_real_pipeline_preflight_refuses_ordinary_flux_and_requires_masked_source(self):
        body = self.pipeline_process().body
        start = next(
            i
            for i, node in enumerate(body)
            if isinstance(node, ast.ImportFrom)
            and node.module == "ruinedfooocus.features.image_editing.native_conditioning"
        )
        stop = next(
            i
            for i, node in enumerate(body)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "checkpoint" for t in node.targets)
        )
        code = compile(
            ast.Module(body=body[start:stop], type_ignores=[]), "native_inpaint_preflight", "exec"
        )

        def preflight(model, **data):
            scope = dict(
                self=SimpleNamespace(
                    xl_base_patched=SimpleNamespace(unet=SimpleNamespace(model=model))
                ),
                gen_data=data,
                architecture=type(model).__name__,
                SDXL=type("SDXL", (), {}),
            )
            exec(code, scope)
            return scope

        with self.assertRaisesRegex(ValueError, "SDXL or Flux inpaint"):
            preflight(flux_model(16), inpaint_toggle=True)
        with self.assertRaisesRegex(ValueError, "requires Inpainting"):
            preflight(flux_model(), inpaint_toggle=False)
        with self.assertRaisesRegex(ValueError, "source image"):
            preflight(flux_model(), inpaint_toggle=True)
        image = Image.new("RGB", (32, 48))
        with self.assertRaisesRegex(ValueError, "Paint a mask"):
            preflight(flux_model(), inpaint_toggle=True, main_view=image, inpaint_view={})
        layer = Image.new("RGBA", image.size)
        ImageDraw.Draw(layer).rectangle((8, 8, 16, 16), fill=(255, 255, 255, 255))
        valid = preflight(
            flux_model(), inpaint_toggle=True, main_view=image, inpaint_view={"layers": [layer]}
        )
        self.assertEqual(valid["inpaint_source"].size, image.size)
        self.assertIsNotNone(valid["inpaint_mask"].getbbox())

    def test_native_config_controls_admission_not_filename_or_display_name(self):
        self.assertTrue(is_flux_inpaint(flux_model()))
        for model in (flux_model(16), flux_model(32), flux_model(96, "flux2"), object()):
            self.assertFalse(is_flux_inpaint(model))
            require_inpaint_input(model, False)
        with self.assertRaisesRegex(ValueError, "Main image and painted mask"):
            require_inpaint_input(flux_model(), False)
        require_inpaint_input(flux_model(), True)

    def test_required_main_image_and_painted_mask_keep_existing_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "source image"):
            source_and_mask(None, {})
        with self.assertRaisesRegex(ValueError, "Paint a mask"):
            source_and_mask(Image.new("RGB", (32, 32)), {})

    def test_native_node_preserves_original_latent_and_conditions_both_prompt_schedules(self):
        image = Image.new("RGB", (117, 101), (204, 51, 102))
        painted = Image.new("L", image.size)
        ImageDraw.Draw(painted).rectangle((40, 30, 60, 50), fill=255)
        pixels, mask, frame = prepare_inpaint(image, painted, (192, 128), "preserve")
        inputs = []

        def encode(value):
            inputs.append(value.clone())
            return value.movedim(-1, 1)

        positive = [
            ["early", {"start_percent": 0.0, "end_percent": 0.4}],
            ["late", {"start_percent": 0.4}],
        ]
        negative = [["negative", {"weight": 2}]]
        with patch.dict("sys.modules", native_mask_nodes()):
            pos, neg, latent = flux_inpaint_conditioning(
                positive, negative, SimpleNamespace(encode=encode), pixels, mask, grow_mask_by=0
            )
        self.assertEqual(len(inputs), 2)
        torch.testing.assert_close(inputs[1], pixels)
        torch.testing.assert_close(latent["samples"], pixels.movedim(-1, 1))
        self.assertTrue((inputs[0][mask.round().bool()] == 0.5).all())
        torch.testing.assert_close(inputs[0][~mask.round().bool()], pixels[~mask.round().bool()])
        for original, conditioned in zip(positive + negative, pos + neg):
            for key, value in original[1].items():
                self.assertEqual(conditioned[1][key], value)
            self.assertNotIn("concat_latent_image", original[1])
            torch.testing.assert_close(
                conditioned[1]["concat_latent_image"], inputs[0].movedim(-1, 1)
            )
            torch.testing.assert_close(conditioned[1]["concat_mask"], latent["noise_mask"])
        self.assertEqual(
            tuple(latent["noise_mask"].shape[-2:]), (frame["canvas_height"], frame["canvas_width"])
        )
        self.assertEqual(tuple(pixels.shape[1:3]), tuple(latent["noise_mask"].shape[-2:]))

    def test_expansion_uses_native_grow_for_conditioning_and_sampler_mask(self):
        pixels = torch.zeros((1, 32, 32, 3))
        mask = torch.zeros((1, 32, 32))
        mask[:, 15, 15] = 1
        with patch.dict("sys.modules", native_mask_nodes()):
            pos, _, latent = flux_inpaint_conditioning(
                [["positive", {}]], [], SimpleNamespace(encode=lambda value: value), pixels, mask, 2
            )
        expected = torch.zeros((1, 1, 32, 32))
        expected[:, :, 13:18, 13:18] = 1
        torch.testing.assert_close(latent["noise_mask"], expected)
        torch.testing.assert_close(pos[0][1]["concat_mask"], expected)
        self.assertEqual(int(mask.sum()), 1, "Native growth must not mutate the captured mask")

    def test_real_pipeline_uses_native_latent_with_scheduled_positive_and_negative(self):
        branch = next(
            node
            for node in self.pipeline_process().body
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "VAEEncodeForInpaint"
                for child in ast.walk(node)
            )
        )
        code = compile(
            ast.Module(body=[branch], type_ignores=[]), "production_masked_conditioning", "exec"
        )
        source = Image.new("RGB", (117, 101), (90, 120, 210))
        mask = Image.new("L", source.size)
        ImageDraw.Draw(mask).rectangle((30, 20, 50, 40), fill=255)
        for scheduled in (False, True):
            request = dict(
                width=192, height=128, inpaint_toggle=True, inpaint_denoise=0.6, inpaint_mask_grow=2
            )
            scope = dict(
                gen_data=request,
                architecture="Flux",
                flux_inpaint=True,
                sd_inpaint=False,
                positive_conditioning=[["plain", {}]],
                switched_prompt=[
                    ["early", {"start_percent": 0.0}],
                    ["late", {"start_percent": 0.5}],
                ]
                if scheduled
                else [],
                negative_conditioning=[["negative", {}]],
                inpaint_source=source,
                inpaint_mask=mask,
                guidance={"image_resize_mode": "preserve"},
                self=SimpleNamespace(
                    xl_base_patched=SimpleNamespace(
                        vae=SimpleNamespace(encode=lambda value: value.movedim(-1, 1)),
                        unet=SimpleNamespace(model=flux_model()),
                    )
                ),
            )
            with patch.dict("sys.modules", native_mask_nodes()):
                exec(code, scope)
            positive = scope["switched_prompt"] if scheduled else scope["positive_conditioning"]
            self.assertEqual(len(positive), 2 if scheduled else 1)
            self.assertTrue(all("concat_latent_image" in entry[1] for entry in positive))
            self.assertIn("concat_latent_image", scope["negative_conditioning"][0][1])
            self.assertEqual(scope["denoise"], 0.6)
            self.assertEqual(request["_image_requested_size"], (192, 128))
            plan = scope["reference_frame"]
            self.assertEqual(
                scope["latent"]["samples"].shape[-2:], (plan["canvas_height"], plan["canvas_width"])
            )

    def test_header_derived_flux_inpaint_enables_mask_with_native_control_choices(self):
        scope, models = functions()
        models.get_model_base.return_value = "SDXL 1.0"  # Deliberately stale metadata.
        models.detect_checkpoint_base.return_value = "FluxInpaint"
        features = scope["checkpoint_features"]("selected")
        self.assertTrue(features["inpaint"])
        self.assertFalse(features["sdxl"])
        self.assertTrue(features["control"])
        self.assertIn("Canny", scope["control_types"](features))
        self.assertTrue(scope["input_context"](True, "None", "Img2img", "selected")[2]["visible"])
        models.detect_checkpoint_base.return_value = "Flux.1 D"
        self.assertEqual(
            scope["input_context"](True, "None", "Img2img", "selected")[2],
            {"visible": False, "value": False},
        )
        models.detect_checkpoint_base.return_value = "FluxInpaint"
        for selection, scale, target in (
            ("Remove background", "Off", "Generated image"),
            ("None", "2× General (RealESRGAN)", "Input image only"),
        ):
            self.assertEqual(
                scope["input_context"](True, selection, "Img2img", "selected", None, scale, target)[
                    2
                ],
                {"visible": False, "value": False},
            )


if __name__ == "__main__":
    unittest.main()
