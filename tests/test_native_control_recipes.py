"""Pinned Union/AliMama bodies, source latents and full-canvas capture on CPU."""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

from ruinedfooocus.app import resource_lifecycle
from ruinedfooocus.features.image_editing import native_control_recipes as recipes
from ruinedfooocus.features.generation.size_plan import KEY, SizePlan, capture_size_plan
from ruinedfooocus.features.models.model_identity import file_signature
from test_native_model_coverage import NATIVE, native_method
from test_native_variant_defaults import write_header


class Control:
    def __init__(self, modes=8, masked=False):
        self.control_model = NS(num_control_type=modes)
        self.concat_mask = masked
        self.extra = {}

    def copy(self):
        result = Control(self.control_model.num_control_type, self.concat_mask)
        result.extra = dict(self.extra)
        return result

    def set_extra_arg(self, name, value):
        self.extra[name] = value

    def set_cond_hint(self, image, strength, interval, **kwargs):
        self.hint = image
        self.strength, self.interval, self.options = strength, interval, kwargs
        return self

    def set_previous_controlnet(self, previous):
        self.previous = previous


class NativeControlRecipeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.checkpoint = write_header(self.root / "checkpoint.safetensors", {"opaque": [1]})
        self.union = write_header(
            self.root / "union.safetensors",
            {
                "zero_convs.0.0.weight": [320, 320, 1, 1],
                "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": [320, 2048],
                "task_embedding": [8, 320],
            },
        )
        self.alimama = write_header(
            self.root / "alimama.safetensors",
            {
                "controlnet_blocks.0.weight": [3072, 3072],
                "controlnet_x_embedder.weight": [3072, 68],
                "transformer_blocks.0.img_mlp.net.0.proj.weight": [24576, 3072],
            },
        )
        self.family = "SDXL"
        self.models = NS(
            get_file=lambda *_: self.checkpoint,
            get_models_by_path=lambda *_, **__: {},
            get_model_base=lambda _: self.family,
        )
        self.manager = NS(get_model_folders=lambda _: [self.root])
        self.image = Image.new("RGB", (17, 19), "red")
        self.mask = Image.new("L", self.image.size, 255)
        self.vae = NS(
            latent_channels=16,
            output_channels=3,
            spacial_compression_encode=lambda: 8,
            encode=Mock(
                side_effect=lambda pixels: torch.zeros(
                    (1, 16, pixels.shape[1] // 8, pixels.shape[2] // 8)
                )
            ),
        )
        self.types = runpy.run_path(str(NATIVE / "comfy/cldm/control_types.py"))[
            "UNION_CONTROLNET_TYPES"
        ]

    @contextmanager
    def native(self):
        advanced = native_method("nodes.py", "ControlNetApplyAdvanced", "apply_controlnet", {})
        scope = dict(
            UNION_CONTROLNET_TYPES=self.types,
            io=NS(NodeOutput=lambda *values: values),
            nodes=NS(ControlNetApplyAdvanced=advanced),
            comfy=NS(utils=NS(common_upscale=lambda value, width, height, *_: value)),
        )
        union = native_method(
            "comfy_extras/nodes_controlnet.py", "SetUnionControlNetType", "execute", scope
        )
        alimama = native_method(
            "comfy_extras/nodes_controlnet.py", "ControlNetInpaintingAliMamaApply", "execute", scope
        )
        modules = {
            "comfy.cldm.control_types": NS(UNION_CONTROLNET_TYPES=self.types),
            "comfy_extras.nodes_controlnet": NS(
                SetUnionControlNetType=union, ControlNetInpaintingAliMamaApply=alimama
            ),
            "nodes": NS(
                ControlNetApplyAdvanced=advanced,
                ImagePadForOutpaint=native_method(
                    "nodes.py", "ImagePadForOutpaint", "expand_image", dict(torch=torch)
                ),
                VAEEncode=native_method("nodes.py", "VAEEncode", "encode", {}),
                SetLatentNoiseMask=native_method("nodes.py", "SetLatentNoiseMask", "set_mask", {}),
            ),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(resource_lifecycle, "require_compute_owner"),
        ):
            yield

    def capture(self, recipe="union", **updates):
        self.family = "QwenImage" if recipe == "alimama-inpaint" else "SDXL"
        name = "alimama.safetensors" if recipe == "alimama-inpaint" else "union.safetensors"
        controls = dict(type="control", model=name, strength=0.7, start=0.1, stop=0.9)
        data = dict(
            base_model_name="checkpoint",
            control_recipe=recipe,
            control_union_type="auto",
            control_inpaint_image=self.image,
            control_inpaint_mask=self.mask,
            controlnet=controls,
        )
        data.update(updates)
        return recipes.capture(data, self.manager, self.models, controls)

    def owner(self, data):
        capture = data[recipes.KEY]
        config = type("QwenImage", (), {})()
        return NS(
            base_file_signature=capture.checkpoint_signature,
            controlnet_file_signature=capture.control_signature,
            xl_base_patched=NS(unet=NS(model=NS(model_config=config))),
        )

    def test_native_enum_and_explicit_union_modes_use_request_local_control_copies(self):
        with self.native():
            self.assertEqual(recipes.union_types(), ("auto", *self.types))
            data = self.capture(control_union_type="canny/lineart/anime_lineart/mlsd")
            original = Control()
            positive = [[torch.ones(1), {"original": True}]]
            negative = [[torch.zeros(1), {}]]
            p, n = recipes.apply(
                self.owner(data),
                positive,
                negative,
                original,
                self.vae,
                torch.ones((1, 16, 16, 3)),
                data,
                lambda: None,
            )
            self.assertEqual(original.extra, {})
            self.assertEqual(p[0][1]["control"].extra["control_type"], [3])
            self.assertIs(p[0][1]["control"], n[0][1]["control"])
            self.assertNotIn("control", positive[0][1])
            data = self.capture(control_union_type="auto")
            p, _ = recipes.apply(
                self.owner(data),
                positive,
                negative,
                original,
                self.vae,
                torch.ones((1, 16, 16, 3)),
                data,
                lambda: None,
            )
            self.assertEqual(p[0][1]["control"].extra["control_type"], [])

    def test_alimama_admits_qwen_68_reader_and_rejects_absent_mask_or_other_source_recipe(self):
        with self.native():
            data = self.capture("alimama-inpaint")
            self.image.paste("blue", (0, 0, *self.image.size))
            self.assertEqual(data["control_inpaint_image"].getpixel((0, 0)), (255, 0, 0))
            self.assertIsNone(data["input_image"])
            with self.assertRaisesRegex(ValueError, "source and a white"):
                self.capture("alimama-inpaint", control_inpaint_mask=None)
            for fields in (
                {"inpaint_toggle": True},
                {"reference_images": [self.image]},
                {"model_patch_recipe": "qwen-diffsynth"},
            ):
                with self.subTest(fields=fields), self.assertRaises(ValueError):
                    self.capture("alimama-inpaint", _api_job=object(), **fields)

    def test_mask_source_is_encoded_once_and_conditioning_uses_exact_frame_with_native_polarity(
        self,
    ):
        with self.native():
            data = self.capture("alimama-inpaint")
            latent, pixels, mask, frame = recipes.source_latent(
                self.vae, data, (32, 32), "pad", lambda: None
            )
            self.vae.encode.assert_called_once_with(pixels)
            self.assertTrue(torch.equal(latent["noise_mask"][:, 0], mask))
            conditioning = [[torch.ones(1), {}]]
            p, _ = recipes.apply(
                self.owner(data),
                conditioning,
                conditioning,
                Control(masked=True),
                self.vae,
                pixels,
                data,
                lambda: None,
                mask=mask,
            )
            control = p[0][1]["control"]
            self.assertTrue(torch.equal(control.options["extra_concat"][0][:, 0], 1 - mask))
            self.assertTrue(
                torch.equal(control.hint.movedim(1, -1), pixels * (1 - mask).round().unsqueeze(-1))
            )
            self.assertEqual(tuple(recipes.finish(pixels, data, frame).shape), (1, 32, 32, 3))

    def test_outpaint_dedicated_borders_override_stale_size_plan_and_preserve_full_canvas(self):
        with (
            self.native(),
            patch(
                "ruinedfooocus.features.image_editing.inpaint_utils.prepare_outpaint",
                wraps=__import__(
                    "ruinedfooocus.features.image_editing.inpaint_utils",
                    fromlist=["prepare_outpaint"],
                ).prepare_outpaint,
            ) as pad,
        ):
            data = dict(
                control_recipe="alimama-inpaint",
                control_inpaint_mode="outpaint",
                controlnet=dict(
                    type="control", model="alimama.safetensors", strength=0.7, start=0.1, stop=0.9
                ),
                control_inpaint_image=self.image,
                control_outpaint_left=7,
                control_outpaint_top=2,
                control_outpaint_right=4,
                control_outpaint_bottom=5,
                control_outpaint_feather=0,
                outpaint_left=999,
                size_mode="legacy_base",
                custom_width=1024,
                custom_height=1024,
                **{KEY: SizePlan.create((512, 512), 1)},
            )
            prepared = recipes.pre_size_capture(data)
            again = recipes.pre_size_capture(prepared)
            pad.assert_called_once()
            self.assertIs(prepared[recipes.SOURCE_KEY], again[recipes.SOURCE_KEY])
            plan = capture_size_plan(again, NS(aspect_ratios={}), controls={}, family="QwenImage")[
                KEY
            ]
            self.assertEqual(plan.target, (28, 26))
            self.assertEqual(plan.canvas, (32, 32))
            captured = self.capture(
                "alimama-inpaint",
                **{key: value for key, value in again.items() if key != "control_recipe"},
            )
            latent, pixels, mask, frame = recipes.source_latent(
                self.vae, captured, (1024, 1024), "pad", lambda: None
            )
            self.assertNotIn("noise_mask", latent)
            self.assertEqual(tuple(pixels.shape), (1, 32, 32, 3))
            finished = recipes.finish(torch.zeros_like(pixels), captured, frame)
            self.assertEqual(tuple(finished.shape), (1, 26, 28, 3))
            self.assertTrue(torch.equal(finished[0, 2:21, 7:24, 0], torch.ones((19, 17))))
            pad.assert_called_once()

    def test_inactive_ui_recipe_cannot_replace_source_or_frozen_size_and_cold_native_import_is_refused(
        self,
    ):
        stale = dict(
            control_recipe="alimama-inpaint",
            control_inpaint_mode="outpaint",
            input_image=self.image,
            custom_width=512,
            custom_height=768,
            **{KEY: SizePlan.create((512, 768), 1)},
        )
        result = recipes.pre_size_capture(stale, {"type": "img2img"})
        self.assertEqual(result["control_recipe"], "ordinary")
        self.assertIs(result["input_image"], self.image)
        self.assertIs(result[KEY], stale[KEY])
        with self.assertRaisesRegex(ValueError, "matching native ControlNet"):
            recipes.pre_size_capture(dict(stale, _api_job=object()), {"type": "img2img"})
        cold = dict(
            stale,
            controlnet={"type": "control"},
            control_inpaint_image=self.image,
            control_outpaint_left=4,
        )
        with (
            patch.dict(sys.modules, {"nodes": None}),
            self.assertRaisesRegex(RuntimeError, "Initialize the native"),
        ):
            recipes.pre_size_capture(cold)

    def test_signature_cancellation_vae_and_union_table_guards_prevent_application(self):
        with self.native():
            data = self.capture(control_union_type="repaint")
            owner = self.owner(data)
            image = torch.ones((1, 16, 16, 3))
            with self.assertRaisesRegex(ValueError, "matching native Union"):
                recipes.apply(owner, [], [], Control(modes=6), self.vae, image, data, lambda: None)
            owner.controlnet_file_signature = ()
            with self.assertRaisesRegex(ValueError, "another checkpoint"):
                recipes.apply(owner, [], [], Control(), self.vae, image, data, lambda: None)
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                recipes.apply(
                    self.owner(data),
                    [],
                    [],
                    Control(),
                    self.vae,
                    image,
                    data,
                    Mock(side_effect=RuntimeError("cancelled")),
                )
            data = self.capture("alimama-inpaint")
            with self.assertRaisesRegex(ValueError, "16-channel"):
                recipes.source_latent(
                    NS(latent_channels=4), data, (32, 32), "preserve", lambda: None
                )
            self.vae.encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
