"""Pinned Cascade control/vision nodes and production geometry on guarded CPU fixtures."""

import ast
from concurrent.futures import CancelledError
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
from pydantic import ValidationError

from ruinedfooocus.features.generation import cascade_guidance as guide
from ruinedfooocus.features.generation import image_preparation as preparation
from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY, capture_size_plan, SizePlan
from ruinedfooocus.features.models.model_identity import file_signature
from test_image_preparation import Manager
from test_native_variant_defaults import write_header
from test_qwen_image21_conditioning import native_class, native_function
from test_ordered_native_references import process_nodes


def control_header():
    return {
        "backbone.0.weight": [16384, 16, 1, 1],
        "backbone.2.weight": [1024, 16384, 1, 1],
        "backbone.10.blocks.0.weight": [1024],
        "backbone.11.weight": [1280, 1024, 1, 1],
        **{f"projections.{index}.2.weight": [2048, 1280, 1, 1] for index in range(8)},
    }


def vision_header():
    return {
        "vision_model.embeddings.patch_embedding.weight": [1024, 3, 14, 14],
        "vision_model.embeddings.position_embedding.weight": [257, 1024],
        "vision_model.encoder.layers.0.layer_norm1.weight": [1024],
        "vision_model.encoder.layers.23.layer_norm1.weight": [1024],
        "vision_model.encoder.layers.22.layer_norm1.weight": [1024],
        "visual_projection.weight": [768, 1024],
    }


class T2IAdapter:
    channels_in = 16
    compression_ratio = 1

    def __init__(self):
        self.t2i_model = NS(proj_blocks=[0, 4, 8, 12, 51, 55, 59, 63])
        self.hint = None

    def copy(self):
        return T2IAdapter()

    def set_cond_hint(self, hint, strength, interval, **kwargs):
        self.hint, self.strength, self.interval = hint, strength, interval
        return self

    def set_previous_controlnet(self, previous):
        self.previous = previous


def runtime():
    values = native_function("node_helpers.py", "conditioning_set_values", {})
    node_scope = dict(node_helpers=NS(conditioning_set_values=values))
    control = native_class(
        "nodes.py", "ControlNetApplyAdvanced", node_scope, methods=("apply_controlnet",)
    )
    unclip = native_class("nodes.py", "unCLIPConditioning", node_scope, methods=("apply_adm",))
    encode = native_class("nodes.py", "CLIPVisionEncode", {}, methods=("encode",))
    sr = native_class(
        "comfy_extras/nodes_stable_cascade.py",
        "StableCascade_SuperResolutionControlnet",
        dict(torch=torch, io=NS(NodeOutput=lambda *args: args)),
        methods=("execute",),
    )
    model = native_class(
        "comfy/model_base.py",
        "StableCascade_C",
        dict(
            torch=torch,
            comfy=NS(
                conds=NS(CONDRegular=lambda tensor: tensor, CONDCrossAttn=lambda tensor: tensor)
            ),
        ),
        methods=("extra_conds",),
    )
    return {
        "nodes": NS(
            ControlNetApplyAdvanced=control, unCLIPConditioning=unclip, CLIPVisionEncode=encode
        ),
        "comfy_extras.nodes_stable_cascade": NS(StableCascade_SuperResolutionControlnet=sr),
    }, model


class CascadeGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        for category in ("checkpoints", "controlnet", "clip_vision"):
            (self.root / category).mkdir()
        self.primary = write_header(
            self.root / "checkpoints/stage_c.safetensors",
            {
                "conditioner.embedders.0.fixture": [1],
                "first_stage_model.fixture": [1],
                "text_encoder.clip_g.text_projection": [1280, 1280],
                "vae.encoder.backbone.1.0.block.0.1.num_batches_tracked": [],
                "model.diffusion_model.clip_txt_mapper.weight": [2048, 1280],
            },
        )
        self.stage_b = write_header(
            self.root / "checkpoints/stage_b.safetensors",
            {
                "text_encoder.clip_g.text_projection": [1280, 1280],
                "vae.vquantizer.codebook.weight": [8192, 4],
            },
        )
        self.control_path = write_header(self.root / "controlnet/sr.safetensors", control_header())
        self.vision_path = write_header(self.root / "clip_vision/L.safetensors", vision_header())
        self.manager = Manager(self.root)
        self.manager.model_paths = dict(embedding_path=[], cache_path=self.root)
        self.models = NS(
            get_file=lambda category, name: self.root / category / name,
            get_models_by_path=lambda *args, **kwargs: dict(baseModel="Stable_Cascade_C"),
            get_model_base=lambda value: value["baseModel"],
            detect_checkpoint_base=lambda path: (
                "Stable_Cascade_B" if Path(path).name == self.stage_b.name else "Stable_Cascade_C"
            ),
        )
        self.data = dict(
            base_model_name=self.primary.name,
            cascade_stage_b=self.stage_b.name,
            cascade_b_steps=2,
            cascade_b_cfg=1.1,
            cascade_b_sampler="euler",
            cascade_b_scheduler="simple",
            cascade_super_resolution=True,
            cascade_super_resolution_model=self.control_path.name,
            cascade_control_strength=0.75,
            cascade_control_start=0.2,
            cascade_control_end=0.8,
            cascade_control_denoise=0.6,
            cascade_vision_enabled=True,
            cascade_vision_model=self.vision_path.name,
            cascade_vision_images=[
                Image.new("RGB", (17, 31), (20, 20, 20)),
                Image.new("RGB", (39, 23), (200, 200, 200)),
            ],
            cascade_vision_strengths=[0.5, -0.25],
            cascade_vision_crop="none",
            prompt="exact prompt",
            input_image=Image.new("RGB", (91, 127)),
            controlnet={"type": "edit"},
            image_resize_mode="pad",
            size_mode="target",
            custom_width=513,
            custom_height=517,
            automatic_upscale="Off",
            positive_prompt="exact prompt",
            negative_prompt="blur",
        )
        self.modules, self.native_model = runtime()
        self.enterContext(
            patch.dict(
                "sys.modules",
                self.modules
                | {"comfy.samplers": NS(KSampler=NS(SAMPLERS=["euler"], SCHEDULERS=["simple"]))},
            )
        )
        self.vae = NS(
            downscale_ratio=32,
            encode=Mock(
                side_effect=lambda pixels: torch.zeros(
                    pixels.shape[0], 16, pixels.shape[1] // 32, pixels.shape[2] // 32
                )
            ),
        )
        self.owner = NS(
            xl_base_patched=NS(
                vae=self.vae,
                unet=NS(model=type("StableCascade_C", (), {"model_config": NS(unet_config={})})()),
            ),
            base_file_signature=file_signature(self.primary),
        )
        self.control = T2IAdapter()
        self.vision = NS(
            encode_image=Mock(
                side_effect=lambda pixels, crop: NS(
                    image_embeds=torch.full((1, 768), float(pixels.mean()))
                )
            )
        )
        self.control_load, self.vision_load = (
            Mock(return_value=self.control),
            Mock(return_value=self.vision),
        )
        self.enterContext(
            patch.dict(
                "sys.modules",
                {
                    "comfy.controlnet": NS(load_controlnet=self.control_load),
                    "comfy.clip_vision": NS(load=self.vision_load),
                },
            )
        )

    def captured(self, **changes):
        data = preparation.capture_image_assets(
            self.data | changes,
            self.manager,
            (self.data | changes)["controlnet"],
            models=self.models,
            offline=True,
        )
        return capture_size_plan(
            data, NS(aspect_ratios={}), family="Stable_Cascade_C", controls=data["controlnet"]
        )

    def test_native_super_resolution_node_composes_twofold_with_exact_target_and_pixel_factor(self):
        from ruinedfooocus.features.image_editing.image_geometry import restore_reference_frame

        for selection, factor in (("Off", 1), ("2× General (RealESRGAN)", 2)):
            if factor > 1:
                (self.root / "upscalers").mkdir(exist_ok=True)
                (self.root / "upscalers/RealESRGAN_x2plus.pth").write_bytes(
                    b"owned inert pixel model"
                )
            data = self.captured(automatic_upscale=selection)
            size = data[SIZE_KEY]
            self.assertEqual(size.scale, 2 * factor)
            pixels, frame, latents = guide.super_resolution_source(
                self.vae, data["input_image"], data, mode="pad"
            )
            self.assertEqual(
                tuple(latents.stage_c["samples"].shape[-2:]),
                tuple(side // 16 for side in pixels.shape[1:3]),
            )
            self.assertEqual(
                tuple(latents.stage_b["samples"].shape[-2:]),
                tuple(side // 2 for side in pixels.shape[1:3]),
            )
            self.assertEqual(
                tuple(latents.control_hint.shape[1:3]),
                tuple(side // 32 for side in pixels.shape[1:3]),
            )
            output = torch.zeros(1, frame["canvas_height"], frame["canvas_width"], 3)
            output = (
                restore_reference_frame(output, frame)
                .repeat_interleave(factor, 1)
                .repeat_interleave(factor, 2)
            )
            output = size.finish(output)
            self.assertEqual(tuple(output.shape[1:3]), (517, 513))
            self.assertEqual(data["native_cascade"]["native_source_scale"], 2)
            imported = dict(
                prompt="literal",
                size_plan=size.metadata(),
                cascade_super_resolution=True,
                automatic_upscale=selection,
            )
            replay = capture_size_plan(
                dict(prompt=json.dumps(imported)), None, family="Stable_Cascade_C"
            )
            self.assertEqual(replay[SIZE_KEY], size)

    def test_preserve_source_returns_natural_twofold_without_exposing_alignment_padding(self):
        from ruinedfooocus.features.image_editing.image_geometry import restore_reference_frame

        data = self.captured(image_resize_mode="preserve")
        self.assertNotIn(SIZE_KEY, data)
        _, frame, _ = guide.super_resolution_source(
            self.vae, data["input_image"], data, mode="preserve"
        )
        output = restore_reference_frame(
            torch.zeros(1, frame["canvas_height"], frame["canvas_width"], 3), frame
        )
        self.assertEqual(tuple(output.shape[1:3]), (254, 182))

    def test_native_conditioning_preserves_prompts_and_ordered_vision_weights_under_captured_leases(
        self,
    ):
        data = self.captured()
        _, _, latents = guide.super_resolution_source(
            self.vae, data["input_image"], data, mode="pad"
        )
        original = [
            ["exact prompt", {"pooled_output": torch.zeros(1, 1280), "timestep_start": 0.6}]
        ]
        negative = [["blur", {"pooled_output": torch.zeros(1, 1280)}]]
        positive, uncond = guide.apply_guidance(
            self.owner, original, negative, data, self.manager, latents.control_hint, lambda: None
        )
        self.assertEqual(positive[0][0], "exact prompt")
        self.assertNotIn("control", original[0][1])
        self.assertNotIn("unclip_conditioning", original[0][1])
        active = positive[0][1]["control"]
        self.assertIsNot(active, self.control)
        self.assertIsNone(self.control.hint)
        self.assertEqual((active.strength, active.interval), (0.75, (0.2, 0.8)))
        self.assertEqual(
            tuple(active.hint.shape[1:]),
            (16, latents.control_hint.shape[1], latents.control_hint.shape[2]),
        )
        options = positive[0][1]
        actual = self.native_model().extra_conds(**options)
        self.assertEqual(tuple(actual["clip_img"].shape), (1, 2, 768))
        self.assertAlmostEqual(float(actual["clip_img"][0, 0, 0]), (20 / 255) * 0.5, places=6)
        self.assertAlmostEqual(float(actual["clip_img"][0, 1, 0]), (200 / 255) * -0.25, places=6)
        self.assertNotIn("unclip_conditioning", uncond[0][1])
        self.assertTrue(
            all(call.kwargs["crop"] is False for call in self.vision.encode_image.call_args_list)
        )
        guide.apply_guidance(
            self.owner, original, negative, data, self.manager, latents.control_hint, lambda: None
        )
        self.control_load.assert_called_once()
        self.vision_load.assert_called_once()
        self.assertEqual(
            [row["strength"] for row in data["native_cascade_guidance"]["vision"]], [0.5, -0.25]
        )

    def test_vision_images_are_frozen_and_zero_strength_keeps_its_position_without_encoding(self):
        data = self.captured(
            cascade_vision_strengths=[0, 0.5],
            cascade_super_resolution=False,
            input_image=None,
            controlnet={},
        )
        data["cascade_vision_images"][1].paste((0, 0, 0), (0, 0, 39, 23))
        original = [["literal", {"pooled_output": torch.zeros(1, 1280)}]]
        positive, _ = guide.apply_guidance(
            self.owner, original, original, data, self.manager, None, lambda: None
        )
        self.vision.encode_image.assert_called_once()
        self.assertAlmostEqual(
            float(
                positive[0][1]["unclip_conditioning"][0]["clip_vision_output"].image_embeds.mean()
            ),
            200 / 255,
            places=6,
        )
        self.assertEqual(
            [row["index"] for row in data["native_cascade_guidance"]["vision"]], [1, 2]
        )
        self.control_load.assert_not_called()

    def test_component_shape_role_model_pair_and_runtime_projection_mismatches_are_rejected(self):
        self.assertTrue(guide.compatible_file("control", self.control_path))
        self.assertFalse(guide.compatible_file("vision", self.control_path))
        self.assertEqual(guide.local_choices(self.manager, "control"), [self.control_path.name])
        write_header(
            self.primary,
            {
                "conditioner.embedders.0.fixture": [1],
                "first_stage_model.fixture": [1],
                "text_encoder.clip_g.text_projection": [1280, 1280],
                "vae.encoder.backbone.1.0.block.0.1.num_batches_tracked": [],
                "model.diffusion_model.clip_txt_mapper.weight": [1536, 1280],
            },
        )
        with self.assertRaisesRegex(ValueError, "full 2048"):
            self.captured()
        data = self.captured(cascade_super_resolution=False, input_image=None, controlnet={})
        self.owner.base_file_signature = file_signature(self.primary)
        self.vision.encode_image.return_value = NS(image_embeds=torch.zeros(1, 1024))
        self.vision.encode_image.side_effect = None
        with self.assertRaisesRegex(ValueError, "768-wide"):
            guide.apply_guidance(
                self.owner, [["x", {}]], [["", {}]], data, self.manager, None, lambda: None
            )

    def test_cancel_and_source_replacement_release_load_scope_without_publishing_metadata(self):
        data = self.captured()
        with self.assertRaises(CancelledError):
            guide.apply_guidance(
                self.owner,
                [["x", {}]],
                [["", {}]],
                data,
                self.manager,
                torch.zeros(1, 2, 2, 16),
                Mock(side_effect=CancelledError()),
            )
        self.control_load.assert_not_called()
        self.assertNotIn("native_cascade_guidance", data)
        self.control_path.write_bytes(b"replaced source")
        with self.assertRaisesRegex(ValueError, "changed after submission"):
            guide.apply_guidance(
                self.owner,
                [["x", {}]],
                [["", {}]],
                data,
                self.manager,
                torch.zeros(1, 2, 2, 16),
                lambda: None,
            )

    def test_changed_stage_or_vision_settings_cannot_reuse_the_captured_request(self):
        data = self.captured()
        for changed in (
            dict(cascade_super_resolution=False),
            dict(cascade_control_strength=0),
            dict(cascade_control_denoise=0),
            dict(cascade_vision_enabled=False),
            dict(cascade_vision_strengths=[0, 1]),
            dict(cascade_vision_crop="center"),
        ):
            with (
                self.subTest(changed=changed),
                self.assertRaisesRegex(ValueError, "settings changed after capture"),
            ):
                guide.require_plan(self.owner, data | changed)
        self.control_load.assert_not_called()
        self.vision.encode_image.assert_not_called()

    def test_pinned_native_control_reader_selects_the_large_effnet_adapter_contract(self):
        constructor = Mock(
            side_effect=lambda **kwargs: NS(
                input_channels=kwargs["c_in"], load_state_dict=Mock(return_value=([], []))
            )
        )
        wrapper = Mock(return_value="native adapter")
        reader = native_function(
            "comfy/controlnet.py",
            "load_t2i_adapter",
            dict(
                comfy=NS(ldm=NS(cascade=NS(controlnet=NS(ControlNet=constructor)))),
                T2IAdapter=wrapper,
            ),
        )
        data = {key: NS(shape=value) for key, value in control_header().items()}
        self.assertEqual(reader(data), "native adapter")
        self.assertEqual(
            constructor.call_args.kwargs,
            dict(c_in=16, bottleneck_mode="large", proj_blocks=[0, 4, 8, 12, 51, 55, 59, 63]),
        )
        self.assertEqual(wrapper.call_args.args[1:], (16, 1, "nearest-exact"))

    def test_pinned_native_vision_reader_uses_the_admitted_l14_dispatch_keys(self):
        clip = NS(load_sd=lambda sd: ([], list(sd)))
        constructor = Mock(return_value=clip)
        reader = native_function(
            "comfy/clip_vision.py",
            "load_clipvision_from_sd",
            dict(
                os=os,
                __file__="owned-native-clip_vision.py",
                ClipVisionModel=constructor,
                comfy=NS(storage=NS(state_dict_fast_disk=lambda sd: None)),
            ),
        )
        for positions, config in (
            (257, "clip_vision_config_vitl.json"),
            (577, "clip_vision_config_vitl_336.json"),
        ):
            header = vision_header()
            header["vision_model.embeddings.position_embedding.weight"] = [positions, 1024]
            write_header(self.vision_path, header)
            self.assertTrue(guide.compatible_file("vision", self.vision_path))
            self.assertIs(reader({name: NS(shape=shape) for name, shape in header.items()}), clip)
            self.assertEqual(Path(constructor.call_args.args[0]).name, config)
            del header["vision_model.encoder.layers.0.layer_norm1.weight"]
            write_header(self.vision_path, header)
            self.assertFalse(guide.compatible_file("vision", self.vision_path))
        for last in (30, 47):
            header = vision_header() | {
                f"vision_model.encoder.layers.{last}.layer_norm1.weight": [1280]
            }
            write_header(self.vision_path, header)
            self.assertFalse(guide.compatible_file("vision", self.vision_path))

    def test_production_read_decorator_holds_all_captured_sources_and_releases_on_cancel(self):
        from contextlib import contextmanager

        data = self.captured()
        held, entered = [], []

        @contextmanager
        def lease(data, role, name, manager, category):
            entered.append(role)
            held.append(role)
            try:
                yield self.root / category / name
            finally:
                held.remove(role)

        @guide.stage_read_leases
        def process(owner, gen_data=None, callback=None):
            self.assertEqual(set(held), {"cascade-primary", "cascade-control", "cascade-vision"})
            _, _, latents = guide.super_resolution_source(
                owner.xl_base_patched.vae, gen_data["input_image"], gen_data, mode="pad"
            )
            guide.apply_guidance(
                owner,
                [["literal", {}]],
                [["", {}]],
                gen_data,
                self.manager,
                latents.control_hint,
                lambda: None,
            )
            self.assertEqual(len(entered), 3)
            raise CancelledError()

        with (
            patch.dict("sys.modules", {"ruinedfooocus.app.state": NS(path_manager=self.manager)}),
            patch.object(preparation, "prepared_asset", lease),
        ):
            with self.assertRaises(CancelledError):
                process(self.owner, gen_data=data)
        self.assertFalse(held)
        self.assertIsNone(guide._read_paths.get())

    def test_api_metadata_replay_and_utility_boundary_keep_typed_opt_ins(self):
        from ruinedfooocus.integrations.openai_api import ImageRequest
        from ruinedfooocus.features.gallery.image_metadata import render_values

        request = ImageRequest(
            prompt="literal",
            cascade_super_resolution=True,
            input_image="source",
            cascade_super_resolution_model=self.control_path.name,
            cascade_control_strength=0,
            cascade_vision_enabled=True,
            cascade_vision_model=self.vision_path.name,
            cascade_vision_images=["encoded"],
            cascade_vision_strengths=[-0.25],
        )
        self.assertEqual(request.cascade_control_strength, 0)
        with self.assertRaises(ValidationError):
            ImageRequest(prompt="x", cascade_vision_strengths=[float("nan")])
        metadata = dict(
            prompt="literal",
            cascade_super_resolution=True,
            cascade_vision_enabled=True,
            cascade_vision_strengths=[0, -0.25],
        )
        self.assertEqual(render_values(metadata)["cascade_vision_strengths"], [0, -0.25])
        replay = preparation.image_control_data(dict(prompt=json.dumps(metadata)))
        self.assertTrue(replay["cascade_super_resolution"])
        ignored = preparation.omit_generation_conditioning(
            self.data | {guide.KEY: object(), "native_cascade_guidance": {"old": True}}
        )
        self.assertFalse(
            set((*guide.FIELDS, guide.IMAGE_FIELD, guide.KEY, "native_cascade_guidance"))
            & ignored.keys()
        )

    def test_public_api_captures_native_twofold_target_and_separate_ordered_vision_images(self):
        from test_api_image_edits import queued_data

        captured = []

        def assets(data, *_args, **_kwargs):
            captured.append(dict(data))
            return dict(data)

        options = dict(
            cascade_super_resolution=True,
            cascade_super_resolution_model="sr.safetensors",
            cascade_control_strength=0,
            cascade_vision_enabled=True,
            cascade_vision_model="L.safetensors",
            cascade_vision_images=["first", "second"],
            cascade_vision_strengths=[0.5, -0.25],
            size="513x517",
        )
        with patch.object(preparation, "capture_image_assets", side_effect=assets):
            data = queued_data(
                "Stable_Cascade_C",
                source_image=Image.new("RGB", (91, 127)),
                payload_options=options,
            )
        self.assertEqual(data[SIZE_KEY].target, (513, 517))
        self.assertEqual(data[SIZE_KEY].scale, 2)
        self.assertEqual(data[SIZE_KEY].alignment, 32)
        self.assertEqual(data["cascade_control_strength"], 0)
        self.assertEqual(len(captured[0]["cascade_vision_images"]), 2)
        self.assertTrue(
            all(isinstance(image, Image.Image) for image in data["cascade_vision_images"])
        )
        self.assertEqual(data["cascade_vision_strengths"], [0.5, -0.25])
        with patch.object(preparation, "capture_image_assets", side_effect=assets):
            only_vision = queued_data(
                "Stable_Cascade_C",
                payload_options=dict(
                    options, input_image=None, prompt="", cascade_super_resolution=False
                ),
            )
        self.assertEqual(only_vision["prompt"], "")
        self.assertTrue(only_vision["cascade_vision_enabled"])

    def test_invalid_active_controls_fail_before_model_or_image_encoding(self):
        for values in (
            dict(cascade_control_start=0.8, cascade_control_end=0.8),
            dict(cascade_vision_strengths=[1]),
            dict(cascade_control_strength=float("nan")),
            dict(cascade_super_resolution_model="../sr.safetensors"),
            dict(cascade_vision_images=[]),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.captured(**values)
        self.control_load.assert_not_called()
        self.vision.encode_image.assert_not_called()

    def test_real_production_source_conditioning_and_denoise_hooks_consume_the_plan(self):
        data = self.captured()
        body = process_nodes()
        preprocess = next(
            node
            for node in body
            if isinstance(node, ast.If) and "super_resolution_source" in ast.unparse(node)
        )
        apply = next(
            node
            for node in body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "gen_data.get('_cascade_guidance') is not None"
        )
        latent = next(
            node
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "cascade_latents is not None"
            and any(
                isinstance(child, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "latent"
                    for target in child.targets
                )
                for child in node.body
            )
        )
        scope = dict(
            self=self.owner,
            gen_data=data,
            qwen_reference_edit=False,
            named_source=False,
            named_source_latent=None,
            cascade_model=True,
            input_image=data["input_image"],
            input_images=0,
            guidance=dict(image_resize_mode="pad"),
            cascade_latents=None,
            switched_prompt=[],
            positive_conditioning=[["literal", {"pooled_output": torch.zeros(1, 1280)}]],
            negative_conditioning=[["blur", {}]],
            path_manager=self.manager,
            worker=NS(check_interrupt=Mock()),
        )
        exec(
            compile(
                ast.Module(body=[preprocess, apply, latent], type_ignores=[]),
                "production Cascade advanced caller",
                "exec",
            ),
            scope,
        )
        self.assertEqual(scope["denoise"], 0.6)
        self.assertIs(scope["latent"], scope["cascade_latents"].stage_c)
        self.assertIn("control", scope["positive_conditioning"][0][1])
        self.assertIn("unclip_conditioning", scope["positive_conditioning"][0][1])
        self.assertEqual(scope["input_images"], 0)


if __name__ == "__main__":
    unittest.main()
