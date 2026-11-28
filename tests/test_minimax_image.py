"""Pinned MiniMax still graph on bounded CPU tensors and inert native headers."""

import ast
from contextlib import contextmanager
from concurrent.futures import CancelledError
import json
import itertools
import math
import pathlib
from pathlib import Path
import sys
import tempfile
import random
import traceback
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from safetensors.numpy import save_file
import torch

from ruinedfooocus.core.preparation import PreparationRequest
from ruinedfooocus.features.generation import minimax_image as image
from ruinedfooocus.features.generation.image_preparation import (
    KEY as ASSETS_KEY,
    ImageArtifact,
    ImageAssets,
    capture_image_assets,
)
from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY, SizePlan
from ruinedfooocus.features.models.component_candidates import classify_shapes, compatibility
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models import native_workflow_admission as admission
from test_native_045_roles import NATIVE, native_sources, minimax_shapes
from test_qwen_image21_conditioning import native_function, native_class


SIGMAS = (1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0)


def runtime():
    # These are exact pinned upstream bodies. Only model computation and the
    # existing hardware sampler boundary are fixtures; no weights are loaded.
    scope = dict(torch=torch, math=math, io=NS(NodeOutput=lambda *values: values))
    resize = native_function("comfy/utils.py", "lanczos", dict(torch=torch, Image=Image, np=np))
    common = native_function("comfy/utils.py", "common_upscale", dict(torch=torch, lanczos=resize))
    comfy = NS(
        utils=NS(common_upscale=common), model_management=NS(intermediate_device=lambda: "cpu")
    )
    scope["comfy"] = comfy
    scope.update(
        CANVAS_MULTIPLE=32,
        IMAGE_REF_SHORT_EDGE=1024,
        IMAGE_REF_VAE_MAX_SHORT_EDGE=2048,
        IMAGE_SIZE=2048,
    )
    for name in ("_resize", "_size_short_edge", "_resize_to", "_image_refs"):
        scope[name] = native_function("comfy_extras/nodes_minimax_h3.py", name, scope)
    setter = native_function("node_helpers.py", "conditioning_set_values", {})
    scope["node_helpers"] = NS(conditioning_set_values=setter)
    conditioning = native_class(
        "comfy_extras/nodes_minimax_h3.py",
        "MiniMaxH3ImageConditioning",
        scope,
        methods=("execute",),
    )
    zero = native_class("nodes.py", "ConditioningZeroOut", dict(torch=torch), methods=("zero_out",))
    return conditioning, zero


class MiniMaxImageTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(native_sources())
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        shape = minimax_shapes()
        shape["final_layer.video_out.weight"] = (96 * 6, 4)
        self.checkpoint = self.file(
            "image.safetensors",
            shape,
            dict(
                config=json.dumps(
                    dict(
                        transformer=dict(
                            pdd_steps=6, pdd_sigmas=SIGMAS, text_dim=5120, latents_dim=24
                        )
                    )
                )
            ),
        )
        self.encoder = self.file(
            "encoder.safetensors",
            {
                "visual.deepstack_merger_list.0.norm.weight": (1,),
                "model.layers.49.self_attn.q_proj.weight": (1, 1),
            },
        )
        self.vae = self.file(
            "vae.safetensors",
            {
                "decoder.proj_out.weight": (768, 1),
                "decoder.x_embedder.weight": (1024, 1),
                "post_quant_conv.weight": (24, 24, 1, 1),
                "encoder.conv_in.weight": (1,),
                "decoder.transformer_blocks.0.scale1": (1,),
            },
        )

    def file(self, name, shapes, metadata=None):
        path = self.root / name
        save_file(
            {name: np.zeros(shape, dtype=np.float16) for name, shape in shapes.items()},
            path,
            metadata=metadata,
        )
        return path

    def request(self, **values):
        data = dict(
            base_model_name=self.checkpoint.name,
            prompt="A patterned still image",
            negative="",
            controlnet={},
            steps=6,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            clip_skip=1,
            seed=42,
            width=65,
            height=97,
            image_number=1,
        )
        data.update(values)
        return image.capture(data, self.checkpoint)

    def assets(self):
        return ImageAssets(
            tuple(
                ImageArtifact(role, name, PreparationRequest("", path), tuple(file_signature(path)))
                for role, name, path in (
                    ("minimax_model", self.checkpoint.name, self.checkpoint),
                    ("clip", image.CLIP_ROLE, self.encoder),
                    ("vae", image.VAE_ROLE, self.vae),
                )
            ),
            True,
        )

    def test_positive_original_pdd_contract_and_missing_metadata_or_video_refusal(self):
        _, steps, sigmas = image.checkpoint_contract(self.checkpoint)
        self.assertEqual((steps, sigmas), (6, SIGMAS))
        with self.assertRaisesRegex(ValueError, "PDD"):
            image.checkpoint_contract(self.file("missing.safetensors", minimax_shapes()))
        with self.assertRaisesRegex(ValueError, "audio"):
            image.checkpoint_contract(self.file("video.safetensors", minimax_shapes(audio=True)))
        for invalid in (
            (1, 0.5, 0.6, 0.4, 0.2, 0.1, 0),
            (1, 0.8, 0.6, 0.4, 0.2, 0.1, float("nan")),
            (1, 0),
        ):
            with self.subTest(sigmas=invalid), self.assertRaisesRegex(ValueError, "PDD"):
                image.checkpoint_contract(
                    self.file(
                        "bad.safetensors",
                        minimax_shapes(),
                        dict(
                            config=json.dumps(
                                dict(transformer=dict(pdd_steps=6, pdd_sigmas=invalid))
                            )
                        ),
                    )
                )

    def test_component_roles_identify_native_32b_and_image_codec_not_optional_prompt_helper(self):
        from ruinedfooocus.features.models.component_candidates import header_shapes

        image.validate_components(self.encoder, self.vae)
        self.assertEqual(
            compatibility(image.CLIP_ROLE, classify_shapes(header_shapes(self.encoder))),
            "compatible",
        )
        self.assertEqual(
            compatibility(image.VAE_ROLE, classify_shapes(header_shapes(self.vae))), "compatible"
        )
        for shapes in (
            {"model.layers.0.post_attention_layernorm.weight": (4096,)},
            {
                "mask_decoder.query_builder.content_proj.weight": (1, 1),
                "model.visual.merger.linear_fc2.weight": (2048, 1),
            },
        ):
            with self.subTest(shapes=shapes), self.assertRaisesRegex(ValueError, "32B"):
                image.validate_components(self.file("wrong-encoder.safetensors", shapes), self.vae)
        with self.assertRaisesRegex(ValueError, "image VAE"):
            image.validate_components(
                self.encoder,
                self.file(
                    "wrong-codec.safetensors",
                    {"decoder.proj_out.weight": (768, 1), "decoder.x_embedder.weight": (2048, 1)},
                ),
            )

    def test_admission_requires_captured_plan_prepared_components_and_unchanged_model(self):
        with self.assertRaisesRegex(ValueError, "Capture"):
            admission.require_available_image_workflow(self.checkpoint)
        data = self.request()
        data[ASSETS_KEY] = self.assets()
        admission.require_available_image_workflow(
            self.checkpoint, prepared_data=data, loading=True
        )
        data[ASSETS_KEY] = ImageAssets((), True)
        with self.assertRaisesRegex(ValueError, "components"):
            admission.require_available_image_workflow(
                self.checkpoint, prepared_data=data, loading=True
            )
        self.checkpoint.write_bytes(self.checkpoint.read_bytes() + b"changed")
        with self.assertRaises((ValueError, OSError)):
            image.require_plan(data, self.checkpoint)

    def test_reference_capture_is_immutable_alpha_explicit_and_replay_requires_original_order(self):
        original = Image.new("RGBA", (65, 97), (255, 0, 0, 0))
        original.putpixel((64, 96), (0, 255, 0, 255))
        second = Image.new("RGBA", (32, 40), (0, 0, 255, 255))
        data = self.request(
            input_image=original, reference_images=[second], controlnet=dict(type="edit")
        )
        plan = data[image.KEY]
        pixels = plan.references[0].tensor()
        torch.testing.assert_close(pixels[0, 0, 0], torch.full((3,), 0.5))
        torch.testing.assert_close(pixels[0, -1, -1], torch.tensor([0.0, 1.0, 0.0]))
        original.paste((255, 255, 255, 255), (0, 0, 65, 97))
        self.assertEqual(plan.references[0].rgba[-4:], bytes([0, 255, 0, 255]))
        self.assertEqual(image.capture(data, self.checkpoint)[image.KEY], plan)
        with self.assertRaisesRegex(ValueError, "original ordered"):
            self.request(
                input_image=second,
                reference_images=[Image.frombytes("RGBA", (65, 97), plan.references[0].rgba)],
                native_minimax_image=plan.metadata(),
            )

    def test_incompatible_masks_overrides_schedules_and_reference_budgets_fail_before_runtime(self):
        for values in (
            dict(inpaint_toggle=True),
            dict(controlnet=dict(type="img2img")),
            dict(steps=20),
            dict(cfg=4),
            dict(native_export_latent=True),
            dict(loras=[dict(name="patch", weight=1)]),
            dict(model_patch_recipe="uso-style"),
            dict(negative="blurry"),
            dict(image_number=0),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.request(**values)
        with self.assertRaisesRegex(ValueError, "four"):
            self.request(
                input_image=Image.new("RGB", (32, 32)),
                reference_images=[Image.new("RGB", (32, 32))] * 4,
            )

    def test_actual_native_conditioning_zeroout_keeps_reference_order_tags_and_24_channel_t1(self):
        conditioning, zero = runtime()
        seen = []
        clip = NS(
            tokenize=lambda prompt, **kw: seen.append((prompt, kw)) or prompt,
            encode_from_tokens_scheduled=lambda tokens: [
                [torch.ones(1, 3, 5120), dict(minimax_token_tags=torch.tensor([0, 1, 1]))]
            ],
        )
        vae = NS(
            encode=lambda pixels: torch.full(
                (1, 24, 1, pixels.shape[1] // 16, pixels.shape[2] // 16), float(pixels[0, 0, 0, 0])
            )
        )
        primary = torch.zeros(1, 1024, 1024, 3)
        second = torch.ones_like(primary)
        positive, latent = conditioning.execute(
            clip,
            "Use <Picture 1> and <Picture 2>",
            96,
            128,
            vae=vae,
            images={"image_2": second, "image_1": primary},
        )
        self.assertEqual(tuple(latent["samples"].shape), (1, 24, 1, 8, 6))
        self.assertEqual(seen[0][1]["picture_label"], "Picture %d: ")
        self.assertEqual(
            [float(ref["latent"].mean()) for ref in positive[0][1]["minimax_refs"]], [0.0, 1.0]
        )
        negative = zero().zero_out(positive)[0]
        self.assertFalse(torch.any(negative[0][0]))
        self.assertIs(negative[0][1]["minimax_refs"], positive[0][1]["minimax_refs"])

    def test_actual_pdd_sigma_class_and_stock_simple_schedule_preserve_native_grid(self):
        path = NATIVE / "comfy/model_sampling.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {"ModelSamplingDiscreteFlow", "CONST", "time_snr_shift"}
        selected = [
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
        ]
        scope = dict(torch=torch, math=math)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
        config = NS(unet_config=dict(pdd_sigmas=SIGMAS), sampling_settings={})
        scope["comfy"] = NS(
            model_sampling=NS(
                ModelSamplingDiscreteFlow=scope["ModelSamplingDiscreteFlow"], CONST=scope["CONST"]
            )
        )
        tree = ast.parse((NATIVE / "comfy/ldm/minimax/sampling.py").read_text(encoding="utf-8"))
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3ImageModelSampling"
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        sampler = scope["MiniMaxH3ImageModelSampling"](config)
        simple = native_function("comfy/samplers.py", "simple_scheduler", dict(torch=torch))
        actual = simple(sampler, 6)
        torch.testing.assert_close(actual, torch.tensor(SIGMAS), atol=0.002, rtol=0)
        self.assertEqual(sampler.percent_to_sigma(0), 1)
        self.assertEqual(sampler.percent_to_sigma(1), 0)

    def execute(self, *, edit=False, cancel=False):
        conditioning, zero = runtime()
        source = Image.new("RGBA", (65, 97), (255, 20, 10, 128)) if edit else None
        data = self.request(input_image=source)
        data[ASSETS_KEY] = self.assets()
        data[SIZE_KEY] = SizePlan.create((65, 97))
        clip = type("MiniMaxH3TEModel", (), {})()
        clip.tokenize = lambda prompt, **kwargs: prompt
        clip.encode_from_tokens_scheduled = lambda tokens: [
            [torch.ones(1, 2, 5120), dict(minimax_token_tags=torch.ones(2, dtype=torch.int64))]
        ]
        vae = type("MiniMaxH3ImageVAE", (), {})()
        vae = NS(
            first_stage_model=vae,
            latent_channels=24,
            spacial_compression_encode=lambda: 16,
            spacial_compression_decode=lambda: 16,
            encode=lambda pixels: torch.zeros(
                1, 24, 1, pixels.shape[1] // 16, pixels.shape[2] // 16
            ),
        )
        model = type(image.FAMILY, (), {})()
        model.model_config = NS(
            unet_config=dict(pdd_sigmas=list(SIGMAS), pdd_steps=6, latents_dim=24)
        )
        model.model_sampling = type("MiniMaxH3ImageModelSampling", (), {})()
        owner = NS(
            xl_base_patched=NS(
                unet=NS(model=model),
                clip=NS(
                    cond_stage_model=clip,
                    tokenize=clip.tokenize,
                    encode_from_tokens_scheduled=clip.encode_from_tokens_scheduled,
                ),
                vae=vae,
            )
        )

        def decode(latent, **kwargs):
            self.assertEqual(tuple(latent["samples"].shape), (1, 24, 1, 8, 6))
            return torch.arange(128 * 96 * 3, dtype=torch.float32).reshape(1, 128, 96, 3) / (
                128 * 96 * 3
            )

        owner.decode_latent = Mock(side_effect=decode)
        check = Mock(side_effect=CancelledError if cancel else None)
        sample = Mock(
            side_effect=lambda *args, **kwargs: kwargs["callback"](0, None, None, 6) or args[8]
        )
        comfy = NS(
            sample=NS(sample=sample, prepare_noise=lambda latent, seed: torch.zeros_like(latent)),
            utils=NS(
                ProgressBar=lambda steps: NS(update_absolute=Mock()), PROGRESS_BAR_ENABLED=False
            ),
        )
        modules = {
            "comfy": comfy,
            "comfy.sample": comfy.sample,
            "comfy.utils": comfy.utils,
            "nodes": NS(ConditioningZeroOut=zero),
            "comfy_extras.nodes_minimax_h3": NS(MiniMaxH3ImageConditioning=conditioning),
        }
        callback = Mock()
        with patch.dict(sys.modules, modules):
            if cancel:
                with self.assertRaises(CancelledError):
                    image.process(owner, data, callback, check)
                sample.assert_not_called()
                owner.decode_latent.assert_not_called()
                return
            result = image.process(owner, data, callback, check)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape[:2], (97, 65))
        self.assertEqual(data["size_plan"]["canvas_width"], 96)
        self.assertEqual(data["size_plan"]["target_width"], 65)
        self.assertEqual(sample.call_args.args[2:6], (6, 1.0, "euler", "simple"))
        self.assertEqual(sample.call_args.kwargs["denoise"], 1)
        self.assertNotIn("sigmas", sample.call_args.kwargs)
        callback.assert_called_once()
        self.assertEqual(data["native_minimax_image"]["final_height"], 97)

    def test_t2i_actual_native_caller_and_ordinary_odd_target_output(self):
        self.execute()

    def test_edit_actual_native_caller_and_original_alpha_policy(self):
        self.execute(edit=True)

    def test_cancel_before_conditioning_never_samples_or_decodes(self):
        self.execute(cancel=True)

    def test_actual_api_captures_existing_typed_t2i_and_rgba_edit_inputs(self):
        import ruinedfooocus.app
        from ruinedfooocus.app import api_runtime as api
        from ruinedfooocus.integrations.openai_api import ImageRequest

        values = {
            image.CLIP_ROLE: self.encoder.name,
            image.VAE_ROLE: self.vae.name,
            "image_engine": "auto",
            "resolution": "default",
        }
        manager = NS(
            paths={
                "path_clip": [str(self.root)],
                "path_vae": [str(self.root)],
                "path_checkpoints": [str(self.root)],
            },
            get_abspath=Path,
            DOWNLOADABLE_FILES={},
            get_model_folders=lambda category: [self.root],
            model_paths={"cache_path": self.root / "cache"},
        )
        models = NS(
            get_models_by_path=lambda *args, **kwargs: {"baseModel": image.FAMILY},
            get_model_base=lambda value: image.FAMILY,
            get_file=lambda *args: self.checkpoint,
            offline=True,
        )
        perf = NS(
            CUSTOM_PERFORMANCE="Own settings",
            choices_for_model=lambda *args: ["Own settings"],
            selection_for_model=lambda *args: "Own settings",
            get_perf_options=lambda *args: dict(
                custom_steps=6, cfg=1, sampler_name="euler", scheduler="simple", clip_skip=1
            ),
        )
        shared = NS(
            settings=NS(default_settings=values),
            models=models,
            path_manager=manager,
            performance_settings=perf,
            resolution_settings=NS(
                CUSTOM_RESOLUTION="Custom...",
                selection_for_model=lambda *args: "default",
                aspect_ratios={"default": (2048, 2048)},
            ),
        )
        source = Image.new("RGBA", (65, 97), (20, 40, 80, 128))

        def decode(value, **options):
            self.assertTrue(options["preserve_alpha"])
            return source.copy()

        with (
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.state": shared,
                    "comfy.samplers": NS(KSampler=NS(SAMPLERS=["euler"], SCHEDULERS=["simple"])),
                },
            ),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(
                api,
                "image_defaults",
                dict(base_model_name=self.checkpoint.name, performance_selection="Own settings"),
            ),
            patch.object(api, "decode_image", side_effect=decode),
            patch.object(api, "submit", side_effect=lambda run: run),
        ):
            for edit in (False, True):
                saved_context = self.request(input_image=source if edit else None)[
                    image.KEY
                ].metadata()
                payload = ImageRequest(
                    prompt="Native still",
                    size="65x97",
                    steps=6,
                    cfg=1,
                    sampler="euler",
                    scheduler="simple",
                    input_image="upload" if edit else None,
                    minimax_image_context=saved_context,
                ).model_dump(exclude_none=True, exclude_unset=True)
                operation = api.images(
                    payload, dict(name=self.checkpoint.name, family=image.FAMILY)
                )
                closure = dict(
                    zip(
                        operation.__code__.co_freevars,
                        (cell.cell_contents for cell in operation.__closure__),
                    )
                )
                data = closure["data"]
                plan = data[image.KEY]
                self.assertEqual(len(plan.references), int(edit))
                self.assertEqual(data[SIZE_KEY].target, (65, 97))
                self.assertEqual(
                    {(a.role, a.name) for a in data[ASSETS_KEY].artifacts},
                    {
                        ("minimax_model", self.checkpoint.name),
                        ("clip", image.CLIP_ROLE),
                        ("vae", image.VAE_ROLE),
                    },
                )
                if edit:
                    self.assertEqual(plan.references[0].rgba[-1], 128)

    def test_saved_component_provenance_is_checked_before_native_weight_loading(self):
        from dataclasses import replace

        data = self.request()
        data[ASSETS_KEY] = self.assets()
        identities = tuple(
            (item.role, *image.source_identity(item.signature))
            for item in data[ASSETS_KEY].artifacts
        )
        data[image.KEY] = replace(data[image.KEY], replay_components=identities)
        image.validate_prepared(data)
        wrong = tuple(
            (row[0], row[1], row[2] + 1, *row[3:]) if row[0] == "clip" else row
            for row in identities
        )
        data[image.KEY] = replace(data[image.KEY], replay_components=wrong)
        with self.assertRaisesRegex(ValueError, "identities"):
            admission.require_available_image_workflow(
                self.checkpoint, prepared_data=data, loading=True
            )

    def test_actual_pipeline_dispatch_precedes_generic_conditioning_and_uses_job_cancellation(self):
        from test_model_cache import method
        from ruinedfooocus.features.generation.cascade_guidance import stage_read_leases

        data = self.request()
        owner = NS(xl_base_patched=NS(unet=NS(model=type(image.FAMILY, (), {})())))
        check = Mock()
        expected = object()
        process = method(
            "ruinedfooocus/features/generation/image_pipeline.py",
            "process",
            dict(
                torch=torch, worker=NS(check_interrupt=check), stage_read_leases=stage_read_leases
            ),
        )
        with patch.object(image, "process", return_value=expected) as native:
            self.assertIs(process(owner, data), expected)
            self.assertIs(native.call_args.args[0], owner)
            native.call_args.args[3]()
        check.assert_called_once_with(data)

    def test_existing_edit_ui_context_and_metadata_replay_keep_original_sources_explicit(self):
        from ruinedfooocus.features.image_editing.image_guidance import capabilities
        from ruinedfooocus.features.models.image_components import edit_control
        from ruinedfooocus.features.gallery.image_metadata import (
            render_values,
            restore_prompt_provenance,
        )
        from ruinedfooocus.features.generation.image_preparation import image_control_data
        from ruinedfooocus.features.models.resource_defaults import load_family_recipe

        self.assertTrue(capabilities(image.FAMILY)["multiple_refs"])
        self.assertEqual(edit_control(image.FAMILY), {"type": "edit"})
        recipe = load_family_recipe("native-minimax-h3-image")
        self.assertTrue(recipe["applicability"]["auto_recipe_supported"])
        self.assertEqual(recipe["sampling"]["custom_steps"], 6)
        source = Image.new("RGBA", (65, 97), (20, 40, 80, 128))
        captured = self.request(input_image=source)
        saved = dict(prompt="Native still", native_minimax_image=captured[image.KEY].metadata())
        self.assertEqual(
            render_values(saved)["native_minimax_image"], saved["native_minimax_image"]
        )
        replay = image_control_data(dict(prompt=json.dumps(saved)))
        self.assertEqual(replay["native_minimax_image"], saved["native_minimax_image"])
        with self.assertRaisesRegex(ValueError, "original ordered"):
            self.request(native_minimax_image=replay["native_minimax_image"])
        restored = self.request(
            input_image=source, native_minimax_image=replay["native_minimax_image"]
        )
        self.assertEqual(restored[image.KEY].references, captured[image.KEY].references)
        # Library's real Use setup -> append_work handoff passes this imported
        # record through restore_prompt_provenance, with literal prompt fields.
        saved.update(base_model_name=self.checkpoint.name, negative="")
        values = dict(
            prompt=saved["prompt"],
            negative="",
            base_model_name=self.checkpoint.name,
            style_selection=[],
            auto_negative=False,
        )
        restore_prompt_provenance(values, saved)
        self.assertTrue(values["_metadata_prompt"])
        self.assertEqual(values["native_minimax_image"], saved["native_minimax_image"])
        with self.assertRaisesRegex(ValueError, "original ordered"):
            self.request(**values)
        self.assertEqual(
            self.request(input_image=source, **values)[image.KEY].references,
            captured[image.KEY].references,
        )

    def test_utility_capture_drops_hidden_minimax_context_and_reference_bytes(self):
        from ruinedfooocus.features.generation.image_preparation import omit_generation_conditioning

        data = self.request(input_image=Image.new("RGBA", (65, 97)))
        result = omit_generation_conditioning(data)
        self.assertNotIn(image.KEY, result)
        self.assertNotIn("native_minimax_image", result)
        self.assertIn(image.KEY, data)

    def test_actual_worker_exports_native_context_and_completed_one_image_output(self):
        from contextlib import nullcontext
        from PIL.PngImagePlugin import PngInfo
        from ruinedfooocus.core.paths import resource_path
        from ruinedfooocus.core.task_errors import describe_error
        from ruinedfooocus.features.gallery.image_metadata import read_metadata
        from ruinedfooocus.features.generation.image_output_batches import (
            ImageOutputBatch,
            ImageBatchArtifacts,
            validate_artifacts,
        )
        from test_image_preparation import function

        data = self.request()
        data.update(
            task_id=1,
            task_type="tool_call",
            _sampling_captured=True,
            base_model_hash="owned fixture",
            custom_steps=6,
            loras=[],
            aspect_ratios_selection="Custom...",
            custom_width=65,
            custom_height=97,
            style_selection=[],
            silent=True,
            index=(0, 1),
        )
        pipe = NS(
            pipeline_type=["sdxl"],
            load_base_model=Mock(),
            load_loras=Mock(),
            process=lambda gen_data, callback: ImageOutputBatch(
                (np.full((97, 65, 3), 63, dtype=np.uint8),),
                gen_data[image.PLAN_KEY],
                (dict(alpha="opaque-RGB"),),
            ),
        )
        forbidden = Mock(side_effect=AssertionError("uncaptured generation dependency"))
        shared = NS(
            models=NS(get_models_by_path=lambda *args: {}, get_model_base=lambda _: image.FAMILY),
            state={},
            performance_settings=NS(apply=forbidden),
            resolution_settings=NS(CUSTOM_RESOLUTION="Custom..."),
            path_manager=NS(model_paths={"temp_outputs_path": self.root}),
            shared_cache={},
        )
        sequence = itertools.count()
        render = function(
            "_process",
            dict(
                image_stage=lambda fn: fn,
                shared=shared,
                settings=NS(default_settings={}),
                control_tools=NS(get_settings=lambda data: data.get("controlnet") or {}),
                check_interrupt=lambda _: None,
                process_metadata=forbidden,
                pipelines=NS(update=lambda _: pipe),
                get_checkpoint_hashes=forbidden,
                process_prompt=lambda styles, p, n, data: (p, n),
                shift_attention=lambda text, distance: text,
                parse_loras=lambda p, n: ([], p, n),
                merge_loras=lambda *args: [],
                TimeIt=lambda _: nullcontext(),
                resource_path=resource_path,
                random=random,
                Path=Path,
                pathlib=pathlib,
                Image=Image,
                PngInfo=PngInfo,
                json=json,
                traceback=traceback,
                model_management=NS(
                    InterruptProcessingException=type("Cancelled", (Exception,), {})
                ),
                describe_error=describe_error,
                generate_temp_filename=lambda **kwargs: str(
                    self.root / f"out-{next(sequence)}.png"
                ),
            ),
        )
        result = render(data)
        self.assertIsInstance(result, ImageBatchArtifacts)
        self.assertIs(validate_artifacts(data, result), result)
        with Image.open(result[0]) as saved:
            metadata = read_metadata(saved.info["parameters"])
            self.assertEqual(saved.size, (65, 97))
            self.assertEqual(metadata["native_minimax_image"]["pdd_sigmas"], list(SIGMAS))
            self.assertEqual(metadata["native_output_group"]["kind"], "minimax-image")
            self.assertEqual(metadata["prompt"], data["prompt"])
        forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
