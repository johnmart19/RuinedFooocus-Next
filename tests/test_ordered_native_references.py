"""Pinned Qwen/LongCat/Flux2 reference nodes with ordered weightless CPU fixtures."""

import math
import asyncio
import ast
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
import gradio as gr
from gradio.state_holder import SessionState

from ruinedfooocus.features.image_editing.ordered_native_references import (
    OrderedNativeReferences,
    flux2_sigmas,
    recipe_for,
    reference_family,
)
from test_qwen_image21_conditioning import native_class, native_function

ROOT = Path(__file__).resolve().parents[1]


def process_nodes():
    tree = ast.parse(
        (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(encoding="utf-8")
    )
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "process"
    ).body


def runtime():
    native = NS(
        utils=NS(
            common_upscale=native_function("comfy/utils.py", "common_upscale", dict(torch=torch))
        )
    )
    values = native_function("node_helpers.py", "conditioning_set_values", {})
    scope = dict(
        torch=torch,
        math=math,
        comfy=native,
        node_helpers=NS(conditioning_set_values=values),
        io=NS(NodeOutput=lambda *values: values),
    )
    qwen = native_class(
        "comfy_extras/nodes_qwen.py", "TextEncodeQwenImageEdit", scope, methods=("execute",)
    )
    plus = native_class(
        "comfy_extras/nodes_qwen.py", "TextEncodeQwenImageEditPlus", scope, methods=("execute",)
    )
    method = native_class(
        "comfy_extras/nodes_flux.py",
        "FluxKontextMultiReferenceLatentMethod",
        scope,
        methods=("execute",),
    )
    references = native_class(
        "comfy_extras/nodes_edit_model.py", "ReferenceMultiLatent", scope, methods=("execute",)
    )
    for name in ("generalized_time_snr_shift", "compute_empirical_mu", "get_schedule"):
        native_function("comfy_extras/nodes_flux.py", name, scope)
    scheduler = native_class(
        "comfy_extras/nodes_flux.py", "Flux2Scheduler", scope, methods=("execute",)
    )
    return {
        "node_helpers": NS(conditioning_set_values=values),
        "comfy_extras.nodes_qwen": NS(
            TextEncodeQwenImageEdit=qwen, TextEncodeQwenImageEditPlus=plus
        ),
        "comfy_extras.nodes_flux": NS(
            FluxKontextMultiReferenceLatentMethod=method, Flux2Scheduler=scheduler
        ),
        "comfy_extras.nodes_edit_model": NS(ReferenceMultiLatent=references),
    }


def clip_fixture():
    return NS(
        tokenize=Mock(side_effect=lambda text, **kwargs: dict(text=text, **kwargs)),
        encode_from_tokens_scheduled=Mock(
            side_effect=lambda tokens: [[tokens["text"], {"native": True}]]
        ),
    )


def vae_fixture(channels=16, scale=8):
    vae = NS(latent_channels=channels, spacial_compression_encode=lambda: scale)
    vae.encode = Mock(
        side_effect=lambda pixels: torch.full(
            (1, channels, pixels.shape[1] // scale, pixels.shape[2] // scale), float(pixels.mean())
        )
    )
    return vae


class OrderedReferenceTests(unittest.TestCase):
    def setUp(self):
        self.modules = runtime()
        self.enterContext(patch.dict("sys.modules", self.modules))

    def test_source_variants_select_native_recipes_without_name_guesses(self):
        for source in (
            None,
            {},
            dict(native_family="QwenImage", behavioral_variant="base"),
            dict(native_family="QwenImage", behavioral_variant="unknown"),
        ):
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(ValueError, "matching Edit recipe"),
            ):
                recipe_for("QwenImage", source=source)
        for variant, expected in (
            ("edit-2509", "qwen-edit-2509"),
            ("edit-2511", "qwen-edit-2511"),
            ("edit", "qwen-edit"),
        ):
            source = dict(native_family="QwenImage", behavioral_variant=variant)
            self.assertEqual(recipe_for("QwenImage", source=source), expected)
        with self.assertRaisesRegex(ValueError, "LongCat Image Edit checkpoint"):
            recipe_for("LongCatImage")
        self.assertEqual(recipe_for("LongCatImage", "longcat-edit"), "longcat-edit")
        with self.assertRaisesRegex(ValueError, "artifact"):
            recipe_for(
                "LongCatImage",
                "longcat-edit",
                dict(native_family="LongCatImage", behavioral_variant="base"),
            )
        with self.assertRaisesRegex(ValueError, "family"):
            recipe_for("SDXL", "qwen-edit-2511")
        self.assertEqual(
            recipe_for("Flux2Klein4B", source=dict(native_family="Flux2Klein4B")), "flux2-reference"
        )

    def test_reference_cache_seals_preprocessing_and_owned_source_versions(self):
        with torch.inference_mode():
            source = torch.full((1, 16, 16, 3), 0.2)
            references = OrderedNativeReferences("qwen-edit", vae_fixture(), (source,))
            references.conditioning(clip_fixture(), "edit", "")
            references.images[0].fill_(0.8)
            with self.assertRaisesRegex(ValueError, "changed after submission"):
                references.conditioning(clip_fixture(), "new prompt", "")

    def test_qwen_edit_plus_uses_ordered_three_images_and_caches_native_vae_across_prompts(self):
        images = tuple(torch.full((1, 16, 16, 3), value) for value in (0.1, 0.3, 0.8))
        vae, clip = vae_fixture(), clip_fixture()
        references = OrderedNativeReferences("qwen-edit-2511", vae, images)
        images[0].fill_(0.9)  # Submission-owned tensors must be independent.
        positive, negative = references.conditioning(clip, "swap <image1> with <image3>", "")
        changed, _ = references.conditioning(clip, "a later scheduled prompt", "")
        self.assertEqual(vae.encode.call_count, 3)
        for conditioning in (positive, negative, changed):
            self.assertEqual(conditioning[0][1]["reference_latents_method"], "index_timestep_zero")
            self.assertEqual(len(conditioning[0][1]["reference_latents"]), 3)
            self.assertAlmostEqual(
                float(conditioning[0][1]["reference_latents"][0].mean()), 0.1, places=5
            )
            self.assertAlmostEqual(
                float(conditioning[0][1]["reference_latents"][2].mean()), 0.8, places=5
            )
        tokens = clip.tokenize.call_args_list[0]
        self.assertTrue(tokens.args[0].startswith("Picture 1:"))
        self.assertIn("Picture 3:", tokens.args[0])
        self.assertEqual(len(tokens.kwargs["images"]), 3)
        for call in clip.tokenize.call_args_list:
            self.assertAlmostEqual(float(call.kwargs["images"][0].mean()), 0.1, places=5)
            self.assertAlmostEqual(float(call.kwargs["images"][2].mean()), 0.8, places=5)
        self.assertEqual([row["index"] for row in references.provenance()["references"]], [1, 2, 3])
        self.assertFalse(references.provenance()["artifact_qualified"])

    def test_2509_keeps_native_default_reference_method_and_longcat_sets_index(self):
        image = torch.full((1, 16, 24, 3), 0.5)
        for recipe, method in (("qwen-edit-2509", None), ("longcat-edit", "index")):
            with self.subTest(recipe=recipe):
                clip = clip_fixture()
                positive, negative = OrderedNativeReferences(
                    recipe, vae_fixture(), (image,)
                ).conditioning(clip, "change daylight", "")
                self.assertEqual(positive[0][1].get("reference_latents_method"), method)
                self.assertEqual(negative[0][1].get("reference_latents_method"), method)
                if recipe == "longcat-edit":
                    self.assertEqual(clip.tokenize.call_args_list[0].args[0], "change daylight")
                    self.assertEqual(len(clip.tokenize.call_args_list[0].kwargs["images"]), 1)

    def test_flux2_native_references_keep_upload_order_scheduled_metadata_and_zero_target(self):
        images = tuple(torch.full((1, 32, 48, 3), index / 20) for index in range(1, 17))
        references = OrderedNativeReferences("flux2-reference", vae_fixture(128, 16), images)
        positive = [
            ["early", {"start_percent": 0, "end_percent": 0.5}],
            ["late", {"start_percent": 0.5}],
        ]
        negative = [["negative", {}]]
        conditioned, unconditioned, latent = references.attach(positive, negative)
        self.assertEqual([row[0] for row in conditioned], ["early", "late"])
        self.assertNotIn("reference_latents", positive[0][1])
        self.assertEqual(conditioned[0][1]["end_percent"], 0.5)
        self.assertEqual(len(unconditioned[0][1]["reference_latents"]), 16)
        self.assertLess(
            float(conditioned[0][1]["reference_latents"][0].mean()),
            float(conditioned[0][1]["reference_latents"][15].mean()),
        )
        self.assertEqual(tuple(latent["samples"].shape), (1, 128, 2, 3))
        self.assertEqual(int(torch.count_nonzero(latent["samples"])), 0)

    def test_reference_bounds_and_wrong_vae_fail_before_encoding(self):
        image = torch.zeros((1, 16, 16, 3))
        for recipe, images, channels in (
            ("qwen-edit-2511", (image,) * 4, 16),
            ("longcat-edit", (image,) * 2, 16),
            ("flux2-reference", (image,), 16),
        ):
            vae = vae_fixture(channels)
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                OrderedNativeReferences(recipe, vae, images)
            vae.encode.assert_not_called()

    def test_flux2_resolution_schedule_uses_native_sigmas_and_native_denoise_slice(self):
        native = self.modules["comfy_extras.nodes_flux"].Flux2Scheduler
        sigmas = flux2_sigmas(20, 1024, 768)
        torch.testing.assert_close(sigmas, native.execute(20, 1024, 768)[0])
        self.assertEqual(len(sigmas), 21)
        self.assertEqual(float(sigmas[-1]), 0)
        self.assertFalse(torch.equal(sigmas, flux2_sigmas(20, 1536, 1536)))
        torch.testing.assert_close(
            flux2_sigmas(4, 1024, 1024, 0.5), native.execute(8, 1024, 1024)[0][-5:]
        )
        for args in (
            (0, 1024, 1024),
            (4, 1025, 1024),
            (4, 1024, 1024, 0),
            (200, 1024, 1024, 0.001),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                flux2_sigmas(*args)

    def test_contextual_recipe_ui_preserves_valid_choice_and_omits_inactive_job_fields(self):
        import ruinedfooocus.app as app_package
        from ruinedfooocus.features.image_editing import ui_ordered_references as ui

        registered = {}
        shared = NS(
            settings=NS(default_settings={}),
            add_ctrl=lambda key, obj: registered.update({key: obj}),
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(app_package, "state", shared, create=True),
            patch(
                "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                side_effect=lambda name: {"qwen": "QwenImage", "longcat": "LongCatImage"}[name],
            ),
        ):
            with gr.Blocks(analytics_enabled=False) as app:
                checkpoint = gr.Textbox(value="qwen")
                tool = gr.Textbox(value="Native image edit")
                kind = gr.Textbox(value="edit")
                ui.create_ui(checkpoint, tool, kind, None, lambda tool, kind: kind)
            from gradio.i18n import I18nData
            from ruinedfooocus.core.translation_manager import catalogue

            label = next(
                label
                for label, value in registered["native_edit_recipe"].choices
                if value == "qwen-edit-2511"
            )
            english = (
                catalogue().native().translations["en"][label.key]
                if isinstance(label, I18nData)
                else label
            )
            self.assertEqual(english, "Qwen Edit Plus 2511: up to three references")
            index = next(key for key, fn in app.fns.items() if fn.fn.__name__ == "context")
            result = asyncio.run(
                app.process_api(
                    index,
                    ["qwen", "Native image edit", "edit", "qwen-edit-2511"],
                    state=SessionState(app),
                )
            )["data"]
            self.assertTrue(result[0]["visible"])
            self.assertEqual(result[1]["value"], "qwen-edit-2511")
            switched = asyncio.run(
                app.process_api(
                    index,
                    ["longcat", "Native image edit", "edit", "qwen-edit-2511"],
                    state=SessionState(app),
                )
            )["data"]
            self.assertEqual(switched[1]["value"], "auto")
        request = dict(native_edit_recipe="qwen-edit-2511")
        self.assertNotIn(
            "native_edit_recipe", ui.omit_inactive_recipe(request, {"type": "upscale"})
        )
        self.assertEqual(request["native_edit_recipe"], "qwen-edit-2511")
        retained = dict(request, reference_images=["retained upload"])
        self.assertEqual(
            ui.omit_inactive_recipe(retained, {"type": "img2img"}, "QwenImage")["reference_images"],
            [],
        )
        self.assertEqual(retained["reference_images"], ["retained upload"])

    def test_production_paired_encoder_caller_records_order_and_keeps_scheduled_native_images(self):
        from PIL import Image
        from ruinedfooocus.features.image_editing.ordered_native_references import _image

        statements = process_nodes()
        prepare = next(
            node
            for node in statements
            if isinstance(node, ast.If) and ast.unparse(node.test) == "ordered_recipe is not None"
        )
        encode = next(
            node
            for node in statements
            if isinstance(node, ast.If)
            and ast.unparse(node.test).startswith(
                "ordered_references is not None and ordered_recipe !="
            )
        )
        vae, clip = vae_fixture(), clip_fixture()
        owner = NS(xl_base_patched=NS(vae=vae, clip=clip))
        scope = dict(
            ordered_recipe="qwen-edit-2511",
            ordered_source=None,
            ordered_references=None,
            OrderedNativeReferences=OrderedNativeReferences,
            self=owner,
            input_image=torch.full((1, 32, 32, 3), 0.2),
            additional_references=[Image.new("RGB", (32, 32), (100, 150, 200))],
            reference_tensor=lambda image, *_: _image(image),
            gen_data=dict(width=32, height=32, steps=4),
            clip_skip=1,
            positive_prompt="[red|blue]",
            negative_prompt="",
            clean_prompt_cond_caches=lambda: {
                key: {"cache": None, "text": None} for key in ("+", "-", "switch")
            },
        )
        exec(
            compile(
                ast.Module(body=[prepare, encode], type_ignores=[]),
                "production ordered native edit caller",
                "exec",
            ),
            scope,
        )
        self.assertEqual(vae.encode.call_count, 2)
        self.assertEqual(
            [
                row[0].split("Picture 2:")[-1].split("<|vision_end|>")[-1]
                for row in scope["switched_prompt"]
            ],
            ["red", "blue", "red", "blue"],
        )
        self.assertTrue(
            all(len(row[1]["reference_latents"]) == 2 for row in scope["switched_prompt"])
        )
        self.assertEqual(scope["gen_data"]["native_edit_recipe"], "qwen-edit-2511")
        self.assertEqual(
            [
                row["index"]
                for row in scope["gen_data"]["native_reference_conditioning"]["references"]
            ],
            [1, 2],
        )

    def test_runtime_klein_variant_preserves_sampler_cfg_in_production_guidance_caller(self):
        node = next(
            node
            for node in process_nodes()
            if isinstance(node, ast.If)
            and "isinstance(self.xl_base.unet.model, Flux)" in ast.unparse(node.test)
        )
        base = type("Flux", (), {})
        model = type(
            "Flux2", (base,), {"model_config": NS(unet_config={"depth_single_blocks": 20})}
        )()
        family = reference_family(model)
        self.assertEqual(family, "Flux2Klein4B")
        scope = dict(
            self=NS(xl_base=NS(unet=NS(model=model))),
            Flux=base,
            architecture="Flux2",
            ordered_family=family,
            native_plan=None,
            controlnet={"type": "edit"},
            cfg=5,
            positive_cond=[["text", {}]],
        )
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]),
                "production native guidance caller",
                "exec",
            ),
            scope,
        )
        self.assertEqual(scope["cfg"], 5)
        self.assertNotIn("guidance", scope["positive_cond"][0][1])

    def test_typed_api_and_request_capture_preserve_native_edit_recipe(self):
        from ruinedfooocus.integrations.openai_api import ImageRequest
        from test_api_image_edits import queued_data

        request = ImageRequest(
            prompt="combine them",
            input_image="fixture",
            reference_images=["a", "b"],
            native_edit_recipe="qwen-edit-2511",
        )
        self.assertEqual(request.native_edit_recipe, "qwen-edit-2511")
        for values in (
            dict(native_edit_recipe="longcat-edit"),
            dict(input_image="fixture", native_edit_recipe="longcat-edit", reference_images=["a"]),
            dict(
                input_image="fixture",
                native_edit_recipe="qwen-edit-2511",
                reference_images=["a"] * 3,
            ),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ImageRequest(**values)
        captured = queued_data(
            "Flux2Klein4B", payload_options={"native_edit_recipe": "flux2-reference"}
        )
        self.assertEqual(captured["controlnet"]["type"], "edit")
        self.assertEqual(captured["native_edit_recipe"], "flux2-reference")
        with self.assertRaisesRegex(ValueError, "native image edit operation"):
            queued_data(
                "QwenImage",
                controlnet={"type": "img2img"},
                payload_options={"reference_images": ["extra"]},
            )


if __name__ == "__main__":
    unittest.main()
