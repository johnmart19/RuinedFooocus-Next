"""Public API, saved intent and actual pipeline callers for named native patches."""

import ast
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

from ruinedfooocus.core.preparation import PreparationBusy
from ruinedfooocus.core.settings_schema import validate_known
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.generation.defaults import generation_defaults
from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY, capture_size_plan
from ruinedfooocus.features.image_editing import model_patches, native_control_recipes
from ruinedfooocus.integrations.openai_api import ImageRequest
import test_api_native_inpaint as api_fixture
import test_gallery_actions as gallery_fixture
import test_native_control_recipes as control_fixture
import test_native_model_patches as patch_fixture

ROOT = Path(__file__).resolve().parents[1]


def process_body():
    tree = ast.parse(
        (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(encoding="utf-8")
    )
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "process"
    ).body


def execute(nodes, scope):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "production P5 caller", "exec"), scope)


class NativePatchIntegrationTests(unittest.TestCase):
    def fixture(self, cls):
        fixture = cls()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def api(self):
        fixture = self.fixture(api_fixture.NativeInpaintApiTests)
        files = self.fixture(patch_fixture.NativeModelPatchTests)
        state = sys.modules["ruinedfooocus.app.state"]
        state.path_manager.get_model_folders = lambda category: [files.root / category]
        return fixture, files

    def post(self, fixture, **values):
        return fixture.client.post(
            "/images/generations", json=dict(prompt="owned native patch", size="512x512") | values
        )

    def test_all_four_named_patch_recipes_reach_real_api_capture_with_only_their_exact_assets(self):
        api, files = self.api()
        guide = api_fixture.inline(api_fixture.image_bytes(files.guide))
        for recipe, (family, _) in model_patches.RECIPES.items():
            with self.subTest(recipe=recipe):
                api.selected["family"] = family
                response = self.post(
                    api,
                    model_patch_recipe=recipe,
                    model_patch_model=recipe + ".safetensors",
                    model_patch_vision="vision.safetensors" if recipe == "uso-style" else "",
                    model_patch_image=guide,
                )
                self.assertEqual(response.status_code, 200, response.text)
                data = api.prepared[-1]
                captured = data[model_patches.KEY]
                self.assertEqual(captured.recipe, recipe)
                self.assertEqual(data["image_engine"], "native")
                self.assertIsNone(data["input_image"])
                self.assertIsNone(data["reference_adapter_image"])
                self.assertEqual(data["model_patch_image"].getpixel((0, 0)), (255, 0, 0))
                self.assertEqual(len(captured.artifacts), 2 if recipe == "uso-style" else 1)
                self.assertTrue(
                    all(item in data[assets.KEY].artifacts for item in captured.artifacts)
                )

    def test_api_fun_source_mask_denoise_and_frame_are_captured_without_an_img2img_upload(self):
        api, files = self.api()
        api.selected["family"] = "ZImage"
        source = Image.new("RGBA", (91, 127), (20, 40, 60, 127))
        source_bytes = api_fixture.image_bytes(source)
        mask = api_fixture.inline(api_fixture.image_bytes(Image.new("L", source.size, 255)))
        response = self.post(
            api,
            model_patch_recipe="zimage-fun",
            model_patch_model="zimage-fun.safetensors",
            model_patch_inpaint_image=api_fixture.inline(source_bytes),
            model_patch_mask=mask,
            model_patch_denoise=0.63,
            image_resize_mode="preserve",
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = api.prepared[-1]
        self.assertEqual(data["model_patch_denoise"], 0.63)
        self.assertEqual(data["model_patch_inpaint_image"].size, source.size)
        self.assertEqual(data["model_patch_mask"].getextrema(), (255, 255))
        self.assertNotIn(SIZE_KEY, data)
        self.assertEqual(api_fixture.image_bytes(source), source_bytes)
        self.assertIsNone(data["input_image"])

    def test_api_union_and_alimama_admission_are_cold_safe_and_capture_the_local_control_file(self):
        api, files = self.api()
        control = self.fixture(control_fixture.NativeControlRecipeTests)
        folder = files.root / "controlnet"
        folder.mkdir()
        for name, path in (("union", control.union), ("alimama", control.alimama)):
            (folder / (name + ".safetensors")).write_bytes(path.read_bytes())
        api.selected["family"] = "SDXL"
        with patch.dict(sys.modules, {"nodes": None, "comfy.cldm.control_types": None}):
            response = self.post(
                api,
                input_image=api_fixture.inline(api.source),
                controlnet=dict(type="control", model="union.safetensors"),
                control_recipe="union",
                control_union_type="depth",
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(api.prepared[-1][native_control_recipes.KEY].union_type, "depth")
        api.selected["family"] = "QwenImage"
        with (
            patch.dict(sys.modules, {"nodes": None}),
            patch(
                "ruinedfooocus.features.image_editing.inpaint_utils.prepare_outpaint"
            ) as native_pad,
        ):
            response = self.post(
                api,
                size="102x134",
                controlnet=dict(type="control", model="alimama.safetensors"),
                control_recipe="alimama-inpaint",
                control_inpaint_image=api_fixture.inline(api.source),
                control_inpaint_mode="outpaint",
                control_outpaint_left=7,
                control_outpaint_right=4,
                control_outpaint_top=2,
                control_outpaint_bottom=5,
                control_inpaint_denoise=0.8,
            )
            native_pad.assert_not_called()
        self.assertEqual(response.status_code, 200, response.text)
        data = api.prepared[-1]
        self.assertEqual(data[SIZE_KEY].target, (102, 134))
        self.assertEqual(data["control_inpaint_denoise"], 0.8)
        self.assertNotIn(native_control_recipes.SOURCE_KEY, data)
        self.assertIsNone(data["input_image"])
        self.assertIn("controlnet", [item.role for item in data[assets.KEY].artifacts])

    def test_api_rejects_incompatible_patch_role_and_alimama_size_before_submission(self):
        api, files = self.api()
        api.selected["family"] = "Anima"
        response = self.post(
            api,
            model_patch_recipe="anima-lllite",
            model_patch_model="zimage-fun.safetensors",
            model_patch_image=api_fixture.inline(api.source),
        )
        self.assertEqual(response.status_code, 400, response.text)
        api.submit.assert_not_called()
        response = self.post(
            api,
            controlnet=dict(type="control", model="missing.safetensors"),
            control_recipe="alimama-inpaint",
            control_inpaint_mode="outpaint",
            control_inpaint_image=api_fixture.inline(api.source),
            control_outpaint_left=8,
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("canvas extension produces", response.text)
        api.submit.assert_not_called()

    def test_typed_api_rejects_mixed_missing_and_non_native_source_contracts(self):
        self.assertEqual(
            ImageRequest.model_json_schema()["properties"]["control_union_type"]["enum"],
            list(native_control_recipes.union_types()),
        )
        for values in (
            dict(model_patch_recipe="anima-lllite"),
            dict(model_patch_recipe="zimage-fun", model_patch_inpaint_image="image"),
            dict(
                model_patch_recipe="uso-style", model_patch_image="image", model_patch_strength=0.5
            ),
            dict(
                model_patch_recipe="anima-lllite", model_patch_image="image", control_recipe="union"
            ),
            dict(model_patch_image="image"),
            dict(control_recipe="union", controlnet=dict(type="img2img")),
            dict(
                control_recipe="alimama-inpaint",
                controlnet=dict(type="control"),
                control_inpaint_image="image",
            ),
            dict(
                control_recipe="alimama-inpaint",
                controlnet=dict(type="control"),
                control_inpaint_image="image",
                control_inpaint_mode="outpaint",
            ),
            dict(control_inpaint_image="image"),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ImageRequest(**values)

    def test_defaults_metadata_and_explicit_full_setup_restore_settings_but_never_uploads(self):
        saved = dict(model_patches.DEFAULTS, **native_control_recipes.DEFAULTS)
        saved.update(
            model_patch_recipe="zimage-fun",
            model_patch_model="zimage-fun.safetensors",
            model_patch_denoise=0.61,
            control_inpaint_denoise=0.82,
        )
        values = generation_defaults(
            dict(saved, model_patch_image=object(), control_inpaint_image=object())
        )
        for key in saved:
            self.assertEqual(values[key], saved[key])
        for key in (*model_patches.IMAGE_FIELDS, *native_control_recipes.IMAGE_FIELDS):
            self.assertNotIn(key, values)
        from ruinedfooocus.features.gallery.image_metadata import render_values

        metadata = dict(
            saved,
            prompt="literal",
            model_patch_image="forged.png",
            control_inpaint_image="forged.png",
        )
        rendered = render_values(metadata)
        self.assertEqual({key: rendered[key] for key in saved}, saved)
        self.assertNotIn("model_patch_image", rendered)
        gallery = self.fixture(gallery_fixture.GalleryActionsTests)
        scope = gallery_fixture.functions()
        scope["apply_metadata"] = Mock(return_value={})
        shared = NS(models=gallery.models, settings=NS(default_settings={}))
        metadata.update(base_model_name="new/model.safetensors", loras=[])
        with patch.dict(
            sys.modules,
            {
                "ruinedfooocus.app": NS(state=shared),
                "ruinedfooocus.core.util": NS(
                    get_lora_thumbnail=lambda name: "thumbnail", load_keywords=lambda name: ""
                ),
            },
        ):
            updates = scope["reuse_setup"](metadata)
        self.assertEqual(updates["model_patch_recipe"]["value"], "zimage-fun")
        self.assertEqual(updates["model_patch_denoise"]["value"], 0.61)
        for key in (*model_patches.IMAGE_FIELDS, *native_control_recipes.IMAGE_FIELDS):
            self.assertIsNone(updates[key]["value"])

    def test_profile_validation_rejects_bad_relative_files_intervals_and_canvas_numbers(self):
        for values in (
            dict(model_patch_model="../outside.safetensors"),
            dict(model_patch_stop=0),
            dict(model_patch_recipe="uso-style", model_patch_start=0.1),
            dict(control_inpaint_denoise=False),
            dict(control_outpaint_left=1.5),
            dict(control_outpaint_feather=129),
            dict(control_recipe="union", control_union_type="unknown"),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_known(values)

    def test_alimama_interval_is_rejected_before_native_load_or_queue_submission(self):
        api, files = self.api()
        api.selected["family"] = "QwenImage"
        fixture = self.fixture(control_fixture.NativeControlRecipeTests)
        folder = files.root / "controlnet"
        folder.mkdir()
        (folder / "alimama.safetensors").write_bytes(fixture.alimama.read_bytes())
        response = self.post(
            api,
            controlnet=dict(type="control", model="alimama.safetensors", start=0.9, stop=0.2),
            control_recipe="alimama-inpaint",
            control_inpaint_image=api_fixture.inline(api.source),
            control_inpaint_mask=api_fixture.inline(
                api_fixture.image_bytes(Image.new("L", api.size, 255))
            ),
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("start", response.text)
        api.submit.assert_not_called()

    def test_production_sampling_scope_uses_native_clone_and_holds_asset_lease_for_the_sampler(
        self,
    ):
        files = self.fixture(patch_fixture.NativeModelPatchTests)
        data = files.capture()
        data.update(steps=8)
        owner, original = files.owner("anima-lllite")
        owner.xl_base_patched.vae = NS()
        observed = []

        def sample(**values):
            observed.append(values["model"])
            self.assertIsNot(values["model"], original)
            with (
                self.assertRaises(PreparationBusy),
                files.service.artifact_lease(
                    files.root / "model_patches/anima-lllite.safetensors",
                    write=True,
                    blocking=False,
                ),
            ):
                self.fail("sampling lost its patch read lease")
            return "owned samples"

        body = next(
            node
            for node in process_body()
            if isinstance(node, ast.With)
            and any(
                "model_patches.apply_scope" in ast.unparse(item.context_expr) for item in node.items
            )
        )
        scope = dict(
            model_patches=model_patches,
            auxiliary_scope=nullcontext,
            patch_scope=None,
            self=owner,
            gen_data=data,
            sampling_model=original,
            path_manager=files.manager,
            worker=NS(check_interrupt=lambda _: None),
            native_plan=None,
            architecture="Anima",
            sampler_name="euler",
            scheduler="simple",
            noise=None,
            cfg=1,
            positive_cond=[],
            negative_conditioning=[],
            latent_image=None,
            denoise=1,
            noise_options=None,
            fixed_sigmas=None,
            noise_mask=None,
            callback_function=None,
            seed=1,
            comfy=NS(sample=NS(sample=sample), utils=NS(PROGRESS_BAR_ENABLED=False)),
        )
        with files.native():
            execute([body], scope)
        self.assertEqual(scope["samples"], "owned samples")
        self.assertEqual(len(observed), 1)
        self.assertEqual(original.patches, {"existing": "preserved"})

    def test_production_source_and_latent_callers_keep_alimama_noise_mask_and_denoise(self):
        fixture = self.fixture(control_fixture.NativeControlRecipeTests)
        with fixture.native():
            data = fixture.capture("alimama-inpaint", control_inpaint_denoise=0.72)
            data.update(width=32, height=32)
            owner = fixture.owner(data)
            owner.xl_base_patched.vae = fixture.vae
            body = process_body()
            source = next(
                node
                for node in body
                if isinstance(node, ast.If) and ast.unparse(node.test) == "named_source"
            )
            scope = dict(
                named_source=True,
                named_source_latent=None,
                tensor_latent=None,
                native_control_recipes=native_control_recipes,
                model_patches=model_patches,
                gen_data=data,
                self=owner,
                guidance=dict(image_resize_mode="pad"),
                worker=NS(check_interrupt=lambda _: None),
            )
            execute([source], scope)
            latent = next(
                node
                for node in body
                if isinstance(node, ast.If)
                and ast.unparse(node.test) == "tensor_latent is not None"
            )
            execute([latent], scope)
            self.assertEqual(scope["denoise"], 0.72)
            self.assertTrue(
                torch.equal(scope["latent"]["noise_mask"][:, 0], scope["named_source_mask"])
            )
            self.assertEqual(tuple(scope["input_image"].shape), (1, 32, 32, 3))
            self.assertEqual(scope["reference_frame"]["mode"], "pad")

    def test_pure_outpaint_admission_precedes_compute_and_preserve_sources_skip_stale_target_plans(
        self,
    ):
        fixture = self.fixture(control_fixture.NativeControlRecipeTests)
        data = dict(
            control_recipe="alimama-inpaint",
            control_inpaint_mode="outpaint",
            control_inpaint_image=fixture.image,
            control_outpaint_left=7,
            controlnet=dict(type="control"),
        )
        with (
            patch.dict(sys.modules, {"nodes": None}),
            patch("ruinedfooocus.features.image_editing.inpaint_utils.prepare_outpaint") as pad,
        ):
            result = native_control_recipes.capture_source_size(data)
            self.assertEqual(result[SIZE_KEY].target, (24, 19))
            pad.assert_not_called()
        for fields, controls in (
            (dict(model_patch_recipe="zimage-fun", model_patch_inpaint_image=fixture.image), {}),
            (
                dict(control_recipe="alimama-inpaint", control_inpaint_image=fixture.image),
                dict(type="control"),
            ),
        ):
            result = capture_size_plan(
                dict(
                    fields,
                    size_mode="target",
                    image_resize_mode="preserve",
                    custom_width=1024,
                    custom_height=1024,
                ),
                NS(aspect_ratios={}),
                family="QwenImage",
                controls=controls,
            )
            self.assertNotIn(SIZE_KEY, result)

    def test_production_alimama_finishes_and_blends_its_full_canvas_only_once(self):
        fixture = self.fixture(control_fixture.NativeControlRecipeTests)
        with fixture.native():
            data = fixture.capture(
                "alimama-inpaint",
                control_inpaint_mode="outpaint",
                control_outpaint_left=7,
                control_outpaint_top=2,
                control_outpaint_right=4,
                control_outpaint_bottom=5,
                control_outpaint_feather=0,
            )
            latent, pixels, mask, frame = native_control_recipes.source_latent(
                fixture.vae, data, (1024, 1024), "preserve", lambda: None
            )
            self.assertNotIn("noise_mask", latent)
            body = process_body()
            finish = next(
                node
                for node in body
                if isinstance(node, ast.If)
                and ast.unparse(node.test) == "native_control_recipes.has_source(gen_data)"
            )
            index = body.index(finish)
            generic_restore = Mock(side_effect=AssertionError("source was restored twice"))
            scope = dict(
                native_control_recipes=native_control_recipes,
                gen_data=data,
                reference_frame=frame,
                decoded_latent=torch.zeros_like(pixels),
                outpaint_composite=None,
                restore_reference_frame=generic_restore,
            )
            execute(body[index : index + 2], scope)
            result = scope["decoded_latent"]
            self.assertEqual(tuple(result.shape), (1, 26, 28, 3))
            self.assertTrue(torch.equal(result[0, 2:21, 7:24, 0], torch.ones((19, 17))))
            self.assertEqual((data["width"], data["height"]), (28, 26))
            generic_restore.assert_not_called()
            from ruinedfooocus.features.generation.automatic_upscale import upscale_decoded

            self.assertTrue(torch.equal(upscale_decoded(NS(), result, data), result))


if __name__ == "__main__":
    unittest.main()
