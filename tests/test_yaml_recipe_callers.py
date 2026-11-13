"""YAML reaches real capture/load/cache/metadata callers without weight loading."""

import ast
from contextlib import contextmanager
from dataclasses import replace
import math
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from ruinedfooocus.core.preparation import PreparationService
from ruinedfooocus.features.models import yaml_recipes as recipes
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.gallery.image_metadata import render_values
import test_model_cache as cache
from test_model_cache import method
from test_native_model_coverage import NATIVE, PIPELINE, native_method
from test_yaml_recipes import TEXT


def definitions(path, names, scope):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def native_sampling():
    scope = dict(torch=torch, np=np, math=math)
    definitions(
        NATIVE / "comfy/ldm/modules/diffusionmodules/util.py", {"make_beta_schedule"}, scope
    )
    definitions(
        NATIVE / "comfy/model_sampling.py",
        {"reshape_sigma", "EPS", "V_PREDICTION", "ModelSamplingDiscrete"},
        scope,
    )
    sampling = NS(
        **{name: scope[name] for name in ("EPS", "V_PREDICTION", "ModelSamplingDiscrete")}
    )
    node = native_method(
        "comfy_extras/nodes_model_advanced.py",
        "ModelSamplingDiscrete",
        "patch",
        {"comfy": NS(model_sampling=sampling)},
    )
    return sampling, node


class Patcher:
    def __init__(self, config, sampling):
        self.model, self.sampling = NS(model_config=config), sampling

    def clone(self):
        return Patcher(self.model.model_config, self.sampling)

    def add_object_patch(self, key, value):
        assert key == "model_sampling"
        self.sampling = value

    def get_model_object(self, key):
        return self.sampling


class Clip:
    def __init__(self, layer=None):
        self.layer = layer

    def clone(self):
        return Clip(self.layer)

    def clip_layer(self, layer):
        self.layer = layer


class YamlCallerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkpoint = self.root / "checkpoint.safetensors"
        self.checkpoint.write_bytes(b"opaque checkpoint, no load")
        self.recipe = self.root / "chosen.yaml"
        self.recipe.write_text(TEXT, encoding="utf-8")
        self.manager = NS(get_model_folders=lambda category: [self.root], DOWNLOADABLE_FILES={})
        self.models = NS(
            get_file=lambda *a: self.checkpoint,
            get_models_by_path=lambda *a, **kw: {},
            get_model_base=lambda *a: "SDXL",
        )

    def capture(self, **kwargs):
        return recipes.capture_recipe(self.recipe, self.checkpoint, family="SDXL", **kwargs)

    def test_discovery_and_capture_are_explicit_separate_and_signature_bound(self):
        (self.root / "not-a-recipe.bin").write_bytes(b"unread weight")
        with patch.object(Path, "open", side_effect=AssertionError("Discovery parsed a file")):
            choices = recipes.discover_local_recipes(self.manager)
        self.assertEqual([path for _, path in choices], [str(self.recipe)])
        data = recipes.capture_request(
            dict(
                base_model_name="checkpoint",
                model_recipe_path=str(self.recipe),
                clip_layer_source="recipe",
                image_engine="auto",
            ),
            self.models,
            self.manager,
            {},
        )
        self.assertEqual(data["image_engine"], "native")
        self.assertEqual(
            data["model_recipe_overrides"], {"prediction": "v_prediction", "clip_layer_idx": -2}
        )
        self.assertEqual(recipes.captured_recipe(data).recipe_clip_layer, -2)
        self.recipe.write_text(TEXT.replace("-2", "-1"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "saved YAML recipe changed"):
            recipes.capture_request(data, self.models, self.manager, {})

    def test_capture_rejects_escaped_paths_utilities_edits_and_optional_engines(self):
        data = dict(base_model_name="checkpoint", model_recipe_path=str(self.recipe))
        for control in ({"type": "upscale"}, {"type": "edit"}, {"type": "canny"}):
            with self.subTest(control=control), self.assertRaises(ValueError):
                recipes.capture_request(data, self.models, self.manager, control)
        with self.assertRaises(ValueError):
            recipes.capture_request(
                dict(data, image_engine="sdcpp-vulkan"), self.models, self.manager, {}
            )
        with tempfile.TemporaryDirectory() as other:
            escaped = Path(other) / "outside.yaml"
            escaped.write_text(TEXT)
            with self.assertRaisesRegex(ValueError, "outside"):
                recipes.capture_request(
                    dict(data, model_recipe_path=str(escaped)), self.models, self.manager, {}
                )

    def test_capture_rechecks_resolved_containment_after_discovery_retarget(self):
        with tempfile.TemporaryDirectory() as other:
            outside = Path(other) / "retargeted.yaml"
            outside.write_text(TEXT)
            captured = recipes.capture_recipe(outside, self.checkpoint, family="SDXL")
            # Discovery saw the configured spelling; the capture resolves its
            # retargeted junction/symlink after that list was returned.
            with (
                patch.object(
                    recipes, "discover_local_recipes", return_value=[("chosen", str(self.recipe))]
                ),
                patch.object(recipes, "capture_recipe", return_value=captured),
            ):
                with self.assertRaisesRegex(ValueError, "location changed outside"):
                    recipes.capture_request(
                        dict(model_recipe_path=str(self.recipe), base_model_name="checkpoint"),
                        self.models,
                        self.manager,
                        {},
                    )

    def test_metadata_replay_legacy_reset_and_api_intent_do_not_guess_from_one(self):
        legacy = recipes.replay_values(dict(prompt="literal", clip_skip=1))
        self.assertEqual(
            (legacy["clip_layer_source"], legacy["model_recipe_path"]), ("request", "")
        )
        metadata = dict(
            prompt="literal",
            clip_skip=1,
            model_recipe_path=str(self.recipe),
            model_recipe_sha256=self.capture().recipe_sha256,
            clip_layer_source="recipe",
            model_recipe_overrides={"prediction": "v_prediction", "clip_layer_idx": -2},
        )
        replay = recipes.replay_values(render_values(metadata))
        self.assertEqual(replay["clip_layer_source"], "recipe")
        captured = recipes.capture_request(
            dict(replay, base_model_name="checkpoint"), self.models, self.manager, {}
        )
        self.assertEqual(recipes.captured_recipe(captured).recipe_clip_layer, -2)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            recipes.normalize_policy(
                dict(clip_layer_source="recipe", clip_skip=1), explicit_clip_skip=True
            )

    def test_real_asset_capture_preparation_and_cancel_validate_frozen_recipe_pair(self):
        from concurrent.futures import CancelledError

        payload = dict(
            base_model_name="checkpoint",
            model_recipe_path=str(self.recipe),
            clip_layer_source="recipe",
            prompt="literal",
        )
        with patch(
            "ruinedfooocus.features.models.component_profiles.component_inventory",
            return_value={"rows": []},
        ):
            data = assets.capture_image_assets(payload, self.manager, {}, models=self.models)
        self.assertEqual(
            {item.role for item in data[assets.KEY].artifacts}, {"yaml-recipe", "yaml-checkpoint"}
        )
        ready = assets.prepare_image_assets(data, check_cancelled=lambda: None)
        self.assertEqual(recipes.recipe_identity(ready), recipes.recipe_identity(data))
        with self.assertRaises(CancelledError):
            assets.prepare_image_assets(data, check_cancelled=Mock(side_effect=CancelledError))
        self.recipe.write_text(TEXT.replace("-2", "-3"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            assets.prepare_image_assets(data, check_cancelled=lambda: None)

    def test_pinned_native_v_patch_changes_denoising_and_eps_does_not_force_prediction(self):
        sampling, node = native_sampling()
        config = type("SDXL", (), {"sampling_settings": {}, "inpaint_model": lambda self: False})()
        original_type = type("Eps", (sampling.ModelSamplingDiscrete, sampling.EPS), {})
        model = Patcher(config, original_type(config))
        modules = {
            "comfy.model_sampling": sampling,
            "comfy_extras.nodes_model_advanced": NS(ModelSamplingDiscrete=node),
        }
        with patch.dict(sys.modules, modules):
            patched = recipes.apply_model_recipe(model, self.capture())
            eps = replace(self.capture(), overrides=recipes.RecipeOverrides(None, -2))
            self.assertIs(recipes.apply_model_recipe(patched, eps), patched)
        self.assertIsNot(patched, model)
        sigma = torch.tensor([1.0])
        args = (sigma, torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2))
        self.assertFalse(
            torch.equal(
                model.sampling.calculate_denoised(*args), patched.sampling.calculate_denoised(*args)
            )
        )
        self.assertIsInstance(model.sampling, sampling.EPS)
        self.assertNotIsInstance(model.sampling, sampling.V_PREDICTION)

    def test_runtime_guard_rejects_wrong_label_flow_and_inpaint(self):
        sampling, node = native_sampling()
        modules = {
            "comfy.model_sampling": sampling,
            "comfy_extras.nodes_model_advanced": NS(ModelSamplingDiscrete=node),
        }
        captured = self.capture()
        for name, flow, inpaint in (
            ("Flux", False, False),
            ("SDXL", True, False),
            ("SDXL", False, True),
            ("SD15", False, False),
        ):
            config = type(
                name, (), {"sampling_settings": {}, "inpaint_model": lambda self: inpaint}
            )()
            model = Patcher(config, object() if flow else sampling.ModelSamplingDiscrete(config))
            with (
                self.subTest(name=name, flow=flow, inpaint=inpaint),
                patch.dict(sys.modules, modules),
            ):
                with self.assertRaisesRegex(ValueError, "ordinary native discrete"):
                    recipes.apply_model_recipe(model, captured)

    def test_clip_layer_policy_and_text_cache_cover_positive_negative_switch_and_lora_clone(self):
        owner = NS(
            conditions={k: {"text": None, "cache": None} for k in ("+", "-", "switch")},
            xl_base_patched=NS(clip=Clip()),
            yaml_recipe=self.capture(clip_layer_source="recipe"),
        )
        calls = []
        encoder = NS(
            encode=lambda clip, text: (calls.append((clip.layer, text)) or [(clip.layer, text)],)
        )
        encode = method(
            PIPELINE,
            "textencode",
            dict(
                path_manager=NS(model_paths={"embedding_path": [], "cache_path": self.root}),
                warm_patched_text_encoder=lambda clip: None,
                CLIPTextEncode=lambda: encoder,
            ),
        )
        for source, layer in (("recipe", -2), ("request", None), ("recipe", -2)):
            owner.yaml_recipe = replace(owner.yaml_recipe, clip_layer_source=source)
            for sign in ("+", "-", "switch"):
                encode(owner, sign, "same literal prompt", 1)
                self.assertEqual(owner.conditions[sign]["cache"][0][0], layer)
                encode(owner, sign, "same literal prompt", 1)  # warm cache
            self.assertIsNone(owner.xl_base_patched.clip.layer)
        self.assertEqual(len(calls), 9)
        owner.xl_base_patched.clip = owner.xl_base_patched.clip.clone()  # LoRA request clone
        self.assertIsNone(recipes.apply_clip_recipe(owner.xl_base_patched.clip, 1, None).layer)

    def test_actual_loader_cache_and_recursive_fallback_apply_recipe_once(self):
        fixture = cache.ModelCacheTests("test_component_choices_reject_malformed_or_unknown_slots")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        checkpoint = fixture.file("checkpoint.safetensors")
        fixture.file("encoder.safetensors")
        fixture.file("vae.safetensors")
        _, owner, native, backend, _ = fixture.base_loader(checkpoint)
        captured = recipes.capture_recipe(self.recipe, checkpoint, family="SDXL")
        data = {recipes.KEY: captured}
        unet = native.return_value[0]
        native.return_value = (unet, None, None, None)
        apply = Mock(side_effect=lambda model, captured: model)
        with patch.object(recipes, "apply_model_recipe", apply):
            owner.load_base_model(checkpoint.name, prepared_data=data)
            owner.load_base_model(checkpoint.name, prepared_data=data)
            self.assertEqual(native.call_count, 1)
            self.assertEqual(apply.call_count, 1)
            owner.load_base_model(checkpoint.name)
            self.assertEqual(native.call_count, 2)
            self.assertIsNone(owner.yaml_recipe)
            self.assertIsNone(owner.base_yaml_recipe_identity)

    def test_actual_outer_scope_reuses_read_lease_when_checkpoint_writer_is_waiting(self):
        captured = self.capture()
        service = PreparationService()
        self.addCleanup(service._executor.shutdown, wait=True)
        real_scope = recipes.recipe_read_scope
        done, inner = threading.Event(), threading.Event()
        writer = None

        def write():
            with service.artifact_lease(self.checkpoint, write=True):
                done.set()

        @contextmanager
        def contended_scope(capture, **kwargs):
            nonlocal writer
            with real_scope(capture, service=service, **kwargs):
                writer = threading.Thread(target=write, daemon=True)
                writer.start()
                with service._condition:
                    self.assertTrue(
                        service._condition.wait_for(
                            lambda: any(
                                item.waiting_writers for item in service._artifacts.values()
                            ),
                            2,
                        )
                    )
                yield capture

        fixture = cache.ModelCacheTests("test_component_choices_reject_malformed_or_unknown_slots")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.file("encoder.safetensors")
        fixture.file("vae.safetensors")
        _, owner, native, _, _ = fixture.base_loader(self.checkpoint)
        unet = native.return_value[0]
        native.return_value = (unet, None, None, None)

        def patched(model, recipe):
            self.assertFalse(done.is_set())
            inner.set()
            return model

        with (
            patch.object(recipes, "recipe_read_scope", contended_scope),
            patch.object(recipes, "apply_model_recipe", side_effect=patched),
        ):
            owner.load_base_model(self.checkpoint.name, prepared_data={recipes.KEY: captured})
        self.assertTrue(inner.is_set())
        self.assertTrue(done.wait(2))
        writer.join(2)
        self.assertEqual(service._artifacts, {})

    def test_api_presence_policy_and_schema_use_real_capture_caller(self):
        from ruinedfooocus.integrations.openai_api import ImageRequest
        from test_api_image_edits import queued_data

        explicit = ImageRequest(clip_skip=1, clip_layer_source="recipe").model_dump(
            exclude_none=True
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            queued_data("SDXL", payload_options=explicit)
        seen = []
        with patch.object(
            assets,
            "capture_image_assets",
            side_effect=lambda data, *a, **kw: seen.append(data) or data,
        ):
            queued_data(
                "SDXL", payload_options={"model_recipe_path": str(self.recipe), "clip_skip": 1}
            )
        self.assertEqual(seen[0]["clip_layer_source"], "request")
        self.assertEqual(seen[0]["clip_skip"], 1)
        self.assertEqual(
            ImageRequest(model_recipe_path=str(self.recipe)).model_recipe_path, str(self.recipe)
        )

    def test_actual_gallery_and_default_consumers_retain_recipe_and_cascade_fields(self):
        import test_image_metadata_boundary as boundary
        from ruinedfooocus.features.generation.defaults import generation_defaults

        captured = self.capture(clip_layer_source="recipe")
        metadata = dict(
            prompt="literal LEFT---RIGHT",
            clip_skip=1,
            model_recipe_path=str(self.recipe),
            model_recipe_sha256=captured.recipe_sha256,
            model_recipe_overrides={"prediction": "v_prediction", "clip_layer_idx": -2},
            clip_layer_source="recipe",
            cascade_stage_b="stage-b.safetensors",
            cascade_b_steps=9,
            cascade_b_cfg=1.5,
            cascade_b_sampler="euler",
            cascade_b_scheduler="simple",
            cascade_stage_b_source={"sha256": "a" * 64},
        )
        updates = boundary.gallery(metadata)
        for key in (
            "model_recipe_path",
            "model_recipe_sha256",
            "model_recipe_overrides",
            "clip_layer_source",
            "cascade_stage_b",
            "cascade_b_steps",
            "cascade_b_cfg",
            "cascade_b_sampler",
            "cascade_b_scheduler",
        ):
            self.assertEqual(updates[key]["value"], metadata[key])
        saved = generation_defaults(metadata)
        self.assertEqual(saved["clip_layer_source"], "recipe")
        self.assertEqual(saved["model_recipe_sha256"], captured.recipe_sha256)
        self.assertEqual(saved["cascade_b_steps"], 9)
        self.assertEqual(
            render_values(metadata)["cascade_stage_b_source"], metadata["cascade_stage_b_source"]
        )
        legacy = boundary.gallery({"prompt": "legacy", "clip_skip": 1})
        self.assertEqual(legacy["clip_layer_source"]["value"], "request")
        self.assertEqual(legacy["model_recipe_path"]["value"], "")


if __name__ == "__main__":
    unittest.main()
