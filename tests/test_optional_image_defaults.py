"""Optional recommendations do not block a connected ordinary native image route."""

from contextlib import ExitStack
from copy import deepcopy
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.model_recommendations import ModelRecommendations
from ruinedfooocus.features.models.performance import PerformanceSettings, UnavailableModelRecipe
from ruinedfooocus.features.models.resource_defaults import performance_defaults


class OptionalImageDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "renamed-unregistered.safetensors"
        self.performance = PerformanceSettings.__new__(PerformanceSettings)
        self.performance.legacy_options = {}
        self.performance.performance_options = performance_defaults()
        self.performance.model_recommendations = ModelRecommendations(
            self.root, self.root / "settings.json", offline=True
        )
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.dict(
                "sys.modules",
                {
                    "comfy.model_detection": SimpleNamespace(),
                    "comfy.samplers": SimpleNamespace(
                        KSampler=SimpleNamespace(
                            SAMPLERS=["euler", "dpmpp_2m", "dpmpp_2m_sde_gpu"],
                            SCHEDULERS=["simple", "normal", "karras"],
                        )
                    ),
                },
            )
        )
        self.contract = dict(architecture="AuraFlow", native_class="AuraFlow", metadata={})
        self.stack.enter_context(
            patch(
                "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                side_effect=lambda path: self.contract.copy(),
            )
        )
        self.stack.enter_context(
            patch(
                "ruinedfooocus.features.models.native_recipes.compatible_native_recipe",
                return_value={},
            )
        )
        self.stack.enter_context(
            patch(
                "ruinedfooocus.features.models.model_identity.cached_file_sha256",
                side_effect=AssertionError("No model hashing"),
            )
        )
        self.header()

    def header(self, dtype="F16", family="AuraFlow"):
        raw = json.dumps({"weight": dict(shape=[1], dtype=dtype, data_offsets=[0, 0])}).encode()
        self.model.write_bytes(len(raw).to_bytes(8, "little") + raw)
        (self.root / (self.model.stem + ".json")).write_text(
            json.dumps(dict(baseModel=family, _rf_file_signature=file_signature(self.model)))
        )

    def options(self, family="AuraFlow"):
        return self.performance.get_perf_options("Model Defaults", str(self.model), family)

    def test_missing_optional_recipe_uses_truthful_fresh_generic_defaults(self):
        for dtype in ("F16", "F8_E4M3"):
            with self.subTest(dtype=dtype):
                self.header(dtype)
                self.assertIsNone(self.performance.family_preset("AuraFlow", str(self.model)))
                values = self.options()
                self.assertEqual(
                    {key: values[key] for key in self.performance.default_settings},
                    self.performance.default_settings,
                )
                self.assertEqual(
                    values["_recipe_baseline"],
                    "Generic app fallback settings; no complete checkpoint recommendation",
                )

    def test_unregistered_native_image_can_use_existing_family_fallback(self):
        self.header(family="Unknown")
        self.contract.update(architecture="SDXL 1.0", native_class="SDXL")
        values = self.options("Unknown")
        self.assertEqual(values["cfg"], self.performance.performance_options["SDXL"]["cfg"])
        self.assertIn("Family fallback settings for SDXL 1.0", values["_recipe_baseline"])

    def test_known_family_fallback_also_captures_its_source(self):
        self.header(family="SDXL 1.0")
        self.contract.update(architecture="SDXL 1.0", native_class="SDXL")
        actual = self.performance.apply(
            dict(performance_selection="Model Defaults", base_model_name=str(self.model)),
            "SDXL 1.0",
        )
        self.assertEqual(actual["cfg"], self.performance.performance_options["SDXL"]["cfg"])
        self.assertIn("Family fallback settings for SDXL 1.0", actual["sampling_defaults_source"])

    def test_current_ordinary_header_can_recover_from_stale_previous_family_metadata(self):
        self.header(family="Unknown")
        self.contract.update(architecture="SDXL 1.0", native_class="SDXL")
        metadata_path = self.root / (self.model.stem + ".json")
        metadata_path.write_text(
            json.dumps(dict(baseModel="Flux3", _rf_file_signature=["previous", 0, 0, 0]))
        )
        self.assertEqual(
            self.performance.model_recommendations.metadata(str(self.model)),
            {"_rf_stale_native_family": "Flux3"},
        )
        values = self.options("Unknown")
        self.assertEqual(values["cfg"], self.performance.performance_options["SDXL"]["cfg"])
        self.assertIn("Family fallback", values["_recipe_baseline"])
        self.assertNotIn("_model_default_sigmas", values)

    def test_saved_override_and_complete_source_recipe_keep_precedence(self):
        store = self.performance.model_recommendations
        values = dict(custom_steps=11, cfg=2, sampler_name="euler", scheduler="simple", clip_skip=2)
        samplers = ["euler"]
        schedulers = ["simple"]
        store.save_override(str(self.model), values, samplers, schedulers)
        self.assertEqual({key: self.options()[key] for key in values}, values)
        store.reset_override(str(self.model))
        store.path.write_text(
            json.dumps({store.identity(str(self.model)): dict(recommended=values)})
        )
        self.assertEqual({key: self.options()[key] for key in values}, values)

    def test_partial_recommendation_keeps_precedence_over_truthfully_labelled_fallback(self):
        store = self.performance.model_recommendations
        store.path.write_text(
            json.dumps({store.identity(str(self.model)): dict(recommended=dict(cfg=2))})
        )
        values = self.options()
        self.assertEqual(values["cfg"], 2)
        self.assertEqual(values["custom_steps"], self.performance.default_settings["custom_steps"])
        self.assertIn("no complete checkpoint recommendation", values["_recipe_baseline"])

    def test_switching_from_previous_automatic_values_replaces_every_sampling_value(self):
        from ruinedfooocus.features.generation.native_sampling import KEY

        previous = dict(
            performance_selection="Model Defaults",
            base_model_name=str(self.model),
            custom_steps=99,
            cfg=19,
            sampler_name="previous",
            scheduler="previous",
            clip_skip=5,
            _model_default_sampling_shift=7,
            _model_default_guidance={"previous": 1},
            _model_default_sigmas=(1, 0),
            sampling_defaults_source="Previous family defaults",
        )
        previous[KEY] = "previous family plan"
        actual = self.performance.apply(previous, "AuraFlow")
        self.assertEqual(
            actual["sampling_defaults_source"],
            "Generic app fallback settings; no complete checkpoint recommendation",
        )
        self.assertEqual(
            {key: actual[key] for key in self.performance.default_settings},
            self.performance.default_settings,
        )
        self.assertFalse(
            any(
                key in actual
                for key in (
                    KEY,
                    "_model_default_sampling_shift",
                    "_model_default_guidance",
                    "_model_default_sigmas",
                )
            )
        )

    def test_fallback_source_is_visible_and_manual_settings_clear_previous_source(self):
        controls = self.performance.control_options("Model Defaults", str(self.model), "AuraFlow")
        source = controls["_recipe_baseline"]
        description = self.performance.model_recommendations.describe(
            str(self.model), controls, ["euler"], ["simple"]
        )
        self.assertIn(source, description)
        manual = dict(
            performance_selection="Own settings",
            base_model_name=str(self.model),
            custom_steps=13,
            cfg=2,
            sampler_name="euler",
            scheduler="simple",
            clip_skip=2,
            sampling_defaults_source=source,
        )
        actual = self.performance.apply(manual, "AuraFlow")
        self.assertEqual(
            {key: actual[key] for key in self.performance.default_settings},
            {key: manual[key] for key in self.performance.default_settings},
        )
        self.assertNotIn("sampling_defaults_source", actual)

    def test_actual_worker_metadata_keeps_only_the_current_captured_fallback_source(self):
        data = self.performance.apply(
            dict(performance_selection="Model Defaults", base_model_name=str(self.model)),
            "AuraFlow",
        )
        source = Path(__file__).resolve().parents[1] / "ruinedfooocus/app/async_worker.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        loop = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Tuple)
            and any(
                isinstance(item, ast.Constant) and item.value == "native_sampling"
                for item in node.iter.elts
            )
        )

        def metadata(gen_data):
            scope = dict(gen_data=gen_data, prompt={}, deepcopy=deepcopy)
            exec(compile(ast.Module(body=[loop], type_ignores=[]), str(source), "exec"), scope)
            return scope["prompt"]

        self.assertEqual(
            metadata(data)["sampling_defaults_source"], data["sampling_defaults_source"]
        )
        manual = self.performance.apply(
            data | dict(performance_selection="Own settings"), "AuraFlow"
        )
        self.assertNotIn("sampling_defaults_source", metadata(manual))

    def test_essential_variants_specialized_and_unsupported_routes_remain_requirements(self):
        for family, native, native_class in (
            ("QwenImage21", "QwenImage21", "QwenImage21"),
            ("Flux3", "Flux3", "Flux3"),
            ("PixArt", "PixArt", "PixArtAlpha"),
            ("Fibo2", "Fibo2", "Fibo2"),
            ("FutureUnsupported", "FutureUnsupported", "FutureUnsupported"),
            ("Wan Video", "Wan Video", "WAN21_T2V"),
        ):
            with self.subTest(family=family):
                self.header(family=family)
                self.contract.update(architecture=native, native_class=native_class)
                with patch.object(self.performance, "family_preset", return_value=None):
                    with self.assertRaises(UnavailableModelRecipe):
                        self.options(family)

    def test_explicit_behavior_or_uninitialized_loader_is_not_a_generic_fallback(self):
        self.contract["metadata"] = {"variant": "turbo"}
        with self.assertRaises(UnavailableModelRecipe):
            self.options()
        self.contract["metadata"] = {}
        with patch.dict("sys.modules", {"comfy.model_detection": None}):
            with self.assertRaises(UnavailableModelRecipe):
                self.options()


if __name__ == "__main__":
    unittest.main()
