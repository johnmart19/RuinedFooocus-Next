"""Pinned graph examples are honest optional defaults, not publisher advice."""

from copy import deepcopy
import builtins
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.model_recommendations import ModelRecommendations
from ruinedfooocus.features.models.performance import PerformanceSettings, UnavailableModelRecipe
from ruinedfooocus.features.models.resolutions import ResolutionSettings
from ruinedfooocus.features.models.resource_defaults import native_workflow_example


class SourceImageDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.model = self.root / "arbitrary-quant.safetensors"
        self.model.write_bytes((2).to_bytes(8, "little") + b"{}")
        (self.root / "arbitrary-quant.json").write_text(
            json.dumps(dict(baseModel="AuraFlow", _rf_file_signature=file_signature(self.model))),
            encoding="utf-8",
        )
        self.performance = PerformanceSettings.__new__(PerformanceSettings)
        self.performance.performance_options = {}
        self.performance.legacy_options = {}
        self.performance.model_recommendations = ModelRecommendations(
            self.root, self.root / "settings.json", offline=False
        )
        self.example = dict(
            recipe_id="official-connected-fixture",
            operation="generate",
            native_classes=["AuraFlow"],
            behavioral_variant=None,
            unresolved=[],
            default_size=[768, 1024],
            sampling=dict(custom_steps=23, cfg=3, sampler_name="euler", scheduler="simple"),
            source="native-workflow-example",
        )
        from ruinedfooocus.features.models.native_snapshot import (
            get_image_family_facts,
            native_sampler_choices,
        )

        self.real_facts_reader = get_image_family_facts
        self.real_sampler_reader = native_sampler_choices
        self.facts_reader = lambda *args, **kwargs: SimpleNamespace(
            metadata=lambda: {"example_recipe": deepcopy(self.example)}
        )
        self.enterContext(
            patch.dict(
                "sys.modules",
                {
                    "comfy.model_detection": SimpleNamespace(),
                    "comfy.samplers": SimpleNamespace(
                        KSampler=SimpleNamespace(
                            SAMPLERS=["euler", "dpmpp_2m_sde_gpu"], SCHEDULERS=["simple", "karras"]
                        )
                    ),
                    "ruinedfooocus.features.models.native_snapshot": SimpleNamespace(
                        get_image_family_facts=lambda *args, **kwargs: self.facts_reader(
                            *args, **kwargs
                        ),
                        native_sampler_choices=self.real_sampler_reader,
                    ),
                },
            )
        )
        self.contract = dict(architecture="AuraFlow", native_class="AuraFlow", metadata={})
        self.enterContext(
            patch(
                "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                return_value=self.contract,
            )
        )
        self.enterContext(
            patch(
                "ruinedfooocus.features.models.native_recipes.compatible_native_recipe",
                return_value={},
            )
        )
        self.refresh = self.enterContext(
            patch.object(
                self.performance.model_recommendations,
                "refresh",
                side_effect=AssertionError("Submission must never refresh network metadata"),
            )
        )

    def options(self, family="AuraFlow"):
        return self.performance.get_perf_options("Model Defaults", str(self.model), family)

    def test_source_example_supplies_usable_sampling_and_truthful_provenance(self):
        options = self.options()
        self.assertEqual(options["custom_steps"], 23)
        self.assertEqual(options["cfg"], 3)
        self.assertIn("Native workflow example", options["_recipe_baseline"])
        self.assertIn("no publisher recommendation", options["_recipe_baseline"])
        self.refresh.assert_not_called()

    def test_unresolved_edit_variant_or_wrong_native_class_cannot_be_auto(self):
        original = deepcopy(self.example)
        for changes in (
            {"unresolved": ["dynamic widget"]},
            {"operation": "edit"},
            {"behavioral_variant": "turbo"},
            {"native_classes": ["FluxSchnell"]},
            {"sampling": dict(original["sampling"], scheduler="SpecializedSchedule")},
        ):
            with self.subTest(changes=changes):
                self.example = original | changes
                self.assertEqual(
                    self.options()["custom_steps"],
                    PerformanceSettings.default_settings["custom_steps"],
                )

    def test_essential_variant_contracts_do_not_use_an_ordinary_example(self):
        for family in ("QwenImage21", "Flux3", "ChromaRadiance"):
            with self.subTest(family=family), self.assertRaises(UnavailableModelRecipe):
                self.options(family)

    def test_explicit_saved_sampling_overrides_source_examples(self):
        self.performance.model_recommendations.save_override(
            str(self.model),
            dict(custom_steps=11, cfg=2, sampler_name="euler", scheduler="simple", clip_skip=2),
            ["euler"],
            ["simple"],
        )
        self.assertEqual(self.options()["custom_steps"], 11)
        self.assertEqual(self.options()["clip_skip"], 2)

    def test_example_canvas_follows_auto_and_preserves_manual_or_replay(self):
        resolutions = ResolutionSettings()
        sizes, source = resolutions.recommendation("AuraFlow", str(self.model), self.performance)
        self.assertEqual(sizes, [(768, 1024)])
        self.assertIn("example canvas", source)
        self.assertIn("no publisher size recommendation", source)
        _, selected, automatic = resolutions.model_selection(
            "AuraFlow", "1000x500 (Saved)", True, str(self.model), self.performance
        )
        self.assertEqual(resolutions.aspect_ratios[selected], (768, 1024))
        self.assertTrue(automatic)
        _, selected, automatic = resolutions.model_selection(
            "AuraFlow", "1000x500 (Saved)", False, str(self.model), self.performance
        )
        self.assertEqual(selected, "1000x500 (Saved)")
        self.assertFalse(automatic)
        _, selected, automatic = resolutions.model_selection(
            "AuraFlow", "Custom...", True, str(self.model), self.performance
        )
        self.assertEqual(selected, "Custom...")
        self.assertFalse(automatic)

    def test_invalid_example_numbers_do_not_leak_into_capture(self):
        self.example["sampling"]["cfg"] = float("nan")
        self.assertEqual(
            native_workflow_example(
                "AuraFlow", native_class="AuraFlow", samplers=["euler"], schedulers=["simple"]
            ),
            {},
        )

    def test_explicit_matching_variant_can_inherit_only_its_source_example(self):
        self.example["behavioral_variant"] = "base"
        self.contract["metadata"] = {"variant": "base"}
        self.assertEqual(self.options()["custom_steps"], 23)
        self.contract["metadata"] = {"variant": "turbo"}
        with self.assertRaises(UnavailableModelRecipe):
            self.options()

    def test_real_snapshot_example_requires_its_actual_declared_variant(self):
        self.facts_reader = self.real_facts_reader
        example = native_workflow_example(
            "SD3",
            native_class="SD3",
            variant="3.5-large",
            samplers=["euler"],
            schedulers=["sgm_uniform"],
        )
        self.assertEqual(example["sampling"]["custom_steps"], 20)
        self.assertEqual(example["source"]["kind"], "native-workflow-example")
        self.assertEqual(native_workflow_example("SD3", native_class="SD3"), {})
        self.assertEqual(
            native_workflow_example("Flux", native_class="Flux", variant="schnell"), {}
        )

    def test_cold_resolution_advice_never_imports_native_or_tensor_runtime(self):
        import sys

        original_import = builtins.__import__

        def passive_only(name, *args, **kwargs):
            if name.split(".", 1)[0] in ("comfy", "torch"):
                raise AssertionError("Passive resolution advice imported a native/tensor runtime")
            return original_import(name, *args, **kwargs)

        with patch.dict(sys.modules):
            sys.modules.pop("comfy.model_detection", None)
            sys.modules.pop("comfy.samplers", None)
            with patch("builtins.__import__", side_effect=passive_only):
                self.assertEqual(self.performance.native_example("AuraFlow", str(self.model)), {})
                sizes, _ = ResolutionSettings().recommendation(
                    "AuraFlow", str(self.model), self.performance
                )
                self.assertIsInstance(sizes, list)

    def test_cold_custom_headers_keep_source_defaults_without_runtime_or_network(self):
        import sys
        import gguf
        import numpy as np
        from test_native_045_roles import native_sources

        self.facts_reader = self.real_facts_reader
        shapes = {
            "joint_blocks.0.context_block.attn.qkv.weight": [64, 64],
            "x_embedder.proj.weight": [64, 16, 2, 2],
        }
        original_import = builtins.__import__

        def passive_only(name, *args, **kwargs):
            if name.split(".", 1)[0] in ("comfy", "torch"):
                raise AssertionError("Cold defaults imported a native/tensor runtime")
            return original_import(name, *args, **kwargs)

        for suffix in (".safetensors", ".gguf"):
            with self.subTest(suffix=suffix):
                self.model = self.root / ("custom-any-name" + suffix)
                if suffix == ".safetensors":
                    header = {
                        key: dict(shape=shape, dtype="F8_E4M3", data_offsets=[0, 0])
                        for key, shape in shapes.items()
                    }
                    header["__metadata__"] = {"variant": "3.5-large"}
                    raw = json.dumps(header).encode()
                    self.model.write_bytes(len(raw).to_bytes(8, "little") + raw)
                else:
                    writer = gguf.GGUFWriter(str(self.model), "sd3")
                    writer.add_string("variant", "3.5-large")
                    for name, shape in shapes.items():
                        tensor = np.zeros(shape, dtype=np.float32)
                        if len(shape) == 2:
                            writer.add_tensor(
                                name,
                                gguf.quantize(tensor, gguf.GGMLQuantizationType.Q8_0),
                                raw_dtype=gguf.GGMLQuantizationType.Q8_0,
                            )
                        else:
                            writer.add_tensor(name, tensor)
                    writer.write_header_to_file()
                    writer.write_kv_data_to_file()
                    writer.write_tensors_to_file()
                    writer.close()
                self.model.with_suffix(".json").write_text(
                    json.dumps(
                        dict(baseModel="SD3", _rf_file_signature=file_signature(self.model))
                    ),
                    encoding="utf-8",
                )
                with (
                    patch.dict(sys.modules),
                    native_sources(),
                    patch(
                        "ruinedfooocus.features.models.model_identity.cached_file_sha256",
                        side_effect=AssertionError("Cold defaults must not hash weights"),
                    ),
                ):
                    sys.modules.pop("comfy.model_detection", None)
                    sys.modules.pop("comfy.samplers", None)
                    with patch("builtins.__import__", side_effect=passive_only):
                        example = self.performance.native_example("SD3", str(self.model))
                        self.assertEqual(example["sampling"]["custom_steps"], 20)
                        values = self.options("SD3")
                        self.assertEqual(values["custom_steps"], 20)
                        self.assertIn("Native workflow example", values["_recipe_baseline"])
                        sizes, _ = ResolutionSettings().recommendation(
                            "SD3", str(self.model), self.performance
                        )
                        self.assertIn((1024, 1024), sizes)
                self.refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
