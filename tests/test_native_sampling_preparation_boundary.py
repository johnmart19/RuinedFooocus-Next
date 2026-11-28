"""Native schedule capture belongs to generation, not standalone image tools."""

from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.features.generation import image_preparation as assets, native_sampling as native
from ruinedfooocus.features.models.resource_defaults import load_family_recipe


class NativeSamplingPreparationBoundaryTests(unittest.TestCase):
    def test_standalone_tools_ignore_unrelated_native_sampling_state(self):
        forbidden = Mock(
            side_effect=AssertionError("Standalone tool resolved generation resources")
        )
        models = NS(get_file=forbidden, get_models_by_path=forbidden, get_model_base=forbidden)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "upscaler.pth").write_bytes(b"owned source fixture")
            manager = NS(DOWNLOADABLE_FILES={}, get_model_folders=lambda category: [root])
            for kind in ("rembg", "upscale"):
                with self.subTest(kind=kind):
                    data = dict(
                        _api_job=object(),
                        _sampling_captured=False,
                        base_model_name="unavailable-generation.safetensors",
                        scheduler="Ideogram4Scheduler",
                        native_negative_model="unavailable-negative.safetensors",
                        native_schedule_alpha=float("nan"),
                        prompt="#! literal utility input",
                    )
                    controls = dict(type=kind, upscaler="upscaler.pth")
                    with patch.object(
                        native, "capture_assets", wraps=native.capture_assets
                    ) as capture:
                        result = assets.capture_image_assets(data, manager, controls, models=models)
                    capture.assert_not_called()
                    self.assertIs(result["_sampling_captured"], False)
                    self.assertEqual(result["prompt"], data["prompt"])
                    self.assertNotIn(assets.KEY, data)
                    self.assertEqual(
                        [item.role for item in result[assets.KEY].artifacts],
                        ["upscaler"] if kind == "upscale" else [],
                    )
        forbidden.assert_not_called()

    def test_ordinary_generation_still_captures_current_native_sampling_plan(self):
        recipe = load_family_recipe("native-flux2-dev")
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        checkpoint = root / "owned-generation.safetensors"
        checkpoint.write_bytes((2).to_bytes(8, "little") + b"{}")
        metadata = dict(baseModel="Flux2")
        models = NS(
            get_file=Mock(return_value=checkpoint),
            get_models_by_path=Mock(return_value=metadata),
            get_model_base=lambda value: value["baseModel"],
        )
        manager = NS(DOWNLOADABLE_FILES={})
        data = dict(
            base_model_name=checkpoint.name,
            _sampling_captured=True,
            scheduler="Flux2Scheduler",
            prompt="ordinary generation",
        )
        with (
            patch(
                "ruinedfooocus.features.models.component_profiles.component_inventory",
                return_value=dict(rows=[]),
            ),
            patch(
                "ruinedfooocus.features.models.native_recipes.current_native_recipe",
                return_value=recipe,
            ) as current_recipe,
        ):
            result = assets.capture_image_assets(data, manager, {}, models=models)
        current_recipe.assert_called_once_with(checkpoint, manager, metadata)
        self.assertEqual(result[native.KEY], native.plan_from_recipe(recipe))
        self.assertEqual(result["native_sampling"]["scheduler"], "Flux2Scheduler")
        self.assertEqual(result["prompt"], data["prompt"])
        self.assertNotIn(native.KEY, data)


if __name__ == "__main__":
    unittest.main()
