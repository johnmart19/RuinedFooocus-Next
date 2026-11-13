"""Unavailable automatic recipes keep the real controls usable and generation strict."""

import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.features.models.performance import PerformanceSettings, UnavailableModelRecipe
from ruinedfooocus.features.models.selection import primary_checkpoint_selection
from test_api_image_edits import queued_data
import test_model_recommendations as recommendation_fixtures

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ruinedfooocus/ui/app.py"


def callback(name, scope):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope[name]


class ModelDefaultsUiTests(unittest.TestCase):
    def test_background_identity_discovery_notifies_family_consumers_without_reselecting(self):
        metadata = {"baseModel": "Illustrious", "id": 123}
        models = SimpleNamespace(
            revision=3,
            get_names=lambda kind: ["selected.safetensors"],
            get_models_by_path=lambda *args: metadata,
        )
        skip = object()
        scope = dict(
            shared=SimpleNamespace(models=models),
            gr=SimpleNamespace(skip=lambda: skip, update=lambda **values: values),
            update_model_filter=lambda value: ("gallery", ["record"]),
            adapters=SimpleNamespace(discover=lambda: (["adapter"], "status")),
            checkpoint_choices=lambda names, selected, metadata: [("renamed", selected)],
            primary_checkpoint_selection=primary_checkpoint_selection,
        )
        refresh = callback("update_model_previews", scope)
        # A manual refresh counter need not equal the background catalogue revision.
        result = refresh(2, "all", "all", [], "selected.safetensors", 12, None)
        self.assertEqual(result[-2], 13)
        self.assertEqual(result[-3], {"choices": [("renamed", "selected.safetensors")]})
        self.assertNotIn("value", result[-3])
        self.assertEqual(
            refresh(3, "all", "all", [], "selected.safetensors", 13, result[-1]), (skip,) * 7
        )
        metadata["preview"] = "new artwork"
        models.revision = 4
        artwork = refresh(3, "all", "all", [], "selected.safetensors", 13, result[-1])
        self.assertIs(artwork[-2], skip)
        metadata["baseModel"] = "Pony"
        models.revision = 5
        changed = refresh(4, "all", "all", [], "selected.safetensors", 13, artwork[-1])
        self.assertEqual(changed[-2], 14)
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        tick = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "tick"
            and any(
                k.arg == "fn"
                and isinstance(k.value, ast.Name)
                and k.value.id == "update_model_previews"
                for k in node.keywords
            )
        )
        outputs = next(k.value for k in tick.keywords if k.arg == "outputs")
        self.assertEqual(outputs.elts[-2].id, "model_metadata_refresh")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.performance = recommendation_fixtures.RecommendationTests()._performance(
            Path(self.temp.name)
        )
        self.performance.model_recommendations.offline = True
        registry = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["simple"])
        )
        self.modules = patch.dict("sys.modules", {"comfy.samplers": registry})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_real_startup_option_bindings_construct_without_claiming_auto_recipe(self):
        scope = dict(
            performance_settings=self.performance,
            initial_performance="Model Defaults",
            initial_checkpoint=primary_checkpoint_selection("model.safetensors"),
            settings={"base_model": "model.safetensors"},
            initial_model_base="Chroma",
            initial_fixed={},
        )
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in ("initial_recipe", "custom_default_values")
                for target in node.targets
            )
        ]
        self.assertEqual(len(assignments), 2)
        for node in assignments:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
        self.assertIn("Choose Own settings", scope["initial_recipe"]["_recipe_unavailable"])
        self.assertEqual(scope["custom_default_values"]["clip_skip"], 1)
        text = self.performance.describe_workflow(
            "Chroma", "model.safetensors", "Model Defaults", 30, 8, "euler", "simple"
        )
        self.assertIn("no qualified Model Defaults recipe", text)
        self.assertNotIn("Steps: 30", text)
        with self.assertRaises(UnavailableModelRecipe):
            self.performance.apply(
                dict(performance_selection="Model Defaults", base_model_name="model.safetensors"),
                "Chroma",
            )

    def test_actual_callbacks_keep_manual_controls_and_show_missing_recipe(self):
        scope = dict(
            performance_settings=self.performance,
            gr=SimpleNamespace(update=lambda **values: values, Request=object),
            t=lambda text: text,
            localize=lambda text, locale: text,
            resolve_locale=lambda **kwargs: None,
            KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["simple"]),
        )
        controls = callback("performance_control_updates", scope)
        automatic = controls("Model Defaults", "Chroma", "local.safetensors")
        self.assertTrue(
            all(not update["visible"] and "value" not in update for update in automatic[:5])
        )
        manual = controls("Own settings", "Chroma", "local.safetensors")
        self.assertTrue(all(update["visible"] and "value" not in update for update in manual[:5]))
        self.performance.model_recommendations.describe = Mock(
            side_effect=AssertionError("No recipe exists")
        )
        recommendation = callback("refresh_recommendation", scope)
        self.assertIn(
            "Choose Own settings",
            recommendation("Model Defaults", "Chroma", "local.safetensors", False),
        )

    def test_control_fallback_does_not_swallow_unrelated_validation_errors(self):
        with patch.object(
            self.performance, "get_perf_options", side_effect=ValueError("invalid checkpoint")
        ):
            with self.assertRaisesRegex(ValueError, "invalid checkpoint"):
                self.performance.control_options("Model Defaults", "local.safetensors", "Chroma")
        self.assertEqual(
            self.performance.control_options("Own settings"), PerformanceSettings.default_settings
        )

    def test_real_api_preserves_complete_explicit_recipe_and_rejects_partial_or_invalid(self):
        explicit = dict(steps=11, cfg=2, sampler="euler", scheduler="simple", clip_skip=2)
        data = queued_data(
            "Chroma",
            recipe_resolver=self.performance,
            payload_options=explicit,
            selection="Model Defaults",
        )
        self.assertEqual(
            tuple(
                data[key]
                for key in ("custom_steps", "cfg", "sampler_name", "scheduler", "clip_skip")
            ),
            (11, 2, "euler", "simple", 2),
        )
        self.assertEqual(data["performance_selection"], "Own settings")
        self.assertTrue(data["_sampling_captured"])
        with self.assertRaises(UnavailableModelRecipe):
            queued_data(
                "Chroma",
                recipe_resolver=self.performance,
                payload_options=dict(steps=11, cfg=2),
                selection="Model Defaults",
            )
        with self.assertRaisesRegex(ValueError, "Unknown sampler"):
            queued_data(
                "Chroma",
                recipe_resolver=self.performance,
                payload_options=explicit | dict(sampler="invented"),
                selection="Model Defaults",
            )


if __name__ == "__main__":
    unittest.main()
