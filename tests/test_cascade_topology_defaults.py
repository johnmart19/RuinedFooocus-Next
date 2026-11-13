"""Only positively identified full Cascade layouts receive staged defaults."""

from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from ruinedfooocus.features.generation.cascade import STAGE_B_RECIPE, stage_b_defaults
from ruinedfooocus.features.generation.image_preparation import capture_image_assets
from ruinedfooocus.features.models.model_architecture import inspect_model_file
from ruinedfooocus.features.models.native_recipes import compatible_native_recipe
from ruinedfooocus.features.models.performance import UnavailableModelRecipe
from test_image_preparation import Manager
from test_model_defaults_ui import callback
import test_model_recommendations as recommendations
from test_model_recommendations import SAMPLERS, SCHEDULERS
import test_native_variant_defaults as native_fixtures
from test_native_variant_defaults import native_backend, write_header


def cascade_shapes(stage="c", *, lite=False, bundle=True):
    prefix = "model.diffusion_model."
    shapes = {prefix + "clf.1.weight": [16, 2048 if stage == "c" else 320, 1, 1]}
    if stage == "c":
        shapes[prefix + "clip_txt_mapper.weight"] = [1536 if lite else 2048, 1280]
        shapes[prefix + "down_blocks.1.0.channelwise.0.weight"] = [8192, 2048]
        codec = "vae.encoder.backbone.1.0.block.0.1.num_batches_tracked"
        codec_shape = []
    else:
        shapes[prefix + "clip_mapper.weight"] = [5120, 1280]
        shapes[prefix + "down_blocks.1.0.channelwise.0.weight"] = [2560, 576 if lite else 640]
        codec, codec_shape = "vae.vquantizer.codebook.weight", [8192, 4]
    if bundle:
        shapes["text_encoder.clip_g.text_projection"] = [1280, 1280]
        shapes[codec] = codec_shape
    return shapes


class CascadeTopologyDefaultsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(
            patch.dict(
                native_fixtures.SHAPES,
                {"Stable_Cascade_C": cascade_shapes(), "Stable_Cascade_B": cascade_shapes("b")},
            )
        )
        self.enterContext(native_backend())
        self.performance = recommendations.RecommendationTests()._performance(self.root / "cache")
        self.performance.model_recommendations.offline = True

    def test_native_full_c_config_qualifies_renames_and_precisions_without_hashing(self):
        with patch(
            "ruinedfooocus.features.models.model_identity.cached_file_sha256",
            side_effect=AssertionError("Defaults must not hash model payloads"),
        ):
            for dtype in ("F16", "BF16", "F8_E4M3"):
                with self.subTest(dtype=dtype):
                    path = write_header(
                        self.root / (dtype + ".safetensors"), cascade_shapes(), dtype
                    )
                    contract = inspect_model_file(path)
                    self.assertEqual(contract["native_config"]["stable_cascade_stage"], "c")
                    self.assertEqual(contract["native_config"]["c_cond"], 2048)
                    recipe = compatible_native_recipe(path, {})
                    self.assertTrue(recipe["auto_recipe_supported"])
                    self.assertFalse(recipe["artifact_qualified"])
                    options = self.performance.get_perf_options(
                        "Model Defaults", str(path), "Stable_Cascade_C"
                    )
                    self.assertEqual(
                        tuple(
                            options[k]
                            for k in (
                                "custom_steps",
                                "cfg",
                                "sampler_name",
                                "scheduler",
                                "clip_skip",
                            )
                        ),
                        (20, 4, "euler_ancestral", "simple", 1),
                    )

    def test_lite_unknown_and_conflicting_behavior_do_not_inherit_full_c_qualification(self):
        wrong_output = cascade_shapes()
        wrong_output["model.diffusion_model.clf.1.weight"] = [4, 2048, 1, 1]
        for shapes, metadata in (
            (cascade_shapes(lite=True), {}),
            ({"unknown.weight": [1]}, {}),
            (wrong_output, {}),
            (cascade_shapes(), {"variant": "turbo"}),
        ):
            with self.subTest(metadata=metadata, shapes=shapes):
                path = write_header(
                    self.root / "stable-cascade-full.safetensors", shapes, metadata=metadata
                )
                self.assertEqual(compatible_native_recipe(path, {}), {})
                with self.assertRaises(UnavailableModelRecipe):
                    self.performance.get_perf_options(
                        "Model Defaults", str(path), "Stable_Cascade_C"
                    )

    def test_full_b_companion_defaults_require_native_layout_clip_and_stage_a(self):
        path = self.root / "unrelated.safetensors"
        models = NS(get_file=lambda *a: path, get_models_by_path=lambda *a, **k: {})
        write_header(path, cascade_shapes("b"))
        with patch(
            "ruinedfooocus.features.models.model_identity.cached_file_sha256",
            side_effect=AssertionError("No hashing"),
        ):
            self.assertEqual(stage_b_defaults(path.name, models), STAGE_B_RECIPE)
        for shapes in (
            cascade_shapes("b", lite=True),
            cascade_shapes("b", bundle=False),
            cascade_shapes(),
        ):
            with self.subTest(shapes=shapes):
                write_header(path, shapes)
                self.assertEqual(stage_b_defaults(path.name, models), {})
        shapes = cascade_shapes("b")
        shapes["text_encoder.clip_g.text_projection"] = [768, 768]
        write_header(path, shapes)
        self.assertEqual(stage_b_defaults(path.name, models), {})

    def test_actual_capture_applies_b_defaults_and_retains_explicit_stage_overrides(self):
        checkpoints = self.root / "checkpoints"
        checkpoints.mkdir()
        c = write_header(checkpoints / "c.safetensors", cascade_shapes())
        b = write_header(checkpoints / "b.safetensors", cascade_shapes("b"))
        files = {c.name: c, b.name: b}
        models = NS(
            get_file=lambda category, name: files.get(name),
            get_models_by_path=lambda *a, **k: {"baseModel": "Stable_Cascade_C"},
            get_model_base=lambda data: data["baseModel"],
            detect_checkpoint_base=lambda path: "Stable_Cascade_B",
        )
        data = dict(
            base_model_name=c.name,
            cascade_stage_b=b.name,
            prompt="test",
            loras=[],
            image_component_overrides={},
        )
        with patch(
            "ruinedfooocus.features.models.component_profiles.component_inventory",
            return_value={"rows": []},
        ):
            captured = capture_image_assets(data, Manager(self.root), {}, models=models)
            self.assertEqual({key: captured[key] for key in STAGE_B_RECIPE}, STAGE_B_RECIPE)
            explicit = dict(
                cascade_b_steps=3,
                cascade_b_cfg=0,
                cascade_b_sampler="euler",
                cascade_b_scheduler="normal",
            )
            captured = capture_image_assets(data | explicit, Manager(self.root), {}, models=models)
            self.assertEqual({key: captured[key] for key in explicit}, explicit)

    def test_actual_api_and_ui_default_callers_receive_full_c_sampling(self):
        import sys
        import ruinedfooocus.app.api_runtime as api_runtime
        from ruinedfooocus.integrations.openai_api import ImageRequest

        path = write_header(self.root / "renamed.safetensors", cascade_shapes())
        b = write_header(self.root / "b.safetensors", cascade_shapes("b"))
        files = {str(path): path, b.name: b}
        # The real API prepares a request; its worker/model execution stays deferred.
        shared = NS(
            settings=NS(
                default_settings={"performance": "Model Defaults", "resolution": "default"}
            ),
            performance_settings=self.performance,
            models=NS(
                get_models_by_path=lambda *a, **k: {"baseModel": "Stable_Cascade_C"},
                get_model_base=lambda data: data["baseModel"],
                get_file=lambda category, name: files.get(name),
                detect_checkpoint_base=lambda path: inspect_model_file(path)["native_class"],
            ),
            path_manager=NS(
                paths={},
                DOWNLOADABLE_FILES={},
                get_abspath=Path,
                get_model_folders=lambda category: [self.root],
            ),
            resolution_settings=NS(
                CUSTOM_RESOLUTION="Custom...",
                selection_for_model=lambda *a: "default",
                aspect_ratios={"default": (1024, 1024)},
            ),
        )
        payload = ImageRequest(
            prompt="test", performance="Model Defaults", size="1024x1024", cascade_stage_b=b.name
        ).model_dump(exclude_none=True)
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(sys.modules["ruinedfooocus.app"], "state", shared, create=True),
            patch.object(
                api_runtime,
                "image_defaults",
                {"base_model_name": str(path), "performance_selection": "Model Defaults"},
            ),
            patch.object(api_runtime, "submit", side_effect=lambda run: run),
            patch(
                "ruinedfooocus.features.models.component_profiles.component_inventory",
                return_value={"rows": []},
            ),
        ):
            operation = api_runtime.images(
                payload, {"family": "Stable_Cascade_C", "name": str(path)}
            )
        cells = dict(
            zip(
                operation.__code__.co_freevars,
                (cell.cell_contents for cell in operation.__closure__),
            )
        )
        api = cells["data"]
        self.assertEqual(
            tuple(api[k] for k in ("custom_steps", "cfg", "sampler_name", "scheduler")),
            (20, 4, "euler_ancestral", "simple"),
        )
        controls = callback(
            "performance_control_updates",
            dict(
                performance_settings=self.performance,
                gr=NS(update=lambda **values: values, Request=object),
                t=lambda value: value,
                localize=lambda value, locale: value,
                resolve_locale=lambda **kwargs: None,
                KSampler=NS(SAMPLERS=SAMPLERS, SCHEDULERS=SCHEDULERS),
            ),
        )
        ui = controls("Model Defaults", "Stable_Cascade_C", str(path))
        self.assertEqual(
            tuple(value["value"] for value in ui[:5]), (4, "euler_ancestral", "simple", 1, 20)
        )

    def test_saved_and_manual_c_sampling_keep_precedence(self):
        path = write_header(self.root / "renamed.safetensors", cascade_shapes())
        explicit = dict(
            custom_steps=7, cfg=0, sampler_name="euler", scheduler="normal", clip_skip=2
        )
        self.performance.model_recommendations.save_override(
            str(path), explicit, SAMPLERS, SCHEDULERS
        )
        automatic = self.performance.get_perf_options(
            "Model Defaults", str(path), "Stable_Cascade_C"
        )
        self.assertEqual({key: automatic[key] for key in explicit}, explicit)
        manual = self.performance.apply(
            dict(base_model_name=str(path), performance_selection="Own settings", **explicit),
            "Stable_Cascade_C",
        )
        self.assertEqual({key: manual[key] for key in explicit}, explicit)


if __name__ == "__main__":
    unittest.main()
