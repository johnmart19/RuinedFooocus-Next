"""Contextual native controls preserve explicit values without loading models."""

import ast
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
import ruinedfooocus.app

from ruinedfooocus.features.image_editing import ui_native_workflows as ui


ROOT = Path(__file__).resolve().parents[1]


class NativeWorkflowUI(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(ui, "t", side_effect=lambda text: text))

    def build(self, defaults=None, family="Stable_Cascade_C"):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(temporary).resolve()
        self.recipe = root / "custom.yaml"
        self.recipe.write_text("model: {}", encoding="utf-8")
        manager = SimpleNamespace(
            get_model_folders=lambda category: [root],
            paths={"path_clip": []},
            get_abspath=Path,
            EXTENSIONS=(".safetensors",),
        )
        registered = {}
        self.names = ["Stage C", "Stage B", "Other"]
        shared = SimpleNamespace(
            models=SimpleNamespace(
                get_names=lambda kind: list(self.names), get_file=lambda *args: None
            ),
            path_manager=manager,
            settings=SimpleNamespace(default_settings=defaults or {}),
            add_ctrl=lambda key, obj: registered.update({key: obj}),
            get_ctrl=registered.__getitem__,
        )
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.state": shared,
                    "comfy.samplers": SimpleNamespace(
                        KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["simple"])
                    ),
                },
            )
        )
        self.enterContext(patch.object(ruinedfooocus.app, "state", shared, create=True))
        self.enterContext(
            patch(
                "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                side_effect=lambda name: name,
            )
        )
        self.enterContext(
            patch(
                "ruinedfooocus.features.image_editing.image_guidance.effective_preset_checkpoint",
                side_effect=lambda name, preset, mode: preset or name,
            )
        )
        with gr.Blocks(analytics_enabled=False) as demo:
            self.clip = gr.Number(value=1)
            shared.add_ctrl("clip_skip", self.clip)
            base = gr.Textbox(value=family)
            preset = gr.Textbox(value="")
            refresh = gr.Number(value=0)
            controls = ui.create_ui(base, preset_selection=preset, model_refresh=refresh)
        return demo, controls, registered

    def test_startup_retains_saved_values_zero_and_recipe_replay_identity(self):
        defaults = dict(
            cascade_stage_b="Unavailable B",
            cascade_b_steps=17,
            cascade_b_cfg=0,
            refiner_noise_augmentation=0,
            pid_degrade_sigma=0,
            model_recipe_path="unavailable.yaml",
            clip_layer_source="recipe",
            model_recipe_sha256="a" * 64,
            model_recipe_overrides={"prediction": "v_prediction"},
        )
        _, controls, _ = self.build(defaults, family="SDXL")
        for key, value in defaults.items():
            self.assertEqual(controls[key].value, value)
        self.assertIn(
            ("Saved: Unavailable B · Unavailable", "Unavailable B"),
            controls["cascade_stage_b"].choices,
        )
        self.assertTrue(controls["clip_layer_source"].interactive)
        _, ordinary, _ = self.build(defaults, family="Flux")
        self.assertEqual(ordinary["model_recipe_path"].value, "")
        self.assertEqual(ordinary["clip_layer_source"].value, "request")
        self.assertIsNone(ordinary["model_recipe_sha256"].value)

    def test_registered_controls_have_blank_stage_b_sampling_and_exact_gui_transport(self):
        demo, controls, registered = self.build()
        self.assertEqual(set(registered), {"clip_skip", *ui.FIELDS})
        for key in ui.FIELDS[:5]:
            self.assertIsNone(controls[key].value)
        self.assertEqual(controls["clip_layer_source"].value, "request")
        self.assertFalse(controls["cascade_b_sampler"].allow_custom_value)
        self.assertEqual(controls["cascade_b_sampler"].choices, [("euler", "euler")])
        tree = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8"))
        image_keys = next(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "image_keys"
                for target in node.targets
            )
        )
        self.assertTrue(set(ui.FIELDS).issubset(ast.literal_eval(image_keys)))
        self.assertTrue(demo.fns)

    def test_sdxl_refiner_context_preserves_explicit_zero_and_unavailable_source(self):
        defaults = dict(
            sdxl_refiner_model="Missing refiner",
            sdxl_refiner_cfg=0,
            sdxl_refiner_switch_step=3,
            sdxl_refiner_clip="Missing G",
        )
        demo, controls, _ = self.build(defaults, family="SDXL")
        for key, value in defaults.items():
            self.assertEqual(controls[key].value, value)
        context = next(fn for fn in demo.fns.values() if fn.fn.__name__ == "refiner_context")
        active = context.fn("SDXL", "Missing refiner", "")
        self.assertTrue(active[0]["visible"])
        self.assertEqual(active[1]["value"], "Missing refiner")
        inactive = context.fn("Flux", "Missing refiner", "")
        self.assertFalse(inactive[0]["visible"])
        self.assertIsNone(inactive[1]["value"])
        self.assertNotIn(controls["sdxl_refiner_cfg"], context.outputs)
        self.assertNotIn(controls["sdxl_refiner_switch_step"], context.outputs)
        refresh = next(
            fn for fn in demo.fns.values() if fn.fn.__name__ == "refiner_encoder_choices"
        )
        with patch(
            "ruinedfooocus.features.models.component_candidates.installed_candidates",
            return_value=[
                dict(name="Installed G", compatibility="compatible"),
                dict(name="Wrong L", compatibility="incompatible"),
            ],
        ):
            choices = refresh.fn("Missing G")
        self.assertEqual(choices["value"], "Missing G")
        self.assertIn("Installed G", choices["choices"])
        self.assertNotIn("Wrong L", choices["choices"])

    def test_cascade_advanced_guidance_opt_ins_are_contextual_and_replay_starts_without_images(
        self,
    ):
        defaults = dict(
            cascade_super_resolution=True,
            cascade_super_resolution_model="Missing SR",
            cascade_control_strength=0,
            cascade_vision_enabled=True,
            cascade_vision_model="Missing L",
            cascade_vision_strengths=[0, -0.25],
        )
        demo, controls, _ = self.build(defaults)
        for key in (
            "cascade_super_resolution",
            "cascade_control_strength",
            "cascade_vision_enabled",
            "cascade_vision_model",
        ):
            self.assertEqual(controls[key].value, defaults[key])
        self.assertEqual(controls["cascade_vision_images"].value, [])
        self.assertEqual(controls["cascade_vision_strengths"].value, "[0, -0.25]")
        callback = next(
            fn for fn in demo.fns.values() if fn.fn.__name__ == "cascade_guidance_context"
        )
        active = callback.fn("Stable_Cascade_C", True, True, "")
        self.assertTrue(active[0]["visible"])
        self.assertEqual(active[1:], (True, True))
        inactive = callback.fn("SDXL", True, True, "")
        self.assertFalse(inactive[0]["visible"])
        self.assertEqual(inactive[1:], (False, False))

    def test_native_and_preset_context_only_exposes_matching_workflow(self):
        demo, controls, _ = self.build()
        context = next(fn for fn in demo.fns.values() if fn.fn.__name__ == "context")
        for family, expected in (
            ("Stable_Cascade_C", [True, False, False, False]),
            ("StableCascade_C", [True, False, False, False]),
            ("HunyuanImage21Refiner", [False, True, False, False]),
            ("PiD", [False, False, True, False]),
            ("SDXL", [False, False, False, True]),
            ("Unknown", [False] * 4),
        ):
            with self.subTest(family=family):
                result = context.fn("Other", family)
                self.assertEqual([row["visible"] for row in result[:4]], expected)
                if family == "SDXL":
                    self.assertEqual(result[4:], [gr.skip()] * 4)
                else:
                    self.assertEqual(result[4:], ["", "request", None, None])
        # Visibility changes cannot replace deliberate Stage B or source values.
        self.assertNotIn(controls["cascade_b_steps"], context.outputs)
        self.assertNotIn(controls["pid_degrade_sigma"], context.outputs)

    def test_recipe_selection_refresh_and_clip_skip_preserve_intent(self):
        demo, controls, _ = self.build()
        refresh = next(
            fn.fn for fn in demo.fns.values() if fn.fn.__name__ == "refresh_recipe_choices"
        )
        result = refresh(str(self.recipe))
        self.assertEqual(result[0]["value"], str(self.recipe))
        self.assertEqual(result[1:], (gr.skip(), gr.skip(), gr.skip()))
        self.recipe.unlink()
        missing = refresh(str(self.recipe))
        self.assertEqual(missing[0]["value"], str(self.recipe))
        self.assertIn(("Selected recipe · Unavailable", str(self.recipe)), missing[0]["choices"])
        self.assertEqual(missing[1:], (gr.skip(), gr.skip(), gr.skip()))
        clip_event = next(fn for fn in demo.fns.values() if (self.clip._id, "input") in fn.targets)
        self.assertEqual(clip_event.fn(), "request")
        preset_event = next(
            fn
            for fn in demo.fns.values()
            if fn.outputs == [controls["clip_layer_source"]]
            and fn.inputs
            and fn.inputs != [controls["model_recipe_path"]]
        )
        self.assertEqual(preset_event.fn("chosen.png"), "request")
        self.assertEqual(preset_event.fn(None), gr.skip())
        recipe_input = next(
            fn
            for fn in demo.fns.values()
            if fn.outputs == [controls["model_recipe_sha256"], controls["model_recipe_overrides"]]
        )
        self.assertEqual(recipe_input.fn(), (None, None))

    def test_vae_choices_reuse_compatibility_and_refresh_preserves_unavailable_intent(self):
        candidates = Mock(
            return_value=[
                {"name": "match", "compatibility": "compatible"},
                {"name": "wrong", "compatibility": "incompatible"},
                {"name": "unknown", "compatibility": "unknown"},
            ]
        )
        with patch(
            "ruinedfooocus.features.models.component_candidates.installed_candidates", candidates
        ):
            result = ui.input_vae_choices("manager", "flux", "match")
            self.assertEqual(result["choices"], ["match"])
            self.assertEqual(result["value"], "match")
            self.assertEqual(
                [call.args for call in candidates.call_args_list],
                [("vae_flux", "manager"), ("vae_flux2", "manager")],
            )
            unavailable = ui.input_vae_choices("manager", "flux", "wrong")
            self.assertEqual(unavailable["value"], "wrong")
            self.assertIn(("Saved: wrong · Unavailable", "wrong"), unavailable["choices"])
            self.assertEqual(ui.input_vae_choices("manager", None)["choices"], [])
        demo, controls, _ = self.build()
        refresh = next(fn.fn for fn in demo.fns.values() if fn.fn.__name__ == "refresh_choices")
        self.names.remove("Stage B")
        with patch.object(ui, "input_vae_choices", return_value=gr.update(choices=[], value=None)):
            result = refresh("Stage B", "flux", "missing")
        self.assertEqual(result[0]["value"], "Stage B")
        self.assertIn(("Saved: Stage B · Unavailable", "Stage B"), result[0]["choices"])
        self.assertFalse(controls["pid_input_vae"].allow_custom_value)

    def test_stage_b_status_discloses_only_verified_recipe_without_filling_controls(self):
        from ruinedfooocus.features.generation.cascade import STAGE_B_RECIPE

        with patch("ruinedfooocus.features.generation.cascade.stage_b_defaults", return_value={}):
            self.assertIn("Enter all four", ui.stage_b_status("custom", object()))
        with patch(
            "ruinedfooocus.features.generation.cascade.stage_b_defaults",
            return_value=STAGE_B_RECIPE,
        ):
            message = ui.stage_b_status("official", object())
        self.assertIn("Verified official", message)
        self.assertIn("10 steps", message)
        self.assertIn("1.1", message)
        self.assertIn("Explicit values override", message)

    def test_stage_b_choices_exclude_other_native_families_unknowns_and_unbundled_files(self):
        from test_native_variant_defaults import write_header

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = write_header(
                root / "renamed.safetensors",
                {
                    "text_encoder.clip_g.text_projection": [1280, 1280],
                    "vae.vquantizer.codebook.weight": [8192, 4],
                },
            )
            missing_codec = write_header(
                root / "split-b.safetensors", {"text_encoder.clip_g.text_projection": [1280, 1280]}
            )
            wrong_codec = write_header(
                root / "wrong-codec.safetensors",
                {
                    "text_encoder.clip_g.text_projection": [1280, 1280],
                    "vae.vquantizer.codebook.weight": [1024, 4],
                },
            )
            families = {
                valid.name: "Stable_Cascade_B",
                missing_codec.name: "Stable_Cascade_B",
                wrong_codec.name: "Stable_Cascade_B",
                "Stage C": "Stable_Cascade_C",
                "refiner": "HunyuanImage21Refiner",
                "Flux": "Flux",
                "unknown": "Unknown",
            }
            paths = {
                valid.name: valid,
                missing_codec.name: missing_codec,
                wrong_codec.name: wrong_codec,
            }
            models = SimpleNamespace(
                get_names=lambda kind: list(families),
                get_file=lambda kind, name: paths.get(name, root / name),
                detect_checkpoint_base=lambda path: families[path.name],
            )
            self.assertEqual(ui.stage_b_choices(models), [valid.name])


if __name__ == "__main__":
    unittest.main()
