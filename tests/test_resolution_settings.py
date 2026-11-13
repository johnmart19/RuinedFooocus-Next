"""Model size changes preserve explicit controls and source/replay ownership."""

import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr

from ruinedfooocus.core.settings import SettingsManager
from ruinedfooocus.features.generation.defaults import STARTUP_DEFAULTS, generation_defaults
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.resolutions import ResolutionSettings

ROOT = Path(__file__).resolve().parents[1]


def callback(name, scope):
    source = ROOT / "ruinedfooocus/ui/app.py"
    node = next(
        node
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope[name]


class ResolutionSettingsTests(unittest.TestCase):
    def test_shared_legacy_image_size_stays_available_and_saved_rows_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "resolutions.json"
            rows = {
                "Shared size": {
                    "width": 832,
                    "height": 640,
                    "image_models": ["SDXL"],
                    "video_models": ["retired"],
                    "custom_note": "preserve",
                },
                "Video only": {"width": 768, "height": 512, "video_models": ["retired"]},
            }
            path.write_text(json.dumps(rows), encoding="utf-8")
            original = path.read_bytes()
            with patch.object(ResolutionSettings, "RESOLUTIONS_FILE", path):
                reader = ResolutionSettings()
                self.assertIn("832x640 (Shared size)", reader.choices_for_model("SDXL"))
                self.assertNotIn("Video only", reader.base_ratios)
                self.assertEqual(path.read_bytes(), original)
                reader.save_resolutions(dict(reader.base_ratios))
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), rows)

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        local = patch.object(ResolutionSettings, "RESOLUTIONS_FILE", self.root / "resolutions.json")
        local.start()
        self.addCleanup(local.stop)
        self.resolutions = ResolutionSettings()
        self.publisher = SimpleNamespace(
            metadata=Mock(return_value={}), cached_resolution=Mock(return_value=(None, None))
        )
        self.performance = SimpleNamespace(
            model_recommendations=self.publisher, family_preset=Mock(return_value="SDXL")
        )

    def test_automatic_switch_uses_recommendation_but_manual_builtin_is_preserved(self):
        current = "1152x896 (4:3)"
        self.assertIn(current, self.resolutions.aspect_ratios)
        choices, selection, automatic = self.resolutions.model_selection("SDXL", current, True)
        self.assertEqual(self.resolutions.aspect_ratios[selection], (1024, 1024))
        self.assertTrue(automatic)
        _, selection, automatic = self.resolutions.model_selection("SDXL", current, False)
        self.assertEqual(selection, current)
        self.assertFalse(automatic)
        self.assertEqual(choices[-1], "Custom...")

    def test_deliberate_model_specific_size_survives_switch_without_becoming_recommendation(self):
        original = self.resolutions.selection_for_model("QwenImage", None)
        choices, selected, automatic = self.resolutions.model_selection("SDXL", original, False)
        self.assertEqual(selected, original)
        self.assertIn(selected, choices)
        self.assertFalse(automatic)
        self.assertNotIn(
            self.resolutions.aspect_ratios[original], self.resolutions.recommended_sizes("SDXL")
        )

    def test_custom_replay_takes_ownership_until_automatic_is_explicitly_enabled(self):
        _, selected, automatic = self.resolutions.model_selection("HiDreamO1", "Custom...", True)
        self.assertEqual((selected, automatic), ("Custom...", False))
        _, selected, automatic = self.resolutions.model_selection(
            "HiDreamO1", "Custom...", True, force=True
        )
        self.assertEqual(self.resolutions.aspect_ratios[selected], (2048, 2048))
        self.assertTrue(automatic)

    def test_checkpoint_variant_recipe_is_used_and_unknown_has_no_invented_size(self):
        self.performance.family_preset.return_value = "HiDream O1 Dev"
        sizes, source = self.resolutions.recommendation(
            "HiDreamO1", "renamed.gguf", self.performance
        )
        self.assertEqual(sizes[0], (2048, 2048))
        self.assertEqual(source, "Local HiDream O1 Dev family recipe")
        self.performance.family_preset.assert_called_with("HiDreamO1", "renamed.gguf")
        self.performance.family_preset.return_value = None
        self.assertEqual(
            self.resolutions.recommended_sizes("Flux3", "renamed.gguf", self.performance), []
        )
        _, selected, automatic = self.resolutions.model_selection(
            "Flux3", "Custom...", True, "renamed.gguf", self.performance, force=True
        )
        self.assertEqual(selected, "1024x1024 (Generic fallback)")
        self.assertTrue(automatic)

    def test_empty_qwen_recipe_drops_the_previous_automatic_size_but_keeps_manual_ownership(self):
        previous = self.resolutions.selection_for_model("HiDreamO1", None)
        self.assertEqual(self.resolutions.aspect_ratios[previous], (2048, 2048))
        self.performance.family_preset.return_value = None
        transition = callback(
            "model_resolution",
            dict(
                gr=gr, resolution_settings=self.resolutions, performance_settings=self.performance
            ),
        )
        automatic, enabled = transition("QwenImage", "uncached.gguf", previous, True)
        self.assertEqual(automatic["value"], "1024x1024 (Generic fallback)")
        self.assertNotIn(previous, automatic["choices"])
        self.assertTrue(enabled)
        manual, enabled = transition("QwenImage", "uncached.gguf", previous, False)
        self.assertEqual(manual["value"], previous)
        self.assertFalse(enabled)

    def test_signed_publisher_resolution_precedes_local_variant_and_stale_file_does_not(self):
        model = self.root / "renamed.safetensors"
        model.write_bytes(b"inert local file, not neural weights")
        metadata = {
            "_rf_file_signature": file_signature(model),
            "generationConfig": {"width": 896, "height": 1152},
        }
        self.publisher.metadata.return_value = metadata
        sizes, source = self.resolutions.recommendation("SDXL", str(model), self.performance)
        self.assertEqual(sizes, [(896, 1152)])
        self.assertIn("publisher", source)
        model.write_bytes(b"replacement with distinct signature")
        self.assertEqual(
            self.resolutions.recommended_sizes("SDXL", str(model), self.performance)[0],
            (1024, 1024),
        )

    def test_publisher_resolution_is_strict_and_never_reads_example_images(self):
        model = self.root / "model.safetensors"
        model.write_bytes(b"inert")
        for invalid in ((True, 1024), (1024.0, 1024), (255, 1024), (1025, 1024), (8192, 1024)):
            with self.subTest(invalid=invalid):
                self.publisher.metadata.return_value = {
                    "_rf_file_signature": file_signature(model),
                    "generationConfig": dict(zip(("width", "height"), invalid)),
                    "images": [{"width": 768, "height": 512}],
                }
                self.assertEqual(
                    self.resolutions.recommended_sizes("SDXL", str(model), self.performance)[0],
                    (1024, 1024),
                )
        self.publisher.cached_resolution.return_value = ((768, 1024), "Civitai version 123")
        self.assertEqual(
            self.resolutions.recommendation("SDXL", str(model), self.performance),
            ([(768, 1024)], "Civitai version 123"),
        )

    def test_successful_refresh_with_no_size_does_not_resurrect_old_local_publisher_size(self):
        model = self.root / "model.safetensors"
        model.write_bytes(b"inert")
        self.publisher.metadata.return_value = {
            "_rf_file_signature": file_signature(model),
            "generationConfig": {"width": 896, "height": 1152},
        }
        self.publisher.cached_resolution.return_value = (None, "Civitai version 123")
        sizes, source = self.resolutions.recommendation("SDXL", str(model), self.performance)
        self.assertEqual((sizes[0], source), ((1024, 1024), "Local SDXL family recipe"))

    def test_source_frame_preview_uses_native_geometry_without_mutating_guidance(self):
        guidance = {"denoise": 0.31, "image_resize_mode": "preserve"}
        before = copy.deepcopy(guidance)
        text = self.resolutions.recommendation_text(
            "SDXL",
            selection="Custom...",
            width=1024,
            height=1024,
            automatic=False,
            source_size=(1600, 900),
            framing=guidance["image_resize_mode"],
        )
        self.assertIn("Selected size: 1024 × 1024", text)
        self.assertIn("Planned source frame: 1365 × 768", text)
        self.assertIn("size override is preserved", text)
        self.assertIn("working pixel budget", text)
        self.assertEqual(guidance, before)

    def test_real_ui_model_callback_preserves_replay_and_manual_then_follows_model(self):
        transition = callback(
            "model_resolution",
            dict(
                gr=gr, resolution_settings=self.resolutions, performance_settings=self.performance
            ),
        )
        update, automatic = transition("SDXL", "model", "Custom...", True)
        self.assertEqual(update["value"], "Custom...")
        self.assertFalse(automatic)
        update, automatic = transition("SDXL", "model", "1152x896 (4:3)", False)
        self.assertEqual(update["value"], "1152x896 (4:3)")
        update, automatic = transition("SDXL", "model", "Custom...", True, force=True)
        self.assertEqual(self.resolutions.aspect_ratios[update["value"]], (1024, 1024))
        self.assertTrue(automatic)

    def test_inpaint_does_not_show_a_stale_image_tools_source_frame(self):
        describe = callback(
            "describe_resolution",
            dict(
                ui_controlnet=SimpleNamespace(tool_kind=lambda *_: "img2img"),
                resolution_settings=SimpleNamespace(
                    recommendation_text=lambda *args, **kwargs: kwargs
                ),
                performance_settings=self.performance,
            ),
        )
        values = (
            None,
            "SDXL",
            "model",
            "Custom...",
            1024,
            1024,
            False,
            SimpleNamespace(size=(1600, 900)),
            "Image to image",
            "Img2img",
            "preserve",
        )
        self.assertIsNone(describe(*values, True)["source_size"])
        self.assertEqual(describe(*values, False)["source_size"], (1600, 900))

    def test_depth_map_size_hints_ignore_checkpoint_and_input_upscale(self):
        tools = SimpleNamespace(tool_kind=lambda *_: "depth-map")
        describe = callback(
            "describe_resolution", dict(ui_controlnet=tools, localize=lambda text, locale: text)
        )
        upscale = callback("update_upscale_size", dict(ui_controlnet=tools, t=lambda text: text))
        expected = "Depth map keeps the source image size."
        self.assertEqual(
            describe(
                None,
                "SD_X4Upscaler",
                "ignored",
                "Custom...",
                512,
                512,
                True,
                None,
                "Depth map",
                "Depth",
                "preserve",
                False,
            ),
            expected,
        )
        self.assertEqual(upscale("4×", "Input image only", "Custom...", 512, 512, None), expected)

    def test_existing_profiles_migrate_as_manual_and_fresh_profiles_are_automatic(self):
        for stored, expected in (
            ({}, True),
            ({"resolution": "1152x896 (4:3)"}, False),
            ({"resolution": "Custom...", "resolution_auto": True}, True),
        ):
            manager = SettingsManager.__new__(SettingsManager)
            manager.settings_path = self.root / "settings.json"
            manager._defaults = copy.deepcopy(STARTUP_DEFAULTS)
            with (
                patch(
                    "ruinedfooocus.core.settings.read_profile", return_value=copy.deepcopy(stored)
                ),
                patch("ruinedfooocus.core.settings.save_json"),
            ):
                manager.load_settings()
            with self.subTest(stored=stored):
                self.assertIs(manager.default_settings["resolution_auto"], expected)
        for automatic in (True, False):
            saved = generation_defaults(
                {
                    "resolution_auto": automatic,
                    "aspect_ratios_selection": "Custom...",
                    "custom_width": 832,
                    "custom_height": 1216,
                }
            )
            self.assertIs(saved["resolution_auto"], automatic)
            self.assertEqual((saved["custom_width"], saved["custom_height"]), (832, 1216))

    def test_profile_rejects_invalid_automatic_intent(self):
        from ruinedfooocus.core.settings_schema import validate_known

        with self.assertRaises(ValueError):
            validate_known({"resolution_auto": "false"})

    def test_recommended_menu_excludes_generic_sizes_but_preserves_numeric_manual_choice(self):
        choices = self.resolutions.choices_for_model("SDXL")
        recommended = set(self.resolutions.recommended_sizes("SDXL"))
        self.assertEqual(
            {self.resolutions.aspect_ratios[label] for label in choices[:-1]}, recommended
        )
        _, selected, automatic = self.resolutions.model_selection("SDXL", "1920x1080", False)
        self.assertEqual((selected, automatic), ("1920x1080", False))
        self.assertEqual(self.resolutions.aspect_ratios[selected], (1920, 1080))

    def test_saved_image_sizes_are_profile_and_family_scoped_without_writing_defaults(self):
        original = self.resolutions.DEFAULT_RESOLUTIONS_FILE.read_bytes()
        with patch(
            "ruinedfooocus.core.profile_store.profile_folder",
            side_effect=lambda profile: self.root / (profile or "default"),
        ):
            self.resolutions.set_profile("art")
            saved = self.resolutions.save_custom("Cinema", 1920, 1080, "SDXL")
            self.assertIn(saved, self.resolutions.choices_for_model("SDXL"))
            self.assertNotIn(saved, self.resolutions.choices_for_model("QwenImage21"))
            self.assertFalse((self.root / "art" / "video_sizes.json").exists())
            self.resolutions.set_profile("portraits")
            self.assertNotIn(saved, self.resolutions.choices_for_model("SDXL"))
            self.resolutions.set_profile("art")
            self.assertIn(saved, self.resolutions.choices_for_model("SDXL"))
            self.resolutions.remove_custom(saved)
            self.assertNotIn(saved, self.resolutions.choices_for_model("SDXL"))
        self.assertEqual(self.resolutions.DEFAULT_RESOLUTIONS_FILE.read_bytes(), original)

    def test_invalid_custom_dimensions_do_not_replace_saved_file(self):
        with patch("ruinedfooocus.core.profile_store.profile_folder", return_value=self.root):
            self.resolutions.save_custom("Exact", 1920, 1080, "SDXL")
            before = self.resolutions._custom_file().read_bytes()
            for invalid in (0, -1, True, 1080.5):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    self.resolutions.save_custom("Exact", invalid, 1080, "SDXL")
            self.assertEqual(self.resolutions._custom_file().read_bytes(), before)

    def test_corrupt_custom_size_file_keeps_model_defaults_and_cannot_be_overwritten(self):
        with patch("ruinedfooocus.core.profile_store.profile_folder", return_value=self.root):
            path = self.resolutions._custom_file()
            path.write_text('{"schema_version":99,"sizes":[]}', encoding="utf-8")
            before = path.read_bytes()
            self.resolutions.load_resolutions()
            self.assertTrue(self.resolutions.choices_for_model("SDXL"))
            self.assertTrue(self.resolutions.custom_errors)
            with self.assertRaises(ValueError):
                self.resolutions.save_custom("Repair", 1920, 1080, "SDXL")
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
