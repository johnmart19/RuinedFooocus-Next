"""The real component editor preserves profiles and reports unresolved choices."""

import copy
import json
import runpy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core import settings as settings_module
from ruinedfooocus.features.models import component_profiles as profiles
from ruinedfooocus.features.models import ui_model_components


class ComponentOverrideTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary)
        self.checkpoint = self.header(self.root / "model.safetensors", {})
        self.identity = profiles.checkpoint_identity(self.checkpoint)
        self.folder = self.root / "profile"
        self.enterContext(patch.object(settings_module, "profile_folder", return_value=self.folder))
        self.enterContext(
            patch.object(
                settings_module, "settings_path", return_value=self.folder / "settings.json"
            )
        )
        self.settings = settings_module.SettingsManager(
            profile="component-test",
            defaults={
                "prompt": "keep exactly",
                "theme": "custom",
                "clip_l": "legacy.safetensors",
                "image_component_profiles": {self.identity: {"clip_g": "other.safetensors"}},
            },
        )
        self.manager = SimpleNamespace(
            paths={"path_clip": [str(self.root / "encoders")], "path_vae": []},
            get_abspath=Path,
            DOWNLOADABLE_FILES={},
            EXTENSIONS=[".safetensors", ".gguf"],
        )
        self.enterContext(
            patch.object(ui_model_components, "selected", return_value=(self.checkpoint, "SDXL"))
        )
        with patch.dict(
            "sys.modules",
            {
                "ruinedfooocus.app.state": SimpleNamespace(
                    settings=self.settings, path_manager=self.manager
                )
            },
        ):
            self.module = runpy.run_path(
                str(
                    Path(__file__).resolve().parents[1]
                    / "ruinedfooocus/features/models/ui_component_overrides.py"
                )
            )

    def header(self, path, tensors):
        payload = json.dumps(tensors).encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(len(payload).to_bytes(8, "little") + payload)
        return path

    def test_save_commits_only_this_checkpoint_role_and_keeps_legacy_settings(self):
        original = copy.deepcopy(self.settings.default_settings)
        pending = {"clip_g": "stale-session.safetensors"}
        with patch.object(self.settings, "save_profile", wraps=self.settings.save_profile) as save:
            choices, scope = self.module["save_choice"](
                "model", "clip_l", "", False, pending, self.identity
            )
        updated = copy.deepcopy(original)
        updated[profiles.PROFILE_KEY][self.identity]["clip_l"] = None
        save.assert_called_once_with(updated, self.manager.paths, profile="component-test")
        self.assertEqual(self.settings.default_settings, updated)
        self.assertEqual(json.loads((self.folder / "settings.json").read_text()), updated)
        self.assertEqual(json.loads((self.folder / "paths.json").read_text()), self.manager.paths)
        self.assertEqual(self.settings.name, "component-test")
        self.assertEqual(choices, {"clip_g": "stale-session.safetensors", "clip_l": None})
        self.assertEqual(scope, self.identity)
        self.assertEqual(pending, {"clip_g": "stale-session.safetensors"})

    def test_failed_save_preserves_memory_profile_disk_and_pending_choices(self):
        original = copy.deepcopy(self.settings.default_settings)
        before = (self.folder / "settings.json").read_bytes()
        pending = {"clip_g": "stale-session.safetensors"}
        with patch.object(settings_module, "save_profile", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(self.module["gr"].Error, "disk full"):
                self.module["save_choice"]("model", "clip_l", "", False, pending, self.identity)
        self.assertEqual(self.settings.default_settings, original)
        self.assertEqual(self.settings.name, "component-test")
        self.assertEqual((self.folder / "settings.json").read_bytes(), before)
        self.assertFalse((self.folder / "paths.json").exists())
        self.assertEqual(pending, {"clip_g": "stale-session.safetensors"})

    def test_saved_missing_or_unverified_choice_is_shown_instead_of_automatic(self):
        self.header(self.root / "encoders" / "custom.safetensors", {})
        with patch.dict("sys.modules", {"comfy.sd": None}):
            for value, status in (
                ("gone.safetensors", "Missing"),
                ("custom.safetensors", "Unverified"),
            ):
                with self.subTest(value=value):
                    choices = {"clip_l": value}
                    result = self.module["role_choices"]("model", "clip_l", choices, self.identity)
                    self.assertEqual(result["value"], value)
                    label = next(label for label, item in result["choices"] if item == value)
                    self.assertIn(status, label)
                    self.assertIn(("Automatic", ""), result["choices"])
                    self.assertEqual(choices, {"clip_l": value})

    def test_alternatives_are_scoped_to_native_role_and_effective_companions(self):
        from ruinedfooocus.features.models import component_candidates

        candidates = [
            {"name": "compatible.safetensors", "compatibility": "compatible"},
            {"name": "unknown.gguf", "compatibility": "unknown"},
            {"name": "other-family.safetensors", "compatibility": "incompatible"},
        ]
        with patch.object(
            component_candidates, "installed_candidates", return_value=candidates
        ) as scan:
            ordinary = self.module["role_choices"]("model", "clip_l", {}, self.identity)
            advanced = self.module["role_choices"]("model", "clip_l", {}, self.identity, True)
        self.assertIn(("compatible.safetensors", "compatible.safetensors"), ordinary["choices"])
        self.assertNotIn("unknown.gguf", {value for _, value in ordinary["choices"]})
        self.assertIn(("unknown.gguf · Unverified", "unknown.gguf"), advanced["choices"])
        self.assertNotIn("other-family.safetensors", {value for _, value in advanced["choices"]})
        self.assertIn("Effective: legacy.safetensors", ordinary["info"])
        self.assertEqual(scan.call_args.args, ("clip_l", self.manager))
        self.assertTrue(scan.call_args.kwargs["companion_rows"])

    def test_component_selection_source_reports_bundled_saved_profile_and_family_precedence(self):
        source = ui_model_components.selection_source
        row = {"key": "clip_l", "status": "Installed"}
        self.assertEqual(source(row, {}, {}), "Family default")
        self.assertEqual(source(row, {}, {"clip_l": "legacy"}), "Profile default")
        self.assertEqual(source(row, {"clip_l": None}, {"clip_l": "legacy"}), "Family default")
        self.assertEqual(source(row, {"clip_l": "custom"}, {}), "Saved for this checkpoint")
        self.assertEqual(
            source({**row, "status": "Embedded in checkpoint"}, {"clip_l": "custom"}, {}),
            "Built into checkpoint",
        )
        self.assertEqual(source({**row, "locked": True}, {"clip_l": "custom"}, {}), "Local bundle")

    def test_legacy_unresolved_choice_remains_visible_until_explicit_reset(self):
        result = self.module["role_choices"]("model", "clip_l", {}, self.identity)
        self.assertEqual(result["value"], "legacy.safetensors")
        self.assertIn(
            "Missing", dict((value, label) for label, value in result["choices"])[result["value"]]
        )
        self.assertEqual(
            self.module["role_choices"]("model", "clip_l", {"clip_l": None}, self.identity)[
                "value"
            ],
            "",
        )

    def test_incompatible_current_file_stays_labelled_and_cannot_be_resaved(self):
        self.header(self.root / "encoders" / "wrong.safetensors", {})
        detector = SimpleNamespace(detect_te_model=lambda shapes: SimpleNamespace(name="CLIP_G"))
        pending = {"clip_l": "wrong.safetensors"}
        with patch.dict("sys.modules", {"comfy.sd": detector}):
            result = self.module["role_choices"]("model", "clip_l", pending, self.identity)
            self.assertEqual(result["value"], "wrong.safetensors")
            self.assertIn(
                ("Current: wrong.safetensors · Incompatible", "wrong.safetensors"),
                result["choices"],
            )
            with self.assertRaisesRegex(self.module["gr"].Error, "different"):
                self.module["save_choice"](
                    "model", "clip_l", "wrong.safetensors", True, pending, self.identity
                )
        self.assertNotIn(
            "clip_l", profiles.saved_choices(self.settings.default_settings, self.checkpoint)
        )

    def test_editor_callbacks_hide_unrelated_progress_overlays(self):
        import gradio as gr

        with gr.Blocks(analytics_enabled=False) as demo:
            model, overrides, scope = gr.State(), gr.State(), gr.State()
            self.module["create_ui"](model, overrides, scope, lambda *args: None, [])
        dependencies = demo.get_config_file()["dependencies"]
        self.assertEqual(len(dependencies), 4)
        self.assertTrue(all(item["show_progress"] == "hidden" for item in dependencies))

    def test_saved_component_keeps_its_role_without_retaining_obsolete_roles(self):
        import gradio as gr

        refreshed = (
            "inventory",
            gr.update(choices=[("Encoder", "clip_l"), ("VAE", "vae")], value="clip_l"),
        )
        refresh = Mock(return_value=refreshed)
        with gr.Blocks(analytics_enabled=False) as demo:
            model, overrides, scope = gr.State(), gr.State(), gr.State()
            self.module["create_ui"](model, overrides, scope, refresh, [gr.State()])
        callback = next(fn for fn in demo.fns.values() if fn.fn.__name__ == "refresh_choice")
        self.assertEqual(len(callback.inputs), 4)
        self.assertEqual(callback.fn("model", {}, "scope", "vae")[-1]["value"], "vae")
        refresh.assert_called_once_with("model", {}, "scope")
        self.assertEqual(callback.fn("model", {}, "scope", "removed")[-1]["value"], "clip_l")
        self.assertEqual(refreshed[-1]["value"], "clip_l")

    def test_explicit_refresh_keeps_vae_role_and_saved_choice_but_drops_obsolete_roles(self):
        import gradio as gr

        refreshed = (
            [["VAE", "compatible-vae.safetensors", "Installed · Custom"]],
            {"vae": {"file": "compatible-vae.safetensors", "selection": "Custom"}},
            "Components present",
            gr.update(interactive=False),
            {"vae": "compatible-vae.safetensors"},
            self.identity,
            gr.update(choices=[("Encoder", "clip_l"), ("VAE", "vae")], value="clip_l"),
        )
        with patch.object(ui_model_components, "refresh", return_value=refreshed) as refresh:
            result = ui_model_components.refresh_selected_role(
                "model", {"vae": "compatible-vae.safetensors"}, self.identity, "vae"
            )
            self.assertEqual(result[-1]["value"], "vae")
            self.assertEqual(result[:-1], refreshed[:-1])
            refresh.assert_called_once_with(
                "model", {"vae": "compatible-vae.safetensors"}, self.identity
            )
            self.assertEqual(
                ui_model_components.refresh_selected_role("model", {}, self.identity, "removed")[
                    -1
                ]["value"],
                "clip_l",
            )
        self.assertEqual(refreshed[-1]["value"], "clip_l")


if __name__ == "__main__":
    unittest.main()
