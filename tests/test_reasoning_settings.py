"""Saved reasoning preferences are distinct from initialized model capability."""

import ast
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr

from ruinedfooocus.core.settings import SettingsManager, prospective_profile
from ruinedfooocus.core.settings_validation import validate_settings
from ruinedfooocus.features.chat.ui_reasoning_settings import (
    create_reasoning_settings,
    preference_controls,
    SAVED_PREFERENCE,
    conversation_reasoning,
)
from ruinedfooocus.features.chat.model_profiles import ProfiledChatClient
from ruinedfooocus.features.chat.model_preparation import capture_chat_request
from ruinedfooocus.features.models.model_registry import load_registry
from ui_test_helpers import function_from

ROOT = Path(__file__).resolve().parents[1]


class ReasoningSettingsTests(unittest.TestCase):
    def adapter(self, values):
        artifact = load_registry().artifact(
            "existing:llm/Qwen3.5-9B-ultra-uncensored-heretic-v2-Q4_K_M.gguf"
        )
        return ProfiledChatClient(
            Mock(), artifact, values, template="{{ enable_thinking }}", initialized=True
        )

    def test_inherited_and_partial_conversation_choices_keep_only_supported_defaults(self):
        saved = {"llm_chat_reasoning_mode": "Off", "llm_chat_reasoning_effort": "High"}
        self.assertIsNone(conversation_reasoning(SAVED_PREFERENCE, SAVED_PREFERENCE, saved))
        boolean = {"modes": ("Auto", "Off", "On"), "efforts": ()}
        self.assertEqual(
            conversation_reasoning(SAVED_PREFERENCE, "Auto", saved, boolean),
            {"mode": "Off", "effort": "Auto"},
        )
        self.assertEqual(
            conversation_reasoning("Auto", SAVED_PREFERENCE, saved, boolean),
            {"mode": "Auto", "effort": "Auto"},
        )
        effort = {"modes": ("Auto", "On"), "efforts": ("Low", "Medium", "High")}
        self.assertEqual(
            conversation_reasoning(SAVED_PREFERENCE, "Low", saved, effort),
            {"mode": "Auto", "effort": "Low"},
        )
        self.assertEqual(
            saved, {"llm_chat_reasoning_mode": "Off", "llm_chat_reasoning_effort": "High"}
        )

    def test_save_updates_next_captured_load_without_overwriting_conversation_auto(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            values = {"llm_chat_reasoning_mode": "Auto", "llm_chat_reasoning_effort": "Auto"}
            manager = SettingsManager("reasoning", defaults=values)
            paths = SimpleNamespace(paths={}, DEFAULT_PATHS={}, set_settings_path=Mock())
            save = function_from(
                "ruinedfooocus/ui/ui_settings.py",
                "save_clicked",
                {
                    "state": {
                        "setting_name": ["llm_chat_reasoning_mode", "llm_chat_reasoning_effort"]
                    },
                    "settings": manager,
                    "path_manager": paths,
                    "validate_settings": validate_settings,
                    "gr": SimpleNamespace(Error=ValueError, Info=Mock()),
                    "models": SimpleNamespace(offline=False),
                    "performance_settings": SimpleNamespace(
                        model_recommendations=SimpleNamespace(offline=False)
                    ),
                },
            )
            tree = ast.parse(
                (ROOT / "ruinedfooocus/features/chat/llama_pipeline.py").read_text(encoding="utf-8")
            )
            owner_class = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "pipeline"
            )
            owner_class.body = [
                node
                for node in owner_class.body
                if isinstance(node, ast.FunctionDef)
                and node.name in ("current_model_settings", "process")
            ]
            namespace = {"settings": manager}
            exec(
                compile(
                    ast.Module(body=[owner_class], type_ignores=[]), "llama_pipeline.py", "exec"
                ),
                namespace,
            )
            owner = namespace["pipeline"]()
            owner.llm, owner.loaded_paths = self.adapter(values), ()
            owner.runtime_name, owner.runtime_label, owner.fallback_reason = "llama.cpp", "CPU", ""
            model = Path(folder) / "fixture.gguf"
            model.write_bytes(b"metadata-only fixture; never loaded by a backend")
            owner.model_settings = owner.current_model_settings(str(model))

            def load(model, progress=None, *, prepared, **options):
                owner.llm = self.adapter(prepared.settings)
                owner.model_settings = owner.current_model_settings(model, values=prepared.settings)

            owner.load_base_model = Mock(side_effect=load)
            catalogue = SimpleNamespace(DOWNLOADABLE_FILES={})
            auto_override = ("Auto", "Auto")

            def request(mode, effort):
                data = capture_chat_request(
                    {"model": str(model), "load_only": True},
                    manager.default_settings,
                    catalogue,
                    {},
                )
                loaded = owner.process(data)
                self.assertTrue(loaded["ready"])
                self.assertEqual(loaded["runtime_name"], "llama.cpp")
                reasoning = conversation_reasoning(
                    mode, effort, manager.default_settings, owner.llm.reasoning_capability
                )
                return owner.llm.parameters(
                    {"_rf_reasoning": reasoning} if reasoning is not None else {}
                )

            launch = SimpleNamespace(args=SimpleNamespace(settings="reasoning", offline=True))
            with patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.arguments": launch,
                    "ruinedfooocus.core.util": SimpleNamespace(
                        url_to_filename=Mock(
                            side_effect=AssertionError("No URL context in this fixture")
                        )
                    ),
                },
            ):
                save("Off", "Auto")
                self.assertEqual(
                    SettingsManager("reasoning").default_settings["llm_chat_reasoning_mode"], "Off"
                )
                self.assertIs(
                    request(SAVED_PREFERENCE, SAVED_PREFERENCE)["chat_template_kwargs"][
                        "enable_thinking"
                    ],
                    False,
                )
                owner.load_base_model.assert_called_once()
                self.assertNotIn("chat_template_kwargs", request(*auto_override))
                owner.load_base_model.assert_called_once()
                save("On", "Auto")
                self.assertIs(
                    request(SAVED_PREFERENCE, SAVED_PREFERENCE)["chat_template_kwargs"][
                        "enable_thinking"
                    ],
                    True,
                )
                self.assertEqual(owner.load_base_model.call_count, 2)
                self.assertNotIn("chat_template_kwargs", request(*auto_override))
                self.assertEqual(auto_override, ("Auto", "Auto"))

    def test_null_rewrite_mode_matches_absent_legacy_policy(self):
        for values in (
            {"llm_rewrite_reasoning": "Low"},
            {"llm_rewrite_reasoning": "Low", "llm_rewrite_reasoning_mode": None},
        ):
            with self.subTest(values=values):
                adapter = self.adapter(validate_settings(values))
                self.assertIs(
                    adapter.parameters({"_rf_task": "rewrite"})["chat_template_kwargs"][
                        "enable_thinking"
                    ],
                    False,
                )
                registered = {}
                with gr.Blocks() as app:
                    create_reasoning_settings(values, registered.__setitem__, gr.State(None))
                self.assertTrue(registered["llm_rewrite_reasoning"].visible)
                updates = preference_controls(
                    None, "Auto", "Auto", values.get("llm_rewrite_reasoning_mode"), "Auto"
                )
                self.assertEqual(updates[2]["value"], "")
                self.assertFalse(updates[4]["visible"])

    def test_unloaded_model_retains_inactive_preferences(self):
        updates = preference_controls(None, "On", "High", "Off", "Low")
        self.assertEqual([item["value"] for item in updates[:4]], ["On", "High", "Off", "Low"])
        self.assertTrue(all(not item["interactive"] for item in updates[:4]))
        self.assertFalse(updates[4]["visible"])

    def test_gpt_oss_omits_off_except_saved_inactive_preference(self):
        cap = {
            "modes": ["Auto", "On"],
            "efforts": ["Low", "Medium", "High"],
            "control": "effort",
            "reason": "verified",
        }
        with patch(
            "ruinedfooocus.features.chat.ui_reasoning_settings.t", side_effect=lambda value: value
        ):
            updates = preference_controls(cap, "Auto", "Low", "Off", "High")
        self.assertNotIn("Off", updates[0]["choices"])
        self.assertIn(("Saved preference (inactive): Off", "Off"), updates[2]["choices"])
        self.assertFalse(updates[0]["visible"])
        self.assertFalse(updates[2]["visible"])
        self.assertTrue(updates[3]["interactive"])

    def test_boolean_template_has_no_effort_controls(self):
        cap = {
            "modes": ["Auto", "Off", "On"],
            "efforts": [],
            "control": "enable_thinking",
            "reason": "verified",
        }
        updates = preference_controls(cap, "Off", "Auto", "", "Auto")
        self.assertTrue(updates[0]["interactive"])
        self.assertFalse(updates[1]["interactive"])
        self.assertFalse(updates[3]["interactive"])
        self.assertTrue(updates[4]["visible"])

    def test_unrelated_save_preserves_absent_rewrite_mode(self):
        current = {"llm_rewrite_reasoning": "Low", "prompt": "literal"}
        values = validate_settings(
            {"llm_rewrite_reasoning_mode": "", "llm_rewrite_reasoning_effort": "Auto"}
        )
        saved, _ = prospective_profile(current, {}, values, set())
        self.assertNotIn("llm_rewrite_reasoning_mode", saved)
        self.assertEqual(saved["llm_rewrite_reasoning"], "Low")
        self.assertEqual(saved["prompt"], "literal")
        current["llm_rewrite_reasoning_mode"] = "On"
        saved, _ = prospective_profile(current, {}, values, set())
        self.assertNotIn("llm_rewrite_reasoning_mode", saved)

    def test_invalid_modes_and_efforts_are_rejected(self):
        for key in (
            "llm_chat_reasoning_mode",
            "llm_chat_reasoning_effort",
            "llm_rewrite_reasoning_mode",
            "llm_rewrite_reasoning_effort",
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_settings({key: "invented"})

    def test_panel_binds_per_browser_state_without_model_import_or_default_migration(self):
        values = {"llm_rewrite_reasoning": "Medium"}
        registered = {}
        with gr.Blocks() as app:
            capability = gr.State(None)
            controls = create_reasoning_settings(values, registered.__setitem__, capability)
        self.assertEqual(len(registered), 5)
        self.assertEqual(controls[2].value, "")
        self.assertEqual(values, {"llm_rewrite_reasoning": "Medium"})
        callback = next(fn for fn in app.fns.values() if fn.fn is preference_controls)
        self.assertIs(callback.inputs[0], capability)
        self.assertEqual(len(callback.outputs), 7)


if __name__ == "__main__":
    unittest.main()
