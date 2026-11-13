"""Independent settings profiles and notifications, without app/GPU imports."""

import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.settings import SettingsManager
from ruinedfooocus.features.generation.defaults import STARTUP_DEFAULTS


class SettingsStoreTests(unittest.TestCase):
    def test_import_does_not_require_app_arguments_or_state(self):
        with patch.dict(
            "sys.modules", {"ruinedfooocus.app.arguments": None, "ruinedfooocus.app.state": None}
        ):
            runpy.run_path(
                str(Path(__file__).resolve().parents[1] / "ruinedfooocus/core/settings.py")
            )

    def test_profiles_copy_defaults_and_notify_only_after_explicit_save(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            defaults = {"style": [], "prompt": ""}
            changed = Mock()
            first = SettingsManager("first", defaults=defaults, on_change=changed)
            second = SettingsManager("second", defaults=defaults)
            changed.assert_not_called()
            first.default_settings["style"].append("custom")
            first.default_settings["prompt"] = 'Label "CAFÉ 42".'
            first.save_settings()
            changed.assert_called_once_with()
            self.assertEqual(second.default_settings["style"], [])
            self.assertEqual(defaults["style"], [])
            self.assertEqual(SettingsManager("first").default_settings, first.default_settings)
            self.assertNotEqual(first.settings_path, second.settings_path)

    def test_saved_choices_and_unknown_keys_survive_neutral_fresh_defaults(self):
        self.assertEqual(STARTUP_DEFAULTS["style"], [])
        self.assertNotIn("advanced_mode", STARTUP_DEFAULTS)
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            path = Path(folder) / "settings/settings.json"
            path.parent.mkdir()
            saved = {
                "style": "custom",
                "prompt": "café",
                "advanced_mode": True,
                "lora_1_model": "original.safetensors",
            }
            path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8-sig")
            owner = SettingsManager(defaults=STARTUP_DEFAULTS)
            self.assertEqual(owner.default_settings["style"], ["custom"])
            self.assertEqual(owner.default_settings["prompt"], "café")
            self.assertTrue(owner.default_settings["advanced_mode"])
            self.assertEqual(owner.default_settings["lora_1_model"], "original.safetensors")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["style"], ["custom"])

    def test_invalid_document_is_reported_without_overwriting_it(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            path = Path(folder) / "settings/settings.json"
            path.parent.mkdir()
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                SettingsManager(defaults=STARTUP_DEFAULTS)
            self.assertEqual(path.read_text(), "[]")


if __name__ == "__main__":
    unittest.main()
