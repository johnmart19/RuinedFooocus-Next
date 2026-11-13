"""Resource discovery must preserve customization without writing to resources."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.core import paths
from ruinedfooocus.core.path import PathManager
from ruinedfooocus.features.chat.assistants import assistant_choices, assistant_info, context_file
from ruinedfooocus.features.prompting.presets import rewrite_presets


class ResourcePathsTests(unittest.TestCase):
    def test_generation_presets_use_user_paths_and_keep_bundled_fallback_without_copying(self):
        from ruinedfooocus.core import path as path_module

        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            root = Path(folder).resolve()
            app = root / "application"
            bundled = app / "ruinedfooocus/resources/generation_presets/example.png"
            bundled.parent.mkdir(parents=True)
            bundled.write_bytes(b"bundled example")
            owner = PathManager.__new__(PathManager)
            owner.paths = PathManager.DEFAULT_PATHS.copy()
            with (
                patch.object(path_module, "APP_ROOT", app),
                patch.object(
                    path_module,
                    "resource_path",
                    lambda *parts: app.joinpath("ruinedfooocus/resources", *parts),
                ),
            ):
                self.assertEqual(owner.get_presets(), [(bundled, "example")])
                self.assertEqual(owner.get_model_paths()["preset_path"], root / "presets")
                user = root / "presets/example.png"
                user.write_bytes(b"custom example")
                self.assertEqual(owner.get_presets(), [(user, "example")])
                legacy = app / "presets/example.png"
                legacy.parent.mkdir(parents=True)
                legacy.write_bytes(b"legacy custom example")
                self.assertEqual(owner.get_presets(), [(user, "example")])
            self.assertEqual(bundled.read_bytes(), b"bundled example")
            self.assertEqual(user.read_bytes(), b"custom example")

    def test_bundled_one_button_presets_and_styles_discover_without_writes(self):
        from ruinedfooocus.features.prompting.one_button.one_button_presets import OneButtonPresets
        from ruinedfooocus.features.prompting.one_button import csv_reader

        self.assertTrue(OneButtonPresets.DEFAULT_OBP_FILE.is_file())
        styles = Path(csv_reader.__file__).parent / "csvfiles/templates/styles.csv"
        self.assertTrue(styles.is_file())
        with tempfile.TemporaryDirectory() as folder:
            user = Path(folder) / "new-data/random_prompt/userfiles/obp_presets.json"
            with patch.object(OneButtonPresets, "OBP_FILE", user):
                presets = OneButtonPresets()
                self.assertIn("Standard", presets.opb_presets)
                self.assertIn("Weird Hats", presets.opb_presets)
                self.assertFalse(user.exists())
                presets.save_obp_preset(presets.opb_presets)
                self.assertEqual(json.loads(user.read_text(encoding="utf-8")), presets.opb_presets)

    def test_startup_markers_and_package_cache_use_writable_data_with_legacy_compatibility(self):
        from ruinedfooocus.core import dependency_cache

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            app, data = root / "application", root / "data"
            app.mkdir()
            data.mkdir()
            legacy = app / "reinstall"
            legacy.write_text("keep legacy request", encoding="utf-8")
            with (
                patch.object(paths, "APP_ROOT", app),
                patch.dict(os.environ, RF_DATA_DIR=str(data)),
                patch.object(dependency_cache, "APP_ROOT", app),
            ):
                self.assertEqual(paths.startup_marker("reinstall"), legacy)
                (data / "reinstall").touch()
                self.assertEqual(paths.startup_marker("reinstall"), data / "reinstall")
                self.assertEqual(
                    dependency_cache.cache_path(app), data / "cache/dependency-checks.json"
                )
            self.assertEqual(legacy.read_text(encoding="utf-8"), "keep legacy request")

    def test_gallery_database_uses_data_directory_without_importing_app_state(self):
        import ast
        import sqlite3
        import time
        from types import SimpleNamespace

        source = (
            Path(__file__).resolve().parents[1] / "ruinedfooocus/features/gallery/imagebrowser.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "connect_database"
        )
        scope = {"sqlite3": sqlite3, "time": time, "version": SimpleNamespace(version="test")}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            connection = scope["connect_database"]()
            try:
                actual = Path(connection.execute("PRAGMA database_list").fetchone()[2]).resolve()
                self.assertEqual(actual, (Path(folder) / "cache/images.db").resolve())
            finally:
                connection.close()

    def test_data_and_legacy_model_paths_are_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            self.assertEqual(
                paths.data_path("settings", "settings.json"),
                (Path(folder) / "settings/settings.json").resolve(),
            )
            owner = PathManager.__new__(PathManager)
            self.assertEqual(
                owner.get_abspath("../models/checkpoints"),
                (Path(folder) / "models/checkpoints").resolve(),
            )
            self.assertEqual(owner.get_abspath(folder), Path(folder))
            self.assertTrue(paths.resource_path("defaults", "styles.default").is_file())

    def test_rewrite_overrides_are_selected_once_and_never_modified(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            old = Path(folder) / "llamas/default.txt"
            old.parent.mkdir()
            old.write_text("My legacy preset\nKeep my words.", encoding="utf-8")
            before = old.read_bytes()
            choices = rewrite_presets()
            self.assertIn(("My legacy preset", str(old.resolve())), choices)
            new = Path(folder) / "prompt_presets/default.txt"
            new.parent.mkdir()
            new.write_text("My current preset\nKeep every count.", encoding="utf-8")
            choices = rewrite_presets()
            self.assertIn(("My current preset", str(new.resolve())), choices)
            self.assertNotIn(("My legacy preset", str(old.resolve())), choices)
            self.assertEqual(old.read_bytes(), before)

    def test_custom_assistant_overrides_bundle_and_general_chat_needs_no_persona(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            bot = Path(folder) / "assistants/cat"
            bot.mkdir(parents=True)
            definition = {"name": "My assistant", "greeting": "Hello", "system": "Be precise."}
            file = bot / "info.json"
            file.write_text(json.dumps(definition), encoding="utf-8")
            before = file.read_bytes()
            choices = assistant_choices()
            self.assertIn(("My assistant", "user:cat"), choices)
            self.assertEqual(assistant_info("user:cat")["system"], "Be precise.")
            self.assertEqual(assistant_info(str(bot))["system"], "Be precise.")
            self.assertEqual(assistant_info("__normal__")["chatstart"], [])
            self.assertEqual(file.read_bytes(), before)

    def test_context_uses_legacy_bundled_path_and_rejects_escape(self):
        file = context_file("chatbots/rf_support_troll/knowledge.md")
        self.assertEqual(
            file, paths.resource_path("assistants/rf_support_troll/knowledge.md").resolve()
        )
        with self.assertRaises(ValueError):
            context_file("../outside.md")


if __name__ == "__main__":
    unittest.main()
