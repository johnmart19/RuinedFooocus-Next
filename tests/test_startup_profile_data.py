"""Pre-install startup policy reads recovered JSON without feature initialization."""

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.core.profile_store import read_profile_data
from ruinedfooocus.packaging.startup_options import saved_startup_policy


ROOT = Path(__file__).resolve().parents[1]


class StartupProfileDataTests(unittest.TestCase):
    def profile(self, folder, values):
        path = Path(folder) / "settings/saved/settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values), encoding="utf-8")
        return path

    def test_startup_policy_never_imports_application_feature_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            profile = Path(folder) / "settings/saved/settings.json"
            profile.parent.mkdir(parents=True)
            profile.write_text(
                json.dumps(
                    {
                        "startup_offline": True,
                        "startup_skip_dependency_install": False,
                        "grid_columns": 0,
                    }
                ),
                encoding="utf-8",
            )
            script = """
import importlib.abc
import sys

class FeatureImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        roots = ('ruinedfooocus.app', 'ruinedfooocus.features', 'ruinedfooocus.backends',
                 'ruinedfooocus.core.settings_schema', 'torch', 'comfy')
        if any(fullname == name or fullname.startswith(name + '.') for name in roots):
            raise AssertionError('Startup imported application validation: ' + fullname)

sys.meta_path.insert(0, FeatureImportsForbidden())
from ruinedfooocus.packaging.startup_options import (
    saved_startup_policy, offline_startup, skip_dependency_install,
)
options = ['--settings=saved']
assert saved_startup_policy(options) == {
    'startup_offline': True, 'startup_skip_dependency_install': False,
}
assert offline_startup(options) is True
assert skip_dependency_install(options) is True
"""
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env=dict(
                    os.environ, RF_DATA_DIR=folder, RF_OFFLINE="0", PYTHONDONTWRITEBYTECODE="1"
                ),
                text=True,
                capture_output=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_startup_flags_reject_non_booleans_without_rewriting_the_profile(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            for field in ("startup_offline", "startup_skip_dependency_install"):
                for value in (0, 1, "false", None, [], {}):
                    with self.subTest(field=field, value=value):
                        path = self.profile(folder, {field: value})
                        original = path.read_bytes()
                        with self.assertRaisesRegex(ValueError, field):
                            saved_startup_policy(["--settings=saved"])
                        self.assertEqual(path.read_bytes(), original)

    def test_profile_data_rejects_malformed_or_non_object_json(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            path = self.profile(folder, {})
            for raw in ("{broken", "[]", "null"):
                with self.subTest(raw=raw):
                    path.write_text(raw, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_profile_data(path)
                    with self.assertRaises(ValueError):
                        saved_startup_policy(["--settings=saved"])
                    self.assertEqual(path.read_text(encoding="utf-8"), raw)

    def test_startup_policy_recovers_original_bytes_before_reading_flags(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            path = self.profile(folder, {"startup_offline": True})
            original = (
                b'\xef\xbb\xbf{"startup_offline":false,"startup_skip_dependency_install":true}'
            )
            journal = path.parent / ".profile-save.json"
            journal.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "before": {
                            "settings.json": base64.b64encode(original).decode("ascii"),
                            "paths.json": None,
                        },
                    }
                ),
                encoding="utf-8",
            )
            paths = path.parent / "paths.json"
            paths.write_text('{"path_outputs":"uncommitted"}', encoding="utf-8")
            self.assertEqual(
                saved_startup_policy(["--settings=saved"]),
                {"startup_offline": False, "startup_skip_dependency_install": True},
            )
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(paths.exists())
            self.assertFalse(journal.exists())


if __name__ == "__main__":
    unittest.main()
