"""Krea style submission uses the real registered adapter preparation request."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.core.path import PathManager
from ruinedfooocus.core.paths import resource_path
from test_krea_style_resources import KEY, SHA256, SIZE, URL


class KreaStyleAssetCaptureTests(unittest.TestCase):
    def test_first_use_captures_real_registered_adapter_and_offline_readiness_blocks(self):
        from ruinedfooocus.core.preparation import preparation_service
        from ruinedfooocus.features.generation.image_preparation import (
            ImageAssets,
            KEY as ASSETS_KEY,
            image_readiness,
        )
        from ruinedfooocus.features.image_editing.krea_style import capture_assets

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manager = PathManager.__new__(PathManager)
            manager.paths = {"path_loras": str(root / "loras")}
            manager.model_paths = {"cache_path": root / "cache"}
            manager.DOWNLOADABLE_FILES = copy.deepcopy(
                json.loads(resource_path("model_catalogue", "native_krea_style.json").read_text())
            )
            data = dict(
                base_model_name="krea", input_image=object(), native_edit_recipe="krea2-style"
            )
            with (
                patch.object(
                    preparation_service, "submit", side_effect=AssertionError("No acquisition")
                ),
                patch.object(manager, "download_file", side_effect=AssertionError("No download")),
                patch.object(manager, "get_file_path", side_effect=AssertionError("No resolver")),
                patch("urllib.request.urlopen", side_effect=AssertionError("No network")),
                patch(
                    "ruinedfooocus.features.generation.image_preparation._control_checkpoint_family",
                    return_value="Krea2",
                ),
            ):
                captured, artifacts = capture_assets(data, manager, object(), {"type": "edit"})
                self.assertEqual(captured["krea_style_lora"], Path(KEY).name)
                self.assertEqual(artifacts[0].role, "krea-style-lora")
                request = artifacts[0].request
                self.assertEqual(request.url, URL)
                self.assertEqual(request.expected_size, SIZE)
                self.assertEqual(request.sha256, SHA256)
                self.assertEqual(request.destination, (root / "loras" / Path(KEY).name).resolve())
                readiness = image_readiness(
                    {ASSETS_KEY: ImageAssets(artifacts=artifacts, offline=True)}
                )
            self.assertEqual(readiness.state, "blocked")
            self.assertEqual(readiness.missing_bytes, SIZE)
            self.assertIn("Offline mode", readiness.reasons[0])
            self.assertIn("not runtime memory", readiness.fit_status)
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
