"""Canonical destinations stay usable after an explicit local folder cleanup."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.path import PathManager


class ModelFolderCleanupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        environment = patch.dict(os.environ, RF_DATA_DIR=str(self.root))
        environment.start()
        self.addCleanup(environment.stop)
        self.paths = PathManager.__new__(PathManager)
        self.paths.paths = PathManager.DEFAULT_PATHS.copy()
        self.paths.DOWNLOADABLE_FILES = {}

    def test_fresh_defaults_persist_only_canonical_names(self):
        self.assertEqual(self.paths.paths["path_diffusion_models"], ["../models/diffusion_models/"])
        self.assertEqual(self.paths.paths["path_embeddings"], ["../models/embeddings/"])
        self.paths.get_model_paths()
        for alias in ("clip", "unet", "embedding", "t2i_adapter"):
            self.assertFalse((self.root / "models" / alias).exists())

    def test_old_profiles_find_moved_files_and_download_to_canonical_roots(self):
        for category, setting, alias, canonical in (
            ("clip", "path_clip", "clip", "text_encoders"),
            ("diffusion_models", "path_diffusion_models", "unet", "diffusion_models"),
            ("embeddings", "path_embeddings", "embedding", "embeddings"),
            ("controlnet", "path_controlnet", "t2i_adapter", "controlnet"),
        ):
            with self.subTest(category=category):
                original = f"../models/{alias}/"
                self.paths.paths[setting] = original
                destination = self.root / "models" / canonical
                destination.mkdir(parents=True, exist_ok=True)
                installed = destination / "installed.safetensors"
                installed.write_bytes(b"preserved weights")
                self.paths.DOWNLOADABLE_FILES = {
                    "missing": {"path": setting, "filename": "missing.safetensors"}
                }
                self.paths.download_file = Mock()
                self.assertEqual(
                    self.paths.get_folder_file_path(category, installed.name), installed
                )
                self.assertEqual(
                    self.paths.get_file_path("missing"), destination / "missing.safetensors"
                )
                self.paths.get_model_paths()
                self.assertFalse((self.root / "models" / alias).exists())
                self.assertEqual(self.paths.paths[setting], original)
                self.assertEqual(installed.read_bytes(), b"preserved weights")

    def test_custom_destination_named_clip_remains_authoritative(self):
        custom = self.root / "shared" / "clip"
        self.paths.paths["path_clip"] = [str(custom)]
        self.assertEqual(self.paths.get_model_download_folder("text_encoders"), custom)
        self.assertEqual(self.paths.get_model_folders("clip"), [custom])
        self.paths.get_model_paths()
        self.assertTrue(custom.is_dir())
        self.assertFalse((self.root / "models/text_encoders").exists())

    def test_download_writes_canonical_destination_for_legacy_local_profile(self):
        self.paths.paths["path_embeddings"] = ["../models/embedding/"]
        self.paths.DOWNLOADABLE_FILES = {
            "embedding/test.safetensors": {
                "path": "path_embeddings",
                "filename": "test.safetensors",
                "url": "https://huggingface.co/example/fixture/resolve/main/test.safetensors",
            }
        }
        destination = self.root / "models/embeddings/test.safetensors"
        with patch("ruinedfooocus.core.preparation.preparation_service.submit") as download:
            self.paths.download_file("embedding/test.safetensors")
        download.assert_called_once()
        request, progress = download.call_args.args
        self.assertEqual(
            request.url, self.paths.DOWNLOADABLE_FILES["embedding/test.safetensors"]["url"]
        )
        self.assertEqual(request.destination, destination)
        self.assertIsNone(progress)
        download.return_value.result.assert_called_once_with()
        self.assertFalse((self.root / "models/embedding").exists())


if __name__ == "__main__":
    unittest.main()
