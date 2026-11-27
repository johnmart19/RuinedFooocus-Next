"""Tests for core/model_catalogue.py, manifests consolidation, and ComfyUI template sync."""

import json
from pathlib import Path
import tempfile
import unittest

from ruinedfooocus.core import model_catalogue
from ruinedfooocus.core.path import PathManager


class ModelCatalogueSyncTests(unittest.TestCase):
    def test_bundled_catalogue_loads_deterministically_and_passes_validation(self):
        entries, provenance = model_catalogue.load_catalogue()
        self.assertGreater(len(entries), 390)
        self.assertEqual(set(entries), set(provenance))
        self.assertEqual(model_catalogue.validate_catalogue(entries), [])
        self.assertIn(
            "controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors",
            entries,
        )
        self.assertEqual(
            provenance["controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors"],
            "comfyui_templates_synced.json",
        )

    def test_path_manager_exposes_catalogue_sources_and_synced_entries(self):
        manager = PathManager()
        self.assertIn("loras/Qwen-Image-Lightning-4steps-V1.0.safetensors", manager.DOWNLOADABLE_FILES)
        self.assertEqual(
            manager.CATALOGUE_SOURCES["loras/Qwen-Image-Lightning-4steps-V1.0.safetensors"],
            "comfyui_templates_synced.json",
        )

    def test_rag_embedding_manifest_loads_from_model_catalogue_manifests(self):
        manifest_path = model_catalogue.manifests_directory() / "rag_embedding_models.json"
        self.assertTrue(manifest_path.is_file())
        data = model_catalogue.load_manifest("rag_embedding_models.json")
        self.assertEqual(data.get("schema_version"), 1)
        self.assertTrue(data.get("models"))

    def test_discover_comfyui_template_models_filters_non_image_and_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "image_demo.json").write_text(
                json.dumps(
                    {
                        "nodes": [
                            {
                                "properties": {
                                    "models": [
                                        {
                                            "name": "new_upscaler.safetensors",
                                            "url": "https://huggingface.co/org/repo/resolve/main/new_upscaler.safetensors",
                                            "directory": "upscale_models",
                                        },
                                        {
                                            "name": "already_there.safetensors",
                                            "url": "https://huggingface.co/org/repo/resolve/main/already_there.safetensors",
                                            "directory": "loras",
                                        },
                                    ]
                                }
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "video_wan_demo.json").write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "name": "wan2_video.safetensors",
                                "url": "https://huggingface.co/org/repo/resolve/main/wan2_video.safetensors",
                                "directory": "diffusion_models",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            existing = {
                "loras/already_there.safetensors": {
                    "url": "https://huggingface.co/org/repo/resolve/main/already_there.safetensors",
                    "path": "path_loras",
                    "filename": "already_there.safetensors",
                }
            }
            found = model_catalogue.discover_comfyui_template_models(
                existing, templates_root=root
            )
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["catalogue_key"], "upscalers/new_upscaler.safetensors")
            self.assertEqual(found[0]["path"], "path_upscalers")

    def test_validate_catalogue_rejects_unsafe_filenames_and_bad_digests(self):
        issues = model_catalogue.validate_catalogue(
            {
                "bad/one": {
                    "url": "ftp://invalid.test/a.safetensors",
                    "path": "checkpoints",
                    "filename": "../escape.safetensors",
                    "size": -5,
                    "sha256": "not-a-sha",
                }
            }
        )
        self.assertEqual(len(issues), 5)


if __name__ == "__main__":
    unittest.main()
