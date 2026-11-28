"""Resolution lists survive pinned metadata acquisition, caching and UI selection."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.features.models import model_sources
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.model_recommendations import (
    ModelRecommendations,
    published_resolutions,
)
from ruinedfooocus.features.models.resolutions import ResolutionSettings
from ruinedfooocus.features.models.resource_defaults import load_family_recipe


class PublisherResolutionTests(unittest.TestCase):
    def test_explicit_lists_and_named_tables_preserve_order_and_ignore_examples(self):
        metadata = {
            "generationConfig": {
                "width": 1024,
                "height": 1024,
                "recommended_resolutions": ["832x1216", [1216, 832], [True, 1024], [1025, 1024]],
            },
            "description": """Example: 2048x2048
## Recommended resolutions

| Aspect ratio | Width | Height |
| --- | --- | --- |
| 9:16 | 768 | 1344 |
| 16:9 | 1344 | 768 |
## Examples
- 3072x3072
```
Recommended sizes: 4096x4096
```
""",
            "images": [{"width": 4096, "height": 4096}],
        }
        self.assertEqual(
            published_resolutions(metadata),
            [
                (1024, 1024),
                (832, 1216),
                (1216, 832),
                (768, 1344),
                (1344, 768),
            ],
        )
        self.assertEqual(
            published_resolutions(
                {"description": "<p><b>Recommended sizes:</b> 832×1216, 1216×832</p>"}
            ),
            [(832, 1216), (1216, 832)],
        )
        self.assertEqual(
            published_resolutions(
                {
                    "description": "<h3>Supported output sizes</h3><table><tr><td>832</td><td>1216</td></tr><tr><td>1216</td><td>832</td></tr></table>"
                }
            ),
            [(832, 1216), (1216, 832)],
        )
        self.assertEqual(
            published_resolutions(
                {
                    "generationConfig": {
                        "recommended_sizes": [
                            {"width": 832, "height": 1216},
                            [832, 1216],
                            [1024.0, 1024],
                            [8192, 8192],
                        ]
                    }
                }
            ),
            [(832, 1216)],
        )

    def test_hf_download_refresh_and_cache_need_no_sampling_recipe_or_extra_repository(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "model.safetensors"
            model.write_bytes(b"inert fixture")
            metadata = {
                "baseModel": "SDXL",
                "hf_repo_id": "owner/image",
                "_rf_local_sha256": "b" * 64,
                "_rf_file_signature": file_signature(model),
            }
            (root / "model.json").write_text(json.dumps(metadata), encoding="utf-8")
            store = ModelRecommendations(root, root / "recommendations.json")
            revision = "a" * 40
            manifest = {
                "id": "owner/image",
                "sha": revision,
                "siblings": [
                    {"rfilename": "model.safetensors", "lfs": {"sha256": "b" * 64}},
                    {"rfilename": "README.md"},
                    {"rfilename": "generation_config.json"},
                ],
            }
            payloads = {
                f"https://huggingface.co/owner/image/resolve/{revision}/README.md": b"Recommended resolutions: 832x1216, 1216x832\nhttps://huggingface.co/other/repack",
                f"https://huggingface.co/owner/image/resolve/{revision}/generation_config.json": b'{"width":1024,"height":1024}',
            }
            with (
                patch.object(
                    model_sources, "huggingface_manifest", return_value=manifest
                ) as manifest_read,
                patch.object(
                    model_sources.hf_metadata,
                    "read",
                    side_effect=lambda url, limit, **kw: payloads[url],
                ) as read,
            ):
                self.assertEqual(
                    store.refresh(str(model), [], []), "Public version metadata refreshed."
                )
                self.assertEqual(
                    store.cached_resolutions(str(model))[0],
                    [(1024, 1024), (832, 1216), (1216, 832)],
                )
                self.assertEqual(store.cached_resolution(str(model))[0], (1024, 1024))
                store.refresh(str(model), [], [])
                manifest_read.assert_called_once_with("owner/image", "", offline=False)
                self.assertEqual(read.call_count, 2)
            with patch.object(
                model_sources,
                "huggingface_recommendations",
                side_effect=AssertionError("UI/cache lookup must be local"),
            ):
                self.assertEqual(len(store.cached_resolutions(str(model))[0]), 3)
                model.write_bytes(b"replacement fixture with different signature")
                self.assertEqual(store.cached_resolutions(str(model)), ([], None))

    def test_legacy_cache_and_explicit_empty_refresh_keep_identity_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "model.safetensors"
            model.write_bytes(b"inert")
            metadata = {"_rf_file_signature": file_signature(model)}
            (root / "model.json").write_text(json.dumps(metadata))
            store = ModelRecommendations(root, root / "recommendations.json")
            key = store.identity(str(model))
            entry = {
                "resolution": [832, 1216],
                "resolution_signature": file_signature(model),
                "source": "Publisher",
            }
            store.path.write_text(json.dumps({key: entry}))
            self.assertEqual(store.cached_resolutions(str(model)), ([(832, 1216)], "Publisher"))
            entry["resolutions"] = []
            store.path.write_text(json.dumps({key: entry}))
            self.assertEqual(store.cached_resolutions(str(model)), ([], "Publisher"))

    def test_common_model_and_unknown_menus_offer_both_orientations(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(ResolutionSettings, "RESOLUTIONS_FILE", Path(folder) / "resolutions.json"),
        ):
            settings = ResolutionSettings()
            for family in (
                "SDXL",
                "Illustrious",
                "NoobAI",
                "Pony",
            ):
                sizes = settings.recommended_sizes(family)
                with self.subTest(family=family):
                    self.assertGreaterEqual(len(sizes), 9)
                    self.assertTrue(any(w < h for w, h in sizes))
                    self.assertTrue(any(w > h for w, h in sizes))
            choices = settings.choices_for_model("Flux3")
            self.assertGreaterEqual(len(choices), 4)
            self.assertEqual(choices[-1], "Custom...")
            self.assertTrue(all("Generic fallback" in choice for choice in choices[:-1]))
            self.assertEqual(
                load_family_recipe("native-qwenimage-layered")["recommended_sizes"], [[640, 640]]
            )
            self.assertEqual(
                load_family_recipe("native-pid-flux1-1024-to-4096-4step")["recommended_sizes"],
                [[4096, 4096]],
            )
            self.assertEqual(
                len(load_family_recipe("native-flux-1-d-kontext-dev")["recommended_sizes"]), 17
            )

    def test_multilingual_card_recommendations_are_kept_as_published(self):
        self.assertEqual(
            published_resolutions({"description": "<p>推荐分辨率：812×1216、1152×1536</p>"}),
            [(812, 1216), (1152, 1536)],
        )
        self.assertEqual(
            published_resolutions({"description": "推奨解像度: 896x1152、1152x896"}),
            [(896, 1152), (1152, 896)],
        )
        self.assertEqual(published_resolutions({"description": "Example output 812×1216"}), [])

    def test_modelspec_resolution_and_optimal_native_card_headings(self):
        self.assertEqual(
            published_resolutions(
                {
                    "modelspec.resolution": "1024x1024",
                    "recommended_resolutions": "832x1216, 1216x832",
                    "description": "## Optimal resolutions\n- 896x1152\n- 1152x896\n",
                }
            ),
            [(1024, 1024), (832, 1216), (1216, 832), (896, 1152), (1152, 896)],
        )
        self.assertEqual(
            published_resolutions({"description": "Native sizes: 1440×1440, 1280×1600"}),
            [(1440, 1440), (1280, 1600)],
        )


if __name__ == "__main__":
    unittest.main()
