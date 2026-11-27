"""Focused CPU tests for Civitai and Hugging Face artwork downloading, caching, and threading."""

import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from ruinedfooocus.features.models import catalogue_previews
from ruinedfooocus.features.models import model_sources
from ruinedfooocus.features.models import popular_models as feed


def _png_bytes(color="steelblue"):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, "PNG")
    return buf.getvalue()


class HuggingFaceArtworkExtractionTests(unittest.TestCase):
    def test_card_data_thumbnail_used_when_siblings_have_no_images(self):
        manifest = {
            "sha": "a" * 40,
            "siblings": [{"rfilename": "model.gguf"}],
            "cardData": {
                "thumbnail": "https://cdn-uploads.huggingface.co/production/uploads/abc/preview.png"
            },
        }
        urls = model_sources.extract_hf_artwork_urls(
            manifest, "org/model", "a" * 40, allow_readme=True
        )
        self.assertEqual(
            urls,
            ["https://cdn-uploads.huggingface.co/production/uploads/abc/preview.png"],
        )

    def test_readme_images_extracted_and_badges_or_svgs_skipped(self):
        readme = (
            b"![badge](https://img.shields.io/badge/test-blue.svg)\n"
            b"![logo](https://huggingface.co/org/model/resolve/main/logo.svg)\n"
            b"![showcase](./assets/showcase.png)\n"
            b'<img src="https://huggingface.co/org/model/blob/main/gallery/demo.jpg" />\n'
            b"![cdn](https://cdn-uploads.huggingface.co/production/uploads/1/sample.webp)\n"
            b"![social](https://cdn-thumbnails.huggingface.co/social-thumbnails/models/org/model.png)\n"
        )
        manifest = {
            "sha": "b" * 40,
            "siblings": [{"rfilename": "model.safetensors"}, {"rfilename": "README.md"}],
        }
        reads = []

        def fake_read(url, limit, *, params=None, offline=False):
            reads.append(url)
            return readme

        with patch.object(model_sources.hf_metadata, "read", side_effect=fake_read):
            urls = model_sources.extract_hf_artwork_urls(
                manifest,
                "org/model",
                "b" * 40,
                rank=catalogue_previews._preview_rank,
            )
        self.assertEqual(
            reads,
            [f"https://huggingface.co/org/model/resolve/{'b' * 40}/README.md"],
        )
        self.assertEqual(
            set(urls),
            {
                f"https://huggingface.co/org/model/resolve/{'b' * 40}/assets/showcase.png",
                "https://huggingface.co/org/model/resolve/main/gallery/demo.jpg",
                "https://cdn-uploads.huggingface.co/production/uploads/1/sample.webp",
            },
        )

    def test_readme_not_fetched_when_absent_from_siblings(self):
        manifest = {
            "sha": "c" * 40,
            "siblings": [{"rfilename": "weights.safetensors"}],
        }
        with patch.object(model_sources.hf_metadata, "read") as mock_read:
            urls = model_sources.extract_hf_artwork_urls(manifest, "org/model", "c" * 40)
        self.assertEqual(urls, [])
        mock_read.assert_not_called()

    def test_fetch_registered_preview_uses_card_data_thumbnail(self):
        digest = "12" * 32
        revision = "d" * 40
        identity = {
            "source": f"https://huggingface.co/org/repo/resolve/{revision}/model.safetensors",
            "sha256": digest,
            "repository": "org/repo",
            "revision": revision,
            "key": "testkey",
        }
        manifest = {
            "sha": revision,
            "siblings": [{"rfilename": "model.safetensors", "lfs": {"sha256": digest}}],
            "cardData": {
                "thumbnail": "https://cdn-uploads.huggingface.co/production/uploads/1/cover.png"
            },
        }
        png = _png_bytes("crimson")
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "testkey"
            with (
                patch.object(model_sources, "huggingface_manifest", return_value=manifest),
                patch.object(catalogue_previews, "_read", return_value=png) as mock_read,
            ):
                published = catalogue_previews.fetch_registered_preview(identity, dest)
            self.assertIsNotNone(published)
            self.assertTrue(Path(published).is_file())
            mock_read.assert_called_once_with(
                "https://cdn-uploads.huggingface.co/production/uploads/1/cover.png",
                12 * 1024 * 1024,
            )


class PopularArtworkStreamingTests(unittest.TestCase):
    def test_parse_huggingface_extracts_card_data_thumbnail(self):
        items = [
            {
                "id": "org/Model-GGUF",
                "tags": ["gguf"],
                "cardData": {
                    "thumbnail": "https://cdn-uploads.huggingface.co/production/uploads/1/thumb.png"
                },
            },
            {
                "id": "org/SocialOnly-GGUF",
                "tags": ["gguf"],
                "cardData": {
                    "thumbnail": "https://cdn-thumbnails.huggingface.co/social-thumbnails/models/org/SocialOnly.png"
                },
            },
        ]
        entries = feed.parse_huggingface(items)
        self.assertEqual(
            entries[0]["image"],
            "https://cdn-uploads.huggingface.co/production/uploads/1/thumb.png",
        )
        self.assertIsNone(entries[1]["image"])

    def test_fetch_artwork_iter_streams_batches_and_resolves_hf_repo(self):
        png = _png_bytes("teal")
        entries = [
            {"source": "Civitai", "image": f"https://image.civitai.com/a/b/width=320/{i}.jpeg"}
            for i in range(4)
        ] + [
            {"source": "Hugging Face", "repository": "org/Chat-GGUF", "image": None},
        ]
        hf_manifest = {
            "sha": "e" * 40,
            "siblings": [{"rfilename": "cover.png"}],
        }
        with tempfile.TemporaryDirectory() as folder:
            with (
                patch.object(feed, "offline", return_value=False),
                patch.object(model_sources, "huggingface_manifest", return_value=hf_manifest),
            ):
                batches = list(
                    feed.fetch_artwork_iter(
                        entries, folder, fetch_image=lambda url: png, batch_size=2
                    )
                )
            self.assertEqual(batches, [2, 4, 5])
            hf_target = feed.thumbnail_path(entries[-1], folder)
            self.assertIsNotNone(hf_target)
            self.assertTrue(hf_target.is_file())


class CivitUpdateLiveRefreshTests(unittest.TestCase):
    def test_civit_update_bumps_revision_and_evicts_shared_cache_per_thumbnail(self):
        from ruinedfooocus.app.state import shared_cache
        from ruinedfooocus.features.models.model_handler import Models
        from ruinedfooocus.features.models.model_previews import write_preview

        png = _png_bytes("navy")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            models_dir = root / "checkpoints"
            cache_dir = root / "cache" / "checkpoints"
            models_dir.mkdir(parents=True)
            cache_dir.mkdir(parents=True)
            (models_dir / "a.safetensors").write_bytes(b"a")
            (models_dir / "b.safetensors").write_bytes(b"b")

            handler = Models.__new__(Models)
            handler.EXTENSIONS = [".safetensors"]
            handler._metadata_lock = threading.RLock()
            handler.civit_workers = []
            handler._metadata_checked = {}
            handler.ready = {"checkpoints": False}
            handler.revision = 0
            handler.offline = False
            handler.cache_paths = {"checkpoints": cache_dir}
            handler.names = {"checkpoints": ["a.safetensors", "b.safetensors"]}
            handler.model_hash = {"checkpoints": {}}

            stale_a = cache_dir / "a.jpeg"
            stale_b = cache_dir / "b.jpeg"
            shared_cache[stale_a] = "stale"
            shared_cache[stale_b] = "stale"

            seen_revisions = []

            def fake_get_models_by_path(model_type, path, fetch=False, discover_hf=False):
                return {"images": [{"url": f"https://image.civitai.com/{Path(path).stem}.jpeg"}]}

            def fake_get_image(model_data, cache_file):
                write_preview(png, cache_file)
                return True

            handler.get_models_by_path = fake_get_models_by_path
            handler.copy_embedded_preview = lambda path, cache_file: False
            handler.get_image = fake_get_image

            orig_notify = handler._notify_preview_updated

            def tracked_notify(model_type, cache_file=None):
                orig_notify(model_type, cache_file)
                seen_revisions.append(handler.revision)

            handler._notify_preview_updated = tracked_notify
            handler._update_models("checkpoints", [models_dir])

            self.assertEqual(len(seen_revisions), 2)
            self.assertEqual(handler.revision, 2)
            self.assertNotIn(stale_a, shared_cache)
            self.assertNotIn(stale_b, shared_cache)


if __name__ == "__main__":
    unittest.main()
