"""Exact publisher-artwork identity, bounded background work and offline cache."""

import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from ruinedfooocus.features.models import catalogue_previews as previews
from ruinedfooocus.features.models import model_sources


class CataloguePreviewTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.row = dict(
            annotation=dict(
                artifact=dict(
                    pinned_url="https://huggingface.co/owner/model/resolve/"
                    + "a" * 40
                    + "/weight.gguf",
                    sha256="b" * 64,
                )
            )
        )

    def image(self):
        output = io.BytesIO()
        Image.new("RGB", (10, 8), "red").save(output, format="PNG")
        return output.getvalue()

    def response(self, *, data=b"raster", redirect=None):
        response = Mock(
            is_redirect=redirect is not None, headers={"location": redirect} if redirect else {}
        )
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.iter_content.return_value = [data]
        return response

    def test_artwork_reads_bound_bytes_redirects_and_source_hosts(self):
        with patch.object(
            model_sources.requests, "get", return_value=self.response(data=b"oversized")
        ):
            with self.assertRaisesRegex(ValueError, "size limit"):
                previews._read("https://huggingface.co/image.png", 3)
        with patch.object(
            model_sources.requests,
            "get",
            return_value=self.response(redirect="https://other.test/private"),
        ) as get:
            with self.assertRaisesRegex(ValueError, "outside"):
                previews._read("https://huggingface.co/image.png", 32)
        self.assertEqual(get.call_count, 1)
        with patch.object(
            model_sources.requests, "get", return_value=self.response(redirect="/again")
        ) as get:
            with self.assertRaisesRegex(ValueError, "redirect limit"):
                previews._read("https://huggingface.co/image.png", 32)
        self.assertEqual(get.call_count, 4)
        with patch.object(
            model_sources.requests,
            "get",
            side_effect=[self.response(redirect="https://cdn-lfs.hf.co/image"), self.response()],
        ) as get:
            self.assertEqual(previews._read("https://huggingface.co/image.png", 32), b"raster")
        self.assertTrue(all(call.kwargs["allow_redirects"] is False for call in get.call_args_list))

    def test_local_preview_is_immediate_and_same_stem_quantizations_need_exact_sidecars(self):
        model = self.root / "model.gguf"
        model.write_bytes(b"local fixture")
        legacy = self.root / "model.preview.png"
        legacy.write_bytes(self.image())
        models = SimpleNamespace(get_file=lambda *_: model, get_names=lambda _: ["model.gguf"])
        self.assertEqual(previews.local_checkpoint_preview("model.gguf", models), str(legacy))
        models.get_names = lambda _: ["model.gguf", "model.safetensors"]
        self.assertIsNone(previews.local_checkpoint_preview("model.gguf", models))
        exact = self.root / "model.gguf.preview.png"
        exact.write_bytes(self.image())
        self.assertEqual(previews.local_checkpoint_preview("model.gguf", models), str(exact))

    def test_only_pinned_exact_registry_identity_can_request_artwork(self):
        self.assertIsNotNone(previews.preview_identity(self.row))
        for source in (
            "https://huggingface.co/owner/model/resolve/main/weight.gguf",
            "https://other.test/owner/model/resolve/" + "a" * 40 + "/weight.gguf",
            "http://huggingface.co/owner/model/resolve/" + "a" * 40 + "/weight.gguf",
        ):
            row = dict(annotation=dict(artifact=dict(pinned_url=source, sha256="b" * 64)))
            self.assertIsNone(previews.preview_identity(row))
        self.assertIsNone(previews.preview_identity(dict(annotation=None)))

    def test_manifest_match_produces_local_raster_and_source_provenance_without_weights(self):
        identity = previews.preview_identity(self.row)
        metadata = dict(
            sha="a" * 40,
            siblings=[
                dict(rfilename="weight.gguf", lfs=dict(sha256="b" * 64)),
                dict(rfilename="00-preview.gif"),
                dict(rfilename="examples/art.png"),
                dict(rfilename="video.mp4"),
            ],
        )
        with (
            patch.object(model_sources, "huggingface_manifest", return_value=metadata) as manifest,
            patch.object(previews, "_read", return_value=self.image()) as read,
        ):
            result = previews.fetch_registered_preview(identity, self.root / identity["key"])
        self.assertTrue(result.is_file())
        urls = [call.args[0] for call in read.call_args_list]
        manifest.assert_called_once_with("owner/model", "a" * 40, offline=False)
        self.assertTrue(urls[0].endswith("/examples/art.png"))
        self.assertFalse(any(url.endswith("weight.gguf") for url in urls))
        receipt = json.loads(result.with_suffix(".source.json").read_text())
        self.assertEqual(receipt["sha256"], "b" * 64)
        self.assertEqual(receipt["image_url"], urls[0])

    def test_wrong_revision_or_hash_and_no_raster_never_requests_an_image(self):
        identity = previews.preview_identity(self.row)
        for metadata in (
            dict(sha="c" * 40, siblings=[]),
            dict(sha="a" * 40, siblings=[dict(rfilename="weight.gguf", lfs=dict(sha256="c" * 64))]),
            dict(
                sha="a" * 40,
                siblings=[
                    dict(rfilename="weight.gguf", lfs=dict(sha256="b" * 64)),
                    dict(rfilename="video.mp4"),
                ],
            ),
        ):
            with (
                patch.object(model_sources, "huggingface_manifest", return_value=metadata),
                patch.object(previews, "_read") as read,
            ):
                self.assertIsNone(
                    previews.fetch_registered_preview(identity, self.root / "preview")
                )
            read.assert_not_called()

    def test_background_jobs_are_bounded_deduplicated_and_cache_works_offline(self):
        started, release = threading.Event(), threading.Event()
        active, maximum, calls = 0, 0, []
        lock = threading.Lock()

        def fetch(identity, destination):
            nonlocal active, maximum
            with lock:
                calls.append(identity["key"])
                active += 1
                maximum = max(maximum, active)
            started.set()
            self.assertTrue(release.wait(5))
            with lock:
                active -= 1
            return previews.write_preview(self.image(), destination)

        cache = previews.CataloguePreviews(self.root, workers=2, pending=3, fetcher=fetch)
        self.addCleanup(cache.close)
        self.addCleanup(release.set)
        rows = [
            dict(
                annotation=dict(
                    artifact=dict(self.row["annotation"]["artifact"], sha256=f"{index:064x}")
                )
            )
            for index in range(6)
        ]
        self.assertFalse(cache.request(rows, offline=True))
        self.assertEqual(calls, [])
        self.assertTrue(cache.request(rows))
        self.assertTrue(started.wait(5))
        cache.request(rows)
        self.assertEqual(len(cache.jobs), 3)
        self.assertLessEqual(maximum, 2)
        futures = list(cache.jobs.values())
        release.set()
        for future in futures:
            future.result(5)
        cache.close()
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(set(calls)), 3)
        self.assertEqual(cache.revision, 3)
        self.assertTrue(Path(cache.cached(rows[0])).is_file())
        self.assertFalse(cache.request(rows, offline=True))

    def test_failed_artwork_frees_slots_and_has_backoff_without_changing_rows(self):
        row_before = json.dumps(self.row, sort_keys=True)
        calls = []

        def fail(identity, destination):
            calls.append(identity["key"])
            raise RuntimeError("fixture artwork unavailable")

        cache = previews.CataloguePreviews(self.root, fetcher=fail)
        self.addCleanup(cache.close)
        cache.request([self.row])
        cache.close()
        self.assertFalse(cache.active)
        cache.request([self.row])
        self.assertEqual(len(calls), 1)
        self.assertIsNone(cache.cached(self.row))
        self.assertEqual(json.dumps(self.row, sort_keys=True), row_before)


if __name__ == "__main__":
    unittest.main()
