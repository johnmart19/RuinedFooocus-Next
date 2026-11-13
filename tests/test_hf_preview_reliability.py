"""Shared artwork policy uses fake HTTP/clock and tiny local rasters only."""

import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import format_datetime
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import requests

from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.features.models import catalogue_previews, model_sources
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.model_previews import valid_preview, write_preview

ROOT = Path(__file__).resolve().parents[1]
REVISION, DIGEST = "a" * 40, "b" * 64
SOURCE = f"https://huggingface.co/owner/image/resolve/{REVISION}/weight.gguf"


def model_method(name, **extra):
    source = ROOT / "ruinedfooocus/features/models/model_handler.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Models"
    )
    node = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    scope = dict(
        Path=Path,
        json=json,
        sys=sys,
        file_signature=file_signature,
        save_json=save_json,
        t=lambda text, **kw: text,
        **extra,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope[name]


class FakeResponse:
    def __init__(self, payload=None, *, body=None, status=200, retry_after=None):
        self.status_code = status
        self.headers = {} if retry_after is None else {"Retry-After": retry_after}
        self.is_redirect = False
        self.body = json.dumps(payload).encode() if body is None else body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)

    def iter_content(self, size):
        yield self.body


class PreviewReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.now = 10.0
        self.wall = 1700000000.0
        self.metadata = model_sources.HuggingFaceMetadata(
            clock=lambda: self.now, wall_clock=lambda: self.wall
        )
        self.enterContext(patch.object(model_sources, "hf_metadata", self.metadata))
        self.row = dict(annotation=dict(artifact=dict(pinned_url=SOURCE, sha256=DIGEST)))

    def manifest(self, *, images=True):
        files = [dict(rfilename="weight.gguf", lfs=dict(sha256=DIGEST))]
        if images:
            files.append(dict(rfilename="00-preview.gif"))
            files.append(dict(rfilename="art.png"))
        return dict(id="owner/image", sha=REVISION, siblings=files)

    def image(self):
        output = io.BytesIO()
        Image.new("RGB", (12, 8), "red").save(output, format="PNG")
        return output.getvalue()

    def test_concurrent_installed_and_catalogue_callers_share_one_manifest(self):
        entered, release = threading.Event(), threading.Event()
        calls = []

        def get(url, **kwargs):
            calls.append(url)
            if "/api/models/" in url:
                entered.set()
                self.assertTrue(release.wait(5))
                return FakeResponse(self.manifest())
            return FakeResponse(body=self.image())

        with patch.object(model_sources.requests, "get", side_effect=get):
            with ThreadPoolExecutor(max_workers=4) as pool:
                installed = pool.submit(
                    model_sources.huggingface_preview, "weight.gguf", DIGEST, SOURCE
                )
                self.assertTrue(entered.wait(5))
                identity = catalogue_previews.preview_identity(self.row)
                catalogue = pool.submit(
                    catalogue_previews.fetch_registered_preview, identity, self.root / "registered"
                )
                recommendation = pool.submit(
                    model_sources.huggingface_recommendations,
                    "owner/image",
                    DIGEST,
                    revision=REVISION,
                )
                release.set()
                self.assertEqual(installed.result(5)["hf_revision"], REVISION)
                self.assertTrue(valid_preview(catalogue.result(5)))
                self.assertIsNotNone(recommendation.result(5))
        self.assertEqual(sum("/api/models/" in url for url in calls), 1)
        self.assertEqual(sum(url.endswith("art.png") for url in calls), 1)
        self.assertFalse(any(url.endswith("weight.gguf") for url in calls))

    def test_mutable_manifest_populates_exact_revision_cache_and_revision_invalidates(self):
        with patch.object(
            model_sources.requests, "get", return_value=FakeResponse(self.manifest())
        ) as get:
            self.assertIsNotNone(
                model_sources.huggingface_preview(
                    "weight.gguf", DIGEST, "https://huggingface.co/owner/image"
                )
            )
            self.assertIsNotNone(model_sources.huggingface_manifest("owner/image", REVISION))
            self.assertEqual(get.call_count, 1)
            self.assertIsNone(model_sources.huggingface_manifest("owner/image", "c" * 40))
            self.assertEqual(get.call_count, 2)
            self.now += 3601
            model_sources.huggingface_manifest("owner/image")
            self.assertEqual(get.call_count, 3)

    def test_failures_are_shared_negative_cached_and_expire_without_inline_retries(self):
        for status, ttl in ((401, 600), (403, 600), (404, 600), (500, 60)):
            with self.subTest(status=status):
                repository = f"owner/image{status}"
                with patch.object(
                    model_sources.requests, "get", return_value=FakeResponse(status=status)
                ) as get:
                    self.assertIsNone(model_sources.huggingface_manifest(repository))
                    self.assertIsNone(model_sources.huggingface_manifest(repository))
                    self.assertEqual(get.call_count, 1)
                    self.now += ttl + 1
                    self.assertIsNone(model_sources.huggingface_manifest(repository))
                    self.assertEqual(get.call_count, 2)
        self.assertEqual(self.metadata.inflight, {})
        with patch.object(
            model_sources.requests, "get", return_value=FakeResponse(body=b"bad json")
        ) as get:
            self.assertIsNone(model_sources.huggingface_manifest("owner/malformed"))
            self.assertIsNone(model_sources.huggingface_manifest("owner/malformed"))
            self.assertEqual(get.call_count, 1)

    def test_retry_after_seconds_date_and_invalid_headers_stop_all_hf_callers(self):
        date = format_datetime(datetime.fromtimestamp(self.wall + 120, timezone.utc), usegmt=True)
        for header, delay in (("90", 90), (date, 120), (None, 60), ("invalid", 60)):
            with self.subTest(header=header):
                self.metadata.not_before = 0
                self.metadata.cache.clear()
                with patch.object(
                    model_sources.requests,
                    "get",
                    side_effect=[
                        FakeResponse(status=429, retry_after=header),
                        FakeResponse(self.manifest()),
                    ],
                ) as get:
                    self.assertIsNone(
                        model_sources.huggingface_preview("weight.gguf", DIGEST, SOURCE)
                    )
                    self.assertIsNone(model_sources.huggingface_manifest("owner/another"))
                    self.assertIsNone(
                        model_sources.huggingface_recommendations("owner/image", DIGEST)
                    )
                    with self.assertRaisesRegex(ValueError, "cooling down"):
                        self.metadata.read("https://cdn-lfs.hf.co/art.png", 100)
                    self.assertEqual(get.call_count, 1)
                    self.now += delay - 0.1
                    self.assertIsNone(model_sources.huggingface_manifest("owner/image"))
                    self.now += 0.2
                    self.assertIsNotNone(model_sources.huggingface_manifest("owner/image"))
                    self.assertEqual(get.call_count, 2)

    def test_image_search_retains_model_tokens_and_never_opens_llm_manifests(self):
        search = [
            dict(id="JoyAI/LLM", pipeline_tag="text-generation", tags=["text-to-image"]),
            dict(id="JoyAI/unknown"),
            dict(id="JoyAI/malformed", pipeline_tag=["image-to-image"]),
            dict(id="JoyAI/malformed-tags", tags=5),
            dict(id="owner/image", pipeline_tag="image-to-image"),
        ]
        with patch.object(
            model_sources.requests,
            "get",
            side_effect=[FakeResponse(search), FakeResponse(self.manifest())],
        ) as get:
            result = model_sources.huggingface_preview("joyai_image_edit_Q4_K.gguf", DIGEST)
            self.assertEqual(result["hf_repo_id"], "owner/image")
            self.assertEqual(get.call_count, 2)
            self.assertEqual(get.call_args_list[0].kwargs["params"]["search"], "joyai image edit")
            self.assertTrue(get.call_args_list[1].args[0].endswith("/owner/image"))
            model_sources.huggingface_preview("joyai_image_edit_Q4_K.gguf", DIGEST)
            self.assertEqual(get.call_count, 2)

    def test_known_wrong_hash_or_revision_never_searches_or_fetches_another_source(self):
        with patch.object(
            model_sources.requests, "get", return_value=FakeResponse(self.manifest())
        ) as get:
            self.assertIsNone(model_sources.huggingface_preview("weight.gguf", "c" * 64, SOURCE))
            wrong_revision = SOURCE.replace(REVISION, "d" * 40)
            self.assertIsNone(
                model_sources.huggingface_preview("weight.gguf", DIGEST, wrong_revision)
            )
            self.assertEqual(get.call_count, 2)
            self.assertTrue(
                all(
                    "/api/models/owner/image/revision/" in call.args[0]
                    for call in get.call_args_list
                )
            )

    def test_offline_helpers_and_existing_worker_make_zero_http_requests(self):
        identity = catalogue_previews.preview_identity(self.row)
        cache = catalogue_previews.CataloguePreviews(self.root)
        self.addCleanup(cache.close)
        with patch.object(
            model_sources.requests, "get", side_effect=AssertionError("offline HTTP")
        ) as get:
            self.assertIsNone(
                model_sources.huggingface_preview("weight.gguf", DIGEST, SOURCE, offline=True)
            )
            self.assertIsNone(
                model_sources.huggingface_recommendations("owner/image", DIGEST, offline=True)
            )
            self.assertIsNone(
                catalogue_previews.fetch_registered_preview(
                    identity, self.root / "offline", offline=True
                )
            )
            self.assertFalse(cache.request([self.row], offline=True))
            get.assert_not_called()

    def test_switching_offline_during_metadata_prevents_raster_and_pending_jobs(self):
        started, release = threading.Event(), threading.Event()

        def get(url, **kwargs):
            self.assertIn("/api/models/", url)
            started.set()
            self.assertTrue(release.wait(5))
            return FakeResponse(self.manifest())

        cache = catalogue_previews.CataloguePreviews(self.root, workers=1, pending=3)
        self.addCleanup(cache.close)
        self.addCleanup(release.set)
        second = dict(annotation=dict(artifact=dict(pinned_url=SOURCE, sha256="c" * 64)))
        with patch.object(model_sources.requests, "get", side_effect=get) as http:
            cache.request([self.row, second])
            self.assertTrue(started.wait(5))
            cache.request([self.row, second], offline=True)
            release.set()
            cache.close()
            self.assertEqual(http.call_count, 1)
            self.assertIsNone(cache.cached(self.row))
            self.assertFalse(cache.active)

    def test_local_exact_sidecar_and_valid_cache_prevent_background_artwork_or_hashing(self):
        model = self.root / "image.gguf"
        model.write_bytes(b"inert file")
        cache = self.root / "cache"
        cache.mkdir()
        exact = model.with_name(model.name + ".preview.png")
        exact.write_bytes(self.image())
        owner = SimpleNamespace(
            EXTENSIONS=(".gguf",),
            offline=False,
            cache_paths={"checkpoints": cache},
            model_hash={"checkpoints": {}},
            get_names=lambda _: [model.name],
            get_models_by_path=Mock(return_value={}),
            copy_embedded_preview=Mock(return_value=False),
            model_sha256=Mock(side_effect=AssertionError("no model hashing")),
            get_image=Mock(),
        )
        with patch.object(
            model_sources.requests, "get", side_effect=AssertionError("local artwork HTTP")
        ):
            model_method("_update_models")(owner, "checkpoints", [self.root])
            owner.get_models_by_path.assert_called_once_with(
                "checkpoints", model, fetch=False, discover_hf=False
            )
            self.assertTrue(valid_preview(cache / "image.jpeg"))
            exact.unlink()
            owner.get_models_by_path.reset_mock()
            model_method("_update_models")(owner, "checkpoints", [self.root])
            owner.get_models_by_path.assert_called_once_with(
                "checkpoints", model, fetch=False, discover_hf=False
            )
            owner.get_image.assert_not_called()
            owner.model_sha256.assert_not_called()

    def test_actual_metadata_worker_has_one_discovery_attempt_and_cached_miss(self):
        model = self.root / "image.safetensors"
        model.write_bytes(b"inert file")
        cache = self.root / "cache"
        cache.mkdir()
        owner = SimpleNamespace(
            EXTENSIONS=(".safetensors",),
            offline=False,
            cache_paths={"checkpoints": cache},
            model_hash={"checkpoints": {}},
            get_names=lambda _: [model.name],
            copy_embedded_preview=Mock(return_value=False),
            get_image=Mock(return_value=False),
            model_sha256=Mock(return_value=DIGEST),
            detect_checkpoint_base=Mock(return_value="SDXL 1.0"),
            read_safetensors_header=lambda _: {"__metadata__": {"modelspec.source": SOURCE}},
            search_civitai_with_hash=Mock(side_effect=AssertionError("known HF source")),
        )
        owner.get_models_by_path = lambda *args, **kw: model_method("get_models_by_path")(
            owner, *args, **kw
        )
        with (
            patch.object(
                model_sources.requests,
                "get",
                return_value=FakeResponse(self.manifest(images=False)),
            ) as get,
            patch.object(
                model_sources, "huggingface_preview", wraps=model_sources.huggingface_preview
            ) as lookup,
        ):
            model_method("_update_models")(owner, "checkpoints", [self.root])
            self.assertEqual(lookup.call_count, 1)
            model_method("_update_models")(owner, "checkpoints", [self.root])
            self.assertEqual(lookup.call_count, 2)
            self.assertEqual(get.call_count, 1)
            self.assertEqual(owner.model_sha256.call_count, 1)

    def test_civitai_identity_with_failed_artwork_retains_one_hash_matched_hf_fallback(self):
        model = self.root / "image.gguf"
        model.write_bytes(b"inert file")
        cache = self.root / "cache"
        cache.mkdir()
        owner = SimpleNamespace(
            EXTENSIONS=(".gguf",),
            offline=False,
            cache_paths={"checkpoints": cache},
            model_hash={"checkpoints": {}},
            get_names=lambda _: [model.name],
            get_models_by_path=Mock(
                return_value={
                    "id": 12,
                    "_rf_local_sha256": DIGEST,
                    "images": [{"url": "https://example.invalid/failed.png"}],
                }
            ),
            copy_embedded_preview=Mock(return_value=False),
            get_image=Mock(side_effect=[False, True]),
            model_sha256=Mock(side_effect=AssertionError("no new weight hashing")),
            read_safetensors_header=lambda _: {"__metadata__": {"modelspec.source": SOURCE}},
        )
        with (
            patch.object(
                model_sources.requests, "get", return_value=FakeResponse(self.manifest())
            ) as get,
            patch.object(
                model_sources, "huggingface_preview", wraps=model_sources.huggingface_preview
            ) as lookup,
        ):
            model_method("_update_models")(owner, "checkpoints", [self.root])
            self.assertEqual(lookup.call_count, 1)
            self.assertEqual(get.call_count, 1)
            self.assertEqual(owner.get_image.call_count, 2)
            self.assertEqual(owner.get_image.call_args.args[0]["id"], 12)
            self.assertEqual(owner.get_image.call_args.args[0]["hf_repo_id"], "owner/image")

    def test_existing_failed_hf_raster_is_not_retried_by_the_same_scan_fallback(self):
        model = self.root / "image.gguf"
        model.write_bytes(b"inert file")
        cache = self.root / "cache"
        cache.mkdir()
        image_url = f"https://huggingface.co/owner/image/resolve/{REVISION}/art.png"
        data = dict(
            hf_repo_id="owner/image",
            hf_revision=REVISION,
            _rf_local_sha256=DIGEST,
            images=[dict(url=image_url), dict(url=image_url)],
        )
        owner = SimpleNamespace(
            EXTENSIONS=(".gguf",),
            offline=False,
            cache_paths={"checkpoints": cache},
            model_hash={"checkpoints": {}},
            get_names=lambda _: [model.name],
            get_models_by_path=Mock(return_value=data),
            copy_embedded_preview=Mock(return_value=False),
            read_safetensors_header=lambda _: {},
        )
        owner.get_image = lambda *args: model_method("get_image")(owner, *args)
        metadata = self.manifest()
        metadata["siblings"] = [
            item for item in metadata["siblings"] if not item["rfilename"].endswith(".gif")
        ]
        state = SimpleNamespace(settings=SimpleNamespace(default_settings={}))
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": state}),
            patch.object(
                model_sources.requests,
                "get",
                side_effect=[FakeResponse(status=404), FakeResponse(metadata)],
            ) as get,
        ):
            model_method("_update_models")(owner, "checkpoints", [self.root])
            self.assertEqual(get.call_count, 2)
            self.assertEqual(sum(call.args[0] == image_url for call in get.call_args_list), 1)

    def test_forced_recommendation_refresh_respects_shared_cooldown_and_retains_saved_settings(
        self,
    ):
        from ruinedfooocus.features.models.model_recommendations import ModelRecommendations

        recommendations = ModelRecommendations(
            cache_dir=self.root, path=self.root / "recommendations.json"
        )
        metadata = dict(hf_repo_id="owner/image", _rf_local_sha256=DIGEST)
        key = recommendations.identity("weight.gguf", metadata)
        saved = dict(recommended=dict(custom_steps=30), source="saved publisher settings")
        save_json(recommendations.path, {key: saved})
        with (
            patch.object(recommendations, "metadata", return_value=metadata),
            patch.object(
                model_sources.requests,
                "get",
                return_value=FakeResponse(status=429, retry_after="120"),
            ) as get,
        ):
            for _ in range(3):
                recommendations.refresh("weight.gguf", ["euler"], ["simple"], force=True)
            self.assertEqual(get.call_count, 1)
        retained = recommendations.load()[key]
        self.assertEqual(retained["recommended"], saved["recommended"])
        self.assertEqual(retained["source"], saved["source"])

    def test_checkpoint_cards_show_local_art_or_default_raster_with_no_remote_work(self):
        import gradio as gr
        from ruinedfooocus.core.util import path_manager
        from ruinedfooocus.features.models import ui_browser
        from ruinedfooocus.features.models.resource_cards import default_preview

        names = ["local.gguf", "missing.gguf"]
        for name in names:
            (self.root / name).write_bytes(b"inert file")
        exact = self.root / "local.gguf.preview.png"
        exact.write_bytes(self.image())
        models = SimpleNamespace(
            cache_paths={"checkpoints": self.root / "cache/checkpoints"},
            get_names=lambda _: names,
            get_file=lambda _, name: self.root / name,
            get_models_by_path=Mock(return_value={"baseModel": "SDXL 1.0"}),
        )
        with (
            patch.object(
                model_sources.requests, "get", side_effect=AssertionError("card HTTP")
            ) as http,
            patch.object(ui_browser, "create_components_ui"),
            patch.object(ui_browser, "create_checkpoint_ui"),
            patch.dict(path_manager.model_paths, {"cache_path": self.root / "cache"}),
        ):
            with gr.Blocks(analytics_enabled=False) as app:
                base = gr.Dropdown(choices=names, value=names[0])
                ui_browser.create_ui(base, models)
            cards = next(
                block.value["cards"]
                for block in app.blocks.values()
                if isinstance(block, gr.HTML) and block.elem_id == "checkpoint-cards"
            )
            self.assertEqual(Path(cards[0]["preview"]["path"]).read_bytes(), exact.read_bytes())
            self.assertTrue(valid_preview(cards[1]["preview"]["path"]))
            expected = Path(default_preview("Image model", self.root / "cache"))
            shown = Path(cards[1]["preview"]["path"])
            # Gradio serves a cached copy; retain the versioned artifact identity.
            self.assertEqual(shown.name, expected.name)
            self.assertEqual(shown.read_bytes(), expected.read_bytes())
            self.assertEqual(cards[1]["preview"], cards[1]["fallback"])
            self.assertTrue(cards[1]["preview_is_default"])
            self.assertTrue(
                all(
                    call.kwargs.get("fetch") is False
                    for call in models.get_models_by_path.call_args_list
                )
            )
            http.assert_not_called()

    def test_invalid_or_animated_artwork_preserves_existing_valid_image(self):
        destination = self.root / "art"
        published = write_preview(self.image(), destination)
        original = published.read_bytes()
        animation = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(
            animation, format="GIF", save_all=True, append_images=[Image.new("RGB", (8, 8), "blue")]
        )
        for body in (b"invalid image", animation.getvalue()):
            with self.assertRaises((OSError, ValueError)):
                write_preview(body, destination)
            self.assertEqual(published.read_bytes(), original)

    def test_metadata_cache_and_transport_concurrency_are_bounded(self):
        entered, release = threading.Event(), threading.Event()
        active, maximum = 0, 0
        lock = threading.Lock()

        def get(url, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                if active == 2:
                    entered.set()
            self.assertTrue(release.wait(5))
            with lock:
                active -= 1
            return FakeResponse({})

        with patch.object(model_sources.requests, "get", side_effect=get) as http:
            with ThreadPoolExecutor(max_workers=3) as pool:
                jobs = [
                    pool.submit(model_sources.huggingface_manifest, f"owner/image{i}")
                    for i in range(2)
                ]
                self.assertTrue(entered.wait(5))
                self.assertIsNone(model_sources.huggingface_manifest("owner/third"))
                release.set()
                for job in jobs:
                    job.result(5)
            self.assertEqual(http.call_count, 2)
            self.assertEqual(maximum, 2)
        with patch.object(model_sources.requests, "get", return_value=FakeResponse({})):
            for index in range(140):
                model_sources.huggingface_manifest(f"owner/cache{index}")
        self.assertLessEqual(len(self.metadata.cache), 128)
        self.assertFalse(self.metadata.inflight)


if __name__ == "__main__":
    unittest.main()
