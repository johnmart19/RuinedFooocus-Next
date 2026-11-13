"""Checkpoint presentation work is bounded by a query, not each selection."""

import ast
import asyncio
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.state_holder import SessionState
from PIL import Image

from ruinedfooocus.core.paths import asset_path
from ruinedfooocus.core.translation_manager import localize
from ruinedfooocus.features.models import catalogue_previews, resource_cards
from ruinedfooocus.features.models.selection import (
    checkpoint_records,
    model_label,
    selected_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


class CheckpointFixture:
    def __init__(self, count=19):
        self.temporary = tempfile.TemporaryDirectory(prefix="rf-model-card-perf-")
        self.root = Path(self.temporary.name).resolve()
        assert self.root.parent == Path(tempfile.gettempdir()).resolve()
        self.counts = Counter()
        self.cache = {}
        self.names = [f"model-{index:02}.safetensors" for index in range(count)]
        for index, name in enumerate(self.names):
            (self.root / name).write_bytes(b"Inert fixture; presentation never opens weights")
            if index < count // 2:
                Image.new("RGB", (512, 512), (index, 40, 90)).save(
                    self.root / f"{name}.preview.jpg"
                )
        self.manager = SimpleNamespace(model_paths={"cache_path": self.root / "setup-cache"})
        self.models = SimpleNamespace(
            get_names=self.get_names,
            get_models_by_path=self.metadata,
            get_file=lambda category, name: self.root / name,
            cache_paths={"checkpoints": self.manager.model_paths["cache_path"] / "checkpoints"},
        )
        utility = ROOT / "ruinedfooocus/core/util.py"
        tree = ast.parse(utility.read_text(encoding="utf-8"))
        self.utility_scope = dict(
            Path=Path, path_manager=self.manager, shared_cache=self.cache, asset_path=asset_path
        )
        exec(
            compile(
                ast.Module(
                    body=[
                        node
                        for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name in {"_get_model_thumbnail", "get_checkpoint_thumbnail"}
                    ],
                    type_ignores=[],
                ),
                str(utility),
                "exec",
            ),
            self.utility_scope,
        )
        source = ROOT / "ruinedfooocus/features/models/ui_browser.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        self.localized = []
        self.scope = dict(
            __file__=str(source),
            Path=Path,
            gr=gr,
            translate_ui=lambda text: text,
            localize=localize,
            resolve_locale=lambda **kwargs: "en",
            localized_output=lambda fn, inputs, outputs: self.localized.append(
                (fn, inputs, outputs)
            ),
            checkpoint_records=checkpoint_records,
            selected_checkpoint=selected_checkpoint,
            model_label=model_label,
            get_checkpoint_thumbnail=self.thumbnail,
            create_components_ui=lambda *args, **kwargs: None,
            create_checkpoint_ui=lambda *args, **kwargs: None,
        )
        exec(
            compile(
                ast.Module(
                    body=[
                        node
                        for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "create_ui"
                    ],
                    type_ignores=[],
                ),
                str(source),
                "exec",
            ),
            self.scope,
        )
        with gr.Blocks(analytics_enabled=False) as self.app:
            self.base = gr.Dropdown(choices=self.names, value=self.names[0])
            self.query, self.gallery, self.rows, self.refresh = self.scope["create_ui"](
                self.base, self.models
            )
        self.cards = next(
            block
            for block in self.app.blocks.values()
            if getattr(block, "elem_id", None) == "checkpoint-cards"
        )
        self.project = next(
            fn.fn
            for fn in self.app.fns.values()
            if getattr(fn.fn, "__name__", "") == "visible_cards_for"
        )
        self.query_event = next(
            event
            for event in self.app.config["dependencies"]
            if self.app.fns[event["id"]].fn is self.refresh
        )
        self.cards_event = next(
            event
            for event in self.app.config["dependencies"]
            if event["trigger_after"] == self.query_event["id"]
        )
        self.state = SessionState(self.app)
        self.request = gr.Request(username="owner", session_hash=f"checkpoint-work-{id(self)}")
        self.query_version = 1
        self.cards_version = 0

    def query_api(self, search=""):
        self.query_version += 1
        return asyncio.run(
            self.app.process_api(
                self.query_event["id"],
                [search, self.query_version],
                state=self.state,
                request=self.request,
            )
        )

    def cards_api(self, current_model=None):
        self.cards_version += 1
        return asyncio.run(
            self.app.process_api(
                self.cards_event["id"],
                [None, current_model or self.names[0], self.cards_version],
                state=self.state,
                request=self.request,
            )
        )["data"][0]

    def get_names(self, category):
        self.counts["catalogue_reads"] += 1
        return self.names

    def metadata(self, category, name, fetch=False):
        assert fetch is False
        self.counts["metadata_reads"] += 1
        return {}

    def thumbnail(self, name):
        self.counts["thumbnail_lookups"] += 1
        return self.utility_scope["get_checkpoint_thumbnail"](name)

    def observe(self):
        stack = ExitStack()
        for module, name, key in (
            (catalogue_previews, "local_checkpoint_preview", "local_preview_lookups"),
            (resource_cards, "default_preview", "default_preview_lookups"),
            (Image, "open", "image_opens"),
        ):
            original = getattr(module, name)

            def wrapped(*args, _original=original, _key=key, **kwargs):
                self.counts[_key] += 1
                return _original(*args, **kwargs)

            stack.enter_context(patch.object(module, name, wrapped))
        return stack

    def cold_cache(self, index):
        self.manager.model_paths["cache_path"] = self.root / f"cache-{index}"
        self.models.cache_paths["checkpoints"] = (
            self.manager.model_paths["cache_path"] / "checkpoints"
        )
        self.cache.clear()
        resource_cards._bundled_default.cache_clear()
        resource_cards._valid_preview.cache_clear()

    def close(self):
        self.temporary.cleanup()


class CheckpointPresentationWorkTests(unittest.TestCase):
    def setUp(self):
        self.fixture = CheckpointFixture()
        self.addCleanup(self.fixture.close)

    def test_query_resolves_artwork_once_for_both_hidden_and_visible_cards(self):
        fixture = self.fixture
        with fixture.observe():
            fixture.counts.clear()
            response = fixture.query_api()
            self.assertEqual(len(response["data"][0]["value"]), 19)
            self.assertEqual(fixture.counts["metadata_reads"], 19)
            self.assertEqual(fixture.counts["local_preview_lookups"], 19)
            self.assertEqual(fixture.counts["thumbnail_lookups"], 10)
            self.assertEqual(fixture.counts["default_preview_lookups"], 11)
            fixture.counts.clear()
            cards = fixture.cards_api()
            self.assertEqual(len(cards["cards"]), 19)
            self.assertEqual(fixture.counts, {})

    def test_selection_and_locale_reuse_query_artwork_without_mutating_rows(self):
        fixture = self.fixture
        fixture.query_api()
        rows = fixture.state[fixture.rows._id]
        original = deepcopy(rows)
        with fixture.observe():
            fixture.counts.clear()
            selected = fixture.cards_api(fixture.names[-1])
            localized = fixture.localized[0][0]("uk", rows, fixture.names[-1])
            self.assertEqual(fixture.counts, {})
        self.assertEqual(selected["selected"], rows[-1]["caption"])
        self.assertEqual(localized["heading"], localize("Image models", "uk"))
        self.assertEqual(localized["selected"], rows[-1]["caption"])
        self.assertEqual(rows, original)

    def test_explicit_refresh_rebuilds_artwork_and_empty_search_does_no_preview_work(self):
        fixture = self.fixture
        fixture.query_api()
        before = fixture.cards_api()
        self.assertTrue(before["cards"][-1]["preview_is_default"])
        preview = fixture.root / f"{fixture.names[-1]}.preview.jpg"
        Image.new("RGB", (512, 512), (90, 30, 20)).save(preview)
        fixture.query_api()
        after = fixture.cards_api()
        self.assertFalse(after["cards"][-1]["preview_is_default"])
        self.assertEqual(
            Path(after["cards"][-1]["preview"]["path"]).read_bytes(), preview.read_bytes()
        )
        self.assertGreater(
            after["requestIdentity"]["version"], before["requestIdentity"]["version"]
        )
        with fixture.observe():
            fixture.counts.clear()
            fixture.query_api("no matching fixture")
            empty = fixture.cards_api()
            self.assertEqual(empty["cards"], [])
            self.assertEqual(fixture.counts["thumbnail_lookups"], 0)
            self.assertEqual(fixture.counts["default_preview_lookups"], 0)
            self.assertEqual(fixture.counts["image_opens"], 0)


if __name__ == "__main__":
    unittest.main()
