"""Deterministic model projection/cache races use real bounded workers and Gradio callbacks."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.context import LocalContext
from PIL import Image

from ruinedfooocus.app.preparation_admission import PreparationAdmission
from ruinedfooocus.app.resource_lifecycle import LatestUIRequests
from ruinedfooocus.features.models.acquisition import Catalogue
from ruinedfooocus.features.models import ui_requests
from ruinedfooocus.features.models.ui_acquisition import create_ui
import test_model_acquisition as fixtures
from ui_test_helpers import function_from

ROOT = Path(__file__).resolve().parents[1]


class CataloguePublicationTests(unittest.TestCase):
    def test_invalidated_slow_scan_cannot_replace_a_newer_shared_snapshot(self):
        catalogue = Catalogue(None)
        entered, release = threading.Event(), threading.Event()
        calls = []

        def row(label):
            return dict(
                key="a",
                label=label,
                support="",
                category_label="Image models",
                installation="Available",
                task="Image",
                quantization="Q",
            )

        def scan(cancelled=None):
            calls.append(None)
            if len(calls) == 1:
                entered.set()
                self.assertTrue(release.wait(5))
                return [row("older")]
            return [row("newer")]

        with patch.object(catalogue, "_scan", side_effect=scan), ThreadPoolExecutor(1) as threads:
            old = threads.submit(catalogue.records)
            self.assertTrue(entered.wait(5))
            try:
                catalogue.invalidate()
                self.assertEqual(catalogue.records()[0]["label"], "newer")
            finally:
                release.set()
            self.assertEqual(old.result(5)[0]["label"], "older")
            self.assertEqual(catalogue.records()[0]["label"], "newer")
            self.assertEqual(len(calls), 2)

    def test_cancelled_scan_never_populates_cache(self):
        catalogue = Catalogue(SimpleNamespace(DOWNLOADABLE_FILES={"a": {}}))
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(CancelledError):
            catalogue.records(cancelled=cancelled)
        self.assertIsNone(catalogue._snapshot)


class ModelProjectionTests(unittest.TestCase):
    setUp = fixtures.AcquisitionTests.setUp

    def ui(self):
        fence, pool = LatestUIRequests(), PreparationAdmission(max_workers=2, max_jobs=4)
        self.addCleanup(pool.close)
        for target, value in (("latest_ui_requests", fence), ("preparation_admission", pool)):
            change = patch.object(ui_requests, target, value)
            change.start()
            self.addCleanup(change.stop)
        with patch(
            "ruinedfooocus.features.models.acquisition.load_registry", return_value=self.registry
        ):
            with gr.Blocks(analytics_enabled=False) as app:
                ui = create_ui(
                    self.manager, SimpleNamespace(update_all_models=Mock()), offline=lambda: True
                )
        ui.catalogue.registry = self.registry
        self.addCleanup(ui.preview_cache.close)
        functions = {
            function.fn.__name__: function
            for function in app.fns.values()
            if function.fn is not None
        }
        return ui, functions, fence, pool

    @staticmethod
    def request(session):
        return SimpleNamespace(username="owner", session_hash=session, request=None)

    def test_real_filter_callback_discards_late_success_or_error_while_second_session_proceeds(
        self,
    ):
        for fail in (False, True):
            with self.subTest(late_error=fail):
                ui, functions, _, _ = self.ui()
                first, other = self.request("first"), self.request("other")
                entered, release = threading.Event(), threading.Event()
                original = ui.catalogue.records

                def records(**kwargs):
                    if kwargs.get("query") == "slow":
                        entered.set()
                        self.assertTrue(release.wait(5))
                        if fail:
                            raise OSError("late metadata error")
                    return original(**kwargs)

                fn = functions["filter_rows"].fn
                with (
                    patch.object(ui.catalogue, "records", side_effect=records),
                    ThreadPoolExecutor(1) as threads,
                ):
                    old = threads.submit(
                        fn, "slow", "All", "discover", "All", "All", None, 1, first, 1
                    )
                    self.assertTrue(entered.wait(5))
                    try:
                        second_result = fn("", "All", "discover", "All", "All", None, 1, other, 1)
                        self.assertTrue(second_result[0])
                        newest = fn("", "All", "discover", "All", "All", None, 1, first, 2)
                        self.assertEqual(newest[0][0]["_view_version"], 2)
                    finally:
                        release.set()
                    self.assertEqual(old.result(5), tuple(gr.skip() for _ in range(6)))
                self.assertFalse(functions["filter_rows"].queue)
                self.assertEqual(functions["filter_rows"].trigger_mode, "multiple")
                self.assertEqual(
                    functions["filter_rows"].show_progress_on, [functions["filter_rows"].outputs[4]]
                )

    def test_selection_ack_is_light_and_old_details_actions_and_other_owner_are_rejected(self):
        ui, functions, fence, _ = self.ui()
        request = self.request("first")
        rows = functions["filter_rows"].fn(
            "", "All", "discover", "All", "All", None, 1, request, 1
        )[0]
        first, second = [row["key"] for row in rows[:2]]
        choose = functions["select_record"].fn
        with patch.object(
            ui.catalogue,
            "companion_status",
            side_effect=AssertionError("selection acknowledgement must not inspect metadata"),
        ):
            result = choose(
                rows, "text", True, gr.EventData(None, {"key": first}), request=request, version=1
            )
            self.assertEqual(result, (first, *(gr.skip() for _ in range(4))))
            choose(
                rows, "text", True, gr.EventData(None, {"key": second}), request=request, version=2
            )
        details = functions["selected_details"].fn(first, rows, request=request, version=1)
        self.assertEqual(details, tuple(gr.skip() for _ in range(4)))
        empty = functions["selected_details"].fn(None, rows, request=request, version=2)
        self.assertEqual(empty, tuple(gr.skip() for _ in range(4)))
        for selected, owner in ((first, request), (second, self.request("other"))):
            with self.assertRaisesRegex(gr.Error, "selection changed"):
                ui.require_source(selected, rows, owner)
        ui.require_source(second, rows, request)
        functions["filter_rows"].fn("", "All", "discover", "All", "All", None, 1, request, 2)
        with self.assertRaises(gr.Error):
            ui.require_source(second, rows, request)
        self.assertEqual(
            choose(
                rows, "text", True, gr.EventData(None, {"key": first}), request=request, version=3
            ),
            tuple(gr.skip() for _ in range(5)),
        )
        with self.assertRaises(CancelledError):
            fence.peek(("ui", "owner", "local", "first"), "acquisition-selection")

    def test_superseding_attached_queued_work_returns_capacity_before_active_work_finishes(self):
        pool = PreparationAdmission(max_workers=1, max_jobs=2)
        self.addCleanup(pool.close)
        fence, entered, release = LatestUIRequests(), threading.Event(), threading.Event()

        def active():
            entered.set()
            self.assertTrue(release.wait(5))

        running = pool.submit_work(active)
        self.assertTrue(entered.wait(5))
        first = fence.begin(("ui", "one"), "model", 1)
        queued = fence.attach(
            first,
            pool.submit_work(lambda: self.fail("superseded work ran"), cancelled=first.cancelled),
        )
        try:
            newest = fence.begin(first.owner, first.channel, 2)
            self.assertTrue(queued.cancelled())
            retry = fence.attach(
                newest, pool.submit_work(lambda: "latest", cancelled=newest.cancelled)
            )
        finally:
            release.set()
        running.result(5)
        self.assertEqual(retry.result(5), "latest")

    def test_filtered_out_selection_remains_source_fenced_and_available_for_deliberate_use(self):
        ui, functions, _, _ = self.ui()
        request = self.request("first")
        rows = functions["filter_rows"].fn(
            "", "All", "discover", "All", "All", None, 1, request, 1
        )[0]
        selected = rows[0]["key"]
        functions["select_record"].fn(
            rows, "text", True, gr.EventData(None, {"key": selected}), request=request, version=1
        )
        result = functions["filter_rows"].fn(
            "no matching files at all", "All", "discover", "All", "All", selected, 1, request, 2
        )
        current_rows, cards, selection = result[:3]
        self.assertEqual(cards["cards"], [])
        self.assertEqual(selection, gr.skip())
        kept = next(row for row in current_rows if row["key"] == selected)
        self.assertTrue(kept["_details_only"])
        self.assertEqual(kept["_view_version"], 2)
        ui.require_source(selected, current_rows, request)
        with self.assertRaises(gr.Error):
            ui.require_source(selected, rows, request)
        with self.assertRaises(gr.Error):
            ui.require_source(selected, current_rows, self.request("other"))

    def test_actual_card_script_does_not_paint_a_delayed_old_view(self):
        script = r"""
const fs = require('fs'), vm = require('vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
function render(version) {
  try {
    const context = {
      window: {__rfModelUIRequests: {'acquisition-view': 2}},
      props: {value: {requestIdentity: {channel: 'acquisition-view', version}}},
      element: {addEventListener() {}, querySelector() {throw new Error('paint attempted');}},
      watch() {}
    };
    vm.runInNewContext(source, context);
    // A query has already changed locally even before its acknowledgement;
    // the previously painted card must not dispatch an action in that gap.
    context.choose('previous file', 'activate');
    return false;
  } catch (error) {
    if (error.message !== 'paint attempted') throw error;
    return true;
  }
}
process.stdout.write(JSON.stringify({old: render(1), current: render(2)}));
"""
        result = subprocess.run(
            ["node", "-e", script, str(ROOT / "ruinedfooocus/ui/assets/adapter_cards.js")],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout), {"old": False, "current": True})

    def test_local_checkpoint_description_keeps_latest_model_during_a_slow_older_lookup(self):
        self.ui()  # Install the isolated production request fence and bounded pool.
        entered, release, block = threading.Event(), threading.Event(), threading.Event()
        preview = self.root / "preview.png"
        Image.new("RGBA", (7, 11)).save(preview)

        def metadata(category, name, fetch=False):
            if block.is_set() and name == "older.safetensors":
                entered.set()
                self.assertTrue(release.wait(5))
            return {"baseModel": name}

        models = SimpleNamespace(
            get_names=lambda _: ["older.safetensors", "newer.safetensors"],
            get_models_by_path=metadata,
            get_file=lambda *args: self.root / args[-1],
            cache_paths={"checkpoints": self.root / "cache/checkpoints"},
        )
        create = function_from(
            "ruinedfooocus/features/models/ui_browser.py",
            "create_ui",
            dict(
                gr=gr,
                Path=Path,
                __file__=str(ROOT / "ruinedfooocus/features/models/ui_browser.py"),
                models=models,
                get_checkpoint_thumbnail=lambda name: str(preview),
                checkpoint_records=lambda names, reader, search: [
                    dict(name=name, caption=name, label=name) for name in names
                ],
                model_label=lambda name, info: name,
                selected_checkpoint=Mock(),
                localized_output=Mock(),
                localize=lambda text, locale=None: text,
                resolve_locale=lambda **kwargs: "en",
                create_components_ui=Mock(),
                create_checkpoint_ui=Mock(),
            ),
        )
        with gr.Blocks(analytics_enabled=False) as app:
            create(gr.Dropdown(choices=models.get_names(None), value="newer.safetensors"), models)
        function = next(
            function for function in app.fns.values() if function.fn.__name__ == "describe"
        )

        def call(name, version):
            token = LocalContext.request.set(gr.Request(username="owner", session_hash="first"))
            try:
                return function.fn(name, version)
            finally:
                LocalContext.request.reset(token)

        block.set()
        with ThreadPoolExecutor(1) as threads:
            old = threads.submit(call, "older.safetensors", 1)
            self.assertTrue(entered.wait(5))
            try:
                newest = call("newer.safetensors", 2)
            finally:
                release.set()
            self.assertIn("newer.safetensors", newest)
            self.assertEqual(old.result(5), gr.skip())
        self.assertFalse(function.queue)
        self.assertEqual(function.trigger_mode, "multiple")


if __name__ == "__main__":
    unittest.main()
