"""Original identity and job ownership across native Latest and in-Editor picker events."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import unittest
import threading
from unittest.mock import Mock, patch

import gradio as gr
from PIL import Image
from PIL.PngImagePlugin import PngImageFile, PngInfo

from ruinedfooocus.features.gallery.actions import source_record
from ruinedfooocus.features.gallery.index_store import connect_index
from ruinedfooocus.features.image_processing.native_image_outputs import (
    latest_generated_result,
    output_role,
    result_presentation,
)
import test_editor_workspace_events as editor_fixtures
from ui_test_helpers import function_from


class EditorGalleryImports(unittest.TestCase):
    def setUp(self):
        self.fixture = editor_fixtures.EditorWorkspaceEventTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.workspace, self.root = self.fixture.workspace, self.fixture.root
        self.session = self.fixture.session("picker-owner")
        self.record = source_record(self.fixture.legacy.path)
        self.browser = SimpleNamespace(
            load_images=Mock(side_effect=self.page), update_images=Mock(return_value=None)
        )
        with self.fixture.app:
            self.workspace.bind_picker(self.browser)

    def page(self, page, **options):
        return (
            [(self.record["path"], self.record["caption"])],
            "fixture image",
            [self.record],
            {"count": 1, "pages": 1, "page": 1},
        )

    def function(self, name):
        return next(
            function
            for function in self.fixture.app.fns.values()
            if getattr(function.fn, "__name__", "") == name
        )

    def browse(self):
        updates = self.fixture.call(self.session, self.fixture.event(self.workspace.choose_gallery))
        descriptor = next(control for control in updates if isinstance(control, gr.JSON))
        self.fixture.call(self.session, self.function("prepare_picker"))
        return descriptor

    def select(self):
        event = gr.SelectData(
            self.workspace.picker_gallery,
            {"index": 0, "value": {"caption": self.record["caption"]}, "selected": True},
        )
        asyncio.run(self.fixture.api(self.session, self.function("pick"), event_data=event))

    def native_browser(self):
        """Use production index/thumbnail methods over private valid image originals."""
        from gradio.processing_utils import save_file_to_cache

        outputs = self.root / "picker-originals"
        outputs.mkdir()
        self.originals = {}
        for number in range(9):
            path = outputs / f"{number}.png"
            Image.new("RGBA", (27, 19), (number, 40, 60, 127)).save(path)
            self.originals[path] = path.read_bytes()
        manager = SimpleNamespace(
            model_paths={"temp_outputs_path": outputs, "cache_path": self.root / "picker-cache"},
            paths={"path_outputs": outputs},
            get_abspath=lambda path: path,
        )
        scope = dict(
            Path=Path,
            Image=Image,
            PngImageFile=PngImageFile,
            Dict=dict,
            settings=SimpleNamespace(default_settings={}),
            path_manager=manager,
            connect_database=connect_index,
            save_file_to_cache=save_file_to_cache,
            get_upload_folder=lambda: str(self.root / "picker-delivery"),
        )
        source = "ruinedfooocus/features/gallery/imagebrowser.py"
        function_from(source, "get_png_metadata", scope)
        methods = {
            name: function_from(source, name, scope)
            for name in ("__init__", "load_images", "update_images")
        }
        browser = type("NativePickerBrowser", (), methods)()
        self.addCleanup(browser.sql_conn.close)
        self.browser.load_images = browser.load_images
        self.browser.update_images = Mock(wraps=browser.update_images)
        return browser, scope

    def test_cold_picker_indexes_originals_once_and_pages_without_scanning_or_changing_draft(self):
        browser, _ = self.native_browser()
        before = {
            name: self.fixture.value(self.session, self.fixture.controls[name])
            for name in ("prompt", "negative", "base_model_name")
        }
        self.browse()
        prepare = self.function("prepare_picker")
        self.assertTrue(prepare.queue)
        self.assertEqual(prepare.show_progress_on, [self.workspace.picker_notice])
        self.assertEqual((browser.last_scan.count, browser.last_scan.parsed), (9, 9))
        self.assertEqual(len(self.fixture.value(self.session, self.workspace.picker_records)), 8)
        self.assertEqual(self.fixture.value(self.session, self.workspace.picker_total), "1 / 2")
        with patch(
            "ruinedfooocus.features.gallery.index_store.os.walk",
            side_effect=AssertionError("picker paging scanned outputs"),
        ):
            self.fixture.call(self.session, self.fixture.event(self.workspace.picker_next))
            self.fixture.call(self.session, prepare)
            self.assertEqual(
                len(self.fixture.value(self.session, self.workspace.picker_records)), 1
            )
            self.assertEqual(self.fixture.value(self.session, self.workspace.picker_total), "2 / 2")
            self.browse()
        self.assertTrue(
            all(path.read_bytes() == contents for path, contents in self.originals.items())
        )
        for name, value in before.items():
            self.assertEqual(self.fixture.value(self.session, self.fixture.controls[name]), value)
        self.assertIsNone(self.fixture.value(self.session, self.workspace.context))
        self.fixture.run_request.assert_not_called()
        self.fixture.stop_request.assert_not_called()

    def test_newer_cold_picker_request_rejects_old_scan_commit_and_retry_initializes(self):
        browser, scope = self.native_browser()
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        metadata = scope["get_png_metadata"]

        def delayed(path):
            entered.set()
            self.assertTrue(release.wait(5))
            return metadata(path)

        self.fixture.call(self.session, self.fixture.event(self.workspace.choose_gallery))
        prepare = self.function("prepare_picker")
        with (
            patch.dict(scope, {"get_png_metadata": delayed}),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            pending = pool.submit(self.fixture.call, self.session, prepare)
            try:
                self.assertTrue(entered.wait(5))
                self.fixture.call(self.session, self.fixture.event(self.workspace.choose_gallery))
            finally:
                release.set()
            pending.result(5)
        self.assertEqual(
            browser.sql_conn.execute("SELECT revision FROM library_snapshot").fetchone()[0], 0
        )
        self.assertEqual(self.fixture.value(self.session, self.workspace.picker_records), [])
        self.fixture.call(self.session, prepare)
        self.assertEqual((browser.last_scan.count, browser.last_scan.parsed), (9, 9))
        self.assertEqual(len(self.fixture.value(self.session, self.workspace.picker_records)), 8)

    def test_picker_stays_inside_editor_and_retains_exact_original_alpha_and_draft(self):
        original = Path(self.record["path"]).read_bytes()
        self.browse()
        self.select()
        before = self.fixture.value(self.session, self.fixture.controls["prompt"])
        self.fixture.call(self.session, self.fixture.event(self.workspace.picker_open))
        context = self.fixture.value(self.session, self.workspace.context)
        snapshot = self.fixture.store.snapshot(context["session_id"], owner=self.session.owner)
        self.assertEqual(Path(snapshot.original.path), Path(self.record["path"]))
        self.assertEqual(snapshot.current.frame.size, (73, 121))
        with Image.open(snapshot.current.path) as image:
            self.assertEqual(image.getpixel((0, 0)), (20, 40, 60, 127))
        self.assertEqual(self.fixture.value(self.session, self.fixture.controls["prompt"]), before)
        self.assertEqual(Path(self.record["path"]).read_bytes(), original)
        self.assertEqual(self.browser.load_images.call_args.kwargs["page_size"], 8)
        self.fixture.run_request.assert_not_called()
        self.fixture.stop_request.assert_not_called()

    def test_new_page_and_deleted_source_reject_old_picker_choice(self):
        self.browse()
        self.select()
        self.fixture.call(self.session, self.fixture.event(self.workspace.choose_gallery))
        with self.assertRaisesRegex(gr.Error, "image list changed"):
            self.fixture.call(self.session, self.fixture.event(self.workspace.picker_open))
        self.fixture.call(self.session, self.function("prepare_picker"))
        self.select()
        Path(self.record["path"]).unlink()
        with self.assertRaisesRegex(gr.Error, "image list changed"):
            self.fixture.call(self.session, self.fixture.event(self.workspace.picker_open))
        self.assertIsNone(self.fixture.value(self.session, self.workspace.context))

    def test_pending_job_defers_picker_and_latest_without_touching_terminal(self):
        self.fixture.open_library(self.session)
        before = self.fixture.value(self.session, self.workspace.context).copy()
        self.browse()
        self.select()
        _, _, records, _ = result_presentation([self.record["path"]])
        for button in (self.workspace.picker_open, self.workspace.open_latest):
            self.fixture.call(
                self.session,
                self.fixture.event(button),
                {
                    self.fixture.remote["task"]: 123,
                    self.workspace.latest_generated: latest_generated_result(records),
                },
            )
            self.assertEqual(self.fixture.value(self.session, self.workspace.context), before)
            self.assertEqual(self.fixture.value(self.session, self.fixture.remote["task"]), 123)
        self.fixture.stop_request.assert_not_called()

    def test_slow_import_cannot_publish_over_a_newer_source(self):
        from ruinedfooocus.features.gallery.editing_workflow import working_source

        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        older = self.fixture.legacy.path
        newer = self.root / "newer-source.png"
        Image.new("RGBA", (35, 53), (31, 57, 81, 129)).save(newer)

        def delayed(record):
            if Path(record["path"]) == older:
                entered.set()
                self.assertTrue(release.wait(5))
            return working_source(record)

        callback = self.fixture.event(self.fixture.library_open).fn

        def load(record, number):
            return callback(
                record,
                {},
                "Rendered prompt",
                "draft",
                "negative",
                "model",
                "Custom...",
                512,
                768,
                True,
                None,
                number,
                {},
                self.session.request,
            )

        with (
            patch(
                "ruinedfooocus.features.gallery.ui_workspace.working_source", side_effect=delayed
            ),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            pending = executor.submit(load, source_record(older), 1)
            try:
                self.assertTrue(entered.wait(2))
                updates = load(source_record(newer), 2)
                self.assertEqual(updates[self.workspace.context]["revision"], 1)
            finally:
                release.set()
            self.assertEqual(pending.result(5), gr.skip())
        current = self.fixture.store.snapshot(
            updates[self.workspace.context]["session_id"], owner=self.session.owner
        )
        self.assertEqual(Path(current.original.path), newer)

    def test_latest_uses_selected_completed_batch_original_and_rejects_replacement(self):
        first, second = self.root / "first.png", self.root / "selected.png"
        Image.new("RGBA", (27, 41), (1, 2, 3, 99)).save(first)
        Image.new("RGBA", (43, 29), (4, 5, 6, 101)).save(second)
        _, _, records, _ = result_presentation([str(first), str(second)])
        selected = latest_generated_result(records, 1)
        self.fixture.call(
            self.session,
            self.fixture.event(self.workspace.open_latest),
            {
                self.workspace.latest_generated: selected,
            },
        )
        context = self.fixture.value(self.session, self.workspace.context)
        snapshot = self.fixture.store.snapshot(context["session_id"], owner=self.session.owner)
        self.assertEqual(Path(snapshot.original.path), second)
        self.assertEqual(snapshot.current.frame.size, (43, 29))
        self.assertEqual(selected["batch_index"], 1)
        Image.new("RGBA", (43, 29), "red").save(second)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.fixture.call(self.session, self.fixture.event(self.workspace.open_latest))

    def test_semantic_technical_outputs_remain_accessible_but_do_not_replace_latest(self):
        cases = [
            ({}, "image"),
            ({"kind": "processing", "image_utility_source": {"operation": "moge-depth"}}, "guide"),
            (
                {
                    "image_utility_source": {
                        "operation": "image-mask",
                        "mask_role": "inpaint-selection",
                    }
                },
                "mask",
            ),
            ({"native_vision_source": {"native": "sam3"}}, "guide"),
            ({"Data": "Preview Grid"}, "preview"),
            ({"output_role": "helper"}, "helper"),
        ]
        paths = []
        import json

        for index, (metadata, role) in enumerate(cases):
            path = self.root / f"role-{index}.png"
            info = PngInfo()
            info.add_text("parameters", json.dumps(metadata))
            Image.new("RGBA", (9, 7), (10, 20, 30, 80)).save(path, pnginfo=info)
            self.assertEqual(output_role(metadata), role)
            paths.append(str(path))
        _, _, records, _ = result_presentation(paths)
        self.assertEqual(len(records), len(cases))
        self.assertEqual(latest_generated_result(records)["path"], str(Path(paths[0]).resolve()))
        for index in range(1, len(cases)):
            self.assertIsNone(latest_generated_result(records, index))
            self.assertTrue(Path(records[index]["path"]).is_file())
        self.fixture.call(
            self.session,
            self.fixture.event(self.fixture.present_results),
            {self.fixture.products: [paths[0]]},
        )
        self.fixture.call(
            self.session,
            self.fixture.event(self.fixture.present_results),
            {self.fixture.products: paths[1:]},
        )
        retained = self.fixture.value(self.session, self.workspace.latest_generated)
        self.assertEqual(retained["path"], paths[0])
        self.assertEqual(
            len(self.fixture.value(self.session, self.fixture.remote["artifacts"])), len(cases) - 1
        )


if __name__ == "__main__":
    unittest.main()
