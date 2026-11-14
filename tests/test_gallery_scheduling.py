"""Construct installed Gradio events and exercise actual Library callbacks without a server."""

import ast
import asyncio
import json
import hashlib
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio import processing_utils
from gradio.state_holder import SessionState
from PIL import Image
from PIL.PngImagePlugin import PngImageFile, PngInfo

from ruinedfooocus.app.preparation_admission import PreparationAdmission
from ruinedfooocus.app.resource_lifecycle import LatestUIRequests
from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui
from ruinedfooocus.features.gallery.actions import select_source, source_path, source_record
from ruinedfooocus.features.gallery.image_metadata import read_metadata
from ruinedfooocus.features.gallery.index_store import MEDIA_SUFFIXES, connect_index
from ui_test_helpers import function_from

ROOT = Path(__file__).resolve().parents[1]


class GallerySchedulingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rf-gallery-events-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "original.png"
        Image.new("RGBA", (7, 11), (20, 40, 60, 127)).save(self.source)
        self.record = source_record(self.source)
        self.pool = PreparationAdmission()
        self.addCleanup(self.pool.close)
        self.fence = LatestUIRequests()
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/gallery/ui.py").read_text(encoding="utf-8")
        )
        create = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "create_image_gallery"
        )
        script = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "request_version_js"
                for target in node.targets
            )
        )

        def load(page, term="", **kwargs):
            result = ([], term or "empty", [self.record.copy()])
            return (
                (*result, {"count": 1, "pages": 1, "page": 1})
                if kwargs.get("include_page")
                else result
            )

        self.browser = SimpleNamespace(
            load_images=load,
            num_images_pages=lambda *args: (1, 1),
            update_images=lambda **kwargs: "refreshed",
        )
        self.scope = dict(
            gr=gr,
            Path=Path,
            MEDIA_SUFFIXES=MEDIA_SUFFIXES,
            json=json,
            Image=Image,
            t=lambda text: text,
            translate_ui=translate_ui,
            localize=localize,
            resolve_locale=resolve_locale,
            read_metadata=read_metadata,
            select_source=select_source,
            source_path=source_path,
            get_png_metadata=lambda path: {"parameters": json.dumps({"prompt": "saved prompt"})},
            get_media_metadata=lambda path: {},
            format_metadata_string=lambda metadata: "selected metadata",
            request_version_js=script,
            browser=self.browser,
        )
        exec(
            compile(ast.Module(body=[create], type_ignores=[]), "Library scheduling", "exec"),
            self.scope,
        )
        with (
            patch("ruinedfooocus.app.preparation_admission.preparation_admission", self.pool),
            patch("ruinedfooocus.app.resource_lifecycle.latest_ui_requests", self.fence),
        ):
            self.app, self.open_view, *_ = self.scope["create_image_gallery"]()
        self.functions = {
            function.fn.__name__: function
            for function in self.app.fns.values()
            if function.fn is not None
        }
        self.request = SimpleNamespace(username="owner", session_hash="first")
        self.gallery = next(
            component
            for component in self.app.blocks.values()
            if component.elem_id == "output-browser"
        )
        self.selected = next(
            component
            for component in self.app.blocks.values()
            if isinstance(component, gr.State) and component.value is None
        )
        self.metadata_output = next(
            component
            for component in self.app.blocks.values()
            if component.elem_id == "browser-metadata"
        )

    def selection_event(self, record=None):
        record = record or self.record
        return SimpleNamespace(index=0, value={"caption": record["caption"]}, selected=True)

    def native_browser(self):
        outputs = self.root / "originals"
        outputs.mkdir()
        originals = {}
        for number in range(3):
            path = outputs / f"{number}.png"
            metadata = PngInfo()
            metadata.add_text(
                "parameters",
                json.dumps(
                    {"prompt": f"saved {number} needle" if number == 2 else f"saved {number}"}
                ),
            )
            with Image.new("RGBA", (1056, 32), (number, 40, 60, 127)) as image:
                image.save(path, pnginfo=metadata)
            originals[path] = path.read_bytes()
        manager = SimpleNamespace(
            model_paths={"temp_outputs_path": outputs, "cache_path": self.root / "cache"},
            paths={"path_outputs": outputs},
            get_abspath=lambda path: path,
        )
        scope = dict(
            Path=Path,
            Image=Image,
            PngImageFile=PngImageFile,
            Dict=dict,
            settings=SimpleNamespace(default_settings={"images_per_page": 2}),
            path_manager=manager,
            connect_database=connect_index,
            save_file_to_cache=processing_utils.save_file_to_cache,
            get_upload_folder=lambda: str(self.root / "delivery"),
        )
        source = "ruinedfooocus/features/gallery/imagebrowser.py"
        function_from(source, "get_png_metadata", scope)
        methods = {
            name: function_from(source, name, scope)
            for name in ("__init__", "num_images_pages", "load_images", "update_images")
        }
        browser = type("NativeGalleryBrowser", (), methods)()
        self.addCleanup(browser.sql_conn.close)
        return browser, scope, originals

    def check_native_first_visit(self, *, indexed):
        import importlib

        # Patch the modules the Gallery imports now; another suite may have
        # replaced them in sys.modules after the package attributes were bound.
        thumbnails = importlib.import_module("ruinedfooocus.features.gallery.thumbnails")
        index_store = importlib.import_module("ruinedfooocus.features.gallery.index_store")

        browser, native, originals = self.native_browser()
        if indexed:
            browser.update_images()
        self.scope["browser"] = browser
        copies = Mock(wraps=native["save_file_to_cache"])
        with (
            patch("ruinedfooocus.app.preparation_admission.preparation_admission", self.pool),
            patch("ruinedfooocus.app.resource_lifecycle.latest_ui_requests", self.fence),
            patch.dict(native, save_file_to_cache=copies),
            patch.object(Image, "open", wraps=Image.open) as opened,
            patch.object(thumbnails, "_thumbnail", wraps=thumbnails._thumbnail) as decoded,
            patch.object(index_store.os, "walk", wraps=os.walk) as walked,
        ):
            app, refresh, *_ = self.scope["create_image_gallery"]()
            startup_work = {
                "output_walks": walked.call_count,
                "original_opens": sum(
                    Path(call.args[0]) in originals for call in opened.call_args_list
                ),
                "grid_thumbnails": decoded.call_count,
                "delivery_copies": copies.call_count,
            }
            self.assertEqual(startup_work, dict.fromkeys(startup_work, 0))
            functions = {
                function.fn.__name__: function
                for function in app.fns.values()
                if function.fn is not None
            }
            gallery = next(
                control for control in app.blocks.values() if control.elem_id == "output-browser"
            )
            self.assertEqual(gallery.value, [])
            state = SessionState(app)
            prepare = functions["prepare_view"]
            records = next(
                control
                for control in prepare.outputs
                if isinstance(control, gr.State) and control.value == []
            )
            self.assertEqual(state[records._id], [])
            # Opening Gallery shows the indexed page at once and requests a scan.
            opened = refresh(1, "", 1, self.request)
            request_output = next(
                control
                for control in opened
                if isinstance(control, gr.JSON) and isinstance(opened[control], dict)
            )
            descriptor = opened[request_output]
            self.assertTrue(descriptor["refresh"])
            self.assertEqual(len(opened[records]), 2 if indexed else 0)
            self.assertTrue(all(record["_view_version"] == 1 for record in opened[records]))

            async def visit():
                result = await prepare.fn(descriptor, self.request)
                processed = await app.postprocess_data(prepare, result, state)
                self.assertEqual(len(processed), len(prepare.outputs))
                return result

            result = asyncio.run(visit())
            self.assertEqual(walked.call_count, 1)
            self.assertEqual(browser.last_scan.count, 3)
            self.assertEqual(browser.last_scan.parsed, 0 if indexed else 3)
            # Each page is rendered once per visit: by the open when the index is
            # unchanged, otherwise by the scan.
            self.assertEqual(decoded.call_count, 2)
            if indexed:
                self.assertNotIn(gallery, result)
                self.assertNotIn(records, result)
                self.assertIn(functions["prepare_view"].outputs[2], result)
            else:
                self.assertEqual(copies.call_count, 2)
                self.assertEqual(len(state[records._id]), 2)
                self.assertTrue(all(record["_view_version"] == 1 for record in state[records._id]))
                self.assertEqual(len(result[gallery].value), 2)
            page = next(
                control for control in app.blocks.values() if control.elem_id == "browser-page"
            )
            rendered = opened if indexed else result
            self.assertEqual(rendered[page]["maximum"], 2)
            for control in prepare.outputs:
                if isinstance(control, gr.Button) and str(control.value) in (
                    "Use prompt",
                    "Reuse full setup",
                    "Open in Editor",
                ):
                    self.assertFalse(control.interactive)
                    self.assertFalse(rendered[control]["interactive"])
            page_request = functions["request_next"].fn(1, "", 2, self.request)
            following = asyncio.run(prepare.fn(page_request, self.request))
            self.assertEqual(len(following[records]), 1)
            search_request = functions["request_search"].fn(1, "needle", 3, self.request)
            searched = asyncio.run(prepare.fn(search_request, self.request))
            self.assertEqual(len(searched[records]), 1)
            self.assertEqual(Path(searched[records][0]["path"]).name, "2.png")
            self.assertEqual(searched[page]["maximum"], 1)
            self.assertEqual(walked.call_count, 1)
        self.assertTrue(all(path.read_bytes() == value for path, value in originals.items()))

    def test_existing_index_defers_native_page_work_until_gallery_visit(self):
        self.check_native_first_visit(indexed=True)

    def test_slow_cached_open_cannot_overwrite_a_newer_search_page(self):
        entered, release = threading.Event(), threading.Event()
        load = self.browser.load_images

        def held(page, term="", **kwargs):
            if not term:
                entered.set()
                self.assertTrue(release.wait(5))
            return load(page, term, **kwargs)

        self.browser.load_images = held

        async def journey():
            old = asyncio.create_task(asyncio.to_thread(self.open_view, 1, "", 1, self.request))
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            try:
                newer = self.functions["request_search"].fn(1, "needle", 2, self.request)
                result = await self.functions["prepare_view"].fn(newer, self.request)
                records = next(
                    control
                    for control in result
                    if isinstance(control, gr.State) and isinstance(result[control], list)
                )
                self.assertEqual(result[records][0]["_view_version"], 2)
            finally:
                release.set()
            self.assertEqual(await old, gr.skip())

        asyncio.run(journey())

    def test_cold_failed_refresh_keeps_the_original_folder_error(self):
        browser, _, _ = self.native_browser()
        self.assertFalse(hasattr(browser, "last_scan"))
        browser.path_manager.paths["path_outputs"] = self.root / "unavailable-output-folder"
        self.scope["browser"] = browser
        descriptor = self.functions["request_refresh"].fn(1, "", 1, self.request)
        result = asyncio.run(self.functions["prepare_view"].fn(descriptor, self.request))
        messages = [value.get("value") for value in result.values() if isinstance(value, dict)]
        message = next(
            (
                text
                for text in messages
                if isinstance(text, str) and "Error updating folder" in text
            ),
            "",
        )
        self.assertIn("Library folder is unavailable", message)
        self.assertNotIn("last_scan", message)

    def test_first_gallery_visit_initializes_native_index_then_pages_and_searches(self):
        self.check_native_first_visit(indexed=False)

    def test_actual_gradio_event_configuration_has_independent_ack_cancel_and_bounded_work(self):
        for name in (
            "request_page",
            "request_refresh",
            "request_search",
            "acknowledge_selection",
            "cancel_refresh",
        ):
            self.assertFalse(self.functions[name].queue)
            self.assertEqual(self.functions[name].trigger_mode, "multiple")
        for name in ("prepare_view", "prepare_selection"):
            function = self.functions[name]
            self.assertTrue(function.queue)
            self.assertEqual(
                (function.concurrency_limit, function.trigger_mode), (2, "always_last")
            )
            self.assertEqual(len(function.show_progress_on), 1)

    def test_request_fallback_uses_the_same_authenticated_login_prefix_as_middleware(self):
        from starlette.requests import Request

        server = SimpleNamespace(cookie_id="fixture", tokens={"fixture-login-A": "owner"})
        request = SimpleNamespace(
            username="owner",
            session_hash="browser tab",
            request=Request(
                {
                    "type": "http",
                    "headers": [(b"cookie", b"access-token-fixture=fixture-login-A")],
                    "app": server,
                }
            ),
        )
        self.functions["request_page"].fn(1, "", 1, request)
        expected = (
            "ui-login",
            "owner",
            hashlib.sha256(b"fixture-login-A").hexdigest(),
            "browser tab",
        )
        self.assertEqual(self.fence.get(expected, "library-view", 1).owner, expected)

    def test_slow_callback_is_replaced_while_another_session_still_browses(self):
        entered, release = threading.Event(), threading.Event()
        original = self.browser.load_images

        def load(page, term="", **kwargs):
            if term == "slow":
                entered.set()
                self.assertTrue(release.wait(5))
            return original(page, term, **kwargs)

        self.browser.load_images = load
        ack = self.functions["request_page"].fn
        prepare = self.functions["prepare_view"].fn

        async def race():
            slow = ack(1, "slow", 1, self.request)
            old = asyncio.create_task(prepare(slow, self.request))
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            try:
                other = SimpleNamespace(username="owner", session_hash="second")
                other_result = await prepare(ack(1, "other session", 1, other), other)
                self.assertTrue(other_result)
                latest = ack(1, "latest", 2, self.request)
            finally:
                release.set()
            self.assertEqual(await old, gr.skip())
            result = await prepare(latest, self.request)
            records = next(
                value
                for key, value in result.items()
                if isinstance(key, gr.State) and isinstance(value, list)
            )
            self.assertEqual(records[0]["_view_version"], 2)

        asyncio.run(race())

    def test_late_page_cannot_activate_selection_after_newer_acknowledgement(self):
        ack = self.functions["request_page"].fn
        first = ack(1, "first", 1, self.request)
        result = asyncio.run(self.functions["prepare_view"].fn(first, self.request))
        records = next(
            value
            for key, value in result.items()
            if isinstance(key, gr.State) and isinstance(value, list)
        )
        # This models a fully prepared response being delayed in transport.
        newer = ack(1, "new input", 2, self.request)
        event = SimpleNamespace(index=0, value={"caption": self.record["caption"]}, selected=True)
        selected_request = self.functions["acknowledge_selection"].fn(
            records, newer, 3, event, self.request
        )
        selected = asyncio.run(
            self.functions["prepare_selection"].fn(records, newer, selected_request, self.request)
        )
        self.assertEqual(selected, gr.skip())

    def test_canceled_and_stale_callbacks_serialize_all_real_outputs_without_mutating_state(self):
        ack = self.functions["request_page"].fn
        view = self.functions["prepare_view"]
        selection = self.functions["prepare_selection"]

        async def check(function, prediction):
            self.assertEqual(prediction, gr.skip())
            state = SessionState(self.app)
            previous = {}
            for output in function.outputs:
                if isinstance(output, gr.State):
                    previous[output._id] = {"preserved": output._id}
                    state[output._id] = previous[output._id].copy()
            processed = await self.app.postprocess_data(function, prediction, state)
            self.assertGreater(len(function.outputs), 1)
            self.assertEqual(len(processed), len(function.outputs))
            self.assertTrue(all(value is None or value == gr.skip() for value in processed))
            for identity, value in previous.items():
                self.assertEqual(state[identity], value)

        async def cases():
            await check(view, await view.fn(None, self.request))
            old = ack(1, "first", 1, self.request)
            latest = ack(1, "newer", 2, self.request)
            await check(view, await view.fn(old, self.request))
            await check(selection, await selection.fn([], None, None, self.request))
            await check(
                selection, await selection.fn([], latest, {"view_version": 1}, self.request)
            )
            # A selection worker can also complete after its page was superseded.
            record = self.record | {"_view_version": 2}
            event = SimpleNamespace(index=0, value={"caption": record["caption"]}, selected=True)
            selected = self.functions["acknowledge_selection"].fn(
                [record], latest, 3, event, self.request
            )
            ack(1, "superseded", 4, self.request)
            await check(selection, await selection.fn([record], latest, selected, self.request))

        asyncio.run(cases())

    def test_cancel_acknowledges_browse_against_previous_index(self):
        descriptor = self.functions["request_refresh"].fn(1, "keep filter", 1, self.request)
        restored, message = self.functions["cancel_refresh"].fn(1, "keep filter", 2, self.request)
        self.assertEqual(
            (restored["version"], restored["term"], restored["refresh"]), (2, "keep filter", False)
        )
        self.assertIn("previous index", message)
        self.assertEqual(
            asyncio.run(self.functions["prepare_view"].fn(descriptor, self.request)), gr.skip()
        )
        self.assertTrue(asyncio.run(self.functions["prepare_view"].fn(restored, self.request)))

    def test_new_selection_fences_a_slow_metadata_result_on_the_same_page(self):
        view = self.functions["request_page"].fn(1, "", 1, self.request)
        asyncio.run(self.functions["prepare_view"].fn(view, self.request))
        second_path = self.root / "second.png"
        second_path.write_bytes(self.source.read_bytes())
        records = [
            record | {"_view_version": 1} for record in (self.record, source_record(second_path))
        ]
        entered, release = threading.Event(), threading.Event()

        def metadata(path):
            if Path(path).resolve() == self.source.resolve():
                entered.set()
                self.assertTrue(release.wait(5))
            return {"parameters": json.dumps({"prompt": Path(path).name})}

        self.scope["get_png_metadata"] = metadata
        acknowledge = self.functions["acknowledge_selection"].fn
        prepare = self.functions["prepare_selection"].fn

        def event(index):
            return SimpleNamespace(
                index=index, value={"caption": records[index]["caption"]}, selected=True
            )

        async def race():
            old_request = acknowledge(records, view, 2, event(0), self.request)
            old = asyncio.create_task(prepare(records, view, old_request, self.request))
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            try:
                current = acknowledge(records, view, 3, event(1), self.request)
                result = await prepare(records, view, current, self.request)
            finally:
                release.set()
            self.assertEqual(await old, gr.skip())
            selected = next(
                value
                for key, value in result.items()
                if isinstance(key, gr.State) and isinstance(value, dict) and "path" in value
            )
            self.assertEqual(Path(selected["path"]), second_path.resolve())
            self.assertEqual(selected["_selection_version"], 3)

        asyncio.run(race())

    def test_stale_page_click_cannot_cancel_the_current_selection(self):
        acknowledge = self.functions["acknowledge_selection"].fn
        prepare = self.functions["prepare_selection"].fn
        self.functions["request_page"].fn(1, "old", 1, self.request)
        view = self.functions["request_page"].fn(1, "current", 2, self.request)
        current_records = [self.record | {"_view_version": 2}]
        stale_records = [self.record | {"_view_version": 1}]
        entered, release = threading.Event(), threading.Event()

        def metadata(path):
            entered.set()
            self.assertTrue(release.wait(5))
            return {"prompt": "current selection"}

        self.scope["get_png_metadata"] = metadata

        async def race():
            selected = acknowledge(current_records, view, 3, self.selection_event(), self.request)
            current = asyncio.create_task(prepare(current_records, view, selected, self.request))
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            try:
                self.assertEqual(
                    acknowledge(stale_records, view, 4, self.selection_event(), self.request),
                    gr.skip(),
                )
            finally:
                release.set()
            result = await current
            self.assertEqual(result[self.selected]["_selection_version"], 3)
            self.assertEqual(result[self.gallery]["elem_classes"], [])

        asyncio.run(race())

    def test_view_replaced_during_selection_ack_cannot_claim_newer_selection_ownership(self):
        acknowledge = self.functions["acknowledge_selection"].fn
        view = self.functions["request_page"].fn(1, "old", 1, self.request)
        original_get = self.fence.get
        replaced = []

        def replace_after_validation(owner, channel, version, **kwargs):
            ticket = original_get(owner, channel, version, **kwargs)
            if channel == "library-view" and version == 1 and not replaced:
                replaced.append(True)
                current = self.functions["request_page"].fn(1, "current", 2, self.request)
                self.assertEqual(
                    acknowledge(
                        [self.record | {"_view_version": 2}],
                        current,
                        3,
                        self.selection_event(),
                        self.request,
                    )["version"],
                    3,
                )
            return ticket

        with patch.object(self.fence, "get", side_effect=replace_after_validation):
            self.assertEqual(
                acknowledge(
                    [self.record | {"_view_version": 1}],
                    view,
                    4,
                    self.selection_event(),
                    self.request,
                ),
                gr.skip(),
            )
        current = original_get(("ui", "owner", "local", "first"), "library-selection", 3)
        self.assertTrue(self.fence.current(current))

    def test_current_selection_error_unlocks_and_clears_actions_and_serializes_outputs(self):
        view = self.functions["request_page"].fn(1, "", 1, self.request)
        records = [self.record | {"_view_version": 1}]
        selected = self.functions["acknowledge_selection"].fn(
            records, view, 2, self.selection_event(), self.request
        )

        def fail(path):
            raise RuntimeError("fixture metadata failure")

        self.scope["get_png_metadata"] = fail

        async def check():
            callback = self.functions["prepare_selection"]
            result = await callback.fn(records, view, selected, self.request)
            self.assertEqual(result[self.gallery]["elem_classes"], [])
            self.assertIsNone(result[self.selected])
            self.assertEqual(result[self.metadata_output], "fixture metadata failure")
            actions = [
                value
                for key, value in result.items()
                if isinstance(key, gr.Button) and key.interactive is False
            ]
            self.assertTrue(actions)
            self.assertTrue(all(value["interactive"] is False for value in actions))
            serialized = await self.app.postprocess_data(callback, result, SessionState(self.app))
            self.assertEqual(len(serialized), len(callback.outputs))

        asyncio.run(check())

    def test_failed_page_unlocks_an_empty_view_then_allows_browsing_again(self):
        load = self.browser.load_images
        view = self.functions["request_page"].fn(1, "broken", 1, self.request)

        def fail(*args, **kwargs):
            raise OSError("fixture page failure")

        self.browser.load_images = fail
        result = asyncio.run(self.functions["prepare_view"].fn(view, self.request))
        self.assertEqual(result[self.gallery].value, [])
        self.assertEqual(result[self.gallery].elem_classes, [])
        self.assertIsNone(result[self.selected])
        self.assertTrue(
            any(value == [] for key, value in result.items() if isinstance(key, gr.State))
        )
        self.browser.load_images = load
        next_view = self.functions["request_page"].fn(1, "working", 2, self.request)
        restored = asyncio.run(self.functions["prepare_view"].fn(next_view, self.request))
        records = next(
            value
            for key, value in restored.items()
            if isinstance(key, gr.State) and isinstance(value, list)
        )
        self.assertEqual(records[0]["_view_version"], 2)

    def test_superseded_metadata_error_cannot_clear_the_latest_selection(self):
        view = self.functions["request_page"].fn(1, "", 1, self.request)
        second = self.root / "second.png"
        second.write_bytes(self.source.read_bytes())
        records = [record | {"_view_version": 1} for record in (self.record, source_record(second))]
        entered, release = threading.Event(), threading.Event()

        def metadata(path):
            if Path(path).resolve() == self.source.resolve():
                entered.set()
                self.assertTrue(release.wait(5))
                raise RuntimeError("superseded failure")
            return {"prompt": "latest selection"}

        self.scope["get_png_metadata"] = metadata
        acknowledge = self.functions["acknowledge_selection"].fn
        prepare = self.functions["prepare_selection"].fn

        async def race():
            stale_request = acknowledge(records, view, 2, self.selection_event(), self.request)
            stale = asyncio.create_task(prepare(records, view, stale_request, self.request))
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            try:
                event = self.selection_event(records[1])
                event.index = 1
                selected = acknowledge(records, view, 3, event, self.request)
                current = await prepare(records, view, selected, self.request)
            finally:
                release.set()
            self.assertEqual(await stale, gr.skip())
            self.assertEqual(Path(current[self.selected]["path"]), second.resolve())
            self.assertEqual(current[self.selected]["_selection_version"], 3)

        asyncio.run(race())


if __name__ == "__main__":
    unittest.main()
