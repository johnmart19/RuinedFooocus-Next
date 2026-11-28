"""Bounded model-free route measurements; browser/transfer time is deliberately absent."""

import json
import ast
import asyncio
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import platform
from time import perf_counter
import unittest
from unittest.mock import patch

import gradio as gr
from PIL import Image
from PIL.PngImagePlugin import PngImageFile

from ruinedfooocus.features.gallery import editing_workflow
from ruinedfooocus.features.gallery.actions import source_record
import test_editor_workspace_events as editor_fixtures
import test_gallery_page_work as page_fixtures
import test_gallery_scheduling as selection_fixtures
import test_library_editing_ui as local_fixtures


class EditorGalleryMeasurements(unittest.TestCase):
    def fixture(self, kind):
        fixture = kind()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_bounded_routes_retain_originals_and_record_work(self):
        root = Path(__file__).resolve().parents[1]
        source = Path(os.environ.get("RF_EDITOR_GALLERY_SOURCE", root))
        workspace_scope = None
        if source != root:
            workspace_scope = {"__name__": "bounded_baseline_workspace"}
            path = source / "ruinedfooocus/features/gallery/ui_workspace.py"
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), workspace_scope)
            base = workspace_scope["EditorWorkspace"]

            class BaselineWorkspace(base):
                def __init__(self, **options):
                    super().__init__(**options)
                    self.latest_generated = gr.State(None)

            self.enterContext(patch.object(editor_fixtures, "EditorWorkspace", BaselineWorkspace))
            self.enterContext(patch.object(page_fixtures, "ROOT", source))
            self.enterContext(patch.object(selection_fixtures, "ROOT", source))
        receipt = {
            "definition": "Each route owns a fresh fixture. Cold is first target-page/import invocation; next/previous prime only the opposite page. Warm is two repeats. OS file cache is not flushed.",
            "limits": "Native Python/Gradio callbacks only; no server, queue wait, network, browser, models or GPU.",
            "dependencies": {
                "python": platform.python_version(),
                "gradio": version("gradio"),
                "pillow": version("pillow"),
            },
            "source_sha256": {
                name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in (
                    "ruinedfooocus/features/gallery/ui_workspace.py",
                    "ruinedfooocus/features/gallery/imagebrowser.py",
                    "ruinedfooocus/features/gallery/ui.py",
                )
            },
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "support_sha256": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in (
                    "ruinedfooocus/features/gallery/actions.py",
                    "ruinedfooocus/features/gallery/editor_session.py",
                    "ruinedfooocus/features/gallery/editing_workflow.py",
                    "ruinedfooocus/features/gallery/ui_editing.py",
                    "ruinedfooocus/features/gallery/index_store.py",
                    "ruinedfooocus/features/image_processing/native_image_outputs.py",
                )
            },
            "fixture_sha256": {},
            "routes": {},
        }
        for route, number in (("first_page", 1), ("next_page", 2), ("previous_page", 1)):
            page = self.fixture(page_fixtures.GalleryPageWorkTests)
            page.browser.images_per_page = 3
            if route != "first_page":
                page.load(1 if route == "next_page" else 2)
            samples = []
            for sample in range(3):
                queries = []
                page.database.set_trace_callback(queries.append)
                start = perf_counter()
                images, _, records = page.load(number)
                samples.append(
                    {
                        "cache": "cold" if sample == 0 else "warm",
                        "server_ms": (perf_counter() - start) * 1000,
                        "records": len(records),
                        "images": len(images),
                        "index_selects": sum(query.startswith("SELECT") for query in queries),
                        "server_callbacks": 2,
                        "client_callbacks": 1,
                    }
                )
            receipt["routes"][route] = samples
            receipt["fixture_sha256"][route] = [
                hashlib.sha256(contents).hexdigest() for contents in page.originals.values()
            ]
            for original, contents in page.originals.items():
                self.assertEqual(original.read_bytes(), contents)
            page.doCleanups()
        for route in ("open_current", "open_gallery"):
            fixture = self.fixture(editor_fixtures.EditorWorkspaceEventTests)
            Image.new("RGBA", (1536, 1024), (20, 40, 60, 127)).save(fixture.legacy.path)
            fixture.legacy.record = source_record(fixture.legacy.path)
            receipt["fixture_sha256"][route] = hashlib.sha256(
                fixture.legacy.path.read_bytes()
            ).hexdigest()
            session = fixture.session("measure-" + route)
            function = fixture.event(
                fixture.workspace.open_current if route == "open_current" else fixture.library_open
            )
            samples = []
            for sample in range(3):
                preparation = []

                def measured(record):
                    start = perf_counter()
                    result = editing_workflow.working_source(record)
                    preparation.append((perf_counter() - start) * 1000)
                    return result

                context = (
                    patch.dict(workspace_scope, {"working_source": measured})
                    if workspace_scope is not None
                    else patch(
                        "ruinedfooocus.features.gallery.ui_workspace.working_source",
                        side_effect=measured,
                    )
                )
                with context:
                    start = perf_counter()
                    if route == "open_current":
                        value = fixture.plain(
                            fixture.remote["preview"].postprocess(str(fixture.legacy.path))
                        )
                        fixture.call(session, function, {fixture.remote["preview"]: value})
                    else:
                        fixture.open_library(session)
                    samples.append(
                        {
                            "cache": "cold" if sample == 0 else "warm",
                            "server_ms": (perf_counter() - start) * 1000,
                            "working_source_calls": len(preparation),
                            "working_source_ms": sum(preparation),
                            "server_callbacks": 1,
                            "client_callbacks": int(route == "open_gallery"),
                            "outputs": len(function.outputs),
                        }
                    )
            receipt["routes"][route] = samples
            fixture.doCleanups()
        fixture = self.fixture(selection_fixtures.GallerySchedulingTests)
        tree = ast.parse(
            (source / "ruinedfooocus/features/gallery/imagebrowser.py").read_text(encoding="utf-8")
        )
        node = next(node for node in tree.body if getattr(node, "name", "") == "get_png_metadata")
        namespace = dict(Image=Image, PngImageFile=PngImageFile, Dict=dict)
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]), "actual PNG selection decoder", "exec"
            ),
            namespace,
        )
        fixture.scope["get_png_metadata"] = namespace["get_png_metadata"]
        local = self.fixture(local_fixtures.LibraryEditingUiTests) if source == root else None
        samples = []
        for sample in range(3):
            descriptor = fixture.functions["request_page"].fn(1, "", sample + 1, fixture.request)
            prepared = asyncio.run(
                fixture.functions["prepare_view"].fn(descriptor, fixture.request)
            )
            records = next(
                value
                for control, value in prepared.items()
                if isinstance(control, gr.State) and isinstance(value, list)
            )
            start = perf_counter()
            selected = fixture.functions["acknowledge_selection"].fn(
                records, descriptor, sample + 100, fixture.selection_event(), fixture.request
            )
            result = asyncio.run(
                fixture.functions["prepare_selection"].fn(
                    records, descriptor, selected, fixture.request
                )
            )
            local_ms, local_preparations = 0, 0
            if local is not None:
                selected_record = result[fixture.selected]
                before_local = perf_counter()
                with patch(
                    "ruinedfooocus.features.gallery.ui_editing.working_source",
                    wraps=editing_workflow.working_source,
                ) as prepared_local:
                    asyncio.run(
                        local.functions["load_selection"].fn(
                            selected_record,
                            {},
                            None if sample == 0 else selected_record,
                            "Rendered prompt",
                            "model",
                            "edit",
                            "draft",
                            "negative",
                            "model",
                            512,
                            768,
                            {},
                            gr.Request(session_hash="measured-local"),
                        )
                    )
                    local_preparations = prepared_local.call_count
                local_ms = (perf_counter() - before_local) * 1000
            samples.append(
                {
                    "cache": "cold" if sample == 0 else "warm",
                    "server_ms": (perf_counter() - start) * 1000,
                    "server_callbacks": 2 + int(local is not None),
                    "client_callbacks": 1 + int(local is not None),
                    "local_draft_ms": local_ms,
                    "local_working_source_calls": local_preparations,
                    "outputs": len(fixture.functions["prepare_selection"].outputs),
                }
            )
            self.assertIsNotNone(result[fixture.selected])
        receipt["routes"]["selection"] = samples
        receipt["fixture_sha256"]["selection"] = hashlib.sha256(
            fixture.source.read_bytes()
        ).hexdigest()
        target = os.environ.get("RF_EDITOR_GALLERY_RECEIPT")
        if target:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            Path(target).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt))


if __name__ == "__main__":
    unittest.main()
