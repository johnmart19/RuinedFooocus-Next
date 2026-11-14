"""Canonical Create editor journeys through installed native Gradio events."""

import asyncio
import ast
from concurrent.futures import CancelledError, ThreadPoolExecutor
import copy
import importlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import subprocess
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState
from PIL import Image

from ruinedfooocus.app import resource_lifecycle as life
from ruinedfooocus.features.gallery import editor_session
from ruinedfooocus.features.gallery.actions import source_record
from ruinedfooocus.features.gallery.ui_workspace import EditorWorkspace
from ruinedfooocus.features.image_editing import image_workflows
from ruinedfooocus.features.image_processing import image_composition
from ruinedfooocus.features.image_processing.native_image_outputs import result_presentation
import test_gallery_actions as gallery_fixtures
import test_library_editing_ui as library_fixtures
import test_gallery_setup_outputs as setup_fixtures
from ui_test_helpers import function_from


class EditorWorkspaceEventTests(unittest.TestCase):
    def setUp(self):
        cache = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(patch.dict(os.environ, {"GRADIO_TEMP_DIR": str(cache)}))
        self.legacy = library_fixtures.LibraryEditingUiTests()
        self.legacy.setUp()
        self.addCleanup(self.legacy.doCleanups)
        self.root, self.app = self.legacy.root, self.legacy.app
        self.store = editor_session.EditorSessionStore(self.root / "editor-sessions")
        self.shared = self.legacy.shared
        self.shared.models = gallery_fixtures.Models(self.root / "models")
        self.controls = self.shared.get_ctrl.__self__
        self.remote = self.legacy.remote
        self.enterContext(patch.dict("sys.modules", {"ruinedfooocus.app.state": self.shared}))
        app_module = importlib.import_module("ruinedfooocus.app")
        self.enterContext(patch.object(app_module, "state", self.shared, create=True))
        self.enterContext(patch.object(editor_session, "default_store", return_value=self.store))
        self.fence = life.LatestUIRequests()
        self.enterContext(
            patch("ruinedfooocus.features.gallery.ui_workspace.latest_ui_requests", self.fence)
        )
        self.jobs, self.requests = life.JobRegistry(), []
        self.worker_mode = "completed"
        self.fallback_ready = False
        scope = dict(
            registry=self.jobs,
            current_job=life.current_job,
            json=json,
            check_interrupt=lambda data: (data.get("_job") or life.current_job()).check(),
        )
        for name in (
            "validate_editor_data",
            "_source_has_opaque_alpha",
            "stage_editor_result",
            "publish_editor_result",
            "discard_editor_result",
            "editor_result",
        ):
            function_from("ruinedfooocus/app/async_worker.py", name, scope)
        self.worker_scope = scope
        worker = SimpleNamespace(editor_result=scope["editor_result"])
        self.enterContext(patch.dict("sys.modules", {"ruinedfooocus.app.async_worker": worker}))
        self.enterContext(patch.object(app_module, "async_worker", worker, create=True))
        self.run_request = Mock(side_effect=self.run_fixture_worker)
        self.stop_request = Mock(side_effect=self.cancel_owned)
        with self.app:
            with gr.Tabs() as self.destinations:
                gr.Tab("Create", id="create")
                gr.Tab("Editor", id="editor")
                gr.Tab("Gallery", id="library")
            self.controls.update(
                prompt=gr.Textbox(value="Create draft"),
                negative=gr.Textbox(value="draft exclusions"),
                base_model_name=gr.Dropdown(
                    choices=["model", "new/model.safetensors"], value="model"
                ),
                custom_width=gr.Number(value=512, precision=0),
                custom_height=gr.Number(value=768, precision=0),
                aspect_ratios_selection=gr.Dropdown(choices=["Custom..."], value="Custom..."),
                image_source=gr.State(None),
                input_image=gr.Image(type="pil", format="png", image_mode="RGBA"),
                reference_images=gr.Gallery(value=[]),
                output_format=gr.Dropdown(choices=["png", "exr", "avif"], value="png"),
                output_bit_depth=gr.Dropdown(choices=["8-bit", "16-bit"], value="8-bit"),
                output_color_space=gr.Dropdown(choices=["sRGB", "Linear"], value="sRGB"),
                output_crf=gr.Slider(minimum=1, maximum=63, value=18),
                _adjustment_hdr_precision=gr.Dropdown(
                    choices=["16-bit float", "32-bit float"], value=None
                ),
                resolution_auto=gr.Checkbox(value=True),
                cn_selection=gr.Dropdown(
                    choices=[
                        "None",
                        "Face landmarks",
                        "Whole-body pose",
                        "Depth map",
                        "Adjust image",
                    ],
                    value="None",
                ),
                cn_type=gr.Dropdown(choices=["None"], value="None"),
            )
            self.remote.update(
                preview=gr.Image(type="filepath", format="png", image_mode=None),
                inpaint_toggle=gr.Checkbox(value=False),
                inpaint=gr.ImageEditor(type="pil"),
                hint=gr.Markdown(),
                metadata=gr.JSON({}),
                image_source=self.controls["image_source"],
            )
            self.workspace = EditorWorkspace(store=self.store)
            self.workspace.destination.value = "editor"
            self.controls["_editor_destination"] = self.workspace.destination
            self.controls["_editor_source_input"] = self.workspace.source_input
            self.controls["_editor_denoise"] = self.workspace.denoise
            self.controls["_editor_output_explicit"] = self.workspace.output_explicit
            self.controls["_native_main_source"] = self.remote["native_source"]
            self.workspace.bind(
                self.controls,
                run_request=self.run_request,
                stop_request=self.stop_request,
                run_controls=self.remote,
                navigation={
                    self.destinations: gr.update(selected="editor"),
                    self.workspace.destination: "editor",
                },
            )
            self.library_open = gr.Button("Open/Edit in Editor")
            self.selected = gr.State(None)
            self.metadata = gr.JSON({})
            self.prompt_version = gr.Radio(
                choices=["Rendered prompt", "Original prompt"], value="Rendered prompt"
            )
            self.workspace.bind_library(
                self.library_open,
                self.selected,
                self.metadata,
                self.prompt_version,
                validate_record=self.legacy.validate,
            )
            self.values = gr.State({})
            self.remote["run"].click(
                self.workspace.run,
                inputs=[self.values, self.workspace.version],
                outputs=self.production_run_outputs(),
                js=self.workspace.version_js,
            )
            self.remote["stop"].click(self.cancel_owned, inputs=self.remote["task"], queue=False)
            gallery_change = function_from(
                "ruinedfooocus/ui/app.py",
                "gallery_change",
                dict(
                    gr=gr,
                    Image=Image,
                    main_view=self.remote["preview"],
                    metadata_json=self.remote["metadata"],
                    native_main_source=self.remote["native_source"],
                    editor_workspace=self.workspace,
                ),
            )
            self.remote["gallery"].select(
                gallery_change,
                inputs=[self.remote["gallery"], self.remote["artifacts"]],
                outputs=[
                    self.remote["preview"],
                    self.remote["metadata"],
                    self.remote["native_source"],
                    self.workspace.latest_generated,
                ],
            )
            update_results = function_from(
                "ruinedfooocus/ui/app.py",
                "update_results",
                dict(
                    gr=gr,
                    shared=self.shared,
                    run_button=self.remote["run"],
                    stop_button=self.remote["stop"],
                    progress_html=self.remote["progress"],
                    main_view=self.remote["preview"],
                    inpaint_toggle=self.remote["inpaint_toggle"],
                    gallery=self.remote["gallery"],
                    metadata_json=self.remote["metadata"],
                    generation_originals=self.remote["originals"],
                    generation_artifacts=self.remote["artifacts"],
                    native_main_source=self.remote["native_source"],
                    editor_workspace=self.workspace,
                ),
            )
            self.products, self.present_results = (
                gr.State([]),
                gr.Button("Fixture terminal results"),
            )
            self.present_results.click(
                update_results,
                inputs=self.products,
                outputs=list(
                    dict.fromkeys(
                        [
                            *self.remote.values(),
                            self.workspace.latest_generated,
                            self.workspace.source_input,
                        ]
                    )
                ),
            )
        self.saved = dict(
            prompt="Rendered café prompt",
            negative="rendered exclusions",
            raw_prompt="Original unprocessed prompt",
            raw_negative="original exclusions",
            base_model_name="new/model.safetensors",
            base_model_hash="a" * 64,
            width=113,
            height=75,
        )

    def production_run_outputs(self):
        """Use the actual app declaration so native journeys cover its wiring."""
        path = Path(__file__).resolve().parents[1] / "ruinedfooocus/ui/app.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and any(
                keyword.arg == "fn"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "generate_clicked"
                for keyword in node.keywords
            )
        )
        expression = next(item.value for item in call.keywords if item.arg == "outputs")
        mapping = dict(
            run_button="run",
            stop_button="stop",
            progress_html="progress",
            main_view="preview",
            inpaint_view="inpaint",
            inpaint_toggle="inpaint_toggle",
            gallery="gallery",
            metadata_json="metadata",
            hint_text="hint",
            generation_task="task",
            generation_originals="originals",
            generation_artifacts="artifacts",
            native_main_source="native_source",
        )
        namespace = {name: self.remote[key] for name, key in mapping.items()}
        namespace.update(shared=self.shared, editor_workspace=self.workspace)
        return eval(compile(ast.Expression(expression), str(path), "eval"), namespace)

    def session(self, name="workspace-A"):
        return SimpleNamespace(
            state=SessionState(self.app),
            values={},
            request=gr.Request(session_hash=name),
            owner=life.ui_request_owner(gr.Request(session_hash=name)),
            browser={},
        )

    def event(self, button, name="click", callback=None):
        dependency = next(
            item
            for item in self.app.config["dependencies"]
            if any(target == (button._id, name) for target in item["targets"])
            and (
                callback is None or getattr(self.app.fns[item["id"]].fn, "__name__", "") == callback
            )
        )
        return self.app.fns[dependency["id"]]

    @staticmethod
    def plain(value):
        return value.model_dump() if hasattr(value, "model_dump") else value

    def browser_script(self, session, function, script, arguments):
        """Reuse the existing draft-fence Node DOM fixture with actual event scripts."""
        nodes = [
            {
                "id": control.elem_id,
                "value": self.value(session, control),
                "checkbox": isinstance(control, gr.Checkbox),
            }
            for control in function.inputs
            if getattr(control, "elem_id", None)
            and not control.stateful
            and not isinstance(control, gr.JSON)
        ]
        harness = r"""
const vm=require('node:vm'), data=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const memory=data.memory||{}, nodes={};
for(const entry of data.nodes){
  const previous=memory[entry.id]||{};
  const input={value:String(entry.value??''),checked:entry.checkbox?Boolean(entry.value):false,
    listeners:{},addEventListener(name,fn){this.listeners[name]=fn;}};
  if(previous.draft) input.rfDraft=previous.draft;
  nodes[entry.id]={input,listeners:{},querySelector(){return input},querySelectorAll(){return [input]},
    addEventListener(name,fn){this.listeners[name]=fn;}};
  if(previous.context) nodes[entry.id].rfPromptContext=previous.context;
}
const scope={window:memory.window||{},document:{getElementById(id){return nodes[id]||null}}};
const output=vm.runInNewContext('('+data.script+')',scope)(...data.arguments);
for(const [id,node] of Object.entries(nodes)) memory[id]={draft:node.input.rfDraft,context:node.rfPromptContext};
memory.window=scope.window;
process.stdout.write(JSON.stringify({output,memory}));
"""
        result = subprocess.run(
            ["node", "-e", harness],
            input=json.dumps(
                dict(nodes=nodes, memory=session.browser, script=script, arguments=arguments)
            ),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        session.browser = value["memory"]
        return value["output"]

    def apply(self, session, outputs, values):
        projected = {}
        for control, raw in zip(outputs, values):
            value = self.plain(raw)
            projected[control] = value
            if control.stateful:
                continue
            if isinstance(value, dict) and value.get("__type__") == "update":
                if "value" in value:
                    session.values[control._id] = value["value"]
            else:
                session.values[control._id] = value
        return projected

    def publish(self, session, function, projected):
        clients = [
            event
            for event in self.app.config["dependencies"]
            if event["trigger_after"] == function._id and "fence.publish" in (event["js"] or "")
        ]
        if not clients:
            return {}
        client = self.app.fns[clients[0]["id"]]
        envelope = projected.get(client.inputs[0])
        if envelope is None or envelope == gr.skip():
            return {}
        updates = self.browser_script(session, function, clients[0]["js"], [envelope])
        return self.apply(session, client.outputs, updates)

    async def api(
        self, session, function, changes=None, *, iterator=None, publish=True, event_data=None
    ):
        for control, value in (changes or {}).items():
            if control.stateful:
                session.state[control._id] = value
            else:
                session.values[control._id] = value
        arguments = [
            None if control.stateful else session.values.get(control._id, control.value)
            for control in function.inputs
        ]
        dependency = next(
            item for item in self.app.config["dependencies"] if item["id"] == function._id
        )
        if iterator is None and dependency["js"]:
            arguments = self.browser_script(session, function, dependency["js"], arguments)
        result = await self.app.process_api(
            function,
            arguments if iterator is None else None,
            state=session.state,
            request=session.request,
            iterator=iterator,
            event_data=event_data,
        )
        if inspect.isgeneratorfunction(function.fn) and not result["is_generating"]:
            return result, {}
        projected = self.apply(session, function.outputs, result["data"])
        if publish:
            self.publish(session, function, projected)
        return result, projected

    def call(self, session, function, changes=None):
        return asyncio.run(self.api(session, function, changes))[1]

    async def finish(self, session, function, result):
        while result["is_generating"]:
            result, _ = await self.api(session, function, iterator=result["iterator"])

    def value(self, session, control):
        return (
            session.state[control._id]
            if control.stateful
            else session.values.get(control._id, control.value)
        )

    def open_library(self, session, version="Rendered prompt"):
        return self.call(
            session,
            self.event(self.library_open),
            {
                self.selected: copy.deepcopy(self.legacy.record),
                self.metadata: copy.deepcopy(self.saved),
                self.prompt_version: version,
            },
        )

    def cancel_owned(self, task_id, request: gr.Request = None):
        self.jobs.cancel(task_id, owner=life.ui_request_owner(request))

    def run_fixture_worker(self, data):
        """A serial CPU native-composition fixture with the real P3 terminal helpers."""
        item = self.jobs.create(owner=life.ui_request_owner())
        request = dict(data, _job=item)
        mode = self.worker_mode
        self.requests.append(request)
        try:
            with life.job_scope(item):
                self.worker_scope["validate_editor_data"](request, start=True)
            initial = self.root / "fixture-init.png"
            if not initial.exists():
                Image.new("RGB", (2, 2), "gray").save(initial)
            yield {
                self.remote["task"]: item.task_id,
                self.remote["preview"]: str(initial),
                self.remote["run"]: gr.update(interactive=False),
                self.remote["stop"]: gr.update(visible=True),
            }
            if mode in ("empty-failed", "cancelled-result"):
                partial = self.root / "partial-preview.png"
                if not partial.exists():
                    Image.new("RGB", (73, 121), "yellow").save(partial)
                yield {self.remote["preview"]: str(partial)}
            with life.job_scope(item):
                if mode == "empty-failed":
                    self.worker_scope["discard_editor_result"](item, status="failed")
                elif mode == "cancelled-result":
                    item.cancelled.set()
                    self.worker_scope["discard_editor_result"](item, status="cancelled")
                elif mode == "fallback-result":
                    output = self.root / "fallback-generation-fixture.png"
                    Image.new("RGBA", (73, 121), (201, 11, 31, 255)).save(output)
                    self.fallback_ready = True
                elif mode == "owned-copy-result":
                    output = self.root / f"owned-copy-{item.task_id}.png"
                    request["input_image"].save(output)
                    self.worker_scope["stage_editor_result"](request, [str(output)])
                    with self.jobs.condition:
                        self.worker_scope["publish_editor_result"](item)
                else:
                    item.check()
                    captured = image_composition.capture(request, "image-composite")
                    image = image_composition.process(captured, item.check)
                    output = self.root / f"native-compose-{item.task_id}.png"
                    image.save(output)
                    self.worker_scope["stage_editor_result"](request, [str(output)])
                    with self.jobs.condition:
                        self.worker_scope["publish_editor_result"](item)
            if mode == "empty-failed":
                yield {
                    self.remote["task"]: None,
                    self.remote["artifacts"]: [],
                    self.remote["preview"]: None,
                    self.controls["input_image"]: None,
                    self.workspace.context: None,
                    self.controls["prompt"]: "old failed draft",
                }
            else:
                result = self.legacy.path if mode == "cancelled-result" else output
                yield {
                    self.remote["task"]: None,
                    self.remote["preview"]: str(result),
                    self.remote["artifacts"]: result_presentation([str(result)])[2],
                }
        except CancelledError:
            with life.job_scope(item):
                self.worker_scope["discard_editor_result"](item, status="cancelled")
            yield {
                self.remote["task"]: None,
                self.remote["artifacts"]: [],
                self.remote["preview"]: None,
            }
        finally:
            with life.job_scope(item):
                if "editor_result" not in item.state:
                    self.worker_scope["discard_editor_result"](item, status="cancelled")
            self.jobs.finish(item)

    def request_values(self, session, action, *, background=None):
        values = {name: self.value(session, control) for name, control in self.controls.items()}
        values.update(
            _editor_context=copy.deepcopy(self.value(session, self.workspace.context)),
            _editor_action=action,
            _editor_framing="preserve",
        )
        values["_native_main_source"] = self.value(session, self.remote["native_source"])
        preview = self.value(session, self.remote["preview"])
        values["main_view"] = preview.get("path") if isinstance(preview, dict) else preview
        if background is not None:
            values["_editor_background"] = str(background)
        return values

    def background(self):
        path = self.root / "background.png"
        if not path.exists():
            Image.new("RGBA", (73, 121), (0, 0, 255, 255)).save(path)
        return path

    async def select_background(self, session):
        function = self.event(self.remote["run"])
        result, _ = await self.api(
            session,
            function,
            {
                self.values: self.request_values(
                    session, "select-background", background=self.background()
                )
            },
        )
        await self.finish(session, function, result)

    def mount_picker(self, session):
        record = copy.deepcopy(self.legacy.record)
        browser = SimpleNamespace(
            update_images=Mock(return_value=None),
            load_images=Mock(
                return_value=(
                    [(record["path"], record["caption"])],
                    "fixture image",
                    [record],
                    {"page": 1, "pages": 1},
                )
            ),
        )
        with self.app:
            self.workspace.bind_picker(browser)
        self.call(session, self.event(self.workspace.choose_gallery))
        prepare = next(
            fn for fn in self.app.fns.values() if getattr(fn.fn, "__name__", "") == "prepare_picker"
        )
        self.call(session, prepare)
        pick = self.event(self.workspace.picker_gallery, "select", "pick")
        dependency = next(
            event for event in self.app.config["dependencies"] if event["trigger_after"] == pick._id
        )
        return browser, pick, self.app.fns[dependency["id"]], dependency

    def picker_event(self, *, caption=None):
        return gr.SelectData(
            self.workspace.picker_gallery,
            {
                "index": 0,
                "value": {"caption": caption or self.legacy.record["caption"]},
                "selected": True,
            },
        )

    def test_latest_without_completed_result_keeps_current_source_and_requires_explicit_gallery(
        self,
    ):
        session = self.session("missing-latest")
        self.open_library(session)
        before = copy.deepcopy(self.value(session, self.workspace.context))
        browser, _, _, _ = self.mount_picker(session)
        browser.load_images.reset_mock()
        for record in (None, dict(self.legacy.record, output_role="guide")):
            updates = self.call(
                session,
                self.event(self.workspace.open_latest),
                {self.workspace.latest_generated: record},
            )
            self.assertEqual(self.value(session, self.workspace.context), before)
            from ruinedfooocus.core.translation_manager import localize

            notice = str(updates[self.workspace.notice])
            if notice.startswith("__i18n__"):
                key = json.loads(notice.removeprefix("__i18n__"))["key"]
                notice = localize(key.removeprefix("ruinedfooocus."), "en")
            self.assertIn("Choose an image from Gallery", notice)
        browser.load_images.assert_not_called()
        self.run_request.assert_not_called()

    def test_latest_in_empty_editor_opens_newest_gallery_image(self):
        session = self.session("empty-latest")
        browser, _, _, _ = self.mount_picker(session)
        browser.load_images.reset_mock()
        self.assertIsNone(self.value(session, self.workspace.context))
        self.call(
            session,
            self.event(self.workspace.open_latest),
            {self.workspace.latest_generated: None},
        )
        browser.load_images.assert_called_once()
        self.assertEqual(browser.load_images.call_args.args[0], 1)
        context = self.value(session, self.workspace.context)
        self.assertIsNotNone(context)
        self.assertEqual(context["revision"], 1)
        self.run_request.assert_not_called()

    def test_view_projections_track_no_progress_outputs_but_source_import_keeps_its_notice(self):
        adjustment = self.event(self.workspace.action, "change", "adjustment_action")
        quick = self.event(self.workspace.quick_tool, "input", "select_quick_tool")
        for function in (adjustment, quick):
            dependency = next(
                event for event in self.app.config["dependencies"] if event["id"] == function._id
            )
            self.assertTrue(dependency["backend_fn"])
            self.assertEqual(dependency["show_progress"], "hidden")
            self.assertEqual(dependency["show_progress_on"], [])
        self.assertEqual(
            adjustment.outputs, [self.controls["cn_selection"], self.controls["cn_type"]]
        )
        source_import = self.event(self.library_open)
        dependency = next(
            event for event in self.app.config["dependencies"] if event["id"] == source_import._id
        )
        self.assertEqual(dependency["show_progress"], "minimal")
        self.assertEqual(dependency["show_progress_on"], [self.workspace.notice._id])
        session = self.session("quiet-view-projection")
        self.open_library(session)
        context = copy.deepcopy(self.value(session, self.workspace.context))
        selected = self.value(session, self.controls["cn_selection"])
        updates = self.call(session, adjustment)
        self.assertTrue(all(update == gr.skip() for update in updates.values()))
        self.assertEqual(self.value(session, self.workspace.context), context)
        self.assertEqual(self.value(session, self.controls["cn_selection"]), selected)
        self.run_request.assert_not_called()

    def test_picker_select_opens_original_through_success_continuation_in_one_action(self):
        session = self.session("one-click-picker")
        original = self.legacy.path.read_bytes()
        _, pick, adopt, dependency = self.mount_picker(session)
        self.assertTrue(dependency["trigger_only_on_success"])
        asyncio.run(self.api(session, pick, event_data=self.picker_event()))
        updates = self.call(session, adopt)
        context = self.value(session, self.workspace.context)
        snapshot = self.store.snapshot(context["session_id"], owner=session.owner)
        self.assertEqual(Path(snapshot.original.path), self.legacy.path)
        self.assertEqual(snapshot.current.frame.size, (73, 121))
        self.assertEqual(self.value(session, self.controls["prompt"]), "Create draft")
        self.assertEqual(self.value(session, self.workspace.action), "prompt-edit")
        self.assertFalse(updates[self.workspace.picker]["visible"])
        self.assertEqual(self.legacy.path.read_bytes(), original)
        self.run_request.assert_not_called()

    def test_failed_or_cancelled_picker_selection_cannot_adopt_previous_choice(self):
        session = self.session("failed-picker")
        self.open_library(session)
        before = copy.deepcopy(self.value(session, self.workspace.context))
        _, pick, adopt, _ = self.mount_picker(session)
        asyncio.run(self.api(session, pick, event_data=self.picker_event()))
        for mode in ("mismatched-caption", "new-page", "cancelled"):
            with self.subTest(mode=mode):
                record = self.value(session, self.workspace.picker_records)[0]
                session.state[self.workspace.picker_selected._id] = record
                if mode == "new-page":
                    self.call(session, self.event(self.workspace.choose_gallery))
                elif mode == "cancelled":
                    self.call(session, self.event(self.workspace.picker_close))
                    session.state[self.workspace.picker_selected._id] = record
                asyncio.run(
                    self.api(
                        session,
                        pick,
                        event_data=self.picker_event(
                            caption="stale caption" if mode == "mismatched-caption" else None
                        ),
                    )
                )
                self.assertIsNone(self.value(session, self.workspace.picker_selected))
                self.call(session, adopt)
                self.assertEqual(self.value(session, self.workspace.context), before)
        self.run_request.assert_not_called()

    def test_all_action_chips_name_run_and_reveal_their_inputs_without_replacing_drafts(self):
        session = self.session("all-action-inputs")
        self.open_library(session)
        context = copy.deepcopy(self.value(session, self.workspace.context))
        draft = self.value(session, self.controls["prompt"])
        strength = 0.46
        session.values[self.workspace.denoise._id] = strength
        callback = self.event(self.workspace.action, "change", "action_context")
        self.assertEqual(self.workspace.action.value, "prompt-edit")
        self.assertTrue(self.workspace.denoise.visible)
        self.assertTrue(self.workspace.action_inputs.open)
        self.assertTrue(self.workspace.action_inputs.visible)
        self.assertFalse(self.workspace.mask_choice.visible)
        actions = [value for _, value in self.workspace.action.choices]
        self.assertEqual(len(actions), 11)
        for action in actions:
            with self.subTest(action=action):
                updates = self.call(session, callback, {self.workspace.action: action})
                self.assertEqual(
                    updates[self.remote["run"]]["value"], str(self.workspace.action_label(action))
                )
                self.assertEqual(updates[self.workspace.mask]["visible"], action == "inpaint")
                self.assertEqual(
                    updates[self.workspace.background]["visible"],
                    action in ("select-background", "compose"),
                )
                self.assertEqual(
                    updates[self.workspace.denoise]["visible"], action == "prompt-edit"
                )
                if action in ("inpaint", "compose", "select-background"):
                    self.assertTrue(updates[self.workspace.action_inputs]["visible"])
                    self.assertTrue(updates[self.workspace.action_inputs]["open"])
                self.assertEqual(self.value(session, self.workspace.context), context)
                self.assertEqual(self.value(session, self.controls["prompt"]), draft)
                self.assertEqual(self.value(session, self.workspace.denoise), strength)
        self.run_request.assert_not_called()

    def test_upload_reads_metadata_without_automatic_caption_inference_and_keeps_draft(self):
        from ruinedfooocus.features.image_processing.interrogate import metadata_prompt
        from PIL.PngImagePlugin import PngInfo

        forbidden = Mock(side_effect=AssertionError("Upload must not run a caption model"))
        handler = function_from(
            "ruinedfooocus/ui/app.py",
            "load_images_handler",
            {
                "Image": Image,
                "metadata_prompt": metadata_prompt,
                "gr": gr,
                "look": forbidden,
            },
        )
        precision = self.root / "upload-no-metadata.tiff"
        Image.new("I;16", (17, 23), 12345).save(precision)
        self.assertEqual(handler(str(precision), "", "create")[0], "")
        self.assertEqual(handler(str(precision), "Make her more detailed", "create")[0], gr.skip())
        saved = self.root / "upload-metadata.png"
        info = PngInfo()
        info.add_text("parameters", '{"prompt":"saved source description"}')
        Image.new("RGB", (17, 23), "red").save(saved, pnginfo=info)
        self.assertEqual(handler(str(saved), "", "create")[0], "saved source description")
        self.assertEqual(handler(str(saved), "Exact user instruction", "create")[0], gr.skip())
        for source in (precision, saved):
            for instruction in ("", "Make her more detailed"):
                with self.subTest(source=source.name, instruction=instruction):
                    self.assertEqual(handler(str(source), instruction, "editor")[0], gr.skip())
        forbidden.assert_not_called()

    def test_adjustment_activation_restores_internal_choices_and_reveals_actual_panel(self):
        from ruinedfooocus.features.image_processing import ui_image_adjustments
        from ruinedfooocus.features.image_processing.still_image_utilities import adjustment_kind

        session = self.session("adjustment-first-use")
        with patch.object(
            self.shared,
            "add_ctrl",
            side_effect=lambda key, value: self.controls.update({key: value}),
            create=True,
        ):
            with self.app:
                ui_image_adjustments.create_ui(
                    self.controls["cn_selection"], self.controls["cn_type"], adjustment_kind
                )
        activate = self.event(self.workspace.action, "change", "adjustment_action")
        updates = self.call(session, activate, {self.workspace.action: "adjust-image"})
        chosen = updates[self.controls["cn_type"]]
        self.assertEqual(chosen["value"], "Image-transform")
        self.assertIn("Image-transform", [value for _, value in chosen["choices"]])
        panel = next(
            fn
            for fn in self.app.fns.values()
            if getattr(fn.fn, "__name__", "") == "adjustment_context"
        )
        shown = self.call(session, panel)
        self.assertTrue(shown[panel.outputs[0]]["visible"])
        self.assertTrue(shown[panel.outputs[2]]["visible"])
        self.assertEqual(self.value(session, self.controls["transform_operation"]), "resize")

    def test_passive_preview_sync_gives_point_tools_current_upload_without_open_or_run(self):
        from gradio.processing_utils import move_files_to_cache
        from ruinedfooocus.features.image_processing import ui_native_vision

        with (
            patch.object(
                self.shared,
                "add_ctrl",
                side_effect=lambda key, value: self.controls.update({key: value}),
                create=True,
            ),
            patch.object(self.shared, "path_manager", object(), create=True),
            patch.object(ui_native_vision, "native_model_choices", return_value=[]),
            patch.object(ui_native_vision, "picker_value", return_value="/segment"),
            patch.object(ui_native_vision, "coco_classes", return_value=[]),
        ):
            with self.app:
                ui_native_vision.create_ui(
                    self.controls["cn_selection"],
                    self.controls["cn_type"],
                    self.controls["input_image"],
                    gr.Number(0),
                    lambda *_: "segment",
                )
        session = self.session("point-upload-source")
        self.open_library(session)
        sync = self.event(self.remote["preview"], "change", "preview_source")
        clear = self.event(self.controls["input_image"], "change", "clear")

        def upload(name, color):
            path = self.root / name
            Image.new("RGBA", (73, 121), color).save(path)
            value = self.plain(
                move_files_to_cache(
                    self.remote["preview"].postprocess(str(path)),
                    self.remote["preview"],
                    postprocess=True,
                )
            )
            session.values[self.remote["preview"]._id] = value
            self.call(session, self.event(self.remote["preview"], "upload", "source_changed"))
            self.call(session, sync)
            self.call(session, clear)
            return path

        first = upload("point-source-B.png", (71, 91, 111, 131))
        first_bytes = first.read_bytes()
        shown = self.value(session, self.controls["input_image"])
        with Image.open(shown["path"]) as image:
            self.assertEqual(image.getpixel((0, 0)), (71, 91, 111, 131))
        self.assertIsNone(self.value(session, self.workspace.context))
        point = self.event(self.controls["input_image"], "select", "add_point")
        asyncio.run(
            self.api(
                session,
                point,
                event_data=gr.SelectData(
                    self.controls["input_image"], {"index": [12, 13], "value": None}
                ),
            )
        )
        self.assertEqual(
            json.loads(self.value(session, self.controls["segment_positive_points"])),
            [{"x": 12, "y": 13}],
        )
        self.assertEqual(self.call(session, sync)[self.controls["input_image"]], gr.skip())
        session.values[self.workspace.mask._id] = "old-mask.png"
        session.values[self.controls["reference_images"]._id] = ["old-reference.png"]
        upload("point-source-C.png", (111, 91, 71, 131))
        self.assertIsNone(self.value(session, self.workspace.mask))
        self.assertEqual(self.value(session, self.controls["segment_positive_points"]), "[]")
        self.assertEqual(self.value(session, self.controls["reference_images"]), [])
        self.assertEqual(first.read_bytes(), first_bytes)
        self.run_request.assert_not_called()

    def test_passive_preview_sync_defers_during_an_accepted_job(self):
        session = self.session("preview-active-job")
        self.open_library(session)
        sync = self.event(self.remote["preview"], "change", "preview_source")
        before = self.value(session, self.controls["input_image"])
        updates = self.call(session, sync, {self.remote["task"]: "accepted-job"})
        self.assertEqual(updates[self.controls["input_image"]], gr.skip())
        self.assertEqual(self.value(session, self.controls["input_image"]), before)
        self.run_request.assert_not_called()

    def test_output_intent_marks_only_real_selector_input_and_survives_source_actions(self):
        session = self.session("explicit-output")
        self.assertFalse(self.value(session, self.workspace.output_explicit))
        fn = self.event(self.controls["output_format"], "input", "mark_output_explicit")
        self.assertTrue(all(name == "input" for _, name in fn.targets))
        self.call(session, fn)
        self.assertTrue(self.value(session, self.workspace.output_explicit))
        self.open_library(session)
        self.call(
            session,
            self.event(self.workspace.action, "change", "action_context"),
            {self.workspace.action: "adjust-image"},
        )
        self.assertTrue(self.value(session, self.workspace.output_explicit))
        self.assertTrue(self.request_values(session, "adjust-image")["_editor_output_explicit"])
        for name in (
            "output_bit_depth",
            "output_color_space",
            "output_crf",
            "_adjustment_hdr_precision",
        ):
            self.assertIn((self.controls[name]._id, "input"), fn.targets)

    def test_upload_then_edit_run_adopts_source_without_open_and_publishes_owned_history(self):
        from gradio.processing_utils import move_files_to_cache

        session = self.session("upload-one-step")
        self.open_library(session)
        old_context = copy.deepcopy(self.value(session, self.workspace.context))
        replacement = self.root / "new-upload.png"
        Image.new("RGBA", (73, 121), (66, 77, 88, 123)).save(replacement)
        original = replacement.read_bytes()
        preview = self.plain(
            move_files_to_cache(
                self.remote["preview"].postprocess(str(replacement)),
                self.remote["preview"],
                postprocess=True,
            )
        )
        session.values[self.remote["preview"]._id] = preview
        self.call(session, self.event(self.remote["preview"], "upload", "source_changed"))
        self.assertIsNone(self.value(session, self.workspace.context))
        self.assertEqual(self.value(session, self.workspace.source_input), "preview")
        self.worker_mode = "owned-copy-result"
        function = self.event(self.remote["run"])

        async def journey():
            with patch(
                "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                return_value="SDXL",
            ):
                result, _ = await self.api(
                    session, function, {self.values: self.request_values(session, "prompt-edit")}
                )
                await self.finish(session, function, result)

        asyncio.run(journey())
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(request["input_image"].getpixel((0, 0)), (66, 77, 88, 123))
        self.assertEqual(request["_image_workflow_step"].parameters["denoise"], 0.3)
        context = self.value(session, self.workspace.context)
        self.assertNotEqual(context["session_id"], old_context["session_id"])
        snapshot = self.store.snapshot(context["session_id"], owner=session.owner)
        self.assertEqual(len(snapshot.steps), 1)
        self.assertEqual(Path(snapshot.original.path).read_bytes(), original)
        self.assertEqual(replacement.read_bytes(), original)
        self.assertEqual(self.value(session, self.controls["prompt"]), self.saved["prompt"])

    def test_slow_old_preview_cannot_replace_input_upload_and_reselect_mirrors_again(self):
        from gradio.processing_utils import move_files_to_cache
        from ruinedfooocus.features.gallery.editing_workflow import working_source

        session = self.session("preview-source-race")
        self.open_library(session)
        sync = self.event(self.remote["preview"], "change", "preview_source")
        self.call(session, sync)
        original_preview = copy.deepcopy(self.value(session, self.remote["preview"]))
        entered, release = threading.Event(), threading.Event()

        def held(record):
            entered.set()
            self.assertTrue(release.wait(5))
            return working_source(record)

        session.state[self.workspace.preview_signature._id] = None
        other = self.root / "input-B.png"
        Image.new("RGBA", (73, 121), (170, 120, 70, 255)).save(other)
        other_value = self.plain(
            move_files_to_cache(
                self.controls["input_image"].postprocess(str(other)),
                self.controls["input_image"],
                postprocess=True,
            )
        )
        with (
            patch("ruinedfooocus.features.gallery.ui_workspace.working_source", side_effect=held),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            pending = pool.submit(self.call, session, sync)
            self.assertTrue(entered.wait(5))
            try:
                session.values[self.controls["input_image"]._id] = other_value
                self.call(
                    session,
                    self.event(self.controls["input_image"], "upload", "input_source_changed"),
                )
            finally:
                release.set()
            updates = pending.result(5)
        self.assertEqual(updates[self.controls["input_image"]], gr.skip())
        self.assertEqual(self.value(session, self.workspace.source_input), "input")
        with Image.open(self.value(session, self.controls["input_image"])["path"]) as image:
            self.assertEqual(image.getpixel((0, 0)), (170, 120, 70, 255))
        self.assertIsNone(self.value(session, self.workspace.preview_signature))
        session.values[self.remote["preview"]._id] = original_preview
        self.call(session, self.event(self.remote["preview"], "upload", "source_changed"))
        self.call(session, sync)
        with Image.open(self.value(session, self.controls["input_image"])["path"]) as image:
            self.assertEqual(image.getpixel((0, 0)), (20, 40, 60, 127))
        self.run_request.assert_not_called()

    def test_quick_tools_reveal_native_panels_and_keep_draft_source_and_saved_values(self):
        from ruinedfooocus.features.image_processing import ui_depth_utilities, ui_native_vision

        session = self.session("quick-tools")
        self.open_library(session)
        before = copy.deepcopy(self.value(session, self.workspace.context))
        prompt = self.value(session, self.controls["prompt"])
        kinds = {
            "Face landmarks": "face-landmarks",
            "Whole-body pose": "pose-map",
            "Depth map": "depth-map",
        }
        with (
            patch.object(
                self.shared,
                "add_ctrl",
                side_effect=lambda key, value: self.controls.update({key: value}),
                create=True,
            ),
            patch.object(self.shared, "path_manager", SimpleNamespace(), create=True),
            patch.object(ui_depth_utilities, "component_choices", return_value=[]),
            patch.object(ui_native_vision, "native_model_choices", return_value=[]),
            patch.object(ui_native_vision, "coco_classes", return_value=[]),
        ):
            with self.app:
                refresh = gr.Number(0)
                mapper = lambda choice, kind: kinds.get(choice)
                ui_depth_utilities.create_ui(
                    self.controls["cn_selection"],
                    self.controls["cn_type"],
                    self.controls["input_image"],
                    refresh,
                    mapper,
                )
                ui_native_vision.create_ui(
                    self.controls["cn_selection"],
                    self.controls["cn_type"],
                    self.controls["input_image"],
                    refresh,
                    mapper,
                )
            native = {
                name: next(
                    fn for fn in self.app.fns.values() if getattr(fn.fn, "__name__", "") == name
                )
                for name in ("utility_context", "context")
            }
            self.assertNotIn(
                "Restore cropped face", {value for _, value in self.workspace.quick_tool.choices}
            )
            for choice in kinds:
                self.call(
                    session,
                    self.event(self.workspace.quick_tool, "input", "select_quick_tool"),
                    {self.workspace.quick_tool: choice},
                )
                fn = native["context" if choice == "Depth map" else "utility_context"]
                self.assertIn((self.controls["cn_selection"]._id, "change"), fn.targets)
                updates = self.call(session, fn)
                self.assertTrue(updates[fn.outputs[0]]["visible"])
                if choice == "Face landmarks":
                    self.assertTrue(updates[fn.outputs[-2]]["visible"])
                elif choice == "Whole-body pose":
                    self.assertTrue(updates[fn.outputs[-1]]["visible"])
                self.assertEqual(self.value(session, self.workspace.context), before)
                self.assertEqual(self.value(session, self.controls["prompt"]), prompt)
                self.assertEqual(self.value(session, self.controls["cn_strength"]), 0.37)
        self.run_request.assert_not_called()

    def test_library_open_edit_keeps_prompt_model_and_source_frame_without_saved_resize(self):
        original = self.legacy.path.read_bytes()
        metadata = self.shared.models.get_models_by_path
        self.enterContext(
            patch.object(
                self.shared.models,
                "get_models_by_path",
                side_effect=lambda category, name, fetch=False: (
                    metadata(category, name, fetch=fetch)
                    | {"model": {"name": "Publisher model"}, "name": "v7.77"}
                ),
            )
        )
        for version in ("Rendered prompt", "Original prompt"):
            with self.subTest(version=version):
                session = self.session("library-" + version.split()[0])
                load = self.legacy.functions["load_selection"]
                legacy = self.call(
                    session,
                    load,
                    {
                        load.inputs[0]: copy.deepcopy(self.legacy.record),
                        load.inputs[1]: copy.deepcopy(self.saved),
                        load.inputs[2]: None,
                        load.inputs[3]: version,
                        load.inputs[5]: "edit",
                        load.inputs[-1]: {},
                    },
                )
                expected = legacy[load.outputs[-1]]["updates"]
                self.assertEqual(expected[3:], [113, 75, ""])
                opened = self.open_library(session, version)
                model_update = opened[self.event(self.library_open).outputs[-1]]["updates"][2]
                self.assertEqual(
                    (dict(model_update["choices"]), model_update["value"]),
                    (
                        {"Publisher model v7.77": "new/model.safetensors"},
                        "new/model.safetensors",
                    ),
                )
                self.assertEqual(session.state[self.destinations._id].selected, "editor")
                self.assertEqual(self.value(session, self.workspace.destination), "editor")
                context = self.value(session, self.workspace.context)
                snapshot = self.store.snapshot(context["session_id"], owner=session.owner)
                self.assertEqual(context, self.workspace.public_context(snapshot))
                self.assertEqual(
                    [
                        self.value(session, self.controls["prompt"]),
                        self.value(session, self.controls["negative"]),
                        self.value(session, self.controls["base_model_name"]),
                        self.value(session, self.controls["custom_width"]),
                        self.value(session, self.controls["custom_height"]),
                    ],
                    [expected[0], expected[1], expected[2]["value"], 73, 121],
                )
                source = self.value(session, self.controls["input_image"])
                with Image.open(source["path"]) as current:
                    self.assertEqual(current.size, (73, 121))
                    self.assertEqual(current.getpixel((0, 0)), (20, 40, 60, 127))
                provenance = self.value(session, self.controls["image_source"])
                self.assertEqual(provenance["filename"], self.legacy.path.name)
                self.assertEqual((provenance["width"], provenance["height"]), (73, 121))
                self.assertEqual(Path(snapshot.current.path), self.legacy.path)
                self.assertEqual(
                    self.legacy.validate.call_args.args[1].session_hash,
                    session.request.session_hash,
                )
                self.assertIs(self.workspace.controls["prompt"], self.controls["prompt"])
                self.assertIs(
                    self.workspace.controls["base_model_name"], self.controls["base_model_name"]
                )
                self.assertIn(self.legacy.editor.run._id, self.app.blocks)
        self.assertEqual(self.legacy.path.read_bytes(), original)
        self.run_request.assert_not_called()

    def test_native_undo_redo_keep_draft_and_opaque_owner_context(self):
        first, second = self.session(), self.session("workspace-B")
        self.open_library(first)
        self.open_library(second)
        context = self.value(first, self.workspace.context)
        foreign = self.value(second, self.workspace.context)
        self.assertNotEqual(context["session_id"], foreign["session_id"])
        ticket = self.store.capture(
            context["session_id"],
            context["revision"],
            "processing",
            source_id=context["source_id"],
            owner=first.owner,
        )
        edited = self.root / "edited.png"
        Image.new("RGBA", (73, 121), (91, 17, 45, 127)).save(edited)
        completed = self.store.complete(
            ticket, edited, execution={"kind": "fixture", "computed": False}, owner=first.owner
        )
        first.state[self.workspace.context._id] = self.workspace.public_context(completed)
        first.values[self.controls["prompt"]._id] = "Keep my new draft"
        self.call(first, self.event(self.workspace.undo))
        undone = self.value(first, self.workspace.context)
        self.assertEqual(undone["source_id"], context["source_id"])
        self.assertGreater(undone["revision"], completed.revision)
        self.assertEqual(self.value(first, self.controls["prompt"]), "Keep my new draft")
        self.call(first, self.event(self.workspace.redo))
        redone = self.value(first, self.workspace.context)
        self.assertEqual(redone["source_id"], completed.current.artifact_id)
        self.assertGreater(redone["revision"], undone["revision"])
        self.assertEqual(self.value(first, self.controls["prompt"]), "Keep my new draft")
        self.assertEqual(self.value(second, self.workspace.context), foreign)
        with self.assertRaises(PermissionError):
            self.call(first, self.event(self.workspace.undo), {self.workspace.context: foreign})
        untouched = self.store.snapshot(foreign["session_id"], owner=second.owner)
        self.assertEqual(untouched.revision, foreign["revision"])
        self.run_request.assert_not_called()

    def test_explicit_setup_replays_saved_target_and_manual_edit_target_is_preserved(self):
        app, event, controls, _, _, _ = setup_fixtures.setup_fixture(random_seed=False)
        metadata = dict(
            self.saved,
            steps=9,
            cfg=3.5,
            sampler_name="euler_ancestral",
            scheduler="normal",
            clip_skip=1,
            loras=[],
            seed=17,
        )
        with (
            patch(
                "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                return_value="SDXL",
            ),
            patch(
                "ruinedfooocus.features.models.component_profiles.replay_choices",
                return_value=({}, None),
            ),
            patch.dict(
                "sys.modules",
                {
                    "ruinedfooocus.core.util": SimpleNamespace(
                        get_lora_thumbnail=lambda *_: None, load_keywords=lambda *_: ""
                    )
                },
            ),
        ):
            result = asyncio.run(
                app.process_api(
                    event["id"],
                    [self.legacy.record, metadata, "Custom...", None, None, None],
                    state=SessionState(app),
                    request=gr.Request(session_hash="explicit-setup"),
                )
            )
        outputs = dict(zip(event["outputs"], result["data"]))
        self.assertEqual(outputs[controls["custom_width"]._id]["value"], 113)
        self.assertEqual(outputs[controls["custom_height"]._id]["value"], 75)
        session = self.session("manual-target")
        session.values.update(
            {
                self.controls["resolution_auto"]._id: False,
                self.controls["custom_width"]._id: 257,
                self.controls["custom_height"]._id: 319,
            }
        )
        self.open_library(session)
        self.assertEqual(self.value(session, self.controls["custom_width"]), 257)
        self.assertEqual(self.value(session, self.controls["custom_height"]), 319)
        self.assertFalse(self.value(session, self.controls["resolution_auto"]))
        self.run_request.assert_not_called()

    def test_auto_preserve_edit_requests_use_owned_frame_without_rewriting_size_draft(self):
        from ruinedfooocus.features.image_editing.inpaint_utils import source_and_mask

        session = self.session("auto-preserve-edit")
        self.open_library(session)
        session.values.update(
            {self.controls["custom_width"]._id: 512, self.controls["custom_height"]._id: 768}
        )
        context = copy.deepcopy(self.value(session, self.workspace.context))
        mask_path = self.root / "auto-preserve-mask.png"
        Image.new("L", (73, 121), 127).save(mask_path)
        mask = self.store.bind_mask(
            context["session_id"], mask_path, context["source_id"], owner=session.owner
        )
        with patch(
            "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
            return_value="SD1",
        ):
            for action in ("prompt-edit", "inpaint"):
                with self.subTest(action=action):
                    values = self.request_values(session, action)
                    values["_editor_context"]["mask_id"] = mask.artifact_id
                    request = self.workspace.build_request(values, request=session.request)
                    self.assertIsInstance(
                        request["_image_workflow_step"], image_workflows.WorkflowStep
                    )
                    self.assertEqual((request["custom_width"], request["custom_height"]), (73, 121))
                    self.assertEqual((values["custom_width"], values["custom_height"]), (512, 768))
                    self.assertEqual(self.value(session, self.controls["custom_width"]), 512)
                    self.assertEqual(self.value(session, self.controls["custom_height"]), 768)
                    self.assertEqual(self.value(session, self.workspace.context), context)
                    if action == "inpaint":
                        photo, selected = source_and_mask(
                            request["main_view"], request["inpaint_view"], preserve_alpha=True
                        )
                        self.assertEqual(photo.size, (73, 121))
                        self.assertEqual(photo.getchannel("A").getextrema(), (127, 127))
                        self.assertEqual(selected.size, (73, 121))
                        self.assertEqual(selected.getextrema(), (127, 127))
        self.run_request.assert_not_called()

    def test_manual_preserve_rejects_size_change_and_explicit_pad_crop_keep_target(self):
        session = self.session("explicit-edit-frame")
        self.open_library(session)
        context = copy.deepcopy(self.value(session, self.workspace.context))
        mask_path = self.root / "explicit-frame-mask.png"
        Image.new("L", (73, 121), 255).save(mask_path)
        mask = self.store.bind_mask(
            context["session_id"], mask_path, context["source_id"], owner=session.owner
        )
        with patch(
            "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
            return_value="SD1",
        ):
            for action in ("prompt-edit", "inpaint"):
                values = self.request_values(session, action)
                values.update(custom_width=257, custom_height=319, resolution_auto=False)
                values["_editor_context"]["mask_id"] = mask.artifact_id
                with self.subTest(action=action, framing="manual-preserve"):
                    with self.assertRaisesRegex(ValueError, "explicit pad or crop"):
                        self.workspace.build_request(values, request=session.request)
                for automatic, framing in ((True, "pad"), (True, "crop"), (False, "crop")):
                    with self.subTest(action=action, automatic=automatic, framing=framing):
                        request = self.workspace.build_request(
                            dict(values, resolution_auto=automatic, _editor_framing=framing),
                            request=session.request,
                        )
                        self.assertEqual(
                            (request["custom_width"], request["custom_height"]), (257, 319)
                        )
                        self.assertEqual(
                            request["_image_workflow_step"].parameters["image_resize_mode"], framing
                        )
        self.run_request.assert_not_called()

    def test_missing_saved_model_adopts_source_and_keeps_current_model(self):
        session = self.session()
        self.call(
            session,
            self.event(self.library_open),
            {
                self.selected: copy.deepcopy(self.legacy.record),
                self.metadata: dict(
                    self.saved, base_model_name="missing-model.safetensors", base_model_hash=None
                ),
            },
        )
        context = self.value(session, self.workspace.context)
        snapshot = self.store.snapshot(context["session_id"], owner=session.owner)
        self.assertEqual(Path(snapshot.current.path), self.legacy.path)
        self.assertEqual(self.value(session, self.controls["base_model_name"]), "model")
        self.assertIn("missing-model", self.value(session, self.workspace.notice))
        self.run_request.assert_not_called()

    def test_delayed_library_load_cannot_paint_over_newer_create_draft_model_or_size(self):
        session = self.session()
        self.open_library(session)
        replacement = self.root / "delayed-library.png"
        Image.new("RGBA", (91, 65), (17, 83, 149, 255)).save(replacement)
        function = self.event(self.library_open)
        module = importlib.import_module("ruinedfooocus.features.gallery.ui_workspace")
        working_source = module.working_source
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def delayed(record):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("delayed Library fixture was not released")
            return working_source(record)

        with (
            patch.object(module, "working_source", delayed),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            pending = executor.submit(
                lambda: asyncio.run(
                    self.api(
                        session,
                        function,
                        {
                            self.selected: source_record(replacement),
                            self.metadata: dict(self.saved, prompt="late saved prompt"),
                        },
                        publish=False,
                    )
                )
            )
            try:
                self.assertTrue(entered.wait(2))
                manual = {
                    "prompt": "Typed during the delayed load",
                    "negative": "new exclusions",
                    "base_model_name": "model",
                    "custom_width": 157,
                    "custom_height": 219,
                    "resolution_auto": False,
                }
                session.values.update(
                    {self.controls[name]._id: value for name, value in manual.items()}
                )
            finally:
                release.set()
            _, projected = pending.result(5)
        for name in ("prompt", "negative", "base_model_name", "custom_width", "custom_height"):
            self.assertEqual(projected.get(self.controls[name], gr.skip()), gr.skip())
        painted = self.publish(session, function, projected)
        for name in ("prompt", "negative", "base_model_name", "custom_width", "custom_height"):
            self.assertEqual(painted[self.controls[name]], gr.skip())
        for name, value in manual.items():
            self.assertEqual(self.value(session, self.controls[name]), value)
        self.run_request.assert_not_called()

    def test_generated_gallery_select_b_then_run_never_emits_old_a_source(self):
        replacement = self.root / "generated-B.png"
        Image.new("RGBA", (73, 121), (31, 151, 47, 255)).save(replacement)
        for order in (("gallery_change", "source_changed"), ("source_changed", "gallery_change")):
            with self.subTest(order=order):
                session = self.session("gallery-" + order[0])
                self.open_library(session)
                old_source = copy.deepcopy(self.value(session, self.workspace.context))

                async def journey():
                    await self.api(
                        session,
                        self.event(self.present_results),
                        {
                            self.products: [str(self.legacy.path), str(replacement)],
                        },
                    )
                    gallery = self.value(session, self.remote["gallery"])
                    event = gr.SelectData(
                        self.remote["gallery"],
                        {
                            "index": 1,
                            "value": gallery[1],
                            "selected": True,
                        },
                    )
                    for callback in order:
                        await self.api(
                            session,
                            self.event(self.remote["gallery"], "select", callback),
                            event_data=event,
                        )
                    self.assertIsNone(self.value(session, self.workspace.context))
                    latest = self.value(session, self.workspace.latest_generated)
                    self.assertEqual(Path(latest["path"]), replacement)
                    self.assertEqual(latest["batch_index"], 1)
                    selected = self.value(session, self.remote["preview"])
                    with Image.open(selected["path"]) as image:
                        self.assertEqual(image.getpixel((0, 0)), (31, 151, 47, 255))
                    held = self.value(session, self.controls["input_image"])
                    with Image.open(held["path"]) as image:
                        self.assertEqual(image.getpixel((0, 0)), (20, 40, 60, 127))
                    with patch(
                        "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                        return_value="SDXL",
                    ):
                        data = self.workspace.build_request(
                            self.request_values(session, "prompt-edit"), request=session.request
                        )
                    self.assertEqual(data["input_image"].getpixel((0, 0)), (31, 151, 47, 255))
                    self.assertNotEqual(
                        data["_editor_operation"].session_id, old_source["session_id"]
                    )
                    self.assertEqual(data["prompt"], self.value(session, self.controls["prompt"]))

                asyncio.run(journey())
                self.assertEqual(self.requests, [])
                self.run_request.assert_not_called()
                self.assertEqual(
                    self.store.snapshot(old_source["session_id"], owner=session.owner).revision,
                    old_source["revision"],
                )

    def test_stale_fallback_terminal_cannot_adopt_before_publication_fence(self):
        session = self.session()
        self.open_library(session)
        context = copy.deepcopy(self.value(session, self.workspace.context))
        session.values[self.controls["prompt"]._id] = "Keep the newer fallback draft"
        self.worker_mode = "fallback-result"
        publish = self.fence.publish
        invalidated = []

        def replace_at_publish(ticket, callback):
            if self.fallback_ready and not invalidated:
                invalidated.append(True)
                self.fence.begin(ticket.owner, ticket.channel, ticket.version + 1)
            return publish(ticket, callback)

        async def journey():
            function = self.event(self.remote["run"])
            result, _ = await self.api(
                session,
                function,
                {
                    self.values: self.request_values(session, "generate"),
                    self.workspace.version: 2,
                },
            )
            await self.finish(session, function, result)

        with (
            patch.object(self.fence, "publish", side_effect=replace_at_publish),
            patch.object(self.store, "start", wraps=self.store.start) as adopt,
        ):
            asyncio.run(journey())
        self.assertTrue(invalidated)
        adopt.assert_not_called()
        self.assertEqual(self.value(session, self.workspace.context), context)
        self.assertEqual(
            self.value(session, self.controls["prompt"]), "Keep the newer fallback draft"
        )

    def test_native_select_background_compose_and_history_keep_one_owned_session(self):
        session = self.session()
        self.open_library(session)
        original = copy.deepcopy(self.value(session, self.workspace.context))
        original_bytes = self.legacy.path.read_bytes()
        session.values[self.controls["prompt"]._id] = "Keep this current draft"

        async def journey():
            await self.select_background(session)
            self.run_request.assert_not_called()
            selected = self.value(session, self.workspace.context)
            self.assertEqual(selected["session_id"], original["session_id"])
            self.assertEqual(selected["additional_image_ids"], [original["source_id"]])
            function = self.event(self.remote["run"])
            result, _ = await self.api(
                session,
                function,
                {self.values: self.request_values(session, "compose"), self.workspace.version: 2},
            )
            task_id = self.value(session, self.remote["task"])
            self.assertEqual(self.jobs.get(task_id, owner=session.owner).owner, session.owner)
            await self.finish(session, function, result)

        asyncio.run(journey())
        self.assertEqual(len(self.requests), 1)
        ticket = self.requests[0][editor_session.KEY]
        self.assertEqual(ticket.session_id, original["session_id"])
        context = self.value(session, self.workspace.context)
        self.assertEqual(context["session_id"], original["session_id"])
        self.assertEqual(context["additional_image_ids"], [])
        snapshot = self.store.snapshot(context["session_id"], owner=session.owner)
        self.assertEqual(context, self.workspace.public_context(snapshot))
        self.assertEqual(
            [step["operation"] for step in snapshot.steps], ["select-background", "compose"]
        )
        self.assertEqual(snapshot.source_generation["prompt"], self.saved["prompt"])
        self.assertNotIn("sampler_name", snapshot.steps[-1]["execution"])
        with Image.open(snapshot.current.path) as composed:
            self.assertEqual((composed.mode, composed.size), ("RGBA", (73, 121)))
            self.assertEqual(composed.getpixel((0, 0)), (10, 20, 158, 255))
        self.assertEqual(self.value(session, self.controls["prompt"]), "Keep this current draft")
        self.assertIsNone(self.value(session, self.remote["task"]))
        self.call(session, self.event(self.workspace.undo))
        undone = self.value(session, self.workspace.context)
        self.assertNotEqual(undone["source_id"], context["source_id"])
        self.call(session, self.event(self.workspace.redo))
        self.assertEqual(
            self.value(session, self.workspace.context)["source_id"], context["source_id"]
        )
        self.assertEqual(self.value(session, self.controls["prompt"]), "Keep this current draft")
        self.assertEqual(self.legacy.path.read_bytes(), original_bytes)

    def test_browsing_and_source_adoption_defer_without_cancelling_accepted_run(self):
        session = self.session()
        self.open_library(session)
        original = copy.deepcopy(self.value(session, self.workspace.context))
        replacement = self.root / "replacement.png"
        Image.new("RGBA", (73, 121), (81, 13, 107, 255)).save(replacement)

        async def journey():
            await self.select_background(session)
            function = self.event(self.remote["run"])
            result, _ = await self.api(
                session,
                function,
                {self.values: self.request_values(session, "compose"), self.workspace.version: 2},
            )
            task_id = self.value(session, self.remote["task"])
            session.values[self.controls["prompt"]._id] = "Typed while the old run waited"
            await self.api(
                session,
                self.event(self.library_open),
                {self.selected: source_record(replacement), self.metadata: {}},
            )
            latest = copy.deepcopy(self.value(session, self.workspace.context))
            invalidate = next(
                function
                for function in self.app.fns.values()
                if getattr(function.fn, "__name__", "") == "invalidate"
            )
            await self.api(session, invalidate, {self.workspace.version: 3})
            self.assertFalse(self.jobs.get(task_id, owner=session.owner).cancelled.is_set())
            self.stop_request.assert_not_called()
            self.assertEqual(latest["session_id"], original["session_id"])
            await self.finish(session, function, result)
            completed = self.value(session, self.workspace.context)
            self.assertEqual(completed["session_id"], original["session_id"])
            self.assertGreater(completed["revision"], latest["revision"])

        asyncio.run(journey())
        self.assertEqual(
            self.value(session, self.controls["prompt"]), "Typed while the old run waited"
        )
        current = self.value(session, self.controls["input_image"])
        with Image.open(current["path"]) as image:
            self.assertEqual(image.getpixel((0, 0)), (10, 20, 158, 255))
        preview = self.value(session, self.remote["preview"])
        with Image.open(preview["path"]) as image:
            self.assertEqual(image.getpixel((0, 0)), (10, 20, 158, 255))
        old = self.store.snapshot(original["session_id"], owner=session.owner)
        self.assertEqual(
            [step["operation"] for step in old.steps], ["select-background", "compose"]
        )

    def test_native_failed_cancelled_and_stop_results_preserve_current_image_history_and_draft(
        self,
    ):
        for mode in ("empty-failed", "cancelled-result", "stop-click"):
            with self.subTest(mode=mode):
                self.worker_mode = "completed" if mode == "stop-click" else mode
                session = self.session("terminal-" + mode)
                self.open_library(session)

                async def journey():
                    await self.select_background(session)
                    context = copy.deepcopy(self.value(session, self.workspace.context))
                    session.values[self.controls["prompt"]._id] = "Preserve this terminal draft"
                    function = self.event(self.remote["run"])
                    result, _ = await self.api(
                        session,
                        function,
                        {
                            self.values: self.request_values(session, "compose"),
                            self.workspace.version: 2,
                        },
                    )
                    if mode == "stop-click":
                        await self.api(session, self.event(self.remote["stop"]))
                    await self.finish(session, function, result)
                    self.assertEqual(self.value(session, self.workspace.context), context)
                    snapshot = self.store.snapshot(context["session_id"], owner=session.owner)
                    self.assertEqual(
                        [step["operation"] for step in snapshot.steps], ["select-background"]
                    )

                asyncio.run(journey())
                self.assertEqual(
                    self.value(session, self.controls["prompt"]), "Preserve this terminal draft"
                )
                current = self.value(session, self.controls["input_image"])
                with Image.open(current["path"]) as image:
                    self.assertEqual(image.getpixel((0, 0)), (0, 0, 255, 255))
                preview = self.value(session, self.remote["preview"])
                with Image.open(preview["path"]) as image:
                    self.assertEqual(image.getpixel((0, 0)), (0, 0, 255, 255))

    def test_pre_admission_empty_result_keeps_run_cleanup_without_terminal_lookup(self):
        session = self.session("empty-before-admission")
        self.open_library(session)
        before = copy.deepcopy(self.value(session, self.workspace.context))

        def stream(data):
            yield {
                self.remote["run"]: gr.update(interactive=False, visible=False),
                self.remote["stop"]: gr.update(visible=True),
                self.remote["progress"]: gr.update(visible=True),
            }
            yield {
                self.remote["artifacts"]: [],
                self.remote["preview"]: None,
                self.remote["task"]: None,
                self.remote["run"]: gr.update(interactive=True, visible=True),
                self.remote["stop"]: gr.update(visible=False),
                self.remote["progress"]: gr.update(visible=False),
                self.workspace.context: None,
            }

        self.run_request.side_effect = stream
        worker = importlib.import_module("ruinedfooocus.app.async_worker")
        with (
            patch.object(self.workspace, "build_request", return_value={}),
            patch.object(
                worker, "editor_result", side_effect=ValueError("Unknown image job")
            ) as lookup,
        ):

            async def journey():
                fn = self.event(self.remote["run"])
                result, started = await self.api(
                    session,
                    fn,
                    {self.values: self.request_values(session, "generate")},
                )
                self.assertFalse(started[self.remote["run"]]["interactive"])
                self.assertTrue(started[self.remote["stop"]]["visible"])
                result, cleanup = await self.api(session, fn, iterator=result["iterator"])
                await self.finish(session, fn, result)
                return cleanup

            cleanup = asyncio.run(journey())
            lookup.assert_not_called()
        self.assertTrue(cleanup[self.remote["run"]]["interactive"])
        self.assertTrue(cleanup[self.remote["run"]]["visible"])
        self.assertFalse(cleanup[self.remote["stop"]]["visible"])
        self.assertFalse(cleanup[self.remote["progress"]]["visible"])
        self.assertIsNone(self.value(session, self.remote["task"]))
        self.assertEqual(self.value(session, self.workspace.context), before)

    def test_completed_zero_match_terminal_adopts_authoritative_revision_and_keeps_source(self):
        session = self.session("empty-completed-segment")
        self.open_library(session)
        before = copy.deepcopy(self.value(session, self.workspace.context))
        prompt = self.value(session, self.controls["prompt"])
        terminal = {}

        def stream(data):
            yield {self.remote["task"]: 77}
            ticket = self.store.capture(
                before["session_id"],
                before["revision"],
                "segment",
                source_id=before["source_id"],
                owner=session.owner,
            )
            refs = self.store.capture_derived_results(ticket, [], owner=session.owner)
            snapshot = self.store.complete_derived_results(
                ticket,
                refs,
                output="mask",
                owner=session.owner,
                execution={
                    "native_vision_source": {
                        "output": "mask",
                        "mask_role": "inpaint-selection",
                        "instance_count": 0,
                    }
                },
            )
            terminal.update(status="completed", snapshot=snapshot.to_dict())
            yield {
                self.remote["artifacts"]: [],
                self.remote["preview"]: None,
                self.remote["task"]: None,
            }

        self.run_request.side_effect = stream
        worker = importlib.import_module("ruinedfooocus.app.async_worker")
        with (
            patch.object(self.workspace, "build_request", return_value={"_editor_context": before}),
            patch.object(worker, "editor_result", side_effect=lambda task_id: terminal),
        ):

            async def journey():
                fn = self.event(self.remote["run"])
                result, _ = await self.api(
                    session,
                    fn,
                    {self.values: {"main_view": self.value(session, self.remote["preview"])}},
                )
                await self.finish(session, fn, result)

            asyncio.run(journey())
        context = self.value(session, self.workspace.context)
        self.assertEqual(context["session_id"], before["session_id"])
        self.assertEqual(context["source_id"], before["source_id"])
        self.assertEqual(context["revision"], before["revision"] + 1)
        self.assertEqual(self.value(session, self.controls["prompt"]), prompt)
        with Image.open(self.value(session, self.remote["preview"])["path"]) as image:
            self.assertEqual(image.getpixel((0, 0)), (20, 40, 60, 127))

    def test_rejected_mask_upload_cleans_only_its_new_owned_copy(self):
        from gradio.processing_utils import move_files_to_cache

        session = self.session("mask-admission")
        self.open_library(session)
        owned = self.root / "owned-mask-cache"
        upload = self.event(self.workspace.mask, "upload")

        def mask_value(path):
            return self.plain(
                move_files_to_cache(
                    self.workspace.mask.postprocess(str(path)),
                    self.workspace.mask,
                    postprocess=True,
                )
            )

        with patch("ruinedfooocus.core.cache_contract.configured_cache_root", return_value=owned):
            for mode, size in (("RGBA", (73, 121)), ("L", (74, 121))):
                path = self.root / (mode + "-invalid-mask.png")
                Image.new(mode, size).save(path)
                original = path.read_bytes()
                with self.assertRaisesRegex(
                    ValueError, "single-channel and match its source frame"
                ):
                    self.call(session, upload, {self.workspace.mask: mask_value(path)})
                self.assertEqual([path for path in owned.rglob("*") if path.is_file()], [])
                self.assertEqual(path.read_bytes(), original)
            valid = self.root / "valid-mask.png"
            Image.new("L", (73, 121), 127).save(valid)
            self.call(session, upload, {self.workspace.mask: mask_value(valid)})
            context = self.value(session, self.workspace.context)
            retained = self.store.artifact(
                context["session_id"], context["mask_id"], owner=session.owner
            )
            with patch.object(self.store, "_save", side_effect=OSError("manifest write failed")):
                with self.assertRaisesRegex(OSError, "manifest write failed"):
                    self.call(session, upload, {self.workspace.mask: mask_value(valid)})
            self.assertEqual(
                {path for path in owned.rglob("*") if path.is_file()}, {Path(retained.path)}
            )
            self.assertEqual(Path(retained.path).read_bytes(), valid.read_bytes())

    def test_open_current_uses_precision_original_record_instead_of_display_png(self):
        from ruinedfooocus.features.image_processing.native_image_outputs import NativeImageOutput

        original = self.root / "precision-original.png"
        display = self.root / "precision-display.png"
        Image.new("I;16", (73, 121), 12345).save(original)
        Image.new("RGB", (73, 121), (20, 130, 70)).save(display)
        original_bytes = original.read_bytes()
        session = self.session("precision-original-owner")

        async def journey():
            await self.api(
                session,
                self.event(self.present_results),
                {self.products: [NativeImageOutput(str(original), str(display))]},
            )
            gallery = self.value(session, self.remote["gallery"])
            selection = gr.SelectData(
                self.remote["gallery"], {"index": 0, "value": gallery[0], "selected": True}
            )
            for callback in ("gallery_change", "source_changed"):
                await self.api(
                    session,
                    self.event(self.remote["gallery"], "select", callback),
                    event_data=selection,
                )
            self.assertIsNone(self.value(session, self.workspace.context))
            await self.api(
                session, self.event(self.workspace.open_current), {self.workspace.version: 100}
            )

        asyncio.run(journey())
        context = self.value(session, self.workspace.context)
        snapshot = self.store.snapshot(context["session_id"], owner=session.owner)
        self.assertEqual(Path(snapshot.original.path), original)
        self.assertEqual(Path(snapshot.current.path), original)
        self.assertNotEqual(snapshot.current.frame.dtype, "uint8")
        self.assertEqual(Path(self.value(session, self.remote["native_source"])["path"]), original)
        data = self.workspace.build_request(
            self.request_values(session, "remove-background"), request=session.request
        )
        self.assertEqual(Path(data["_editor_operation"].source.path), original)
        self.assertEqual(data["input_image"].mode, "RGBA")
        self.assertEqual(data["input_image"].size, (73, 121))
        self.assertNotEqual(data["image_source"]["conversion"], "none")
        self.assertEqual(self.store.snapshot(context["session_id"], owner=session.owner).steps, ())
        self.assertEqual(original.read_bytes(), original_bytes)
        self.run_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
