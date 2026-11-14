"""Installed Gradio constructs and submits direct Library edits without compute."""

import asyncio
import copy
from concurrent.futures import CancelledError, ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import subprocess
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.data_classes import JsonData
from gradio.helpers import special_args
from gradio.i18n import I18nData
from gradio.state_holder import SessionState
from PIL import Image

from ruinedfooocus.app.preparation_admission import PreparationAdmission
from ruinedfooocus.features.gallery.actions import source_record
from ruinedfooocus.features.gallery.ui_editing import LibraryEditor


class LibraryEditingUiTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.path = self.root / "selected.png"
        Image.new("RGBA", (73, 121), (20, 40, 60, 127)).save(self.path)
        self.record = source_record(self.path)
        self.captured = []
        self.closed = False
        self.defaults = dict(
            base_model_name="model",
            custom_width=512,
            custom_height=768,
            cn_strength=0.37,
            prompt="Create draft",
            negative="old negative",
            automatic_upscale="Off",
            seed_random=False,
            seed=42,
            rembg_native_model="birefnet.safetensors",
            native_vision_model="",
            image_component_overrides={"vae": "current.safetensors"},
            loras=[("thumbnail", "0.5 - detail.safetensors")],
        )
        self.policy = SimpleNamespace(
            checkpoint_features=lambda *_: dict(native_edit=False, inpaint=True),
            controlnet=SimpleNamespace(cn_options={"Img2Img": dict(type="img2img")}),
            NEWCN="Custom...",
        )
        self.validate = Mock(return_value=None)
        self.stop = Mock()
        self.stop.__name__ = "stop_fixture"
        pool = PreparationAdmission()
        self.addCleanup(pool.close)
        with gr.Blocks(analytics_enabled=False) as self.app:
            controls = {name: gr.State(value) for name, value in self.defaults.items()}
            shared = SimpleNamespace(
                get_ctrl=controls.__getitem__,
                state={"ctrls_name": list(controls), "ctrls_obj": list(controls.values())},
                models=SimpleNamespace(get_names=lambda _: ["model"]),
                settings=SimpleNamespace(default_settings={}),
            )
            self.shared = shared
            self.remote = dict(
                run=gr.Button(),
                stop=gr.Button(),
                progress=gr.HTML(),
                preview=gr.Image(),
                artifacts=gr.State([]),
                native_source=gr.State(None),
                originals=gr.File(file_count="multiple"),
                gallery=gr.Gallery(),
                task=gr.State(None),
            )
            modules = {
                "ruinedfooocus.app": SimpleNamespace(state=shared),
                "ruinedfooocus.app.preparation_admission": SimpleNamespace(
                    preparation_admission=pool
                ),
                "ruinedfooocus.features.image_editing.ui": self.policy,
            }
            with patch.dict("sys.modules", modules):
                self.editor = LibraryEditor()
                self.local_prompt = gr.Button("Use prompt locally")
                self.local_setup = gr.Button("Use setup locally")
                self.editor.bind(
                    gr.State(None),
                    gr.JSON({}),
                    gr.Radio(["Rendered prompt", "Original prompt"]),
                    validate_record=self.validate,
                    resets=dict(
                        input_image=None,
                        reference_images=[],
                        inpaint_toggle=False,
                        inpaint_view=None,
                        _native_main_source=None,
                        _generation_artifacts=[],
                    ),
                    run_request=self.run_request,
                    stop_request=self.stop,
                    run_controls=self.remote,
                    prompt_button=self.local_prompt,
                    setup_button=self.local_setup,
                    setup_values=lambda saved: {
                        controls["prompt"]: gr.update(value="saved local setup"),
                        controls["negative"]: gr.update(value="saved local exclusions"),
                    },
                )
        self.functions = {
            function.fn.__name__: function
            for function in self.app.fns.values()
            if function.fn is not None
        }

    def browser_json_result(self, callback, prediction):
        """Use the registered Gradio transport before handing an envelope to JS."""
        component = callback.outputs[-1]
        self.assertIsInstance(component, gr.JSON)
        raw = prediction if len(callback.outputs) == 1 else prediction[component]
        self.assertIsInstance(raw["message"], I18nData)
        processed = asyncio.run(
            self.app.postprocess_data(callback, prediction, SessionState(self.app))
        )[-1]
        self.assertIsInstance(processed, JsonData)
        envelope = processed.model_dump()
        self.assertEqual(envelope["guard"], raw["guard"])
        self.assertEqual(envelope["message"], str(raw["message"]))
        self.assertTrue(envelope["message"].startswith("__i18n__"))
        return envelope

    def test_canceled_editor_metadata_serializes_every_output_without_changing_source(self):
        load = self.functions["load_selection"]
        self.validate.side_effect = CancelledError("superseded selection")
        prediction = asyncio.run(
            load.fn(
                self.record,
                {},
                None,
                "Rendered prompt",
                "model",
                "edit",
                "draft",
                "negative",
                "model",
                512,
                768,
                {},
                gr.Request(session_hash="A"),
            )
        )
        self.assertEqual(prediction, gr.skip())
        state = SessionState(self.app)
        state[self.editor.source._id] = self.record.copy()
        processed = asyncio.run(self.app.postprocess_data(load, prediction, state))
        self.assertEqual(len(processed), len(load.outputs))
        self.assertGreater(len(processed), 1)
        self.assertTrue(all(value is None or value == gr.skip() for value in processed))
        self.assertEqual(state[self.editor.source._id], self.record)

    def test_codec_context_keeps_the_saved_unavailable_selection_valid(self):
        selected = "missing-saved-codec.safetensors"
        self.shared.path_manager = SimpleNamespace(get_model_folders=lambda _: [])
        self.editor.seedvr2_vae.choices = [(selected, selected)]
        function = self.functions["codec_context"]
        state = SessionState(self.app)
        asyncio.run(self.app.process_api(function, ["restore-seedvr2", selected], state=state))
        field = state.blocks_config.blocks[self.editor.seedvr2_vae._id]
        self.assertEqual(field.preprocess(selected), selected)
        self.assertFalse(field.allow_custom_value)

    def test_precision_draft_notice_resolves_registered_translation_before_concatenation(self):
        with patch(
            "ruinedfooocus.features.gallery.ui_editing.working_source",
            return_value=(
                Image.new("RGBA", (73, 121)),
                {"filename": "original.exr", "conversion": "working copy"},
            ),
        ):
            result = asyncio.run(
                self.functions["load_selection"].fn(
                    self.record,
                    {},
                    None,
                    "Rendered prompt",
                    "model",
                    "edit",
                    "draft",
                    "negative",
                    "model",
                    73,
                    121,
                    {},
                    gr.Request(session_hash="precision"),
                )
            )
        self.assertIsInstance(result[self.editor.notice], str)
        self.assertIn("prepared automatically", result[self.editor.notice])
        self.assertIn("original is kept", result[self.editor.notice])

    def test_delayed_metadata_uses_existing_browser_fence_and_preserves_manual_text(self):
        load = self.functions["load_selection"]
        dependency = next(
            item for item in self.app.config["dependencies"] if item["id"] == load._id
        )
        publication = next(
            item for item in self.app.config["dependencies"] if item["trigger_after"] == load._id
        )
        arguments = [
            None,
            {"prompt": "saved", "negative": "saved negative"},
            None,
            "Rendered prompt",
            None,
            "edit",
            "before",
            "negative before",
            "model",
            512,
            768,
            {},
        ]
        harness = r"""
const assert=require('node:assert/strict'),vm=require('node:vm');
const data=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const prompt={value:'before',listeners:{},addEventListener(n,fn){this.listeners[n]=fn}};
const negative={value:'negative before',checked:false};
const nodes={}; const scope={document:{getElementById(id){
  if(id===data.prompt) return {querySelector:()=>prompt};
  return nodes[id] ||= {listeners:{},querySelectorAll(){return id===data.negative?[negative]:[]},addEventListener(n,fn){this.listeners[n]=fn}};
}}};
const args=vm.runInNewContext('('+data.capture+')',scope)(...data.args);
if(!data.result) process.stdout.write(JSON.stringify(args));
else {
  prompt.value='typed while image loaded'; prompt.listeners.input();
  negative.value='typed negative'; nodes[data.negative].listeners.input();
  const painted=vm.runInNewContext('('+data.publish+')',scope)(data.result);
  assert.equal(painted[0].value,undefined); assert.equal(painted[1].value,undefined);
  assert.equal(prompt.value,'typed while image loaded'); assert.equal(negative.value,'typed negative');
  assert.equal(painted.at(-1).visible,true);
}
"""
        payload = dict(
            prompt=self.editor.prompt.elem_id,
            negative=self.editor.negative.elem_id,
            capture=dependency["js"],
            publish=publication["js"],
            args=arguments,
        )

        def browser(values):
            result = subprocess.run(
                ["node", "-e", harness],
                input=json.dumps(values),
                text=True,
                capture_output=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout

        captured = json.loads(browser(payload))
        state = SessionState(self.app)
        state[load.inputs[0]._id] = self.record
        state[load.inputs[2]._id] = None
        state[load.inputs[4]._id] = "model"
        entered, release = threading.Event(), threading.Event()
        from ruinedfooocus.features.gallery.editing_workflow import working_source

        def delayed(record):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("source fixture was not released")
            return working_source(record)

        with patch("ruinedfooocus.features.gallery.ui_editing.working_source", delayed):
            with ThreadPoolExecutor(max_workers=1) as executor:
                pending = executor.submit(
                    lambda: asyncio.run(
                        self.app.process_api(
                            load, captured, state=state, request=gr.Request(session_hash="A")
                        )
                    )
                )
                try:
                    self.assertTrue(entered.wait(2))
                finally:
                    release.set()
                result = pending.result(5)
        self.assertEqual(state[self.editor.source._id]["path"], self.record["path"])
        self.assertNotIn(self.editor.prompt, load.outputs)
        self.assertNotIn(self.editor.negative, load.outputs)
        envelope = result["data"][-1].model_dump()
        self.assertEqual(envelope["updates"][:2], ["saved", "saved negative"])
        browser(dict(payload, result=envelope))

    def test_real_session_state_stop_is_bound_to_its_own_job(self):
        from ruinedfooocus.app.resource_lifecycle import JobRegistry
        from test_resource_lifecycle import function

        jobs = JobRegistry()
        first = jobs.create(owner=("ui", "", "local", "A"))
        second = jobs.create(owner=("ui", "", "local", "B"))
        callback = function(
            "ruinedfooocus/ui/app.py",
            "stop_clicked",
            {"worker": SimpleNamespace(interrupt_processing=jobs.cancel)},
        )
        with gr.Blocks() as app:
            task, button = gr.State(None), gr.Button()
            button.click(callback, inputs=task, queue=False)
        a, b = SessionState(app), SessionState(app)
        a[task._id], b[task._id] = first.task_id, second.task_id
        asyncio.run(app.process_api(0, [None], state=a, request=gr.Request(session_hash="A")))
        self.assertTrue(first.cancelled.is_set())
        self.assertFalse(second.cancelled.is_set())
        a[task._id] = second.task_id
        with self.assertRaisesRegex(PermissionError, "different session"):
            asyncio.run(app.process_api(0, [None], state=a, request=gr.Request(session_hash="A")))
        self.assertFalse(second.cancelled.is_set())

    def run_request(self, data):
        self.captured.append(copy.deepcopy(data))
        try:
            if getattr(self, "stream_error", None):
                raise self.stream_error
            yield {
                self.remote["preview"]: gr.update(value=str(self.path)),
                self.remote["artifacts"]: [self.record],
            }
        finally:
            self.closed = True

    def arguments(self, source=None):
        return [
            self.record,
            source or copy.deepcopy(self.record),
            "Changed instruction",
            "No blur",
            "model",
            "edit",
            0.37,
            1024,
            768,
            None,
            None,
            False,
            *copy.deepcopy(list(self.defaults.values())),
        ]

    def test_real_gradio_signature_injects_owner_before_variadic_create_capture(self):
        self.assertEqual(self.editor.denoise.value, 0.3)
        self.assertEqual(self.defaults["cn_strength"], 0.37)
        callback = self.functions["run"].fn
        request = gr.Request(session_hash="Library-session-A")
        inputs, *_ = special_args(callback, self.arguments(), request=request)
        stream = callback(*inputs)
        next(stream)
        output = next(stream)
        self.assertEqual(self.validate.call_args.args[1].session_hash, "Library-session-A")
        self.assertEqual(len(self.captured), 1)
        request = self.captured[0]
        self.assertEqual(request["prompt"], "Changed instruction")
        self.assertEqual(request["negative"], "No blur")
        self.assertEqual(request["cn_strength"], 0.37)
        self.assertEqual(request["seed"], 42)
        self.assertEqual(
            request["image_component_overrides"], self.defaults["image_component_overrides"]
        )
        self.assertEqual(request["loras"], self.defaults["loras"])
        from ruinedfooocus.features.gallery import editor_session

        self.assertIn(editor_session.KEY, request)
        self.assertEqual(request[editor_session.KEY].parameters["custom_width"], 1024)
        self.assertTrue(set(output).isdisjoint(self.remote.values()))
        self.assertEqual(request["input_image"].getpixel((0, 0)), (20, 40, 60, 127))
        self.assertIsNone(request["_native_main_source"])
        self.assertEqual(output[self.editor.downloads]["value"], [str(self.path)])
        stream.close()
        self.assertTrue(self.closed)

    def test_prompt_and_setup_actions_write_only_the_gallery_draft(self):
        local = self.functions["use_local_prompt"].fn(
            self.record,
            {"prompt": "local instruction", "negative": "local negative"},
            "Rendered prompt",
            "prior",
            "negative",
            "model",
            512,
            768,
            {},
        )
        self.assertEqual(local["updates"], ["local instruction", "local negative"])
        updates = self.functions["use_local_setup"].fn(
            self.record,
            {"prompt": "saved"},
            {},
            "",
            "prior",
            "negative",
            "model",
            512,
            768,
            0.37,
            {},
        )
        result = updates[self.functions["use_local_setup"].outputs[-1]]
        self.assertEqual(result["updates"][0], "saved local setup")
        self.assertEqual(
            updates[self.editor.setup][result["updates"][-2]]["values"]["negative"],
            "saved local exclusions",
        )
        self.assertTrue(set(updates).isdisjoint(self.shared.state["ctrls_obj"]))
        event = self.functions["run"]
        self.assertEqual((event.concurrency_id, event.concurrency_limit), ("compute:shared", 1))
        stop = next(
            fn for fn in self.app.fns.values() if fn.targets == [(self.editor.stop._id, "click")]
        )
        self.assertEqual(stop.inputs, [self.editor.task])

    def test_local_reuse_paints_only_when_the_registered_browser_draft_is_current(self):
        harness = r"""
const vm=require('node:vm'),data=JSON.parse(require('node:fs').readFileSync(0,'utf8')),nodes={};
for(const entry of data.nodes){const input={value:String(entry.value??''),listeners:{},addEventListener(n,f){this.listeners[n]=f}};
nodes[entry.id]={input,querySelector(){return input},querySelectorAll(){return[input]},addEventListener(){}};}
const scope={document:{getElementById(id){return nodes[id]||null}},window:{}};
const args=vm.runInNewContext('('+data.capture+')',scope)(...data.arguments);
if(!data.result)process.stdout.write(JSON.stringify(args));else{
const prompt=nodes[data.prompt].input;prompt.value='typed after server validation';prompt.listeners.input();
const result={...data.result,guard:args.at(-1)};
const updates=vm.runInNewContext('('+data.publish+')',scope)(result);
if(updates.slice(0,-1).some(value=>value&&Object.hasOwn(value,'value')))throw Error('late result overwrote a newer draft');
process.stdout.write(JSON.stringify(updates));}
"""
        for name in ("use_local_prompt", "use_local_setup"):
            callback = self.functions[name]
            event = next(
                item for item in self.app.config["dependencies"] if item["id"] == callback._id
            )
            publication = next(
                item
                for item in self.app.config["dependencies"]
                if item["trigger_after"] == callback._id
            )
            arguments = [control.value for control in callback.inputs]
            arguments[0], arguments[1] = (
                self.record,
                {"prompt": "saved", "negative": "saved negative"},
            )
            nodes = [
                {"id": control.elem_id, "value": control.value}
                for control in callback.inputs
                if control.elem_id and not isinstance(control, (gr.State, gr.JSON))
            ]
            data = dict(
                nodes=nodes,
                prompt=self.editor.prompt.elem_id,
                capture=event["js"],
                publish=publication["js"],
                arguments=arguments,
            )

            def browser(payload):
                result = subprocess.run(
                    ["node", "-e", harness],
                    input=json.dumps(payload),
                    text=True,
                    capture_output=True,
                    timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(result.stdout)

            captured = browser(data)
            # Native Gradio replaces State inputs on the server; JSON roundtrip
            # must not turn its authoritative signature tuple into client data.
            captured[0] = self.record
            result = callback.fn(*captured)
            envelope = self.browser_json_result(callback, result)
            browser(dict(data, result=envelope))
            if name == "use_local_setup":
                self.assertNotIn(self.editor.prompt, callback.outputs)
                self.assertNotIn(self.editor.setup_choice, callback.outputs)
                self.assertLessEqual(len(result[self.editor.setup]), 8)

    def test_failed_local_stream_releases_its_ticket_and_restores_local_controls(self):
        self.stream_error = RuntimeError("owned stream failed")
        request = gr.Request(session_hash="local-failure")
        inputs, *_ = special_args(self.functions["run"].fn, self.arguments(), request=request)
        outputs = list(self.functions["run"].fn(*inputs))
        self.assertEqual(outputs[-2][self.editor.notice], "owned stream failed")
        self.assertTrue(outputs[-1][self.editor.run]["interactive"])
        self.assertIsNone(outputs[-1][self.editor.task])
        self.assertTrue(all(set(output).isdisjoint(self.remote.values()) for output in outputs))
        from ruinedfooocus.features.gallery import editor_session
        from ruinedfooocus.app.resource_lifecycle import ui_request_owner

        ticket = self.captured[0][editor_session.KEY]
        with self.assertRaises(CancelledError):
            editor_session.default_store().validate(ticket, owner=ui_request_owner(request))

    def test_run_uses_only_the_painted_setup_token_and_retains_it_when_snapshots_are_pruned(self):
        previous = {
            "painted": {"source": self.record, "values": {"custom_steps": 5}},
            **{f"older-{index}": {"custom_steps": index} for index in range(10)},
        }
        result = self.functions["use_local_setup"].fn(
            self.record, {}, previous, "painted", "prior", "negative", "model", 512, 768, 0.37, {}
        )
        snapshots = result[self.editor.setup]
        self.assertEqual(len(snapshots), 8)
        self.assertEqual(snapshots["painted"]["values"], {"custom_steps": 5})
        arguments = [*self.arguments(), None, 4, "none", snapshots, "painted"]
        inputs, *_ = special_args(
            self.functions["run"].fn, arguments, request=gr.Request(session_hash="painted-setup")
        )
        stream = self.functions["run"].fn(*inputs)
        next(stream)
        next(stream)
        stream.close()
        self.assertEqual(self.captured[0]["custom_steps"], 5)

    def test_missing_or_other_source_painted_setup_fails_closed_before_capture(self):
        for snapshots in (
            {},
            {
                "painted": {
                    "source": dict(self.record, signature="older-source"),
                    "values": {"custom_steps": 99},
                }
            },
        ):
            arguments = [*self.arguments(), None, 4, "none", snapshots, "painted"]
            inputs, *_ = special_args(
                self.functions["run"].fn, arguments, request=gr.Request(session_hash="source-fence")
            )
            with self.assertRaisesRegex(gr.Error, "setup no longer matches"):
                next(self.functions["run"].fn(*inputs))
        self.assertEqual(self.captured, [])

    def test_explicit_current_sampling_recovery_preserves_text_and_clears_only_setup(self):
        callback = self.functions["clear_local_setup"]
        updates = callback.fn("typed Gallery draft", "missing", {})
        self.assertEqual(updates[self.editor.setup], {})
        envelope = updates[callback.outputs[-1]]
        self.assertEqual(envelope["updates"][0], gr.skip())
        self.assertEqual(envelope["updates"][1], "")
        self.assertNotIn(self.editor.prompt, callback.outputs)
        event = next(item for item in self.app.config["dependencies"] if item["id"] == callback._id)
        self.assertIn("fence.capture", event["js"])
        publication = next(
            item
            for item in self.app.config["dependencies"]
            if item["trigger_after"] == callback._id
        )
        self.assertIn("fence.publish", publication["js"])
        envelope = self.browser_json_result(callback, updates)
        harness = r"""
const vm=require('node:vm'),data=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const prompt={value:'typed Gallery draft',addEventListener(){}};
const other={value:'missing'};
const scope={window:{},document:{getElementById(id){return {querySelector(){return id===data.prompt?prompt:other},querySelectorAll(){return[other]},addEventListener(){}}}}};
const args=vm.runInNewContext('('+data.capture+')',scope)('typed Gallery draft','missing',{});
const before={...prompt.rfDraft};
const result={...data.result,guard:args.at(-1)};
const output=vm.runInNewContext('('+data.publish+')',scope)(result);
if(prompt.rfDraft.text!==before.text||prompt.rfDraft.revision!==before.revision)throw Error('clear changed canonical prompt state');
if(output[1]!==''||Object.hasOwn(output[0],'value'))throw Error('clear did not preserve prompt slot');
"""
        result = subprocess.run(
            ["node", "-e", harness],
            input=json.dumps(
                dict(
                    prompt=self.editor.prompt.elem_id,
                    capture=event["js"],
                    publish=publication["js"],
                    result=envelope,
                )
            ),
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_old_draft_source_cannot_run_and_mask_is_never_borrowed(self):
        old = dict(self.record, signature="old")
        callback = self.functions["run"].fn
        inputs, *_ = special_args(
            callback, self.arguments(old), request=gr.Request(session_hash="B")
        )
        with self.assertRaisesRegex(gr.Error, "selected image changed"):
            next(callback(*inputs))
        self.assertEqual(self.captured, [])
        arguments = self.arguments()
        arguments[5] = "inpaint"
        inputs, *_ = special_args(callback, arguments, request=gr.Request(session_hash="B"))
        with self.assertRaisesRegex(gr.Error, "Paint a mask"):
            next(callback(*inputs))
        self.assertEqual(self.captured, [])

    def test_utility_submission_uses_explicit_resource_without_generation_model_lookup(self):
        self.policy.checkpoint_features = Mock(side_effect=AssertionError("generation lookup"))
        arguments = self.arguments()
        arguments[4] = None
        arguments[5] = "rembg"
        arguments[9] = "selected-birefnet.safetensors"
        callback = self.functions["run"].fn
        inputs, *_ = special_args(callback, arguments, request=gr.Request(session_hash="utility"))
        stream = callback(*inputs)
        next(stream)
        next(stream)
        stream.close()
        self.assertEqual(self.captured[0]["rembg_native_model"], "selected-birefnet.safetensors")
        self.assertEqual(self.captured[0]["input_image"].size, (73, 121))
        self.policy.checkpoint_features.assert_not_called()

    def test_reselecting_same_source_preserves_draft_and_explicit_prompt_version_changes_it(self):
        callback = self.functions["load_selection"].fn
        with patch.dict("sys.modules", {"ruinedfooocus.app": SimpleNamespace(state=self.shared)}):
            result = asyncio.run(
                callback(
                    self.record,
                    {"prompt": "saved"},
                    self.record,
                    "Rendered prompt",
                    "model",
                    "edit",
                    "manual",
                    "manual negative",
                    "model",
                    512,
                    768,
                    {},
                    gr.Request(session_hash="A"),
                )
            )
        self.assertNotIn(self.editor.prompt, result)
        self.assertNotIn(self.editor.negative, result)
        self.assertNotIn("value", result[self.editor.mask])
        prompt = self.functions["select_prompt"].fn(
            {
                "prompt": "rendered",
                "negative": "rendered negative",
                "raw_prompt": "original",
                "raw_negative": "",
            },
            "Original prompt",
        )
        self.assertEqual(prompt, ("original", ""))


if __name__ == "__main__":
    unittest.main()
