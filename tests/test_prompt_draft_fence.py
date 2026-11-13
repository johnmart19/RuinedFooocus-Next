"""The final browser publication step protects edits after server validation."""

import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import threading
import unittest
import warnings

import gradio as gr
from gradio.state_holder import SessionState

from ruinedfooocus.core.paths import asset_path
from ruinedfooocus.features.prompting.ui_draft_fence import fence_event
from ruinedfooocus.features.prompting.prompt_preview import create_prompt_action


class PromptDraftFenceTests(unittest.TestCase):
    def test_prompt_ticket_apply_and_undo_round_trip_without_json_type_warnings(self):
        original = " Original draft\nwith exact spacing "
        for generated in ("Random prompt text", "One Button prompt text"):
            with self.subTest(generated=generated):
                with gr.Blocks(analytics_enabled=False) as app:
                    prompt = gr.Textbox(value=original, elem_id="prompt")
                    create_prompt_action(prompt)(lambda: generated, [])
                functions = {
                    fn.fn.__name__: fn
                    for fn in app.fns.values()
                    if fn.fn is not None
                    and fn.fn.__name__ in {"build_prompt", "publish_prompt", "undo_prompt"}
                }
                state = SessionState(app)
                guard = {"revision": 0, "serial": 1, "text": original}

                def call(name, values):
                    value = asyncio.run(app.process_api(functions[name]._id, values, state=state))[
                        "data"
                    ][0]
                    return (
                        value.model_dump() if isinstance(value, gr.data_classes.JsonData) else value
                    )

                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter("always", UserWarning)
                    ticket = call("build_prompt", [original, None, False, guard])
                    published = call("publish_prompt", [original, None, ticket])
                    undo_record = published["updates"][1]
                    undone = call("undo_prompt", [generated, undo_record, guard])
                self.assertEqual(ticket, [original, 0, generated, False, guard])
                self.assertEqual(published["updates"][:2], [generated, [original, generated]])
                self.assertEqual(undone["updates"], [original, None])
                self.assertEqual(
                    [
                        str(item.message)
                        for item in observed
                        if "JSON component received unexpected type" in str(item.message)
                    ],
                    [],
                )

    def test_real_generate_bindings_capture_before_delayed_cross_action_builds_and_reuse_at_publish(
        self,
    ):
        started_a, started_b, release_a, release_b = (threading.Event() for _ in range(4))

        def builder(started, release, text):
            started.set()
            if not release.wait(5):
                raise TimeoutError("fixture builder was not released")
            return text

        with gr.Blocks() as app:
            prompt = gr.Textbox(value="original", elem_id="prompt")
            create_prompt_action(prompt)(lambda: builder(started_a, release_a, "Random A"), [])
            create_prompt_action(prompt)(lambda: builder(started_b, release_b, "One Button B"), [])
        builds = [fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "build_prompt"]
        publishes = [
            fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "publish_prompt"
        ]
        dependencies = {item["id"]: item for item in app.config["dependencies"]}
        generation_js, publication_js = [], []
        for build, publish in zip(builds, publishes):
            generation_js.append(dependencies[build._id]["js"])
            self.assertIn("return fence.capture", generation_js[-1])
            self.assertIsNone(dependencies[publish._id]["js"])
            client = next(
                item for item in dependencies.values() if item["trigger_after"] == publish._id
            )
            publication_js.append(client["js"])
            self.assertEqual(
                dependencies[publish._id]["inputs"],
                [prompt._id, publish.inputs[1]._id, build.outputs[0]._id],
            )

        harness = r"""
const assert = require('node:assert/strict'), vm = require('node:vm');
const data=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const input={value:'original',addEventListener(){}};
const scope={document:{getElementById(id){return id==='prompt'?{querySelector:()=>input}:null}}};
const args=data.generate.map(source=>vm.runInNewContext('('+source+')',scope)('original',0,false,{}));
assert.equal(args[0].at(-1).serial,1); assert.equal(args[1].at(-1).serial,2);
if (!data.envelopes) { process.stdout.write(JSON.stringify(args)); }
else {
  const first=vm.runInNewContext('('+data.publish[0]+')',scope)(data.envelopes[0]);
  assert.equal(first[0].value,undefined); assert.equal(input.rfDraft.serial,2);
  assert.equal(input.rfDraft.revision,0); assert.equal(input.value,'original');
  const second=vm.runInNewContext('('+data.publish[1]+')',scope)(data.envelopes[1]);
  assert.equal(second[0],'One Button B'); assert.equal(input.rfDraft.serial,2);
  assert.equal(input.rfDraft.revision,1);
}
"""

        def browser(payload):
            result = subprocess.run(
                ["node", "-e", harness],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout

        arguments = json.loads(browser(dict(generate=generation_js)))
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                older = executor.submit(builds[0].fn, *arguments[0])
                newer = executor.submit(builds[1].fn, *arguments[1])
                self.assertTrue(started_a.wait(2))
                self.assertTrue(started_b.wait(2))
                release_a.set()
                first_ticket = older.result(2)
                self.assertFalse(newer.done())
                first = publishes[0].fn("original", 0, first_ticket)
                self.assertEqual(first["guard"], arguments[0][-1])
                release_b.set()
                second_ticket = newer.result(2)
                second = publishes[1].fn("original", 0, second_ticket)
                self.assertEqual(second["guard"], arguments[1][-1])
        finally:
            release_a.set()
            release_b.set()
        browser(dict(generate=generation_js, publish=publication_js, envelopes=[first, second]))

    def test_server_returns_envelope_and_only_client_success_owns_prompt_outputs(self):
        with gr.Blocks() as app:
            prompt = gr.Textbox(value="original")
            previous = gr.JSON(None, visible="hidden")
            button = gr.Button("Apply")
            fence_event(
                button.click,
                lambda current: (current + " changed", (current, current + " changed")),
                prompt=prompt,
                inputs=[prompt],
                outputs=[prompt, previous],
                history_output=1,
            )
        backend, client = list(app.fns.values())
        guard = {"revision": 2, "serial": 3, "operation": "owned", "text": "original"}
        envelope = backend.fn("original", guard)
        self.assertEqual(envelope["guard"], guard)
        self.assertEqual(envelope["updates"][0], "original changed")
        self.assertEqual(len(backend.outputs), 1)
        self.assertIsInstance(backend.outputs[0], gr.JSON)
        self.assertNotIn(prompt, backend.outputs)
        self.assertIsNone(client.fn)
        self.assertEqual(client.outputs[:2], [prompt, previous])
        self.assertTrue(app.config["dependencies"][1]["trigger_only_on_success"])

    def test_live_draft_revision_context_and_request_order_fence_actual_paint(self):
        source = asset_path("prompt_draft_fence.js").read_text(encoding="utf-8")
        harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const source = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
function browser() {
  const input = {value:'original',listeners:{},addEventListener(name,fn){this.listeners[name]=fn;}};
  const negative = {value:'negative',checked:false};
  const context = {listeners:{},querySelectorAll(){return [negative]},addEventListener(name,fn){this.listeners[name]=fn;}};
  const scope = {document:{getElementById(id){return id==='prompt'?{querySelector:()=>input}:id==='negative'?context:null}}};
  vm.runInNewContext(source+';globalThis.fence=fence;',scope);
  return {input,negative,context,fence:scope.fence};
}
const a=browser(), b=browser();
const config={prompt:0,history_input:null,history_output:1,history_revision:2,contexts:['negative']};
function begin(browser, operation='apply', extra=[]) {return browser.fence.capture('prompt',operation,[browser.input.value,...extra,{}],config).at(-1);}
function result(guard){return {guard,updates:['reviewed',['original','reviewed'],false],stale:null,message:'draft changed'};}
function edit(browser,value) {browser.input.value=value;browser.input.listeners.input();}
let guard=begin(a);
// The backend already validated "original". A later keystroke must win at paint.
edit(a,'newer manual edit');
let output=a.fence.publish('prompt',result(guard),3,config);
assert.equal(output[0].__type__,'update'); assert.equal(output[0].value,undefined);
assert.equal(output[3].value,'draft changed');
assert.equal(a.input.value,'newer manual edit');
assert.equal(b.input.value,'original');
// Restoring identical text remains a newer revision, including across tools.
guard=begin(a,'bake'); edit(a,'temporary'); edit(a,'newer manual edit');
assert.equal(a.fence.publish('prompt',result(guard),3,config)[0].value,undefined);
// A negative/context edit after validation also cannot be painted over.
guard=begin(a); a.negative.value='typed negative'; a.context.listeners.input();
assert.equal(a.fence.publish('prompt',result(guard),3,config)[0].value,undefined);
// The second request owns this operation; the first cannot replace its result.
const first=begin(b), second=begin(b);
assert.equal(b.fence.publish('prompt',result(first),3,config)[0].value,undefined);
output=b.fence.publish('prompt',result(second),3,config);
assert.equal(output[0],'reviewed');
// Random A and One Button B share the same textarea and unchanged text. B is
// still pending when A returns: A must not publish and invalidate B's revision.
const c=browser();
const randomA=begin(c,'random'), oneButtonB=begin(c,'one-button');
assert.equal(c.fence.publish('prompt',result(randomA),3,config)[0].value,undefined);
assert.equal(c.input.value,'original');
assert.equal(c.fence.publish('prompt',result(oneButtonB),3,config)[0],'reviewed');
const history=output[1];
b.input.value='reviewed';
// Apply/Undo history includes the applied browser revision. Editing away and
// back must invalidate Undo even though the final text still matches it.
edit(b,'other'); edit(b,'reviewed');
const undoConfig={...config,history_input:1,history_output:null};
const undoGuard=b.fence.capture('prompt','undo',['reviewed',history,{}],undoConfig).at(-1);
assert.equal(b.fence.publish('prompt',{guard:undoGuard,updates:['original',null],message:'draft changed'},2,undoConfig)[0].value,undefined);
// A late Random/One Button result is still reviewable rather than discarded.
guard=begin(a,'random'); edit(a,'typed while callback returned');
const pending={guard,updates:['old'],stale:[{__type__:'update'},'pending suggestion',{__type__:'update',visible:true}],message:'draft changed'};
output=a.fence.publish('prompt',pending,3,config);
assert.equal(output[1],'pending suggestion'); assert.equal(output[2].visible,true);
// Chat discards permission/assistant/model context changes, including an edit
// and restore, while ordinary tools retain their existing pending behavior.
guard=begin(a,'chat'); a.negative.value='permission revoked'; a.context.listeners.input();
output=a.fence.publish('prompt',{...pending,guard},3,{...config,discard_context_stale:true});
assert.equal(output[1].value,undefined); assert.equal(output[2].visible,undefined);
guard=begin(a,'chat'); a.negative.value='assistant changed'; a.context.listeners.input();
a.negative.value='permission revoked'; a.context.listeners.input();
assert.equal(a.fence.publish('prompt',{...pending,guard},3,{...config,discard_context_stale:true})[1].value,undefined);
guard=begin(a,'chat'); edit(a,'new manual draft');
assert.equal(a.fence.publish('prompt',{...pending,guard},3,{...config,discard_context_stale:true})[1],'pending suggestion');
"""
        result = subprocess.run(
            ["node", "-e", harness],
            input=json.dumps(source),
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
