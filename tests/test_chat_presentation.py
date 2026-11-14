"""Real Gradio presentation events use inert workers and captured identities."""

import asyncio
import json
from itertools import product
from pathlib import Path
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState

import test_chat_image_ui_events as fixtures
from ruinedfooocus.features.chat.model_preparation import runtime_selection_key
from ruinedfooocus.features.chat.ui_load_state import progress_text
from ruinedfooocus.core.translation_manager import catalogue
from ruinedfooocus.core.paths import asset_path


class ChatPresentationTests(unittest.TestCase):
    def function(self, app, name):
        return next(fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == name)

    def test_all_three_choices_are_captured_independently_and_never_overwrite_create_draft(self):
        fixture = fixtures.ChatImageUiEventsTests()
        app, _, submit, _, _ = fixture.make_ui(prompt=True)
        self.assertEqual([choice[1] for choice in submit.inputs[2].choices], ["text", "run"])
        state = None
        previous = None
        for generation, rewrite, update in product((False, True), repeat=3):
            fixture.events = [("results", "A reply")]
            state, _ = fixture.submit(
                app,
                submit,
                intent="run" if generation else "text",
                state=state,
                overrides={
                    len(submit.inputs) - 9: rewrite,
                    len(submit.inputs) - 8: update,
                    len(submit.inputs) - 7: "  exact supplied visual prompt  ",
                },
            )
            data = fixture.sent[-1]
            policy = data["_chat_tool_permissions"]
            self.assertEqual(
                (
                    data["enable_image_generation"],
                    data["rewrite_image_prompt"],
                    data["update_create_prompt"],
                ),
                (generation, rewrite, update),
            )
            self.assertEqual(policy.original_prompt, "  exact supplied visual prompt  ")
            policy.check()
            if previous is not None:
                with self.assertRaises(ValueError):
                    previous.check()
            previous = policy
            self.assertNotIn(fixture.prompt, submit.outputs)
        self.assertEqual(fixture.prompt.value, "Create draft")

    def test_combined_image_result_stages_only_the_validated_prompt_with_captured_authority(self):
        fixture = fixtures.ChatImageUiEventsTests()
        app, _, submit, _, _ = fixture.make_ui(prompt=True)
        calls = 0

        def result(_):
            nonlocal calls
            calls += 1
            policy = fixture.sent[-1]["_chat_tool_permissions"]
            return (
                (
                    "image_tool_result",
                    dict(
                        status="completed",
                        name="generate_image",
                        call_id="combined",
                        prompt_proposal={
                            "prompt": "Validated visual prompt",
                            "turn_id": policy.turn_id,
                        },
                    ),
                )
                if calls == 1
                else ("results", "Done")
            )

        fixture.worker.task_result = result
        state, replies = fixture.submit(
            app,
            submit,
            intent="run",
            overrides={
                len(submit.inputs) - 8: True,
                len(submit.inputs) - 1: {"text": "Create draft", "revision": 0, "serial": 1},
            },
        )
        ticket = next(component for component in submit.outputs if isinstance(component, gr.JSON))
        proposals = [reply["data"][submit.outputs.index(ticket)] for reply in replies]
        proposals = [getattr(proposal, "root", proposal) for proposal in proposals]
        self.assertTrue(
            any(
                isinstance(proposal, list) and proposal[2] == "Validated visual prompt"
                for proposal in proposals
            ),
            repr(proposals),
        )
        self.assertEqual(fixture.prompt.value, "Create draft")
        self.assertEqual(len(fixture.sent), 1)

    def test_runtime_progress_unknown_totals_do_not_invent_percentages(self):
        message, bar = progress_text({"phase": "downloading_runtime", "received": 1234})
        self.assertIn("0.0 MB", message)
        self.assertNotIn(' value="', bar)
        message, bar = progress_text(
            {"phase": "downloading_runtime", "transfer": {"received": 25, "total": 100}}
        )
        self.assertIn("25%", message)
        self.assertIn('value="25.0"', bar)

    def test_registered_cancellation_events_block_prevalidated_late_browser_publication(self):
        fixture = fixtures.ChatImageUiEventsTests()
        app, workspace, submit, _, _ = fixture.make_ui(prompt=True)
        dependencies = app.config["dependencies"]
        submit_row = next(row for row in dependencies if row["id"] == submit._id)
        cancels = [
            row
            for row in dependencies
            if getattr(app.fns[row["id"]].fn, "__name__", "") == "revoke_permissions"
        ]
        protected_targets = {tuple(target) for row in cancels for target in row["targets"]}
        new_conversation = self.function(app, "new_conversation")
        stop_row = next(
            row for row in dependencies if getattr(app.fns[row["id"]].fn, "__name__", "") == "stop"
        )
        self.assertIn((workspace.context._id, "change"), protected_targets)
        self.assertTrue(any(event == "click" for _, event in protected_targets))
        self.assertTrue(all(row["js"] for row in cancels))
        new_row = next(row for row in dependencies if row["id"] == new_conversation._id)
        self.assertTrue(set(map(tuple, new_row["targets"])).issubset(protected_targets))
        self.assertTrue(set(map(tuple, stop_row["targets"])).issubset(protected_targets))
        publication = self.function(app, "publish_chat_prompt")
        client = next(row for row in dependencies if row["trigger_after"] == publication._id)
        apply = self.function(app, "apply_chat_prompt")
        apply_row = next(row for row in dependencies if row["id"] == apply._id)
        apply_client = next(row for row in dependencies if row["trigger_after"] == apply._id)
        harness = r"""
const vm=require('node:vm'), assert=require('node:assert/strict');
const data=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
for (const cancel of [null,...data.cancels]) {
  const input={value:'Create draft',addEventListener(){}};
  const nodes=new Map();
  const scope={document:{
    getElementById(id){if(!nodes.has(id)) nodes.set(id,{querySelector:()=>id===data.prompt?input:null,querySelectorAll:()=>[],addEventListener(){}});return nodes.get(id)},
    querySelector(){return {dataset:{attachmentKey:'owned'}}}
  }};
  const args=vm.runInNewContext('('+data.capture+')',scope)(...Array(data.inputCount).fill(null));
  const guard=args.at(-1);
  const envelope={guard,updates:['late prompt',null,'',null,null],stale:[null,null,'late prompt',{visible:true},null],message:'draft changed'};
  if(cancel) vm.runInNewContext('('+cancel+')',scope)(null);
  const painted=vm.runInNewContext('('+data.publish+')',scope)(envelope);
  if(cancel) {assert.equal(painted[0]?.value,undefined);assert.notEqual(painted[2],'late prompt');assert.equal(input.rfDraft.serial,2)}
  else assert.equal(painted[0],'late prompt');
  // Apply can queue before server-side revoke finishes. Its newly captured
  // serial must keep the original ticket's revoked authority epoch.
  const ticket=['Create draft',0,'late prompt',false,guard,'captured-turn'];
  const applyValues=Array(data.applyInputCount).fill(null); applyValues[0]='Create draft';applyValues[1]=ticket;
  const applied=vm.runInNewContext('('+data.applyCapture+')',scope)(...applyValues);
  const applyEnvelope={guard:applied.at(-1),updates:envelope.updates};
  const applyPaint=vm.runInNewContext('('+data.applyPublish+')',scope)(applyEnvelope);
  if(cancel) {assert.equal(applied.at(-1).revoked,true);assert.equal(applyPaint[0]?.value,undefined)}
  else assert.equal(applyPaint[0],'late prompt');
  assert.equal(input.value,'Create draft');
}
"""
        result = subprocess.run(
            ["node", "-e", harness],
            input=json.dumps(
                dict(
                    capture=submit_row["js"],
                    cancels=[row["js"] for row in cancels],
                    publish=client["js"],
                    prompt=fixture.prompt.elem_id,
                    inputCount=len(submit.inputs),
                    applyCapture=apply_row["js"],
                    applyPublish=apply_client["js"],
                    applyInputCount=len(apply.inputs),
                )
            ),
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_registered_load_streams_exact_identity_and_selection_changes_fence_old_completion(
        self,
    ):
        for superseded in (False, True, "cancel", "cancel_verification"):
            with self.subTest(superseded=superseded), tempfile.TemporaryDirectory() as temporary:
                model = Path(temporary) / "Fixture-Q4.gguf"
                model.write_bytes(b"inert metadata fixture")
                fixture = fixtures.ChatImageUiEventsTests()
                app, _, _, _, _ = fixture.make_ui(catalogue={"Fixture": [("Q4", str(model))]})
                load = self.function(app, "use_model")
                scope = load.fn.__globals__
                values = dict(scope["settings"].default_settings)
                snapshot = dict(
                    load_id="accepted-fixture-instance",
                    selection_key=runtime_selection_key(model, values),
                    model_identity={"path": str(model), "quantization": "Q4"},
                    runtime_identity={"tag": "fixture-build", "backend": "CPU"},
                    placement={"state": "mixed"},
                )
                fixture.worker.task_result = Mock(
                    side_effect=lambda _: (
                        (
                            "runtime_progress",
                            {
                                "load_id": fixture.sent[-1]["_runtime_load_id"],
                                "phase": "probing_device",
                                "message": "Fixture probe",
                            },
                        )
                        if fixture.worker.task_result.call_count == 1
                        else ("results", {"ready": True, "runtime_snapshot": snapshot})
                    )
                )
                state = SessionState(app)
                session = state[load.inputs[-1]._id]
                session.select(str(model))
                state[load.inputs[0]._id] = str(model)
                state[load.inputs[1]._id] = "Fixture"
                replies = []

                async def run():
                    iterator = None
                    while True:
                        reply = await app.process_api(
                            load,
                            [str(model), "Fixture", "Local models", "Load", None],
                            state=state,
                            iterator=iterator,
                            request=gr.Request(headers={"accept-language": "uk"}),
                        )
                        replies.append(reply)
                        if superseded == "cancel_verification" and "Verifying candidate" in str(
                            reply["data"]
                        ):
                            await app.process_api(
                                self.function(app, "cancel_model_load"), [None, None], state=state
                            )
                        if superseded and "Fixture probe" in str(reply["data"]):
                            if superseded == "cancel":
                                await app.process_api(
                                    self.function(app, "cancel_model_load"),
                                    [None, None],
                                    state=state,
                                )
                            else:
                                session.select("new-selected.gguf")
                        if not reply["is_generating"]:
                            break
                        iterator = reply["iterator"]

                with (
                    patch.dict(
                        catalogue().translations["uk"],
                        {"Observed placement: {placement}": "Спостережене розміщення: {placement}"},
                    ),
                    patch.dict(
                        scope,
                        capture_model=lambda *args, **kwargs: SimpleNamespace(
                            model=model, present=True, settings=values
                        ),
                        ModelTransfer=lambda _: SimpleNamespace(result=lambda: None),
                        initialized_runtime_status=lambda: dict(
                            active_model=None if superseded else str(model),
                            runtime_snapshot=None if superseded else snapshot,
                        ),
                    ),
                ):
                    asyncio.run(run())
                wire = str([reply["data"] for reply in replies])
                if superseded == "cancel_verification":
                    self.assertEqual(fixture.sent, [])
                    self.assertIn("Loading cancelled. You can retry.", wire)
                    self.assertNotIn("Stopping…", str(replies[-2]["data"]))
                else:
                    self.assertIn("Fixture probe", wire)
                if superseded:
                    self.assertNotIn("'value': 'Loaded'", wire)
                    self.assertIsNone(session.initialized_runtime)
                else:
                    self.assertIn("'value': 'Loaded'", wire)
                    self.assertIn("fixture-build", wire)
                    from ruinedfooocus.core.translation_manager import localize

                    # The request locale reaches server-built text, translated or not.
                    self.assertIn(
                        localize("Observed placement: {placement}", "uk", placement="mixed"),
                        wire,
                    )
                    self.assertNotIn("{placement}", wire)
                    self.assertNotIn("__i18n__", wire)
                    self.assertEqual(session.initialized_runtime, snapshot)
                self.assertIsNone(session.load_operation)
                if superseded == "cancel":
                    fixture.worker.interrupt_processing.assert_called_once_with(1)
                else:
                    fixture.worker.interrupt_processing.assert_not_called()

    def test_cancelled_same_file_reload_keeps_only_the_previously_accepted_native_instance(self):
        for resident_case in (
            "previous_instance",
            "new_context_instance",
            "new_identical_config_instance",
        ):
            with self.subTest(resident=resident_case), tempfile.TemporaryDirectory() as temporary:
                model = Path(temporary) / "Same-Q4.gguf"
                model.write_bytes(b"inert fixture; no native load")
                fixture = fixtures.ChatImageUiEventsTests()
                app, _, submit, _, _ = fixture.make_ui(catalogue={"Fixture": [("Q4", str(model))]})
                load = self.function(app, "use_model")
                scope = load.fn.__globals__
                previous_values = dict(scope["settings"].default_settings, llm_n_ctx=2048)
                current_values = dict(
                    previous_values,
                    llm_n_ctx=2048 if resident_case == "new_identical_config_instance" else 4096,
                )
                accepted = dict(
                    load_id="accepted-native-instance",
                    selection_key=runtime_selection_key(model, previous_values),
                    model_identity={"path": str(model), "quantization": "Q4"},
                    runtime_identity={"tag": "fixture-build", "backend": "CPU"},
                    requested={"context": 2048},
                    effective={"context": 2048},
                    placement={"state": "unknown"},
                )
                preserved = resident_case == "previous_instance"
                resident = (
                    dict(accepted, placement={"state": "mixed"})
                    if preserved
                    else dict(
                        accepted,
                        load_id="late-cancelled-native-instance",
                        selection_key=runtime_selection_key(model, current_values),
                        requested={"context": current_values["llm_n_ctx"]},
                        effective={"context": current_values["llm_n_ctx"]},
                    )
                )
                fixture.worker.task_result = Mock(
                    side_effect=lambda _: (
                        (
                            "runtime_progress",
                            {
                                "load_id": fixture.sent[-1]["_runtime_load_id"],
                                "phase": "initializing_model",
                                "message": "Cancel checkpoint",
                            },
                        )
                        if fixture.worker.task_result.call_count == 1
                        else ("results", {"ready": True, "runtime_snapshot": resident})
                    )
                )
                state = SessionState(app)
                session = state[load.inputs[-1]._id]
                session.select(str(model))
                session.active, session.initialized_runtime = str(model), accepted
                state[load.inputs[0]._id], state[load.inputs[1]._id] = str(model), "Fixture"
                state[submit.inputs[7]._id] = str(model)
                replies = []
                gate = next(
                    fn
                    for fn in app.fns.values()
                    if fn.inputs == [submit.inputs[7]] and fn.outputs == [submit.outputs[9]]
                )

                async def run():
                    iterator = None
                    while True:
                        reply = await app.process_api(
                            load,
                            [None, None, "Local models", "Load", None],
                            state=state,
                            iterator=iterator,
                        )
                        replies.append(reply)
                        if "Cancel checkpoint" in str(reply["data"]):
                            await app.process_api(
                                self.function(app, "cancel_model_load"), [None, None], state=state
                            )
                        if not reply["is_generating"]:
                            return await app.process_api(gate, [None], state=state)
                        iterator = reply["iterator"]

                with (
                    patch.dict(scope["settings"].default_settings, current_values),
                    patch.dict(
                        scope,
                        capture_model=lambda *args, **kwargs: SimpleNamespace(
                            model=model, present=True, settings=current_values
                        ),
                        ModelTransfer=lambda _: SimpleNamespace(result=lambda: None),
                        initialized_runtime_status=lambda: dict(
                            active_model=str(model), runtime_snapshot=resident
                        ),
                    ),
                ):
                    eligibility = asyncio.run(run())
                self.assertEqual(session.active, str(model) if preserved else None)
                self.assertEqual(state[submit.inputs[7]._id], str(model) if preserved else None)
                self.assertEqual(session.initialized_runtime, resident if preserved else None)
                self.assertEqual(eligibility["data"][0]["interactive"], preserved)
                self.assertNotIn("'value': 'Loaded'", str([reply["data"] for reply in replies]))
                self.assertIsNone(session.load_operation)
                self.assertEqual(accepted["load_id"], "accepted-native-instance")
                fixture.worker.interrupt_processing.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
