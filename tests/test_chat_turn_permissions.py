"""Independent captured permissions and actual fenced prompt callbacks; inert compute."""

import asyncio
from copy import deepcopy
from dataclasses import replace, FrozenInstanceError
from itertools import product
import json
import subprocess
from types import SimpleNamespace
from threading import Event
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState

from ruinedfooocus.app import api_runtime
from ruinedfooocus.features.chat.attachments import attachment_capabilities
from ruinedfooocus.features.chat.document_extraction import document_capabilities
from ruinedfooocus.features.chat.image_tool_requests import (
    ChatPermissionSession,
    ImageToolContext,
    ImageToolValidationError,
    check_tool_permissions,
    create_prompt_proposal,
    dispatch_image_tool,
    image_tool_schemas,
    prompt_tool_result,
    validate_image_tool_call,
)
from ruinedfooocus.features.chat.llama_vision import reference_capabilities
from ruinedfooocus.features.prompting.prompt_preview import create_chat_prompt_action
from ui_test_helpers import function_from

OWNER = ("ui", "permissions", "browser")
CAPABILITY = {"available": True, "binding": "inert-initialized-template"}


def call(name="generate_image", prompt="A new bounded visual prompt"):
    return {
        "id": "call_permissions",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps({"prompt": prompt})},
    }


def captured(session, **choices):
    policy = session.capture(OWNER, **choices)
    return ImageToolContext(
        OWNER, "run" if policy.enable_image_generation else "text", {}, permissions=policy
    )


class TurnPermissionsTests(unittest.TestCase):
    def test_all_eight_combinations_keep_three_independent_authorities(self):
        for generation, rewrite, update in product((False, True), repeat=3):
            with self.subTest(generation=generation, rewrite=rewrite, update=update):
                context = captured(
                    ChatPermissionSession(),
                    enable_image_generation=generation,
                    rewrite_image_prompt=rewrite,
                    update_create_prompt=update,
                )
                names = {
                    tool["function"]["name"] for tool in image_tool_schemas(context, CAPABILITY)
                }
                self.assertEqual("generate_image" in names, generation)
                self.assertEqual("propose_create_prompt" in names, update)
                self.assertEqual(context.permissions.rewrite_image_prompt, rewrite)
                if generation:
                    image_request = validate_image_tool_call(call(), context, CAPABILITY)
                    check_tool_permissions(image_request)
                    proposal = create_prompt_proposal(image_request)
                    self.assertEqual(proposal is not None, update)
                    if update:
                        self.assertEqual(proposal["prompt"], image_request.arguments["prompt"])
                else:
                    with self.assertRaises(ImageToolValidationError):
                        validate_image_tool_call(call(), context, CAPABILITY)
                if update:
                    request = validate_image_tool_call(
                        call("propose_create_prompt"), context, CAPABILITY
                    )
                    self.assertEqual(
                        prompt_tool_result(request)["prompt_proposal"]["prompt"],
                        "A new bounded visual prompt",
                    )

    def test_default_off_strict_booleans_and_immutable_snapshot(self):
        session = ChatPermissionSession()
        policy = session.capture(OWNER)
        self.assertEqual(
            image_tool_schemas(ImageToolContext(OWNER, "run", {}, permissions=policy), CAPABILITY),
            [],
        )
        self.assertIs(deepcopy(policy), policy)
        with self.assertRaises(FrozenInstanceError):
            policy.enable_image_generation = True
        with self.assertRaises(ValueError):
            session.capture(OWNER, enable_image_generation="true")
        with self.assertRaises(ImageToolValidationError):
            replace(policy, enable_image_generation=True).check()

    def test_exact_supplied_target_survives_model_rewrite_and_new_target_is_not_chat_prose(self):
        original = '  Exactly two blue cups with "A & B"\nKeep spacing.  '
        for rewrite in (False, True):
            context = captured(
                ChatPermissionSession(),
                enable_image_generation=True,
                rewrite_image_prompt=rewrite,
                update_create_prompt=True,
                prompt_target="supplied",
                original_prompt=original,
            )
            for name in ("generate_image", "propose_create_prompt"):
                request = validate_image_tool_call(
                    call(name, "Model tried a different prompt"), context, CAPABILITY
                )
                self.assertEqual(
                    request.arguments["prompt"],
                    "Model tried a different prompt" if rewrite else original,
                )
                if not rewrite:
                    with self.assertRaises(ImageToolValidationError):
                        check_tool_permissions(replace(request, arguments={"prompt": "tampered"}))
        context = captured(ChatPermissionSession(), enable_image_generation=True)
        self.assertEqual(
            validate_image_tool_call(call(), context, CAPABILITY).arguments["prompt"],
            "A new bounded visual prompt",
        )
        with self.assertRaises(ValueError):
            ChatPermissionSession().capture(OWNER, original_prompt="unrelated old prompt")

    def test_queued_dispatch_retry_and_final_publication_check_revocation(self):
        session = ChatPermissionSession()
        context = captured(session, enable_image_generation=True)
        request = validate_image_tool_call(call(), context, CAPABILITY)
        service = Mock(return_value={"status": "completed"})
        session.revoke(OWNER)
        result = dispatch_image_tool(request, service)
        self.assertEqual(result["error"]["code"], "revoked_permissions")
        service.assert_not_called()
        self.assertEqual(image_tool_schemas(context, CAPABILITY), [])
        context = captured(session, enable_image_generation=True)
        request = validate_image_tool_call(call(), context, CAPABILITY)

        def late_service(_request):
            session.revoke(OWNER)
            return {"status": "completed"}

        self.assertEqual(dispatch_image_tool(request, late_service)["status"], "error")

    def test_new_turn_and_cross_browser_capture_do_not_reinterpret_old_permissions(self):
        session = ChatPermissionSession()
        old = session.capture(OWNER, enable_image_generation=True)
        other = session.capture(("ui", "other", "browser"), update_create_prompt=True)
        old.check()
        session.capture(OWNER, update_create_prompt=True)
        with self.assertRaises(ImageToolValidationError):
            old.check()
        other.check()
        with self.assertRaises(ValueError):
            ImageToolContext(OWNER, "text", {}, permissions=other)

    def test_api_capture_migration_ignores_legacy_preference_and_new_flags_stay_off(self):
        with patch.object(api_runtime, "image_defaults", {}):
            default = api_runtime.capture_image_tool_context(
                {"enable_llm_tools": True}, owner=OWNER
            )
            self.assertEqual(image_tool_schemas(default, CAPABILITY), [])
            legacy = api_runtime.capture_image_tool_context(
                {"image_tool_intent": "run"}, owner=OWNER
            )
            self.assertTrue(legacy.permissions.enable_image_generation)
            self.assertFalse(legacy.permissions.rewrite_image_prompt)
            self.assertFalse(legacy.permissions.update_create_prompt)
            denied = api_runtime.capture_image_tool_context(
                {"image_tool_intent": "run", "enable_image_generation": False}, owner=OWNER
            )
            self.assertEqual(image_tool_schemas(denied, CAPABILITY), [])

    def test_worker_captures_policy_without_preference_upgrade_or_image_assets_for_prompt_only(
        self,
    ):
        class StopAtRegistry(Exception):
            pass

        scope = dict(
            settings=SimpleNamespace(default_settings={"enable_llm_tools": True}),
            shared=SimpleNamespace(path_manager=object()),
            registry=SimpleNamespace(create=Mock(side_effect=StopAtRegistry)),
        )
        add_task = function_from("ruinedfooocus/app/async_worker.py", "add_task", scope)
        for choices in ({}, {"update_create_prompt": True}, {"enable_image_generation": True}):
            observed = []

            def capture(data, *_args):
                observed.append(data)
                return data

            policy = ChatPermissionSession().capture(OWNER, **choices)
            with (
                patch(
                    "ruinedfooocus.features.chat.model_preparation.capture_chat_request",
                    side_effect=capture,
                ),
                patch("ruinedfooocus.features.chat.llama_models.local_models", return_value=[]),
                patch.object(
                    api_runtime, "_image_runtime_snapshot", return_value="snapshot"
                ) as snapshot,
            ):
                with self.assertRaises(StopAtRegistry):
                    add_task(
                        {
                            "task_type": "llama",
                            "_chat_tool_permissions": policy,
                            "image_settings": {"prompt": "unrelated Create draft"},
                        }
                    )
            self.assertEqual(
                "image_settings" in observed[0], bool(choices.get("enable_image_generation"))
            )
            self.assertEqual(snapshot.call_count, int(bool(choices.get("enable_image_generation"))))

    def test_explicit_rf_image_api_retains_authority_and_prompt_only_api_never_prepares_images(
        self,
    ):
        pipeline = SimpleNamespace(image_tool_capability=lambda: CAPABILITY, vision_projector=None)
        with (
            patch("ruinedfooocus.app.resource_lifecycle.service_pipeline", return_value=pipeline),
            patch.object(api_runtime, "image_defaults", {}),
            patch.object(api_runtime, "_image_runtime_snapshot", return_value="inert"),
            patch.object(api_runtime, "submit", side_effect=lambda operation: operation) as submit,
            patch.object(
                api_runtime,
                "_submit_passive_image_tool",
                side_effect=lambda operation, request: operation,
            ) as passive,
            patch.object(api_runtime, "prepare_chat_image_tool") as prepare,
        ):
            explicit = api_runtime.image_tool({"intent": "run", "call": call()}, owner=OWNER)
            self.assertEqual(explicit.resource_group, "image")
            submit.assert_called_once_with(explicit)
            passive.assert_not_called()
            with self.assertRaises(ImageToolValidationError):
                api_runtime.image_tool(
                    {"intent": "run", "enable_image_generation": False, "call": call()}, owner=OWNER
                )
            prompt_only = api_runtime.image_tool(
                {
                    "intent": "text",
                    "enable_image_generation": False,
                    "update_create_prompt": True,
                    "call": call("propose_create_prompt"),
                },
                owner=OWNER,
            )
            submit.assert_called_once_with(explicit)
            passive.assert_called_once()
            self.assertIs(passive.call_args.args[0], prompt_only)
            request = passive.call_args.args[1]
            self.assertEqual((request.name, request.intent), ("propose_create_prompt", "text"))
            self.assertEqual(prompt_only.resource_group, "chat")
            result = prompt_only(SimpleNamespace(cancelled=Event()), 1)
            self.assertEqual(result["prompt_proposal"]["prompt"], "A new bounded visual prompt")
            prepare.assert_not_called()

    def test_attachment_projection_matches_configured_parser_eligibility_and_vision_is_independent(
        self,
    ):
        for available in (False, True):
            documents = {
                "suffixes": (".pdf", ".docx") if available else (".docx",),
                "pdf_available": available,
                "pdf_reason": "" if available else "missing",
                "notes": "bounded",
            }
            with patch(
                "ruinedfooocus.features.chat.document_extraction.document_capabilities",
                return_value=documents,
            ):
                projection = attachment_capabilities()
            self.assertEqual(".pdf" in projection["suffixes"], available)
            self.assertEqual(projection["accept"], ",".join(projection["suffixes"]))
            self.assertEqual(projection["limits"]["files"], 4)
        with patch("ruinedfooocus.features.chat.document_extraction.version", return_value="wrong"):
            self.assertNotIn(".pdf", document_capabilities()["suffixes"])
        vision = reference_capabilities(
            vision_supported=True,
            source="Uploaded image",
            source_available=True,
            projector_required=True,
        )
        self.assertTrue(vision["visible"])
        self.assertTrue(vision["upload_visible"])
        self.assertFalse(vision["denoise_visible"])
        self.assertIn("projector", vision["reason"])
        self.assertFalse(reference_capabilities()["visible"])


class ChatDraftCallerTests(unittest.TestCase):
    def setUp(self):
        self.session = ChatPermissionSession()
        self.policy = self.session.capture(OWNER, update_create_prompt=True)
        with gr.Blocks(analytics_enabled=False) as self.app:
            self.prompt = gr.Textbox(value="original")
            self.controls = create_chat_prompt_action(
                self.prompt, owner_for_request=lambda _request: OWNER
            )
        self.guard = {"text": "original", "revision": 2, "serial": 3, "contexts": []}
        request = validate_image_tool_call(
            call("propose_create_prompt", "candidate"),
            ImageToolContext(OWNER, "text", {}, permissions=self.policy),
            CAPABILITY,
        )
        self.ticket = self.controls.stage(prompt_tool_result(request), self.policy, self.guard)
        self.functions = {fn.fn.__name__: fn for fn in self.app.fns.values() if fn.fn is not None}

    def invoke(self, name, values):
        state = SessionState(self.app)
        state[self.controls.permissions._id] = self.policy
        result = asyncio.run(self.app.process_api(self.functions[name]._id, values, state=state))[
            "data"
        ][0]
        return result.model_dump() if isinstance(result, gr.data_classes.JsonData) else result

    def test_actual_registered_publish_keeps_changed_draft_and_apply_undo_are_explicit(self):
        published = self.invoke("publish_chat_prompt", ["original", self.ticket, None])
        self.assertEqual(published["updates"][:2], ["candidate", ["original", "candidate"]])
        self.assertEqual(published["guard"], self.guard)
        pending = self.invoke("publish_chat_prompt", ["manual draft", self.ticket, None])
        self.assertEqual(pending["updates"][0], {"__type__": "update"})
        self.assertEqual(pending["updates"][2], "candidate")
        applied = self.invoke(
            "apply_chat_prompt",
            [
                "manual draft",
                self.ticket,
                None,
                {"text": "manual draft", "revision": 4, "serial": 4},
            ],
        )
        self.assertEqual(applied["updates"][:2], ["candidate", ["manual draft", "candidate"]])
        undone = self.invoke(
            "undo_chat_prompt",
            [
                "candidate",
                ["manual draft", "candidate", 5],
                {"text": "candidate", "revision": 5, "serial": 5},
            ],
        )
        self.assertEqual(undone["updates"], ["manual draft", None])
        # Refusals return to the browser step, which applies nothing and shows why.
        refused = self.functions["undo_chat_prompt"].fn(
            "new manual edit", ["manual draft", "candidate", 5], self.guard
        )
        self.assertIsNone(refused["updates"])
        self.assertIn("overwrite", refused["error"])

    def test_revoked_pending_or_superseded_turn_cannot_apply(self):
        self.session.revoke(OWNER)
        for name, values in (
            ("publish_chat_prompt", ["original", self.ticket, self.policy]),
            ("apply_chat_prompt", ["original", self.ticket, self.policy, self.guard]),
        ):
            with self.subTest(name=name):
                refused = self.functions[name].fn(*values)
                self.assertIsNone(refused["updates"])
                self.assertTrue(refused["error"])
        self.assertEqual(self.prompt.value, "original")

    def test_dismiss_captures_new_serial_and_apply_rechecks_live_context(self):
        dismiss = self.functions["dismiss_chat_prompt"]
        dependencies = {item["id"]: item for item in self.app.config["dependencies"]}
        self.assertIn("return fence.capture", dependencies[dismiss._id]["js"])
        envelope = dismiss.fn("original", self.guard)
        self.assertIsNone(envelope["updates"][-1])
        apply = self.functions["apply_chat_prompt"]
        client = next(item for item in dependencies.values() if item["trigger_after"] == apply._id)
        self.assertIn('"discard_context_stale": true', client["js"])

    def test_registered_stop_and_new_conversation_invalidate_after_validation_before_paint(self):
        def stop_pending():
            self.session.revoke(OWNER)

        with self.app:
            for label in ("Stop reply", "New conversation"):
                button = gr.Button(label)
                button.click(stop_pending, js=self.controls.invalidate_js, queue=False)
        envelope = self.invoke("apply_chat_prompt", ["original", self.ticket, None, self.guard])
        self.assertEqual(envelope["updates"][0], "candidate")
        dependencies = self.app.config["dependencies"]
        cancellation_ids = {fn._id for fn in self.app.fns.values() if fn.fn is stop_pending}
        cancellations = [item["js"] for item in dependencies if item["id"] in cancellation_ids]
        apply_id = self.functions["apply_chat_prompt"]._id
        begin = next(item["js"] for item in dependencies if item["id"] == apply_id)
        publish = next(item["js"] for item in dependencies if item["trigger_after"] == apply_id)
        harness = r"""
const assert=require('node:assert/strict'), vm=require('node:vm');
const data=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const guards=[];
for (const [index,cancelSource] of data.cancellations.entries()) {
  const input={value:'original',rfDraft:{revision:2,serial:3,text:'original'},addEventListener(){}};
  const scope={document:{getElementById(id){return id===data.prompt?{querySelector:()=>input}:null}}};
  const cancel=vm.runInNewContext('('+cancelSource+')',scope);
  const publish=vm.runInNewContext('('+data.publish+')',scope);
  const begin=vm.runInNewContext('('+data.begin+')',scope);
  const values=cancel('unchanged',7);
  assert.equal(values[0],'unchanged'); assert.equal(values[1],7);
  assert.equal(input.rfDraft.serial,4); assert.equal(input.rfDraft.revocation,1);
  assert.equal(input.value,'original');
  const painted=publish(data.envelope);
  assert.equal(painted[0].value,undefined); assert.equal(painted[0].__type__,'update');
  assert.equal(input.value,'original'); assert.equal(input.rfDraft.revision,2);
  // Apply is clicked before the cancellation reaches the server. It captures a
  // newer serial but cannot renew authority from the original revoked ticket.
  const fresh=begin('original',data.ticket,null,{}).at(-1);
  assert.equal(fresh.serial,5); assert.equal(fresh.revocation,1); assert.equal(fresh.revoked,true);
  guards.push(fresh);
  if (data.fresh_envelopes) {
    const freshPaint=publish(data.fresh_envelopes[index]);
    assert.equal(freshPaint[0].value,undefined);
    assert.equal(input.value,'original'); assert.equal(input.rfDraft.revision,2);
  }
}
process.stdout.write(JSON.stringify(guards));
"""

        def browser(**extra):
            result = subprocess.run(
                ["node", "-e", harness],
                input=json.dumps(
                    {
                        "cancellations": cancellations,
                        "begin": begin,
                        "publish": publish,
                        "envelope": envelope,
                        "ticket": self.ticket,
                        "prompt": self.prompt.elem_id,
                        **extra,
                    }
                ),
                text=True,
                capture_output=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

        fresh_guards = browser()
        # The server has not processed Stop yet and would still authorize Apply.
        self.policy.check()
        fresh_envelopes = [
            self.invoke("apply_chat_prompt", ["original", self.ticket, None, guard])
            for guard in fresh_guards
        ]
        self.assertTrue(all(result["updates"][0] == "candidate" for result in fresh_envelopes))
        browser(fresh_envelopes=fresh_envelopes)
        stop_pending()
        with self.assertRaises(ImageToolValidationError):
            self.policy.check()
        self.assertEqual(len(cancellations), 2)


if __name__ == "__main__":
    unittest.main()
