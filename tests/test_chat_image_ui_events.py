"""Actual Chat Gradio events carry explicit image authority to an inert worker."""

import asyncio
from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState
from gradio.utils import get_upload_folder
from PIL import Image


class ChatImageUiEventsTests(unittest.TestCase):
    def make_ui(self, *, catalogue=None, prompt=False):
        self.sent = []
        self.events = [("results", "ordinary reply")]
        self.worker = SimpleNamespace(
            add_task=lambda data: self.sent.append(deepcopy(data)) or len(self.sent),
            task_result=lambda _: self.events.pop(0),
            interrupt_processing=Mock(return_value=True),
        )
        with patch.dict(sys.modules, {"ruinedfooocus.app.async_worker": self.worker}):
            source = Path(__file__).resolve().parents[1] / "ruinedfooocus/features/chat/ui.py"
            spec = importlib.util.spec_from_file_location("_rf_chat_image_ui_fixture", source)
            ui = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ui)
        with gr.Blocks(analytics_enabled=False) as app:
            context = gr.State(None)
            workspace = SimpleNamespace(
                context=context,
                image_context=Mock(return_value=None),
                result_context=Mock(
                    return_value={"session_id": "session", "revision": 1, "source_id": "source"}
                ),
                bind_chat_open=Mock(),
            )
            image_prompt = gr.Textbox(value="saved negative", label="Captured negative")
            self.prompt = (
                gr.Textbox(value="Create draft", label="Create prompt") if prompt else None
            )
            with (
                patch.object(ui, "worker", self.worker),
                patch.object(ui, "local_models", return_value={}),
                patch.object(ui, "model_catalogue", return_value=catalogue or {}),
            ):
                ui.create_chat(
                    {"negative": image_prompt},
                    editor_workspace=workspace,
                    prompt_control=self.prompt,
                )
        ui.worker = self.worker
        submit = next(
            fn
            for fn in app.fns.values()
            if fn.fn is not None and fn.fn.__name__ == "submit_with_attachments"
        )
        result_button, result_state = workspace.bind_chat_open.call_args.args
        return app, workspace, submit, result_button, result_state

    def submit(
        self,
        app,
        function,
        *,
        intent="text",
        state=None,
        mode="New image",
        overrides=None,
        after_step=None,
    ):
        state = state or SessionState(app)
        values = [component.value for component in function.inputs]
        values[0:4] = [None, "A literal message", "run" if intent == "preview" else intent, None]
        if intent == "preview":
            function = next(
                fn
                for fn in app.fns.values()
                if fn.fn is not None and fn.fn.__name__ == "preview_with_attachments"
            )
        values[6:8] = [[], "fixture.gguf"]
        values[12] = mode
        for index, value in (overrides or {}).items():
            values[index] = value
        for component, value in zip(function.inputs, values):
            if component.stateful:
                state[component._id] = value

        async def run():
            iterator = None
            replies = []
            while True:
                reply = await app.process_api(
                    function,
                    values,
                    state=state,
                    request=gr.Request(session_hash="chat-fixture"),
                    iterator=iterator,
                )
                replies.append(reply)
                if after_step is not None:
                    await after_step(state, reply)
                if not reply["is_generating"]:
                    return replies
                iterator = reply["iterator"]

        return state, asyncio.run(run())

    def test_text_default_and_native_event_preserve_the_existing_chat_transport(self):
        app, workspace, function, button, result = self.make_ui()
        radio = function.inputs[2]
        self.assertEqual(radio.value, "text")
        self.assertIn(workspace.context, function.inputs)
        self.assertNotIn(workspace.context, function.outputs)
        self.submit(app, function)
        self.assertEqual(self.sent[0]["image_tool_intent"], "text")
        self.assertFalse(self.sent[0]["force_image"])
        self.assertFalse(self.sent[0]["enable_image_tools"])
        self.assertEqual(self.sent[0]["image_settings"], {"negative": "saved negative"})
        self.assertEqual(self.sent[0]["history"][-1]["content"], "A literal message")
        workspace.image_context.assert_not_called()
        self.assertFalse(button.interactive)

    def test_preview_native_terminal_keeps_readiness_beside_the_action_without_source_publication(
        self,
    ):
        app, workspace, function, button, result = self.make_ui()
        terminal = dict(
            call_id="preview-call",
            name="generate_image",
            status="preview",
            readiness={"missing": [{"name": "Selected image model"}]},
        )
        self.events = [
            ("download", None),
            ("image_tool_result", terminal),
            ("results", "Preview finished"),
        ]
        state, replies = self.submit(app, function, intent="preview")
        self.assertEqual(self.sent[0]["image_tool_intent"], "preview")
        self.assertTrue(self.sent[0]["enable_image_tools"])
        self.assertEqual(state[result._id], terminal)
        self.assertNotIn("editor_context", self.sent[0])
        workspace.image_context.assert_called_once()
        workspace.result_context.assert_not_called()
        self.assertIn("Selected image model", str([reply["data"] for reply in replies]))
        self.assertEqual(
            state[function.inputs[2]._id]
            if function.inputs[2].stateful
            else function.inputs[2].value,
            "text",
        )
        self.assertNotIn(workspace.context, function.outputs)

    def test_run_keeps_only_public_typed_result_and_registers_deliberate_open(self):
        app, workspace, function, button, result = self.make_ui()
        terminal = dict(
            call_id="run-call",
            name="generate_image",
            status="completed",
            editor={"session_id": "owned", "revision": 2, "current": {"artifact_id": "image"}},
            private="not public",
        )
        self.events = [("image_tool_result", terminal), ("results", "Image finished")]
        state, replies = self.submit(app, function, intent="run")
        self.assertEqual(self.sent[0]["image_tool_intent"], "run")
        self.assertNotIn("private", state[result._id])
        self.assertEqual(state[result._id]["editor"]["revision"], 2)
        workspace.result_context.assert_called_once()
        self.assertEqual(
            workspace.result_context.call_args.kwargs["request"].session_hash, "chat-fixture"
        )
        output_index = function.outputs.index(button)
        self.assertTrue(
            any(
                reply["data"][output_index].get("interactive") is True
                for reply in replies
                if isinstance(reply["data"][output_index], dict)
            )
        )
        self.assertNotIn(workspace.context, function.outputs)

    def test_previous_reference_uses_owned_terminal_context_and_keeps_create_values(self):
        app, workspace, function, _, _ = self.make_ui()
        owned = dict(
            session_id="owned",
            revision=4,
            source_id="pixel",
            mask_id="mask",
            additional_image_ids=["background"],
        )
        workspace.result_context.return_value = owned
        workspace.image_context.return_value = owned
        previous = dict(status="completed", editor={"session_id": "owned", "revision": 4})
        self.submit(
            app,
            function,
            intent="run",
            mode="Previous generated image",
            overrides={17: {"session_id": "old"}, 18: previous},
        )
        self.assertEqual(self.sent[0]["editor_context"], owned)
        self.assertEqual(self.sent[0]["image_settings"], {"negative": "saved negative"})
        self.assertIsNone(self.sent[0]["image_reference"])
        self.assertEqual(
            workspace.image_context.call_args.args[:2], (owned, "Previous generated image")
        )

    def test_foreign_or_replaced_terminal_never_enables_open_or_publishes_old_chat_images(self):
        for error in (PermissionError("another owner"), ValueError("replaced source")):
            with self.subTest(error=error):
                app, workspace, function, button, result = self.make_ui()
                workspace.result_context.side_effect = error
                self.events = [
                    (
                        "image_tool_result",
                        dict(
                            call_id="old",
                            name="edit_image",
                            status="completed",
                            editor={"session_id": "stale", "revision": 1},
                        ),
                    ),
                    ("results", "![Image](old-image.png)"),
                ]
                state, replies = self.submit(app, function, intent="run")
                self.assertEqual(state[result._id]["status"], "stale")
                self.assertNotIn("editor", state[result._id])
                self.assertNotIn("old-image.png", str([reply["data"] for reply in replies]))
                output_index = function.outputs.index(button)
                self.assertFalse(
                    any(
                        reply["data"][output_index].get("interactive") is True
                        for reply in replies
                        if isinstance(reply["data"][output_index], dict)
                    )
                )

    def test_native_stop_cancels_only_this_turn_and_fences_late_terminal(self):
        app, workspace, function, _, result = self.make_ui()
        cancel = next(
            fn
            for fn in app.fns.values()
            if fn.fn is not None and fn.fn.__name__ == "cancel_image_request"
        )
        self.events = [
            (
                "image_tool_result",
                dict(
                    call_id="late",
                    name="generate_image",
                    status="completed",
                    editor={"session_id": "old", "revision": 1},
                ),
            ),
            ("results", "![Image](late-image.png)"),
        ]
        cancelled = False

        async def stop(state, _):
            nonlocal cancelled
            if self.sent and not cancelled:
                cancelled = True
                await app.process_api(
                    cancel, [], state=state, request=gr.Request(session_hash="chat-fixture")
                )

        state, replies = self.submit(app, function, intent="run", after_step=stop)
        self.worker.interrupt_processing.assert_called_once_with(1)
        self.assertIsNone(state[result._id])
        self.assertNotIn("late-image.png", str([reply["data"] for reply in replies]))
        workspace.result_context.assert_not_called()

    def assert_native_noop_cancellation(self, *, finished):
        from ruinedfooocus.app.resource_lifecycle import latest_ui_requests, ui_request_owner

        app, _, _, button, result = self.make_ui()
        cancel = next(
            fn
            for fn in app.fns.values()
            if fn.fn is not None and fn.fn.__name__ == "cancel_image_request"
        )
        self.assertEqual(len(cancel.outputs), 3)
        self.assertEqual(
            [type(component) for component in cancel.outputs], [gr.State, gr.Markdown, gr.Button]
        )
        with app:
            seed = gr.Button(visible=False).click(
                lambda: (
                    {"call_id": "preserved-result"},
                    "Preserved image status",
                    gr.update(value="Preserved open result", interactive=True),
                ),
                outputs=cancel.outputs,
            )
        state = SessionState(app)
        request = gr.Request(session_hash=f"noop-cancel-{finished}")
        seeded = asyncio.run(app.process_api(seed["id"], [], state=state, request=request))
        self.assertEqual(seeded["data"][1], "Preserved image status")
        self.assertEqual(seeded["data"][2]["value"], "Preserved open result")
        if finished:
            turn = latest_ui_requests.begin(
                ui_request_owner(request), f"chat-image-{result._id}", 1
            )
            turn.source = "completed-worker-task"
            latest_ui_requests.finish(turn)
        with (
            patch.object(Path, "open", side_effect=AssertionError("Cancellation opened a file")),
            patch("builtins.open", side_effect=AssertionError("Cancellation opened a file")),
        ):
            reply = asyncio.run(app.process_api(cancel, [], state=state, request=request))
        self.assertEqual(reply["data"], [None, gr.skip(), gr.skip()])
        self.assertEqual(reply["changed_state_ids"], [])
        self.assertEqual(state[result._id], {"call_id": "preserved-result"})
        # Markdown/button values live in the browser: the three serialized skips
        # preserve the seeded wire values rather than replacing them with defaults.
        self.assertTrue(state[button._id].interactive)
        self.assertEqual(self.sent, [])
        self.worker.interrupt_processing.assert_not_called()

    def test_idle_image_input_cancel_preserves_three_native_outputs_without_file_io(self):
        self.assert_native_noop_cancellation(finished=False)

    def test_finished_image_input_cancel_preserves_three_native_outputs_without_file_io(self):
        self.assert_native_noop_cancellation(finished=True)

    def test_cancel_on_initial_reset_prevents_any_queued_work(self):
        app, workspace, function, _, result = self.make_ui()
        cancel = next(
            fn
            for fn in app.fns.values()
            if fn.fn is not None and fn.fn.__name__ == "cancel_image_request"
        )
        cancelled = False

        async def stop_before_enqueue(state, reply):
            nonlocal cancelled
            reset = reply["data"][function.outputs.index(function.inputs[2])] == "text"
            if reset and not cancelled:
                self.assertFalse(self.sent)
                cancelled = True
                await app.process_api(
                    cancel, [], state=state, request=gr.Request(session_hash="chat-fixture")
                )

        state, _ = self.submit(app, function, intent="run", after_step=stop_before_enqueue)
        self.assertTrue(cancelled)
        self.assertFalse(self.sent)
        self.worker.interrupt_processing.assert_not_called()
        self.assertIsNone(state[result._id])
        workspace.result_context.assert_not_called()

    def test_cancellation_during_enqueue_interrupts_the_just_attached_task(self):
        app, workspace, function, _, result = self.make_ui()
        cancel = next(
            fn
            for fn in app.fns.values()
            if fn.fn is not None and fn.fn.__name__ == "cancel_image_request"
        )

        def enqueue(data):
            self.sent.append(deepcopy(data))
            cancel.fn(request=gr.Request(session_hash="chat-fixture"))
            return 1

        self.worker.add_task = enqueue
        self.events = []
        state, _ = self.submit(app, function, intent="run")
        self.worker.interrupt_processing.assert_called_once_with(1)
        self.assertIsNone(state[result._id])
        workspace.result_context.assert_not_called()

    def test_registered_denoise_and_model_events_fence_pending_typed_results(self):
        for input_index, event in ((14, "input"), (3, "change"), (7, "change")):
            with self.subTest(input_index=input_index):
                app, workspace, function, _, result = self.make_ui()
                control = function.inputs[input_index]
                dependency = next(
                    row
                    for row in app.config["dependencies"]
                    if any(tuple(target) == (control._id, event) for target in row["targets"])
                    and app.fns[row["id"]].fn.__name__ == "revoke_permissions"
                )
                change = app.fns[dependency["id"]]
                self.assertNotIn(control, change.outputs)
                self.assertNotIn(workspace.context, change.outputs)
                self.assertNotIn(function.inputs[1], change.outputs)
                self.events = [
                    (
                        "image_tool_result",
                        dict(
                            call_id="late",
                            name="generate_image",
                            status="completed",
                            editor={"session_id": "old", "revision": 1},
                        ),
                    ),
                    ("results", "![Image](obsolete.png)"),
                ]
                changed = False

                async def change_selection(state, _):
                    nonlocal changed
                    if self.sent and not changed:
                        changed = True
                        await app.process_api(
                            change,
                            [None],
                            state=state,
                            request=gr.Request(session_hash="chat-fixture"),
                        )

                state, replies = self.submit(
                    app, function, intent="run", after_step=change_selection
                )
                self.worker.interrupt_processing.assert_called_once_with(1)
                self.assertIsNone(state[result._id])
                self.assertNotIn("obsolete.png", str([reply["data"] for reply in replies]))
                workspace.result_context.assert_not_called()

    def test_cancellation_during_result_lookup_prevents_terminal_open_and_history_publication(self):
        app, workspace, function, button, result = self.make_ui()
        cancel = next(
            fn
            for fn in app.fns.values()
            if fn.fn is not None and fn.fn.__name__ == "cancel_image_request"
        )

        def lookup(terminal, *, request):
            cancel.fn(request=request)
            return dict(session_id="valid", revision=1, source_id="valid-pixels")

        workspace.result_context.side_effect = lookup
        self.events = [
            (
                "image_tool_result",
                dict(
                    call_id="old",
                    name="generate_image",
                    status="completed",
                    editor={"session_id": "valid", "revision": 1},
                ),
            ),
            ("results", "![Image](replaced-during-lookup.png)"),
        ]
        state, replies = self.submit(app, function, intent="run")
        self.assertIsNone(state[result._id])
        self.assertNotIn("replaced-during-lookup.png", str([reply["data"] for reply in replies]))
        output_index = function.outputs.index(button)
        self.assertFalse(
            any(
                reply["data"][output_index].get("interactive") is True
                for reply in replies
                if isinstance(reply["data"][output_index], dict)
            )
        )
        self.worker.interrupt_processing.assert_called_once_with(1)

    def test_error_terminal_is_bounded_and_does_not_reload_other_controls(self):
        app, workspace, function, button, result = self.make_ui()
        self.events = [
            (
                "image_tool_result",
                dict(
                    call_id="failed",
                    name="process_image",
                    status="error",
                    error={"message": "Missing selected resource"},
                ),
            ),
            ("results", "Please choose the resource"),
        ]
        state, replies = self.submit(app, function, intent="run")
        self.assertEqual(state[result._id]["status"], "error")
        self.assertIn("Missing selected resource", str([reply["data"] for reply in replies]))
        self.assertNotIn(workspace.context, function.outputs)
        self.assertNotIn(function.inputs[19], function.outputs)

    def test_uploaded_reference_crosses_the_native_image_dto_to_the_single_workspace(self):
        app, workspace, function, _, _ = self.make_ui()
        owned = dict(session_id="upload", revision=0, source_id="uploaded-pixels")
        workspace.image_context.return_value = owned
        Path(get_upload_folder()).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=get_upload_folder()) as directory:
            path = Path(directory) / "rgba.png"
            Image.new("RGBA", (3, 2), (20, 40, 60, 128)).save(path)
            self.submit(
                app,
                function,
                intent="preview",
                mode="Uploaded image",
                overrides={13: {"path": str(path), "meta": {"_type": "gradio.FileData"}}},
            )
        self.assertEqual(self.sent[0]["editor_context"], owned)
        reference = workspace.image_context.call_args.args[2]
        self.assertIsInstance(reference, Image.Image)
        self.assertEqual((reference.mode, reference.size), ("RGBA", (3, 2)))
        self.assertEqual(reference.getpixel((0, 0)), (20, 40, 60, 128))

    def test_source_input_event_cancels_pending_turn_and_retry_uses_new_context(self):
        app, workspace, function, _, result = self.make_ui()
        cancel = next(
            fn
            for fn in app.fns.values()
            if fn.fn is not None and fn.fn.__name__ == "cancel_image_request"
        )
        self.events = [
            (
                "image_tool_result",
                dict(
                    call_id="late",
                    name="edit_image",
                    status="completed",
                    editor={"session_id": "old", "revision": 1},
                ),
            ),
            ("results", "Old image"),
        ]
        cancelled = False

        async def changed(state, _):
            nonlocal cancelled
            if self.sent and not cancelled:
                cancelled = True
                await app.process_api(
                    cancel, [], state=state, request=gr.Request(session_hash="chat-fixture")
                )

        state, _ = self.submit(app, function, intent="run", after_step=changed)
        self.assertIsNone(state[result._id])
        self.events = [("results", "New reply")]
        current = dict(session_id="fresh", revision=2, source_id="current")
        workspace.image_context.return_value = current
        self.submit(
            app,
            function,
            intent="preview",
            state=state,
            mode="Previous generated image",
            overrides={17: current, 18: None},
        )
        self.assertEqual(self.sent[1]["editor_context"], current)


if __name__ == "__main__":
    unittest.main()
