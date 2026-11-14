"""Create request capture and navigation keep the saved Editor draft separate."""

import asyncio
import copy
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState

from ruinedfooocus.app import resource_lifecycle as life
from ruinedfooocus.core.translation_manager import translate_ui
import test_editor_workspace_events as editor_fixtures
import test_navigation_shell as navigation_fixtures
from ui_test_helpers import function_from


class CreateModeIsolationTests(unittest.TestCase):
    def setUp(self):
        fixture = navigation_fixtures.NavigationShellTests()
        self.shell, self.workspace, _, self.scope, *_ = fixture.shell()
        self.context = {"session_id": "saved-editor", "source_id": "owned-source", "revision": 4}
        self.run_button = self.scope["run_button"]
        self.prompt = self.scope["navigation_drafts"]["create"]
        self.source = "saved-editor-source.png"
        self.requests = []
        self.browser_values = {}
        self.request = gr.Request(session_hash="create-isolation")
        self.owner = life.ui_request_owner(self.request)
        self.jobs = life.JobRegistry()
        self.stop = Mock()
        with self.shell.app:
            self.source_control = gr.State(self.source)
            self.mask_control = gr.State("saved-editor-mask.png")
            self.task = gr.State(None)
            self.stop_button = gr.Button("Stop", visible=False)
            self.progress = gr.HTML()
            self.controls = {
                "_editor_destination": self.workspace.destination,
                "_editor_action": self.workspace.action,
                "_editor_context": self.workspace.context,
                "prompt": self.prompt,
                "input_image": self.source_control,
                "inpaint_view": self.mask_control,
                "inpaint_toggle": gr.State(True),
                "base_model_name": gr.State("chosen.safetensors"),
                "image_component_overrides": gr.State({"vae": "chosen-vae.safetensors"}),
                "reference_adapter": gr.State("redux"),
                "reference_adapter_strength": gr.State(0.6),
                "native_latent_input": gr.State("intentional-latent.safetensors"),
                "cn_selection": gr.State("Background removal"),
            }
            self.workspace.run_controls = dict(
                run=self.run_button,
                task=self.task,
                stop=self.stop_button,
                progress=self.progress,
                preview=gr.Image(),
                artifacts=gr.State([]),
                hint=gr.Markdown(),
            )
            self.workspace.run_request = self.accept_request
            self.workspace.stop_request = self.stop
            generate = function_from(
                "ruinedfooocus/ui/app.py",
                "generate_clicked",
                dict(
                    editor_workspace=self.workspace,
                    state={"ctrls_name": list(self.controls)},
                ),
            )
            self.run_button.click(
                generate,
                inputs=[*self.controls.values(), self.workspace.version],
                outputs=[self.run_button, self.task, self.stop_button, self.workspace.notice],
            )
        self.state = SessionState(self.shell.app)
        self.state[self.workspace.context._id] = copy.deepcopy(self.context)
        self.state[self.workspace.action._id].value = "prompt-edit"
        self.fence = life.LatestUIRequests()
        self.enterContext(
            patch("ruinedfooocus.features.gallery.ui_workspace.latest_ui_requests", self.fence)
        )

    def accept_request(self, data):
        """Capture at the worker boundary without preparing or computing images."""
        self.requests.append(copy.deepcopy(data))
        job = self.jobs.create(owner=life.ui_request_owner())
        try:
            yield {
                self.task: job.task_id,
                self.run_button: gr.update(interactive=False),
                self.stop_button: gr.update(visible=True),
            }
            job.check()
            yield {
                self.task: None,
                self.run_button: gr.update(interactive=True),
                self.stop_button: gr.update(visible=False),
            }
        finally:
            self.jobs.finish(job)

    def event(self, component, event):
        return next(
            function
            for function in self.shell.app.fns.values()
            if (component._id, event) in function.targets
        )

    async def navigate(self, destination):
        function = self.event(getattr(self.shell, destination), "select")
        self.navigation_serial = getattr(self, "navigation_serial", 0) + 1
        result = await self.shell.app.process_api(
            function,
            [self.state[self.workspace.action._id].value, self.navigation_serial],
            state=self.state,
            request=self.request,
        )
        payload = result["data"][0]
        payload = payload.model_dump() if hasattr(payload, "model_dump") else payload
        self.assertEqual(payload["version"], self.navigation_serial)
        self.assertEqual(payload["button"], f"rf-nav-{destination}-button")
        # The browser publishes the server's updates through the follow-up step.
        publish = next(
            candidate
            for candidate in self.shell.app.fns.values()
            if candidate.trigger_after == function._id
        )
        self.paint(publish, {"data": payload["updates"]})
        return result

    def paint(self, function, result):
        for control, value in zip(function.outputs, result["data"]):
            if isinstance(value, dict) and "value" in value:
                self.browser_values[control._id] = value["value"]

    async def click_generate(self):
        function = self.event(self.run_button, "click")
        arguments = [
            None if component.stateful else self.state[component._id].value
            for component in function.inputs
        ]
        result = await self.shell.app.process_api(
            function, arguments, state=self.state, request=self.request
        )
        self.paint(function, result)
        return function, result

    async def finish(self, function, result):
        while result["is_generating"]:
            result = await self.shell.app.process_api(
                function, None, state=self.state, request=self.request, iterator=result["iterator"]
            )
            self.paint(function, result)

    def assert_editor_draft(self):
        self.assertEqual(self.state[self.workspace.action._id].value, "prompt-edit")
        self.assertEqual(self.state[self.workspace.context._id], self.context)
        self.assertEqual(self.state[self.source_control._id], self.source)
        self.assertEqual(self.state[self.mask_control._id], "saved-editor-mask.png")

    def test_native_create_editor_roundtrip_restores_label_without_clearing_draft(self):
        async def journey():
            await self.navigate("editor")
            self.assertEqual(self.browser_values[self.run_button._id], "Edit image")
            await self.navigate("create")
            self.assertEqual(self.browser_values[self.run_button._id], "Generate")
            await self.navigate("editor")
            self.assertEqual(self.browser_values[self.run_button._id], "Edit image")

        asyncio.run(journey())
        self.assert_editor_draft()
        self.assertEqual(self.requests, [])

    def test_actual_generate_callback_captures_new_image_and_keeps_intentional_settings(self):
        self.state[self.prompt._id].value = "Fresh image prompt"

        async def journey():
            await self.navigate("editor")
            await self.navigate("create")
            function, result = await self.click_generate()
            self.assertTrue(result["is_generating"])
            captured = self.requests[0]
            self.assertEqual(captured["_editor_action"], "generate")
            self.assertEqual(captured["_editor_destination"], "create")
            for name in ("_editor_context", "editor_context", "input_image", "inpaint_view"):
                self.assertIsNone(captured[name])
            self.assertFalse(captured["inpaint_toggle"])
            self.assertEqual(captured["cn_selection"], "None")
            self.assertEqual(captured["prompt"], "Fresh image prompt")
            self.assertEqual(captured["base_model_name"], "chosen.safetensors")
            self.assertEqual(
                captured["image_component_overrides"], {"vae": "chosen-vae.safetensors"}
            )
            self.assertEqual(captured["reference_adapter"], "redux")
            self.assertEqual(captured["reference_adapter_strength"], 0.6)
            self.assertEqual(captured["native_latent_input"], "intentional-latent.safetensors")
            self.assertNotIn("_image_workflow_step", captured)
            await self.finish(function, result)
            await self.navigate("editor")

        asyncio.run(journey())
        self.assert_editor_draft()
        self.assertEqual(self.state[self.prompt._id].value, "Fresh image prompt")
        self.stop.assert_not_called()

    def test_navigation_during_accepted_job_keeps_task_button_ownership_and_editor_draft(self):
        async def journey():
            await self.navigate("create")
            function, result = await self.click_generate()
            task_id = self.state[self.task._id]
            self.assertIsNotNone(task_id)
            await self.navigate("editor")
            self.assertEqual(self.browser_values[self.run_button._id], "Edit image")
            self.assertFalse(self.state[self.run_button._id].interactive)
            await self.navigate("create")
            self.assertEqual(self.browser_values[self.run_button._id], "Generate")
            self.assertFalse(self.state[self.run_button._id].interactive)
            self.assertEqual(self.state[self.task._id], task_id)
            self.assertFalse(self.jobs.get(task_id, owner=self.owner).cancelled.is_set())
            self.assert_editor_draft()
            await self.finish(function, result)
            self.assertIsNone(self.state[self.task._id])
            self.assertTrue(self.state[self.run_button._id].interactive)

        asyncio.run(journey())
        self.stop.assert_not_called()

    def test_action_callback_uses_current_destination_for_hidden_editor_changes(self):
        editor = editor_fixtures.EditorWorkspaceEventTests()
        editor.setUp()
        self.addCleanup(editor.doCleanups)
        session = editor.session()
        callback = next(
            function
            for function in editor.app.fns.values()
            if getattr(function.fn, "__name__", "") == "action_context"
        )
        self.assertIn((editor.workspace.action._id, "change"), callback.targets)
        self.assertIn((editor.workspace.destination._id, "change"), callback.targets)
        for destination, expected in (("create", "Generate"), ("editor", "Edit image")):
            editor.call(
                session,
                callback,
                {editor.workspace.action: "prompt-edit", editor.workspace.destination: destination},
            )
            value = editor.value(session, editor.remote["run"])
            self.assertEqual(value, str(translate_ui(expected)))


if __name__ == "__main__":
    unittest.main()
