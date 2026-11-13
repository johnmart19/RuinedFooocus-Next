"""Production Chat constructor and app binding join the actual canonical editor."""

import asyncio
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from PIL import Image

import test_chat_image_ui_events as chat_events
import test_editor_workspace_events as workspace_events
from ui_test_helpers import function_from


class ChatEditorWorkspaceJoinTests(unittest.TestCase):
    def setUp(self):
        self.editor = workspace_events.EditorWorkspaceEventTests()
        self.editor.setUp()
        self.addCleanup(self.editor.doCleanups)
        self.editor.shared.path_manager = SimpleNamespace(
            DOWNLOADABLE_FILES={},
            paths={"path_llm": [], "path_vision": []},
            model_paths={"cache_path": self.editor.root / "cache"},
            get_abspath=Path,
        )
        self.editor.shared.add_ctrl = function_from(
            "ruinedfooocus/app/state.py", "add_ctrl", {"state": self.editor.shared.state}
        )
        self.chat = chat_events.ChatImageUiEventsTests()
        self.chat.sent, self.chat.events = [], []
        worker = SimpleNamespace(
            add_task=lambda data: self.chat.sent.append(deepcopy(data)) or len(self.chat.sent),
            task_result=lambda _: self.chat.events.pop(0),
            interrupt_processing=Mock(return_value=True),
        )
        self.chat.worker = worker
        with patch.dict("sys.modules", {"ruinedfooocus.app.async_worker": worker}):
            source = Path(__file__).resolve().parents[1] / "ruinedfooocus/features/chat/ui.py"
            spec = importlib.util.spec_from_file_location("_rf_chat_workspace_join", source)
            self.ui = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.ui)
        self.ui.worker = worker
        create = function_from(
            "ruinedfooocus/ui/app.py",
            "create_chat_for_workspace",
            {
                "ui_llama_chat": self.ui,
                "shared": SimpleNamespace(get_ctrl=lambda name: self.editor.controls[name]),
            },
        )
        with (
            self.editor.app,
            patch.object(self.ui, "local_models", return_value={}),
            patch.object(self.ui, "model_catalogue", return_value={}),
            patch.object(
                self.editor.workspace, "bind_chat_open", wraps=self.editor.workspace.bind_chat_open
            ) as registered,
        ):
            create({"negative": self.editor.controls["negative"]}, self.editor.workspace)
        self.button, self.result = registered.call_args.args
        self.submit = next(
            fn
            for fn in self.editor.app.fns.values()
            if getattr(fn.fn, "__name__", "") == "submit_with_attachments"
        )

    def assert_current(self, session, context, pixels, prompt, provenance):
        self.assertEqual(session.state[self.editor.workspace.context._id], context)
        self.assertEqual(self.editor.value(session, self.editor.controls["prompt"]), prompt)
        self.assertEqual(
            self.editor.value(session, self.editor.controls["image_source"]), provenance
        )
        current = self.editor.value(session, self.editor.controls["input_image"])
        with Image.open(current["path"]) as image:
            self.assertEqual(image.size, (73, 121))
            self.assertEqual(image.convert("RGBA").getpixel((0, 0)), pixels)

    def test_native_terminal_then_deliberate_open_uses_same_context_and_rejects_stale_or_foreign(
        self,
    ):
        session = self.editor.session("chat-fixture")
        self.editor.open_library(session)
        old = deepcopy(session.state[self.editor.workspace.context._id])
        old_provenance = deepcopy(self.editor.value(session, self.editor.controls["image_source"]))
        session.values[self.editor.controls["prompt"]._id] = "Keep the current Create draft"
        prompt = self.editor.value(session, self.editor.controls["prompt"])
        result_path = self.editor.root / "chat-owned-result.png"
        Image.new("RGBA", (73, 121), (15, 35, 75, 128)).save(result_path)
        completed = self.editor.store.start(result_path, owner=session.owner)
        terminal = dict(
            call_id="native-join",
            name="generate_image",
            status="completed",
            editor=completed.to_dict(public=True),
        )
        self.chat.events = [("image_tool_result", terminal), ("results", "Image finished")]
        context_index = self.submit.inputs.index(self.editor.workspace.context)

        async def retained(state, _):
            self.assertIs(state, session.state)
            self.assert_current(session, old, (20, 40, 60, 127), prompt, old_provenance)

        state, _ = self.chat.submit(
            self.editor.app,
            self.submit,
            intent="run",
            state=session.state,
            overrides={
                context_index: deepcopy(old),
                self.submit.inputs.index(self.editor.controls["negative"]): self.editor.value(
                    session, self.editor.controls["negative"]
                ),
            },
            after_step=retained,
        )
        session.state = state
        self.assertIn(self.editor.workspace.context, self.submit.inputs)
        self.assertNotIn(self.editor.workspace.context, self.submit.outputs)
        self.assertNotIn(self.editor.controls["input_image"], self.submit.outputs)
        self.assertEqual(state[self.result._id]["call_id"], "native-join")
        self.assert_current(session, old, (20, 40, 60, 127), prompt, old_provenance)

        open_event = self.editor.event(self.button)
        _, updates = asyncio.run(
            self.editor.api(session, open_event, {self.editor.workspace.version: 100})
        )
        context = state[self.editor.workspace.context._id]
        self.assertEqual(state[self.editor.destinations._id].selected, "editor")
        self.assertEqual(state[self.editor.workspace.destination._id], "editor")
        self.assertEqual(context, self.editor.workspace.public_context(completed))
        image_data = self.editor.value(session, self.editor.controls["input_image"])
        with Image.open(image_data["path"]) as image:
            self.assertEqual(image.convert("RGBA").getpixel((0, 0)), (15, 35, 75, 128))
        self.assertNotIn(self.editor.controls["prompt"], updates)
        self.assertEqual(self.editor.value(session, self.editor.controls["prompt"]), prompt)
        provenance = deepcopy(self.editor.value(session, self.editor.controls["image_source"]))

        ticket = self.editor.store.capture(
            completed.session_id,
            completed.revision,
            "fixture-edit",
            source_id=completed.current.artifact_id,
            owner=session.owner,
        )
        newer_path = self.editor.root / "newer-result.png"
        Image.new("RGBA", (73, 121), (1, 2, 3, 128)).save(newer_path)
        self.editor.store.complete(ticket, newer_path, owner=session.owner)
        with self.assertRaisesRegex(ValueError, "replaced"):
            asyncio.run(self.editor.api(session, open_event, {self.editor.workspace.version: 101}))
        self.assert_current(session, context, (15, 35, 75, 128), prompt, provenance)
        foreign = self.editor.store.start(result_path, owner=("ui", "foreign", "local", "other"))
        with self.assertRaises(PermissionError):
            asyncio.run(
                self.editor.api(
                    session,
                    open_event,
                    {
                        self.result: dict(terminal, editor=foreign.to_dict(public=True)),
                        self.editor.workspace.version: 102,
                    },
                )
            )
        self.assert_current(session, context, (15, 35, 75, 128), prompt, provenance)

    def test_real_chat_rejects_stale_and_foreign_terminal_without_workspace_mutation(self):
        session = self.editor.session("chat-fixture")
        self.editor.open_library(session)
        current = deepcopy(session.state[self.editor.workspace.context._id])
        provenance = deepcopy(self.editor.value(session, self.editor.controls["image_source"]))
        prompt = self.editor.value(session, self.editor.controls["prompt"])
        source = self.editor.root / "invalid-terminal-original.png"
        Image.new("RGBA", (73, 121), (37, 71, 109, 128)).save(source)
        stale = self.editor.store.start(source, owner=session.owner)
        ticket = self.editor.store.capture(
            stale.session_id,
            stale.revision,
            "fixture-edit",
            source_id=stale.current.artifact_id,
            owner=session.owner,
        )
        updated = self.editor.root / "invalid-terminal-updated.png"
        Image.new("RGBA", (73, 121), (19, 13, 11, 128)).save(updated)
        self.editor.store.complete(ticket, updated, owner=session.owner)
        foreign = self.editor.store.start(source, owner=("ui", "foreign", "local", "other"))
        context_index = self.submit.inputs.index(self.editor.workspace.context)

        async def retained(state, _):
            self.assertIs(state, session.state)
            self.assert_current(session, current, (20, 40, 60, 127), prompt, provenance)

        for name, snapshot in (("stale", stale), ("foreign", foreign)):
            with self.subTest(terminal=name):
                self.chat.events = [
                    (
                        "image_tool_result",
                        dict(
                            call_id=name,
                            name="generate_image",
                            status="completed",
                            editor=snapshot.to_dict(public=True),
                        ),
                    ),
                    ("results", "Invalid image finished"),
                ]
                state, replies = self.chat.submit(
                    self.editor.app,
                    self.submit,
                    intent="run",
                    state=session.state,
                    overrides={
                        context_index: deepcopy(current),
                        self.submit.inputs.index(
                            self.editor.controls["negative"]
                        ): self.editor.value(session, self.editor.controls["negative"]),
                    },
                    after_step=retained,
                )
                self.assertEqual(state[self.result._id]["status"], "stale")
                self.assertNotIn("editor", state[self.result._id])
                button_index = self.submit.outputs.index(self.button)
                self.assertFalse(
                    any(
                        isinstance(reply["data"][button_index], dict)
                        and reply["data"][button_index].get("interactive") is True
                        for reply in replies
                    )
                )
                self.assert_current(session, current, (20, 40, 60, 127), prompt, provenance)


if __name__ == "__main__":
    unittest.main()
