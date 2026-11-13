"""Native Chat control transitions that do not load a model or start work."""

import asyncio
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState

import test_chat_image_ui_events as image_ui_fixture
from ruinedfooocus.features.chat.ui_reasoning_settings import SAVED_PREFERENCE


class ChatUiAcceptanceTests(unittest.TestCase):
    def functions(self, app):
        return [fn for fn in app.fns.values() if fn.fn is not None]

    def fixture(self):
        fixture = image_ui_fixture.ChatImageUiEventsTests()
        app, _, submit, button, result = fixture.make_ui()
        state = SessionState(app)
        state[submit.inputs[6]._id] = [{"role": "user", "content": "Keep my conversation"}]
        state[submit.inputs[7]._id] = "keep-loaded.gguf"
        search = next(fn for fn in self.functions(app) if fn.fn.__name__ == "search_models")
        session = state[search.inputs[-1]._id]
        session.active = "keep-loaded.gguf"
        session.select("keep-selected.gguf")
        return fixture, app, state, submit, session, button, result

    def call(self, app, fn, state, values=()):
        return asyncio.run(
            app.process_api(
                fn,
                list(values),
                state=state,
                request=gr.Request(session_hash="chat-control-acceptance"),
            )
        )

    def test_new_and_cancel_assistant_editor_preserve_conversation_and_model_without_saving(self):
        fixture, app, state, submit, session, _, _ = self.fixture()
        opened = next(
            fn
            for fn in self.functions(app)
            if fn.fn.__name__ == "open_assistant_editor" and not fn.inputs
        )
        cancelled = next(
            fn
            for fn in self.functions(app)
            if fn.fn.__name__ == "<lambda>" and fn.outputs == opened.outputs and not fn.inputs
        )
        before = deepcopy(state[submit.inputs[6]._id])
        save = Mock(side_effect=AssertionError("Opening/cancelling an editor must not save"))
        with patch.dict(opened.fn.__globals__, {"save_assistant": save}):
            showing = self.call(app, opened, state)
            hidden = self.call(app, cancelled, state)
        self.assertEqual(len(showing["data"]), 7)
        self.assertTrue(showing["data"][0]["visible"])
        self.assertEqual(len(hidden["data"]), 7)
        self.assertFalse(hidden["data"][0]["visible"])
        self.assertEqual(state[submit.inputs[6]._id], before)
        self.assertEqual(state[submit.inputs[7]._id], "keep-loaded.gguf")
        self.assertEqual(
            (session.active, session.selected), ("keep-loaded.gguf", "keep-selected.gguf")
        )
        save.assert_not_called()
        self.assertEqual(fixture.sent, [])
        fixture.worker.interrupt_processing.assert_not_called()

    def test_idle_new_conversation_and_assistant_selection_have_valid_native_output_contracts(self):
        fixture, app, state, submit, session, button, result = self.fixture()
        for name in ("new_conversation", "select_assistant_and_clear"):
            with self.subTest(event=name):
                fn = next(fn for fn in self.functions(app) if fn.fn.__name__ == name)
                with patch.dict(fn.fn.__globals__, {"assistant_avatar": lambda _: None}):
                    response = self.call(app, fn, state, [None, "__normal__"])
                self.assertEqual(len(response["data"]), len(fn.outputs))
                self.assertIsNone(state[result._id])
                self.assertEqual(state[submit.inputs[7]._id], "keep-loaded.gguf")
                self.assertEqual(session.active, "keep-loaded.gguf")
                self.assertEqual(session.selected, "keep-selected.gguf")
                self.assertFalse(response["data"][fn.outputs.index(button)]["interactive"])
                self.assertEqual(response["data"][fn.outputs.index(submit.inputs[2])], "text")
                self.assertEqual(response["data"][fn.outputs.index(submit.inputs[12])], "New image")
                self.assertEqual(response["data"][fn.outputs.index(submit.inputs[-7])], "")
        self.assertEqual(fixture.sent, [])
        fixture.worker.interrupt_processing.assert_not_called()

    def test_reasoning_capability_transitions_serialize_valid_controls_without_persisting_preferences(
        self,
    ):
        fixture, app, state, submit, session, _, _ = self.fixture()
        fn = next(fn for fn in self.functions(app) if fn.fn.__name__ == "reasoning_controls")
        preferences = fn.fn.__globals__["settings"].default_settings
        before = deepcopy(preferences)

        def apply(capability, mode=SAVED_PREFERENCE, effort=SAVED_PREFERENCE):
            state[fn.inputs[0]._id] = capability
            result = self.call(app, fn, state, [None, mode, effort])["data"]
            self.assertEqual(len(result), 3)
            return result

        unknown = apply(None, "Off", "High")
        self.assertFalse(unknown[0]["visible"])
        self.assertFalse(unknown[1]["visible"])
        boolean = {
            "modes": ("Auto", "Off", "On"),
            "efforts": (),
            "control": "enable_thinking",
            "reason": "Boolean template",
        }
        apply(boolean)
        off = apply(boolean, "Off")
        self.assertTrue(off[0]["visible"])
        self.assertFalse(off[1]["visible"])
        effort = {
            "modes": ("Auto", "On"),
            "efforts": ("Low", "Medium", "High"),
            "control": "reasoning_effort",
            "reason": "Effort template; no off switch",
        }
        graded = apply(effort, "Off")
        self.assertFalse(graded[0]["visible"])
        self.assertEqual(graded[0]["value"], SAVED_PREFERENCE)
        self.assertNotIn("Off", [choice[1] for choice in graded[0]["choices"]])
        self.assertTrue(graded[1]["visible"])
        apply(effort, "On", "High")
        unloaded = apply(None, "On", "High")
        self.assertFalse(unloaded[0]["visible"])
        self.assertFalse(unloaded[1]["visible"])
        display = next(
            candidate
            for candidate in self.functions(app)
            if candidate.inputs == [fn.inputs[0], submit.inputs[6]]
            and len(candidate.outputs) == 1
            and isinstance(candidate.outputs[0], gr.Checkbox)
        )
        state[submit.inputs[6]._id] = [{"role": "assistant", "content": "<think>Retained thought"}]
        self.assertTrue(self.call(app, display, state, [None, None])["data"][0]["visible"])
        self.assertEqual(preferences, before)
        self.assertEqual(session.active, "keep-loaded.gguf")
        self.assertEqual(fixture.sent, [])
        fixture.worker.interrupt_processing.assert_not_called()

    def test_reference_wrappers_follow_relevance_and_missing_sources_hide_denoise(self):
        fixture, app, state, _, _, _, _ = self.fixture()
        function = next(fn for fn in self.functions(app) if fn.fn.__name__ == "reference_controls")
        image = self.call(app, function, state, ["run", False, None, "Uploaded image", None, None])[
            "data"
        ]
        self.assertTrue(image[0]["visible"])
        self.assertTrue(image[1]["visible"])
        self.assertFalse(image[2]["visible"])
        self.assertIn("provide an image reference", image[3]["value"])
        text = self.call(app, function, state, ["text", False, None, "Uploaded image", None, None])[
            "data"
        ]
        self.assertFalse(text[0]["visible"])
        self.assertFalse(text[1]["visible"])
        self.assertFalse(text[2]["visible"])
        self.assertFalse(text[4]["visible"])
        self.assertEqual(fixture.sent, [])


if __name__ == "__main__":
    unittest.main()
