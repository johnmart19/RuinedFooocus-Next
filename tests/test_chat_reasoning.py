"""Reasoning controls snapshot browser choices without changing persona/tool data."""

import ast
from contextlib import nullcontext
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ui_test_helpers import function_from
from ruinedfooocus.features.chat.chat_tools import IMAGE_TOOL, IMAGE_TOOL_GUIDANCE, ImageToolContext
from ruinedfooocus.features.chat.model_profiles import ProfiledChatClient
from ruinedfooocus.features.chat.ui_reasoning_settings import (
    SAVED_PREFERENCE,
    conversation_reasoning,
)

ROOT = Path(__file__).resolve().parents[1]


class ChatReasoningTests(unittest.TestCase):
    def test_display_control_follows_actual_history_independently_of_computation(self):
        namespace = {"re": re}
        display = function_from("ruinedfooocus/features/chat/ui.py", "display_history", namespace)
        available = function_from(
            "ruinedfooocus/features/chat/ui.py",
            "reasoning_display_available",
            {"display_history": display},
        )
        self.assertFalse(available(None, [{"role": "assistant", "content": "Answer"}]))
        self.assertTrue(available({"control": "enable_thinking"}, []))
        history = [{"role": "assistant", "content": "<think>Private streamed thought"}]
        self.assertTrue(available(None, history))
        self.assertEqual(history[0]["content"], "<think>Private streamed thought")

    def test_controls_hide_unknowns_and_expose_only_initialized_efforts(self):
        controls = function_from(
            "ruinedfooocus/features/chat/ui.py",
            "reasoning_controls",
            {
                "gr": SimpleNamespace(update=lambda **values: values),
                "SAVED_PREFERENCE": SAVED_PREFERENCE,
                "translate_ui": lambda text: text,
            },
        )
        mode, effort, _ = controls(None, "Off", "High")
        self.assertFalse(mode["visible"])
        self.assertFalse(effort["visible"])
        self.assertEqual(mode["value"], SAVED_PREFERENCE)
        mode, effort, _ = controls(
            {
                "modes": ("Auto", "On"),
                "efforts": ("Low", "Medium", "High"),
                "control": "reasoning_effort",
                "reason": "No off switch",
            },
            "Off",
            "Low",
        )
        self.assertNotIn("Off", mode["choices"])
        self.assertEqual(mode["value"], SAVED_PREFERENCE)
        self.assertTrue(effort["visible"])
        _, effort, _ = controls(
            {
                "modes": ("Auto", "Off", "On"),
                "efforts": (),
                "control": "enable_thinking",
                "reason": "Boolean",
            },
            "Off",
            "High",
        )
        self.assertFalse(effort["visible"])
        self.assertEqual(effort["value"], SAVED_PREFERENCE)

    def test_event_wrapper_keeps_reasoning_separate_from_positional_image_controls(self):
        submit = Mock(return_value=iter(("done",)))
        callback = function_from(
            "ruinedfooocus/features/chat/ui.py",
            "submit_with_reasoning",
            {
                "submit_message": submit,
                "conversation_reasoning": conversation_reasoning,
                "settings": SimpleNamespace(default_settings={}),
            },
        )
        self.assertEqual(
            list(callback("message", "image-checkpoint", 0, "Off", "Auto", None)), ["done"]
        )
        submit.assert_called_once_with(
            "message", "image-checkpoint", 0, reasoning={"mode": "Off", "effort": "Auto"}
        )

    def test_event_wrapper_omits_inherited_controls_instead_of_forcing_auto(self):
        submit = Mock(return_value=iter(()))
        callback = function_from(
            "ruinedfooocus/features/chat/ui.py",
            "submit_with_reasoning",
            {
                "submit_message": submit,
                "conversation_reasoning": conversation_reasoning,
                "settings": SimpleNamespace(default_settings={"llm_chat_reasoning_mode": "Off"}),
            },
        )
        list(callback("message", SAVED_PREFERENCE, SAVED_PREFERENCE, None))
        submit.assert_called_once_with("message")

    def test_persona_and_tool_advertisement_keep_explicit_false_and_captured_defaults(self):
        from ruinedfooocus.features.chat.chat_tools import ChatPermissionSession

        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/chat/llama_pipeline.py").read_text(encoding="utf-8")
        )
        owner = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pipeline"
        )
        process = next(
            node
            for node in owner.body
            if isinstance(node, ast.FunctionDef) and node.name == "process"
        )
        requests = []

        def answer(request, callback):
            requests.append(request)
            callback({"choices": [{"delta": {"content": "Meow"}, "finish_reason": "stop"}]})

        adapter = ProfiledChatClient(SimpleNamespace(handle_chat_completions=answer), None, {})
        pipeline = SimpleNamespace(
            llm=adapter,
            model_settings="same",
            current_model_settings=lambda *args, **kwargs: "same",
            vision_projector=None,
            embeddings=None,
            embeddings_hash="[]",
            image_tool_capability=lambda: {"available": True, "binding": "initialized-fixture"},
        )
        scope = {
            "settings": SimpleNamespace(default_settings={"enable_llm_tools": True}),
            "DEFAULT_MODEL": "fixture.gguf",
            "worker": SimpleNamespace(add_result=Mock()),
            "selected_image_reference": lambda *args: (None, ""),
            "json": json,
            "IMAGE_TOOL": IMAGE_TOOL,
            "IMAGE_TOOL_GUIDANCE": IMAGE_TOOL_GUIDANCE,
            "re": re,
            "transient_vision_messages": lambda messages: messages,
            "TimeIt": lambda *args: nullcontext(),
        }
        exec(compile(ast.Module(body=[process], type_ignores=[]), "llama_pipeline", "exec"), scope)
        base = {
            "task_id": 1,
            "model": "fixture.gguf",
            "system": "You are Cat. Use only meows.",
            "history": [{"role": "user", "content": "Just chat"}],
            "reasoning": {"mode": "Auto", "effort": "Auto"},
            "_chat_preparation": SimpleNamespace(
                vision=False, settings={"enable_llm_tools": False}
            ),
            "_image_tool_context": ImageToolContext(("fixture", "owner"), "run", {}),
        }
        self.assertEqual(scope["process"](pipeline, dict(base)), "Meow")
        self.assertNotIn("tools", requests[-1])
        self.assertEqual(requests[-1]["messages"][0]["content"], base["system"])
        scope["process"](pipeline, dict(base, enable_image_tools=True))
        self.assertNotIn("tools", requests[-1])
        permissions = ChatPermissionSession().capture(
            ("fixture", "owner"), enable_image_generation=True
        )
        scope["process"](
            pipeline,
            dict(
                base,
                _chat_tool_permissions=permissions,
                _image_tool_context=ImageToolContext(
                    permissions.owner, "run", {}, permissions=permissions
                ),
            ),
        )
        request = requests[-1]
        self.assertEqual(request["tools"], [IMAGE_TOOL])
        self.assertEqual(request["tool_choice"], "auto")
        self.assertTrue(request["messages"][0]["content"].startswith(base["system"]))
        self.assertIn("do not redefine your persona", request["messages"][0]["content"])
        permissions = ChatPermissionSession().capture(
            ("fixture", "owner"), enable_image_generation=False
        )
        scope["process"](
            pipeline,
            dict(
                base,
                enable_image_tools=True,
                _chat_tool_permissions=permissions,
                _image_tool_context=ImageToolContext(
                    permissions.owner, "text", {}, permissions=permissions
                ),
            ),
        )
        self.assertNotIn("tools", requests[-1])
        self.assertNotIn("_rf_reasoning", requests[-1])
