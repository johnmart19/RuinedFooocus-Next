"""Chat setup context stays separate from runtime loading and conversation drafts."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.i18n import I18nData

from ruinedfooocus.core.translation_manager import catalogue
from ruinedfooocus.features.chat.assistants import assistant_info
from ruinedfooocus.features.chat.ui_context import context_text

ROOT = Path(__file__).resolve().parents[1]


class ChatContextTests(unittest.TestCase):
    def test_actual_general_assistant_metadata_is_resolved_before_markdown_interpolation(self):
        assistant = assistant_info("__normal__")["name"]
        self.assertIsInstance(assistant, I18nData)
        text = context_text("candidate.gguf", None, assistant, locale="en")
        with gr.Blocks():
            rendered = gr.Markdown(text).get_config()["value"]
        self.assertIn("Assistant: General chat", rendered)
        self.assertNotIn("__i18n__", rendered)
        self.assertNotIn("translation_metadata", rendered)

    def test_browser_locale_resolves_only_typed_metadata_and_keeps_custom_names_literal(self):
        assistant = assistant_info("__normal__")["name"]
        manager = catalogue()
        with patch.dict(manager.translations["uk"], {"chat.general_chat": "Translated assistant"}):
            localized = context_text("candidate.gguf", None, assistant, locale="uk")
            english = context_text("candidate.gguf", None, assistant, locale="en")
            custom = context_text("candidate.gguf", None, "General chat", locale="uk")
        self.assertIn("Translated assistant", localized)
        self.assertIn("General chat", english)
        self.assertNotIn("Translated assistant", english)
        self.assertIn("General chat", custom)
        self.assertNotIn("Translated assistant", custom)

    def test_unloaded_candidate_and_offline_reason_are_nearby_without_runtime_claims(self):
        text = context_text("candidate.gguf", None, "My assistant", offline=True)
        self.assertIn("Selected: candidate.gguf", text)
        self.assertIn("Loaded: No model loaded", text)
        self.assertIn("Press Load to start chatting", text)
        self.assertIn("only installed or imported models work", text)
        loaded = context_text("candidate.gguf", "active.gguf", "My assistant")
        self.assertIn("Loaded: active.gguf", loaded)
        self.assertIn("Selected: candidate.gguf", loaded)
        self.assertNotIn("Press Load", loaded)

    def test_filenames_and_assistant_names_are_rendered_as_text(self):
        text = context_text("[link](bad).gguf", None, "<script>unsafe</script>")
        self.assertIn(r"\[link\]", text)
        self.assertIn(r"\<script\>", text)

    def test_send_eligibility_is_independent_of_the_removed_context_banner(self):
        tree = ast.parse((ROOT / "ruinedfooocus/features/chat/ui.py").read_text(encoding="utf-8"))
        event = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "llama_active_model"
            and any(
                keyword.arg == "outputs"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "send_message"
                for keyword in node.keywords
            )
        )
        callback = next(keyword.value for keyword in event.keywords if keyword.arg == "fn")
        function = eval(
            compile(ast.Expression(callback), "send eligibility", "eval"),
            {"gr": SimpleNamespace(update=lambda **values: values)},
        )
        for active, allowed in ((None, False), ("loaded.gguf", True)):
            self.assertEqual(function(active), {"interactive": allowed})
        self.assertNotIn(
            "conversation_context",
            {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)},
        )


if __name__ == "__main__":
    unittest.main()
