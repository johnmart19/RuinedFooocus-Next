"""New UI messages retain per-browser language and literal user/device data."""

from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.translation_manager import catalogue, localize, native_key, resolve_locale
from ruinedfooocus.ui import optional_image_runtime as optional
from ui_test_helpers import function_from


class NewUiLocalizationTests(unittest.TestCase):
    def test_automatic_llama_fallback_note_uses_each_browser_locale(self):
        manager = catalogue()
        message = (
            "Automatic prefers {backend}, which is not installed yet. Press Install now to add it."
        )
        key = manager.aliases[message]
        cached_path = r"C:\literal model folder\llama-server.exe"
        status = {"installed": ["CPU"], "paths": {"CPU": cached_path}, "server_override": None}
        callback = function_from(
            "ruinedfooocus/ui/ui_hardware.py",
            "llama_status_text",
            {
                "inventory": SimpleNamespace(component_status=lambda: {"llm": status}),
                "settings": SimpleNamespace(default_settings={"llama_backend": "Auto"}),
                "localize": localize,
            },
        )
        policy = ModuleType("ruinedfooocus.backends.llama.runtime_policy")
        policy.preferred_backend = Mock(return_value="Vulkan")
        with patch.dict(sys.modules, {policy.__name__: policy}):
            for locale in manager.manifest["locales"]:
                with self.subTest(locale=locale):
                    expected = manager.translations[locale][key].format(backend="Vulkan")
                    rendered = callback(locale, "Auto")
                    self.assertIn(expected, rendered)
                    self.assertIn(cached_path, rendered)
                    self.assertNotIn("{backend}", rendered)
                    if locale != "en":
                        self.assertNotEqual(expected, message.format(backend="Vulkan"))
            ukrainian_note = manager.translations["uk"][key].format(backend="Vulkan")
            self.assertNotIn(ukrainian_note, callback("uk", "CPU"))
            status["paths"]["Vulkan"] = cached_path
            self.assertNotIn(ukrainian_note, callback("uk", "Auto"))
        self.assertEqual(manager.language, "en")

    def test_optional_status_localizes_each_browser_without_changing_wire_device(self):
        manager = catalogue()
        key = manager.aliases["Requested image engine: {engine}."]
        status = {"available": False, "reason": "Not installed"}
        with (
            patch.dict(manager.translations["uk"], {key: "Запитаний рушій: {engine}."}),
            patch.object(optional.activation, "status", return_value=status),
        ):
            ukrainian = optional.status_text("sdcpp-vulkan", "Vulkan12", locale="uk")
            english = optional.status_text("sdcpp-vulkan", "Vulkan12", locale="en")
        self.assertIn("Запитаний рушій: sdcpp-vulkan.", ukrainian)
        self.assertIn("Requested image engine: sdcpp-vulkan.", english)
        self.assertIn("Vulkan12", ukrainian)
        self.assertIn("Vulkan12", english)
        self.assertEqual(manager.language, "en")

    def test_assistant_save_localizes_confirmation_and_preserves_literal_instructions(self):
        from ruinedfooocus.core.translation_manager import translate_ui

        arguments = ModuleType("ruinedfooocus.app.arguments")
        arguments.args = SimpleNamespace(guest=False)
        save = Mock(return_value=("assistant-id", "revision"))
        callback = function_from(
            "ruinedfooocus/features/chat/ui.py",
            "save_assistant_draft",
            {
                "save_assistant": save,
                "gr_llama_get_assistants": lambda value: value,
                "llama_assistants": "selected",
                "assistant_draft": "draft",
                "assistant_version": "revision",
                "assistant_status": "status",
                "translate_ui": translate_ui,
            },
        )
        with patch.dict(sys.modules, {"ruinedfooocus.app.arguments": arguments}):
            result = callback(
                "猫 {name}", "Keep <tag> and {literal} unchanged", None, None, "current"
            )
        self.assertEqual(
            save.call_args.args[:2], ("猫 {name}", "Keep <tag> and {literal} unchanged")
        )
        message = "Assistant saved. Select it to start a conversation with the saved instructions."
        self.assertEqual(result["status"].key, native_key(catalogue().aliases[message]))
        self.assertEqual(result["selected"], "current")

    def test_path_help_language_refresh_expands_base_without_replacing_draft(self):
        from ruinedfooocus.core.translation_manager import localized_outputs
        from ruinedfooocus.ui import path_picker
        import gradio as gr

        with gr.Blocks():
            field = path_picker.create_path_control("path_checkpoints", "Folders", ["owned draft"])
        render, inputs, outputs = localized_outputs()[-1]
        self.assertEqual(inputs, [])
        self.assertEqual(outputs, [field])
        update = render("uk")
        self.assertIn("info", update)
        self.assertNotIn("{base}", update["info"])
        self.assertNotIn("value", update)
        self.assertEqual(field.value, "owned draft")


if __name__ == "__main__":
    unittest.main()
