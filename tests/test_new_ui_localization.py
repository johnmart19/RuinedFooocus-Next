"""New UI messages retain per-browser language and literal user/device data."""

from pathlib import Path
import ast
import asyncio
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.translation_manager import catalogue, localize, native_key, resolve_locale
from ruinedfooocus.ui import optional_image_runtime as optional
from ui_test_helpers import function_from


class NewUiLocalizationTests(unittest.TestCase):
    def test_staged_russian_run_label_refreshes_for_explicit_english_without_changing_other_client(
        self,
    ):
        import gradio as gr
        from gradio.state_holder import SessionState
        from ruinedfooocus.core import translation_manager as translations
        from ruinedfooocus.core.translation_manager import translate_ui
        import test_navigation_shell as navigation

        shell, workspace, *_ = navigation.NavigationShellTests().shell(translate=translate_ui)
        event = next(
            row
            for row in shell.app.config["dependencies"]
            if row["targets"] == [(shell.create._id, "select")]
        )
        request = gr.Request(session_hash="russian-client", cookies={"rf_ui_locale": "ru"})
        result = asyncio.run(
            shell.app.process_api(
                event["id"], ["generate", 1], state=SessionState(shell.app), request=request
            )
        )
        envelope = result["data"][0].model_dump()
        self.assertEqual(envelope["updates"][-1]["value"], localize("Generate", "ru"))
        # Staged navigation emits a plain string; the native locale store cannot
        # retranslate it. It must be registered with the existing dynamic refresh.
        path = Path(__file__).resolve().parents[1] / "ruinedfooocus/ui/app.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        binding = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "localized_output"
            and isinstance(node.value.args[0], ast.Name)
            and node.value.args[0].id == "render_run_label"
        )
        with gr.Blocks():
            tool, scale, target = gr.Textbox(), gr.Textbox(), gr.Textbox()
        scope = dict(
            gr=gr,
            editor_workspace=workspace,
            translate_ui=translate_ui,
            catalogue=catalogue,
            localized_output=translations.localized_output,
            shared=SimpleNamespace(get_ctrl=lambda name: tool),
            automatic_upscale=scale,
            automatic_upscale_target=target,
            run_button=shell.outputs[-1],
            ui_controlnet=SimpleNamespace(
                controlnet=SimpleNamespace(
                    cn_options={
                        "Segment by points": {"type": "segment"},
                    }
                )
            ),
        )
        function_from("ruinedfooocus/ui/app.py", "render_run_label", scope)
        with patch.object(translations, "_LOCALIZED_OUTPUTS", []):
            exec(compile(ast.Module(body=[binding], type_ignores=[]), str(path), "exec"), scope)
            render, inputs, outputs = translations.localized_outputs()[0]
        self.assertEqual(outputs, [shell.outputs[-1]])
        self.assertEqual(inputs[:2], [workspace.action, workspace.destination])
        wrapper = function_from(
            "ruinedfooocus/ui/app.py",
            "localized_renderer",
            dict(resolve_locale=resolve_locale, gr=gr),
        )(render, True)
        self.assertEqual(
            wrapper(request, "en", "generate", "create", "None", "Off", "Generated image")["value"],
            "Generate",
        )
        self.assertEqual(
            wrapper(request, "en", "prompt-edit", "editor", "None", "Off", "Generated image")[
                "value"
            ],
            "Edit image",
        )
        self.assertEqual(
            wrapper(request, "en", "generate", "create", "None", "2×", "Input image only")["value"],
            "Upscale",
        )
        self.assertEqual(
            wrapper(
                request, "en", "generate", "editor", "Segment by points", "Off", "Generated image"
            )["value"],
            "Segment by points",
        )
        self.assertEqual(
            wrapper(request, "ru", "generate", "create", "None", "Off", "Generated image")["value"],
            localize("Generate", "ru"),
        )
        self.assertEqual(catalogue().language, "en")

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
