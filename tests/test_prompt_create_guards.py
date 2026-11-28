"""Guards for user_errors console quietness and Prompt/Create/Settings button usability."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
from PIL import Image

from ruinedfooocus.core.paths import asset_path
from ruinedfooocus.features.prompting import (
    prompt_preview,
    ui_caption,
    ui_evolve,
    ui_rewrite,
    ui_styles,
)
from ruinedfooocus.ui import ui_settings
from ruinedfooocus.ui.settings_navigation import source_text
from ruinedfooocus.ui.user_errors import as_user_errors

ROOT = Path(__file__).resolve().parents[1]


class UserErrorsQuietConsoleTests(unittest.TestCase):
    def test_expected_user_errors_suppress_console_tracebacks_while_preserving_cause(self):
        for exc in (
            ValueError("Enter a name first."),
            PermissionError("Sign in again."),
            FileNotFoundError("Model file missing."),
        ):

            def sync_fail(exc=exc):
                raise exc

            def gen_fail(exc=exc):
                yield "step"
                raise exc

            async def async_fail(exc=exc):
                raise exc

            async def async_gen_fail(exc=exc):
                yield "step"
                raise exc

            async def drain(fn):
                return [item async for item in fn()]

            for invoke in (
                lambda: as_user_errors(sync_fail)(),
                lambda: list(as_user_errors(gen_fail)()),
                lambda: asyncio.run(as_user_errors(async_fail)()),
                lambda: asyncio.run(drain(as_user_errors(async_gen_fail))),
            ):
                with self.subTest(error=type(exc).__name__, invoke=invoke):
                    with self.assertRaises(gr.Error) as raised:
                        invoke()
                    self.assertEqual(raised.exception.message, str(exc))
                    self.assertIs(raised.exception.__cause__, exc)
                    self.assertFalse(raised.exception.print_exception)


class PromptToolButtonGuardTests(unittest.TestCase):
    def test_caption_buttons_disable_without_real_image_and_enable_on_upload(self):
        with gr.Blocks(analytics_enabled=False) as app:
            prompt = gr.Textbox(value="")
            preview = gr.Image(value=str(asset_path("init_image.png")), type="filepath")
            run_event = gr.Number(value=0)
            describe, undo = ui_caption.add_describe_controls(
                prompt, preview, allow_upload=True, run_event=run_event
            )
        self.assertFalse(describe.interactive)
        self.assertFalse(undo.interactive)
        append_btn = next(
            c
            for c in app.blocks.values()
            if getattr(c, "elem_id", None) == "describe-append-prompt"
        )
        gen_btn = next(
            c
            for c in app.blocks.values()
            if getattr(c, "elem_id", None) == "describe-generate-image"
        )
        self.assertFalse(append_btn.interactive)
        self.assertFalse(gen_btn.interactive)

        sync = next(
            fn.fn
            for fn in app.fns.values()
            if getattr(fn.fn, "__name__", None) == "sync_describe_button"
        )
        self.assertFalse(sync(str(asset_path("init_image.png")), None)["interactive"])
        self.assertTrue(sync(None, Image.new("RGB", (16, 16)))["interactive"])

        with self.assertRaises(gr.Error) as raised:
            ui_caption.describe_image("", None)
        self.assertFalse(raised.exception.print_exception)

    def test_rewrite_preview_button_requires_prompt_and_preset(self):
        with (
            patch.object(ui_rewrite, "rewrite_presets", return_value=[("Crisp", "crisp.txt")]),
            gr.Blocks(analytics_enabled=False) as app,
        ):
            prompt = gr.Textbox(value="")
            ui_rewrite.add_rewrite_controls(prompt)
        buttons = {
            source_text(getattr(c, "value", None)): c
            for c in app.blocks.values()
            if isinstance(c, gr.Button)
        }
        preview_btn = buttons["Preview rewrite"]
        apply_btn = buttons["Apply rewrite"]
        undo_btn = buttons["Undo rewrite"]
        self.assertFalse(preview_btn.interactive)
        self.assertFalse(apply_btn.interactive)
        self.assertFalse(undo_btn.interactive)

        sync = next(
            fn.fn for fn in app.fns.values() if fn.fn is not None and fn.outputs == [preview_btn]
        )
        self.assertFalse(sync("", "crisp.txt")["interactive"])
        self.assertFalse(sync("a cat", None)["interactive"])
        self.assertTrue(sync("a cat", "crisp.txt")["interactive"])

        suggest = next(
            fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", None) == "suggest"
        )
        with self.assertRaises(gr.Error) as raised:
            suggest("crisp.txt", "   ")
        self.assertFalse(raised.exception.print_exception)

    def test_evolve_buttons_disable_until_prompt_and_variants_exist(self):
        with (
            patch.object(ui_evolve, "translate_ui", side_effect=lambda text: text),
            gr.Blocks(analytics_enabled=False) as app,
        ):
            prompt = gr.Textbox(value="")
            run_event = gr.Number(value=0)
            image_number = gr.Number(value=1)
            ui_evolve.add_evolve_controls(prompt, run_event=run_event, image_number=image_number)
        buttons = {c.value: c for c in app.blocks.values() if isinstance(c, gr.Button)}
        for label in (
            "Preview variations",
            "Apply candidate to prompt",
            "Undo prompt change",
            "Generate reviewed candidates",
            "Use current prompts as parents",
            "Use preview as parents",
            "Copy selected parent to prompt",
        ):
            with self.subTest(button=label):
                self.assertFalse(buttons[label].interactive)

        sync = next(
            fn.fn
            for fn in app.fns.values()
            if getattr(fn.fn, "__name__", None) == "sync_evolve_buttons"
        )
        # Prompt typed, no session yet -> preview & load_parents enabled, variant actions still disabled.
        updates = sync("a forest", None)
        self.assertEqual(
            [u["interactive"] for u in updates], [True, True, False, False, False, False]
        )
        # Session with parents and variants matching current prompt -> all enabled.
        session = {
            "source": "a forest",
            "parents": ["a forest"],
            "variants": ["a forest", "a pine"],
        }
        updates = sync("a forest", session)
        self.assertEqual([u["interactive"] for u in updates], [True, True, True, True, True, True])
        # Prompt edited after preview -> variant actions disable until re-previewed.
        updates = sync("edited", session)
        self.assertEqual(
            [u["interactive"] for u in updates], [True, True, False, False, True, False]
        )

    def test_prompt_preview_undo_buttons_start_disabled_and_sync_with_history(self):
        with gr.Blocks(analytics_enabled=False) as app:
            prompt = gr.Textbox(value="draft")
            prompt_preview.create_prompt_action(prompt)
            prompt_preview.create_chat_prompt_action(prompt, owner_for_request=lambda _: "owner")
        undo_buttons = [
            c
            for c in app.blocks.values()
            if isinstance(c, gr.Button) and "Undo" in source_text(c.value)
        ]
        self.assertEqual(len(undo_buttons), 2)
        for btn in undo_buttons:
            self.assertFalse(btn.interactive)
            sync = next(
                fn.fn for fn in app.fns.values() if fn.fn is not None and fn.outputs == [btn]
            )
            self.assertFalse(sync(None)["interactive"])
            self.assertTrue(sync(("old", "new"))["interactive"])

    def test_styles_clear_button_disabled_without_visual_styles(self):
        with gr.Blocks(analytics_enabled=False) as app:
            controls = ui_styles.create_ui({})
        clear_btn = next(
            c
            for c in app.blocks.values()
            if isinstance(c, gr.Button) and "Clear styles" in source_text(c.value)
        )
        self.assertFalse(clear_btn.interactive)
        sync = next(
            fn.fn for fn in app.fns.values() if fn.fn is not None and fn.outputs == [clear_btn]
        )
        self.assertFalse(sync(["Local chat model", "LoRA keywords"])["interactive"])
        self.assertTrue(sync(["Style: sai-anime"])["interactive"])


class CreateAndSettingsGuardTests(unittest.TestCase):
    def test_app_sets_hf_hub_disable_symlinks_warning_on_windows(self):
        source = (ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8")
        self.assertIn("HF_HUB_DISABLE_SYMLINKS_WARNING", source)

    def test_create_custom_size_save_and_delete_guards(self):
        source = (ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        # Verify ratio_save is created with interactive=False.
        ratio_save_assign = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "ratio_save" for t in node.targets)
            and isinstance(node.value, ast.Call)
        )
        kw = {k.arg: k.value for k in ratio_save_assign.value.keywords}
        self.assertIn("interactive", kw)
        self.assertFalse(ast.literal_eval(kw["interactive"]))

        # Execute ratio_save_click and ratio_delete_click with mock resolution_settings.
        save_fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "ratio_save_click"
        )
        delete_fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "ratio_delete_click"
        )
        save_fn.decorator_list = []
        delete_fn.decorator_list = []
        scope = dict(
            gr=gr,
            resolution_settings=SimpleNamespace(
                is_custom=lambda name: name == "My Custom",
                save_custom=lambda *a: "My Custom",
                remove_custom=lambda *a: None,
                choices_for_model=lambda *a: ["1024x1024"],
                selection_for_model=lambda *a: "1024x1024",
            ),
        )
        exec(
            compile(ast.Module(body=[save_fn, delete_fn], type_ignores=[]), "app.py", "exec"), scope
        )

        with self.assertRaises(gr.Error) as raised:
            scope["ratio_save_click"]("   ", 1024, 1024, "SDXL")
        self.assertIn("name", raised.exception.message.lower())
        self.assertFalse(raised.exception.print_exception)

        with self.assertRaises(gr.Error) as raised:
            scope["ratio_delete_click"]("1024x1024", "SDXL")
        self.assertIn("custom size", raised.exception.message.lower())
        self.assertFalse(raised.exception.print_exception)

    def test_settings_save_as_and_theme_download_guards(self):
        source = (ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        download_fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "download_theme_clicked"
        )
        scope = dict(gr=gr, t=lambda s: s)
        exec(
            compile(ast.Module(body=[download_fn], type_ignores=[]), "ui_settings.py", "exec"),
            scope,
        )
        with self.assertRaises(gr.Error) as raised:
            scope["download_theme_clicked"]("   ")
        self.assertIn("theme", raised.exception.message.lower())
        self.assertFalse(raised.exception.print_exception)

        # Verify download_theme button starts interactive=False in AST.
        download_assign = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "download_theme" for t in node.targets)
            and isinstance(node.value, ast.Call)
        )
        kw = {k.arg: k.value for k in download_assign.value.keywords}
        self.assertIn("interactive", kw)
        self.assertFalse(ast.literal_eval(kw["interactive"]))

        # Verify save_as_btn starts interactive=False when settings.name is empty.
        function = next(
            node for node in tree.body if getattr(node, "name", None) == "create_settings"
        )
        layout = function.body[0]
        for node in layout.body:
            if isinstance(node, ast.With) and any(
                isinstance(item.optional_vars, ast.Name)
                and item.optional_vars.id == "settings_tabs"
                for item in node.items
            ):
                node.body = ast.parse(
                    'with gr.Tab("Fixture") as models_tab:\n'
                    '    field = gr.Slider(0, 10, value=5, label="Fixture setting")\n'
                    '    add_setting("fixture", field)\n'
                ).body
        endpoint = next(
            index
            for index, node in enumerate(layout.body)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "bind_navigation"
        )
        layout.body = layout.body[: endpoint + 1]
        registry = {"setting_name": [], "setting_obj": []}

        def register(name, control):
            registry["setting_name"].append(name)
            registry["setting_obj"].append(control)

        scope = dict(
            gr=gr,
            t=lambda text: text,
            state=registry,
            add_setting=register,
            settings=SimpleNamespace(name="", default_settings={}),
            path_manager=SimpleNamespace(paths={}),
            SETTINGS_INDEX={},
            models_in_settings=True,
        )
        with patch("ruinedfooocus.core.profile_store.selected_profile", return_value=""):
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[layout], type_ignores=[])),
                    "Settings guards",
                    "exec",
                ),
                scope,
            )
        self.assertFalse(scope["save_as_btn"].interactive)
        app = scope["app_settings"]
        sync = next(
            fn.fn
            for fn in app.fns.values()
            if fn.fn is not None and fn.outputs == [scope["save_as_btn"]]
        )
        self.assertFalse(sync("  ")["interactive"])
        self.assertTrue(sync("studio")["interactive"])


if __name__ == "__main__":
    unittest.main()
