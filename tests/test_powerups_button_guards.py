"""Button interactivity guards, panel visibility, validation errors, and tool ordering in Power-ups."""

import ast
import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.state_holder import SessionState
from PIL import Image

import ruinedfooocus.app as app
from ruinedfooocus.features.image_editing import lora_hooks, ui_guidance, ui_lora_hooks
from ruinedfooocus.features.image_processing import (
    latent_composition,
    ui_image_composition,
    ui_image_histogram,
    ui_latent_composition,
    ui_native_formats,
    ui_native_vision,
)
from test_powerup_family_controls import functions as powerup_functions

ROOT = Path(__file__).resolve().parents[1]
POWERUP_UI_SOURCE = ROOT / "ruinedfooocus/features/image_editing/ui.py"
POWERUP_DEFAULTS = ROOT / "ruinedfooocus/resources/defaults/powerup.default"


def button_by_label(demo, expected):
    from ruinedfooocus.core.translation_manager import translate_ui

    expected_str = str(translate_ui(expected))
    for block in demo.blocks.values():
        if not isinstance(block, gr.Button):
            continue
        if str(block.value) in (expected, expected_str):
            return block
    raise KeyError(expected)


def extract_function(source_path, name, scope):
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), scope)
    return scope[name]


class PowerupsVisibilityAndOrderingTests(unittest.TestCase):
    def test_generation_context_and_sd4x_hide_for_image_to_prompt_and_utilities(self):
        scope, models = powerup_functions()
        scope["generation_groups"] = [object(), object()]
        generation_context = extract_function(POWERUP_UI_SOURCE, "generation_context", scope)
        for selection, expected in (
            ("None", True),
            ("Img2Img", True),
            ("Canny", True),
            ("Image to prompt", False),
            ("Remove background", False),
        ):
            with self.subTest(selection=selection):
                updates = generation_context(selection, "Canny")
                self.assertEqual([u["visible"] for u in updates], [expected, expected])
        models.detect_checkpoint_base.return_value = "SD_X4Upscaler"
        self.assertFalse(scope["sd4x_output_info"](None, "selected", "Image to prompt")["visible"])

    def test_source_status_text_hides_for_tools_without_single_input_image(self):
        scope, _ = powerup_functions()
        scope["localize"] = lambda text, locale=None: text
        presets = json.loads(POWERUP_DEFAULTS.read_text(encoding="utf-8"))
        scope["controlnet"].cn_options = presets
        source_status_text = extract_function(POWERUP_UI_SOURCE, "source_status_text", scope)
        for hidden_tool in (
            "None",
            "Qwen Image 2.1 Union",
            "Compose saved latents",
            "Make image grid",
            "Compose images",
        ):
            with self.subTest(tool=hidden_tool):
                self.assertFalse(source_status_text(hidden_tool, None)["visible"])
        self.assertTrue(source_status_text("Image to prompt", None)["visible"])
        self.assertTrue(source_status_text("Img2Img", None)["visible"])

    def test_powerup_choices_are_logically_grouped_and_unique(self):
        scope, _ = powerup_functions()
        scope["controlnet"].cn_options = json.loads(POWERUP_DEFAULTS.read_text(encoding="utf-8"))
        choices = scope["powerup_choices"](
            sdxl=True, native_edit=True, control=True, qwen_union=True
        )
        values = [value for _, value in choices]
        self.assertEqual(values[0], "None")
        self.assertEqual(values[-1], scope["NEWCN"])
        self.assertEqual(len(values), len(set(values)))
        self.assertEqual(values.count("Image to prompt"), 1)
        self.assertLess(values.index("Img2Img"), values.index("ControlNet / T2I adapter"))
        self.assertLess(values.index("ControlNet / T2I adapter"), values.index("Canny (low)"))
        self.assertLess(values.index("Canny (low)"), values.index("Restore still image (SeedVR2)"))
        self.assertLess(values.index("Restore still image (SeedVR2)"), values.index("Detect objects"))
        self.assertLess(values.index("Detect objects"), values.index("Blur image"))
        self.assertLess(values.index("Blur image"), values.index("Make image grid"))

    def test_save_tool_preset_guard_and_blank_name_error(self):
        scope, _ = powerup_functions()
        scope["gr"] = NS(update=lambda **kwargs: kwargs, Error=gr.Error)
        scope.update(cn_sliders=[None] * 6, cn_outputs=[None] * 3)
        save_button_state = extract_function(POWERUP_UI_SOURCE, "save_button_state", scope)
        cn_changed = extract_function(POWERUP_UI_SOURCE, "cn_changed", scope)
        cn_type_changed = extract_function(POWERUP_UI_SOURCE, "cn_type_changed", scope)
        scope["cn_type_changed"] = cn_type_changed
        cn_save = extract_function(POWERUP_UI_SOURCE, "cn_save", scope)

        self.assertFalse(save_button_state("")["interactive"])
        self.assertFalse(save_button_state("   ")["interactive"])
        self.assertTrue(save_button_state("My Canny")["interactive"])

        new_updates = cn_changed(scope["NEWCN"], "Canny", "selected")
        self.assertEqual(new_updates[1], {"visible": True, "interactive": False})

        with self.assertRaisesRegex(gr.Error, "Enter a tool preset name"):
            cn_save("   ", None, "Canny", 0.0, 1.0, 1.0, 0.2, 0.8, "None", "selected")

    def test_image_guidance_hides_for_image_to_prompt_and_composition(self):
        registered = {}
        shared = NS(
            settings=NS(default_settings={}),
            path_manager=object(),
            add_ctrl=lambda key, value: registered.update({key: value}),
            get_ctrl=lambda key: registered.get(key),
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(app, "state", shared, create=True),
            patch.object(ui_guidance, "family_for_checkpoint", return_value="SDXL"),
        ):
            with gr.Blocks(analytics_enabled=False) as demo:
                base_model = gr.Textbox(value="sdxl.safetensors")
                tool_selection = gr.Textbox(value="None")
                tool_type = gr.Textbox(value="Canny")
                ui_guidance.create_ui(
                    base_model,
                    tool_selection=tool_selection,
                    tool_type=tool_type,
                    kind_for=lambda sel, kind: (
                        "image-to-prompt"
                        if sel == "Image to prompt"
                        else sel
                    ),
                )
        fn_index = next(
            idx
            for idx, fn in demo.fns.items()
            if fn.inputs == [tool_selection, tool_type]
        )
        for tool, expected in (
            ("None", True),
            ("img2img", True),
            ("Image to prompt", False),
            ("latent-compose", False),
            ("image-grid", False),
            ("image-composite", False),
        ):
            with self.subTest(tool=tool):
                out = asyncio.run(demo.process_api(fn_index, [tool, ""], state=None))["data"][0]
                self.assertEqual(out["visible"], expected)


class PowerupsButtonGuardTests(unittest.TestCase):
    def test_histogram_hides_for_non_applicable_tools_and_guards_buttons(self):
        self.assertFalse(ui_image_histogram.histogram_visible("Image to prompt", object()))
        self.assertFalse(ui_image_histogram.histogram_visible("Compose saved latents", object()))
        self.assertFalse(ui_image_histogram.histogram_visible("Make image grid", object()))
        self.assertFalse(ui_image_histogram.histogram_visible("Compose images", object()))
        self.assertFalse(ui_image_histogram.histogram_visible("Qwen Image 2.1 Union", object()))
        self.assertTrue(ui_image_histogram.histogram_visible("Img2Img", None))
        self.assertFalse(ui_image_histogram.histogram_visible("None", None))

        with gr.Blocks(analytics_enabled=False) as demo:
            source = gr.Image(type="pil", image_mode="RGBA")
            selection = gr.Textbox(value="None")
            ui_image_histogram.create_ui(source, tool_selection=selection)

        self.assertFalse(button_by_label(demo, "Analyze histogram").interactive)
        self.assertFalse(button_by_label(demo, "Cancel analysis").interactive)

        sync_idx = next(
            idx for idx, fn in demo.fns.items() if fn.fn and fn.fn.__name__ == "sync_analyze_button"
        )
        self.assertFalse(asyncio.run(demo.process_api(sync_idx, [None], state=None))["data"][0]["interactive"])

        with self.assertRaisesRegex(gr.Error, "Add one source image"):
            asyncio.run(ui_image_histogram.analyze_source(None))

    def test_native_vision_undo_and_clear_buttons_follow_point_state(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.png"
            Image.new("RGBA", (64, 64), (10, 20, 30, 255)).save(path)
            registered = {}
            shared = NS(
                settings=NS(default_settings={}),
                path_manager=object(),
                add_ctrl=lambda key, obj: registered.update({key: obj}),
            )
            with (
                patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
                patch.object(app, "state", shared, create=True),
                patch.object(ui_native_vision, "native_model_choices", lambda *_: [("sam", "/sam")]),
                patch.object(ui_native_vision, "coco_classes", lambda: ["person"]),
            ):
                with gr.Blocks(analytics_enabled=False) as demo:
                    selection = gr.Textbox(value="Custom...")
                    tool = gr.Textbox(value="segment")
                    image = gr.Image(type="pil", image_mode="RGBA", format="png")
                    refresh = gr.Number(value=0)
                    ui_native_vision.create_ui(selection, tool, image, refresh, lambda s, k: k)
            self.assertFalse(button_by_label(demo, "Undo last point").interactive)
            self.assertFalse(button_by_label(demo, "Clear points").interactive)

            state = SessionState(demo)
            add_idx = next(idx for idx, fn in demo.fns.items() if fn.fn.__name__ == "add_point")
            clear_idx = next(idx for idx, fn in demo.fns.items() if fn.fn.__name__ == "clear")
            source_payload = {"path": str(path), "meta": {"_type": "gradio.FileData"}}
            event = gr.SelectData(image, {"index": [10, 12], "value": None})
            added = asyncio.run(
                demo.process_api(
                    add_idx,
                    [source_payload, "Custom...", "segment", ui_native_vision.POSITIVE, "[]", "[]", None],
                    state=state,
                    event_data=event,
                )
            )["data"]
            self.assertTrue(added[5]["interactive"])
            self.assertTrue(added[6]["interactive"])

            cleared = asyncio.run(demo.process_api(clear_idx, [], state=state))["data"]
            self.assertFalse(cleared[5]["interactive"])
            self.assertFalse(cleared[6]["interactive"])

    def test_image_composition_buttons_start_disabled_and_track_selection(self):
        registered = {}
        shared = NS(
            settings=NS(default_settings={}),
            add_ctrl=lambda key, obj: registered.update({key: obj}),
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(app, "state", shared, create=True),
        ):
            with gr.Blocks(analytics_enabled=False) as demo:
                selection = gr.Textbox(value="image-grid")
                kind = gr.Textbox(value="ignored")
                ui_image_composition.create_ui(selection, kind, lambda s, k: s)

        self.assertFalse(button_by_label(demo, "Move selected earlier").interactive)
        self.assertFalse(button_by_label(demo, "Move selected later").interactive)
        self.assertFalse(button_by_label(demo, "Clear image inputs").interactive)

        fn = next(item.fn for item in demo.fns.values() if item.fn.__name__ == "button_states")
        self.assertEqual([u["interactive"] for u in fn([], None, None)], [False, False, False])
        self.assertEqual([u["interactive"] for u in fn(["a", "b"], 0, None)], [False, True, True])
        self.assertEqual([u["interactive"] for u in fn(["a", "b"], 1, None)], [True, False, True])
        self.assertEqual([u["interactive"] for u in fn([], None, object())], [False, False, True])

    def test_latent_composition_clear_button_tracks_inputs(self):
        registered = {}
        shared = NS(
            settings=NS(default_settings={}),
            add_ctrl=lambda name, obj: registered.update({name: obj}),
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(app, "state", shared, create=True),
        ):
            with gr.Blocks(analytics_enabled=False) as demo:
                selection = gr.Textbox(value="Compose saved latents")
                kind = gr.Textbox(value=latent_composition.KIND)
                ui_latent_composition.create_ui(selection, kind, lambda s, v: v)

        clear_btn = next(b for b in demo.blocks.values() if isinstance(b, gr.Button))
        self.assertFalse(clear_btn.interactive)
        fn = next(item.fn for item in demo.fns.values() if item.fn.__name__ == "clear_button_state")
        self.assertFalse(fn(None, None, None)["interactive"])
        self.assertTrue(fn(b"x", None, None)["interactive"])
        self.assertTrue(fn(None, None, object())["interactive"])

    def test_native_formats_import_guard_and_export_missing_selection_error(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = NS(model_paths={"temp_outputs_path": folder, "cache_path": folder})
            with gr.Blocks(analytics_enabled=False) as demo:
                selected = gr.State(None)
                ui_native_formats.create_library_ui(selected, manager)

        self.assertFalse(button_by_label(demo, "Export selected image").interactive)
        self.assertFalse(button_by_label(demo, "Import original copy").interactive)
        fn = next(item.fn for item in demo.fns.values() if item.fn.__name__ == "import_button_state")
        self.assertFalse(fn(None)["interactive"])
        self.assertTrue(fn("sample.png")["interactive"])

        with self.assertRaisesRegex(gr.Error, "Select an image"):
            ui_native_formats.export_selection(None, "png", "8-bit", "sRGB", 18, folder)

    def test_lora_hooks_hides_for_image_to_prompt_and_guards_copy_button(self):
        registered = {}
        shared = NS(
            settings=NS(default_settings={}),
            path_manager=object(),
            add_ctrl=lambda key, value: registered.update({key: value}),
            get_ctrl=lambda key: registered.get(key),
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(app, "state", shared, create=True),
            patch(
                "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                return_value="SDXL",
            ),
            patch.object(lora_hooks, "choices", return_value=[]),
        ):
            with gr.Blocks(analytics_enabled=False) as demo:
                registered["loras"] = gr.Gallery(value=[])
                checkpoint = gr.Textbox(value="SDXL")
                selection = gr.Textbox(value="None")
                kind = gr.Textbox(value="")
                refresh = gr.Number(value=0)
                ui_lora_hooks.create_ui(
                    checkpoint,
                    selection,
                    kind,
                    refresh,
                    lambda s, k: "image-to-prompt" if s == "Image to prompt" else s,
                )

        copy_btn = next(b for b in demo.blocks.values() if isinstance(b, gr.Button))
        self.assertFalse(copy_btn.interactive)
        sync_fn = next(
            item.fn for item in demo.fns.values() if item.fn.__name__ == "sync_copy_selected"
        )
        self.assertFalse(sync_fn([])["interactive"])
        self.assertTrue(sync_fn([("thumb.png", "1.0 - style.safetensors")])["interactive"])

        ctx_idx = next(idx for idx, fn in demo.fns.items() if fn.fn.__name__ == "context")
        for tool in ("Image to prompt", "latent-compose", "image-grid", "image-composite"):
            with self.subTest(tool=tool):
                out = asyncio.run(
                    demo.process_api(ctx_idx, ["SDXL", "off", "", tool, "", 0, "lora"], state=None)
                )["data"]
                self.assertFalse(out[0]["visible"])


if __name__ == "__main__":
    unittest.main()
