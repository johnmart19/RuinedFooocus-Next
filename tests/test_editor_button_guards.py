"""Empty-state button guards, dynamic interactivity and clean notices for Editor & Gallery."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr
from PIL import Image

from ruinedfooocus.app.resource_lifecycle import ui_request_owner
from ruinedfooocus.core.paths import asset_path
from ruinedfooocus.features.gallery.editor_session import EditorSessionStore
from ruinedfooocus.features.gallery.ui_workspace import EditorWorkspace


def _find_event(app, trigger_component, trigger_name="click", fn_name=None):
    matches = []
    for block_fn in app.fns.values():
        if fn_name is not None and (
            block_fn.fn is None or getattr(block_fn.fn, "__name__", None) != fn_name
        ):
            continue
        for target_id, event_name in block_fn.targets:
            if target_id == trigger_component._id and event_name == trigger_name:
                matches.append(block_fn)
                break
    if not matches:
        raise AssertionError(
            f"No event found for {type(trigger_component).__name__}.{trigger_name}"
            + (f" ({fn_name})" if fn_name else "")
        )
    return matches[0]


class EditorButtonGuardTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.request = gr.Request(session_hash="guard-owner")
        self.owner = ui_request_owner(self.request)
        self.store = EditorSessionStore(self.root / "sessions")
        self.source = self.root / "source.png"
        Image.new("RGBA", (64, 48), (20, 40, 60, 255)).save(self.source)
        self.edited = self.root / "edited.png"
        Image.new("RGBA", (64, 48), (80, 120, 160, 255)).save(self.edited)

        with gr.Blocks(analytics_enabled=False) as self.app:
            self.workspace = EditorWorkspace(store=self.store)
            self.controls = {
                "_editor_context": self.workspace.context,
                "_editor_source_input": self.workspace.source_input,
                "_editor_destination": self.workspace.destination,
                "_editor_action": self.workspace.action,
                "_editor_denoise": self.workspace.denoise,
                "_editor_framing": self.workspace.framing,
                "_editor_mask": self.workspace.mask,
                "_editor_mask_choice": self.workspace.mask_choice,
                "_editor_mask_index": self.workspace.mask_index,
                "_editor_mask_count": self.workspace.mask_count,
                "_editor_mask_invert": self.workspace.invert,
                "_editor_background": self.workspace.background,
                "_editor_prompt_processing": self.workspace.prompt_processing,
                "_editor_upscale_target": self.workspace.upscale_target,
                "prompt": gr.Textbox(""),
                "negative": gr.Textbox(""),
                "base_model_name": gr.Dropdown(["fixture.safetensors"], value="fixture.safetensors"),
                "aspect_ratios_selection": gr.Dropdown(["Custom..."], value="Custom..."),
                "custom_width": gr.Number(64),
                "custom_height": gr.Number(48),
                "resolution_auto": gr.Checkbox(True),
                "image_source": gr.State(None),
                "input_image": gr.Image(type="pil", value=None),
                "cn_selection": gr.Dropdown(["None"], value="None"),
                "cn_type": gr.Dropdown([""], value=""),
                "main_view": gr.Image(type="filepath", value=str(asset_path("init_image.png"))),
                "_native_main_source": gr.State(None),
            }
            self.run_controls = {
                "preview": self.controls["main_view"],
                "native_source": self.controls["_native_main_source"],
                "gallery": gr.Gallery(value=[]),
                "inpaint_toggle": gr.Checkbox(False),
                "inpaint": gr.ImageEditor(value=None),
                "run": gr.Button("Generate"),
                "stop": gr.Button("Stop", visible=False),
                "progress": gr.HTML(visible=False),
                "task": gr.State(None),
                "artifacts": gr.State([]),
                "hint": gr.Markdown(""),
            }
            self.workspace.bind(
                self.controls,
                run_request=lambda data: iter(()),
                stop_request=lambda task_id: None,
                run_controls=self.run_controls,
                guide_control_plan=lambda spec, name, selected, signature: {},
            )
            self.chat_open_btn = gr.Button("Open chat result")
            self.chat_result = gr.State(None)
            self.workspace.bind_chat_open(self.chat_open_btn, self.chat_result)
            self.lib_open_btn = gr.Button("Open in Editor")
            self.lib_selected = gr.State(None)
            self.lib_metadata = gr.JSON({})
            self.lib_prompt_version = gr.Radio(["Rendered prompt"], value="Rendered prompt")
            self.workspace.bind_library(
                self.lib_open_btn,
                self.lib_selected,
                self.lib_metadata,
                self.lib_prompt_version,
                validate_record=lambda record, req: None,
            )
            self.workspace.bind_picker(browser=None)

    def test_initial_button_states_are_disabled_until_prerequisites_exist(self):
        for button in (
            self.workspace.open_current,
            self.workspace.undo,
            self.workspace.redo,
            self.workspace.reopen,
            self.workspace.preview,
            self.workspace.picker_previous,
            self.workspace.picker_next,
            self.workspace.picker_open,
            self.workspace.paint.load,
            self.workspace.paint.apply,
        ):
            self.assertFalse(button.interactive, f"{button.value} should start non-interactive")
        self.assertFalse(self.workspace.use_selection.visible)
        self.assertTrue(self.workspace.choose_gallery.interactive)
        self.assertTrue(self.workspace.open_latest.interactive)

    def test_check_resources_empty_state_returns_clean_notice_without_traceback(self):
        from types import SimpleNamespace
        from ruinedfooocus import app as rf_app

        event = _find_event(self.app, self.workspace.preview, "click", "preview_readiness")
        values = [
            None if name == "_editor_context"
            else "placeholder" if name == "_editor_source_input"
            else "editor" if name == "_editor_destination"
            else "prompt-edit" if name == "_editor_action"
            else str(asset_path("init_image.png")) if name == "main_view"
            else ctrl.value
            for name, ctrl in self.controls.items()
        ]
        worker = SimpleNamespace(capture_image_data=lambda data: data)
        with (
            patch.dict("sys.modules", {"ruinedfooocus.app.async_worker": worker}),
            patch.object(rf_app, "async_worker", worker, create=True),
        ):
            notice = event.fn(*values, request=self.request)
            self.assertIsInstance(notice, str)
            self.assertIn("Choose or upload an image", notice)

            # Power-ups action with no tool chosen also returns a clean inline notice.
            values_generate = [
                "generate" if name == "_editor_action" else val
                for name, val in zip(self.controls, values)
            ]
            notice_generate = event.fn(*values_generate, request=self.request)
            self.assertIsInstance(notice_generate, str)
            self.assertIn("Choose a tool in Power-ups", notice_generate)

    def test_sync_action_buttons_tracks_source_and_history_transitions(self):
        sync = _find_event(self.app, self.workspace.context, "change", "sync_action_buttons")

        # Branding placeholder does not enable image-dependent actions.
        placeholder_state = sync.fn(
            None, str(asset_path("init_image.png")), None, "placeholder", request=self.request
        )
        for button in self.workspace.action_buttons:
            self.assertFalse(placeholder_state[button]["interactive"])

        # Real preview image enables Open current image & Check resources, not history/paint.
        preview_state = sync.fn(None, str(self.source), None, "preview", request=self.request)
        self.assertTrue(preview_state[self.workspace.open_current]["interactive"])
        self.assertTrue(preview_state[self.workspace.preview]["interactive"])
        self.assertFalse(preview_state[self.workspace.undo]["interactive"])
        self.assertFalse(preview_state[self.workspace.redo]["interactive"])
        self.assertFalse(preview_state[self.workspace.reopen]["interactive"])
        self.assertFalse(preview_state[self.workspace.paint.load]["interactive"])

        # Freshly opened session enables Reopen & Paint Load, while Undo/Redo remain disabled.
        initial = self.store.start(self.source, owner=self.owner)
        self.assertFalse(initial.can_undo)
        self.assertFalse(initial.can_redo)
        open_state = sync.fn(
            self.workspace.public_context(initial),
            str(self.source),
            None,
            "preview",
            request=self.request,
        )
        self.assertTrue(open_state[self.workspace.reopen]["interactive"])
        self.assertTrue(open_state[self.workspace.paint.load]["interactive"])
        self.assertTrue(open_state[self.workspace.preview]["interactive"])
        self.assertFalse(open_state[self.workspace.undo]["interactive"])
        self.assertFalse(open_state[self.workspace.redo]["interactive"])

        # After one completed step, Undo is enabled and Redo is disabled.
        ticket = self.store.capture(
            initial.session_id,
            initial.revision,
            "processing",
            source_id=initial.current.artifact_id,
            owner=self.owner,
        )
        completed = self.store.complete(
            ticket, self.edited, execution={"kind": "fixture"}, owner=self.owner
        )
        self.assertTrue(completed.can_undo)
        self.assertFalse(completed.can_redo)
        step_updates = sync.fn(
            self.workspace.public_context(completed),
            str(self.edited),
            None,
            "preview",
            request=self.request,
        )
        self.assertTrue(step_updates[self.workspace.undo]["interactive"])
        self.assertFalse(step_updates[self.workspace.redo]["interactive"])

        # After Undo, Undo is disabled again and Redo is enabled.
        undone = self.store.undo(completed.session_id, completed.revision, owner=self.owner)
        self.assertFalse(undone.can_undo)
        self.assertTrue(undone.can_redo)
        undo_updates = sync.fn(
            self.workspace.public_context(undone),
            str(self.source),
            None,
            "preview",
            request=self.request,
        )
        self.assertFalse(undo_updates[self.workspace.undo]["interactive"])
        self.assertTrue(undo_updates[self.workspace.redo]["interactive"])

        # After Redo, Undo is re-enabled and Redo is disabled.
        redone = self.store.redo(undone.session_id, undone.revision, owner=self.owner)
        self.assertTrue(redone.can_undo)
        self.assertFalse(redone.can_redo)
        redo_updates = sync.fn(
            self.workspace.public_context(redone),
            str(self.edited),
            None,
            "preview",
            request=self.request,
        )
        self.assertTrue(redo_updates[self.workspace.undo]["interactive"])
        self.assertFalse(redo_updates[self.workspace.redo]["interactive"])

    def test_empty_and_invalid_clicks_raise_clean_gradio_errors(self):
        # Open current image with no image or branding placeholder.
        open_current = _find_event(self.app, self.workspace.open_current, "click")
        for path in (None, str(asset_path("init_image.png"))):
            with self.assertRaisesRegex(gr.Error, "Open a source image first"):
                open_current.fn(path, None, None, True, None, 1, request=self.request)

        # Undo / Redo / Reopen with no context.
        for btn in (self.workspace.undo, self.workspace.redo, self.workspace.reopen):
            ev = _find_event(self.app, btn, "click")
            with self.assertRaisesRegex(gr.Error, "Open an image history first"):
                ev.fn(None, None, 2, request=self.request)

        # Undo / Redo when history has no earlier/later step -> clean gr.Error, not raw ValueError.
        initial = self.store.start(self.source, owner=self.owner)
        ctx = self.workspace.public_context(initial)
        undo_ev = _find_event(self.app, self.workspace.undo, "click")
        redo_ev = _find_event(self.app, self.workspace.redo, "click")
        with self.assertRaisesRegex(gr.Error, "no earlier step"):
            undo_ev.fn(ctx, None, 3, request=self.request)
        with self.assertRaisesRegex(gr.Error, "no retained editor result to redo"):
            redo_ev.fn(ctx, None, 4, request=self.request)

        # Paint Load & Apply with no context or empty canvas.
        paint_load = _find_event(self.app, self.workspace.paint.load, "click")
        with self.assertRaisesRegex(gr.Error, "Open an image in the Editor first"):
            paint_load.fn(None, request=self.request)
        paint_apply = _find_event(self.app, self.workspace.paint.apply, "click")
        with self.assertRaisesRegex(gr.Error, "Load the current image"):
            paint_apply.fn(None, ctx, None, 5, request=self.request)
        with self.assertRaisesRegex(gr.Error, "Open an image in the Editor first"):
            paint_apply.fn(
                {"composite": Image.new("RGBA", (8, 8), (255, 0, 0, 255))},
                None,
                None,
                6,
                request=self.request,
            )

        # GuidedResultActions use_map with no context.
        use_map = _find_event(self.app, self.workspace.use_selection, "click")
        with self.assertRaisesRegex(gr.Error, "Open a current owned image result first"):
            use_map.fn(None, "fixture.safetensors", "", None, None, [], 7, request=self.request)

        # Chat open & Library open with empty result/record.
        open_chat = _find_event(self.app, self.chat_open_btn, "click")
        with self.assertRaisesRegex(gr.Error, "no owned image"):
            open_chat.fn(None, None, 8, request=self.request)
        open_lib = _find_event(self.app, self.lib_open_btn, "click")
        with self.assertRaisesRegex(gr.Error, "Select an image in Gallery first"):
            open_lib.fn(
                None,
                {},
                "Rendered prompt",
                "",
                "",
                "fixture.safetensors",
                "Custom...",
                64,
                48,
                True,
                None,
                9,
                {},
                request=self.request,
            )

    def test_paint_buttons_enable_on_load_and_picker_close_disables_open(self):
        initial = self.store.start(self.source, owner=self.owner)
        ctx = self.workspace.public_context(initial)
        paint_load = _find_event(self.app, self.workspace.paint.load, "click")
        loaded_img, apply_update = paint_load.fn(ctx, request=self.request)
        self.assertIsInstance(loaded_img, Image.Image)
        self.assertTrue(apply_update["interactive"])

        sync_paint = _find_event(self.app, self.workspace.paint.canvas, "change", "sync_paint_buttons")
        load_u, apply_u = sync_paint.fn(None, {"composite": loaded_img})
        self.assertFalse(load_u["interactive"])
        self.assertFalse(apply_u["interactive"])
        load_u, apply_u = sync_paint.fn(ctx, None)
        self.assertTrue(load_u["interactive"])
        self.assertFalse(apply_u["interactive"])
        load_u, apply_u = sync_paint.fn(ctx, {"composite": loaded_img})
        self.assertTrue(load_u["interactive"])
        self.assertTrue(apply_u["interactive"])

        close_picker = _find_event(self.app, self.workspace.picker_close, "click")
        picker_vis, selected, open_update = close_picker.fn(request=self.request)
        self.assertFalse(picker_vis["visible"])
        self.assertIsNone(selected)
        self.assertFalse(open_update["interactive"])

    def test_gallery_empty_selection_buttons_are_disabled_and_guarded(self):
        from ruinedfooocus.features.gallery import ui as gallery_ui

        with patch.object(gallery_ui.browser, "num_images_pages", return_value=(0, 1)):
            app, _, _, _, _ = gallery_ui.create_image_gallery()
        disabled_labels = {
            "Previous",
            "Next",
            "Use prompt",
            "Reuse full setup",
            "Open in Editor",
            "Edit image",
            "Inpaint",
            "Upscale",
            "Remove background",
            "Start image grid",
            "Start image composition",
        }
        from ruinedfooocus.core.translation_manager import translate_ui

        expected_keys = {
            str(translate_ui(label)): label for label in disabled_labels
        }
        found = {}
        for block in app.blocks.values():
            if isinstance(block, gr.Button):
                raw = str(block.value)
                label = expected_keys.get(raw, raw)
                if label in disabled_labels:
                    found[label] = block.interactive
        self.assertEqual(set(found), disabled_labels)
        for label, interactive in found.items():
            self.assertFalse(interactive, f"Gallery button {label!r} should start non-interactive")


if __name__ == "__main__":
    unittest.main()
