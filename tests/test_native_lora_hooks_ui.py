"""Actual mounted Gradio callbacks for scheduled LoRA intent."""

import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from PIL import Image
import ruinedfooocus.app
from ruinedfooocus.features.image_editing import lora_hooks
from ruinedfooocus.features.image_editing import ui_lora_hooks as ui


class LoraHookUITests(unittest.TestCase):
    def build(self, family="SDXL", defaults=None, cards=None):
        registered = {}
        shared = NS(
            settings=NS(default_settings=defaults or {}),
            path_manager=object(),
            add_ctrl=lambda key, value: registered.update({key: value}),
            get_ctrl=lambda key: registered.get(key),
        )
        self.enterContext(patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}))
        self.enterContext(patch.object(ruinedfooocus.app, "state", shared, create=True))
        self.enterContext(
            patch(
                "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                side_effect=lambda value: value,
            )
        )
        self.scan = self.enterContext(
            patch.object(lora_hooks, "choices", return_value=["style.safetensors"])
        )
        with gr.Blocks(analytics_enabled=False) as demo:
            if cards is not None:
                registered["loras"] = gr.Gallery(value=cards)
            checkpoint = gr.Textbox(value=family)
            selection, kind = gr.Textbox(value="None"), gr.Textbox(value="")
            refresh = gr.Number(value=0)
            controls = ui.create_ui(
                checkpoint,
                selection,
                kind,
                refresh,
                lambda selected, tool: tool.lower() if selected == "Custom..." else None,
            )
        return demo, controls, registered, refresh

    def callback(self, demo, values, source_type="lora"):
        event = next(index for index, fn in demo.fns.items() if fn.fn.__name__ == "context")
        return asyncio.run(demo.process_api(event, [*values, source_type], state=None))["data"]

    def test_default_off_registers_all_fields_without_scanning_or_loading_files(self):
        demo, controls, registered, refresh = self.build()
        self.assertEqual(set(controls), set(lora_hooks.FIELDS))
        self.assertEqual(set(registered), set(lora_hooks.FIELDS))
        self.assertEqual(controls["lora_hook_mode"].value, "off")
        raw_count = controls["lora_hook_keyframe_count"].preprocess(7.0)
        self.assertEqual(
            lora_hooks.replay_values(dict(lora_hook_keyframe_count=raw_count))[
                "lora_hook_keyframe_count"
            ],
            7,
        )
        self.scan.assert_not_called()
        self.assertTrue(
            any(
                (refresh._id, "change") in event["targets"] for event in demo.config["dependencies"]
            )
        )

    def test_actual_callback_shows_named_modes_and_preserves_saved_missing_name(self):
        demo, _, _, _ = self.build(defaults={"lora_hook_model": "missing.pt"})
        updates = self.callback(demo, ["SDXL", "keyframes", "missing.pt", "None", "", 0])
        self.assertTrue(updates[0]["visible"])
        self.assertTrue(updates[1]["visible"])
        self.assertIn(["Unavailable: missing.pt", "missing.pt"], updates[2]["choices"])
        self.assertTrue(updates[4]["visible"])
        self.assertFalse(updates[5]["visible"])
        updates = self.callback(demo, ["SDXL", "interpolated", "missing.pt", "None", "", 0])
        self.assertFalse(updates[4]["visible"])
        self.assertTrue(updates[5]["visible"])

    def test_model_only_family_exposes_invalid_clip_for_correction_and_utility_hides_panel(self):
        demo, _, _, _ = self.build()
        updates = self.callback(demo, ["SenseNovaU15", "keyframes", "", "None", "", 0])
        self.assertTrue(updates[0]["visible"])
        self.assertFalse(updates[3]["visible"])
        updates = self.callback(demo, ["SenseNovaU15", "keyframes", "", "None", "", 0.5])
        self.assertTrue(updates[3]["visible"])
        self.assertTrue(updates[6]["visible"])
        self.assertTrue(all("value" not in row for row in updates))
        updates = self.callback(demo, ["SDXL", "off", "", "Custom...", "Segment", 0])
        self.assertFalse(updates[0]["visible"])

    def test_modern_native_scheduled_clip_controls_are_contextual(self):
        demo, _, _, _ = self.build("Flux")
        for family in ("Flux", "QwenImage21", "Anima", "MOSSImage", "ZImage"):
            updates = self.callback(demo, [family, "keyframes", "", "None", "", 0.5])
            self.assertTrue(updates[0]["visible"])
            self.assertTrue(updates[3]["visible"])
            self.assertFalse(updates[6]["visible"])

    def test_local_file_labels_are_actual_text_with_i18n_enabled(self):
        from gradio.i18n import I18nData

        with (
            patch.object(ui, "t", side_effect=lambda text: I18nData(text)),
            patch.object(lora_hooks, "choices", return_value=["owned.pt"]),
        ):
            values = ui.local_choices(object(), "keyframes", "missing.pt", "en")
        self.assertEqual(values[-1], ("Unavailable: missing.pt", "missing.pt"))

    def test_float_list_and_checkpoint_source_use_actual_context_controls(self):
        demo, controls, _, _ = self.build()
        updates = self.callback(demo, ["SDXL", "floats", "", "None", "", 0], "checkpoint")
        self.assertFalse(updates[4]["visible"])
        self.assertFalse(updates[5]["visible"])
        self.assertTrue(updates[7]["visible"])
        self.assertTrue(updates[8]["visible"])
        self.assertEqual(self.scan.call_args.args[1], "checkpoint")
        self.assertEqual(controls["lora_hook_strengths"].value, "[1.0,0.0]")
        self.assertEqual(controls["lora_hook_sources"].value, "[]")

    def test_copy_cards_uses_real_gallery_preprocessing_and_keeps_card_source_intact(self):
        from gradio.components.gallery import GalleryData

        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        thumbnail = root / "thumbnail.png"
        Image.new("RGB", (8, 8)).save(thumbnail)
        cards = [
            (str(thumbnail), "0.75 - detail.safetensors"),
            (str(thumbnail), "-0.5 - other.sft"),
        ]
        demo, controls, registered, _ = self.build(cards=cards)
        event = next(fn for fn in demo.fns.values() if fn.fn.__name__ == "copy_cards")
        self.assertIs(event.inputs[0], registered["loras"])
        self.assertIsInstance(event.inputs[0], gr.Gallery)
        payload = GalleryData(
            root=[dict(image=dict(path=path), caption=caption) for path, caption in cards]
        )
        actual_cards = event.inputs[0].preprocess(payload)
        before = list(actual_cards)
        file_update, list_update, mode_update = event.fn(actual_cards, "off")
        self.assertEqual(actual_cards, before)
        self.assertEqual(file_update["value"], "")
        self.assertEqual(mode_update["value"], "keyframes")
        self.assertEqual(
            [row["name"] for row in json.loads(list_update["value"])],
            ["detail.safetensors", "other.sft"],
        )
        self.assertEqual(controls["lora_hook_mode"].value, "off")


if __name__ == "__main__":
    unittest.main()
