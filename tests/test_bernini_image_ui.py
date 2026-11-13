"""Actual Bernini form events and public request/replay transports on CPU."""

import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.state_holder import SessionState
import ruinedfooocus.app

from ruinedfooocus.features.generation import ui_bernini_image as ui
from ruinedfooocus.features.generation.bernini_image import DEFAULTS, FIELDS
from ruinedfooocus.features.gallery.image_metadata import render_values


class BerniniImageUI(unittest.TestCase):
    def build(self, defaults=None):
        self.enterContext(patch.object(ui, "t", side_effect=lambda text: text))
        shared = NS(
            models=NS(get_names=lambda kind: [], get_file=lambda *a: None),
            settings=NS(default_settings=defaults or {}),
        )
        self.enterContext(patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}))
        self.enterContext(patch.object(ruinedfooocus.app, "state", shared, create=True))
        self.enterContext(
            patch(
                "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                side_effect=lambda name: name,
            )
        )
        self.enterContext(
            patch(
                "ruinedfooocus.features.image_editing.image_guidance.effective_preset_checkpoint",
                side_effect=lambda name, preset, mode: preset or name,
            )
        )
        with gr.Blocks(analytics_enabled=False) as demo:
            model = gr.Textbox(value="BerniniImage")
            preset = gr.Textbox(value="")
            refresh = gr.Number(value=0)
            controls = ui.create_ui(model, preset_selection=preset, model_refresh=refresh)
        return demo, controls

    def test_saved_values_zero_and_unavailable_selected_stage_remain_explicit(self):
        values = dict(
            DEFAULTS,
            bernini_low_model="Missing low stage",
            bernini_split_step=4,
            bernini_high_lora_strength=0,
            bernini_low_lora_strength=0,
        )
        demo, controls = self.build(values)
        self.assertEqual(set(controls), set(FIELDS))
        for key, value in values.items():
            self.assertEqual(controls[key].value, value)
        self.assertIn(
            ("Missing low stage", "Missing low stage"), controls["bernini_low_model"].choices
        )
        self.assertFalse(
            any(
                "length" in getattr(component, "label", "").casefold()
                for component in controls.values()
            )
        )

    def test_actual_gradio_model_and_preset_events_change_only_panel_visibility(self):
        demo, controls = self.build()
        index = next(index for index, fn in demo.fns.items() if fn.fn.__name__ == "bernini_context")
        self.assertEqual(len(demo.fns[index].outputs), 1)
        state = SessionState(demo)

        async def exercise():
            ordinary = await demo.process_api(index, ["Other", ""], state=state)
            preset = await demo.process_api(index, ["Other", "BerniniImage"], state=state)
            unresolved = await demo.process_api(index, [None, None], state=state)
            return ordinary, preset, unresolved

        ordinary, preset, _ = asyncio.run(exercise())
        self.assertFalse(ordinary["data"][0]["visible"])
        self.assertTrue(preset["data"][0]["visible"])
        self.assertEqual(controls["bernini_high_lora_strength"].value, 3)
        self.assertIsNone(controls["bernini_split_step"].value)

    def test_refresh_retains_selected_component_and_does_not_change_sampling(self):
        demo, controls = self.build()
        fn = next(fn for fn in demo.fns.values() if fn.fn.__name__ == "refresh_bernini_choices")
        result = fn.fn("Renamed local low.safetensors")
        self.assertEqual(result["value"], "Renamed local low.safetensors")
        self.assertIn("Renamed local low.safetensors", result["choices"])
        self.assertEqual(fn.outputs, [controls["bernini_low_model"]])

    def test_metadata_fields_roundtrip_and_product_fields_do_not_contain_video_controls(self):
        metadata = dict(
            DEFAULTS,
            native_bernini_image=dict(version=1, source=dict(rgb_sha256="a" * 64)),
            bernini_source_frame=dict(width=928, height=1280),
        )
        restored = render_values(metadata)
        for key, value in metadata.items():
            self.assertEqual(restored[key], value)
        schema = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "ruinedfooocus/resources/bernini_image_components.json"
            ).read_text()
        )
        self.assertEqual(set(schema["components"]), {"high", "low", "clip", "vae", "lora"})
        self.assertTrue(all(len(row["sha256"]) == 64 for row in schema["components"].values()))


if __name__ == "__main__":
    unittest.main()
