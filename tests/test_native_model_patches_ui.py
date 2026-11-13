"""Real registered Gradio patch controls retain family and source ownership."""

import asyncio
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import gradio as gr
import ruinedfooocus.app
from ruinedfooocus.features.image_editing import model_patches, native_control_recipes
from ruinedfooocus.features.image_editing import ui_model_patches as ui


class NativePatchUITests(unittest.TestCase):
    def test_dynamic_file_labels_resolve_text_before_combining_with_filenames(self):
        from gradio.i18n import I18nData

        with (
            patch.object(model_patches, "local_choices", return_value=[("opaque.pt", "unknown")]),
            patch.object(ui, "t", side_effect=lambda text: I18nData(text)),
        ):
            values = ui.choices(object(), "anima-lllite", "missing.pt", locale="en")
        self.assertEqual(values[1], ("opaque.pt · Native validation on Run", "opaque.pt"))
        self.assertEqual(values[2], ("Unavailable: missing.pt", "missing.pt"))
        self.assertTrue(all("__i18n__" not in label for label, _ in values[1:]))

    def build(self, family="Flux", defaults=None):
        registered = {}
        shared = NS(
            settings=NS(default_settings=defaults or {}),
            path_manager=object(),
            add_ctrl=lambda key, value: registered.update({key: value}),
        )
        self.enterContext(patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}))
        self.enterContext(patch.object(ruinedfooocus.app, "state", shared, create=True))
        self.enterContext(
            patch(
                "ruinedfooocus.features.image_editing.ui_guidance.family_for_checkpoint",
                side_effect=lambda value: value,
            )
        )
        self.enterContext(
            patch.object(
                model_patches, "local_choices", return_value=[("local.safetensors", "compatible")]
            )
        )
        self.enterContext(
            patch.object(native_control_recipes, "union_types", return_value=("auto", "depth"))
        )
        with gr.Blocks(analytics_enabled=False) as demo:
            checkpoint = gr.Textbox(value=family)
            selection, kind = gr.Textbox(value="Custom..."), gr.Textbox(value="Control")
            refresh = gr.Number(value=0)
            controls = ui.create_ui(
                checkpoint,
                selection,
                kind,
                refresh,
                lambda selected, kind: kind.lower() if selected == "Custom..." else None,
            )
        return demo, controls, registered, checkpoint

    def test_fields_are_registered_once_and_source_uploads_never_come_from_defaults(self):
        _, controls, registered, _ = self.build(
            "ZImage", dict(model_patch_recipe="zimage-fun", model_patch_image="stale.png")
        )
        self.assertEqual(
            set(registered),
            set(
                model_patches.FIELDS
                + model_patches.IMAGE_FIELDS
                + native_control_recipes.FIELDS
                + native_control_recipes.IMAGE_FIELDS
            ),
        )
        self.assertEqual(set(controls), set(registered))
        for name in model_patches.IMAGE_FIELDS + native_control_recipes.IMAGE_FIELDS:
            self.assertIsNone(controls[name].value)

    def test_actual_family_callback_hides_unsupported_patch_and_uso_parameter_controls(self):
        demo, controls, _, checkpoint = self.build(defaults=dict(model_patch_recipe="uso-style"))
        event = next(
            index for index, function in demo.fns.items() if function.fn.__name__ == "context"
        )
        self.assertIs(demo.fns[event].inputs[0], checkpoint)
        update = asyncio.run(
            demo.process_api(
                event,
                ["Flux", "uso-style", "local.safetensors", "local.safetensors", "None", ""],
                state=None,
            )
        )["data"]
        self.assertFalse(update[4]["visible"])
        self.assertEqual([item["value"] for item in update[7:]], [1.0, 0.0, 1.0])
        update = asyncio.run(
            demo.process_api(
                event,
                ["SDXL", "uso-style", "local.safetensors", "local.safetensors", "None", ""],
                state=None,
            )
        )["data"]
        self.assertFalse(update[0]["visible"])
        self.assertEqual(update[1]["value"], "none")

    def test_actual_control_context_offers_family_specific_recipes_and_dedicated_canvas_controls(
        self,
    ):
        demo, _, _, _ = self.build("QwenImage")
        event = next(
            index
            for index, function in demo.fns.items()
            if function.fn.__name__ == "control_context"
        )
        update = asyncio.run(
            demo.process_api(
                event,
                ["QwenImage", "Custom...", "Control", "alimama-inpaint", "outpaint"],
                state=None,
            )
        )["data"]
        self.assertEqual(
            [value for _, value in update[1]["choices"]], ["ordinary", "alimama-inpaint"]
        )
        self.assertFalse(update[5]["visible"])
        self.assertTrue(update[6]["visible"])
        update = asyncio.run(
            demo.process_api(
                event, ["Flux", "Custom...", "Control", "alimama-inpaint", "mask"], state=None
            )
        )["data"]
        self.assertEqual(update[1]["value"], "ordinary")
        self.assertEqual([value for _, value in update[1]["choices"]], ["ordinary", "union"])
        update = asyncio.run(
            demo.process_api(
                event, ["QwenImage", "Rembg", "Control", "alimama-inpaint", "mask"], state=None
            )
        )["data"]
        self.assertFalse(update[0]["visible"])


if __name__ == "__main__":
    unittest.main()
