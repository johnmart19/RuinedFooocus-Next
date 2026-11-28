"""Hub activation retains actual compatible Dropdown labels and all choices."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState

from ruinedfooocus.features.image_processing import ui_native_vision
from ui_test_helpers import function_from


class ToolActivationChoiceTests(unittest.TestCase):
    def test_hub_sam_activation_reuses_complete_friendly_choices_in_real_gradio_callback(self):
        choices = [
            ("sam3.1_multiplex.safetensors · SAM31", "D:/fixture/sam3.1_multiplex.safetensors"),
            ("sam3.safetensors · SAM3", "D:/fixture/sam3.safetensors"),
        ]
        manager = object()
        with gr.Blocks(analytics_enabled=False) as app:
            picker = gr.Dropdown(
                choices=[("RT-DETR", "D:/fixture/detector.safetensors")],
                value="D:/fixture/detector.safetensors",
            )
            operation = gr.Dropdown(choices=["Segment"], value="Segment")
            target = gr.Radio(choices=["Generated image"], value="Generated image")
            button = gr.Button()
            controls = dict(
                native_vision_model=picker, cn_selection=operation, automatic_upscale_target=target
            )
            checkpoint_features = Mock(
                side_effect=AssertionError("SAM activation must retain the image checkpoint")
            )
            activate = function_from(
                "ruinedfooocus/features/image_editing/ui.py",
                "activate_tool",
                dict(
                    gr=gr,
                    shared=SimpleNamespace(get_ctrl=controls.get, path_manager=manager),
                    controlnet=SimpleNamespace(cn_options={"Segment": {"type": "segment"}}),
                    NEWCN="Custom...",
                    checkpoint_features=checkpoint_features,
                ),
            )

            def use():
                return activate(
                    "segment",
                    dict(kind="segment", resource="native_vision_model", value=choices[0][1]),
                    checkpoint="unchanged image.safetensors",
                )

            event = button.click(use, outputs=list(controls.values()))
        state = SessionState(app)
        with patch.object(
            ui_native_vision, "native_model_choices", return_value=choices.copy()
        ) as scan:
            result = asyncio.run(app.process_api(event["id"], [], state=state))
        scan.assert_called_once_with(manager, "segment")
        checkpoint_features.assert_not_called()
        projected = result["data"][0]
        self.assertEqual(projected["choices"], [list(value) for value in choices])
        self.assertEqual(projected["value"], choices[0][1])
        actual = state.blocks_config.blocks[picker._id]
        self.assertIsInstance(actual, gr.Dropdown)
        self.assertEqual(actual.choices, choices)
        self.assertEqual(actual.value, choices[0][1])
        self.assertNotIn(choices[0][1], {label for label, _ in actual.choices})


if __name__ == "__main__":
    unittest.main()
