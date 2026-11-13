"""Queued tool values use current choices when checkpoint capabilities change."""

import ast
from pathlib import Path
import unittest

import gradio as gr

from ruinedfooocus.features.image_editing.tool_type_dropdown import ToolTypeDropdown
from test_powerup_family_controls import functions

ROOT = Path(__file__).resolve().parents[1]


class ImageToolTransitionTests(unittest.TestCase):
    def test_known_stale_type_is_normalized_in_both_checkpoint_directions(self):
        scope, models = functions()
        scope["controlnet"].controlnet_models["edit"] = None
        scope["controlnet"].cn_options["Native image edit"] = {"type": "edit"}
        source = ROOT / "ruinedfooocus/features/image_editing/ui.py"
        callback = next(
            node
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef) and node.name == "model_changed"
        )
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(source), "exec"), scope)
        for family, old_type in (
            ("SD15_instructpix2pix", "Img2img"),
            ("SDXL_instructpix2pix", "Img2img"),
            ("SD_X4Upscaler", "Img2img"),
            ("Flux", "Edit"),
            ("PiD", "Control"),
        ):
            with self.subTest(family=family, old_type=old_type):
                models.detect_checkpoint_base.return_value = family
                update = scope["model_changed"]("selected", "None")[1]
                # Gradio reconstructs the session component from the updated
                # choices, separately from the browser's value update.
                current = ToolTypeDropdown(choices=update["choices"], render=False)
                self.assertEqual(current.preprocess(old_type), update["value"])
                self.assertFalse(current.allow_custom_value)
                self.assertEqual(current.get_block_name(), "dropdown")

    def test_supported_type_is_preserved_and_unknown_values_are_rejected(self):
        dropdown = ToolTypeDropdown(choices=["Edit", "Upscale", "Rembg"], render=False)
        self.assertEqual(dropdown.preprocess("Upscale"), "Upscale")
        self.assertIsNone(dropdown.preprocess(None))
        with self.assertRaises(gr.Error):
            dropdown.preprocess("arbitrary unsupported tool")

    def test_real_callback_graph_accepts_stale_value_after_session_choice_update(self):
        import asyncio
        from gradio.state_holder import SessionState

        with gr.Blocks(analytics_enabled=False) as demo:
            tool = ToolTypeDropdown(choices=["Img2img", "Upscale", "Rembg"], value="Img2img")
            switch = gr.Button()
            switch.click(
                lambda: gr.update(choices=["Edit", "Upscale", "Rembg"], value="Edit"),
                outputs=tool,
            )
            tool.change(lambda value: value, inputs=tool, outputs=tool)
        state = SessionState(demo)

        async def transition():
            await demo.process_api(0, [], state=state)
            # The browser's dependent event can still carry the previous model's value.
            result = await demo.process_api(1, ["Img2img"], state=state)
            self.assertEqual(result["data"], ["Edit"])

        asyncio.run(transition())


if __name__ == "__main__":
    unittest.main()
