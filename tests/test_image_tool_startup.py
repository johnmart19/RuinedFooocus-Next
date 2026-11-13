"""Construct the production tool selector from complete checkpoint capabilities."""

import ast
from copy import deepcopy
import unittest

import gradio as gr

from test_powerup_family_controls import SOURCE, functions


class ImageToolStartupTests(unittest.TestCase):
    def test_actual_constructor_accepts_complete_standard_and_minimax_capabilities(self):
        for family in ("SDXL 1.0", "MiniMaxH3Image"):
            with self.subTest(family=family):
                scope, models = functions()
                models.get_model_base.return_value = family
                models.detect_checkpoint_base.return_value = family
                scope["controlnet"].cn_options["Native image edit"] = {"type": "edit"}
                scope["controlnet"].controlnet_models["edit"] = None
                registered = {}
                scope.update(gr=gr, add_ctrl=lambda name, value: registered.update({name: value}))
                features = scope["checkpoint_features"]("selected")
                self.assertEqual(features["minimax_image"], family == "MiniMaxH3Image")

                # Execute the real constructor through its first selector. The
                # independent feature panels below it have their own suites.
                tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
                constructor = deepcopy(
                    next(
                        node
                        for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "add_controlnet_tab"
                    )
                )
                tab = constructor.body[1]
                self.assertIsInstance(tab, ast.With)
                tab.body = tab.body[:2]
                constructor.body = [constructor.body[0], tab]
                exec(
                    compile(ast.Module(body=[constructor], type_ignores=[]), str(SOURCE), "exec"),
                    scope,
                )
                with gr.Blocks(analytics_enabled=False):
                    base = gr.Dropdown(choices=["selected"], value="selected")
                    with gr.Tabs():
                        scope["add_controlnet_tab"](None, None, None, None, None, base)

                selector = registered["cn_selection"]
                values = {value for _, value in selector.choices}
                self.assertEqual(selector.value, "None")
                if family == "MiniMaxH3Image":
                    self.assertIn("Native image edit", values)
                    self.assertNotIn("Img2Img", values)
                    types = scope["control_types"](features)
                    self.assertIn("Edit", types)
                    self.assertNotIn("Img2img", types)
                else:
                    self.assertIn("Img2Img", values)


if __name__ == "__main__":
    unittest.main()
