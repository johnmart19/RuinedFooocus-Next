"""Size paste and validation stay bounded without treating input mistakes as crashes."""

import ast
import asyncio
from pathlib import Path
import subprocess
import unittest

import gradio as gr

from ruinedfooocus.features.generation.dimension_number import DimensionNumber

ROOT = Path(__file__).resolve().parents[1]


class DimensionInputTests(unittest.TestCase):
    def test_native_dimension_components_keep_their_renderer_bounds_and_ids(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8"))
        nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "DimensionNumber"
        ]
        scope = dict(
            DimensionNumber=DimensionNumber,
            t=str,
            default_resolution=(1440, 2560),
            custom_resolution=True,
        )
        with gr.Blocks() as app:
            exec(
                compile(
                    ast.Module(body=nodes, type_ignores=[]), "actual dimension controls", "exec"
                ),
                scope,
            )
        numbers = [c for c in app.config["components"] if c["type"] == "number"]
        self.assertEqual(len(numbers), 2)
        self.assertEqual(
            [c["props"]["elem_id"] for c in numbers],
            ["output-target-width", "output-target-height"],
        )
        self.assertEqual(scope["custom_width"].preprocess(1440), 1440)
        self.assertEqual(scope["custom_height"].preprocess(2560), 2560)

    def test_invalid_dimensions_never_call_the_action_and_keep_readable_quiet_errors(self):
        seen = []
        with gr.Blocks() as app:
            dimension = DimensionNumber(minimum=1, maximum=32768, precision=0)
            button = gr.Button()
            button.click(lambda value: seen.append(value), inputs=dimension)
        for value in (0, 14402560):
            with self.subTest(value=value), self.assertRaises(gr.Error) as raised:
                asyncio.run(app.process_api(0, [value]))
            self.assertFalse(raised.exception.print_exception)
            self.assertIn("value", raised.exception.message.lower())
        self.assertEqual(seen, [])
        asyncio.run(app.process_api(0, [2560]))
        self.assertEqual(seen, [2560])

    def test_dimension_paste_browser_handlers(self):
        result = subprocess.run(
            ["node", "tests/test_dimension_inputs.js"], cwd=ROOT, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        source = (ROOT / "ruinedfooocus/ui/html.py").read_text(encoding="utf-8")
        self.assertIn('asset_path("dimension_inputs.js")', source)
