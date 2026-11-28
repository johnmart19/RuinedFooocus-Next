"""Use the actual UI constructors and Gradio codecs without model inference."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import gradio as gr
import numpy as np
from PIL import Image

from ruinedfooocus.features.image_editing.inpaint_utils import editor_payload
from ruinedfooocus.features.image_editing.qwen_image21_local_edit import PAINT_COLOR
from ruinedfooocus.core.translation_manager import translate_ui

ROOT = Path(__file__).resolve().parents[1]


def component(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    call = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    scope = dict(
        gr=gr,
        t=lambda text: text,
        translate_ui=translate_ui,
        initial_features={},
        input_label=lambda _: "Image to image",
        family="QwenImage",
        edit_format=SimpleNamespace(value="Disabled"),
        references_visible=lambda *args: True,
        PAINT_COLOR=PAINT_COLOR,
    )
    return eval(compile(ast.Expression(call), path, "eval"), scope)


def fixture(mode="RGBA"):
    pixels = np.arange(17 * 23 * 4, dtype=np.uint8).reshape(23, 17, 4)
    image = Image.fromarray(pixels).convert(mode)
    image.info["parameters"] = '{"prompt":"Keep fine details."}'
    return image


class ImageInputTransportTests(unittest.TestCase):
    def test_image_tool_roundtrip_preserves_pixels_and_embedded_settings(self):
        control = component("ruinedfooocus/features/image_editing/ui.py", "input_image")
        for mode in ("RGB", "RGBA"):
            with self.subTest(mode=mode):
                image = fixture(mode)
                with control.preprocess(control.postprocess(image)) as restored:
                    self.assertEqual(restored.size, image.size)
                    self.assertEqual(restored.mode, "RGBA")
                    self.assertEqual(restored.tobytes(), image.convert("RGBA").tobytes())
                    self.assertEqual(restored.info["parameters"], image.info["parameters"])

    def test_standalone_upscale_upload_preserves_rgba_pixels(self):
        control = component("ruinedfooocus/ui/app.py", "upscale_upload")
        image = fixture()
        with control.preprocess(control.postprocess(image)) as restored:
            self.assertEqual(restored.tobytes(), image.tobytes())

    def test_inpaint_background_survives_editor_transport_without_compression(self):
        control = component("ruinedfooocus/ui/app.py", "inpaint_view")
        image = fixture("RGB")
        restored = control.preprocess(control.postprocess(editor_payload(image)))
        np.testing.assert_array_equal(restored["background"], np.asarray(image.convert("RGBA")))

    def test_additional_reference_gallery_keeps_fine_rgb_details(self):
        control = component("ruinedfooocus/features/image_editing/ui_guidance.py", "references")
        image = fixture("RGB")
        with control.preprocess(control.postprocess([image]))[0][0] as restored:
            self.assertEqual(restored.size, image.size)
            self.assertEqual(restored.tobytes(), image.tobytes())


if __name__ == "__main__":
    unittest.main()
