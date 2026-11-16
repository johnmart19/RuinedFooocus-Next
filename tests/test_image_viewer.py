"""Display controls must not enqueue inference or change image request values."""

import json
from pathlib import Path
import subprocess
import unittest

import gradio as gr

from ruinedfooocus.ui.image_viewer import create_ui

ROOT = Path(__file__).resolve().parents[1]


class ImageViewerTests(unittest.TestCase):
    def test_native_controls_are_client_only_and_do_not_change_the_image(self):
        with gr.Blocks() as demo:
            image = gr.Image(value=None, elem_id="main_view")
            controls = create_ui()
        self.assertEqual(set(controls), {"fit", "fill", "actual", "zoom-out", "zoom-in", "resize"})
        self.assertIsNone(image.value)
        self.assertEqual(len(demo.fns), 5)
        for fn in demo.fns.values():
            self.assertIsNone(fn.fn)
            self.assertFalse(fn.queue)
            self.assertEqual(fn.outputs, [])
            self.assertIn("window.rfImageViewer.", fn.js)

    def test_pan_zoom_and_resize_geometry(self):
        result = subprocess.run(
            ["node", "tests/test_image_viewer.js"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_viewer_labels_exist_in_every_shipped_language(self):
        folder = ROOT / "ruinedfooocus/resources/translations"
        locales = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["locales"]
        for locale in locales:
            data = json.loads((folder / f"{locale}.json").read_text(encoding="utf-8"))
            for name in ("fit", "fill", "actual_size", "zoom_in", "zoom_out", "resize"):
                with self.subTest(locale=locale, label=name):
                    self.assertTrue(data[f"viewer.{name}"].strip())


if __name__ == "__main__":
    unittest.main()
