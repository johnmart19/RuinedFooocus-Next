"""Real Gradio callbacks retain task-local precision artifacts and original follow-ups."""

import ast
import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.state_holder import SessionState
import numpy as np
from PIL import Image

from ruinedfooocus.features.image_processing import native_image_formats as formats
from ruinedfooocus.features.image_processing.native_image_outputs import (
    NativeImageOutput,
    capture_original_source,
    result_presentation,
    selected_result,
)
from test_native_image_formats import native_codecs
from ruinedfooocus.features.gallery.actions import source_record

ROOT = Path(__file__).resolve().parents[1]


class NativeOutputTransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rf-native-output-transport-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.native = native_codecs()
        self.enterContext(patch.object(formats, "_native", return_value=self.native))
        self.source = formats.write_image(
            self.root / "original.exr",
            np.full((5, 7, 4), 0.5, np.float32),
            dict(output_format="exr", output_bit_depth="32-bit float", output_color_space="linear"),
            {"Prompt": "original prompt"},
        )
        self.original = self.source.read_bytes()
        self.display = self.root / "derived.png"
        Image.new("RGBA", (3, 2), (255, 0, 0, 127)).save(self.display)
        self.product = [NativeImageOutput(str(self.source), str(self.display))]

    def test_real_create_result_callback_downloads_original_and_captures_original_source(self):
        with gr.Blocks(analytics_enabled=False) as app:
            namespace = dict(gr=gr)
            for name in ("run_button", "stop_button"):
                namespace[name] = gr.Button()
            namespace["progress_html"] = gr.HTML()
            namespace["main_view"] = gr.Image(type="filepath")
            namespace["inpaint_toggle"] = gr.Checkbox()
            namespace["gallery"] = gr.Gallery()
            namespace["metadata_json"] = gr.JSON()
            namespace["generation_originals"] = gr.File(file_count="multiple")
            namespace["generation_artifacts"] = gr.State([])
            namespace["native_main_source"] = gr.State(None)
            namespace["latest_generated"] = gr.State(None)
            namespace["editor_workspace"] = SimpleNamespace(
                latest_generated=namespace["latest_generated"]
            )
            namespace["image_source"] = gr.State({"filename": "previous.png"})
            namespace["shared"] = SimpleNamespace(get_ctrl=lambda _: namespace["image_source"])
            tree = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8"))
            node = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "update_results"
            )
            exec(
                compile(
                    ast.Module(body=[node], type_ignores=[]), "real Create result callback", "exec"
                ),
                namespace,
            )
            outputs = [
                value for value in namespace.values() if isinstance(value, gr.components.Component)
            ]
            trigger = gr.Button()
            trigger.click(lambda: namespace["update_results"](self.product), outputs=outputs)
        state = SessionState(app)
        result = asyncio.run(app.process_api(0, [], state=state))["data"]
        by_control = dict(zip(outputs, result))
        download = by_control[namespace["generation_originals"]]["value"][0]
        self.assertEqual(Path(download["path"]).read_bytes(), self.original)
        self.assertTrue(download["orig_name"].endswith(".exr"))
        self.assertEqual(by_control[namespace["main_view"]]["buttons"], ["fullscreen"])
        record = state[namespace["native_main_source"]._id]
        latest = state[namespace["latest_generated"]._id]
        self.assertEqual(Path(latest["path"]), self.source.resolve())
        self.assertEqual(latest["batch_index"], 0)
        self.assertEqual(latest["output_role"], "image")
        self.assertIsNone(state[namespace["image_source"]._id])
        displayed = by_control[namespace["main_view"]]["value"]["path"]
        captured = capture_original_source(dict(main_view=displayed, _native_main_source=record))
        self.assertEqual(captured["main_view"].size, (7, 5))
        self.assertGreater(captured["main_view"].getpixel((0, 0))[1], 165)
        self.assertEqual(captured["image_source"]["original_dtype"], "float32")
        self.assertEqual(captured["image_source"]["working_dtype"], "uint8")
        self.assertNotIn(str(self.root), str(captured["image_source"]))

    def test_separate_result_states_select_their_own_original(self):
        _, originals, first, _ = result_presentation(self.product)
        other = self.root / "other.png"
        Image.new("RGB", (4, 4), "blue").save(other)
        _, _, second, _ = result_presentation([str(other)])
        self.assertEqual(Path(selected_result(first, 0)[2]["path"]), self.source.resolve())
        self.assertIsNone(selected_result(second, 0)[2])
        self.assertEqual(originals, [str(self.source)])
        self.source.write_bytes(self.original + b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            selected_result(first, 0)
        with self.assertRaisesRegex(ValueError, "changed before"):
            result_presentation(self.product)

    def test_library_handoff_clears_old_original_even_when_the_new_preview_bytes_match(self):
        from test_gallery_actions import functions

        other = self.root / "new-selection.png"
        other.write_bytes(self.display.read_bytes())
        policy = SimpleNamespace(
            checkpoint_features=lambda *_: dict(sdxl=True, inpaint=True, native_edit=False),
            edit_tool=lambda _: "Img2Img",
            tool_kind=lambda *_: "img2img",
            controlnet=SimpleNamespace(cn_options={"Img2Img": dict(type="img2img", denoise=0.25)}),
        )
        with patch.dict("sys.modules", {"ruinedfooocus.features.image_editing.ui": policy}):
            updates = functions()["prepare_image"](
                source_record(other), "edit", "model", "Disabled", "Off"
            )
        _, _, records, _ = result_presentation(self.product)
        with gr.Blocks(analytics_enabled=False) as app:
            controls = {
                "main_view": gr.Image(type="filepath"),
                "_native_main_source": gr.State(records[0]),
                "_generation_artifacts": gr.State(records),
                "_generation_originals": gr.File(file_count="multiple"),
                "image_source": gr.State(None),
            }
            button = gr.Button()
            button.click(
                lambda: {control: updates[name] for name, control in controls.items()},
                outputs=list(controls.values()),
            )
        state = SessionState(app)
        rendered = asyncio.run(app.process_api(0, [], state=state))["data"]
        self.assertIsNone(state[controls["_native_main_source"]._id])
        self.assertEqual(state[controls["_generation_artifacts"]._id], [])
        self.assertEqual(state[controls["image_source"]._id]["filename"], other.name)
        data = dict(
            main_view=rendered[0]["value"]["path"],
            _native_main_source=state[controls["_native_main_source"]._id],
        )
        self.assertEqual(capture_original_source(data), data)

    def test_oversized_codec_result_is_not_published_and_retry_can_reuse_destination(self):
        destination = self.root / "retry.png"
        pixels = np.full((2, 3, 3), 0.5, np.float32)
        with (
            patch.object(formats, "MAX_BYTES", 1024),
            patch.object(self.native, "_encode_image", return_value=b"x" * 1025),
        ):
            with self.assertRaisesRegex(ValueError, "file limit"):
                formats.write_image(destination, pixels)
        self.assertFalse(destination.exists())
        formats.write_image(destination, pixels)
        self.assertEqual(formats.read_image(destination).pixels.shape, (2, 3, 3))


if __name__ == "__main__":
    unittest.main()
