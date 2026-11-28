"""Real Gradio point events preserve source pixels and independent request fields."""

import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.i18n import I18nData
from gradio.state_holder import SessionState
from PIL import Image
import ruinedfooocus.app
from ruinedfooocus.features.image_processing import ui_native_vision as ui
from ruinedfooocus.core.translation_manager import catalogue


class NativeVisionUI(unittest.TestCase):
    def build(self, kind="segment", saved=None):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "source.png"
        Image.new("RGBA", (1800, 900), (25, 50, 75, 128)).save(self.path)
        self.registered = {}
        shared = NS(
            settings=NS(default_settings=saved or {}),
            path_manager=object(),
            add_ctrl=lambda key, obj: self.registered.update({key: obj}),
        )
        self.enterContext(patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}))
        self.enterContext(patch.object(ruinedfooocus.app, "state", shared, create=True))
        self.enterContext(
            patch.object(ui, "native_model_choices", lambda manager, kind: [(kind, "/" + kind)])
        )
        self.enterContext(patch.object(ui, "coco_classes", lambda: ["person", "cat"]))
        with gr.Blocks(analytics_enabled=False) as demo:
            selection = gr.Textbox(value="Custom...")
            tool = gr.Textbox(value=kind)
            self.image = gr.Image(type="pil", image_mode="RGBA", format="png")
            refresh = gr.Number(value=0)
            ui.create_ui(selection, tool, self.image, refresh, lambda selection, kind: kind)
        self.demo, self.state = demo, SessionState(demo)
        return demo

    def call(self, name, values, event=None):
        index = next(key for key, fn in self.demo.fns.items() if fn.fn.__name__ == name)
        return asyncio.run(
            self.demo.process_api(index, values, state=self.state, event_data=event)
        )["data"]

    def source(self):
        return {"path": str(self.path), "meta": {"_type": "gradio.FileData"}}

    def click(self, mode, positive="[]", negative="[]", at=(1500, 700)):
        event = gr.SelectData(self.image, {"index": list(at), "value": None})
        return self.call(
            "add_point",
            [self.source(), "Custom...", "segment", mode, positive, negative, None],
            event,
        )

    def test_initial_classes_registered_capture_keys_and_explicit_zero(self):
        self.build("detect", dict(native_vision_model="/missing", vision_threshold=0))
        self.assertEqual(self.registered["vision_threshold"].value, 0)
        self.assertEqual(
            [v for _, v in self.registered["detect_class"].choices], ["all", "person", "cat"]
        )
        self.assertIn("segment_positive_points", self.registered)
        self.assertIn("segment_negative_points", self.registered)
        self.assertEqual(self.registered["native_vision_model"].value, "/missing")

    def test_refresh_and_tool_context_preserve_unavailable_explicit_selection(self):
        self.build(saved=dict(native_vision_model="/missing"))
        result = self.call("context", ["Custom...", "detect", "/missing", "cat"])
        self.assertEqual(result[3]["value"], "/missing")
        self.assertIn("/missing", [v for _, v in result[3]["choices"]])
        self.assertTrue(result[1]["visible"])
        self.assertFalse(result[2]["visible"])
        self.assertTrue(result[5]["visible"])
        self.assertTrue(
            self.call("context", ["Custom...", "segment", "/missing", "cat"])[5]["visible"]
        )
        self.assertFalse(
            self.call("context", ["Custom...", "rembg", "/missing", "cat"])[0]["visible"]
        )

    def test_sam_text_boxes_and_instance_controls_keep_spatial_prompts_request_owned(self):
        self.build(
            saved=dict(
                segment_text="stale saved category",
                segment_boxes='[{"x":0,"y":0,"width":1,"height":1}]',
                segment_individual_masks=True,
                segment_max_instances=4,
            )
        )
        self.assertEqual(self.registered["segment_text"].value, "")
        self.assertEqual(self.registered["segment_boxes"].value, "[]")
        self.assertTrue(self.registered["segment_individual_masks"].value)
        self.assertEqual(self.registered["segment_max_instances"].value, 4)
        result = self.call("context", ["Custom...", "segment", "/segment", "all"])
        self.assertTrue(result[2]["visible"])
        self.assertTrue(result[5]["visible"])
        self.assertFalse(result[1]["visible"])

    def test_depth_map_context_exposes_only_the_depth_model_picker(self):
        self.build("depth-map")
        expected_label = "Depth Anything 3 model"
        manager = catalogue()
        key = manager.aliases.get(expected_label, expected_label)
        with patch.dict(manager.translations["en"], {key: expected_label}):
            result = self.call("context", ["Depth map", "depth-map", "/depth-map", "cat"])
            label = result[3]["label"]
            if isinstance(label, I18nData):
                label = manager.translate(label.key.removeprefix("ruinedfooocus."), locale="en")
            self.assertEqual(label, expected_label)
        self.assertTrue(result[0]["visible"])
        self.assertEqual(result[3]["choices"], [["depth-map", "/depth-map"]])
        self.assertTrue(all(not result[index]["visible"] for index in (1, 2, 5)))

    def test_select_uses_source_pixels_and_preview_never_changes_input(self):
        self.build()
        original = self.path.read_bytes()
        result = self.click(ui.POSITIVE)
        self.assertEqual(json.loads(result[0]), [{"x": 1500, "y": 700}])
        preview = Image.open(result[4]["value"]["path"])
        self.assertEqual(preview.size, (768, 384))
        self.assertNotEqual(preview.getpixel((640, 299))[:3], (25, 50, 75))
        self.assertEqual(self.path.read_bytes(), original)
        self.assertNotIn(
            self.image,
            next(fn for fn in self.demo.fns.values() if fn.fn.__name__ == "add_point").outputs,
        )

    def test_positive_negative_undo_and_image_replacement_clear(self):
        self.build()
        first = self.click(ui.POSITIVE)
        second = self.click(ui.NEGATIVE, first[0], first[1], (20, 30))
        self.assertEqual(json.loads(second[1]), [{"x": 20, "y": 30}])
        undone = self.call("undo", [self.source(), second[0], second[1], None])
        self.assertEqual(undone[0], first[0])
        self.assertEqual(undone[1], "[]")
        cleared = self.call("clear", [])
        self.assertEqual(cleared[:2], ["[]", "[]"])
        self.assertFalse(cleared[4]["visible"])

    def test_manual_json_edit_clears_stale_click_history_and_invalid_preview_is_safe(self):
        self.build()
        first = self.click(ui.NEGATIVE)
        self.call("edited", [self.source(), '[{"x":2,"y":3}]', "[]", None])
        undone = self.call("undo", [self.source(), '[{"x":2,"y":3}]', "[]", None])
        self.assertEqual(undone[:2], ["[]", "[]"])
        self.assertEqual(ui.parse_points('[{"x":NaN,"y":0},{"x":true,"y":1}]'), [])
        self.assertEqual(ui.parse_points(json.dumps([{"x": 1, "y": 2}] * 257)), [])
        self.assertIsNone(ui.preview(None, first[0], first[1]))

    def test_server_summary_locale_and_english_fallback_do_not_change_global_language(self):
        manager = catalogue()
        text = "{included} points to keep, {excluded} points to exclude."
        key = manager.aliases[text]
        with patch.dict(
            manager.translations["uk"], {key: "Залишити {included}, виключити {excluded}."}
        ):
            self.assertEqual(ui.summary('[{"x":2,"y":3}]', "[]", "uk"), "Залишити 1, виключити 0.")
        self.assertIn("1 points to keep", ui.summary('[{"x":2,"y":3}]', "[]", "en"))
        self.assertEqual(manager.language, "en")


if __name__ == "__main__":
    unittest.main()
