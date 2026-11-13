"""Mounted installed Gradio uploads, ordering, visibility and session state."""

import asyncio
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.state_holder import SessionState
from PIL import Image

import ruinedfooocus.app as app
from ruinedfooocus.features.image_processing import ui_image_composition as ui
from ruinedfooocus.features.image_processing import image_composition as composition


class ImageCompositionUITests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.paths = [self.root / "first.png", self.root / "second.png"]
        for path, color in zip(self.paths, ("red", "blue")):
            Image.new("RGBA", (8, 6), color).save(path)
        self.registered = {}
        shared = NS(
            settings=NS(
                default_settings=dict(
                    grid_padding=0,
                    composition_x=0,
                    composition_images=["never restore"],
                    composition_mask="stale",
                )
            ),
            add_ctrl=lambda key, obj: self.registered.update({key: obj}),
        )
        self.enterContext(patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}))
        self.enterContext(patch.object(app, "state", shared, create=True))
        with gr.Blocks(analytics_enabled=False) as self.demo:
            selection = gr.Textbox(value="image-grid")
            kind = gr.Textbox(value="ignored")
            ui.create_ui(selection, kind, lambda selected, kind: selected)
        self.state = SessionState(self.demo)

    def call(self, name, values, *, event=None, state=None, occurrence=0):
        indexes = [key for key, fn in self.demo.fns.items() if fn.fn.__name__ == name]
        return asyncio.run(
            self.demo.process_api(
                indexes[occurrence], values, state=state or self.state, event_data=event
            )
        )["data"]

    def images(self):
        return [
            {
                "image": {"path": str(path), "meta": {"_type": "gradio.FileData"}},
                "caption": path.stem,
            }
            for path in self.paths
        ]

    def test_controls_are_registered_once_uploads_start_empty_and_numeric_zero_survives(self):
        self.assertEqual(set(self.registered), set(composition.FIELDS + composition.IMAGE_FIELDS))
        self.assertEqual(self.registered["composition_images"].value, [])
        self.assertIsNone(self.registered["composition_mask"].value)
        self.assertEqual(self.registered["grid_padding"].value, 0)
        self.assertEqual(self.registered["composition_x"].value, 0)
        self.assertEqual(
            [self.registered[key].precision for key in ("grid_columns", "composition_x")], [0, 0]
        )

    def test_real_change_events_keep_grid_and_composite_controls_in_their_context(self):
        self.assertEqual(
            [value["visible"] for value in self.call("context", ["image-grid", "unused"])],
            [True, True, False],
        )
        self.assertEqual(
            [value["visible"] for value in self.call("context", ["image-composite", "unused"])],
            [True, False, True],
        )
        self.assertEqual(
            [value["visible"] for value in self.call("context", ["edit", "unused"])],
            [False, False, False],
        )
        trigger = next(fn for fn in self.demo.fns.values() if fn.fn.__name__ == "context")
        self.assertTrue(all(event == "change" for _, event in trigger.targets))

    def test_gallery_selection_reorders_actual_pil_uploads_and_does_not_leak_between_sessions(self):
        gallery = self.registered["composition_images"]
        event = gr.SelectData(gallery, {"index": 1, "value": None, "selected": True})
        self.call("select", [], event=event)
        result = self.call("<lambda>", [self.images(), None], occurrence=0)
        self.assertEqual([entry["caption"] for entry in result[0]], ["second", "first"])
        with Image.open(result[0][0]["image"]["path"]) as moved:
            self.assertEqual(moved.getpixel((0, 0)), (0, 0, 255, 255))
        fresh = SessionState(self.demo)
        untouched = self.call("<lambda>", [self.images(), None], state=fresh, occurrence=0)
        self.assertEqual([entry["caption"] for entry in untouched[0]], ["first", "second"])
        self.call("<lambda>", [], occurrence=2)
        original = [Image.open(path).getpixel((0, 0)) for path in self.paths]
        self.assertEqual(original, [(255, 0, 0, 255), (0, 0, 255, 255)])

    def test_saved_defaults_keep_only_scalars_and_validation_stays_bounded(self):
        from ruinedfooocus.features.generation.defaults import generation_defaults
        from ruinedfooocus.core.settings_schema import validate_known

        values = dict(
            composition.DEFAULTS,
            composition_images=[Image.new("RGB", (3, 4))],
            composition_mask=Image.new("L", (3, 4)),
            image_number=1,
        )
        saved = generation_defaults(values)
        for key in composition.FIELDS:
            self.assertEqual(saved[key], values[key])
        self.assertNotIn("composition_images", saved)
        self.assertNotIn("composition_mask", saved)
        for changed in (
            dict(grid_columns=True),
            dict(grid_padding=-1),
            dict(composition_y=float("inf")),
            dict(composition_mode="unknown"),
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_known(changed)

    def test_upload_and_move_bounds_run_before_any_gallery_media_publication(self):
        payload = self.images() * 17
        with self.assertRaisesRegex(gr.Error, "32 ordered"):
            self.call("validate_uploads", [payload])
        with (
            patch.object(composition, "MAX_INPUT_PIXELS", 47),
            self.assertRaisesRegex(ValueError, "total"),
        ):
            ui.move([(Image.new("RGBA", (8, 6)), "caption")], 0, 1)


if __name__ == "__main__":
    unittest.main()
