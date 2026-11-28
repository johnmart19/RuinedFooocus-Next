"""Paint canvas results become ordinary undoable Editor steps; no browser or model runs."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from PIL import Image

from ruinedfooocus.features.gallery import editor_session as editor
from ruinedfooocus.features.gallery import ui_paint as paint


class EditorPaintTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.owner = ("paint-owner",)
        self.store = editor.EditorSessionStore(self.root / "sessions")
        self.source = self.root / "original.png"
        Image.new("RGB", (8, 6), (10, 20, 30)).save(self.source)
        self.initial = self.store.start(self.source, owner=self.owner)

    def manager(self):
        return NS(model_paths={"temp_outputs_path": str(self.root / "outputs")})

    def complete(self, image, layers=1):
        snapshot = self.store.snapshot(self.initial.session_id, owner=self.owner)
        ticket = self.store.capture(
            snapshot.session_id,
            snapshot.revision,
            paint.OPERATION,
            dict(layers=layers),
            source_id=snapshot.current.artifact_id,
            owner=self.owner,
        )
        with patch("ruinedfooocus.app.state.path_manager", self.manager(), create=True):
            path = paint.save_result(image, "original.png")
        return path, self.store.complete(
            ticket,
            path,
            execution=paint.execution(snapshot.current.frame, image, layers),
            owner=self.owner,
        )

    def test_flattened_canvas_prefers_the_composite_and_reports_layers(self):
        background = Image.new("RGB", (4, 4), "red")
        composite = Image.new("RGBA", (4, 4), (0, 0, 255, 128))
        value = dict(background=background, layers=[composite, composite], composite=composite)
        self.assertEqual(paint.flattened(value).getpixel((0, 0)), (0, 0, 255, 128))
        self.assertEqual(paint.painted_layers(value), 2)
        self.assertEqual(paint.flattened(dict(background=background)).mode, "RGBA")
        for empty in (None, {}, dict(composite=None, background=None), "path"):
            self.assertIsNone(paint.flattened(empty))
            self.assertEqual(paint.painted_layers(empty), 0)

    def test_painting_is_saved_with_metadata_and_becomes_an_undoable_step(self):
        painted = Image.new("RGBA", (8, 6), (200, 0, 0, 255))
        path, completed = self.complete(painted)
        self.assertEqual(completed.revision, self.initial.revision + 1)
        self.assertEqual(Path(completed.current.path), path)
        self.assertTrue(path.is_relative_to(self.root / "outputs"))
        with Image.open(path) as saved:
            self.assertEqual(saved.getpixel((0, 0)), (200, 0, 0, 255))
            information = json.loads(saved.info["parameters"])
        self.assertEqual(information["controlnet"], {"type": "paint"})
        self.assertEqual((information["width"], information["height"]), (8, 6))
        undone = self.store.undo(completed.session_id, completed.revision, owner=self.owner)
        self.assertEqual(undone.current.artifact_id, self.initial.current.artifact_id)
        redone = self.store.redo(undone.session_id, undone.revision, owner=self.owner)
        self.assertEqual(redone.current.artifact_id, completed.current.artifact_id)
        with Image.open(self.source) as original:
            self.assertEqual(original.getpixel((0, 0)), (10, 20, 30))

    def test_a_cropped_canvas_records_its_geometry_and_an_unexplained_one_is_refused(self):
        cropped = Image.new("RGBA", (5, 3), (0, 200, 0, 255))
        frame = self.initial.current.frame
        self.assertEqual(
            paint.execution(frame, cropped, 0)["geometry"],
            dict(policy="canvas crop", source=[8, 6], result=[5, 3]),
        )
        self.assertNotIn("geometry", paint.execution(frame, Image.new("RGBA", (8, 6)), 0))
        _, completed = self.complete(cropped)
        self.assertEqual((completed.current.frame.width, completed.current.frame.height), (5, 3))
        snapshot = self.store.snapshot(self.initial.session_id, owner=self.owner)
        ticket = self.store.capture(
            snapshot.session_id,
            snapshot.revision,
            paint.OPERATION,
            source_id=snapshot.current.artifact_id,
            owner=self.owner,
        )
        other = self.root / "other.png"
        Image.new("RGBA", (2, 2)).save(other)
        with self.assertRaises(ValueError):
            self.store.complete(ticket, other, execution=dict(operation="paint"), owner=self.owner)


if __name__ == "__main__":
    unittest.main()
