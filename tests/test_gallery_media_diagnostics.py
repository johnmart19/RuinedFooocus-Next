"""Real image readers and SQLite keep valid rows beside rejected media files."""

import ast
import errno
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from PIL import Image
from PIL.PngImagePlugin import PngImageFile

from ruinedfooocus.features.gallery.index_store import (
    MediaMetadataError,
    connect_index,
    incremental_scan,
    page_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


class GalleryMediaDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rf-library-media-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        self.connection = connect_index(self.root / "images.db")
        self.addCleanup(self.connection.close)
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/gallery/imagebrowser.py").read_text(encoding="utf-8")
        )
        browser = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ImageBrowser"
        )
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("get_png_metadata", "get_media_metadata")
        ]
        update = next(
            node
            for node in browser.body
            if isinstance(node, ast.FunctionDef) and node.name == "update_images"
        )
        self.scope = dict(
            Image=Image,
            PngImageFile=PngImageFile,
            Path=Path,
            Dict=dict,
            json=json,
            settings=SimpleNamespace(default_settings={}),
        )
        exec(
            compile(
                ast.Module(body=[*functions, update], type_ignores=[]),
                "actual Library metadata",
                "exec",
            ),
            self.scope,
        )
        self.browser = SimpleNamespace(
            path_manager=SimpleNamespace(
                paths={"path_outputs": str(self.outputs)}, get_abspath=Path
            ),
            sql_conn=self.connection,
        )

    def png(self, name):
        path = self.outputs / name
        Image.new("RGB", (3, 2), (40, 90, 120)).save(path)
        return path

    def test_animated_rasters_are_skipped_without_changing_the_originals(self):
        self.png("valid.png")
        originals = {}
        for name, format in (
            ("animated.gif", "GIF"),
            ("animated.webp", "WEBP"),
            ("animated.png", "PNG"),
        ):
            path = self.outputs / name
            image = Image.new("RGB", (3, 2), "red")
            image.save(
                path,
                format=format,
                save_all=True,
                append_images=[Image.new("RGB", image.size, "blue")],
                duration=100,
            )
            originals[path] = path.read_bytes()
        self.scope["update_images"](self.browser)
        self.assertEqual(self.browser.last_scan.count, 1)
        self.assertEqual(self.browser.last_scan.skipped, 3)
        self.assertEqual(
            self.connection.execute("SELECT path FROM images").fetchall(), [("valid.png",)]
        )
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)

    def refresh(self):
        return self.scope["update_images"](self.browser)

    def rows(self):
        return self.connection.execute("SELECT path FROM images ORDER BY path").fetchall()

    def test_actual_png_and_avif_rejections_report_names_and_keep_valid_images(self):
        self.png("good.png")
        (self.outputs / "bad.png").write_bytes(b"malformed PNG")
        (self.outputs / "unsupported.avif").write_bytes(b"malformed AVIF header")
        status = self.refresh()
        self.assertIn("Found 1 images", status)
        self.assertIn("Skipped 2 unreadable files", status)
        self.assertIn("bad.png", status)
        self.assertIn("unsupported.avif", status)
        self.assertIn("UnidentifiedImageError", status)
        self.assertIn("ValueError", status)
        self.assertEqual(self.rows(), [("good.png",)])
        self.assertEqual(self.browser.last_scan.skipped, 2)
        self.assertTrue((self.outputs / "bad.png").exists())
        self.assertTrue((self.outputs / "unsupported.avif").exists())
        self.assertEqual(page_snapshot(self.connection, 1, "", 10)[0], 1)

    def test_rejected_replacement_drops_only_invalid_index_row_and_retry_recovers(self):
        self.png("good.png")
        replaced = self.png("replaced.png")
        self.refresh()
        replaced.write_bytes(b"corrupt replacement")
        self.png("new.png")
        self.assertIn("Skipped 1 unreadable files", self.refresh())
        self.assertEqual(self.rows(), [("good.png",), ("new.png",)])
        self.assertEqual((self.browser.last_scan.reused, self.browser.last_scan.removed), (1, 1))
        self.png("replaced.png")
        self.assertNotIn("Skipped", self.refresh())
        self.assertEqual(self.rows(), [("good.png",), ("new.png",), ("replaced.png",)])

    def test_filesystem_and_unexpected_reader_errors_preserve_previous_snapshot(self):
        self.png("good.png")
        self.refresh()
        before = self.rows()
        self.png("new.png")
        for error in (PermissionError(errno.EACCES, "denied"), RuntimeError("parser bug")):
            with self.subTest(error=type(error).__name__):
                self.scope["get_png_metadata"] = Mock(side_effect=error)
                self.assertIn("Error updating folder", self.refresh())
                self.assertEqual(self.rows(), before)

    def test_warning_details_are_bounded_and_html_escaped(self):
        for index in range(25):
            (self.outputs / f"bad{index:02d}.png").write_bytes(b"invalid")
        self.scope["get_png_metadata"] = Mock(side_effect=ValueError("<decoder>" + "x" * 2000))
        status = self.refresh()
        result = self.browser.last_scan
        self.assertEqual(result.skipped, 25)
        self.assertEqual(len(result.warnings), 20)
        self.assertTrue(all(len(warning) <= 1024 for warning in result.warnings))
        self.assertIn("&lt;decoder&gt;", status)
        self.assertNotIn("<decoder>", status)
        self.assertEqual(self.rows(), [])

    def test_cancellation_and_source_change_during_rejection_roll_back(self):
        self.png("good.png")
        self.refresh()
        before = self.rows()
        changed = self.outputs / "new.png"
        changed.write_bytes(b"invalid")
        cancelled = threading.Event()

        def reject(path):
            cancelled.set()
            raise MediaMetadataError("bad media")

        from concurrent.futures import CancelledError

        with self.assertRaises(CancelledError):
            incremental_scan(self.connection, [self.outputs], reject, cancelled=cancelled)
        self.assertEqual(self.rows(), before)

        def changing(path):
            path.write_bytes(b"replacement during failed read")
            raise MediaMetadataError("bad media")

        with self.assertRaisesRegex(OSError, "changed while scanning"):
            incremental_scan(self.connection, [self.outputs], changing)
        self.assertEqual(self.rows(), before)


if __name__ == "__main__":
    unittest.main()
