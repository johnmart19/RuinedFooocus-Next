"""Thumbnail retention and concurrent copies use only private image fixtures."""

import ast
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from ruinedfooocus.core import cache_maintenance
from ruinedfooocus.core.cache_contract import file_identity, library_thumbnail_root, record_key
from ruinedfooocus.features.gallery import thumbnails
from ruinedfooocus.features.gallery.index_store import ensure_schema

ROOT = Path(__file__).resolve().parents[1]


class GalleryThumbnailTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rf-thumbnails-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.cache = self.root / "cache"
        self.directory = library_thumbnail_root(self.cache)
        self.source = self.root / "original.png"
        Image.new("RGBA", (1600, 800), (20, 40, 60, 127)).save(self.source)
        self.original = self.source.read_bytes()
        self.manager = SimpleNamespace(model_paths={"cache_path": self.cache})
        state = patch.dict(thumbnails._dirty_roots, clear=True)
        state.start()
        self.addCleanup(state.stop)

    def fixture(self, number, size=20):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / (f"{number:064x}" + ".webp")
        path.write_bytes(b"x" * size)
        os.utime(path, ns=(number, number))
        return path

    def test_large_transparent_image_is_bounded_and_original_and_small_image_are_preserved(self):
        result = thumbnails.grid_thumbnail(self.source, self.cache)
        self.assertEqual(result.parent, self.directory)
        self.assertEqual(result.suffix, ".webp")
        with Image.open(result) as image:
            self.assertEqual(image.size, (1024, 512))
            self.assertIn("A", image.getbands())
        self.assertEqual(self.source.read_bytes(), self.original)
        small = self.root / "small.png"
        Image.new("RGB", (8, 16), "red").save(small)
        self.assertEqual(thumbnails.grid_thumbnail(small, self.cache), small)
        self.assertEqual(len(list(self.directory.glob("*.webp"))), 1)

    def test_warm_hit_does_not_decode_source_and_replacement_gets_a_new_key(self):
        result = thumbnails.grid_thumbnail(self.source, self.cache)
        os.utime(result, ns=(1, 1))
        with patch.object(Image, "open", side_effect=AssertionError("Warm hit decoded original")):
            self.assertEqual(thumbnails.grid_thumbnail(self.source, self.cache), result)
        self.assertGreater(result.stat().st_mtime_ns, 1)
        Image.new("RGB", (1400, 900), "blue").save(self.source)
        changed = thumbnails.grid_thumbnail(self.source, self.cache)
        self.assertNotEqual(changed, result)
        with Image.open(changed) as image:
            self.assertEqual(max(image.size), 1024)

    def test_exif_orientation_rotates_pixels_and_invalidates_old_copy_without_changing_source(self):
        source = self.root / "camera.jpg"
        with Image.new("RGB", (60, 30), "red") as image:
            image.paste("blue", (30, 0, 60, 30))
            exif = Image.Exif()
            exif[0x0112] = 6  # Camera pixels need a 90-degree clockwise display rotation.
            image.save(source, "JPEG", quality=95, exif=exif)
        original, mtime = source.read_bytes(), source.stat().st_mtime_ns
        self.directory.mkdir(parents=True)
        with patch.object(thumbnails, "GRID_THUMBNAIL_SIDE", 32):
            old_key = record_key("library-grid-thumbnail", file_identity(source) | {"side": 32})
            old_copy = self.directory / f"{old_key}.webp"
            with Image.open(source) as image:
                image.thumbnail((32, 32))
                image.save(old_copy, "WEBP")
            result = thumbnails.grid_thumbnail(source, self.cache)
        self.assertNotEqual(result, old_copy)
        with Image.open(result) as image:
            self.assertEqual(image.size, (16, 32))
            top, bottom = image.getpixel((8, 4)), image.getpixel((8, 27))
            self.assertGreater(top[0], 150)
            self.assertLess(top[2], 100)
            self.assertGreater(bottom[2], 150)
            self.assertLess(bottom[0], 100)
            self.assertNotIn(0x0112, image.getexif())
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(source.stat().st_mtime_ns, mtime)

    def test_failed_atomic_replacement_cleans_unique_partial_and_preserves_original(self):
        with patch.object(thumbnails.os, "replace", side_effect=OSError("fixture full disk")):
            self.assertEqual(thumbnails.grid_thumbnail(self.source, self.cache), self.source)
        self.assertEqual(list(self.directory.iterdir()), [])
        self.assertEqual(self.source.read_bytes(), self.original)
        unsupported = self.root / "untouched.exr"
        unsupported.write_bytes(b"native fixture")
        self.assertEqual(thumbnails.grid_thumbnail(unsupported, self.cache), unsupported)

    def test_two_simultaneous_requests_share_one_complete_write(self):
        entered, release = threading.Event(), threading.Event()
        save = Image.Image.save
        partials = []

        def blocked_save(image, path, *args, **kwargs):
            partials.append(Path(path))
            entered.set()
            self.assertTrue(release.wait(5))
            return save(image, path, *args, **kwargs)

        with (
            ThreadPoolExecutor(max_workers=2) as pool,
            patch.object(Image.Image, "save", autospec=True, side_effect=blocked_save),
        ):
            first = pool.submit(thumbnails.grid_thumbnail, self.source, self.cache)
            self.assertTrue(entered.wait(5))
            try:
                second = pool.submit(thumbnails.grid_thumbnail, self.source, self.cache)
            finally:
                release.set()
            self.assertEqual(first.result(5), second.result(5))
        self.assertEqual(len(partials), 1)
        self.assertNotEqual(partials[0].name, f"{first.result().name}.{os.getpid()}.tmp")
        self.assertEqual(list(self.directory.glob("*.tmp")), [])
        with Image.open(first.result()) as image:
            image.load()

    def test_count_and_byte_limits_remove_oldest_owned_entries_only(self):
        oldest, middle, newest = [self.fixture(number) for number in (1, 2, 3)]
        manual = self.directory / "manual.webp"
        manual.write_bytes(b"manual image")
        nested = self.directory / "nested" / ("f" * 64 + ".webp")
        nested.parent.mkdir()
        nested.write_bytes(b"retained nested fixture")
        with patch.object(thumbnails, "LIBRARY_THUMBNAIL_MAX_FILES", 2):
            thumbnails._prune(self.directory)
        self.assertFalse(oldest.exists())
        self.assertTrue(middle.exists() and newest.exists())
        self.assertTrue(manual.exists() and nested.exists())
        self.fixture(4)
        with (
            patch.dict(thumbnails._dirty_roots, {str(self.directory): True}, clear=True),
            patch.object(thumbnails, "LIBRARY_THUMBNAIL_MAX_BYTES", 25),
        ):
            thumbnails._prune(self.directory)
        self.assertFalse(middle.exists() or newest.exists())
        self.assertTrue((self.directory / (f"{4:064x}" + ".webp")).is_file())

    def test_oversized_entry_is_not_cached_and_active_copy_defers_eviction(self):
        with patch.object(thumbnails, "LIBRARY_THUMBNAIL_MAX_BYTES", 1):
            self.assertEqual(thumbnails.grid_thumbnail(self.source, self.cache), self.source)
        self.assertEqual(list(self.directory.iterdir()), [])
        oldest = self.fixture(1)
        with patch.object(thumbnails, "LIBRARY_THUMBNAIL_MAX_FILES", 1):
            with thumbnails.grid_thumbnail_path(self.source, self.cache) as result:
                self.assertTrue(oldest.exists() and result.exists())
                thumbnails._prune(self.directory)
                self.assertTrue(oldest.exists() and result.exists())
                cleared = cache_maintenance.clear_category("library-thumbnails", self.manager)
                self.assertEqual((cleared["removed"], cleared["skipped"]), (0, 1))
            self.assertFalse(oldest.exists())
            self.assertTrue(result.exists())

    def test_real_browser_copy_is_leased_but_source_records_stay_original(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/gallery/imagebrowser.py").read_text(encoding="utf-8")
        )
        browser_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ImageBrowser"
        )
        load = next(
            node for node in browser_class.body if getattr(node, "name", None) == "load_images"
        )
        copied = self.root / "gradio-copy.webp"

        def copy(path, destination):
            result = cache_maintenance.clear_category("library-thumbnails", self.manager)
            self.assertEqual((result["removed"], result["skipped"]), (0, 1))
            shutil.copyfile(path, copied)
            return str(copied)

        scope = dict(Path=Path, save_file_to_cache=copy, get_upload_folder=lambda: str(self.root))
        exec(compile(ast.Module(body=[load], type_ignores=[]), "thumbnail caller", "exec"), scope)
        database = sqlite3.connect(":memory:")
        self.addCleanup(database.close)
        ensure_schema(database)
        database.execute(
            "INSERT INTO images(fullpath,path,json) VALUES(?,?,?)",
            (str(self.source), self.source.name, "{}"),
        )
        database.commit()
        browser = SimpleNamespace(sql_conn=database, images_per_page=20, path_manager=self.manager)
        images, _, records = scope["load_images"](browser, 1, include_paths=True)
        self.assertEqual(images[0][0], str(copied))
        self.assertEqual(records[0]["path"], str(self.source))
        cleared = cache_maintenance.clear_category("library-thumbnails", self.manager)
        self.assertEqual(cleared["removed"], 1)
        self.assertTrue(copied.is_file())
        self.assertEqual(self.source.read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
