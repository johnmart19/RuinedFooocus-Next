"""Page-level work budgets exercise real thumbnail and native Gradio delivery."""

import ast
from concurrent.futures import CancelledError
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
from gradio import processing_utils
from PIL import Image

from ruinedfooocus.core import cache_maintenance
from ruinedfooocus.core.cache_contract import library_thumbnail_root
from ruinedfooocus.features.gallery import thumbnails
from ruinedfooocus.features.gallery.index_store import ensure_schema

ROOT = Path(__file__).resolve().parents[1]


class GalleryPageWorkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rf-gallery-page-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.cache = self.root / "cache"
        self.delivery = self.root / "delivery"
        self.directory = library_thumbnail_root(self.cache)
        self.manager = SimpleNamespace(model_paths={"cache_path": self.cache})
        self.database = sqlite3.connect(":memory:")
        self.addCleanup(self.database.close)
        ensure_schema(self.database)
        self.originals = {}
        for number in range(6):
            path = self.root / f"{number}.png"
            with Image.new("RGBA", (80, 60), (number * 30, 40, 60, 127)) as image:
                image.save(path)
            self.originals[path] = path.read_bytes()
            self.database.execute(
                "INSERT INTO images(fullpath,path,json) VALUES(?,?,?)",
                (str(path), path.name, "{}"),
            )
        self.database.commit()
        self.browser = SimpleNamespace(
            sql_conn=self.database, path_manager=self.manager, images_per_page=100
        )
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/gallery/imagebrowser.py").read_text(encoding="utf-8")
        )
        browser = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ImageBrowser"
        )
        load = next(node for node in browser.body if getattr(node, "name", None) == "load_images")
        self.scope = dict(
            Path=Path,
            save_file_to_cache=processing_utils.save_file_to_cache,
            get_upload_folder=lambda: str(self.delivery),
        )
        exec(
            compile(ast.Module(body=[load], type_ignores=[]), "Gallery page caller", "exec"),
            self.scope,
        )
        for context in (
            patch.dict(thumbnails._dirty_roots, clear=True),
            patch.object(thumbnails, "GRID_THUMBNAIL_SIDE", 32),
        ):
            context.start()
            self.addCleanup(context.stop)

    def load(self, page=1, **kwargs):
        return self.scope["load_images"](self.browser, page, include_paths=True, **kwargs)

    def test_cold_page_scans_retention_once_and_all_copies_survive_eviction(self):
        self.directory.mkdir(parents=True)
        for number in range(8):
            path = self.directory / f"{number:064x}.webp"
            path.write_bytes(b"old owned entry")
            os.utime(path, ns=(number + 1, number + 1))
        scans = []
        iterdir = Path.iterdir

        def count(directory):
            if directory == self.directory:
                scans.append(directory)
            return iterdir(directory)

        with (
            patch.object(Path, "iterdir", count),
            patch.object(thumbnails, "LIBRARY_THUMBNAIL_MAX_FILES", 4),
        ):
            images, _, records = self.load()
        self.assertEqual(len(scans), 1)
        self.assertEqual(len(list(self.directory.glob("*.webp"))), 4)
        self.assertEqual(len(images), 6)
        self.assertTrue(all(Path(path).is_file() for path, _ in images))
        self.assertEqual({Path(record["path"]) for record in records}, set(self.originals))
        self.assertTrue(
            all(path.read_bytes() == content for path, content in self.originals.items())
        )

    def test_warm_page_and_native_gallery_postprocess_do_not_rehash_delivered_files(self):
        with patch.object(
            processing_utils, "hash_file", wraps=processing_utils.hash_file
        ) as hashed:
            images, _, _ = self.load()
            self.assertEqual(hashed.call_count, 6)
            first = Path(images[0][0])
            os.utime(first, ns=(1, 1))
            self.assertEqual(self.load()[0], images)
            self.assertGreater(first.stat().st_mtime_ns, 1)
            self.assertEqual(hashed.call_count, 6)
            gallery = gr.Gallery(render=False)
            gallery.GRADIO_CACHE = str(self.delivery)
            with patch.object(
                processing_utils, "get_upload_folder", return_value=str(self.delivery)
            ):
                processing_utils.move_files_to_cache(
                    gallery.postprocess(images), gallery, postprocess=True
                )
            self.assertEqual(hashed.call_count, 6)
        self.assertEqual(gallery.temp_files, {path for path, _ in images})

    def test_source_replacement_missing_delivery_and_upload_root_each_invalidate_reuse(self):
        with patch.object(
            processing_utils, "hash_file", wraps=processing_utils.hash_file
        ) as hashed:
            images, _, records = self.load()
            Path(images[0][0]).unlink()
            self.load()
            self.assertEqual(hashed.call_count, 7)
            original = Path(records[1]["path"])
            with Image.new("RGB", (84, 64), "yellow") as image:
                image.save(original)
            changed, _, changed_records = self.load()
            self.assertEqual(hashed.call_count, 8)
            self.assertNotEqual(changed[1][0], images[1][0])
            self.assertNotEqual(changed_records[1]["signature"], records[1]["signature"])
            self.scope["get_upload_folder"] = lambda: str(self.root / "other-delivery")
            relocated, _, _ = self.load()
            self.assertEqual(hashed.call_count, 14)
            self.assertTrue(
                all(
                    Path(path).is_relative_to(self.root / "other-delivery") for path, _ in relocated
                )
            )

    def test_cancelled_copy_batch_preserves_delivery_and_releases_maintenance_lease(self):
        cancelled = threading.Event()
        copied = []

        def copy(path, destination):
            result = processing_utils.save_file_to_cache(path, destination)
            copied.append(result)
            clear = cache_maintenance.clear_category("library-thumbnails", self.manager)
            self.assertEqual((clear["removed"], clear["skipped"]), (0, 1))
            cancelled.set()
            return result

        self.scope["save_file_to_cache"] = copy
        with self.assertRaises(CancelledError):
            self.load(cancelled=cancelled)
        self.assertEqual(len(copied), 1)
        self.assertTrue(Path(copied[0]).is_file())
        clear = cache_maintenance.clear_category("library-thumbnails", self.manager)
        self.assertEqual((clear["removed"], clear["skipped"]), (1, 0))
        self.assertTrue(
            all(path.read_bytes() == content for path, content in self.originals.items())
        )

    def test_cancelled_before_start_does_not_prepare_or_deliver_any_file(self):
        cancelled = threading.Event()
        cancelled.set()
        with patch.object(
            processing_utils, "hash_file", wraps=processing_utils.hash_file
        ) as hashed:
            with self.assertRaises(CancelledError):
                self.load(cancelled=cancelled)
        self.assertEqual(hashed.call_count, 0)
        self.assertFalse(self.directory.exists())

    def test_delivery_reuse_is_bounded_to_the_most_recent_page(self):
        self.browser.images_per_page = 2
        with patch.object(
            processing_utils, "hash_file", wraps=processing_utils.hash_file
        ) as hashed:
            for page in (1, 2, 3):
                self.load(page)
                self.assertEqual(len(self.browser._page_copies), 2)
            self.assertEqual(hashed.call_count, 6)
            self.load(3)
            self.assertEqual(hashed.call_count, 6)
            self.load(1)
            self.assertEqual(hashed.call_count, 8)


if __name__ == "__main__":
    unittest.main()
