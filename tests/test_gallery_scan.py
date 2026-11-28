"""Library scanning deduplicates root aliases and overlapping folders in order."""

import ast
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.path import PathManager


ROOT = Path(__file__).resolve().parents[1]


class GalleryScanTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="rf-gallery-")
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/gallery/imagebrowser.py").read_text(encoding="utf-8")
        )
        browser = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ImageBrowser"
        )
        methods = [
            node
            for node in browser.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("update_images", "add_image", "num_images_pages")
        ]
        self.settings = SimpleNamespace(default_settings={})
        self.metadata = Mock(side_effect=lambda path: {"source": str(path)})
        self.scope = dict(
            Path=Path,
            os=os,
            json=json,
            settings=self.settings,
            TimeIt=lambda _: nullcontext(),
            get_png_metadata=self.metadata,
        )
        exec(compile(ast.Module(body=methods, type_ignores=[]), "Library scan", "exec"), self.scope)
        self.database = sqlite3.connect(":memory:")
        self.addCleanup(self.database.close)
        self.database.execute("CREATE TABLE images(fullpath text, path text, json text)")

    def image(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Enumeration/metadata identity is the contract; no generated image is needed.
        path.write_bytes(b"private enumeration fixture")
        return path

    def scan(self, output, archives=(), manager=None):
        self.settings.default_settings["archive_folders"] = list(archives)
        manager = manager or SimpleNamespace(paths={"path_outputs": output}, get_abspath=Path)
        browser = SimpleNamespace(path_manager=manager, sql_conn=self.database, images_per_page=1)
        browser.add_image = lambda *args: self.scope["add_image"](browser, *args)
        status = self.scope["update_images"](browser)
        self.assertFalse(status.startswith("Error"), status)
        return (
            status,
            self.database.execute("SELECT fullpath, path FROM images ORDER BY rowid").fetchall(),
            browser,
        )

    def test_identical_and_relative_alias_roots_are_scanned_once(self):
        image = self.image("outputs/fixture.png")
        output = image.parent
        alias = output / "child/.."
        (output / "child").mkdir()
        with patch.object(os, "walk", wraps=os.walk) as walk:
            status, rows, browser = self.scan(output, [output, output.resolve(), str(alias)])
        self.assertEqual(rows, [(str(image), "fixture.png")])
        self.assertEqual(walk.call_count, 1)
        self.metadata.assert_called_once_with(str(image))
        self.assertIn("Found 1 images", status)
        self.assertEqual(status.count(f"`{output.resolve()}`"), 1)
        self.assertEqual(self.scope["num_images_pages"](browser), (1, 1))

    def test_overlapping_roots_keep_first_source_relative_path_and_order(self):
        root_image = self.image("outputs/root.png")
        child_image = self.image("outputs/child/child.png")
        status, rows, _ = self.scan(child_image.parent, [root_image.parent])
        self.assertEqual(rows, [(str(child_image), "child.png"), (str(root_image), "root.png")])
        self.assertEqual(self.metadata.call_count, 2)
        self.assertIn("Found 2 images", status)

        self.metadata.reset_mock()
        _, rows, _ = self.scan(root_image.parent, [child_image.parent])
        self.assertEqual(
            dict(rows),
            {str(root_image): "root.png", str(child_image): str(Path("child/child.png"))},
        )
        # A reordered root changes relative identity without reparsing unchanged files.
        self.assertEqual(self.metadata.call_count, 0)

    def test_distinct_files_with_same_name_in_distinct_roots_are_retained(self):
        first = self.image("first/same.png")
        second = self.image("second/same.png")
        _, rows, _ = self.scan(first.parent, [second.parent])
        self.assertEqual(rows, [(str(first), "same.png"), (str(second), "same.png")])

    def test_default_profile_path_policy_is_one_root_and_uses_private_data(self):
        image = self.image("outputs/default.png")
        manager = PathManager.__new__(PathManager)
        manager.paths = {"path_outputs": PathManager.DEFAULT_PATHS["path_outputs"]}
        with patch.dict(os.environ, RF_DATA_DIR=str(self.root)):
            status, rows, _ = self.scan(image.parent, manager=manager)
        self.assertEqual(rows, [(str(image.resolve()), "default.png")])
        self.assertIn("Found 1 images", status)
        self.assertEqual(self.metadata.call_count, 1)


if __name__ == "__main__":
    unittest.main()
