"""Bundled wildcard ownership and bounded, refreshable reads without models."""

import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.prompting import wildcards


class WildcardCatalogTests(unittest.TestCase):
    def test_all_bundled_nested_references_resolve(self):
        files = wildcards.wildcard_files([])
        bundled = list(resource_path("wildcards").rglob("*.txt"))
        self.assertEqual(len(bundled), 32)
        references = 0
        for path in bundled:
            for name in re.findall(r"__([\w-]+)__", path.read_text(encoding="utf-8-sig")):
                references += 1
                self.assertIn(name, files, path)
        self.assertGreater(references, 0)

    def test_custom_precedence_and_cached_choices_refresh_without_hashing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            custom = root / "wildcards"
            legacy = root / "application/wildcards_official"
            custom.mkdir()
            legacy.mkdir(parents=True)
            path = custom / "words.txt"
            path.write_text("café\n# comment\n\n", encoding="utf-8-sig")
            (legacy / "words.txt").write_text("legacy", encoding="utf-8")
            with (
                patch.dict(os.environ, RF_DATA_DIR=folder),
                patch.object(wildcards, "APP_ROOT", root / "application"),
            ):
                self.assertEqual(wildcards.wildcard_files()["words"], path)
                first = wildcards.wildcard_choices(path)
                self.assertIs(first, wildcards.wildcard_choices(path))
                self.assertEqual(first, ("café", ""))
                self.assertEqual(wildcards.wildcard_word_index(), (("café",), frozenset(("café",))))
                path.write_text("fresh\nwords\n", encoding="utf-8")
                self.assertEqual(wildcards.wildcard_choices(path), ("fresh", "words"))
                self.assertIn("fresh", wildcards.wildcard_word_index()[1])


if __name__ == "__main__":
    unittest.main()
