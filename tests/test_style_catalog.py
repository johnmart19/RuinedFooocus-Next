"""Style discovery must not change user files or turn category rows into prompts."""

import ast
import csv
from pathlib import Path
import random
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.prompting import style_catalog
from ruinedfooocus.features.prompting.style_catalog import (
    is_heading,
    load_catalog,
    normalize_selection,
    style_templates,
)

ROOT = Path(__file__).resolve().parents[1]


class StyleCatalogTests(unittest.TestCase):
    def test_all_bundled_rows_have_groups_and_no_header_is_a_style(self):
        bundled = resource_path("defaults", "styles.default")
        with bundled.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with tempfile.TemporaryDirectory() as folder:
            catalog = load_catalog(bundled, Path(folder) / "missing.csv")
        headings = [row for row in rows if is_heading(row["name"])]
        self.assertEqual(len(headings), 9)
        self.assertEqual(len(catalog), len(rows) - len(headings))
        self.assertTrue(all(style.group for style in catalog.values()))
        self.assertFalse(any(is_heading(name) for name in catalog))
        self.assertEqual(catalog["Style: papercraft-collage"].group, "Papercraft")

    def test_custom_overrides_by_name_without_any_write_and_refreshes_after_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            default, user = Path(folder) / "default.csv", Path(folder) / "custom.csv"
            default.write_text(
                "name,prompt,negative_prompt\n>>>>>> PHOTO,,\nFrame,{prompt} photo,blur\n",
                encoding="utf-8",
            )
            original = "name,prompt,negative_prompt\nFrame,{prompt} café,noise\nCustom,{prompt} 猫,\n".encode(
                "utf-8-sig"
            )
            user.write_bytes(original)
            catalog = load_catalog(default, user)
            self.assertEqual(catalog["Frame"].prompt, "{prompt} café")
            self.assertEqual(catalog["Frame"].group, "Photography")
            self.assertEqual(catalog["Custom"].group, "Custom")
            self.assertEqual(user.read_bytes(), original)
            user.write_bytes(original.replace(b"noise", b"grain"))
            self.assertEqual(load_catalog(default, user)["Frame"].negative_prompt, "grain")

    def test_same_metadata_edits_refresh_while_unchanged_content_stays_parsed_once(self):
        with tempfile.TemporaryDirectory() as folder:
            default, user = Path(folder) / "default.csv", Path(folder) / "custom.csv"
            default.write_text(
                "name,prompt,negative_prompt\nFrame,{prompt} baseline,blur\n", encoding="utf-8"
            )
            original = (
                'name,prompt,negative_prompt\r\nFrame,"{prompt} café\r\n猫",noise\r\n'.encode(
                    "utf-8-sig"
                )
            )
            user.write_bytes(original)
            unchanged_metadata = user.stat()
            style_catalog._read_styles.cache_clear()
            with (
                patch.object(Path, "stat", return_value=unchanged_metadata),
                patch.object(style_catalog.csv, "DictReader", wraps=csv.DictReader) as parse,
            ):
                self.assertEqual(load_catalog(default, user)["Frame"].negative_prompt, "noise")
                self.assertEqual(load_catalog(default, user)["Frame"].prompt, "{prompt} café\n猫")
                self.assertEqual(load_catalog(default, user)["Frame"].negative_prompt, "noise")
                self.assertEqual(parse.call_count, 2)
                user.write_bytes(original.replace(b"noise", b"grain"))
                self.assertEqual(load_catalog(default, user)["Frame"].negative_prompt, "grain")
                self.assertEqual(parse.call_count, 3)

    def test_missing_custom_file_stays_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "custom.csv"
            self.assertIn("Style: sai-anime", style_templates(user_file=missing))
            self.assertFalse(missing.exists())

    def test_legacy_selection_keeps_real_styles_order_and_does_not_mutate_input(self):
        selected = [
            ">>>>>> PAPERCRAFT",
            "Style: papercraft-collage",
            "None",
            "Erniehancer",
            "Style: papercraft-collage",
        ]
        before = selected.copy()
        self.assertEqual(
            normalize_selection(selected), ["Style: papercraft-collage", "Local chat model"]
        )
        self.assertEqual(selected, before)

    def test_every_bundled_style_keeps_the_literal_prompt(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/prompting/sdxl_styles.py").read_text(encoding="utf-8")
        )
        node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_style"
        )
        with tempfile.TemporaryDirectory() as folder:
            templates = style_templates(user_file=Path(folder) / "missing.csv")
        namespace = dict(
            re=re,
            random=random,
            STYLE_ALIASES={},
            load_styles=lambda: templates,
            build_dynamic_prompt=SimpleNamespace(
                artify_prompt=lambda prompt, artists: prompt + ", " + artists
            ),
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), "apply_style", "exec"), namespace)
        prompt = 'Keep exactly two red apples. "café" {prompt} \\1'
        for name in templates:
            if name in ("Flufferizer", "Hyperprompt", "Local chat model"):
                continue
            with self.subTest(style=name):
                selected = [name]
                positive, _ = namespace["apply_style"](selected, prompt, "No text.", "trigger")
                self.assertIn(prompt, positive)
                self.assertEqual(selected, [name])


if __name__ == "__main__":
    unittest.main()
