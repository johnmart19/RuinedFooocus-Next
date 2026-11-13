"""Style catalogs preserve UTF-8 and literal custom content."""

import ast
import csv
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), scope)
    return scope[name]


class StyleEncodingTests(unittest.TestCase):
    def test_utf8_styles_and_bom_preserve_words_across_load_merge_and_reload(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            default = Path(folder) / "default.csv"
            custom = Path(folder) / "styles.csv"
            default.write_text(
                "name,prompt,negative_prompt\ncafé,猫 près de la tasse,sans bruit\n",
                encoding="utf-8-sig",
            )
            custom.write_text(
                "name,prompt,negative_prompt\nCustom,lumière café,текст\n", encoding="utf-8-sig"
            )
            execute = function(
                "ruinedfooocus/features/prompting/sdxl_styles.py",
                "load_styles",
                {
                    "DEFAULT_STYLES_FILE": default,
                    "STYLES_FILE": custom,
                    "exists": os.path.exists,
                    "csv": csv,
                    "DictReader": csv.DictReader,
                    "locale": __import__("locale"),
                    "STYLE_ALIASES": {},
                },
            )
            styles = execute()
            self.assertEqual(styles["café"], ("猫 près de la tasse", "sans bruit"))
            self.assertEqual(styles["Custom"], ("lumière café", "текст"))
            self.assertEqual(execute(), styles)
            self.assertIn("lumière café", custom.read_text(encoding="utf-8"))

    def test_legacy_locale_styles_are_read_losslessly_without_incidental_rewrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            default = Path(folder) / "default.csv"
            custom = Path(folder) / "styles.csv"
            text = "name,prompt,negative_prompt\nCustom,свет,текст\n"
            default.write_text(text, encoding="utf-8")
            original = text.encode("cp1251")
            custom.write_bytes(original)
            execute = function(
                "ruinedfooocus/features/prompting/sdxl_styles.py",
                "load_styles",
                {
                    "DEFAULT_STYLES_FILE": default,
                    "STYLES_FILE": custom,
                    "exists": os.path.exists,
                    "csv": csv,
                    "DictReader": csv.DictReader,
                    "locale": SimpleNamespace(getpreferredencoding=lambda flag: "cp1251"),
                    "STYLE_ALIASES": {},
                },
            )
            with patch(
                "ruinedfooocus.features.prompting.style_catalog.locale.getpreferredencoding",
                return_value="cp1251",
            ):
                self.assertEqual(execute()["Custom"], ("свет", "текст"))
            self.assertEqual(custom.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
