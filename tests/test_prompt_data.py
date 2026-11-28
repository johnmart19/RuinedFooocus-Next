"""Prompt data contracts without models, generation, or user-file writes."""

import csv
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.features.prompting.one_button import csv_reader as reader
from ruinedfooocus.features.prompting.one_button import data_store as store
from ruinedfooocus.features.prompting.one_button.one_button_presets import OneButtonPresets


class PromptDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.user = self.root / "user"
        self.user.mkdir()
        self.package = self.root / "package"
        self.package.mkdir()
        for target, name, value in (
            (store, "PACKAGE", self.package),
            (reader, "PACKAGE", self.package),
            (store, "user_roots", lambda: (self.user,)),
        ):
            context = patch.object(target, name, value)
            context.start()
            self.addCleanup(context.stop)
        store.clear_cache()
        self.addCleanup(store.clear_cache)

    def write(self, name, text, user=False):
        path = (self.user if user else self.package) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_direct_user_lists_and_wildcards_have_precedence_without_writes(self):
        names = (
            "custom_subjects",
            "personal_artists",
            "custom_outfits",
            "custom_input_prefix",
            "custom_input_mid",
            "custom_input_suffix",
            "antilist",
            "styles_ti_lora",
        )
        for name in names:
            self.write(name + ".csv", "header\nSentinel " + name, user=True)
            self.assertEqual(
                reader.csv_to_list(name, directory="./userfiles/", skipheader=True),
                ["Sentinel " + name],
            )
        path = self.write("wildcards/test.csv", "café?;extra\n", user=True)
        before = {p: p.read_bytes() for p in self.user.rglob("*") if p.is_file()}
        self.assertEqual(reader.csv_to_list("test", directory="./userfiles/wildcards/"), ["café?"])
        self.assertEqual(before, {p: p.read_bytes() for p in self.user.rglob("*") if p.is_file()})
        self.assertEqual(reader.csv_to_list("missing", directory="./userfiles/"), [])
        path.unlink()
        self.assertEqual(reader.csv_to_list("test", directory="./userfiles/wildcards/"), [])

    def test_replacement_variant_addon_and_legacy_matrix_shape(self):
        self.write("csvfiles/example.csv", "Header\nBase\n")
        self.write("example_replace.csv", "Header\nCustom\nCUSTOM\n", user=True)
        self.write("example_replace_addon.csv", "Header\nExtra\n", user=True)
        self.write("example_addon.csv", "Header\nUnused\n", user=True)
        with patch.object(random, "randint") as draw:
            self.assertEqual(
                reader.csv_to_list("example", skipheader=True, insanitylevel=1), ["Custom", "Extra"]
            )
            draw.assert_not_called()
        matrix = reader.csv_to_list("example", skipheader=True, listoflistmode=True)
        self.assertEqual(matrix, [["Custom"], ["CUSTOM"], [["Extra"]]])
        matrix[-1][0][0] = "mutated"
        self.assertEqual(
            reader.csv_to_list("example", skipheader=True, listoflistmode=True)[-1], [["Extra"]]
        )

    def test_light_and_medium_exist_independently_and_rng_draw_order_is_retained(self):
        self.write("csvfiles/onlylight.csv", "full")
        self.write("csvfiles/onlylight_light.csv", "light")
        self.write("csvfiles/onlymedium.csv", "full")
        self.write("csvfiles/onlymedium_medium.csv", "medium")
        with patch.object(random, "randint", return_value=0) as draw:
            self.assertEqual(reader.csv_to_list("onlylight", insanitylevel=1), ["light"])
            draw.assert_called_once_with(0, 13)
        with patch.object(random, "randint", return_value=0):
            self.assertEqual(reader.csv_to_list("onlymedium", insanitylevel=3), ["medium"])
        with patch.object(random, "randint", return_value=13):
            self.assertEqual(reader.csv_to_list("onlymedium", insanitylevel=6), ["full"])
        for level in (1, 3, 4, 6, 7, 10):
            random.seed(1234)
            store.clear_cache()
            cold = reader.csv_to_list("onlymedium", insanitylevel=level), random.getstate()
            random.seed(1234)
            warm = reader.csv_to_list("onlymedium", insanitylevel=level), random.getstate()
            self.assertEqual(cold, warm)

    def test_utf8_bom_quoted_delimiter_empty_and_gender_short_rows(self):
        self.write(
            "test.csv",
            '\ufeffvalue;gender\n\n"café; ?";female\nShort\nBoth;both\nMale;male\n',
            user=True,
        )
        with self.assertWarnsRegex(RuntimeWarning, "missing gender column"):
            self.assertEqual(
                reader.csv_to_list(
                    "test", directory="./userfiles/", gender="female", skipheader=True
                ),
                ["café; ?", "Both"],
            )
        self.write("empty.csv", "", user=True)
        self.assertEqual(reader.csv_to_list("empty", directory="./userfiles/", skipheader=True), [])
        self.write("broken.csv", '"unterminated', user=True)
        with self.assertRaisesRegex(ValueError, "broken.csv:1: malformed CSV"):
            reader.csv_to_list("broken", directory="./userfiles/")

    def test_stable_dedup_case_antilist_and_transform(self):
        self.write("test.csv", "Header\nWord\nword\n Word \nOther\n", user=True)
        self.assertEqual(
            reader.csv_to_list("test", directory="./userfiles/", skipheader=True),
            ["Word", " Word ", "Other"],
        )
        self.assertEqual(
            reader.csv_to_list("test", ["word"], "./userfiles/", 1, skipheader=True), ["other"]
        )

    def test_cache_reuses_immutable_rows_and_invalidates_edit_create_delete(self):
        path = self.write("test.csv", "old", user=True)
        with patch("builtins.open", wraps=open) as opened:
            first = store.read_rows(path)
            self.assertIs(first, store.read_rows(path))
            self.assertEqual(opened.call_count, 1)
        with self.assertRaises(TypeError):
            first[0][0] = "mutation"
        path.write_text("new-value", encoding="utf-8")
        self.assertEqual(reader.csv_to_list("test", directory="./userfiles/"), ["new-value"])
        path.unlink()
        self.assertEqual(reader.csv_to_list("test", directory="./userfiles/"), [])
        path.write_text("recreated", encoding="utf-8")
        self.assertEqual(reader.csv_to_list("test", directory="./userfiles/"), ["recreated"])
        with patch.object(store, "CACHE_MAX_ENTRIES", 2):
            for index in range(3):
                store.read_rows(self.write(str(index) + ".csv", str(index), user=True))
            self.assertLessEqual(store.cache_info()["entries"], 2)
        with patch.object(store, "CACHE_MAX_BYTES", 1):
            store.clear_cache()
            store.read_rows(path)
            self.assertEqual(store.cache_info()["entries"], 0)

    def test_artist_order_duplicates_projections_are_indexed_once(self):
        self.write(
            "csvfiles/artists.csv",
            "Artist,Tags,Medium,Description,popular\nA,t,m,d,1\nB,u,n,e,0\nA,v,o,f,1\n",
        )
        with patch("builtins.open", wraps=open) as opened:
            self.assertEqual(reader.artist_category_csv_to_list("artists", "popular"), ["A", "A"])
            self.assertEqual(
                reader.artist_category_by_category_csv_to_list("artists", "A"),
                (["t", "v"], ["m", "o"], ["d", "f"]),
            )
            descriptions = reader.artist_descriptions_csv_to_list("artists")
            self.assertEqual(descriptions, ["d", "e", "f"])
            descriptions.clear()
            self.assertEqual(reader.artist_descriptions_csv_to_list("artists"), ["d", "e", "f"])
            self.assertEqual(opened.call_count, 1)

    def test_negative_pairs_short_rows_and_config_fallback_are_read_only(self):
        self.write(
            "csvfiles/special_lists/negativewords.csv",
            "primer;negative\nphoto;painting\nshort\nanime;photo\n",
        )
        self.write("negativewords_addon.csv", "primer;negative\nnew;avoid\n", user=True)
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(
                reader.load_negative_list(),
                (["photo", "anime", "new"], ["painting", "photo", "avoid"]),
            )
        self.write(
            "csvfiles/config/default_config_test.csv", "header;option\n#comment;\nkeep;on\nshort\n"
        )
        with self.assertWarnsRegex(RuntimeWarning, "expected 2 config columns"):
            self.assertEqual(reader.load_config_csv("test"), [["keep", "on"]])
        self.assertFalse((self.user / "config_test.csv").exists())

    def test_table_diagnostic_counts_blank_records(self):
        path = self.write("csvfiles/table.csv", "primer;negative\n\nshort\n")
        with self.assertWarnsRegex(RuntimeWarning, "record 3: expected 2 columns"):
            self.assertEqual(
                store.table_rows(path, ("primer", "negative")), (("primer", "negative"), ())
            )

    def test_token_alias_preserves_header_options_and_requires_explicit_copy(self):
        source = self.write("csvfiles/tokens.csv", "10-point\n1080\n11-point\n")
        self.assertEqual(
            reader.csv_to_list("tokenwriter", directory="./csvfiles/special_lists/"),
            ["10-point", "1080", "11-point"],
        )
        self.assertEqual(reader.csv_to_list("tokens", skipheader=True), ["1080", "11-point"])
        copy = self.root / "copy.csv"
        reader.export_token_dictionary(copy)
        self.assertEqual(copy.read_text(encoding="utf-8"), source.read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            reader.export_token_dictionary(source)
        with self.assertRaises(TypeError):
            reader.sort_and_dedupe_csv_file()

    def test_preset_discovery_preserves_bytes_and_save_is_explicit(self):
        file = self.write(
            "obp_presets.json", json.dumps({"Mine": {"subject": "custom"}}), user=True
        )
        before = file.read_bytes()
        target = self.root / "new-data/userfiles/obp_presets.json"
        with patch.object(OneButtonPresets, "OBP_FILE", target):
            presets = OneButtonPresets()
            self.assertIn("Standard", presets.opb_presets)
            self.assertIn("Mine", presets.opb_presets)
            self.assertFalse(target.parent.exists())
            self.assertEqual(file.read_bytes(), before)
            presets.save_obp_preset(presets.opb_presets)
            self.assertTrue(target.is_file())
            self.assertEqual(file.read_bytes(), before)


class RealRootTests(unittest.TestCase):
    def test_rf_data_and_legacy_package_roots_are_ordered_without_mkdir(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            app, data, package = root / "app", root / "data", root / "package"
            with (
                patch.dict(os.environ, RF_DATA_DIR=str(data)),
                patch.object(store, "APP_ROOT", app),
                patch.object(store, "PACKAGE", package),
            ):
                roots = store.user_roots()
                self.assertEqual(
                    roots,
                    (
                        data / "random_prompt/userfiles",
                        app / "random_prompt/userfiles",
                        package / "userfiles",
                    ),
                )
                self.assertFalse(data.exists())
                for index, candidate in reversed(list(enumerate(roots))):
                    candidate.mkdir(parents=True, exist_ok=True)
                    (candidate / "custom.csv").write_text(str(index), encoding="utf-8")
                    self.assertEqual(store.user_file("custom.csv"), candidate / "custom.csv")


if __name__ == "__main__":
    unittest.main()
