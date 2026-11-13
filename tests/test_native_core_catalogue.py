"""Native core acquisition rows bind sources, variants, bundles and usable defaults."""

import json
import unittest

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.features.models.model_recommendations import family_recipe
from ruinedfooocus.features.models.performance import PAYLOAD_RECIPES


class NativeCoreCatalogueTests(unittest.TestCase):
    def test_source_classified_tuples_keep_exact_companions_and_native_recipe_identity(self):
        registry = load_registry()
        count = 0
        for path in resource_path("model_catalogue").glob("native_core_*.json"):
            for key, download in json.loads(path.read_text(encoding="utf-8")).items():
                count += 1
                row = registry.annotations_for(key, download)
                self.assertEqual(row["artifact"]["sha256"], download["sha256"])
                self.assertEqual(row["artifact"]["bytes"], download["size"])
                self.assertEqual(row["artifact"]["pinned_url"], download["url"])
                self.assertNotIn("/resolve/main/", download["url"])
                self.assertGreater(row["artifact"]["header_bytes"], 8)
                self.assertFalse(row["artifact"]["full_file_hash_recomputed"])
                self.assertFalse(row["chat_weight"])
                self.assertEqual(
                    row["native_class"], row["evidence"]["native_classification"]["native_class"]
                )
                self.assertEqual(
                    row["evidence"]["native_classification"]["header_sha256"],
                    row["artifact"]["header_sha256"],
                )
                self.assertTrue(row["evidence"]["default_recipe_id"])
                for identity in row["companions"]["text"]:
                    self.assertIsNotNone(registry.artifact(identity))
                if key.startswith("checkpoints/"):
                    self.assertEqual(row["companions"]["text"], ())
                else:
                    self.assertTrue(row["companions"]["text"] or row["native_class"] == "HiDreamO1")
        self.assertEqual(count, 10)

    def test_new_omni_and_kandinsky_geometry_match_existing_qualified_native_baselines(self):
        for family in ("Omnigen2", "Kandinsky5Image"):
            recipe = family_recipe(family)
            self.assertEqual(recipe["default_size"], [1024, 1024])
            self.assertEqual(recipe["sampling"], PAYLOAD_RECIPES[family][1])
            self.assertEqual(recipe["provenance"]["kind"], "native_workflow")

    def test_coverage_additions_resolve_eight_classes_without_certifying_execution(self):
        rows = {row["id"]: row for row in load_registry().native_dispositions()}
        for family in (
            "Anima",
            "Omnigen2",
            "Kandinsky5Image",
            "FluxSchnell",
            "HiDream",
            "HiDreamO1",
            "MageFlow",
            "Krea2",
        ):
            self.assertEqual(rows[family]["download_disposition"], "registered-artifacts")
            self.assertTrue(rows[family]["available_artifact_ids"])
            self.assertEqual(rows[family]["qualification"], "not-inherited-from-source-support")


if __name__ == "__main__":
    unittest.main()
