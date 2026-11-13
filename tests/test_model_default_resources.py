"""Family recipes separate geometry, behavior, quantization and exact evidence."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models.model_registry import RegistryError, metadata_digest
from ruinedfooocus.features.models.model_recommendations import family_recipe
from ruinedfooocus.features.models.resource_defaults import load_family_recipe, performance_defaults


class FamilyResourceTests(unittest.TestCase):
    def test_all_legacy_presets_are_an_exact_projection_of_small_media_resources(self):
        presets = performance_defaults()
        self.assertEqual(len(presets), 26)
        self.assertEqual(
            metadata_digest(presets),
            "5dde2b88f603d0cc76a070fbaddac7f0ed9c3894e8eb83a33ee44c09c9d7beb9",
        )
        index = json.loads(resource_path("defaults", "index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["legacy_sha256"], metadata_digest(presets))
        self.assertEqual(len(index["recipes"]), 93)
        self.assertEqual(len({item["id"] for item in index["recipes"].values()}), 93)
        bernini = index["recipes"]["Bernini-R image edit"]
        self.assertEqual(
            (bernini["id"], bernini["family"], bernini["path"], bernini["media"]),
            ("native-bernini-image", "BerniniImage", "image/native-bernini-image.json", "image"),
        )
        for label, descriptor in index["recipes"].items():
            recipe = load_family_recipe(label, media=descriptor["media"])
            self.assertEqual(recipe["id"], descriptor["id"])
            self.assertTrue(recipe["provenance"]["source_revision"])
            self.assertIn(
                recipe["provenance"]["kind"],
                ("local_family_recipe", "publisher_geometry", "native_workflow"),
            )
            self.assertEqual(descriptor["media"], "image")

    def test_qwen_publisher_2k_and_comfy_1k_are_distinct_with_override_free_data(self):
        base = family_recipe("QwenImage21", variant="base")
        turbo = family_recipe("Qwen Image 2.1 Turbo")
        self.assertEqual(base["default_size"], [2048, 2048])
        self.assertEqual(len(base["recommended_sizes"]), 7)
        self.assertNotIn([1024, 1024], base["recommended_sizes"])
        self.assertEqual(base["working_recipe"]["default_size"], [1024, 1024])
        self.assertEqual(base["provenance"]["kind"], "publisher_geometry")
        self.assertEqual(base["working_recipe"]["provenance"]["kind"], "native_workflow")
        self.assertEqual(base["sampling"]["custom_steps"], 25)
        self.assertEqual(base["publisher"]["sampling"]["custom_steps"], 40)
        self.assertEqual(turbo["sampling"]["custom_steps"], 8)
        self.assertEqual(len(turbo["fixed_sigmas"]), 9)
        self.assertTrue(base["applicability"]["quantization_independent"])
        self.assertEqual(
            base["applicability"]["component_contract"]["vae_role"], "vae_qwen_image21"
        )
        self.assertEqual(family_recipe("QwenImage21", variant="unknown"), {})
        self.assertEqual(family_recipe("custom-turbo-Q4_K_M.gguf"), {})

    def test_aliases_are_deterministic_and_returned_values_cannot_poison_cache(self):
        for alias in ("SDXL", "SDXL 1.0"):
            self.assertEqual(family_recipe(alias)["default_size"], [1024, 1024])
        self.assertEqual(
            family_recipe("Anima")["applicability"]["component_contract"]["vae_role"],
            "vae_qwen_image",
        )
        first = family_recipe("Anima")
        first["sampling"]["cfg"] = 99
        self.assertEqual(family_recipe("Anima")["sampling"]["cfg"], 4)

    def test_corrupt_missing_escaped_and_unknown_schema_recipes_have_explicit_errors(self):
        original = json.loads(resource_path("defaults", "index.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index_path = root / "index.json"
            index_path.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaises(RegistryError):
                load_family_recipe("Anima", root=root)
            original["recipes"]["Anima"]["path"] = "../outside.json"
            index_path.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(RegistryError, "escapes"):
                load_family_recipe("Anima", root=root)
            index_path.write_text('{"schema_version":99}', encoding="utf-8")
            with self.assertRaises(RegistryError):
                load_family_recipe("Anima", root=root)

    def test_recipe_lookup_reads_neither_weights_nor_qualification_nor_tensor_runtimes(self):
        script = """
import sys
from pathlib import Path
from unittest.mock import patch
from ruinedfooocus.features.models.model_recommendations import family_recipe
original = Path.read_text
def read(path, *a, **kw):
    assert 'qualification' not in str(path) and 'model_metadata' not in str(path)
    return original(path, *a, **kw)
with patch.object(Path, 'read_text', read):
    assert family_recipe('QwenImage21')['default_size'] == [2048,2048]
assert not any(name in sys.modules for name in ('torch','comfy','ruinedfooocus.app.state'))
"""
        subprocess.run(
            [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1], check=True
        )


if __name__ == "__main__":
    unittest.main()
