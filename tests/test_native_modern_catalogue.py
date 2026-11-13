"""Modern image sources retain exact variants and complete workflow requirements."""

import json
import unittest

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.features.models.resource_defaults import load_family_recipe


class ModernCatalogueTests(unittest.TestCase):
    def test_all_source_tuples_bind_current_registry_components_and_small_recipe_resources(self):
        registry = load_registry()
        rows = []
        for path in resource_path("model_catalogue").glob("native_modern_*.json"):
            for key, download in json.loads(path.read_text(encoding="utf-8")).items():
                row = registry.annotations_for(key, download)
                rows.append(row)
                self.assertEqual(row["artifact"]["pinned_url"], download["url"])
                self.assertEqual(row["artifact"]["sha256"], download["sha256"])
                self.assertEqual(row["artifact"]["bytes"], download["size"])
                self.assertRegex(download["url"], r"/resolve/[a-f0-9]{40}/")
                self.assertFalse(row["artifact"]["full_file_hash_recomputed"])
                self.assertGreater(row["artifact"]["header_bytes"], 8)
                self.assertFalse(row["chat_weight"])
                for kind in ("text", "image"):
                    for identity in row["companions"][kind]:
                        self.assertIsNotNone(registry.artifact(identity))
                recipe_id = row["evidence"].get("default_recipe_id")
                if recipe_id:
                    recipe = load_family_recipe(recipe_id)
                    self.assertEqual(recipe["id"], recipe_id)
                    self.assertTrue(recipe["applicability"]["quantization_independent"])
        self.assertEqual(len(rows), 35)
        self.assertEqual(len({row.get("native_class") for row in rows} - {None}), 17)
        self.assertEqual(sum(row["role"] == "image_diffusion_checkpoint" for row in rows), 33)
        self.assertEqual(sum(row["role"] == "image_diffusion_companion" for row in rows), 1)
        self.assertEqual(sum(row["role"] == "image_codec" for row in rows), 1)

    def test_conditional_model_and_rgba_codec_cannot_masquerade_as_ordinary_primary_files(self):
        registry = load_registry()
        primary = registry.artifact("existing:diffusion_models/ideogram4_fp8_scaled.safetensors")
        secondary_id = "existing:diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors"
        self.assertEqual(primary["companions"]["image"], (secondary_id,))
        self.assertEqual(registry.artifact(secondary_id)["role"], "image_diffusion_companion")
        self.assertTrue(primary["evidence"]["auto_recipe_supported"])
        self.assertIn("DualModelGuider", primary["evidence"]["recipe_requirements"])
        self.assertIn("Ideogram4Scheduler", primary["evidence"]["recipe_requirements"])
        coverage = {row["id"]: row for row in registry.native_dispositions()}
        self.assertNotIn(secondary_id, coverage["Ideogram4"]["available_artifact_ids"])
        codec = registry.artifact("existing:vae/qwen_image_layered_vae.safetensors")
        self.assertEqual(codec["component_contract"]["image_channels"], 4)
        self.assertEqual(codec["component_contract"]["latent_channels"], 16)
        self.assertEqual(codec["component_contract"]["encoder_input_shape"], (96, 4, 3, 3, 3))
        layered = registry.artifact("existing:diffusion_models/qwen_image_layered_bf16.safetensors")
        self.assertIn(codec["id"], layered["companions"]["text"])
        self.assertFalse(layered["evidence"]["auto_recipe_supported"])

    def test_variant_sampling_and_operation_contracts_are_not_filename_guesses(self):
        base = load_family_recipe("native-zimage-base")
        turbo = load_family_recipe("native-zimage-turbo")
        self.assertEqual(
            (base["sampling"]["custom_steps"], turbo["sampling"]["custom_steps"]), (25, 8)
        )
        self.assertEqual(load_family_recipe("boogu_image_turbo_fp8_scaled.safetensors"), {})
        chroma = load_family_recipe("native-chromaradiance-radiance-x0")
        self.assertTrue(chroma["applicability"]["auto_recipe_supported"])
        self.assertEqual(chroma["sampling"]["scheduler"], "BetaSamplingScheduler")
        for family in ("QwenImage", "LongCatImage"):
            variants = {
                row["behavioral_variant"]
                for row in load_registry().entries(role="image_diffusion_checkpoint")
                if row.get("native_family") == family
            }
            self.assertIn("edit", variants)
            self.assertIn("base", variants)


if __name__ == "__main__":
    unittest.main()
