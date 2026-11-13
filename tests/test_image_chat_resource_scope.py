"""RF product catalogues stay image/Chat scoped when native registrations change."""

import ast
import json
from pathlib import Path
import unittest

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.features.models.resource_defaults import load_family_recipe, performance_defaults

ROOT = Path(__file__).resolve().parents[1]
REMOVED_NATIVE_CLASSES = {
    "ACEStep",
    "ACEStep15",
    "CogVideoX_I2V",
    "CogVideoX_Inpaint",
    "CogVideoX_T2V",
    "CosmosI2V",
    "CosmosI2VPredict2",
    "CosmosT2V",
    "GenmoMochi",
    "Hunyuan3Dv2",
    "Hunyuan3Dv2_1",
    "Hunyuan3Dv2mini",
    "HunyuanVideo",
    "HunyuanVideo15",
    "HunyuanVideo15_SR_Distilled",
    "HunyuanVideoI2V",
    "HunyuanVideoSkyreelsI2V",
    "Kandinsky5",
    "LTXAV",
    "LTXV",
    "MiniMaxH3",
    "MiniMaxMusic3",
    "SV3D_p",
    "SV3D_u",
    "SVD_img2vid",
    "StableAudio",
    "StableAudio3",
    "Stable_Zero123",
    "Trellis2",
    "TripoSplat",
    "WAN21_Camera",
    "WAN21_CausalAR_T2V",
    "WAN21_FlowRVS",
    "WAN21_FunControl2V",
    "WAN21_HuMo",
    "WAN21_I2V",
    "WAN21_SCAIL",
    "WAN21_SCAIL2",
    "WAN21_T2V",
    "WAN21_Vace",
    "WAN22_Animate",
    "WAN22_Camera",
    "WAN22_RelayVSR",
    "WAN22_S2V",
    "WAN22_T2V",
    "WAN22_WanDancer",
    "WAN_Animate2",
    "YuE2",
}
REMOVED_CLIP_TYPES = {
    "STABLE_AUDIO",
    "MOCHI",
    "LTXV",
    "HUNYUAN_VIDEO",
    "WAN",
    "ACE",
    "HUNYUAN_VIDEO_15",
    "OVIS",
    "KANDINSKY5",
    "COGVIDEOX",
    "YUE2",
}
REMOVED_FOLDER_ROLES = {
    "latent_upscale_models",
    "audio_encoders",
    "frame_interpolation",
    "optical_flow",
    "vae_gguf",
    "latent_upscale_models_gguf",
    "custom_nodes",
    "datasets",
    "classifiers",
}
RETAINED_TE_MODELS = {
    "BYT5_SMALL_GLYPH",
    "CLIP_G",
    "CLIP_H",
    "CLIP_L",
    "GEMMA_2_2B",
    "GEMMA_3_4B",
    "GPT_OSS_20B",
    "JINA_CLIP_2",
    "LLAMA3_8",
    "MING_IMAGE",
    "MINISTRAL_3_3B",
    "MISTRAL3_24B_PRUNED_FLUX2",
    "QWEN25_3B",
    "QWEN25_7B",
    "QWEN35_9B",
    "QWEN3VL_32B",
    "QWEN3VL_4B",
    "QWEN3VL_8B",
    "QWEN3_06B",
    "QWEN3_4B",
    "QWEN3_8B",
    "T5_XXL",
    "T5_XXL_OLD",
    "T5_XL",
    "WEVISONE_2B",
    "WEVISONE_4B",
}


class ImageChatResourceScopeTests(unittest.TestCase):
    def test_admission_helper_rejects_retired_families_and_keeps_every_retained_row(self):
        from ruinedfooocus.features.models.image_scope import require_image_family

        for family in REMOVED_NATIVE_CLASSES:
            with (
                self.subTest(retired_family=family),
                self.assertRaisesRegex(ValueError, "still images"),
            ):
                require_image_family(family)
        for row in load_registry().native_dispositions():
            with self.subTest(retained_family=row["id"]):
                require_image_family(row["id"])

    def test_all_recipe_indexes_and_performance_defaults_are_retained_image_consumers(self):
        root = resource_path("defaults")
        index = json.loads((root / "index.json").read_text(encoding="utf-8"))
        for label, descriptor in index["recipes"].items():
            with self.subTest(recipe=label):
                self.assertEqual(descriptor["media"], "image")
                recipe = load_family_recipe(label)
                if recipe["id"] == "native-bernini-image":
                    # A source-bound still adapter is distinct from admitting
                    # the shared Wan class or a generic temporal workflow.
                    self.assertEqual(recipe["family"], "BerniniImage")
                    self.assertEqual(recipe["applicability"]["architecture"], "BerniniImage")
                    self.assertFalse(recipe["applicability"]["default_for_native_configuration"])
                    self.assertNotIn("WAN21_T2V", index["native_recipes"])
                    continue
                self.assertNotIn(
                    recipe.get("applicability", {}).get("native_class"), REMOVED_NATIVE_CLASSES
                )
        self.assertFalse((root / "video_resolutions.default").exists())
        self.assertEqual(list((root / "video").glob("*.json")), [])
        self.assertTrue(all("video_models" not in row for row in performance_defaults().values()))

    def test_native_product_rows_never_inherit_media_or_unused_source_roles(self):
        registry = load_registry()
        rows = registry.native_dispositions()
        self.assertEqual({row["media"] for row in rows}, {"image", "utility"})
        self.assertFalse(REMOVED_NATIVE_CLASSES & {row["id"] for row in rows})
        self.assertTrue(
            all(row["media"] == "image" for row in registry.native_dispositions(kind="clip_types"))
        )
        self.assertEqual(
            {row["id"] for row in registry.native_dispositions(kind="te_models")},
            RETAINED_TE_MODELS,
        )
        self.assertTrue(
            all(
                row["media"] == "image-or-utility"
                for row in registry.native_dispositions(kind="roles")
            )
        )
        self.assertFalse(
            REMOVED_FOLDER_ROLES & {row["id"] for row in registry.native_dispositions(kind="roles")}
        )

    def test_every_download_and_companion_has_an_active_non_media_registry_identity(self):
        registry = load_registry()
        manifest = json.loads(resource_path("bernini_image_components.json").read_text())
        bernini = manifest["components"]
        high = bernini["high"]
        expected_primary = "existing:" + high["catalogue_key"]
        for row in registry.entries():
            self.assertNotIn(row["role"], {"audio_video_diffusion_checkpoint", "latent_upscaler"})
            if row["id"] == expected_primary:
                self.assertEqual(manifest["comfy_pin"], "989d93d0d968e704c589ce0146bb18c77f6b50cc")
                self.assertEqual(
                    manifest["blueprint_sha256"],
                    "da28b3c871ac9f5e09c30ca1ed182bcf3ce29cd028cf2383bd43b99013c42ed2",
                )
                self.assertEqual(row["profile_id"], "generation:bernini_image")
                self.assertEqual(row["native_family"], "BerniniImage")
                self.assertEqual(row["architecture"], "BerniniImage")
                self.assertEqual(row["native_class"], "WAN21_T2V")
                self.assertEqual(row["behavioral_variant"], "image-edit")
                self.assertEqual(row["role"], "image_diffusion_checkpoint")
                for field in ("bytes", "sha256", "header_sha256"):
                    self.assertEqual(row["artifact"][field], high[field])
                self.assertEqual(row["artifact"]["pinned_url"], high["url"])
                self.assertEqual(row["evidence"]["default_recipe_id"], "native-bernini-image")
                self.assertEqual(
                    row["component_contract"]["source_workflow"],
                    "blueprints/Image Edit (Bernini-R).json",
                )
                companions = [
                    "existing:" + bernini[role]["catalogue_key"] for role in ("low", "clip", "vae")
                ]
                for mode in ("text", "image"):
                    self.assertEqual(list(row["companions"][mode]), companions)
            else:
                self.assertNotIn(row.get("native_class"), REMOVED_NATIVE_CLASSES)
            for ids in row["companions"].values():
                for identity in ids:
                    self.assertIsNotNone(registry.artifact(identity))
        for path in resource_path("model_catalogue").glob("*.json"):
            for key, download in json.loads(path.read_text(encoding="utf-8")).items():
                with self.subTest(catalogue=key):
                    self.assertIsNotNone(registry.annotations_for(key, download))

    def test_shared_video_derived_components_have_explicit_connected_image_consumers(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/models/image_components.py").read_text(encoding="utf-8")
        )
        components = ast.literal_eval(
            next(
                node.value
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "_COMPONENT_KEYS"
                    for target in node.targets
                )
            )
        )
        registry = load_registry()
        cases = {
            "generation:wan21_vae": {"CosmosPredict2": "vae_wan", "JoyImage": "vae_wan"},
            "generation:minimax_official_encoder": {"MiniMaxH3Image": "clip_minimax_h3_image"},
        }
        for identity, consumers in cases.items():
            profile = registry.profile(identity)
            self.assertEqual(set(profile["consumers"]), set(consumers))
            for family, role in consumers.items():
                self.assertIn(role, (*components[family][0], components[family][1]))
        for key in (
            "existing:clip/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            "existing:vae/wan_2.1_vae.safetensors",
            "existing:vae/hunyuan_image_2.1_vae_fp16.safetensors",
            "existing:vae/hunyuan_image_refiner_vae_fp16.safetensors",
            "existing:vae/seedvr2_ema_vae_fp16.safetensors",
        ):
            self.assertIsNotNone(registry.artifact(key))


if __name__ == "__main__":
    unittest.main()
