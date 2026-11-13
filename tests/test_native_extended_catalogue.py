"""Final catalogue coverage keeps native payloads, operation and blockers distinct."""

from collections import Counter
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models.acquisition import Catalogue
from ruinedfooocus.features.models.component_candidates import classify_shapes, compatibility
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.features.models.resource_defaults import load_family_recipe, resource_name
from test_component_candidates import qwen_wan21_shapes


class ExtendedCatalogueTests(unittest.TestCase):
    def rows(self):
        registry = load_registry()
        return [
            (registry.annotations_for(key, value), value)
            for path in resource_path("model_catalogue").glob("native_extended_*.json")
            for key, value in json.loads(path.read_text(encoding="utf-8")).items()
        ]

    def test_exact_published_tuples_bind_native_headers_recipes_and_all_companions(self):
        registry = load_registry()
        rows = self.rows()
        self.assertEqual(len(rows), 27)
        self.assertEqual(len({row.get("native_class") for row, _ in rows} - {None}), 23)
        self.assertEqual(
            Counter(row["role"] for row, _ in rows),
            {
                "image_diffusion_checkpoint": 20,
                "image_utility_model": 5,
                "depth_estimation": 1,
                "image_codec": 1,
            },
        )
        for row, download in rows:
            with self.subTest(identity=row["id"]):
                self.assertEqual(row["artifact"]["pinned_url"], download["url"])
                self.assertRegex(download["url"], r"/resolve/[a-f0-9]{40}/")
                self.assertEqual(row["artifact"]["bytes"], download["size"])
                self.assertEqual(row["artifact"]["sha256"], download["sha256"])
                self.assertGreater(row["artifact"]["header_bytes"], 8)
                self.assertFalse(row["artifact"]["full_file_hash_recomputed"])
                self.assertFalse(row["chat_weight"])
                for identities in row["companions"].values():
                    for identity in identities:
                        self.assertIsNotNone(registry.artifact(identity))
                recipe_id = row["evidence"].get("default_recipe_id")
                if recipe_id:
                    recipe = load_family_recipe(recipe_id)
                    self.assertEqual(recipe["applicability"]["native_class"], row["native_class"])
                    self.assertEqual(recipe["variant"], row["behavioral_variant"])
                    self.assertEqual(
                        recipe["applicability"]["auto_recipe_supported"],
                        row["auto_recipe_supported"],
                    )

    def test_remaining_acquisition_rows_distinguish_access_and_local_ingestion(self):
        registry = load_registry()
        rows = [
            row for row in registry.native_dispositions() if row["media"] in ("image", "utility")
        ]
        self.assertEqual(len(rows), 62)
        self.assertEqual(sum(bool(row["available_artifact_ids"]) for row in rows), 59)
        self.assertEqual(sum(len(row["available_artifact_ids"]) for row in rows), 84)
        missing = {row["id"]: row for row in rows if not row["available_artifact_ids"]}
        self.assertEqual(set(missing), {"SAM3", "Flux3", "SenseNovaU15"})
        for family, disposition in (
            ("SAM3", "publisher-access-gated"),
            ("Flux3", "publisher-access-gated"),
            ("SenseNovaU15", "no-reviewed-exact-checkpoint-tuple"),
        ):
            row = missing[family]
            self.assertEqual(row["download_disposition"], disposition)
            blocker = row["acquisition_blocker"]
            self.assertRegex(blocker["source_revision"], r"^[a-f0-9]{40}$")
            self.assertIn(blocker["source_revision"], blocker["source_url"])
            self.assertEqual(blocker["runtime_implementation_complete"], family == "SenseNovaU15")
            self.assertRegex(blocker["evidence_sha256"], r"^[a-f0-9]{64}$")
        self.assertIn("HTTP 401", missing["SAM3"]["download_reason"])
        pixel = next(row for row in rows if row["id"] == "ZImagePixelSpace")
        self.assertIn("convolutional", pixel["excluded_source"]["reason"].lower())
        self.assertEqual(pixel["download_disposition"], "registered-artifacts")
        self.assertIn("eight safetensors shards", missing["SenseNovaU15"]["download_reason"])

    def test_native_class_distinctions_and_published_integrity_do_not_follow_local_names(self):
        registry = load_registry()
        expected = {
            "sd15_unet_fp16.safetensors": (
                "SD15",
                "c83908253f9a64d08c25fc90874c9c8aef9a329ce1ca5fb909d73b0c83d1ea21",
            ),
            "sdxl_instructpix2pix_unet.safetensors": (
                "SDXL_instructpix2pix",
                "01d538927edfdf63511a38840b9d0cd8d8dcd4e40001566b83c9104ef1a2f92f",
            ),
            "pixart_alpha_xl_2_1024_ms.safetensors": (
                "PixArtAlpha",
                "809a92d52a4a228f381a4b4f4b76051294b73285fb0cbb02f0ad24f9372217a8",
            ),
            "pixart_sigma_xl_2_1024_ms.safetensors": (
                "PixArtSigma",
                "2fa4dee9229c02b03163f57bdb8e80c7a5ee364b7161796abe9c05e8dd13f239",
            ),
        }
        for filename, (native_class, digest) in expected.items():
            row = registry.artifact("existing:diffusion_models/" + filename)
            self.assertEqual(row["native_class"], native_class)
            self.assertEqual(row["artifact"]["sha256"], digest)
        for version in ("1.0", "1.1"):
            row = registry.artifact("existing:checkpoints/hunyuan_dit_" + version + ".safetensors")
            self.assertEqual(row["native_class"], "HunyuanDiT1")
            self.assertEqual(row["behavioral_variant"], version)
            self.assertTrue(row["component_contract"]["embedded"]["clip"])
            self.assertTrue(row["component_contract"]["embedded"]["vae"])
            self.assertEqual(row["companions"]["text"], ())

    def test_connected_seedvr2_still_restoration_keeps_its_exact_codec(self):
        registry = load_registry()
        seed = registry.artifact("existing:diffusion_models/seedvr2_3b_int8_convrot.safetensors")
        self.assertIn("existing:vae/seedvr2_ema_vae_fp16.safetensors", seed["companions"]["text"])
        coverage = {row["id"]: row for row in registry.native_dispositions()}
        self.assertEqual(
            coverage["SeedVR2"]["RF_source_route"], "features/image_processing/native_seedvr2.py"
        )
        self.assertEqual(coverage["SeedVR2"]["app_route"], "specialized-source-route")
        self.assertFalse(seed["auto_recipe_supported"])

    def test_auto_admission_projection_reaches_hub_without_claiming_utility_activation(self):
        registry = load_registry()
        downloads = {row["catalogue_key"]: download for row, download in self.rows()}
        with tempfile.TemporaryDirectory() as folder:
            manager = SimpleNamespace(
                DOWNLOADABLE_FILES=downloads,
                get_model_folders=lambda category: [folder],
            )
            catalogue = Catalogue(manager, registry=registry)
            for row, download in self.rows():
                if row.get("auto_recipe_supported") is not False:
                    continue
                path = Path(folder) / download["filename"]
                path.write_bytes(b"x")
                with patch(
                    "ruinedfooocus.features.models.acquisition.file_signature",
                    return_value=(str(path), download["size"], 1),
                ):
                    record = catalogue.record(row["catalogue_key"])
                self.assertTrue(record["allowed"])
                self.assertTrue(record["size_matches"])
                self.assertFalse(record["can_activate"])
                if row["role"] == "image_utility_model":
                    self.assertNotEqual(record["task"], "Image")
        for row in registry.entries():
            evidence = row.get("evidence", {})
            if "auto_recipe_supported" in evidence:
                self.assertEqual(row["auto_recipe_supported"], evidence["auto_recipe_supported"])

    def test_sampling_baselines_and_unreviewed_canvases_retain_their_real_provenance(self):
        cosmos = load_family_recipe("native-extended-cosmost2ipredict2-2b-t2i")
        self.assertEqual(cosmos["sampling"]["custom_steps"], 30)
        self.assertEqual(cosmos["sampling"]["cfg"], 4)
        self.assertEqual(cosmos["sampling"]["scheduler"], "simple")
        self.assertTrue(cosmos["applicability"]["auto_recipe_supported"])
        unclip = load_family_recipe("native-extended-sd21uncliph-unclip-h")
        self.assertEqual(unclip["sampling"]["custom_steps"], 26)
        self.assertFalse(unclip["applicability"]["auto_recipe_supported"])
        pixart = load_family_recipe("native-extended-pixartalpha-xl-2-1024-ms")
        self.assertEqual(pixart["sampling"], {})
        self.assertEqual(pixart["provenance"]["kind"], "local_family_recipe")
        self.assertIn("not a publisher", pixart["provenance"]["geometry_evidence"])
        self.assertFalse(pixart["applicability"]["default_for_native_configuration"])

    def test_layered_codec_slot_validates_full_rgba_layout_and_rejects_rgb(self):
        shapes = qwen_wan21_shapes()
        rgb = classify_shapes(shapes, detector=lambda _: None)
        self.assertEqual(compatibility("vae_qwen_image_layered", rgb), "incompatible")
        shapes["encoder.conv1.weight"] = (96, 4, 3, 3, 3)
        shapes["decoder.head.2.weight"] = (4, 96, 3, 3, 3)
        rgba = classify_shapes(shapes, detector=lambda _: None)
        self.assertEqual(compatibility("vae_qwen_image_layered", rgba), "compatible")
        self.assertEqual(compatibility("vae_qwen_image", rgba), "incompatible")
        self.assertEqual(compatibility("vae_qwen_image21", rgba), "incompatible")
        shapes.pop("encoder.downsamples.8.resample.1.weight")
        self.assertEqual(
            compatibility(
                "vae_qwen_image_layered", classify_shapes(shapes, detector=lambda _: None)
            ),
            "incompatible",
        )
        self.assertEqual(
            resource_name("vae_qwen_image_layered", {}), "qwen_image_layered_vae.safetensors"
        )


if __name__ == "__main__":
    unittest.main()
