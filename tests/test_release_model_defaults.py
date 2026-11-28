"""Release defaults use existing native recipes without manual qualification."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.features.generation.sdxl_refiner import settings
from ruinedfooocus.features.models.model_identity import file_signature, cached_native_source
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.features.models.native_recipes import compatible_native_recipe
from ruinedfooocus.features.models.resource_defaults import load_family_recipe
import test_model_recommendations as recommendations
from test_native_recipe_contracts import boogu_shapes
from test_native_variant_defaults import native_backend, write_header


class ReleaseDefaultsTests(unittest.TestCase):
    def test_cached_exact_boogu_identity_without_receipt_or_render_hashing(self):
        original = load_registry().artifact(
            "existing:diffusion_models/boogu_image_edit_fp8_scaled.safetensors"
        )
        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            path = write_header(root / "renamed.safetensors", boogu_shapes(), "F8_E4M3")
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            row = dict(original) | {
                "artifact": dict(original["artifact"])
                | {"sha256": checksum, "bytes": path.stat().st_size}
            }
            metadata = {
                "_rf_local_sha256": checksum.upper(),
                "_rf_file_signature": file_signature(path),
            }
            with patch("ruinedfooocus.features.models.model_registry.load_registry") as registry:
                registry.return_value.entries.return_value = [row]
                with patch(
                    "ruinedfooocus.features.models.model_identity.cached_file_sha256",
                    side_effect=AssertionError("no hashing in defaults"),
                ):
                    recipe = compatible_native_recipe(path, metadata)
                    self.assertEqual(recipe["id"], "native-boogu-edit")
                    self.assertTrue(recipe["auto_recipe_supported"])
                    perf = recommendations.RecommendationTests()._performance(root)
                    with (
                        patch.object(perf.model_recommendations, "metadata", return_value=metadata),
                        patch.object(
                            perf,
                            "sampling_choices",
                            return_value=(("euler", "dpmpp_2m"), ("normal", "simple")),
                        ),
                    ):
                        values = perf.get_perf_options("Model Defaults", str(path), "Boogu")
                    self.assertEqual((values["custom_steps"], values["cfg"]), (25, 3.5))
                    self.assertEqual(
                        (values["sampler_name"], values["scheduler"]), ("dpmpp_2m", "simple")
                    )
                for bad in (
                    metadata | {"_rf_file_signature": ["old", 1, 2, 3]},
                    metadata | {"_rf_local_sha256": "0" * 64},
                    {},
                ):
                    self.assertEqual(compatible_native_recipe(path, bad), {})

    def test_ambiguous_registered_artifact_contracts_remain_a_miss(self):
        original = load_registry().artifact(
            "existing:diffusion_models/boogu_image_edit_fp8_scaled.safetensors"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.safetensors"
            path.write_bytes(b"bounded identity fixture")
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            row = dict(original) | {
                "artifact": dict(original["artifact"])
                | {"sha256": checksum, "bytes": path.stat().st_size}
            }
            with patch("ruinedfooocus.features.models.model_registry.load_registry") as registry:
                registry.return_value.entries.return_value = [
                    row,
                    row | {"behavioral_variant": "turbo"},
                ]
                self.assertIsNone(
                    cached_native_source(
                        path,
                        {"_rf_local_sha256": checksum, "_rf_file_signature": file_signature(path)},
                    )
                )

    def test_refiner_native_recipe_is_automatic_for_current_configuration(self):
        recipe = load_family_recipe("native-extended-sdxlrefiner-base")
        self.assertEqual(
            recipe["sampling"],
            dict(custom_steps=25, cfg=8, sampler_name="euler", scheduler="normal", clip_skip=1),
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = write_header(root / "renamed.safetensors", {})
            contract = dict(
                native_class="SDXLRefiner",
                architecture="SDXLRefiner",
                native_config={"in_channels": 4},
                metadata={},
            )
            with (
                patch(
                    "ruinedfooocus.features.models.model_architecture.native_detector_ready",
                    return_value=True,
                ),
                patch(
                    "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                    return_value=contract,
                ),
            ):
                self.assertTrue(compatible_native_recipe(path, {})["auto_recipe_supported"])
                perf = recommendations.RecommendationTests()._performance(root)
                values = perf.get_perf_options("Model Defaults", str(path), "SDXLRefiner")
                self.assertEqual(
                    (values["custom_steps"], values["cfg"], values["scheduler"]), (25, 8, "normal")
                )

    def test_blank_refiner_handoff_tracks_native_example_and_steps(self):
        self.assertEqual(settings(dict(custom_steps=25)), (25, 20, None))
        self.assertEqual(
            settings(dict(custom_steps=25, sdxl_refiner_switch_step=0)), (25, 20, None)
        )
        self.assertEqual(settings(dict(custom_steps=30)), (30, 24, None))
        self.assertEqual(settings(dict(custom_steps=2)), (2, 1, None))
        self.assertEqual(
            settings(dict(custom_steps=25, sdxl_refiner_switch_step=12, sdxl_refiner_cfg=0)),
            (25, 12, 0),
        )

    def test_optional_refiner_cfg_accepts_automatic_and_custom_numeric_values(self):
        self.assertEqual(settings(dict(custom_steps=25, sdxl_refiner_cfg="")), (25, 20, None))
        self.assertEqual(settings(dict(custom_steps=25, sdxl_refiner_cfg="0")), (25, 20, 0))
        self.assertEqual(settings(dict(custom_steps=25, sdxl_refiner_cfg="3.25")), (25, 20, 3.25))
        for value in ("invalid", "nan", "101", -1, False):
            with self.subTest(value=value), self.assertRaises(ValueError):
                settings(dict(custom_steps=25, sdxl_refiner_cfg=value))

    def test_refiner_auto_help_is_localized_in_every_shipped_language(self):
        import json
        from ruinedfooocus.core.paths import resource_path

        for path in resource_path("translations").glob("*.json"):
            if path.stem in ("manifest", "legacy-keys"):
                continue
            with self.subTest(locale=path.stem):
                text = json.loads(path.read_text(encoding="utf-8"))[
                    "generation.blank_refiner_switch_uses_80_percent"
                ]
                self.assertIn("0", text)
                self.assertTrue(text.strip())


if __name__ == "__main__":
    unittest.main()
