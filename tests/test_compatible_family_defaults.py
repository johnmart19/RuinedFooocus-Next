"""Recognized family/variant recipes survive alternate storage and explicit overrides."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.performance import UnavailableModelRecipe
import test_model_recommendations as recommendations
from test_model_recommendations import SAMPLERS, SCHEDULERS
from test_native_variant_defaults import SHAPES, native_backend, write_header


class CompatibleFamilyDefaultsTests(unittest.TestCase):
    def test_qwen_base_turbo_alternate_quantization_and_renames_keep_behavior_separate(self):
        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            performance = recommendations.RecommendationTests()._performance(root)
            performance.model_recommendations.offline = True
            for dtype in ("F16", "BF16", "F8_E4M3"):
                for variant, steps in (("base", 25), ("turbo", 8)):
                    with self.subTest(dtype=dtype, variant=variant):
                        model = write_header(
                            root / f"unrelated-{dtype}-{variant}.safetensors",
                            SHAPES["QwenImage21"],
                            dtype,
                            {"variant": variant},
                        )
                        with patch(
                            "ruinedfooocus.features.models.model_identity.cached_file_sha256",
                            side_effect=AssertionError("no render hashing"),
                        ):
                            settings = performance.get_perf_options(
                                "Model Defaults", str(model), "QwenImage21"
                            )
                            self.assertEqual(settings["custom_steps"], steps)
                            self.assertEqual(
                                "_model_default_sigmas" in settings, variant == "turbo"
                            )
                            details = performance.model_recommendations.family_details(
                                str(model), family="QwenImage21"
                            )
                            self.assertEqual(details["default_size"], [2048, 2048])
                            self.assertEqual(
                                details["identity_basis"],
                                "current_native_header_and_explicit_variant",
                            )
                            self.assertFalse(details["artifact_qualified"])
                        performance.model_recommendations.save_override(
                            str(model), settings | dict(cfg=2), SAMPLERS, SCHEDULERS
                        )
                        self.assertEqual(
                            performance.get_perf_options(
                                "Model Defaults", str(model), "QwenImage21"
                            )["cfg"],
                            2,
                        )

    def test_qwen_logical_gguf_header_qualifies_base_without_full_file_hash(self):
        import gguf
        import numpy as np

        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            model = root / "arbitrary.gguf"
            writer = gguf.GGUFWriter(str(model), "qwen_image21")
            writer.add_string("modelspec.variant", "base")
            for name, shape in SHAPES["QwenImage21"].items():
                data = np.zeros(shape, dtype=np.float32)
                writer.add_tensor(
                    name,
                    gguf.quantize(data, gguf.GGMLQuantizationType.Q8_0),
                    raw_dtype=gguf.GGMLQuantizationType.Q8_0,
                )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            performance = recommendations.RecommendationTests()._performance(root)
            performance.model_recommendations.offline = True
            with patch(
                "ruinedfooocus.features.models.model_identity.cached_file_sha256",
                side_effect=AssertionError("no render hashing"),
            ):
                self.assertEqual(
                    performance.get_perf_options("Model Defaults", str(model), "QwenImage21")[
                        "custom_steps"
                    ],
                    25,
                )
            model.unlink()  # Header inspection must release its mmap on Windows.

    def test_ambiguous_conflicting_or_fake_family_cannot_select_behavior_from_filename(self):
        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            performance = recommendations.RecommendationTests()._performance(root)
            performance.model_recommendations.offline = True
            model = root / "qwen-image21-turbo-Q8.safetensors"
            for shapes, declaration in (
                (SHAPES["QwenImage21"], {}),
                ({"unknown.weight": [1]}, {"variant": "turbo"}),
            ):
                write_header(model, shapes, metadata=declaration)
                with self.assertRaises(UnavailableModelRecipe):
                    performance.get_perf_options("Model Defaults", str(model), "QwenImage21")
            write_header(model, SHAPES["QwenImage21"], metadata={"variant": "base"})
            (root / (model.stem + ".json")).write_text(
                json.dumps(
                    dict(
                        baseModel="QwenImage21",
                        variant="turbo",
                        _rf_file_signature=file_signature(model),
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaises(UnavailableModelRecipe):
                performance.get_perf_options("Model Defaults", str(model), "QwenImage21")


if __name__ == "__main__":
    unittest.main()
