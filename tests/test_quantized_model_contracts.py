"""Variant defaults use current native contracts and explicit publisher metadata."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.model_recommendations import published_resolution
from ruinedfooocus.features.models.performance import UnavailableModelRecipe
import test_model_recommendations as recommendations
from test_model_recommendations import SAMPLERS, SCHEDULERS
from test_native_variant_defaults import native_backend, write_header


FLUX3 = {
    "content_mode_blocks.image.0.attn_out.weight": [32, 32],
    "single_stream_modulations.txt.lin.weight": [96, 32],
    "ref_time_in.in_layer.weight": [32, 256],
}


class QuantizedModelContractTests(unittest.TestCase):
    def test_imported_flux3_flow_map_defaults_match_safetensors_and_actual_q8_gguf(self):
        import gguf

        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            perf = recommendations.RecommendationTests()._performance(root)
            perf.model_recommendations.offline = True
            model = root / "misleading-base.gguf"
            writer = gguf.GGUFWriter(str(model), "flux3")
            quant = gguf.GGMLQuantizationType.Q8_0
            for name, shape in FLUX3.items():
                writer.add_tensor(
                    name, gguf.quantize(np.zeros(shape, dtype=np.float32), quant), raw_dtype=quant
                )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            for selected in (model, write_header(root / "renamed.SFT", FLUX3, "F8_E4M3")):
                with self.subTest(selected=selected):
                    values = perf.get_perf_options("Model Defaults", str(selected))
                    self.assertEqual((values["custom_steps"], values["cfg"]), (15, 1))
                    self.assertIn("flow-map variant baseline", values["_recipe_baseline"])
            # A cached positive classification never survives an incompatible replacement.
            write_header(root / "renamed.SFT", FLUX3 | {"ref_time_in.in_layer.weight": [32, 128]})
            with self.assertRaises(UnavailableModelRecipe):
                perf.get_perf_options("Model Defaults", str(root / "renamed.SFT"), "Flux3")
            model.unlink()

    def test_declared_variants_survive_rename_dtype_and_reject_conflict_or_stale_cache(self):
        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            perf = recommendations.RecommendationTests()._performance(root)
            perf.model_recommendations.offline = True
            for family, declaration, preset in (
                ("HiDream", "full", "HiDream I1 Full"),
                ("HiDream", "dev", "HiDream I1 Dev"),
                ("HiDream", "fast", "HiDream I1 Fast"),
                ("HiDreamO1", "dev", "HiDream O1 Dev"),
                ("QwenImage", "edit", "Qwen Image Edit"),
                ("MageFlow", "turbo", "Mage Flow Turbo"),
            ):
                for dtype in ("F16", "F8_E4M3"):
                    model = write_header(
                        root / "misleading_full.safetensors",
                        {"unknown.weight": [1]},
                        dtype,
                        {"variant": declaration},
                    )
                    with self.subTest(family=family, declaration=declaration, dtype=dtype):
                        values = perf.get_perf_options("Model Defaults", str(model), family)
                        expected = perf.performance_options[preset]
                        self.assertEqual(
                            (values["custom_steps"], values["cfg"]),
                            (expected["custom_steps"], expected["cfg"]),
                        )
            for family in ("HiDream", "HiDreamO1"):
                model = write_header(
                    root / f"fast_dev_{family}.safetensors", {"unknown.weight": [1]}
                )
                with self.assertRaises(UnavailableModelRecipe):
                    perf.get_perf_options("Model Defaults", str(model), family)
                cache = root / (model.stem + ".json")
                cache.write_text(
                    json.dumps(
                        dict(
                            baseModel=family,
                            variant="dev",
                            _rf_file_signature=file_signature(model),
                        )
                    )
                )
                self.assertIn("Dev", perf.family_preset(family, str(model)))
                write_header(model, {"unknown.weight": [1]}, metadata={"variant": "full"})
                self.assertNotIn("Dev", perf.family_preset(family, str(model)))
                cache.write_text(
                    json.dumps(
                        dict(
                            baseModel=family,
                            variant="dev",
                            _rf_file_signature=file_signature(model),
                        )
                    )
                )
                with self.assertRaises(UnavailableModelRecipe):
                    perf.get_perf_options("Model Defaults", str(model), family)
                perf.model_recommendations.save_override(
                    str(model),
                    dict(
                        custom_steps=7, cfg=0, sampler_name="euler", scheduler="simple", clip_skip=1
                    ),
                    SAMPLERS,
                    SCHEDULERS,
                )
                self.assertEqual(
                    perf.get_perf_options("Model Defaults", str(model), family)["cfg"], 0
                )

    def test_exact_publisher_titles_qualify_edit_turbo_but_local_filenames_do_not(self):
        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            perf = recommendations.RecommendationTests()._performance(root)
            perf.model_recommendations.offline = True
            for family in ("QwenImage", "MageFlow"):
                model = write_header(
                    root / f"edit_turbo_{family}.safetensors", {"unknown.weight": [1]}
                )
                with self.assertRaises(UnavailableModelRecipe):
                    perf.get_perf_options("Model Defaults", str(model), family)
                title = (
                    "Qwen-Image-Edit FP8 & GGUF"
                    if family == "QwenImage"
                    else "Mage-Flow-Edit-4B-Turbo"
                )
                (root / (model.stem + ".json")).write_text(
                    json.dumps(
                        dict(
                            baseModel=family,
                            id=42,
                            name=title,
                            _rf_file_signature=file_signature(model),
                        )
                    )
                )
                expected = "Qwen Image Edit" if family == "QwenImage" else "Mage Flow Turbo"
                self.assertEqual(perf.family_preset(family, str(model)), expected)

    def test_publisher_resolution_refresh_is_bounded_offline_and_signature_current(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = write_header(root / "model.safetensors", {"unknown.weight": [1]})
            (root / "model.json").write_text(
                json.dumps(
                    dict(id=42, baseModel="SDXL 1.0", _rf_file_signature=file_signature(model))
                )
            )
            store = recommendations.RecommendationTests()._performance(root).model_recommendations
            remote = dict(id=42, generationConfig=dict(steps=30, width=832, height=1216))
            with patch(
                "ruinedfooocus.features.models.model_sources.civitai_metadata", return_value=remote
            ):
                store.refresh(str(model), SAMPLERS, SCHEDULERS)
            self.assertEqual(
                store.cached_resolution(str(model)), ((832, 1216), "Civitai version 42")
            )
            with patch(
                "ruinedfooocus.features.models.model_sources.civitai_metadata", return_value=None
            ):
                store.refresh(str(model), SAMPLERS, SCHEDULERS, force=True)
            self.assertEqual(store.cached_resolution(str(model))[0], (832, 1216))
            with patch(
                "ruinedfooocus.features.models.model_sources.civitai_metadata",
                return_value=dict(id=42),
            ):
                store.refresh(str(model), SAMPLERS, SCHEDULERS, force=True)
            self.assertEqual(store.cached_resolution(str(model)), (None, "Civitai version 42"))
            with patch(
                "ruinedfooocus.features.models.model_sources.civitai_metadata", return_value=remote
            ):
                store.refresh(str(model), SAMPLERS, SCHEDULERS, force=True)
            store.offline = True
            store.refresh(str(model), SAMPLERS, SCHEDULERS, force=True)
            self.assertEqual(store.cached_resolution(str(model))[0], (832, 1216))
            model.write_bytes(model.read_bytes() + b"replacement")
            self.assertEqual(store.cached_resolution(str(model)), (None, None))
            for signature in (None, "malformed", [str(model)]):
                (root / "model.json").write_text(json.dumps(dict(id=42, baseModel="SDXL 1.0")))
                data = store.load()
                entry = data[store.identity(str(model))]
                entry["resolution_signature"] = signature
                store.path.write_text(json.dumps(data))
                self.assertEqual(store.cached_resolution(str(model)), (None, None))
            for size in ((True, 1024), (255, 1024), (1024, 4098), (1023, 1024), ("1024", 1024)):
                self.assertIsNone(
                    published_resolution({"generationConfig": dict(zip(("width", "height"), size))})
                )

    def test_unindexed_native_hidream_imports_cannot_bypass_variant_guard(self):
        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            perf = recommendations.RecommendationTests()._performance(root)
            perf.model_recommendations.offline = True
            for shapes in (
                {"caption_projection.0.linear.weight": [1]},
                {"t_embedder1.mlp.0.weight": [1], "x_embedder.proj1.weight": [1]},
            ):
                model = write_header(root / "renamed.safetensors", shapes)
                with self.assertRaises(UnavailableModelRecipe):
                    perf.get_perf_options("Model Defaults", str(model))
                write_header(model, shapes, metadata={"variant": "dev"})
                self.assertEqual(
                    perf.get_perf_options("Model Defaults", str(model))["custom_steps"], 28
                )

    def test_actual_gguf_keeps_declared_behavioral_variant_across_quantization(self):
        import gguf

        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            perf = recommendations.RecommendationTests()._performance(root)
            perf.model_recommendations.offline = True
            for declaration, expected in (("dev", "HiDream I1 Dev"), ("fast", "HiDream I1 Fast")):
                model = root / "misleading_full.gguf"
                writer = gguf.GGUFWriter(str(model), "hidream")
                writer.add_string("modelspec.variant", declaration)
                quant = gguf.GGMLQuantizationType.Q8_0
                writer.add_tensor(
                    "unknown.weight",
                    gguf.quantize(np.zeros(32, dtype=np.float32), quant),
                    raw_dtype=quant,
                )
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                self.assertEqual(perf.family_preset("HiDream", str(model)), expected)
                model.unlink()


if __name__ == "__main__":
    unittest.main()
