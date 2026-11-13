"""Native configuration recipes survive storage changes without variant guessing."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.core.cache_contract import file_identity
from ruinedfooocus.core.preparation import PreparationRequest
from ruinedfooocus.features.models.model_identity import (
    file_signature,
    record_prepared_native_source,
)
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.features.models.native_recipes import compatible_native_recipe
from ruinedfooocus.features.models.resource_defaults import load_family_recipe
from ruinedfooocus.features.models.performance import UnavailableModelRecipe
import test_model_recommendations as recommendations
from test_native_variant_defaults import native_backend, write_header


def boogu_shapes(layers=32):
    return {
        "x_embedder.weight": [3360, 32],
        "time_caption_embed.caption_embedder.0.weight": [4096, 32],
        **{
            f"double_stream_layers.{n}.img_instruct_attn.processor.img_to_q.weight": [32]
            for n in range(8)
        },
        **{f"single_stream_layers.{n}.weight": [32] for n in range(layers)},
        **{f"noise_refiner.{n}.weight": [32] for n in range(2)},
    }


class NativeRecipeTests(unittest.TestCase):
    def test_real_performance_caller_uses_compatible_variant_and_preserves_saved_overrides(self):
        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            performance = recommendations.RecommendationTests()._performance(root)
            performance.model_recommendations.offline = True
            samplers = ["euler", "lcm", "dpmpp_2m"]
            schedulers = ["simple", "normal", "sgm_uniform"]
            with patch.dict(
                "sys.modules",
                {
                    "comfy.samplers": SimpleNamespace(
                        KSampler=SimpleNamespace(SAMPLERS=samplers, SCHEDULERS=schedulers)
                    )
                },
            ):
                for variant, steps in (("edit", 25), ("turbo", 4)):
                    path = write_header(
                        root / (variant + ".safetensors"),
                        boogu_shapes(),
                        "F8_E4M3",
                        {"variant": variant},
                    )
                    values = performance.get_perf_options("Model Defaults", str(path), "Boogu")
                    self.assertEqual(values["custom_steps"], steps)
                    self.assertEqual(
                        performance.family_preset("Boogu", str(path)), "native-boogu-" + variant
                    )
                    performance.model_recommendations.save_override(
                        str(path),
                        dict(
                            custom_steps=19,
                            cfg=2,
                            sampler_name="euler",
                            scheduler="simple",
                            clip_skip=1,
                        ),
                        samplers,
                        schedulers,
                    )
                    values = performance.get_perf_options("Model Defaults", str(path), "Boogu")
                    self.assertEqual((values["custom_steps"], values["cfg"]), (19, 2))

    def test_real_native_header_keeps_base_turbo_and_quantization_independent(self):
        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            for dtype in ("F16", "BF16", "F8_E4M3"):
                for variant, steps in (("edit", 25), ("turbo", 4)):
                    with self.subTest(dtype=dtype, variant=variant):
                        path = write_header(
                            root / "renamed.safetensors",
                            boogu_shapes(),
                            dtype,
                            {"variant": variant},
                        )
                        with patch(
                            "ruinedfooocus.features.models.model_identity.cached_file_sha256",
                            side_effect=AssertionError("no render hashing"),
                        ):
                            recipe = compatible_native_recipe(path, {})
                        self.assertEqual(recipe["sampling"]["custom_steps"], steps)
                        self.assertEqual(recipe["default_size"], [1024, 1024])
                        self.assertTrue(recipe["auto_recipe_supported"])
                        self.assertFalse(recipe["artifact_qualified"])

    def test_logical_gguf_header_uses_current_variant_not_storage_or_filename(self):
        import gguf
        import numpy as np

        with tempfile.TemporaryDirectory() as folder, native_backend():
            path = Path(folder) / "different-family-name.gguf"
            writer = gguf.GGUFWriter(str(path), "boogu")
            writer.add_string("modelspec.variant", "turbo")
            for name, shape in boogu_shapes().items():
                writer.add_tensor(
                    name,
                    gguf.quantize(
                        np.zeros(shape, dtype=np.float32), gguf.GGMLQuantizationType.Q8_0
                    ),
                    raw_dtype=gguf.GGMLQuantizationType.Q8_0,
                )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            self.assertEqual(compatible_native_recipe(path, {})["sampling"]["custom_steps"], 4)
            path.unlink()

    def test_ambiguous_conflicting_wrong_config_stale_and_runtime_uninitialized_are_misses(self):
        with tempfile.TemporaryDirectory() as folder, native_backend():
            path = Path(folder) / "boogu-turbo-Q8.safetensors"
            write_header(path, boogu_shapes())
            self.assertEqual(compatible_native_recipe(path, {}), {})
            write_header(path, boogu_shapes(31), metadata={"variant": "turbo"})
            self.assertEqual(compatible_native_recipe(path, {}), {})
            write_header(path, boogu_shapes(), metadata={"variant": "turbo"})
            metadata = dict(variant="edit", _rf_file_signature=file_signature(path))
            self.assertEqual(compatible_native_recipe(path, metadata), {})
            metadata = dict(variant="turbo", _rf_file_signature=["old", 1, 2, 3])
            self.assertEqual(compatible_native_recipe(path, metadata), {})
        with patch.dict("sys.modules", {"comfy.model_detection": None}):
            # Presence alone is insufficient when an importer explicitly has no backend.
            with patch(
                "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                side_effect=ImportError("uninitialized"),
            ):
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "unrelated.safetensors"
                    path.write_bytes(b"header placeholder")
                    self.assertEqual(compatible_native_recipe(path, {}), {})

    def test_exact_source_receipt_resolves_offline_and_exposes_unimplemented_graph_requirements(
        self,
    ):
        original = load_registry().artifact("existing:diffusion_models/flux1-fill-dev.safetensors")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "renamed.safetensors"
            path.write_bytes(b"small exact source fixture")
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            row = dict(original) | dict(
                artifact=dict(original["artifact"])
                | dict(sha256=checksum, bytes=path.stat().st_size)
            )
            registry = SimpleNamespace(
                entries=lambda **kw: (row,),
                artifact=lambda identity: row if identity == row["id"] else None,
            )
            request = PreparationRequest(
                row["artifact"]["pinned_url"],
                path,
                sha256=checksum,
                expected_size=path.stat().st_size,
            )
            with patch(
                "ruinedfooocus.features.models.model_registry.load_registry", return_value=registry
            ):
                proof = record_prepared_native_source(
                    path, request, file_identity(path), root / "cache"
                )
                with patch(
                    "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                    side_effect=AssertionError("exact proof needs no native runtime"),
                ):
                    recipe = compatible_native_recipe(path, proof)
                performance = recommendations.RecommendationTests()._performance(
                    root / "cache/checkpoints"
                )
                performance.model_recommendations.offline = True
                with patch.dict(
                    "sys.modules",
                    {
                        "comfy.samplers": SimpleNamespace(
                            KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["simple"])
                        )
                    },
                ):
                    with self.assertRaises(UnavailableModelRecipe):
                        performance.get_perf_options("Model Defaults", str(path), "FluxInpaint")
                    store = performance.model_recommendations
                    store.path.parent.mkdir(parents=True, exist_ok=True)
                    store.path.write_text(
                        json.dumps(
                            {
                                store.identity(str(path)): dict(
                                    recommended=dict(
                                        custom_steps=20,
                                        cfg=7,
                                        sampler_name="euler",
                                        scheduler="simple",
                                        clip_skip=1,
                                    )
                                )
                            }
                        ),
                        encoding="utf-8",
                    )
                    with self.assertRaises(UnavailableModelRecipe):
                        performance.get_perf_options("Model Defaults", str(path), "FluxInpaint")
            self.assertEqual(recipe["identity_basis"], "verified_publisher_artifact")
            self.assertFalse(recipe["artifact_qualified"])
            self.assertFalse(recipe["auto_recipe_supported"])
            self.assertIn(
                "InpaintModelConditioning", recipe["applicability"]["runtime_requirements"]
            )


if __name__ == "__main__":
    unittest.main()
