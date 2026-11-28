"""Native Auto qualification uses logical tensor configuration, not payload identity."""

import ast
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ruinedfooocus.features.models.model_architecture import inspect_model_file, _inspect_model_file
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.performance import UnavailableModelRecipe
from ruinedfooocus.features.models.resolutions import ResolutionSettings
import test_model_recommendations as recommendations
from test_model_recommendations import SAMPLERS, SCHEDULERS

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))
SHAPES = {
    "QwenImage21": {
        "txt_in.text_norm.weight": [32],
        "modulation.1.weight": [32, 32],
        "transformer_blocks.0.attn.norm_q.weight": [32],
        "img_in.weight": [32, 256],
        "proj_out.weight": [256, 32],
        "transformer_blocks.0.img_mlp.gate_up.weight": [64, 32],
    },
    "Omnigen2": {"time_caption_embed.timestep_embedder.linear_1.bias": [1]},
    "HunyuanDiT": {
        "mlp_t5.0.weight": [8192, 2048],
        "x_embedder.proj.weight": [1408, 4, 2, 2],
        "extra_embedder.0.weight": [5632, 1024],
    },
    "Kandinsky5Image": {
        "visual_transformer_blocks.0.cross_attention.key_norm.weight": [128],
        "visual_embeddings.in_layer.bias": [2560],
        "visual_embeddings.in_layer.weight": [2560, 64],
        "time_embeddings.in_layer.bias": [512],
        "visual_transformer_blocks.0.feed_forward.in_layer.weight": [10240, 2560],
    },
}


def write_header(path, shapes, dtype="F16", metadata=None):
    header = {k: dict(shape=v, dtype=dtype, data_offsets=[0, 0]) for k, v in shapes.items()}
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw)
    return path


@contextmanager
def native_backend():
    """Execute pinned detector and class matcher; replace only runtime construction."""
    tree = ast.parse((NATIVE / "comfy/model_detection.py").read_text(encoding="utf-8"))
    scope = dict(torch=torch, logging=logging)
    for name in (
        "count_blocks",
        "any_suffix_in",
        "detect_unet_config",
        "model_config_from_unet_config",
        "model_config_from_unet",
        "unet_prefix_from_state_dict",
    ):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(NATIVE), "exec"), scope)
    base_tree = ast.parse((NATIVE / "comfy/supported_models_base.py").read_text(encoding="utf-8"))
    base = next(n for n in base_tree.body if isinstance(n, ast.ClassDef) and n.name == "BASE")
    match = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "matches")
    exec(compile(ast.Module(body=[match], type_ignores=[]), str(NATIVE), "exec"), scope)
    classes = []
    supported = ast.parse((NATIVE / "comfy/supported_models.py").read_text(encoding="utf-8"))
    for node in supported.body:
        if not isinstance(node, ast.ClassDef) or node.name not in (
            *SHAPES,
            "HunyuanDiT1",
            "Boogu",
            "Flux3",
            "HiDream",
            "HiDreamO1",
        ):
            continue
        config = next(
            n.value
            for n in node.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "unet_config" for t in n.targets)
        )
        classes.append(
            type(
                node.name,
                (),
                dict(
                    unet_config=ast.literal_eval(config),
                    required_keys={},
                    matches=scope["matches"],
                    __init__=lambda self, config: setattr(self, "unet_config", config),
                ),
            )
        )
    scope["comfy"] = NS(
        supported_models=NS(models=classes), utils=NS(detect_layer_quantization=lambda *a: {})
    )
    detection = NS(**{k: v for k, v in scope.items() if callable(v)})
    samplers = NS(KSampler=NS(SAMPLERS=SAMPLERS, SCHEDULERS=SCHEDULERS + ["sgm_uniform"]))
    _inspect_model_file.cache_clear()
    with patch.dict(
        "sys.modules",
        {
            "comfy": NS(model_detection=detection),
            "comfy.model_detection": detection,
            "comfy.samplers": samplers,
        },
    ):
        yield
    _inspect_model_file.cache_clear()


class NativeVariantDefaultsTests(unittest.TestCase):
    def test_native_config_baseline_survives_quantization_rename_import_and_missing_source(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            native_backend(),
            patch(
                "ruinedfooocus.features.models.model_identity.cached_file_sha256",
                side_effect=AssertionError("Auto must not hash weights"),
            ),
        ):
            root = Path(folder)
            perf = recommendations.RecommendationTests()._performance(root)
            perf.model_recommendations.offline = True
            for family, expected in (
                ("HunyuanDiT", (30, 6)),
                ("Kandinsky5Image", (50, 3.5)),
                ("Omnigen2", (20, 5)),
            ):
                for dtype in ("F16", "BF16", "F8_E4M3"):
                    with self.subTest(family=family, dtype=dtype):
                        model = write_header(
                            root / f"import-{family}-{dtype}.safetensors", SHAPES[family], dtype
                        )
                        options = perf.get_perf_options("Model Defaults", str(model))
                        self.assertEqual((options["custom_steps"], options["cfg"]), expected)
                        self.assertIn("Compatible native", options["_recipe_baseline"])
                        self.assertEqual(inspect_model_file(model)["architecture"], family)
                        self.assertGreater(_inspect_model_file.cache_info().hits, 0)

    def test_actual_gguf_logical_header_qualifies_without_loading_tensor_runtime(self):
        import gguf

        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            model = root / "custom.gguf"
            writer = gguf.GGUFWriter(str(model), "omnigen2")
            quant = gguf.GGMLQuantizationType.Q8_0
            writer.add_tensor(
                next(iter(SHAPES["Omnigen2"])),
                gguf.quantize(np.zeros(32, dtype=np.float32), quant),
                raw_dtype=quant,
            )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            perf = recommendations.RecommendationTests()._performance(root)
            perf.model_recommendations.offline = True
            self.assertEqual(perf.get_perf_options("Model Defaults", str(model))["cfg"], 5)
            contract = inspect_model_file(model)
            contract["native_config"]["image_model"] = "poisoned"
            contract["metadata"]["variant"] = "distilled"
            self.assertEqual(inspect_model_file(model)["native_config"]["image_model"], "omnigen2")
            model.unlink()  # Reader must release the mmap on Windows.

    def test_variant_mismatch_current_declaration_publisher_and_manual_precedence(self):
        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            model = write_header(
                root / "variant.safetensors", SHAPES["Omnigen2"], metadata={"variant": "distilled"}
            )
            perf = recommendations.RecommendationTests()._performance(root)
            perf.model_recommendations.offline = True
            with self.assertRaises(UnavailableModelRecipe):
                perf.get_perf_options("Model Defaults", str(model))
            (root / "variant.json").write_text(
                json.dumps(
                    dict(
                        _rf_file_signature=file_signature(model),
                        baseModel="Omnigen2",
                        generationConfig=dict(
                            steps=4, cfgScale=1, sampler="euler", scheduler="simple"
                        ),
                    )
                )
            )
            self.assertEqual(perf.get_perf_options("Model Defaults", str(model))["custom_steps"], 4)
            perf.model_recommendations.save_override(
                str(model),
                dict(custom_steps=7, cfg=0, sampler_name="euler", scheduler="normal", clip_skip=2),
                SAMPLERS,
                SCHEDULERS,
            )
            self.assertEqual(perf.get_perf_options("Model Defaults", str(model))["cfg"], 0)
            manual = dict(
                performance_selection="Own settings",
                base_model_name=str(model),
                cfg=0,
                custom_steps=3,
                denoise=0.7,
                width=832,
                height=1216,
            )
            self.assertEqual(perf.apply(manual, "Omnigen2"), manual)
            # A structurally distinct native HyDiT1 cannot borrow the1.2 baseline.
            hydit1 = SHAPES["HunyuanDiT"] | {"extra_embedder.0.weight": [5632, 3968]}
            write_header(model, hydit1)
            perf.model_recommendations.reset_override(str(model))
            with self.assertRaises(UnavailableModelRecipe):
                perf.get_perf_options("Model Defaults", str(model), "HunyuanDiT1")

    def test_signature_cache_replacement_and_same_basename_use_current_native_file(self):
        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            model = write_header(root / "model.safetensors", SHAPES["Omnigen2"])
            perf = recommendations.RecommendationTests()._performance(root)
            perf.model_recommendations.offline = True
            for malformed in (None, "invalid", {}, [str(model)]):
                (root / "model.json").write_text(
                    json.dumps(
                        dict(
                            baseModel="Omnigen2",
                            _rf_file_signature=malformed,
                            generationConfig=dict(steps=2, cfgScale=1),
                        )
                    )
                )
                values = perf.get_perf_options("Model Defaults", str(model))
                self.assertEqual((values["custom_steps"], values["cfg"]), (20, 5))
            stale = dict(
                baseModel="Omnigen2",
                _rf_file_signature=file_signature(model),
                variant="distilled",
                generationConfig=dict(steps=2),
            )
            (root / "model.json").write_text(json.dumps(stale))
            with self.assertRaises(UnavailableModelRecipe):
                perf.get_perf_options("Model Defaults", str(model))
            write_header(model, SHAPES["HunyuanDiT"])
            self.assertEqual(perf.get_perf_options("Model Defaults", str(model))["cfg"], 6)
            self.assertEqual(
                perf.get_perf_options("Model Defaults", str(model))["custom_steps"], 30
            )
            with patch.dict(
                "sys.modules", {"ruinedfooocus.app.state": NS(models=NS(get_file=lambda *a: model))}
            ):
                self.assertEqual(perf.get_perf_options("Model Defaults", model.name)["cfg"], 6)
            model.write_bytes(b"invalid replacement")
            with self.assertRaises(UnavailableModelRecipe):
                perf.get_perf_options("Model Defaults", str(model), "Omnigen2")
            # An explicit current classification outside this slice outranks an old hint.
            self.assertEqual(
                perf.get_perf_options("Model Defaults", str(model), "SDXL 1.0")["cfg"],
                perf.performance_options["SDXL"]["cfg"],
            )

    def test_native_image_video_derivative_and_incomplete_config_do_not_share_baseline(self):
        boogu = {
            "double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight": [1],
            "x_embedder.weight": [2520, 64],
            "time_caption_embed.caption_embedder.0.weight": [2048, 2048],
        }
        video = SHAPES["Kandinsky5Image"] | {
            "visual_embeddings.in_layer.bias": [1792],
            "visual_embeddings.in_layer.weight": [1792, 132],
        }
        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            perf = recommendations.RecommendationTests()._performance(root)
            perf.model_recommendations.offline = True
            for shapes, family in (
                (boogu, "Omnigen2"),
                (video, "Kandinsky5Image"),
                ({"unknown.weight": [1]}, "HunyuanDiT"),
            ):
                model = write_header(root / "mismatch.safetensors", shapes)
                with self.subTest(family=family), self.assertRaises(UnavailableModelRecipe):
                    perf.get_perf_options("Model Defaults", str(model), family)
            model = write_header(root / "normal.safetensors", SHAPES["Omnigen2"])
            for declaration in ("distill", "distilled", "turbo", "refiner", "edit"):
                write_header(model, SHAPES["Omnigen2"], metadata={"modelspec.variant": declaration})
                with (
                    self.subTest(declaration=declaration),
                    self.assertRaises(UnavailableModelRecipe),
                ):
                    perf.get_perf_options("Model Defaults", str(model), "Omnigen2")
            write_header(model, SHAPES["Omnigen2"])
            write_header(
                model,
                SHAPES["Omnigen2"],
                metadata={"capabilities": "text-to-image, edit", "operation": "edit"},
            )
            options = perf.get_perf_options("Model Defaults", str(model))
            description = perf.model_recommendations.describe(
                str(model), options, SAMPLERS, SCHEDULERS
            )
            self.assertIn("Compatible native Omnigen2 variant baseline", description)
            self.assertNotIn("exact publisher", description)
            with patch(
                "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                return_value=dict(
                    architecture="Omnigen2",
                    native_class="Omnigen2",
                    native_config=dict(image_model="omnigen2", hidden_size=1024),
                ),
            ):
                with self.assertRaises(UnavailableModelRecipe):
                    perf.get_perf_options("Model Defaults", str(model), "Omnigen2")

    def test_resolution_recommendations_do_not_replace_explicit_manual_size(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(ResolutionSettings, "RESOLUTIONS_FILE", Path(folder) / "resolutions.json"),
        ):
            resolutions = ResolutionSettings()
            for family in SHAPES:
                before = resolutions.recommended_sizes(family)
                resolutions.base_ratios["My manual size"] = (832, 1216)
                resolutions.image_models["My manual size"] = []
                choice = "832x1216 (My manual size)"
                resolutions.aspect_ratios[choice] = (832, 1216)
                self.assertEqual(resolutions.selection_for_model(family, choice), choice)
                self.assertEqual(resolutions.recommended_sizes(family), before)
                self.assertEqual(
                    resolutions.selection_for_model(family, resolutions.CUSTOM_RESOLUTION),
                    resolutions.CUSTOM_RESOLUTION,
                )


if __name__ == "__main__":
    unittest.main()
