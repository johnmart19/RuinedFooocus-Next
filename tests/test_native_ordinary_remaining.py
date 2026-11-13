"""Remaining ordinary image routes preserve native components and paired callers."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.features.models.image_components import IMAGE_COMPONENTS
from ruinedfooocus.features.models.component_candidates import classify_shapes, compatibility
import test_model_cache as cache
from test_model_cache import method, Clip
from test_native_model_coverage import route_info, PIPELINE
from test_qwen_image21_conditioning import runtime, clip_fixture
from test_native_variant_defaults import native_backend, write_header
import test_model_recommendations as recommendations
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.features.models.performance import UnavailableModelRecipe
import tempfile


class RemainingOrdinaryTests(unittest.TestCase):
    def test_exact_qwen_base_source_recipe_does_not_guess_turbo_schedule(self):
        registry = load_registry()
        row = registry.artifact("existing:checkpoints/qwen_image_2.1_int8_convrot.safetensors")
        from ruinedfooocus.features.models.model_identity import is_native_checkpoint_source

        self.assertTrue(is_native_checkpoint_source(row))
        self.assertEqual(row["quantization"]["label"], "INT8 ConvRot")
        self.assertEqual((row["native_family"], row["behavioral_variant"]), ("QwenImage21", "base"))
        self.assertEqual(len(row["companions"]["text"]), 2)
        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            perf = recommendations.RecommendationTests()._performance(root)
            perf.model_recommendations.offline = True
            model = write_header(
                root / "renamed.safetensors", {"unknown.weight": [1]}, metadata={"variant": "base"}
            )
            with self.assertRaises(UnavailableModelRecipe):
                perf.get_perf_options("Model Defaults", str(model), "QwenImage21")
            for variant in ("turbo", ""):
                write_header(model, {"unknown.weight": [1]}, metadata={"variant": variant})
                with self.assertRaises(UnavailableModelRecipe):
                    perf.get_perf_options("Model Defaults", str(model), "QwenImage21")
            # No declaration in these weights: the current verified INT8 source
            # must inherit its base behavior, without guessing from the filename.
            from ruinedfooocus.features.models.model_identity import (
                file_signature,
                verified_native_source,
            )

            signature = file_signature(model)
            signature[1] = row["artifact"]["bytes"]
            metadata = dict(
                baseModel="QwenImage21",
                _rf_file_signature=signature,
                _rf_local_sha256=row["artifact"]["sha256"],
                _rf_verified_native_source=dict(
                    artifact=row["id"], family="QwenImage21", sha256=row["artifact"]["sha256"]
                ),
            )
            with (
                patch(
                    "ruinedfooocus.features.models.model_identity.file_signature",
                    return_value=signature,
                ),
                patch.object(perf.model_recommendations, "metadata", return_value=metadata),
            ):
                self.assertEqual(
                    verified_native_source(model, metadata)["behavioral_variant"], "base"
                )
                self.assertEqual(perf.family_preset("QwenImage21", str(model)), "Qwen Image 2.1")
                inherited = perf.get_perf_options("Model Defaults", str(model), "QwenImage21")
                self.assertEqual(
                    tuple(
                        inherited[key]
                        for key in ("custom_steps", "cfg", "sampler_name", "scheduler")
                    ),
                    (25, 1, "euler", "simple"),
                )
                with patch(
                    "ruinedfooocus.features.models.model_identity.file_signature",
                    return_value=signature[:2] + [0, 0],
                ):
                    self.assertIsNone(verified_native_source(model, metadata))

    def test_distinct_64_channel_codecs_reject_wrong_geometry_and_rgba(self):
        shapes = {
            "decoder.middle.0.residual.0.gamma": (1152, 1, 1, 1),
            "decoder.upsamples.0.upsamples.0.residual.2.weight": (1152, 1152, 1, 3, 3),
            "encoder.conv1.weight": (96, 4, 1, 3, 3),
            "encoder.head.2.weight": (128, 768, 1, 3, 3),
            "decoder.head.0.gamma": (144, 1, 1, 1),
            "decoder.conv1.weight": (1152, 64, 1, 3, 3),
            "decoder.head.2.weight": (4, 144, 1, 3, 3),
        }
        for index, channels in enumerate((96, 192, 384, 768)):
            shapes[f"encoder.downsamples.{index}.downsamples.2.resample.1.weight"] = (
                channels,
                channels,
                3,
                3,
            )
        for index, channels in enumerate((1152, 1152, 576, 288)):
            shapes[f"decoder.upsamples.{index}.upsamples.3.resample.1.weight"] = (
                channels,
                channels,
                3,
                3,
            )
        classify = lambda value: classify_shapes(value, detector=lambda _: None)
        self.assertEqual(compatibility("vae_qwen_image21", classify(shapes)), "compatible")
        self.assertEqual(compatibility("vae_hunyuan_image21", classify(shapes)), "incompatible")
        self.assertEqual(compatibility("vae_qwen_image", classify(shapes)), "incompatible")
        for key, shape in (
            ("decoder.head.2.weight", (3, 144, 1, 3, 3)),
            ("encoder.downsamples.3.downsamples.2.resample.1.weight", (1, 1, 3, 3)),
        ):
            self.assertEqual(
                compatibility("vae_qwen_image21", classify(shapes | {key: shape})), "incompatible"
            )
        hunyuan = {
            "decoder.conv_in.weight": (1024, 64, 3, 3),
            "encoder.conv_in.weight": (128, 3, 3, 3),
            "encoder.conv_out.weight": (128, 1024, 3, 3),
            "decoder.conv_out.weight": (3, 128, 3, 3),
        }
        for i, (inp, out) in enumerate(
            ((128, 64), (256, 128), (512, 128), (512, 256), (1024, 256))
        ):
            hunyuan[f"encoder.down.{i}.downsample.conv.weight"] = (out, inp, 3, 3)
        for i, (inp, out) in enumerate(
            ((1024, 4096), (1024, 2048), (512, 2048), (512, 1024), (256, 512))
        ):
            hunyuan[f"decoder.up.{i}.upsample.conv.weight"] = (out, inp, 3, 3)
        self.assertEqual(compatibility("vae_hunyuan_image21", classify(hunyuan)), "compatible")
        self.assertEqual(compatibility("vae_qwen_image21", classify(hunyuan)), "incompatible")
        del hunyuan["encoder.down.4.downsample.conv.weight"]
        self.assertEqual(compatibility("vae_hunyuan_image21", classify(hunyuan)), "incompatible")

    def test_native_routes_have_exact_component_roles(self):
        for family, clip_type, keys, vae in (
            ("SenseNovaU15", "STABLE_DIFFUSION", [], "pixel_space"),
            (
                "HunyuanImage21",
                "HUNYUAN_IMAGE",
                ["clip_qwen25", "clip_hunyuan_byt5"],
                "vae_hunyuan_image21",
            ),
            ("QwenImage21", "QWEN_IMAGE", ["clip_qwen_image21"], "vae_qwen_image21"),
        ):
            self.assertEqual(route_info()[family]["clip_type"], clip_type)
            self.assertEqual(IMAGE_COMPONENTS[family], dict(clip_names=keys, vae_name=vae))

    def test_qwen_pipeline_pair_preserves_native_flags_and_request_clip_layer(self):
        modules, _ = runtime()
        clip, _ = clip_fixture()
        patched, _ = clip_fixture()
        layer = Mock(return_value=[patched])
        pair = method(
            PIPELINE,
            "textencode_pair",
            dict(
                path_manager=NS(model_paths={"embedding_path": [], "cache_path": Path("unused")}),
                CLIPSetLastLayer=lambda: NS(set_last_layer=layer),
            ),
        )
        owner = NS(xl_base_patched=NS(clip=clip))
        with (
            patch.dict("sys.modules", modules),
            patch(
                "ruinedfooocus.features.models.embedding_browser.native_embedding_text",
                side_effect=lambda text, *args: text,
            ),
            patch(
                "ruinedfooocus.features.image_editing.qwen_image21_conditioning.warm_patched_text_encoder"
            ),
        ):
            positive, negative, latent = pair(owner, "cat", "", 2, 96, 64)
        layer.assert_called_once_with(clip, -2)
        self.assertEqual((positive[0][0], negative[0][0]), ("cat", ""))
        self.assertEqual(tuple(latent["samples"].shape), (1, 64, 4, 6))
        clip.tokenize.assert_not_called()
        self.assertTrue(
            all(
                call.kwargs["prevent_empty_text"] and call.kwargs["keep_vision"]
                for call in patched.tokenize.call_args_list
            )
        )

    def test_pipeline_scheduled_qwen_prompts_use_pair_not_generic_encoder(self):
        tree = ast.parse(Path(PIPELINE).read_text(encoding="utf-8"))
        # Execute the real ordinary text branch, including its per-step schedule.
        process = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process"
        )
        outer = next(
            n
            for n in ast.walk(process)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "image_edit and input_images > 0"
        )
        owner = NS(
            conditions=None,
            textencode=Mock(side_effect=AssertionError("generic encoder")),
            textencode_pair=Mock(
                side_effect=lambda pos, neg, *a: ([[pos, {}]], [[neg, {}]], {"samples": "latent"})
            ),
        )
        scope = dict(
            self=owner,
            callback=None,
            architecture="QwenImage21",
            qwen_references=None,
            positive_prompt="cat",
            negative_prompt="",
            clip_skip=1,
            gen_data=dict(width=96, height=64, steps=3),
            clean_prompt_cond_caches=lambda: {
                key: {"text": None, "cache": None} for key in ("+", "-", "switch")
            },
            pp=NS(prompt_switch_per_step=lambda *a: ["cat", "dog", "cat"]),
            conditioning_set_values=lambda cond, values: [[x, meta | values] for x, meta in cond],
            switched_prompt=[],
        )
        exec(compile(ast.Module(body=outer.orelse, type_ignores=[]), PIPELINE, "exec"), scope)
        self.assertEqual(
            [call.args[0] for call in owner.textencode_pair.call_args_list], ["cat", "cat", "dog"]
        )
        self.assertEqual(len(scope["switched_prompt"]), 3)
        self.assertEqual(scope["edit_latent"], {"samples": "latent"})

    def test_split_sensenova_constructs_native_sentinel_under_checkpoint_lease(self):
        fixture = cache.ModelCacheTests("test_component_choices_reject_malformed_or_unknown_slots")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        checkpoint = fixture.file("checkpoint.safetensors")
        load, owner, diffusion, backend, _ = fixture.base_loader(checkpoint, bare=True)
        model = type("SenseNovaU15", (), {})()
        model.model_config = type("SenseNovaU15", (), {})()
        diffusion.return_value = NS(model=model)
        owner.known_models = {"SenseNovaU15"}
        owner.get_clip_and_vae.return_value = dict(
            clip_names=[], clip_keys=[], vae_key="pixel_space", vae_name="pixel_space"
        )
        held = []

        @contextmanager
        def lease(path, **kwargs):
            held.append(path)
            try:
                yield path
            finally:
                held.pop()

        constructor = Mock(
            side_effect=lambda *args: Clip() if held else self.fail("missing source lease")
        )
        with (
            patch(
                "ruinedfooocus.core.preparation.preparation_service.artifact_lease",
                side_effect=lease,
            ),
            patch("ruinedfooocus.backends.comfy.sensenova_clip.load_sensenova_clip", constructor),
        ):
            load()
        self.assertIsNotNone(owner.xl_base)
        self.assertEqual(constructor.call_args.args[1][0], str(checkpoint.resolve()))
        backend.sd.load_clip.assert_not_called()
        backend.utils.load_torch_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
