"""Pinned native SD4x geometry and required-source helpers without model loading."""

import ast
import os
from pathlib import Path
import sys
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import numpy as np
from PIL import Image

from ruinedfooocus.features.image_editing.sd4x_conditioning import (
    is_sd4x,
    require_sd4x_source,
    require_sd4x_codec,
    sd4x_pixels,
    restore_sd4x_alpha,
    sd4x_conditioning,
)
from ruinedfooocus.features.models.component_candidates import classify_shapes, compatibility
from ruinedfooocus.features.models.image_components import inventory
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.features.generation.image_preparation import capture_image_assets
from ruinedfooocus.features.generation.automatic_upscale import describe_size
from test_model_cache import method
from test_native_model_coverage import route_info
from test_powerup_family_controls import functions
from test_api_image_edits import queued_data
from ruinedfooocus.features.models.performance import UnavailableModelRecipe
import test_model_recommendations as recommendations

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))


def native_node(upscale):
    source = NATIVE / "comfy_extras/nodes_sdupscale.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SD_4XUpscale_Conditioning"
    )
    node.bases = []
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "execute"]
    scope = dict(
        torch=torch,
        comfy=SimpleNamespace(utils=SimpleNamespace(common_upscale=upscale)),
        io=SimpleNamespace(NodeOutput=lambda *values: values),
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return SimpleNamespace(SD_4XUpscale_Conditioning=scope[node.name])


class SD4xConditioningTests(unittest.TestCase):
    def test_primary_output_size_uses_native_source_then_existing_optional_upscale_math(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8"))
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "update_upscale_size"
        )
        features = Mock(return_value={"sd4x": True})
        describe = Mock(wraps=describe_size)
        scope = dict(
            ui_controlnet=SimpleNamespace(
                checkpoint_features=features,
                tool_kind=lambda selection, control_type: (
                    "rembg" if selection == "Remove background" else None
                ),
            ),
            describe_size=describe,
            resolution_settings=SimpleNamespace(aspect_ratios={"Generic": (1024, 1024)}),
        )
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "primary_output_size", "exec"), scope
        )
        source = Image.new("RGB", (117, 101))
        callback = scope["update_upscale_size"]
        for scale, width, height in (
            ("Off", 468, 404),
            ("2× General (RealESRGAN)", 936, 808),
            ("4× General (RealESRGAN)", 1872, 1616),
        ):
            result = callback(
                scale, "Generated image", "Generic", 1024, 1024, None, "selected", source
            )
            self.assertIn(f"{width} × {height} pixels", result)
            self.assertEqual(describe.call_args.args[1:4], (None, 468, 404))
        self.assertIn(
            "Add a source image",
            callback("Off", "Generated image", "Generic", 1024, 1024, None, "selected", None),
        )
        features.return_value = {"sd4x": False}
        self.assertEqual(
            callback("Off", "Generated image", "Generic", 128, 128, None, "other", source),
            "1024 × 1024 pixels",
        )
        features.return_value = {"sd4x": True}
        self.assertIn(
            "234 × 202 pixels",
            callback(
                "2× General (RealESRGAN)",
                "Input image only",
                "Generic",
                1024,
                1024,
                source,
                "selected",
                None,
            ),
        )
        self.assertIn(
            "utility",
            callback(
                "2× General (RealESRGAN)",
                "Generated image",
                "Generic",
                1024,
                1024,
                None,
                "selected",
                source,
                "Remove background",
                "Rembg",
            ),
        )

    def test_native_model_owns_latent_scaling_without_app_normalization(self):
        source = NATIVE / "comfy/latent_formats.py"
        node = next(
            n
            for n in ast.parse(source.read_text(encoding="utf-8")).body
            if isinstance(n, ast.ClassDef) and n.name == "SD_X4"
        )
        node.bases = []
        scope = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        self.assertEqual(scope["SD_X4"]().scale_factor, 0.08333)
        config = next(
            n
            for n in ast.parse(
                (NATIVE / "comfy/supported_models.py").read_text(encoding="utf-8")
            ).body
            if isinstance(n, ast.ClassDef) and n.name == "SD_X4Upscaler"
        )
        latent = next(
            n.value
            for n in config.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "latent_format" for t in n.targets)
        )
        self.assertEqual(ast.unparse(latent), "latent_formats.SD_X4")

    def test_output_size_message_control_state_and_explicit_target_refusal(self):
        scope, models = functions()
        models.detect_checkpoint_base.return_value = "SD_X4Upscaler"
        source = Image.new("RGB", (117, 101))
        info = scope["sd4x_output_info"](source, "selected")
        self.assertTrue(info["visible"])
        self.assertIn("468", info["value"])
        self.assertIn("404", info["value"])
        self.assertIn("before optional post-upscaling", info["value"])
        self.assertFalse(
            scope["sd4x_output_info"](source, "selected", "Remove background", "Rembg")["visible"]
        )
        self.assertFalse(
            scope["sd4x_output_info"](
                source, "selected", None, None, "2× General (RealESRGAN)", "Input image only"
            )["visible"]
        )
        body = ast.parse(
            (ROOT / "ruinedfooocus/features/image_editing/ui.py").read_text(encoding="utf-8")
        )
        callback = next(
            n
            for n in ast.walk(body)
            if isinstance(n, ast.FunctionDef) and n.name == "sd4x_size_changed"
        )
        scope["size_controls"] = [object(), object(), object()]
        exec(compile(ast.Module(body=[callback], type_ignores=[]), "size_ui", "exec"), scope)
        self.assertTrue(
            all(
                row == {"interactive": False}
                for row in scope["sd4x_size_changed"](source, "selected")[1:]
            )
        )
        models.detect_checkpoint_base.return_value = "SDXL 1.0"
        restored = scope["sd4x_size_changed"](source, "selected")
        self.assertFalse(restored[0]["visible"])
        self.assertTrue(all(row == {"interactive": True} for row in restored[1:]))
        for marker in ({"_api_job": object()}, {"task_type": "tool_call"}):
            self.assertEqual(
                tuple(sd4x_pixels(source, dict(marker, width=1024, height=1024)).shape),
                (1, 101, 117, 3),
            )
        for marker in ({"_sd4x_requested_size": (1024, 1024)}, {"_metadata_prompt": True}):
            with self.assertRaisesRegex(ValueError, "SD4x output must be 468×404"):
                sd4x_pixels(source, dict(marker, width=1024, height=1024))
        self.assertEqual(
            tuple(sd4x_pixels(source, {"_sd4x_requested_size": (468, 404)}).shape), (1, 101, 117, 3)
        )
        source.getexif()[274] = 6
        self.assertEqual(tuple(sd4x_pixels(source, {}).shape), (1, 117, 101, 3))

    def test_official_component_identities_and_auto_remain_data_only_and_strict(self):
        registry = load_registry()
        from ruinedfooocus.core.paths import resource_path

        downloads = json.loads(
            resource_path("model_catalogue", "native_image_components.json").read_text(
                encoding="utf-8"
            )
        )
        for key, checksum in (
            (
                "clip/sd4x_clip_h.safetensors",
                "cce6febb0b6d876ee5eb24af35e27e764eb4f9b1d0b7c026c8c3333d4cfc916c",
            ),
            (
                "vae/sd4x_vae.safetensors",
                "33478c297ec29218100f8ee86007b3ab4c2701896d5ca5c9e3a84fc29f678183",
            ),
        ):
            annotation = registry.annotations_for(key, downloads[key])
            self.assertIn(
                "/resolve/572c99286543a273bfd17fac263db5a77be12c4c/", downloads[key]["url"]
            )
            self.assertEqual(annotation["artifact"]["sha256"], checksum)
            self.assertEqual(annotation["artifact"]["availability"], "publisher_metadata_only")
            self.assertFalse(annotation["chat_weight"])
        self.assertEqual(
            queued_data("SD_X4Upscaler", payload_options={"size": "auto"})["controlnet"]["type"],
            "edit",
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                sys.modules,
                {
                    "comfy.samplers": SimpleNamespace(
                        KSampler=SimpleNamespace(
                            SAMPLERS=recommendations.SAMPLERS, SCHEDULERS=recommendations.SCHEDULERS
                        )
                    )
                },
            ),
        ):
            root = Path(directory)
            (root / "source.json").write_text(
                json.dumps({"baseModel": "SD_X4Upscaler"}), encoding="utf-8"
            )
            performance = recommendations.RecommendationTests()._performance(root)
            performance.model_recommendations.offline = True
            with self.assertRaises(UnavailableModelRecipe):
                performance.get_perf_options("Model Defaults", "source.safetensors")
            request = dict(
                performance_selection="Own settings",
                base_model_name="source.safetensors",
                cfg=3.25,
                custom_steps=7,
            )
            self.assertEqual(performance.apply(request, "SD_X4Upscaler"), request)

    def test_transparency_is_neutrally_conditioned_and_preserved_separately_from_generated_rgb(
        self,
    ):
        source = Image.new("RGBA", (2, 3), (255, 0, 0, 0))
        source.putpixel((1, 0), (20, 40, 60, 255))
        source.info["owner"] = "original upload"
        before = source.tobytes(), dict(source.info)
        pixels = sd4x_pixels(source, {})
        torch.testing.assert_close(pixels[0, 1, 0], torch.full((3,), 128 / 255))
        decoded = torch.rand(1, 12, 8, 3)
        result = restore_sd4x_alpha(decoded, source)
        self.assertEqual(tuple(result.shape), (1, 12, 8, 4))
        torch.testing.assert_close(result[..., :3], decoded)
        self.assertEqual(result[0, -1, 0, 3].item(), 0)
        self.assertEqual(result[0, 0, -1, 3].item(), 1)
        self.assertTrue(((result[..., 3] > 0) & (result[..., 3] < 1)).any())
        self.assertEqual((source.tobytes(), source.info), before)
        opaque = Image.new("RGBA", (2, 3), (20, 40, 60, 255))
        self.assertTrue((restore_sd4x_alpha(decoded, opaque)[..., 3] == 1).all())
        self.assertIs(restore_sd4x_alpha(decoded, opaque.convert("RGB")), decoded)
        source.getexif()[274] = 6
        self.assertEqual(
            tuple(restore_sd4x_alpha(torch.zeros(1, 8, 12, 3), source).shape), (1, 8, 12, 4)
        )
        self.assertEqual(source.getexif()[274], 6)

    def test_inventory_and_native_runtime_keep_dedicated_clip_h_and_four_times_vae(self):
        get_info = method(
            "ruinedfooocus/features/generation/image_pipeline.py",
            "get_clip_and_vae",
            dict(settings=SimpleNamespace(default_settings={})),
        )
        native = type("SD_X4Upscaler", (), {"model_config": SimpleNamespace(unet_config={})})()
        info = get_info(
            SimpleNamespace(known_model_info=route_info()), SimpleNamespace(model=native)
        )
        self.assertEqual(info["clip_type"], "STABLE_DIFFUSION")
        self.assertEqual(info["clip_keys"], ["clip_sd4x_h"])
        self.assertEqual(info["vae_key"], "vae_sd4x")
        self.assertEqual(compatibility("clip_sd4x_h", dict(text_type="CLIP_H")), "compatible")
        self.assertEqual(compatibility("clip_sd4x_h", dict(text_type="CLIP_L")), "incompatible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "source.safetensors"
            raw = json.dumps({"weight": {"shape": [1]}}).encode()
            checkpoint.write_bytes(len(raw).to_bytes(8, "little") + raw)
            paths = SimpleNamespace(
                paths={"path_clip": [str(root)], "path_vae": str(root)},
                get_abspath=Path,
                DOWNLOADABLE_FILES={},
            )
            rows = inventory(checkpoint, "SD_X4Upscaler", {}, paths)["rows"]
            self.assertEqual([row["key"] for row in rows], ["clip_sd4x_h", "vae_sd4x"])

    def test_four_times_vae_contract_rejects_ordinary_eightfold_codec(self):
        shapes = {
            "encoder.conv_in.weight": (128, 3, 3, 3),
            "decoder.conv_in.weight": (512, 4, 3, 3),
            "decoder.conv_out.weight": (3, 128, 3, 3),
            "encoder.down.1.downsample.conv.weight": (256, 256, 3, 3),
            "decoder.up.2.upsample.conv.weight": (512, 512, 3, 3),
        }
        four = classify_shapes(shapes, detector=lambda _: None)
        self.assertEqual(compatibility("vae_sd4x", four), "compatible")
        self.assertEqual(compatibility("vae_sd", four), "incompatible")
        ordinary = classify_shapes(
            {
                **shapes,
                "encoder.down.2.downsample.conv.weight": (512, 512, 3, 3),
                "decoder.up.3.upsample.conv.weight": (512, 512, 3, 3),
            },
            detector=lambda _: None,
        )
        self.assertEqual(compatibility("vae_sd4x", ordinary), "incompatible")
        require_sd4x_codec(SimpleNamespace(spacial_compression_decode=lambda: 4))
        with self.assertRaisesRegex(ValueError, "four-times VAE codec"):
            require_sd4x_codec(SimpleNamespace(spacial_compression_decode=lambda: 8))

    def test_ui_requires_explicit_diffusion_upscale_source_and_hides_plain_img2img(self):
        scope, models = functions()
        scope["controlnet"].controlnet_models["edit"] = None
        scope["controlnet"].cn_options["Native image edit"] = {"type": "edit"}
        models.detect_checkpoint_base.return_value = "SD_X4Upscaler"
        features = scope["checkpoint_features"]("selected")
        choices = dict(scope["powerup_choices"](**features))
        self.assertEqual(choices["Diffusion 4× upscale"], "Native image edit")
        self.assertNotIn("Img2Img", choices.values())
        self.assertFalse(features["inpaint"])
        self.assertFalse(features["sdxl"])
        self.assertEqual(scope["input_label"](features), "Image to upscale (4×)")
        self.assertIn("four times", scope["image_tool_help"](features))
        self.assertEqual(
            scope["fallback_selection"](features, "Img2Img", False), ("Native image edit", False)
        )

    def test_asset_capture_rejects_missing_source_before_resolving_model_or_components(self):
        models = SimpleNamespace(
            get_model_base=lambda info: "SD_X4Upscaler",
            get_models_by_path=lambda *args, **kwargs: {},
            get_file=Mock(side_effect=AssertionError("Source guard must precede lookup")),
        )
        with self.assertRaisesRegex(ValueError, "requires Diffusion 4× upscale"):
            capture_image_assets(
                {"base_model_name": "source.safetensors"}, object(), {}, models=models
            )
        models.get_file.assert_not_called()

    def test_actual_process_keeps_native_four_times_source_geometry_through_sampling_and_decode(
        self,
    ):
        native_config = type("SD_X4Upscaler", (), {"unet_config": {}})()
        native = type("SD_X4Upscaler", (), {"model_config": native_config})()
        unet = SimpleNamespace(model=native)
        codec = SimpleNamespace(spacial_compression_decode=lambda: 4)
        owner = SimpleNamespace(
            xl_base_patched=SimpleNamespace(unet=unet, vae=codec),
            xl_base=SimpleNamespace(unet=unet),
            known_models=route_info().keys(),
            model_info={},
            conditions={"+": {"cache": [["positive", {}]]}, "-": {"cache": [["negative", {}]]}},
            textencode=Mock(),
            refresh_controlnet=Mock(),
            xl_controlnet=None,
            decode_latent=Mock(return_value=torch.ones(1, 404, 468, 3)),
        )
        memory = SimpleNamespace(get_torch_device=lambda: "cpu")
        sample = Mock(side_effect=lambda **kwargs: kwargs["latent_image"] + 0.5)
        guidance = dict(
            moss_image_edit_format="Disabled",
            moss_image_edit_strength=1,
            vae_tiled=False,
            vae_tile_size=512,
            vae_overlap=64,
        )
        scope = dict(
            torch=torch,
            np=np,
            path_manager=SimpleNamespace(),
            settings=SimpleNamespace(default_settings={}),
            resolve_guidance=lambda *args: guidance,
            request_model=lambda *args, **kwargs: unet,
            effective_guidance=lambda *args: {},
            normalize_reference_images=lambda *args: [],
            control_tools=SimpleNamespace(
                get_settings=lambda data: {"type": "edit"}, controlnet_models={"edit": None}
            ),
            comfy=SimpleNamespace(
                model_management=memory,
                sample=SimpleNamespace(
                    prepare_noise=lambda latent, *args: torch.ones_like(latent), sample=sample
                ),
                utils=SimpleNamespace(
                    PROGRESS_BAR_ENABLED=False,
                    ProgressBar=lambda steps: SimpleNamespace(update_absolute=Mock()),
                ),
            ),
            pp=SimpleNamespace(prompt_switch_per_step=lambda *args: []),
            Flux=type("Flux", (), {}),
            fix_empty_latent_channels=lambda model, latent, *args: latent,
            restore_reference_frame=lambda pixels, frame: pixels,
        )
        source = "ruinedfooocus/features/generation/image_pipeline.py"
        scope["reference_tensor"] = method(source, "reference_tensor", {"torch": torch, "np": np})
        process = method(source, "process", scope)
        request = dict(
            cfg=3.25,
            sampler_name="euler",
            scheduler="simple",
            clip_skip=1,
            seed=77,
            positive_prompt="enhance",
            negative_prompt="blur",
            steps=4,
            width=32,
            height=32,
            sampling_shift=1,
            input_image=Image.new("RGB", (117, 101), (100, 150, 200)),
        )
        with (
            patch.dict(
                sys.modules,
                {"comfy_extras.nodes_sdupscale": native_node(lambda image, *args: image)},
            ),
            patch(
                "ruinedfooocus.features.generation.automatic_upscale.upscale_decoded",
                side_effect=lambda owner, pixels, data: pixels,
            ),
        ):
            output = process(owner, request)
        call = sample.call_args.kwargs
        self.assertEqual(tuple(call["latent_image"].shape), (1, 4, 101, 117))
        self.assertEqual((request["width"], request["height"]), (468, 404))
        self.assertEqual(tuple(call["positive"][0][1]["concat_image"].shape), (1, 3, 101, 117))
        self.assertEqual(output[0].shape, (404, 468, 3))
        self.assertIsNone(call["denoise"])
        self.assertEqual((call["cfg"], call["seed"]), (3.25, 77))

    def test_native_family_requires_source_edit_and_refuses_masked_or_plain_generation(self):
        model = SimpleNamespace(model_config=type("SD_X4Upscaler", (), {})())
        ordinary = SimpleNamespace(model_config=type("SD20", (), {})())
        self.assertTrue(is_sd4x(model))
        self.assertFalse(is_sd4x(ordinary))
        for control_type, source in (("none", None), ("img2img", object()), ("edit", None)):
            with self.assertRaisesRegex(ValueError, "requires Diffusion 4× upscale"):
                require_sd4x_source(model, control_type, source)
        with self.assertRaisesRegex(ValueError, "Masked inpainting"):
            require_sd4x_source(model, "edit", object(), inpainting=True)
        require_sd4x_source(model, "edit", object())
        require_sd4x_source(ordinary, "none", None)

    def test_actual_native_node_preserves_odd_source_geometry_batch_and_condition_metadata(self):
        pixels = torch.linspace(0.0, 1.0, 2 * 101 * 117 * 3).reshape(2, 101, 117, 3)
        positive = [[torch.ones(1), {"pooled_output": "positive", "keep": 1}]]
        negative = [[torch.zeros(1), {"pooled_output": "negative", "keep": 2}]]
        upscale = Mock(side_effect=lambda image, width, height, *_: image)
        module = native_node(upscale)
        with patch.dict(sys.modules, {"comfy_extras.nodes_sdupscale": module}):
            cp, cn, latent = sd4x_conditioning(positive, negative, pixels, 0.125)
        self.assertEqual(tuple(latent["samples"].shape), (2, 4, 101, 117))
        self.assertEqual(torch.count_nonzero(latent["samples"]).item(), 0)
        self.assertEqual(upscale.call_args.args[1:], (117, 101, "bilinear", "center"))
        expected = pixels.movedim(-1, 1) * 2.0 - 1.0
        self.assertTrue(torch.equal(cp[0][1]["concat_image"], expected))
        self.assertIs(cp[0][1]["concat_image"], cn[0][1]["concat_image"])
        self.assertEqual(cp[0][1]["noise_augmentation"], 0.125)
        self.assertEqual(cn[0][1]["noise_augmentation"], 0.125)
        self.assertEqual(cp[0][1]["pooled_output"], "positive")
        self.assertEqual(cn[0][1]["pooled_output"], "negative")
        self.assertNotIn("concat_image", positive[0][1])
        self.assertNotIn("concat_image", negative[0][1])

    def test_noise_range_refuses_before_native_execution(self):
        module = SimpleNamespace(SD_4XUpscale_Conditioning=SimpleNamespace(execute=Mock()))
        with patch.dict(sys.modules, {"comfy_extras.nodes_sdupscale": module}):
            for noise in (-0.01, 1.01, float("nan")):
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    sd4x_conditioning([], [], None, noise)
        module.SD_4XUpscale_Conditioning.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
