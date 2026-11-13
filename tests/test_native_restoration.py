"""Native PiD and Hunyuan refiner callers require their source codec contracts."""

from contextlib import contextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from ruinedfooocus.features.image_editing.hunyuan_refiner import (
    refiner_conditioning,
    noise_augmentation,
)
from ruinedfooocus.features.image_editing.pid_conditioning import (
    pid_conditioning,
    pid_vae_roles,
    degrade_sigma,
    validate_input_codec,
)
from test_native_model_coverage import native_method, PIPELINE, route_info
from test_model_cache import method
from ruinedfooocus.features.models.component_candidates import classify_shapes, compatibility


class NativeRestorationTests(unittest.TestCase):
    def test_wrong_native_pid_variant_is_rejected_before_codec_payload_loading(self):
        with (
            patch(
                "ruinedfooocus.features.models.component_candidates.inspect_local_component",
                side_effect=lambda role, path: dict(
                    compatibility="compatible" if role == "vae_flux" else "incompatible"
                ),
            ),
            patch(
                "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                return_value=dict(
                    native_class="PiD",
                    native_config=dict(lq_latent_channels=128, latent_spatial_down_factor=16),
                ),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "channels do not match"):
                validate_input_codec(Path("input.safetensors"), "flux", Path("model.safetensors"))

    def test_refiner_codec_requires_native_spatial_and_temporal_layout(self):
        shapes = {
            "encoder.conv_in.weight": (128, 3, 3, 3, 3),
            "decoder.conv_in.weight": (1024, 32, 3, 3, 3),
            "encoder.conv_out.weight": (64, 1024, 3, 3, 3),
            "decoder.conv_out.weight": (3, 128, 3, 3, 3),
        }
        for i, (inp, out) in enumerate(((128, 64), (256, 128), (512, 128), (1024, 128))):
            shapes[f"encoder.down.{i}.downsample.conv.weight"] = (out, inp, 3, 3, 3)
        for i, (inp, out) in enumerate(((1024, 8192), (1024, 4096), (512, 1024), (256, 512))):
            shapes[f"decoder.up.{i}.upsample.conv.weight"] = (out, inp, 3, 3, 3)
        classify = lambda value: classify_shapes(value, detector=lambda _: None)
        self.assertEqual(compatibility("vae_hunyuan_refiner", classify(shapes)), "compatible")
        self.assertEqual(compatibility("vae_hunyuan_image21", classify(shapes)), "incompatible")
        del shapes["decoder.up.3.upsample.conv.weight"]
        self.assertEqual(compatibility("vae_hunyuan_refiner", classify(shapes)), "incompatible")

    def test_parameter_bounds_preserve_explicit_zero(self):
        self.assertEqual(noise_augmentation({}), 0.1)
        self.assertEqual(noise_augmentation(dict(refiner_noise_augmentation=0)), 0)
        self.assertEqual(degrade_sigma({}), 0)
        self.assertEqual(pid_vae_roles("flux"), ("vae_flux", "vae_flux2"))
        for value in (True, -0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                noise_augmentation(dict(refiner_noise_augmentation=value))
            with self.assertRaises(ValueError):
                degrade_sigma(dict(pid_degrade_sigma=value))

    def test_actual_refiner_node_and_native_channel_folding(self):
        values = lambda cond, update: [[tensor, meta | update] for tensor, meta in cond]
        native = NS(model_management=NS(intermediate_device=lambda: "cpu"))
        node = native_method(
            "comfy_extras/nodes_hunyuan.py",
            "HunyuanRefinerLatent",
            "execute",
            dict(
                torch=torch,
                comfy=native,
                node_helpers=NS(conditioning_set_values=values),
                io=NS(NodeOutput=lambda *values: values),
            ),
        )
        source = torch.ones(1, 32, 1, 2, 3)
        vae = NS(
            latent_channels=32,
            latent_dim=3,
            spacial_compression_decode=lambda: 16,
            encode=Mock(return_value=source),
        )
        encoder = native_method("nodes.py", "VAEEncode", "encode", {})
        data = {}
        with patch.dict(
            "sys.modules",
            {
                "nodes": NS(VAEEncode=encoder),
                "comfy_extras.nodes_hunyuan": NS(HunyuanRefinerLatent=node),
            },
        ):
            positive, negative, latent = refiner_conditioning(
                [["p", {}]], [["n", {}]], vae, torch.zeros(1, 32, 48, 3), data
            )
        self.assertIs(positive[0][1]["concat_latent_image"], source)
        self.assertIs(negative[0][1]["concat_latent_image"], source)
        self.assertEqual(tuple(latent["samples"].shape), (1, 32, 1, 2, 3))
        self.assertEqual(data["refiner_noise_augmentation"], 0.1)
        forward = native_method(
            "comfy/latent_formats.py", "HunyuanImage21Refiner", "process_in", dict(torch=torch)
        )()
        inverse = native_method(
            "comfy/latent_formats.py", "HunyuanImage21Refiner", "process_out", dict(torch=torch)
        )()
        forward.scale_factor = inverse.scale_factor = 1.03682
        folded = forward.process_in(source)
        self.assertEqual(tuple(folded.shape), (1, 64, 1, 2, 3))
        torch.testing.assert_close(inverse.process_out(folded), source)

    def test_refiner_rejects_base_codec_before_encoding(self):
        vae = NS(latent_channels=64, encode=Mock())
        with patch.dict(
            "sys.modules",
            {
                "nodes": NS(VAEEncode=Mock()),
                "comfy_extras.nodes_hunyuan": NS(HunyuanRefinerLatent=Mock()),
            },
        ):
            with self.assertRaisesRegex(ValueError, "dedicated"):
                refiner_conditioning([], [], vae, None, {})
        vae.encode.assert_not_called()

    def test_pid_actual_native_node_receives_encoded_source_and_sigma_under_lease(self):
        values = lambda cond, update: [[tensor, meta | update] for tensor, meta in cond]
        source = torch.ones(1, 16, 2, 2)
        fmt = type("NativeFormatFixture", (), {"process_in": lambda self, value: value * 2})
        native = NS(
            latent_formats=NS(Flux=fmt, Flux2=fmt, SD3=fmt, SDXL=fmt, Wan21=fmt),
            model_management=NS(intermediate_device=lambda: "cpu"),
        )
        node = native_method(
            "comfy_extras/nodes_pid.py",
            "PiDConditioning",
            "execute",
            dict(
                torch=torch,
                comfy=native,
                node_helpers=NS(conditioning_set_values=values),
                io=NS(NodeOutput=lambda *values: values),
            ),
        )
        empty = native_method(
            "comfy_extras/nodes_chroma_radiance.py",
            "EmptyChromaRadianceLatentImage",
            "execute",
            dict(torch=torch, comfy=native, io=NS(NodeOutput=lambda *values: values)),
        )
        held = []

        @contextmanager
        def lease(*a, **k):
            held.append(True)
            try:
                yield Path("input.safetensors")
            finally:
                held.pop()

        vae = NS(
            latent_channels=16,
            throw_exception_if_invalid=Mock(),
            encode=Mock(
                side_effect=lambda pixels: (
                    source if held else self.fail("codec encode escaped lease")
                )
            ),
        )
        native.sd = NS(VAE=Mock(return_value=vae))
        native.utils = NS(load_torch_file=Mock(return_value=({}, {})))
        encoder = native_method("nodes.py", "VAEEncode", "encode", {})
        modules = {
            "comfy": native,
            "comfy.sd": native.sd,
            "comfy.utils": native.utils,
            "nodes": NS(VAEEncode=encoder),
            "comfy_extras.nodes_pid": NS(PiDConditioning=node),
            "comfy_extras.nodes_chroma_radiance": NS(EmptyChromaRadianceLatentImage=empty),
        }
        model = NS(diffusion_model=NS(lq_proj=NS(latent_channels=16, sr_scale=4)))
        data = dict(
            pid_input_vae="input.safetensors",
            pid_latent_format="flux",
            pid_degrade_sigma=0.2,
            width=64,
            height=64,
        )
        with (
            patch.dict("sys.modules", modules),
            patch("ruinedfooocus.features.generation.image_preparation.prepared_asset", lease),
            patch("ruinedfooocus.features.image_editing.pid_conditioning.validate_input_codec"),
        ):
            positive, negative, latent = pid_conditioning(
                model, [["p", {}]], [["n", {}]], torch.zeros(1, 16, 16, 3), data, object()
            )
            with self.assertRaisesRegex(ValueError, "output size"):
                pid_conditioning(model, [], [], torch.zeros(1, 32, 16, 3), data, object())
        torch.testing.assert_close(positive[0][1]["lq_latent"], source * 2)
        self.assertAlmostEqual(float(negative[0][1]["degrade_sigma"][0]), 0.2)
        self.assertEqual(tuple(latent["samples"].shape), (1, 3, 64, 64))
        self.assertFalse(held)

    def test_runtime_refiner_text_contract_uses_native_qwen_not_video_llama(self):
        get_info = method(PIPELINE, "get_clip_and_vae", dict(settings=NS(default_settings={})))
        owner = NS(known_model_info=route_info())
        model = type("HunyuanImage21Refiner", (), {})()
        model.model_config = NS(unet_config=dict(context_in_dim=3584, byt5=False))
        info = get_info(owner, NS(model=model))
        self.assertEqual(info["clip_type"], "HUNYUAN_IMAGE")
        self.assertEqual(info["clip_keys"], ["clip_qwen25"])
        model.model_config.unet_config["context_in_dim"] = 4096
        with self.assertRaisesRegex(ValueError, "different native text"):
            get_info(owner, NS(model=model))


if __name__ == "__main__":
    unittest.main()
