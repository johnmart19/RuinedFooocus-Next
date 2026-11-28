"""Native ComfyUI two-pass Latent Hires Fix (LatentUpscaleBy + second KSampler pass)."""

import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from ruinedfooocus.features.generation.hires_fix import (
    HIRES_UPSCALE_METHODS,
    HIRES_UPSCALER_LATENT,
    apply_latent_hires_fix,
    hires_upscaler_choices,
    resolve_hires_seed,
    resolve_hires_upscaler,
    scale_reference_frame,
)
from ruinedfooocus.features.image_editing.image_guidance import (
    FIELD_SCHEMA,
    resolve_guidance,
    validate_guidance,
)


class HiresFixTests(unittest.TestCase):
    def test_resolve_hires_seed_preserves_64bit_explicit_seed_and_derives_auto(self):
        self.assertEqual(resolve_hires_seed(975149368229252, 673438479270215), 673438479270215)
        self.assertEqual(resolve_hires_seed(975149368229252, None), 975149368229253)
        self.assertEqual(resolve_hires_seed(975149368229252, -1), 975149368229253)

    def test_guidance_schema_and_validation_for_hires_fields(self):
        for key in (
            "hires_enabled",
            "hires_scale",
            "hires_upscale_method",
            "hires_steps",
            "hires_denoise",
            "hires_seed",
        ):
            self.assertIn(key, FIELD_SCHEMA)
        resolved = resolve_guidance(
            {
                "hires_enabled": True,
                "hires_scale": 1.5,
                "hires_upscale_method": "nearest-exact",
                "hires_steps": 20,
                "hires_denoise": 0.6,
                "hires_seed": "673438479270215",
            },
            {},
            "Anima",
        )
        self.assertTrue(resolved["hires_enabled"])
        self.assertEqual(resolved["hires_scale"], 1.5)
        self.assertEqual(resolved["hires_upscale_method"], "nearest-exact")
        self.assertEqual(resolved["hires_steps"], 20)
        self.assertEqual(resolved["hires_denoise"], 0.6)
        self.assertEqual(resolved["hires_seed"], 673438479270215)

        for method in HIRES_UPSCALE_METHODS:
            self.assertEqual(
                validate_guidance({"hires_upscale_method": method})["hires_upscale_method"], method
            )
        with self.assertRaises(ValueError):
            validate_guidance({"hires_upscale_method": "invalid"})
        with self.assertRaises(ValueError):
            validate_guidance({"hires_steps": 12.5})
        with self.assertRaises(ValueError):
            validate_guidance({"hires_seed": "not-a-seed"})

    def test_apply_latent_hires_fix_runs_native_upscale_and_second_sample(self):
        calls = []

        class FakeLatentUpscaleBy:
            def upscale(self, samples, upscale_method, scale_by):
                calls.append(("upscale", upscale_method, scale_by))
                return ({"samples": "upscaled-tensor", "batch_index": [0]},)

        def fake_prepare_noise(latent_image, seed, batch_inds):
            calls.append(("prepare_noise", latent_image, seed, batch_inds))
            return "second-noise"

        def fake_sample(**kwargs):
            calls.append(("sample", kwargs))
            return "refined-tensor"

        fake_comfy_sample = NS(prepare_noise=fake_prepare_noise, sample=fake_sample)
        fake_nodes = NS(LatentUpscaleBy=FakeLatentUpscaleBy)
        guidance = resolve_guidance(
            {
                "hires_enabled": True,
                "hires_scale": 1.5,
                "hires_upscale_method": "nearest-exact",
                "hires_steps": 20,
                "hires_denoise": 0.6,
                "hires_seed": 673438479270215,
            },
            {},
            "Anima",
        )
        with patch.dict(
            sys.modules,
            {
                "comfy": NS(sample=fake_comfy_sample),
                "comfy.sample": fake_comfy_sample,
                "nodes": fake_nodes,
            },
        ):
            out = apply_latent_hires_fix(
                {"samples": "pass1-tensor"},
                model="model",
                positive="pos",
                negative="neg",
                cfg=5.0,
                sampler_name="euler_ancestral",
                scheduler="beta",
                base_seed=975149368229252,
                guidance=guidance,
            )
        self.assertEqual(out["samples"], "refined-tensor")
        self.assertEqual(calls[0], ("upscale", "nearest-exact", 1.5))
        self.assertEqual(calls[1], ("prepare_noise", "upscaled-tensor", 673438479270215, [0]))
        self.assertEqual(calls[2][0], "sample")
        sample_kwargs = calls[2][1]
        self.assertEqual(sample_kwargs["steps"], 20)
        self.assertEqual(sample_kwargs["cfg"], 5.0)
        self.assertEqual(sample_kwargs["sampler_name"], "euler_ancestral")
        self.assertEqual(sample_kwargs["scheduler"], "beta")
        self.assertEqual(sample_kwargs["denoise"], 0.6)
        self.assertEqual(sample_kwargs["seed"], 673438479270215)

    def test_disabled_hires_returns_input_untouched(self):
        original = {"samples": "pass1-tensor"}
        out = apply_latent_hires_fix(
            original,
            model=None,
            positive=None,
            negative=None,
            cfg=5.0,
            sampler_name="euler_ancestral",
            scheduler="beta",
            base_seed=1,
            guidance={"hires_enabled": False},
        )
        self.assertIs(out, original)

    def test_openai_api_request_accepts_64bit_seed_and_hires_fields(self):
        from ruinedfooocus.integrations.openai_api import ImageRequest

        req = ImageRequest(
            prompt="cyber_samurai",
            seed=975149368229252,
            hires_enabled=True,
            hires_scale=1.5,
            hires_upscale_method="nearest-exact",
            hires_steps=20,
            hires_denoise=0.6,
            hires_seed=673438479270215,
        )
        self.assertEqual(req.seed, 975149368229252)
        self.assertTrue(req.hires_enabled)
        self.assertEqual(req.hires_seed, 673438479270215)

    def test_scale_reference_frame_matches_upscaled_canvas(self):
        plan = dict(canvas_width=832, canvas_height=1216, width=832, height=1216, left=0, top=0)
        scaled = scale_reference_frame(plan, (1, 16, 1, 152, 104), (1, 16, 1, 228, 156))
        self.assertEqual(
            scaled,
            dict(canvas_width=1248, canvas_height=1824, width=1248, height=1824, left=0, top=0),
        )
        self.assertIsNone(scale_reference_frame(None, (1, 16, 1, 152, 104), (1, 16, 1, 228, 156)))

    def test_hires_upscaler_choices_and_validation(self):
        choices = hires_upscaler_choices()
        self.assertEqual(choices[0], HIRES_UPSCALER_LATENT)
        self.assertIn("2× AMD FSR 4", choices)
        self.assertIn("2× Anime sharp (AnimeSharp V4 Fast)", choices)
        self.assertNotIn("Off", choices)

        self.assertEqual(resolve_hires_upscaler("Latent"), (None, 1))
        self.assertEqual(resolve_hires_upscaler("2× AMD FSR 4"), ("amd-fsr:fsr4", 2))
        self.assertEqual(
            resolve_hires_upscaler("2× Anime sharp (AnimeSharp V4 Fast)"),
            ("2x-AnimeSharpV4_Fast_RCAN_PU.safetensors", 2),
        )

        self.assertEqual(
            validate_guidance({"hires_upscaler": "2× AMD FSR 4"})["hires_upscaler"],
            "2× AMD FSR 4",
        )
        self.assertEqual(
            validate_guidance({"hires_upscaler": ""})["hires_upscaler"],
            HIRES_UPSCALER_LATENT,
        )
        with self.assertRaises(ValueError):
            validate_guidance({"hires_upscaler": "nonexistent-upscaler.pth"})

    def test_apply_hires_fix_with_amd_fsr4_upscaler(self):
        import torch

        calls = []

        class FakeVAE:
            def spacial_compression_decode(self):
                return 8

        class FakeVAEEncode:
            def encode(self, vae, pixels):
                calls.append(("vae_encode", tuple(pixels.shape)))
                return ({"samples": "encoded-fsr4-latent"},)

        def fake_decode(lat):
            calls.append(("decode", lat))
            return torch.zeros((1, 64, 48, 3), dtype=torch.float32)

        def fake_fsr_upscale(version, pixels, width, height, check_cancelled=None):
            calls.append(("fsr_upscale", version, tuple(pixels.shape), width, height))
            return torch.full((1, height, width, 3), 0.5, dtype=torch.float32)

        def fake_prepare_noise(latent_image, seed, batch_inds):
            calls.append(("prepare_noise", latent_image, seed))
            return "noise"

        def fake_sample(**kwargs):
            calls.append(("sample", kwargs["steps"], kwargs["denoise"], kwargs["seed"]))
            return "refined-fsr4-latent"

        fake_comfy_sample = NS(prepare_noise=fake_prepare_noise, sample=fake_sample)
        fake_nodes = NS(VAEEncode=FakeVAEEncode)
        guidance = resolve_guidance(
            {
                "hires_enabled": True,
                "hires_scale": 1.5,
                "hires_upscaler": "2× AMD FSR 4",
                "hires_steps": 15,
                "hires_denoise": 0.5,
                "hires_seed": 42,
            },
            {},
            "Anima",
        )
        gen_data = {}
        with (
            patch.dict(
                sys.modules,
                {
                    "comfy": NS(sample=fake_comfy_sample),
                    "comfy.sample": fake_comfy_sample,
                    "nodes": fake_nodes,
                },
            ),
            patch(
                "ruinedfooocus.features.image_processing.amd_fsr.upscale",
                side_effect=fake_fsr_upscale,
            ),
        ):
            out = apply_latent_hires_fix(
                {"samples": "pass1"},
                model="model",
                positive="pos",
                negative="neg",
                cfg=5.0,
                sampler_name="euler_ancestral",
                scheduler="beta",
                base_seed=10,
                guidance=guidance,
                vae=FakeVAE(),
                decode_fn=fake_decode,
                gen_data=gen_data,
            )
        self.assertEqual(out["samples"], "refined-fsr4-latent")
        self.assertEqual(gen_data["hires_upscaler_model"], "amd-fsr:fsr4")
        # 48 * 1.5 = 72, 64 * 1.5 = 96 (both multiples of 8)
        self.assertEqual(calls[0][0], "decode")
        self.assertEqual(calls[1], ("fsr_upscale", "fsr4", (1, 64, 48, 3), 72, 96))
        self.assertEqual(calls[2], ("vae_encode", (1, 96, 72, 3)))
        self.assertEqual(calls[3], ("prepare_noise", "encoded-fsr4-latent", 42))
        self.assertEqual(calls[4], ("sample", 15, 0.5, 42))

    def test_apply_hires_fix_with_neural_upscaler_and_lanczos_fit(self):
        import torch

        calls = []

        class FakeVAE:
            def spacial_compression_decode(self):
                return 8

        class FakeVAEEncode:
            def encode(self, vae, pixels):
                calls.append(("vae_encode", tuple(pixels.shape)))
                return ({"samples": "encoded-neural-latent"},)

        class FakeUpscalePipe:
            def load_upscaler_model(self, name, gen_data):
                calls.append(("load_upscaler", name))
                return NS(scale=2, input_channels=3, output_channels=3)

        class FakeImageUpscaleWithModel:
            @staticmethod
            def execute(model, image):
                calls.append(("neural_upscale", tuple(image.shape)))
                return (torch.zeros((1, image.shape[1] * 2, image.shape[2] * 2, 3)),)

        def fake_common_upscale(tensor, w, h, method, crop):
            calls.append(("common_upscale", tuple(tensor.shape), w, h, method, crop))
            return torch.zeros((tensor.shape[0], tensor.shape[1], h, w), dtype=tensor.dtype)

        fake_comfy_sample = NS(
            prepare_noise=lambda lat, seed, batch: "noise",
            sample=lambda **kwargs: "refined-neural-latent",
        )
        fake_comfy_utils = NS(common_upscale=fake_common_upscale)
        fake_nodes = NS(VAEEncode=FakeVAEEncode)
        fake_upscale_mod = NS(ImageUpscaleWithModel=FakeImageUpscaleWithModel)
        fake_upscale_pipe_mod = NS(
            InterruptibleUpscaler=lambda model, gen_data: model,
            pipeline=lambda: FakeUpscalePipe(),
        )
        guidance = resolve_guidance(
            {
                "hires_enabled": True,
                "hires_scale": 1.5,
                "hires_upscaler": "2× Anime sharp (AnimeSharp V4 Fast)",
                "hires_steps": 18,
                "hires_denoise": 0.55,
                "hires_seed": 99,
            },
            {},
            "Anima",
        )
        owner = NS(automatic_upscale_pipeline=FakeUpscalePipe())
        gen_data = {}
        with patch.dict(
            sys.modules,
            {
                "comfy": NS(sample=fake_comfy_sample, utils=fake_comfy_utils),
                "comfy.sample": fake_comfy_sample,
                "comfy.utils": fake_comfy_utils,
                "comfy_extras": NS(nodes_upscale_model=fake_upscale_mod),
                "comfy_extras.nodes_upscale_model": fake_upscale_mod,
                "nodes": fake_nodes,
                "ruinedfooocus.features.image_processing.upscale_pipeline": fake_upscale_pipe_mod,
            },
        ):
            out = apply_latent_hires_fix(
                {"samples": "pass1"},
                model="model",
                positive="pos",
                negative="neg",
                cfg=5.0,
                sampler_name="euler_ancestral",
                scheduler="beta",
                base_seed=10,
                guidance=guidance,
                vae=FakeVAE(),
                decode_fn=lambda lat: torch.zeros((1, 64, 48, 3), dtype=torch.float32),
                owner=owner,
                gen_data=gen_data,
            )
        self.assertEqual(out["samples"], "refined-neural-latent")
        self.assertEqual(
            gen_data["hires_upscaler_model"], "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors"
        )
        self.assertEqual(calls[0], ("load_upscaler", "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors"))
        self.assertEqual(calls[1], ("neural_upscale", (1, 64, 48, 3)))
        # 2x neural output is 128x96 -> resized by Lanczos to 1.5x target (96x72)
        self.assertEqual(
            calls[2], ("common_upscale", (1, 3, 128, 96), 72, 96, "lanczos", "disabled")
        )
        self.assertEqual(calls[3], ("vae_encode", (1, 96, 72, 3)))

    def test_openai_api_request_accepts_hires_upscaler(self):
        from ruinedfooocus.integrations.openai_api import ImageRequest

        req = ImageRequest(
            prompt="cyber_samurai",
            hires_enabled=True,
            hires_upscaler="2× Anime sharp (AnimeSharp V4 Fast)",
            automatic_upscale="2× AMD FSR 4",
        )
        self.assertEqual(req.hires_upscaler, "2× Anime sharp (AnimeSharp V4 Fast)")
        self.assertEqual(req.automatic_upscale, "2× AMD FSR 4")

    def test_custom_recipe_scheduler_falls_back_to_simple_for_hires_pass(self):
        captured = {}

        class FakeLatentUpscaleBy:
            def upscale(self, samples, upscale_method, scale_by):
                return ({"samples": "upscaled"},)

        def fake_sample(**kwargs):
            captured.update(kwargs)
            return "refined"

        fake_comfy_sample = NS(prepare_noise=lambda *a: "noise", sample=fake_sample)
        fake_comfy_samplers = NS(KSampler=NS(SCHEDULERS=("simple", "karras", "beta")))
        fake_nodes = NS(LatentUpscaleBy=FakeLatentUpscaleBy)
        guidance = resolve_guidance({"hires_enabled": True, "hires_steps": 10}, {}, "Flux2")
        with patch.dict(
            sys.modules,
            {
                "comfy": NS(sample=fake_comfy_sample, samplers=fake_comfy_samplers),
                "comfy.sample": fake_comfy_sample,
                "comfy.samplers": fake_comfy_samplers,
                "nodes": fake_nodes,
            },
        ):
            apply_latent_hires_fix(
                {"samples": "pass1"},
                model="model",
                positive="pos",
                negative="neg",
                cfg=1.0,
                sampler_name="euler",
                scheduler="Flux2Scheduler",
                base_seed=1,
                guidance=guidance,
            )
        self.assertEqual(captured["scheduler"], "simple")


if __name__ == "__main__":
    unittest.main()
