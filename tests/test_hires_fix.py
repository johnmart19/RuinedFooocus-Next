"""Native ComfyUI two-pass Latent Hires Fix (LatentUpscaleBy + second KSampler pass)."""

import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from ruinedfooocus.features.generation.hires_fix import (
    HIRES_UPSCALE_METHODS,
    apply_latent_hires_fix,
    resolve_hires_seed,
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


if __name__ == "__main__":
    unittest.main()
