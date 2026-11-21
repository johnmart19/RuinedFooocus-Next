"""Built-in AMD FSR 1-4 presets in Create -> Main -> Upscale (automatic_upscale)."""

from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch

from ruinedfooocus.features.generation import automatic_upscale


class AutomaticUpscaleFSRTests(unittest.TestCase):
    def test_choices_and_resolve_expose_all_four_fsr_versions(self):
        all_choices = automatic_upscale.choices()
        x2_choices = automatic_upscale.choices(factor=2)
        x4_choices = automatic_upscale.choices(factor=4)
        for version, label in (
            ("fsr4", "2× AMD FSR 4"),
            ("fsr3", "2× AMD FSR 3"),
            ("fsr2", "2× AMD FSR 2"),
            ("fsr1", "2× AMD FSR 1"),
        ):
            self.assertIn(label, all_choices)
            self.assertIn(label, x2_choices)
            self.assertNotIn(label, x4_choices)
            self.assertEqual(automatic_upscale.resolve(label), (f"amd-fsr:{version}", 2))
            self.assertEqual(automatic_upscale.fsr_version_for(f"amd-fsr:{version}"), version)
        self.assertIsNone(automatic_upscale.fsr_version_for("2x-AniScale.pth"))

    def test_upscale_decoded_routes_fsr_presets_through_amd_fsr(self):
        calls = []

        def fake_upscale(version, pixels, width, height, *, check_cancelled=None):
            calls.append((version, tuple(pixels.shape), width, height))
            return torch.zeros((pixels.shape[0], height, width, pixels.shape[3]))

        image = torch.zeros((2, 64, 48, 3))
        gen_data = {"task_id": "t1", "automatic_upscale": "2× AMD FSR 4"}
        import sys
        import ruinedfooocus.app as app_pkg

        fake_worker = NS(check_interrupt=lambda *a, **k: None, add_result=lambda *a, **k: None)
        with (
            patch("ruinedfooocus.features.image_processing.amd_fsr.upscale", side_effect=fake_upscale),
            patch.dict(sys.modules, {"ruinedfooocus.app.async_worker": fake_worker}),
            patch.object(app_pkg, "async_worker", fake_worker, create=True),
        ):
            out = automatic_upscale.upscale_decoded(NS(), image, gen_data)
        self.assertEqual(tuple(out.shape), (2, 128, 96, 3))
        self.assertEqual(
            calls,
            [
                ("fsr4", (1, 64, 48, 3), 96, 128),
                ("fsr4", (1, 64, 48, 3), 96, 128),
            ],
        )
        self.assertEqual(gen_data["automatic_upscale_model"], "amd-fsr:fsr4")


if __name__ == "__main__":
    unittest.main()
