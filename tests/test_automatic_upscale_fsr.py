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
        for version in ("fsr4", "fsr3", "fsr2", "fsr1"):
            label_2x = f"2× AMD FSR {version[-1]}"
            label_4x = f"4× AMD FSR {version[-1]}"
            self.assertIn(label_2x, all_choices)
            self.assertIn(label_2x, x2_choices)
            self.assertNotIn(label_2x, x4_choices)
            self.assertEqual(automatic_upscale.resolve(label_2x), (f"amd-fsr:{version}", 2))
            self.assertIn(label_4x, all_choices)
            self.assertIn(label_4x, x4_choices)
            self.assertNotIn(label_4x, x2_choices)
            self.assertEqual(automatic_upscale.resolve(label_4x), (f"amd-fsr:{version}", 4))
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

        calls.clear()
        gen_data_4x = {"task_id": "t2", "automatic_upscale": "4× AMD FSR 4"}
        with (
            patch("ruinedfooocus.features.image_processing.amd_fsr.upscale", side_effect=fake_upscale),
            patch.dict(sys.modules, {"ruinedfooocus.app.async_worker": fake_worker}),
            patch.object(app_pkg, "async_worker", fake_worker, create=True),
        ):
            out4 = automatic_upscale.upscale_decoded(NS(), image[:1], gen_data_4x)
        self.assertEqual(tuple(out4.shape), (1, 256, 192, 3))
        self.assertEqual(calls, [("fsr4", (1, 64, 48, 3), 192, 256)])
        self.assertEqual(gen_data_4x["automatic_upscale_model"], "amd-fsr:fsr4")

    def test_switch_factor_selection_preserves_active_upscale_instead_of_resetting_to_off(self):
        x4_choices = automatic_upscale.choices(factor=4)
        x2_choices = automatic_upscale.choices(factor=2)
        self.assertEqual(
            automatic_upscale.switch_factor_selection("2× AMD FSR 4", x4_choices, "4"),
            "4× AMD FSR 4",
        )
        self.assertEqual(
            automatic_upscale.switch_factor_selection("4× AMD FSR 2", x2_choices, "2"),
            "2× AMD FSR 2",
        )
        self.assertEqual(
            automatic_upscale.switch_factor_selection("Off", x4_choices, "4"),
            "Off",
        )
        switched = automatic_upscale.switch_factor_selection("2× Anime (Futsuu)", x4_choices, "4")
        self.assertNotEqual(switched, "Off")
        self.assertIn(switched, x4_choices)


if __name__ == "__main__":
    unittest.main()
