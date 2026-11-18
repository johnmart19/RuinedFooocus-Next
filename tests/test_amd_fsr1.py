"""AMD RCAS sharpening and the Straighten transform: bounded values and geometry."""

import unittest

import torch

from ruinedfooocus.features.image_processing import amd_fsr1
from ruinedfooocus.features.image_processing import image_geometry_tools as transform


class RcasTests(unittest.TestCase):
    def test_sharpening_never_leaves_the_local_range_and_respects_its_stops(self):
        generator = torch.Generator().manual_seed(7)
        pixels = torch.rand(1, 12, 12, 3, generator=generator)
        strong, weak = amd_fsr1.rcas(pixels, 0.0), amd_fsr1.rcas(pixels, 2.0)
        for result in (strong, weak):
            self.assertTrue(torch.isfinite(result).all())
            self.assertGreaterEqual(float(result.min()), 0.0)
            self.assertLessEqual(float(result.max()), 1.0)
        self.assertGreater(
            float((strong - pixels).abs().mean()), float((weak - pixels).abs().mean())
        )
        flat = torch.full((1, 4, 4, 3), 0.5)
        self.assertTrue(torch.equal(amd_fsr1.rcas(flat), flat))
        for black_or_white in (torch.zeros(1, 4, 4, 3), torch.ones(1, 4, 4, 3)):
            self.assertTrue(torch.equal(amd_fsr1.rcas(black_or_white), black_or_white))
        with self.assertRaises(ValueError):
            amd_fsr1.rcas(pixels, -0.1)


class StraightenTests(unittest.TestCase):
    def test_fitted_rectangle_matches_known_geometry_and_never_exceeds_the_source(self):
        self.assertEqual(transform.straightened_size((100, 50), 0), (100, 50))
        self.assertEqual(transform.straightened_size((100, 50), 180), (100, 50))
        self.assertEqual(transform.straightened_size((100, 50), 90), (50, 100))
        self.assertEqual(transform.straightened_size((100, 100), 45), (70, 70))
        for angle in (-30, -5, 3, 12.5, 60, 135):
            width, height = transform.straightened_size((640, 480), angle)
            self.assertTrue(0 < width <= 640 and 0 < height <= 640, (angle, width, height))
            self.assertEqual(transform.straightened_size((640, 480), -angle), (width, height))

    def test_rotation_is_clockwise_keeps_the_centre_and_leaves_no_empty_corner(self):
        pixels = torch.zeros(1, 41, 41, 3)
        pixels[:, :10, 16:25, 0] = 1.0  # A red bar at the top centre.
        pixels[..., 2] = 0.5
        width, height = transform.straightened_size((41, 41), 90)
        turned = transform._straighten(torch, pixels, 90, width, height)
        # Clockwise by a quarter turn moves the top bar to the right edge.
        self.assertGreater(float(turned[0, 16:25, -6:, 0].mean()), 0.9)
        self.assertLess(float(turned[0, :6, 16:25, 0].mean()), 0.1)
        width, height = transform.straightened_size((41, 41), 20)
        tilted = transform._straighten(torch, pixels, 20, width, height)
        self.assertEqual(tuple(tilted.shape), (1, height, width, 3))
        # Every output pixel comes from inside the source: blue stays at its constant value.
        self.assertTrue(
            torch.allclose(tilted[..., 2], torch.full_like(tilted[..., 2], 0.5), atol=1e-4)
        )
        same = transform._straighten(torch, pixels, 0, 41, 41)
        self.assertTrue(torch.allclose(same, pixels, atol=1e-5))

    def test_capture_validates_the_angle_and_reports_the_cropped_size(self):
        from PIL import Image

        options = dict(
            input_image=Image.new("RGB", (100, 50)),
            controlnet={"type": "image-transform"},
            transform_operation="straighten",
        )
        captured, _ = transform.capture_assets(dict(options, transform_angle=90), None)
        self.assertEqual(captured[transform.KEY].output_size, (50, 100))
        for angle in (181, -180.5, float("nan"), "10", True):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                transform.capture_assets(dict(options, transform_angle=angle), None)


if __name__ == "__main__":
    unittest.main()
