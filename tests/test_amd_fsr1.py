"""AMD FSR 1 (EASU and RCAS) and the Straighten transform: bounded values and geometry."""

import unittest
from unittest.mock import patch

import torch

from ruinedfooocus.features.image_processing import amd_fsr, amd_fsr1
from ruinedfooocus.features.image_processing import image_geometry_tools as transform


class AmdFsr1Tests(unittest.TestCase):
    def test_flat_colour_and_alpha_survive_every_pass_exactly(self):
        pixels = torch.tensor([0.25, 0.5, 0.75, 0.6]).expand(1, 6, 5, 4).clone()
        result = amd_fsr1.upscale(pixels, 15, 18)
        self.assertEqual(tuple(result.shape), (1, 18, 15, 4))
        self.assertTrue(torch.allclose(result, pixels[:, :1, :1].expand_as(result), atol=1e-5))

    def test_edges_stay_finite_inside_the_source_range_and_sharper_than_bilinear(self):
        pixels = torch.zeros(1, 16, 16, 3)
        pixels[:, :, 8:] = 1.0
        result = amd_fsr1.upscale(pixels, 32, 32)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)
        bilinear = torch.nn.functional.interpolate(
            pixels.movedim(-1, 1), size=(32, 32), mode="bilinear", align_corners=False
        ).movedim(1, -1)

        def soft(image):
            return int(((image[0, 16, :, 0] > 0.02) & (image[0, 16, :, 0] < 0.98)).sum())

        self.assertLessEqual(soft(result), soft(bilinear))

        def steepest(image):
            return float((image[0, 16, 1:, 0] - image[0, 16, :-1, 0]).max())

        self.assertGreater(steepest(result), steepest(bilinear))
        # The step keeps its place: dark on the left half, light on the right.
        self.assertLess(float(result[0, :, :14].max()), 0.05)
        self.assertGreater(float(result[0, :, 18:].min()), 0.95)

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

    def test_large_targets_take_passes_of_at_most_two_and_reductions_are_refused(self):
        self.assertEqual(amd_fsr1.pass_sizes((100, 50), (200, 100)), [(200, 100)])
        self.assertEqual(amd_fsr1.pass_sizes((100, 50), (100, 50)), [(100, 50)])
        steps = amd_fsr1.pass_sizes((100, 50), (400, 200))
        self.assertEqual(steps, [(200, 100), (400, 200)])
        steps = amd_fsr1.pass_sizes((100, 100), (300, 250))
        self.assertEqual(steps[-1], (300, 250))
        previous = (100, 100)
        for step in steps:
            self.assertLessEqual(step[0] / previous[0], 2.0 + 1e-6)
            self.assertLessEqual(step[1] / previous[1], 2.0 + 1e-6)
            previous = step
        for target in ((99, 200), (200, 49)):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "only enlarges"):
                amd_fsr1.pass_sizes((100, 50), target)

    def test_transform_resize_offers_it_and_refuses_unsuitable_requests_at_capture(self):
        from PIL import Image

        self.assertIn(amd_fsr.INTERPOLATION, transform.INTERPOLATIONS)
        source = Image.new("RGBA", (8, 6), (10, 20, 30, 255))
        options = dict(
            input_image=source,
            controlnet={"type": "image-transform"},
            transform_operation="resize",
            transform_interpolation="amd-fsr",
            transform_fsr_version="fsr1",
        )
        captured, _ = transform.capture_assets(
            dict(options, transform_width=16, transform_height=12), None
        )
        self.assertEqual(captured[transform.KEY].output_size, (16, 12))
        # Centre-crop enlarges to cover the target, then trims.
        self.assertEqual(
            transform._fsr_cover((8, 6), (16, 16), dict(transform_fit="center-crop")),
            (22, 16),
        )
        with self.assertRaisesRegex(ValueError, "only enlarges"):
            transform.capture_assets(dict(options, transform_width=4, transform_height=3), None)
        # FSR 1 needs neither a GPU nor the AMD runtime.
        with patch(
            "ruinedfooocus.features.image_processing.image_shaders.cpu_mode", return_value=True
        ):
            transform.capture_assets(dict(options, transform_width=16, transform_height=12), None)
        with self.assertRaisesRegex(ValueError, "fsr version"):
            transform.capture_assets(
                dict(
                    options, transform_fsr_version="fsr5", transform_width=16, transform_height=12
                ),
                None,
            )


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
