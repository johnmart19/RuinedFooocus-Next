"""Pixel fixtures for exact targets and removable native-only canvas padding."""

from dataclasses import FrozenInstanceError
import unittest

import torch

from ruinedfooocus.features.generation.size_plan import SizePlan, fit_tensor, replay_size


class SizePlanTests(unittest.TestCase):
    def test_inverse_fullhd_2x_and_4x_are_exact_and_immutable(self):
        for scale, content in ((2, (960, 540)), (4, (480, 270))):
            plan = SizePlan.create((1920, 1080), scale, alignment=16)
            self.assertEqual(plan.content, content)
            self.assertEqual(plan.canvas, (content[0], (content[1] + 15) // 16 * 16))
            self.assertEqual(plan.restoration, "padding_only")
            self.assertEqual(SizePlan.from_metadata(plan.metadata()), plan)
            with self.assertRaises(FrozenInstanceError):
                plan.target_width = 42

    def test_padding_removal_preserves_pattern_before_native_upscale(self):
        for scale in (2, 4):
            plan = SizePlan.create((120, 68), scale, alignment=16)
            content = (
                torch.arange(plan.content_width * plan.content_height)
                .reshape(1, plan.content_height, plan.content_width, 1)
                .float()
            )
            canvas = torch.full((1, plan.canvas_height, plan.canvas_width, 1), -999.0)
            canvas[
                :,
                plan.top : plan.top + plan.content_height,
                plan.left : plan.left + plan.content_width,
            ] = content
            restored = plan.restore_canvas(canvas)
            self.assertTrue(torch.equal(restored, content))
            enlarged = restored.repeat_interleave(scale, 1).repeat_interleave(scale, 2)
            self.assertTrue(torch.equal(plan.finish(enlarged), enlarged))
            self.assertEqual(tuple(enlarged.shape[1:3]), (68, 120))

    def test_odd_target_uniform_fit_transparent_padding_preserves_every_edge(self):
        plan = SizePlan.create((19, 11), 4, alignment=8)
        self.assertEqual(plan.content, (5, 3))
        self.assertEqual(plan.restoration, "uniform_fit_pad")
        pixels = torch.ones((1, 12, 20, 4))
        pixels[:, :, :2, 0] = 0
        pixels[:, :, -2:, 2] = 0
        result = plan.finish(pixels)
        self.assertEqual(tuple(result.shape), (1, 11, 19, 4))
        self.assertEqual(float(result[0, 5, -1, 3]), 0)
        self.assertLess(float(result[0, 5, 0, 0]), 0.2)
        self.assertLess(float(result[0, 5, -2, 2]), 0.2)
        self.assertTrue(torch.equal(pixels[..., 3], torch.ones_like(pixels[..., 3])))

    def test_uniform_fit_never_stretches_or_crops_source(self):
        pixels = torch.ones((1, 8, 16, 4))
        result = fit_tensor(pixels, (16, 16))
        self.assertTrue(torch.equal(result[:, 4:12], pixels))
        self.assertTrue(torch.equal(result[:, :4, :, 3], torch.zeros((1, 4, 16))))

    def test_invalid_constraints_and_corrupt_metadata_fail_before_native_work(self):
        for target in ((0, 1080), (-1, 1080), (True, 1080), (1920.5, 1080), (float("nan"), 1)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                SizePlan.create(target)
        with self.assertRaisesRegex(ValueError, "different factor"):
            SizePlan.create((1920, 1080), 4, minimum=512)
        with self.assertRaisesRegex(ValueError, "maximum"):
            SizePlan.create((4095, 2048), alignment=32, maximum=4095)
        value = SizePlan.create((1920, 1080), 2).metadata()
        value["left"] += 1
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            SizePlan.from_metadata(value)

    def test_replay_migrates_old_png_without_applying_inverse_scale_twice(self):
        old = dict(width=1920, height=1080, generation_width=960, generation_height=540)
        self.assertEqual(replay_size(old), (960, 540))
        new = dict(old, size_plan=SizePlan.create((1920, 1080), 2).metadata())
        self.assertEqual(replay_size(new), (1920, 1080))


if __name__ == "__main__":
    unittest.main()
