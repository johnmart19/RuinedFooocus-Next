"""Actual pinned CPU text, Kornia morphology and SDR-to-HDR operations."""

from concurrent.futures import CancelledError
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ruinedfooocus.features.image_processing import color_native as native
from ruinedfooocus.features.image_processing import image_color_tools as colors
from ruinedfooocus.features.image_processing.native_image_formats import ImportedImage


class ImageColorFinishingTests(unittest.TestCase):
    def capture(self, operation, source, **options):
        data = dict(input_image=source, color_operation=operation, **options)
        result, assets = colors.capture_assets(data, None)
        self.assertEqual(assets, ())
        return result

    def run_color(self, data, cancel=lambda: None):
        return np.asarray(colors.process(None, data, None, cancel)[0])

    def test_native_text_rgba_union_draws_on_transparency_and_keeps_original(self):
        source = Image.new("RGBA", (64, 48), (0, 0, 0, 0))
        original = source.tobytes()
        data = self.capture(
            "text-overlay",
            source,
            color_text="A",
            color_text_font_size=30,
            color_text_outline=False,
            color_text_color="#FF0000",
        )
        result = self.run_color(data)
        painted = result[..., 3] > 0
        self.assertTrue(painted.any())
        self.assertTrue((result[..., 0][painted] > 0).all())
        self.assertTrue((result[..., 1:3][painted] == 0).all())
        self.assertEqual(source.tobytes(), original)
        self.assertIn("glyph alpha union", data["image_utility_source"]["alpha"])

    def test_native_text_preserves_nonopaque_pixels_outside_glyphs(self):
        source = Image.new("RGBA", (64, 48), (25, 50, 75, 64))
        result = self.run_color(
            self.capture(
                "text-overlay",
                source,
                color_text="A",
                color_text_font_size=30,
                color_text_outline=False,
            )
        )
        painted = result[..., 3] > 64
        self.assertTrue(painted.any())
        self.assertTrue((~painted).any())
        expected = np.broadcast_to(
            np.array([25, 50, 75, 64], dtype=np.uint8), result[~painted].shape
        )
        np.testing.assert_array_equal(result[~painted], expected)
        self.assertEqual(source.getpixel((0, 0)), (25, 50, 75, 64))

    def test_native_text_blank_is_identity_and_literal_newline_is_native(self):
        source = Image.new("RGB", (64, 48), (25, 50, 75))
        identity = self.run_color(self.capture("text-overlay", source, color_text=" \t\n"))
        np.testing.assert_array_equal(identity[..., :3], np.asarray(source))
        escaped = self.run_color(
            self.capture(
                "text-overlay",
                source,
                color_text="A\\nB",
                color_text_font_size=25,
                color_text_position="bottom",
                color_text_align="right",
            )
        )
        literal = self.run_color(
            self.capture(
                "text-overlay",
                source,
                color_text="A\nB",
                color_text_font_size=25,
                color_text_position="bottom",
                color_text_align="right",
            )
        )
        np.testing.assert_array_equal(escaped, literal)
        self.assertFalse(np.array_equal(escaped, identity))

    def test_native_text_frozen_content_metadata_and_replay_ignore_prompt_state(self):
        source = Image.new("RGBA", (64, 48), (10, 20, 30, 64))
        data = self.capture("text-overlay", source, color_text="Before", color_text_font_size=20)
        data.update(
            color_text="After", prompt="#! generation prompt is irrelevant", loras=[object()]
        )
        worker = colors.worker_data(data)
        self.assertEqual(worker["prompt"], "")
        self.assertEqual(worker["loras"], [])
        self.assertEqual(worker["color_text"], "Before")
        self.run_color(worker)
        metadata = colors.output_metadata(worker)
        self.assertEqual(metadata["color_text"], "Before")
        replay = colors.replay_values(metadata)
        self.assertEqual(replay["color_text"], "Before")
        self.assertIsNone(replay["input_image"])
        self.assertIsNone(replay["color_reference"])

    def test_text_rejects_precision_hdr_invalid_fields_and_excessive_font_work(self):
        source = Image.new("RGBA", (16, 16))
        for bad_source, options in (
            (np.zeros((16, 16, 4), dtype=np.uint16), dict(output_bit_depth="16-bit")),
            (np.zeros((16, 16, 4), dtype=np.float32), dict(output_bit_depth="16-bit")),
            (
                ImportedImage(np.zeros((16, 16, 3), dtype=np.float32), "HDR", "32-bit float", {}),
                dict(
                    output_format="exr", output_bit_depth="32-bit float", output_color_space="HDR"
                ),
            ),
            (source, dict(color_text="x" * 2049)),
            (source, dict(color_text="x\\n" * 64)),
            (source, dict(color_text=1)),
            (source, dict(color_text_color="white")),
            (source, dict(color_text_outline=1)),
            (source, dict(color_text_position="center")),
            (source, dict(color_text_font_size=float("nan"))),
            (Image.new("RGBA", (8, 1027)), dict(color_text="A", color_text_font_size=50)),
        ):
            with (
                self.subTest(source=type(bad_source).__name__, options=list(options)),
                self.assertRaises(ValueError),
            ):
                self.capture("text-overlay", bad_source, **options)

    def test_all_native_morphology_modes_keep_rgb_float_and_exact_alpha(self):
        source = np.zeros((7, 7, 4), dtype=np.float32)
        source[2:5, 2:5, :3] = 0.7
        source[..., 3] = 0.1234567
        original = source.copy()
        expected_counts = dict(
            erode=1, dilate=25, open=9, close=9, gradient=24, bottom_hat=0, top_hat=0
        )
        for operation in colors.MORPHOLOGY_OPERATIONS:
            with self.subTest(operation=operation):
                data = self.capture(
                    "morphology",
                    source,
                    color_morphology_operation=operation,
                    output_bit_depth="16-bit",
                )
                result = self.run_color(data)
                self.assertEqual(np.count_nonzero(result[..., 0]), expected_counts[operation])
                self.assertEqual(result.dtype, np.float32)
                np.testing.assert_array_equal(result[..., 3], source[..., 3])
                self.assertEqual(data["image_utility_source"]["native_node"], "Morphology")
        np.testing.assert_array_equal(source, original)

    def test_native_morphology_accepts_even_kernel_and_retains_extended_hdr_rgb(self):
        source = np.full((5, 5, 4), 2.25, dtype=np.float32)
        source[..., 3] = 0.1234567
        original = source.copy()
        imported = ImportedImage(source, "HDR", "32-bit float", {})
        data = self.capture(
            "morphology",
            imported,
            color_morphology_kernel=4,
            output_format="exr",
            output_bit_depth="32-bit float",
            output_color_space="HDR",
        )
        result = self.run_color(data)
        np.testing.assert_array_equal(result, original)
        np.testing.assert_array_equal(source, original)

    def test_morphology_admission_checks_types_kernel_and_work_before_native(self):
        source = Image.new("RGB", (16, 16))
        for options in (
            dict(color_morphology_kernel=2),
            dict(color_morphology_kernel=3.5),
            dict(color_morphology_kernel=True),
            dict(color_morphology_kernel=999),
            dict(color_morphology_operation="invented"),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.capture("morphology", source, **options)
        with self.assertRaisesRegex(ValueError, "requires EXR"):
            self.capture(
                "morphology", np.full((5, 5, 3), 2.25, dtype=np.float32), output_bit_depth="16-bit"
            )

    def test_native_sdr_to_hdr_ramp_and_target_luminance_keep_float_alpha(self):
        levels = np.linspace(0, 1, 5, dtype=np.float32)
        source = np.empty((3, 5, 4), dtype=np.float32)
        source[..., :3] = levels[None, :, None]
        source[..., 3] = 0.1234567
        original = source.copy()
        outputs = []
        for luminance in (0, 250, 650, 1000):
            data = self.capture(
                "sdr-to-hdr",
                source,
                color_hdr_luminance=luminance,
                output_format="exr",
                output_bit_depth="32-bit float",
                output_color_space="HDR",
            )
            result = self.run_color(data)
            outputs.append(result)
            self.assertEqual(result.dtype, np.float32)
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue((np.diff(result[0, :, 0]) > 0).all())
            np.testing.assert_allclose(result[..., 0], result[..., 1], atol=1e-6)
            np.testing.assert_allclose(result[..., 1], result[..., 2], atol=1e-6)
            np.testing.assert_array_equal(result[:, 0, :3], 0)
            np.testing.assert_array_equal(result[..., 3], source[..., 3])
            self.assertEqual(data["image_utility_source"]["color_space"], "HDR")
            self.assertEqual(data["image_utility_source"]["source_color_space"], "sRGB")
            self.assertEqual(colors.output_metadata(data)["output_color_space"], "HDR")
        self.assertFalse(np.array_equal(outputs[1][..., :3], outputs[2][..., :3]))
        np.testing.assert_array_equal(source, original)

    def test_sdr_to_hdr_uses_uint16_original_and_freezes_exr_hdr_export(self):
        source = np.full((3, 5, 4), 10001, dtype=np.uint16)
        source[..., 3] = 12345
        data = self.capture(
            "sdr-to-hdr",
            source,
            output_format="exr",
            output_bit_depth="32-bit float",
            output_color_space="HDR",
        )
        data.update(output_format="png", output_bit_depth="8-bit", output_color_space="sRGB")
        worker = colors.worker_data(data)
        self.assertEqual((worker["output_format"], worker["output_color_space"]), ("exr", "HDR"))
        result = self.run_color(data)
        np.testing.assert_allclose(result[..., 3], 12345 / 65535, atol=1e-8)
        self.assertEqual((data["output_format"], data["output_color_space"]), ("exr", "HDR"))

    def test_hdr_rejects_non_sdr_invalid_luminance_and_non_float_export(self):
        source = Image.new("RGB", (5, 5))
        for bad_source, options in (
            (source, {}),
            (
                source,
                dict(output_format="png", output_bit_depth="16-bit", output_color_space="HDR"),
            ),
            (
                source,
                dict(
                    output_format="exr", output_bit_depth="32-bit float", output_color_space="sRGB"
                ),
            ),
            (
                source,
                dict(
                    output_format="exr",
                    output_bit_depth="32-bit float",
                    output_color_space="HDR",
                    color_hdr_luminance=-1,
                ),
            ),
            (
                np.full((5, 5, 3), 1.5, dtype=np.float32),
                dict(
                    output_format="exr", output_bit_depth="32-bit float", output_color_space="HDR"
                ),
            ),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.capture("sdr-to-hdr", bad_source, **options)

    def test_each_new_native_operation_cancels_without_metadata_or_source_mutation(self):
        for operation in ("text-overlay", "morphology", "sdr-to-hdr"):
            with self.subTest(operation=operation):
                source = Image.new("RGBA", (32, 24), (25, 50, 75, 64))
                original = source.tobytes()
                options = dict(color_text="A", color_text_font_size=25)
                if operation == "sdr-to-hdr":
                    options.update(
                        output_format="exr",
                        output_bit_depth="32-bit float",
                        output_color_space="HDR",
                    )
                data = self.capture(operation, source, **options)
                checks = 0

                def cancel():
                    nonlocal checks
                    checks += 1
                    if checks == 3:
                        raise CancelledError()

                with self.assertRaises(CancelledError):
                    self.run_color(data, cancel)
                self.assertNotIn("image_utility_source", data)
                self.assertEqual(source.tobytes(), original)
                with patch.object(native, "source_signatures", return_value=()):
                    with self.assertRaisesRegex(ValueError, "source changed"):
                        self.run_color(data)


if __name__ == "__main__":
    unittest.main()
