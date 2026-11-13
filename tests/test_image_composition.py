"""Actual pinned pure-image bodies, bounded captures, alpha and ordered frames."""

from concurrent.futures import CancelledError
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from ruinedfooocus.features.image_processing import image_composition as composition
from ruinedfooocus.features.image_processing import composition_native as native


class ImageCompositionTests(unittest.TestCase):
    def capture(self, images, *, kind="image-composite", **values):
        return composition.capture(dict(composition_images=images, **values), kind)

    def test_native_grid_keeps_order_complete_framing_and_explicit_black_alpha(self):
        first = Image.new("RGBA", (32, 16), (255, 0, 0, 255))
        first.putpixel((0, 0), (0, 255, 0, 255))
        first.putpixel((31, 15), (255, 255, 255, 255))
        second = Image.new("RGBA", (16, 32), (0, 0, 255, 128))
        data = self.capture(
            [first, second],
            kind="image-grid",
            grid_columns=2,
            grid_cell_width=32,
            grid_cell_height=32,
            grid_padding=4,
        )
        result = composition.process(data, lambda: None)
        self.assertEqual((result.mode, result.size), ("RGB", (68, 32)))
        self.assertEqual(result.getpixel((0, 8)), (0, 255, 0))
        self.assertEqual(result.getpixel((31, 23)), (255, 255, 255))
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(result.getpixel((44, 0)), (0, 0, 128))
        self.assertEqual(result.getpixel((32, 16)), (0, 0, 0))
        self.assertEqual(data["native_composition_source"]["native"], "ImageGrid._group_process")
        self.assertEqual(
            [item["index"] for item in data["native_composition_source"]["inputs"]], [0, 1]
        )

    def test_porter_duff_uses_native_inverted_masks_and_preserves_rgba(self):
        base = Image.new("RGBA", (2, 2), (0, 0, 255, 255))
        layer = Image.new("RGBA", (2, 2), (255, 0, 0, 128))
        result = composition.process(self.capture([base, layer]), lambda: None)
        self.assertEqual(result.getpixel((0, 0)), (128, 0, 127, 255))
        for mode in native.modes():
            with self.subTest(mode=mode):
                data = self.capture([base, layer], composition_mode=mode)
                pixels = np.asarray(composition.process(data, lambda: None))
                self.assertEqual(pixels.shape, (2, 2, 4))
                self.assertTrue(np.isfinite(pixels).all())
        self.assertEqual(base.getpixel((0, 0)), (0, 0, 255, 255))
        self.assertEqual(layer.getpixel((0, 0)), (255, 0, 0, 128))

    def test_masked_native_composition_uses_white_overlay_and_one_mask_frame(self):
        base = Image.new("RGBA", (32, 32), (0, 0, 255, 255))
        layer = Image.new("RGBA", (32, 16), (0, 255, 0, 255))
        mask = Image.new("L", layer.size, 0)
        mask.paste(255, (0, 0, 16, 16))
        data = self.capture([base, layer], composition_mask=mask, composition_mode="masked")
        result = composition.process(data, lambda: None)
        self.assertEqual(result.getpixel((0, 8)), (0, 255, 0, 255))
        self.assertEqual(result.getpixel((31, 23)), (0, 0, 255, 255))
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 255, 255))
        self.assertEqual(data["native_composition_source"]["frames"][0]["top"], 8)

    def test_queued_pixels_and_order_ignore_later_upload_mutation_and_generation_choices(self):
        base = Image.new("RGBA", (2, 2), (0, 0, 255, 255))
        layer = Image.new("RGBA", (2, 2), (255, 0, 0, 255))
        original = dict(
            composition_images=[base, layer],
            prompt="#! ignored",
            loras=object(),
            base_model_name="missing",
            native_negative_model="missing",
            scheduler=object(),
        )
        data = composition.capture(original, "image-composite")
        layer.paste((0, 255, 0, 255), (0, 0, 2, 2))
        original["composition_images"].reverse()
        result = composition.process(data, lambda: None)
        self.assertEqual(result.getpixel((0, 0)), (255, 0, 0, 255))
        self.assertEqual((data["prompt"], data["loras"], data["image_number"]), ("", [], 1))
        self.assertIs(data["_sampling_captured"], False)
        self.assertNotIn("composition_images", data)
        self.assertNotIn("native_negative_model", data)

    def test_capture_rejects_paths_oversized_aggregate_crop_and_bad_numeric_modes(self):
        image = Image.new("RGBA", (8, 8))
        for images, values in (
            (["private.png"], {}),
            ([], {}),
            ([image] * 33, {}),
            ([image], {}),
            ([image, image], dict(composition_x=8)),
            ([image, image], dict(composition_width=9)),
            ([image, image], dict(composition_mode="invented")),
            ([image, image], dict(grid_columns=True)),
            ([image, image], dict(composition_y=float("nan"))),
            ([image, image], dict(composition_mask=Image.new("L", (7, 8)))),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.capture(images, **values)
        with patch.object(composition, "MAX_INPUT_PIXELS", 64):
            with self.assertRaisesRegex(ValueError, "total"):
                self.capture([image, image])
        with self.assertRaisesRegex(ValueError, "output"):
            self.capture(
                [image] * 32,
                kind="image-grid",
                grid_columns=20,
                grid_cell_width=2048,
                grid_cell_height=2048,
            )

    def test_cancellation_or_changed_native_source_cannot_publish_metadata(self):
        image = Image.new("RGBA", (2, 2), "red")
        data = self.capture([image, image])
        stop = Mock(side_effect=CancelledError())
        with self.assertRaises(CancelledError):
            composition.process(data, stop)
        self.assertNotIn("native_composition_source", data)
        bad = replace(data[composition.KEY], native_sources=())
        with self.assertRaisesRegex(ValueError, "changed"):
            composition.process(dict(data, **{composition.KEY: bad}), lambda: None)

    def test_precision_output_keeps_the_native_float_alpha_without_preview_round_trip(self):
        base = Image.new("RGBA", (2, 2), (0, 0, 255, 127))
        layer = Image.new("RGBA", (2, 2), (255, 0, 0, 127))
        data = self.capture([base, layer], output_format="png", output_bit_depth="16-bit")
        result = composition.process(data, lambda: None)
        self.assertEqual((result.dtype, result.shape), (np.float32, (2, 2, 4)))
        expected = 127 / 255 + (1 - 127 / 255) * 127 / 255
        self.assertAlmostEqual(float(result[0, 0, 3]), expected, places=6)


if __name__ == "__main__":
    unittest.main()
