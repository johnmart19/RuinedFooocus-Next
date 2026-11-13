"""Exact selection bounds use real pinned CPU BoundingBox and ImageCropV2 bodies."""

from concurrent.futures import CancelledError
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from ruinedfooocus.features.image_processing import image_geometry_tools as geometry
from ruinedfooocus.features.image_processing.native_image_formats import ImportedImage


class ImageCropToSelectionTests(unittest.TestCase):
    def setUp(self):
        self.manager = Mock()
        self.manager.get_model_folders.side_effect = AssertionError("model lookup")
        self.source = np.arange(4 * 6 * 4, dtype=np.uint8).reshape(4, 6, 4)
        self.source[..., 3] = np.array([0, 17, 63, 127, 191, 255], dtype=np.uint8)
        self.selection = np.zeros((4, 6), dtype=np.uint8)
        # Nonrectangular and antialiased selections still describe one tight bbox.
        self.selection[1, 2] = 1
        self.selection[3, 4] = 255

    def capture(self, source=None, mask=None, **values):
        return geometry.capture_assets(
            dict(
                input_image=Image.fromarray(self.source) if source is None else source,
                mask_image=Image.fromarray(self.selection) if mask is None else mask,
                transform_operation="crop-to-mask",
                **values,
            ),
            self.manager,
        )[0]

    def test_exact_nonrectangular_selection_crops_every_rgba_channel_without_masking(self):
        original = self.source.copy()
        data = self.capture()
        result = geometry.process(None, data, self.manager, lambda: None)[0]
        np.testing.assert_array_equal(np.asarray(result), original[1:4, 2:5])
        np.testing.assert_array_equal(self.source, original)
        self.assertEqual(data[geometry.KEY].output_size, (3, 3))
        info = geometry.output_metadata(data)["image_utility_source"]
        self.assertEqual(info["native_nodes"], ["PrimitiveBoundingBox", "ImageCropV2"])
        self.assertEqual(info["crop_region"], dict(x=2, y=1, width=3, height=3))
        self.assertEqual(info["selection_sha256"], data[geometry.KEY].selection.source_sha256)
        self.assertNotIn("mask_image", data)
        self.assertEqual(self.manager.mock_calls, [])

    def test_edge_single_pixel_and_full_frame_masks_keep_inclusive_selected_pixels(self):
        for selected in ((0, 0), (3, 5), None):
            with self.subTest(selected=selected):
                mask = np.zeros((4, 6), dtype=np.uint8)
                if selected is None:
                    mask[:] = 255
                    expected = self.source
                else:
                    row, column = selected
                    mask[row, column] = 255
                    expected = self.source[row : row + 1, column : column + 1]
                data = self.capture(mask=mask)
                output = geometry.process(None, data, self.manager, lambda: None)[0]
                np.testing.assert_array_equal(np.asarray(output), expected)

    def test_capture_freezes_mask_source_bounds_and_worker_dimensions(self):
        image, mask = Image.fromarray(self.source), Image.fromarray(self.selection)
        data = self.capture(image, mask)
        identity = data[geometry.KEY].selection.source_sha256
        image.paste((255, 255, 255, 255), (0, 0, 6, 4))
        mask.paste(255, (0, 0, 6, 4))
        data.update(mask_image=mask, transform_operation="resize", transform_width=200)
        worker = geometry.worker_data(data)
        self.assertEqual((worker["custom_width"], worker["custom_height"]), (3, 3))
        self.assertNotIn("mask_image", worker)
        result = geometry.process(None, worker, self.manager, lambda: None)[0]
        np.testing.assert_array_equal(np.asarray(result), self.source[1:4, 2:5])
        self.assertEqual(worker[geometry.KEY].selection.source_sha256, identity)
        replay = geometry.replay_values(geometry.output_metadata(worker))
        self.assertEqual(replay["transform_operation"], "crop-to-mask")
        self.assertIsNone(replay["mask_image"])

    def test_uint16_source_and_mask_keep_subbyte_samples_without_lanczos(self):
        source = (self.source.astype(np.uint16) * 257 + 1).astype(np.uint16)
        data = self.capture(
            source,
            self.selection.astype(np.uint16) * 257,
            output_bit_depth="16-bit",
        )
        result = geometry.process(None, data, self.manager, lambda: None)[0]
        np.testing.assert_array_equal(result, source[1:4, 2:5].astype(np.float32) / 65535)
        self.assertEqual(data[geometry.KEY].source.dtype, "uint16")

    def test_hdr_float_crop_preserves_signed_values_and_transparency_exactly(self):
        source = self.source.astype(np.float32) / 17 - 2
        source[..., 3] = self.source[..., 3].astype(np.float32) / 255
        original = source.copy()
        imported = ImportedImage(source, "linear", "32-bit float", {})
        data = self.capture(
            imported,
            output_format="exr",
            output_bit_depth="32-bit float",
            output_color_space="linear",
        )
        result = geometry.process(None, data, self.manager, lambda: None)[0]
        np.testing.assert_array_equal(result, original[1:4, 2:5])
        np.testing.assert_array_equal(source, original)

    def test_invalid_empty_mismatched_colored_and_nonopaque_masks_are_rejected(self):
        color = np.repeat(self.selection[..., None], 3, axis=-1)
        color[1, 2, 1] = 2
        transparent = np.dstack([np.repeat(self.selection[..., None], 3, axis=-1), self.selection])
        for mask in (
            np.zeros_like(self.selection),
            np.ones((4, 7), dtype=np.uint8),
            color,
            transparent,
            np.full((4, 6), np.nan, dtype=np.float32),
            np.full((4, 6), -0.01, dtype=np.float32),
            np.full((4, 6), 1.01, dtype=np.float32),
        ):
            with self.subTest(shape=mask.shape, dtype=mask.dtype), self.assertRaises(ValueError):
                self.capture(mask=mask)
        with self.assertRaisesRegex(ValueError, "Add a grayscale"):
            geometry.capture_assets(
                dict(input_image=self.source, transform_operation="crop-to-mask"), self.manager
            )
        with self.assertRaisesRegex(ValueError, "Only Crop"):
            geometry.capture_assets(
                dict(input_image=self.source, mask_image=self.selection), self.manager
            )
        with self.assertRaisesRegex(ValueError, "bounds must fit"):
            geometry.output_size(
                (6, 4), geometry.parameters(dict(transform_operation="crop-to-mask")), (5, 3, 2, 2)
            )

    def test_late_cancel_and_native_source_change_do_not_publish_metadata(self):
        data = self.capture()
        checks = Mock(side_effect=[None, None, CancelledError()])
        with self.assertRaises(CancelledError):
            geometry.process(None, data, self.manager, checks)
        self.assertNotIn("image_utility_source", data)
        signatures = data[geometry.KEY].native_signatures
        with patch.object(geometry.native, "source_signatures", side_effect=[signatures, ()]):
            with self.assertRaisesRegex(ValueError, "source changed during"):
                geometry.process(None, data, self.manager, lambda: None)
        self.assertNotIn("image_utility_source", data)


if __name__ == "__main__":
    unittest.main()
