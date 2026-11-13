"""Real pinned CPU geometry, precision/alpha capture and native-source fences."""

from concurrent.futures import CancelledError
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from ruinedfooocus.features.image_processing import image_geometry_tools as geometry
from ruinedfooocus.features.image_processing.native_image_formats import ImportedImage


class ImageGeometryToolTests(unittest.TestCase):
    def setUp(self):
        self.manager = Mock()
        self.manager.get_model_folders.side_effect = AssertionError("model lookup")
        self.source = np.arange(3 * 5 * 4, dtype=np.uint8).reshape(3, 5, 4)

    def capture(self, source=None, **values):
        return geometry.capture_assets(
            dict(input_image=Image.fromarray(self.source) if source is None else source, **values),
            self.manager,
        )[0]

    def run_image(self, source=None, **values):
        data = self.capture(source, **values)
        result = geometry.process(None, data, self.manager, lambda: None)[0]
        return np.asarray(result), data

    def test_native_crop_rotate_flip_keep_exact_rgba_pixels(self):
        cropped, _ = self.run_image(
            transform_operation="crop",
            transform_x=1,
            transform_y=1,
            transform_width=3,
            transform_height=2,
        )
        np.testing.assert_array_equal(cropped, self.source[1:3, 1:4])
        for angle, k in (("none", 0), ("90 degrees", -1), ("180 degrees", 2), ("270 degrees", 1)):
            with self.subTest(rotation=angle):
                rotated, _ = self.run_image(transform_operation="rotate", transform_rotation=angle)
                np.testing.assert_array_equal(rotated, np.rot90(self.source, k))
        for choice, axis in (("horizontal", 1), ("vertical", 0)):
            with self.subTest(flip=choice):
                flipped, _ = self.run_image(transform_operation="flip", transform_flip=choice)
                np.testing.assert_array_equal(flipped, np.flip(self.source, axis=axis))

    def test_identity_and_contain_resize_preserve_the_frame_and_default_has_no_resize(self):
        identity, data = self.run_image()
        np.testing.assert_array_equal(identity, self.source)
        self.assertEqual(data[geometry.KEY].output_size, (5, 3))
        enlarged, data = self.run_image(
            transform_width=10, transform_height=10, transform_interpolation="nearest-exact"
        )
        self.assertEqual(data[geometry.KEY].output_size, (10, 6))
        np.testing.assert_array_equal(enlarged, self.source.repeat(2, 0).repeat(2, 1))
        auto, _ = self.run_image(transform_height=6, transform_interpolation="nearest-exact")
        np.testing.assert_array_equal(auto, enlarged)

    def test_explicit_stretch_and_center_crop_have_different_results(self):
        stretch, _ = self.run_image(
            transform_width=3,
            transform_height=3,
            transform_fit="stretch",
            transform_interpolation="nearest-exact",
        )
        crop, _ = self.run_image(
            transform_width=3,
            transform_height=3,
            transform_fit="center-crop",
            transform_interpolation="nearest-exact",
        )
        self.assertEqual(stretch.shape, (3, 3, 4))
        np.testing.assert_array_equal(crop, self.source[:, 1:4])
        self.assertFalse(np.array_equal(crop, stretch))

    def test_native_transparent_color_and_mirror_padding_preserve_original_alpha(self):
        transparent, _ = self.run_image(
            transform_operation="pad", transform_left=2, transform_bottom=1
        )
        self.assertEqual(transparent.shape, (4, 7, 4))
        np.testing.assert_array_equal(transparent[:3, 2:], self.source)
        self.assertTrue((transparent[:, :2, 3] == 0).all())
        self.assertTrue((transparent[3, :, 3] == 0).all())
        colored, _ = self.run_image(
            transform_operation="pad",
            transform_right=1,
            transform_padding_style="color",
            transform_padding_color="#12ab34",
        )
        np.testing.assert_array_equal(colored[:, :5], self.source)
        np.testing.assert_array_equal(colored[:, 5], np.tile([18, 171, 52, 255], (3, 1)))
        mirrored, _ = self.run_image(
            transform_operation="pad", transform_left=1, transform_padding_style="mirror"
        )
        np.testing.assert_array_equal(mirrored[:, 1:], self.source)
        np.testing.assert_array_equal(mirrored[:, 0], self.source[:, 1])

    def test_capture_freezes_pixels_parameters_and_output_plan(self):
        image = Image.fromarray(self.source)
        data = self.capture(image, transform_operation="flip")
        image.paste((255, 255, 255, 255), (0, 0, 5, 3))
        data["transform_flip"] = "vertical"
        result = geometry.process(None, data, self.manager, lambda: None)[0]
        np.testing.assert_array_equal(np.asarray(result), self.source[:, ::-1])
        metadata = geometry.output_metadata(data)
        self.assertEqual(metadata["transform_flip"], "horizontal")
        self.assertEqual(metadata["image_utility_source"]["native_node"], "ImageFlip")
        self.assertNotIn("base_model_name", metadata)
        self.assertEqual(self.manager.mock_calls, [])

    def test_png16_preserves_uint16_precision_and_interpolation_has_no_byte_roundtrip(self):
        source = np.array([[[1, 32769, 65534, 43123], [12345, 2, 65535, 0]]], dtype=np.uint16)
        output, data = self.run_image(source, transform_operation="flip", output_bit_depth="16-bit")
        self.assertEqual(output.dtype, np.float32)
        np.testing.assert_allclose(
            output, source[:, ::-1].astype(np.float32) / 65535, atol=0, rtol=0
        )
        output, _ = self.run_image(
            source,
            transform_width=4,
            transform_interpolation="nearest-exact",
            output_bit_depth="16-bit",
        )
        np.testing.assert_allclose(
            output, (source.astype(np.float32) / 65535).repeat(2, 0).repeat(2, 1), atol=0, rtol=0
        )
        self.assertEqual(data[geometry.KEY].source.dtype, "uint16")

    def test_float_exr_samples_keep_out_of_range_values_and_exact_color_space(self):
        source = np.array([[[-0.25, 1.25, 0.123456, 0.5], [4, 0, -0.1, 1]]], dtype=np.float32)
        imported = ImportedImage(source, "linear", "32-bit float", {})
        output, _ = self.run_image(
            imported,
            transform_operation="flip",
            output_format="exr",
            output_color_space="linear",
            output_bit_depth="32-bit float",
        )
        np.testing.assert_array_equal(output, source[:, ::-1])
        with self.assertRaisesRegex(ValueError, "match the source"):
            self.capture(imported, output_format="exr")
        with self.assertRaisesRegex(ValueError, "Out-of-range"):
            self.capture(source, output_bit_depth="16-bit")
        with self.assertRaisesRegex(ValueError, "PNG16 or EXR"):
            self.capture(source)

    def test_rejects_unsafe_working_copy_invalid_pixels_and_unsupported_native_lanczos(self):
        for values in (
            dict(transform_width=True),
            dict(transform_x=0.5),
            dict(transform_width=float("nan")),
            dict(transform_width=16384, transform_height=16384, transform_fit="stretch"),
            dict(transform_operation="crop", transform_x=5),
            dict(transform_operation="crop", transform_width=6),
            dict(transform_operation="pad", transform_left=16384),
            dict(transform_interpolation="lanczos"),
            dict(transform_padding_color="#GG0011"),
            dict(output_format="avif"),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.capture(**values)
        with self.assertRaisesRegex(ValueError, "8-bit working copy"):
            self.capture(
                image_source=dict(original_dtype="uint16", conversion="native working copy")
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            self.capture(np.full((2, 2, 3), np.nan, dtype=np.float32))

    def test_cancel_and_native_source_change_never_publish_provenance(self):
        data = self.capture()
        cancel = Mock(side_effect=CancelledError())
        with self.assertRaises(CancelledError):
            geometry.process(None, data, self.manager, cancel)
        self.assertNotIn("image_utility_source", data)
        with patch.object(geometry.native, "source_signatures", return_value=()):
            with self.assertRaisesRegex(ValueError, "source changed"):
                geometry.process(None, data, self.manager, lambda: None)
        self.assertNotIn("image_utility_source", data)
        with patch.object(geometry.native, "operations") as operations:
            operations.return_value.resize.return_value = (np.zeros((1, 1, 1, 4)),)
            with self.assertRaises((AttributeError, ValueError)):
                geometry.process(None, data, self.manager, lambda: None)

    def test_replay_clears_source_and_generation_intent(self):
        values = geometry.replay_values(
            dict(transform_operation="crop", transform_width=3, output_bit_depth="16-bit")
        )
        self.assertEqual(values["transform_operation"], "crop")
        self.assertEqual(values["transform_width"], 3)
        self.assertIsNone(values["input_image"])
        self.assertEqual(values["cn_selection"], "Transform image")
        self.assertEqual(values["automatic_upscale"], "Off")
        self.assertFalse(values["inpaint_toggle"])
        self.assertEqual(values["output_bit_depth"], "16-bit")

    def test_output_precision_and_color_settings_are_immutable_across_worker_processing(self):
        source = np.array([[[1, 32769, 65534, 43123], [12345, 2, 65535, 0]]], dtype=np.uint16)
        data = self.capture(source, transform_operation="flip", output_bit_depth="16-bit")
        data.update(
            output_bit_depth="8-bit",
            output_format="avif",
            output_color_space="HDR",
            private_extra_pixels=source.copy(),
        )
        worker = geometry.worker_data(data)
        self.assertEqual(worker["output_format"], "png")
        self.assertEqual(worker["output_bit_depth"], "16-bit")
        self.assertEqual(worker["output_color_space"], "sRGB")
        self.assertNotIn("private_extra_pixels", worker)
        # Even a direct caller after capture cannot silently downgrade the frozen export.
        result = geometry.process(None, data, self.manager, lambda: None)[0]
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(data["output_format"], "png")
        metadata = geometry.output_metadata(data)
        self.assertEqual(metadata["output_bit_depth"], "16-bit")
        self.assertEqual(geometry.replay_values(metadata)["output_bit_depth"], "16-bit")
        np.testing.assert_array_equal(result, source[:, ::-1].astype(np.float32) / 65535)

    def test_late_cancel_does_not_publish_metadata(self):
        data = self.capture(transform_operation="crop", transform_x=1)
        checks = Mock(side_effect=[None, None, CancelledError()])
        with self.assertRaises(CancelledError):
            geometry.process(None, data, self.manager, checks)
        self.assertNotIn("image_utility_source", data)

    def test_worker_data_uses_actual_output_plan_and_clears_model_sampling(self):
        data = self.capture(
            transform_operation="rotate",
            transform_rotation="90 degrees",
            prompt="ignored",
            negative="ignored",
            image_number=8,
            base_model_name="missing.safetensors",
            automatic_upscale="missing",
        )
        captured = geometry.worker_data(data)
        self.assertEqual((captured["custom_width"], captured["custom_height"]), (3, 5))
        self.assertEqual(captured["prompt"], "")
        self.assertEqual(captured["negative"], "")
        self.assertEqual(captured["image_number"], 1)
        self.assertEqual(captured["automatic_upscale"], "Off")
        self.assertNotIn("native_vision_model", captured)
        self.assertIs(captured[geometry.KEY], data[geometry.KEY])

    def test_worker_keeps_bounded_source_generation_without_raw_images_or_paths(self):
        from ruinedfooocus.features.generation.request_metadata import (
            SOURCE_GENERATION,
            attach_source_generation,
        )

        data = self.capture(transform_operation="flip")
        original = dict(
            prompt="original source recipe",
            negative="source negative",
            seed=123,
            base_model_name="source.safetensors",
            input_image=Image.fromarray(self.source),
            image_source=dict(filename="private-source.png"),
            private_pixels=self.source,
            model_recipe_path="private-recipe.json",
        )
        data[SOURCE_GENERATION] = original
        worker = geometry.worker_data(data)
        provenance = worker[SOURCE_GENERATION]
        self.assertEqual(provenance["prompt"], "original source recipe")
        self.assertEqual(provenance["seed"], 123)
        self.assertEqual(provenance["base_model_name"], "source.safetensors")
        for name in ("input_image", "image_source", "private_pixels", "model_recipe_path"):
            self.assertNotIn(name, provenance)
        original["seed"] = 999
        self.assertEqual(provenance["seed"], 123)
        geometry.process(None, worker, self.manager, lambda: None)
        metadata = attach_source_generation(worker, geometry.output_metadata(worker))
        self.assertEqual(metadata["kind"], "processing")
        self.assertEqual(metadata["source_generation"]["seed"], 123)


if __name__ == "__main__":
    unittest.main()
