"""Pinned CPU color algorithms, palette semantics, alpha and precision capture."""

from concurrent.futures import CancelledError
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ruinedfooocus.features.image_processing import image_color_tools as colors
from ruinedfooocus.features.image_processing import color_native as native
from ruinedfooocus.features.image_processing.native_image_formats import ImportedImage


class ImageColorToolsTests(unittest.TestCase):
    def capture(self, operation="blend", *, source=None, reference=None, **options):
        data = dict(
            input_image=source
            if source is not None
            else Image.new("RGBA", (5, 5), (64, 128, 192, 51)),
            color_operation=operation,
            **options,
        )
        if operation != "quantize":
            data["color_reference"] = (
                reference if reference is not None else Image.new("RGB", (5, 5), (192, 64, 128))
            )
        elif reference is not None:
            data["color_reference"] = reference
        result, assets = colors.capture_assets(data, None)
        self.assertEqual(assets, ())
        return result

    def run_color(self, data, cancel=lambda: None):
        return np.asarray(colors.process(None, data, None, cancel)[0])

    def test_all_native_blend_modes_and_factor_preserve_source_alpha(self):
        a = np.full((5, 5, 4), 0.25, dtype=np.float32)
        a[..., 3] = 0.1234567
        b = np.full((5, 5, 3), 0.75, dtype=np.float32)
        expected = dict(
            normal=0.75,
            multiply=0.1875,
            screen=0.8125,
            overlay=0.375,
            soft_light=0.375,
            difference=0,
        )
        for mode in colors.BLEND_MODES:
            with self.subTest(mode=mode):
                data = self.capture(
                    source=a,
                    reference=b,
                    color_blend_mode=mode,
                    color_blend_factor=1,
                    output_bit_depth="16-bit",
                )
                result = self.run_color(data)
                np.testing.assert_allclose(result[..., :3], expected[mode])
                np.testing.assert_array_equal(result[..., 3], a[..., 3])
        data = self.capture(
            source=a, reference=b, color_blend_factor=0.25, output_bit_depth="16-bit"
        )
        np.testing.assert_allclose(self.run_color(data)[..., :3], 0.375)

    def test_reference_alpha_does_not_promote_rgb_or_select_blend_pixels(self):
        source = np.full((5, 5, 3), 0.25, dtype=np.float32)
        reference = np.full((5, 5, 4), 0.75, dtype=np.float32)
        reference[..., 3] = 0
        result = self.run_color(
            self.capture(source=source, reference=reference, output_bit_depth="16-bit")
        )
        self.assertEqual(result.shape, (5, 5, 3))
        np.testing.assert_allclose(result, 0.5)

    def test_real_transfer_methods_allow_independent_reference_frame_and_preserve_alpha(self):
        source = np.full((5, 5, 4), (0.2, 0.3, 0.4, 0.1234567), dtype=np.float32)
        reference = np.full((3, 7, 3), (0.6, 0.5, 0.4), dtype=np.float32)
        for method in colors.TRANSFER_METHODS:
            with self.subTest(method=method):
                data = self.capture(
                    "transfer",
                    source=source,
                    reference=reference,
                    color_transfer_method=method,
                    output_bit_depth="16-bit",
                )
                result = self.run_color(data)
                self.assertEqual(result.shape, source.shape)
                self.assertTrue(np.isfinite(result).all())
                np.testing.assert_array_equal(result[..., 3], source[..., 3])
                np.testing.assert_allclose(
                    result[..., :3],
                    np.broadcast_to(reference[0, 0], result[..., :3].shape),
                    atol=1 / 255 + 1e-5,
                )
                self.assertEqual(data["image_utility_source"]["reference_size"], [7, 3])

    def test_zero_transfer_strength_is_identity(self):
        source = np.linspace(0, 1, 75, dtype=np.float32).reshape(5, 5, 3)
        data = self.capture(
            "transfer", source=source, color_transfer_strength=0, output_bit_depth="16-bit"
        )
        np.testing.assert_array_equal(self.run_color(data), source)

    def test_every_native_dither_uses_palette_color_count_and_preserves_float_alpha(self):
        source = np.linspace(0, 1, 100, dtype=np.float32).reshape(5, 5, 4)
        source[..., 3] = 0.1234567
        for dither in colors.DITHERS:
            with self.subTest(dither=dither):
                data = self.capture(
                    "quantize",
                    source=source,
                    color_quantize_colors=3,
                    color_quantize_dither=dither,
                    output_bit_depth="16-bit",
                )
                result = self.run_color(data)
                self.assertLessEqual(len(np.unique(result[..., :3].reshape(-1, 3), axis=0)), 3)
                np.testing.assert_array_equal(result[..., 3], source[..., 3])
                np.testing.assert_allclose(
                    result[..., :3] * 255, (result[..., :3] * 255).round(), atol=1e-5
                )
                self.assertIn("uint8", data["image_utility_source"]["output_precision"])

    def test_uint16_blend_keeps_distinctions_below_byte_precision(self):
        source = np.full((5, 5, 3), 10001, dtype=np.uint16)
        reference = np.full((5, 5, 3), 10003, dtype=np.uint16)
        result = self.run_color(
            self.capture(source=source, reference=reference, output_bit_depth="16-bit")
        )
        np.testing.assert_allclose(result, 10002 / 65535, atol=1e-8)

    def test_native_hdr_blend_clamp_is_explicit_and_source_alpha_stays_exact(self):
        source = np.full((5, 5, 4), 2.0, dtype=np.float32)
        source[..., 3] = 0.2345678
        reference = np.full((5, 5, 3), 1.5, dtype=np.float32)
        source = ImportedImage(source, "HDR", "32-bit float", {})
        reference = ImportedImage(reference, "HDR", "32-bit float", {})
        data = self.capture(
            source=source,
            reference=reference,
            output_format="exr",
            output_bit_depth="32-bit float",
            output_color_space="HDR",
        )
        result = self.run_color(data)
        np.testing.assert_array_equal(result[..., :3], 1)
        np.testing.assert_array_equal(result[..., 3], source.pixels[..., 3])
        self.assertIn("clamp", data["image_utility_source"]["output_precision"])

    def test_queued_images_and_parameters_are_owned_and_never_mutate_sources(self):
        source = Image.new("RGBA", (5, 5), (20, 40, 60, 123))
        reference = Image.new("RGB", (5, 5), (60, 80, 100))
        data = self.capture(source=source, reference=reference)
        source.paste((255, 0, 0, 255), (0, 0, 5, 5))
        reference.paste((255, 255, 255), (0, 0, 5, 5))
        data["color_blend_factor"] = 0
        result = self.run_color(data)
        np.testing.assert_array_equal(result[0, 0], [40, 60, 80, 123])
        np.testing.assert_array_equal(np.asarray(source)[0, 0], [255, 0, 0, 255])

    def test_worker_prunes_raw_inputs_unknown_generation_state_and_preserves_ticket(self):
        from ruinedfooocus.features.gallery.editor_session import KEY as EDITOR_KEY

        data = self.capture()
        data.update(prompt="ignored", color_reference=object(), unexpected=object(), loras=["bad"])
        data[EDITOR_KEY] = object()
        worker = colors.worker_data(data)
        self.assertEqual((worker["prompt"], worker["loras"], worker["image_number"]), ("", [], 1))
        self.assertNotIn("color_reference", worker)
        self.assertNotIn("unexpected", worker)
        self.assertIs(worker[EDITOR_KEY], data[EDITOR_KEY])

    def test_metadata_is_detached_and_replay_restores_options_without_images(self):
        data = self.capture(color_blend_mode="screen")
        self.run_color(data)
        metadata = colors.output_metadata(data)
        metadata["image_utility_source"]["reference_size"][0] = 999
        self.assertEqual(data["image_utility_source"]["reference_size"], [5, 5])
        replay = colors.replay_values(metadata)
        self.assertEqual(replay["color_blend_mode"], "screen")
        for name in ("input_image", "image_source", "color_reference"):
            self.assertIsNone(replay[name])

    def test_rejects_missing_reference_mismatched_blend_frames_and_raw_paths(self):
        for options in (dict(reference=Image.new("RGB", (4, 5))), dict(reference="private.png")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.capture(**options)
        with self.assertRaisesRegex(ValueError, "Add a reference"):
            colors.capture_assets(dict(input_image=Image.new("RGB", (5, 5))), None)
        with self.assertRaises(ValueError):
            self.capture("quantize", reference=Image.new("RGB", (5, 5)))

    def test_rejects_invalid_scalar_types_precision_and_implicit_color_conversion(self):
        for options in (
            dict(color_blend_factor=True),
            dict(color_quantize_colors=3.5),
            dict(color_transfer_strength=float("inf")),
            dict(color_blend_mode={}),
            dict(inpaint_toggle=True),
            dict(output_format="avif"),
            dict(source=np.zeros((5, 5, 3), dtype=np.uint16)),
            dict(
                source=ImportedImage(
                    np.zeros((5, 5, 3), dtype=np.float32), "linear", "32-bit float", {}
                )
            ),
        ):
            with self.subTest(options=list(options)), self.assertRaises(ValueError):
                self.capture(**options)
        for operation in ("transfer", "quantize"):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                self.capture(
                    operation,
                    source=np.full((5, 5, 3), 2, dtype=np.float32),
                    output_bit_depth="16-bit",
                )

    def test_cancellation_and_stale_sources_do_not_publish_metadata(self):
        data = self.capture()

        def cancel():
            raise CancelledError()

        with self.assertRaises(CancelledError):
            self.run_color(data, cancel)
        self.assertNotIn("image_utility_source", data)
        with patch.object(native, "source_signatures", return_value=()):
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.run_color(data)
        self.assertNotIn("image_utility_source", data)

    def test_missing_capture_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_color({})

    def test_worker_preserves_detached_source_recipe_without_image_or_path_authority(self):
        from ruinedfooocus.features.generation.request_metadata import (
            SOURCE_GENERATION,
            attach_source_generation,
        )

        data = self.capture()
        recipe = dict(
            prompt="source recipe",
            base_model_name="source.safetensors",
            seed=17,
            loras=[["style.safetensors", 0.5]],
            input_image=Image.new("RGB", (5, 5)),
            image_source={"path": "private.png"},
            model_recipe_path="private.json",
        )
        data[SOURCE_GENERATION] = recipe
        worker = colors.worker_data(data)
        recipe["prompt"] = "changed"
        recipe["loras"][0][1] = 1
        retained = worker[SOURCE_GENERATION]
        self.assertEqual(retained["prompt"], "source recipe")
        self.assertEqual(retained["loras"], [["style.safetensors", 0.5]])
        for name in ("input_image", "image_source", "model_recipe_path"):
            self.assertNotIn(name, retained)
        attached = attach_source_generation(worker, {"controlnet": {"type": colors.KINDS[0]}})
        self.assertEqual(attached["kind"], "processing")
        self.assertEqual(attached["source_generation"]["prompt"], "source recipe")

    def test_frozen_export_choice_keeps_precision_and_replay(self):
        source = np.full((5, 5, 3), 10001, dtype=np.uint16)
        data = self.capture(source=source, output_bit_depth="16-bit")
        data["output_bit_depth"] = "8-bit"
        self.assertEqual(colors.worker_data(data)["output_bit_depth"], "16-bit")
        self.assertEqual(self.run_color(data).dtype, np.float32)
        metadata = colors.output_metadata(data)
        self.assertEqual(metadata["output_bit_depth"], "16-bit")
        self.assertEqual(colors.replay_values(metadata)["output_bit_depth"], "16-bit")

    def test_final_cancellation_and_nonfinite_native_result_never_publish_metadata(self):
        data = self.capture()
        count = 0

        def cancel():
            nonlocal count
            count += 1
            if count == 3:
                raise CancelledError()

        with self.assertRaises(CancelledError):
            self.run_color(data, cancel)
        self.assertNotIn("image_utility_source", data)
        import torch
        from ruinedfooocus.features.image_processing.utility_device import UtilityPolicy

        policy = UtilityPolicy(
            torch.device("cpu"), torch.device("cpu"), torch.float32, requested="cpu"
        )
        ops = native.operations(data[colors.KEY].native_signatures, lambda: None, policy=policy)
        ops.Blend.execute = lambda *args: (torch.full((1, 5, 5, 4), float("nan")),)
        with patch.object(native, "operations", return_value=ops):
            with self.assertRaisesRegex(ValueError, "invalid dimensions"):
                self.run_color(data)
        self.assertNotIn("image_utility_source", data)


if __name__ == "__main__":
    unittest.main()
