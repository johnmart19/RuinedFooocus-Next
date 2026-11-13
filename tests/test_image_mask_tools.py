"""Actual pinned CPU mask bodies, soft selection frames and immutable requests."""

from concurrent.futures import CancelledError
from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ruinedfooocus.features.image_processing import image_mask_tools as masks
from ruinedfooocus.features.image_processing import mask_native as native


class ImageMaskToolsTests(unittest.TestCase):
    def capture(self, operation="grow", *, source=None, mask=None, reference=None, **options):
        data = dict(
            input_image=source if source is not None else Image.new("RGB", (5, 5)),
            mask_operation=operation,
        )
        data.update(options)
        if operation not in ("channel", "color"):
            data["mask_image"] = mask if mask is not None else Image.new("L", (5, 5), 128)
        if reference is not None:
            data["mask_reference"] = reference
        captured, assets = masks.capture_assets(data, None)
        self.assertEqual(assets, ())
        return captured

    def run_mask(self, data, cancel=lambda: None):
        return np.asarray(masks.process(None, data, None, cancel)[0])

    def test_real_native_grow_taper_and_shrink_keep_soft_values(self):
        mask = np.zeros((5, 5), dtype=np.float32)
        mask[2, 2] = 0.4
        for tapered, count in ((True, 5), (False, 9)):
            data = self.capture(
                mask=mask, mask_expand=1, mask_tapered_corners=tapered, output_bit_depth="16-bit"
            )
            result = self.run_mask(data)[..., 0]
            self.assertEqual(np.count_nonzero(result), count)
            self.assertEqual(float(result.max()), float(mask.max()))
            self.assertEqual(data["image_utility_source"]["native_node"], "GrowMask")
        full = np.full((5, 5), 0.7, dtype=np.float32)
        full[0] = full[-1] = 0
        full[:, 0] = full[:, -1] = 0
        data = self.capture(mask=full, mask_expand=-1, output_bit_depth="16-bit")
        self.assertEqual(np.count_nonzero(self.run_mask(data)[..., 0]), 1)

    def test_native_feather_has_independent_frame_edges(self):
        data = self.capture("feather", mask=Image.new("L", (5, 5), 255), mask_left=2, mask_top=2)
        result = self.run_mask(data)[..., 0]
        self.assertEqual(int(result[0, 0]), 64)
        self.assertEqual(int(result[2, 0]), 128)
        self.assertEqual(int(result[2, 2]), 255)

    def test_all_native_composite_modes_keep_frame_and_soft_arithmetic(self):
        a = np.full((5, 5), 0.25, dtype=np.float32)
        b = np.full((5, 5), 0.75, dtype=np.float32)
        expected = dict(multiply=0.1875, add=1, subtract=0, **{"and": 0, "or": 1, "xor": 1})
        for mode in masks.COMPOSITE_OPERATIONS:
            with self.subTest(mode=mode):
                data = self.capture(
                    "composite",
                    mask=a,
                    reference=b,
                    mask_composite_operation=mode,
                    output_bit_depth="16-bit",
                )
                np.testing.assert_allclose(self.run_mask(data), expected[mode])

    def test_channel_extraction_retains_uint16_precision_and_float_alpha(self):
        image = np.zeros((5, 5, 4), dtype=np.uint16)
        image[..., 0] = 10001
        image[..., 3] = 12345
        for channel, value in (("red", 10001), ("alpha", 12345)):
            data = self.capture(
                "channel", source=image, mask_channel=channel, output_bit_depth="16-bit"
            )
            np.testing.assert_allclose(self.run_mask(data), value / 65535, atol=1e-8)
        floating = image.astype(np.float32) / 65535
        floating[..., 3] = 0.1234567
        data = self.capture(
            "channel", source=floating, output_format="exr", output_bit_depth="32-bit float"
        )
        np.testing.assert_array_equal(self.run_mask(data)[..., 0], floating[..., 3])

    def test_color_key_uses_native_rgb8_matching_and_ignores_alpha(self):
        image = Image.new("RGBA", (5, 5), (1, 2, 3, 0))
        image.putpixel((0, 0), (4, 5, 6, 255))
        data = self.capture("color", source=image, mask_color="#010203")
        result = self.run_mask(data)
        self.assertEqual(int(result[0, 0, 0]), 0)
        self.assertEqual(int(result[1, 1, 0]), 255)

    def test_threshold_is_native_strict_greater_and_inversion_is_explicit(self):
        values = np.full((5, 5), 0.5, dtype=np.float32)
        values[0, 0] = 0.6
        data = self.capture("threshold", mask=values, output_bit_depth="16-bit")
        result = self.run_mask(data)
        self.assertEqual(np.count_nonzero(result[..., 0]), 1)
        inverted = self.capture(
            "threshold", mask=values, mask_invert=True, output_bit_depth="16-bit"
        )
        self.assertFalse(self.run_mask(inverted).any())
        invert = self.capture("invert", mask=values, output_bit_depth="16-bit")
        np.testing.assert_allclose(self.run_mask(invert)[..., 0], 1 - values)

    def test_mask_input_preserves_grayscale_pixels_and_never_uses_alpha_as_selection(self):
        rgba = Image.new("RGBA", (5, 5), (51, 51, 51, 0))
        result = self.run_mask(self.capture(mask=rgba))
        np.testing.assert_array_equal(result, 51)

    def test_queued_pixels_parameters_and_metadata_ignore_later_mutation(self):
        mask = Image.new("L", (5, 5), 51)
        data = self.capture(mask=mask)
        mask.paste(255, (0, 0, 5, 5))
        data["mask_expand"] = 999
        np.testing.assert_array_equal(self.run_mask(data), 51)
        metadata = masks.output_metadata(data)
        metadata["image_utility_source"]["native_node"] = "changed"
        self.assertEqual(data["image_utility_source"]["native_node"], "GrowMask")
        with self.assertRaises(FrozenInstanceError):
            data[masks.KEY].mask = None

    def test_worker_transport_clears_generation_and_raw_pixels_but_keeps_lifecycle(self):
        from ruinedfooocus.features.gallery.editor_session import KEY as EDITOR_KEY

        data = self.capture()
        data.update(
            prompt="irrelevant",
            image_utility_source={"stale": True},
            loras=["bad"],
            mask_image=object(),
            color_reference=object(),
            unexpected=object(),
            _job=object(),
        )
        data[EDITOR_KEY] = object()
        worker = masks.worker_data(data)
        self.assertEqual((worker["prompt"], worker["loras"], worker["image_number"]), ("", [], 1))
        for field in ("mask_image", "color_reference", "unexpected", "image_utility_source"):
            self.assertNotIn(field, worker)
        self.assertIs(worker[EDITOR_KEY], data[EDITOR_KEY])
        self.assertIs(worker["_job"], data["_job"])

    def test_replay_restores_scalars_and_clears_all_mask_sources(self):
        data = self.capture("feather", mask_left=2)
        self.run_mask(data)
        replay = masks.replay_values(masks.output_metadata(data))
        self.assertEqual(replay["mask_left"], 2)
        for name in (*masks.IMAGE_FIELDS, "input_image", "image_source"):
            self.assertIsNone(replay[name])

    def test_rejects_mismatched_frames_color_masks_and_missing_inputs(self):
        for operation, options in (
            ("grow", dict(mask=Image.new("L", (4, 5)))),
            ("grow", dict(mask=Image.new("RGB", (5, 5), (255, 0, 0)))),
            ("grow", dict(mask=np.full((5, 5), 1.1, dtype=np.float32), output_bit_depth="16-bit")),
            ("composite", {}),
            ("composite", dict(reference=Image.new("L", (4, 5)))),
            ("channel", dict(source=np.zeros((5, 5, 3), dtype=np.uint8))),
            ("color", dict(source=np.zeros((5, 5, 1), dtype=np.uint8))),
        ):
            with (
                self.subTest(operation=operation, options=list(options)),
                self.assertRaises(ValueError),
            ):
                self.capture(operation, **options)
        with self.assertRaisesRegex(ValueError, "Add a grayscale"):
            masks.capture_assets(dict(input_image=Image.new("RGB", (5, 5))), None)

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
        worker = masks.worker_data(data)
        recipe["prompt"] = "changed"
        recipe["loras"][0][1] = 1
        retained = worker[SOURCE_GENERATION]
        self.assertEqual(retained["prompt"], "source recipe")
        self.assertEqual(retained["loras"], [["style.safetensors", 0.5]])
        for name in ("input_image", "image_source", "model_recipe_path"):
            self.assertNotIn(name, retained)
        attached = attach_source_generation(worker, {"controlnet": {"type": masks.KINDS[0]}})
        self.assertEqual(attached["kind"], "processing")
        self.assertEqual(attached["source_generation"]["prompt"], "source recipe")

    def test_rejects_scalar_type_coercion_invalid_precision_and_unrelated_conditioning(self):
        for options in (
            dict(mask_expand=True),
            dict(mask_expand=float("nan")),
            dict(mask_left=0.5),
            dict(mask_invert=1),
            dict(mask_color="red"),
            dict(mask_operation={}),
            dict(mask_threshold=float("inf")),
            dict(inpaint_toggle=True),
            dict(output_format="avif"),
            dict(output_color_space="HDR"),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.capture(**options)
        with self.assertRaisesRegex(ValueError, "Precision masks"):
            self.capture(mask=np.full((5, 5), 0.12345, dtype=np.float32))

    def test_cancellation_and_stale_native_capture_never_publish_metadata(self):
        data = self.capture(mask_expand=4)
        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            if calls == 3:
                raise CancelledError()

        with self.assertRaises(CancelledError):
            self.run_mask(data, cancel)
        self.assertNotIn("image_utility_source", data)
        with patch.object(native, "source_signatures", return_value=()):
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.run_mask(data)
        self.assertNotIn("image_utility_source", data)

    def test_missing_capture_and_invalid_native_result_are_rejected(self):
        with self.assertRaises(ValueError):
            self.run_mask({})
        data = self.capture()
        ops = native.operations(data[masks.KEY].native_signatures)
        ops.MaskToImage = lambda *args: (np.zeros((5, 5)),)
        with patch.object(native, "operations", return_value=ops):
            with self.assertRaisesRegex(ValueError, "invalid selection"):
                self.run_mask(data)

    def test_precision_reference_and_frozen_export_choice_cannot_be_downgraded(self):
        reference = np.full((5, 5), 0.1234567, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Precision masks"):
            self.capture("composite", reference=reference)
        data = self.capture("composite", reference=reference, output_bit_depth="16-bit")
        data["output_bit_depth"] = "8-bit"
        worker = masks.worker_data(data)
        self.assertEqual(worker["output_bit_depth"], "16-bit")
        result = self.run_mask(data)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(data["output_bit_depth"], "16-bit")
        self.assertEqual(masks.output_metadata(data)["output_bit_depth"], "16-bit")

    def test_final_cancellation_leaves_no_publishable_metadata(self):
        data = self.capture()
        count = 0

        def cancel():
            nonlocal count
            count += 1
            if count == 3:
                raise CancelledError()

        with self.assertRaises(CancelledError):
            self.run_mask(data, cancel)
        self.assertNotIn("image_utility_source", data)


if __name__ == "__main__":
    unittest.main()
