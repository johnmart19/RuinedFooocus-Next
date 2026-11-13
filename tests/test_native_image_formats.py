"""Real pinned native codec round trips on tiny CPU images; no weight loading."""

import ast
import asyncio
from fractions import Fraction
import io
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import zlib

import av
from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc
import numpy as np
from PIL import Image
from PIL.Image import Exif
import torch
import gradio as gr
from gradio.state_holder import SessionState

from ruinedfooocus.features.image_processing import native_image_formats as formats

ROOT = Path(__file__).resolve().parents[1]


def native_preview():
    spec = importlib.util.spec_from_file_location(
        "comfy_execution.preview_tonemap",
        ROOT / "repositories/ComfyUI/comfy_execution/preview_tonemap.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_codecs():
    """Execute unchanged native codec/color routines without importing the GPU node registry."""
    source = ROOT / "repositories/ComfyUI/comfy_extras/nodes_images.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    selected = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    selected += [
        node
        for node in tree.body
        if node.lineno >= 1185 and isinstance(node, (ast.Assign, ast.FunctionDef))
    ]
    selected += [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ImageColorSpace"
    ]
    namespace = dict(globals())
    namespace["IO"] = SimpleNamespace(ComfyNode=object, NodeOutput=lambda *values: values)
    namespace["comfy"] = SimpleNamespace(
        model_management=SimpleNamespace(
            intermediate_device=lambda: "cpu", intermediate_dtype=lambda: torch.float32
        )
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    return SimpleNamespace(**namespace)


class NativeImageFormatsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = native_codecs()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rf-native-image-formats-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.enterContext(patch.object(formats, "_native", return_value=self.native))

    def write(self, pixels, kind="png", depth="16-bit", color="sRGB"):
        path = self.root / f"image-{uuid.uuid4().hex}.{kind}"
        values = dict(output_format=kind, output_bit_depth=depth, output_color_space=color)
        return formats.write_image(path, pixels, values, {"Prompt": "весна", "seed": 7})

    def test_png16_rgba_preserves_sub_byte_precision_alpha_and_metadata(self):
        pixels = np.array([[[0.12345, 0.34567, 0.56789, 0.33333], [1, 0, 0.75, 0]]], np.float32)
        path = self.write(pixels)
        imported = formats.read_image(path)
        self.assertEqual(imported.pixels.dtype, np.uint16)
        np.testing.assert_array_equal(imported.pixels, (pixels * 65535).astype(np.uint16))
        self.assertGreater(int(imported.pixels[0, 0, 0]) % 257, 0)
        saved = json.loads(imported.metadata["parameters"])
        self.assertEqual(saved["Prompt"], "весна")
        self.assertEqual(saved["seed"], 7)
        self.assertTrue(saved["native_export"]["alpha"])

    def test_exr_float_preserves_hdr_negative_values_and_straight_alpha(self):
        pixels = np.array([[[2.75, -0.25, 0.1234567, 0.37], [1.2, 0, 3.4, 0.8]]], np.float32)
        path = self.write(pixels, "exr", "32-bit float", "linear")
        imported = formats.read_image(path)
        self.assertEqual(imported.pixels.dtype, np.float32)
        np.testing.assert_array_equal(imported.pixels, pixels)
        self.assertEqual(imported.color_space, "linear")
        self.assertEqual(json.loads(imported.metadata["parameters"])["seed"], 7)

    def test_exr_half_rounding_and_srgb_conversion_use_native_source(self):
        pixels = torch.tensor([[[0.5, 0.25, 0.75, 0.4], [0.9, 0.03, 0.11, 0.0]]])
        path = self.write(pixels, "exr", "16-bit float", "sRGB")
        imported = formats.read_image(path)
        expected = self.native.srgb_to_linear(pixels).half().float().numpy()
        np.testing.assert_array_equal(imported.pixels, expected)
        self.assertEqual(imported.pixels[0, 0, 3], np.float32(np.float16(0.4)))

    def test_pipeline_keeps_float_precision_only_for_explicit_advanced_output(self):
        image = torch.tensor([[[[0.12345, 0.23456, 0.34567]]]])
        ordinary = formats.pipeline_images(image, {})[0]
        self.assertEqual(ordinary.dtype, np.uint8)
        advanced = formats.pipeline_images(image, {"output_bit_depth": "16-bit"})[0]
        np.testing.assert_array_equal(advanced, image[0].numpy())
        advanced[0, 0, 0] = 1
        self.assertAlmostEqual(float(image[0, 0, 0, 0]), 0.12345)

    def test_native_still_avif_round_trip_and_embedded_metadata(self):
        pixels = np.full((64, 64, 3), 0.4, dtype=np.float32)
        path = self.write(pixels, "avif", "10-bit YUV420")
        imported = formats.read_image(path)
        self.assertEqual(imported.pixels.shape, pixels.shape)
        self.assertEqual(imported.pixels.dtype, np.uint16)
        self.assertEqual(json.loads(imported.metadata["parameters"])["Prompt"], "весна")
        self.assertLess(float(np.abs(imported.pixels.astype(float) / 65535 - pixels).max()), 0.02)

    def test_alpha_and_invalid_choice_rejected_before_native_encoder(self):
        with patch.object(formats, "_native") as native:
            with self.assertRaisesRegex(ValueError, "alpha"):
                self.write(np.zeros((64, 64, 4)), "avif", "auto")
            native.assert_not_called()
        for values in (
            {"output_format": "video"},
            {"output_bit_depth": "32-bit float"},
            {"output_crf": True},
            {"output_color_space": "linear"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                formats.format_settings(values)

    def test_original_is_immutable_and_import_is_byte_identical(self):
        path = self.write(np.full((2, 3, 3), 0.55555, dtype=np.float32))
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            formats.write_image(path, np.zeros((2, 3, 3)))
        copied, imported = formats.import_image(path, self.root / "outputs")
        self.assertNotEqual(path, copied)
        self.assertEqual(copied.read_bytes(), original)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(imported.pixels.dtype, np.uint16)

    def test_grayscale_png16_and_float_exr_round_trip(self):
        pixels = np.array([[0.01, 0.23, 0.98]], dtype=np.float32)
        png = formats.read_image(self.write(pixels))
        np.testing.assert_array_equal(png.pixels, (pixels * 65535).astype(np.uint16))
        exr = formats.read_image(self.write(pixels, "exr", "32-bit float", "linear"))
        np.testing.assert_array_equal(exr.pixels, pixels)

    def test_animated_and_mislabeled_inputs_rejected(self):
        animated = self.root / "animated.png"
        Image.new("RGB", (8, 8), "red").save(
            animated, save_all=True, append_images=[Image.new("RGB", (8, 8), "blue")], duration=100
        )
        with self.assertRaisesRegex(ValueError, "Animated"):
            formats.read_image(animated)
        mislabeled = self.root / "mislabeled.exr"
        mislabeled.write_bytes(animated.read_bytes())
        with self.assertRaisesRegex(ValueError, "contents"):
            formats.read_image(mislabeled)
        outputs = self.root / "outputs"
        with self.assertRaisesRegex(ValueError, "contents"):
            formats.import_image(mislabeled, outputs)
        self.assertEqual(list(outputs.iterdir()), [])

    def test_bounds_reject_oversized_headers_nonfinite_and_oversized_images(self):
        malformed = self.root / "bad.exr"
        malformed.write_bytes(
            b"\x76\x2f\x31\x01\x02\x00\x00\x00parameters\x00string\x00"
            + struct.pack("<i", formats.MAX_METADATA + 1)
        )
        with self.assertRaisesRegex(ValueError, "header"):
            formats.image_metadata(malformed)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.write(np.full((1, 1, 3), np.nan))
        with patch.object(formats, "MAX_PIXELS", 1), self.assertRaisesRegex(ValueError, "pixels"):
            self.write(np.zeros((1, 2, 3)))

    def test_api_output_schema_and_capture_keep_explicit_native_options(self):
        from ruinedfooocus.integrations.openai_api import ImageRequest
        from test_api_image_edits import queued_data

        request = ImageRequest(
            prompt="spring",
            output_format="exr",
            output_bit_depth="32-bit float",
            output_color_space="linear",
        )
        data = queued_data("SDXL", payload_options=request.model_dump(exclude_none=True))
        self.assertEqual(
            formats.format_settings(data), formats.format_settings(request.model_dump())
        )
        with self.assertRaises(ValueError):
            ImageRequest(prompt="spring", output_format="png", output_bit_depth="32-bit float")

    def test_derived_preview_retains_replay_metadata_and_original_bytes(self):
        source = self.write(
            np.full((2, 3, 4), 0.4321, dtype=np.float32), "exr", "32-bit float", "linear ACEScg"
        )
        original = source.read_bytes()
        spec = importlib.util.spec_from_file_location(
            "comfy_execution.preview_tonemap",
            ROOT / "repositories/ComfyUI/comfy_execution/preview_tonemap.py",
        )
        preview_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preview_module)
        with patch.dict(sys.modules, {"comfy_execution.preview_tonemap": preview_module}):
            preview = formats.display_image(source, self.root / "cache")
            self.assertEqual(formats.display_image(source, self.root / "cache"), preview)
        with Image.open(preview) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.size, (3, 2))
            self.assertEqual(json.loads(image.info["parameters"])["seed"], 7)
        self.assertEqual(source.read_bytes(), original)
        self.assertNotEqual(source, preview)

    def test_real_gradio_format_event_and_replay_change_valid_depth_choices(self):
        from ruinedfooocus.features.image_processing import ui_native_formats as ui

        with gr.Blocks(analytics_enabled=False) as app:
            controls = ui.create_controls()
        self.assertIs(type(controls["output_crf"].preprocess(18.0)), int)
        index = next(key for key, fn in app.fns.items() if fn.fn.__name__ == "format_choices")
        result = asyncio.run(app.process_api(index, ["exr"], state=SessionState(app)))["data"]
        self.assertEqual(result[0]["value"], "16-bit float")
        self.assertIn(["32-bit float", "32-bit float"], result[0]["choices"])
        restored = ui.replay_values(
            dict(
                output_format="avif", output_bit_depth="10-bit YUV420", output_color_space="HDR PQ"
            )
        )
        self.assertEqual(restored["output_color_space"]["value"], "HDR PQ")
        self.assertTrue(restored["output_crf"]["visible"])

    def test_library_exports_linear_exr_to_display_png_without_reinterpreting_samples(self):
        from ruinedfooocus.features.gallery.actions import source_record
        from ruinedfooocus.features.image_processing.ui_native_formats import export_selection

        source = self.write(
            np.full((2, 3, 4), 0.5, dtype=np.float32), "exr", "32-bit float", "linear"
        )
        original = source.read_bytes()
        exported = export_selection(
            source_record(source), "png", "8-bit", "sRGB", 18, self.root / "cache"
        )
        with Image.open(exported) as image:
            self.assertGreater(image.getpixel((0, 0))[0], 165)
            self.assertLess(image.getpixel((0, 0))[0], 215)
            self.assertEqual(image.getpixel((0, 0))[3], 127)
        self.assertEqual(source.read_bytes(), original)

    def test_public_api_file_and_base64_are_the_original_precision_file(self):
        from test_api_live_contracts import ApiLiveContracts, job
        import base64

        fixture = ApiLiveContracts()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        pixels = np.full((2, 3, 3), 1.234567, np.float32)
        path = self.write(pixels, "exr", "32-bit float", "linear")
        fixture.backend.images.side_effect = lambda *args: job(("result", [str(path)]))
        values = dict(
            prompt="spring",
            output_format="exr",
            output_bit_depth="32-bit float",
            output_color_space="linear",
        )
        response = fixture.client.post(
            "/v1/images/generations", json=values, headers=fixture.headers
        )
        self.assertEqual(response.status_code, 200, response.text)
        asset = fixture.client.get(response.json()["data"][0]["url"])
        self.assertEqual(asset.headers["content-type"], "image/x-exr")
        self.assertEqual(asset.content, path.read_bytes())
        response = fixture.client.post(
            "/v1/images/generations",
            json=dict(values, response_format="b64_json"),
            headers=fixture.headers,
        )
        self.assertEqual(
            base64.b64decode(response.json()["data"][0]["b64_json"]), path.read_bytes()
        )


if __name__ == "__main__":
    unittest.main()
