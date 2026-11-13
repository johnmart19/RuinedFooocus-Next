"""Focused native/Pillow AVIF auxiliary-alpha import and refusal contracts."""

from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import av
import numpy as np
from PIL import Image
import torch

from ruinedfooocus.features.image_processing import native_image_formats as formats
from ruinedfooocus.features.image_processing.ui_native_formats import import_selection
from test_native_image_formats import native_codecs


class NativeAvifAlphaImport(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def avif(self, mode="RGBA", *, metadata=False, animated=False):
        color = (32, 64, 128, 37) if mode == "RGBA" else (32, 64, 128)
        source = Image.new(mode, (16, 16), color)
        options = {}
        if metadata:
            exif = Image.Exif()
            exif[0x010E] = "parameters:" + json.dumps({"Prompt": "alpha codec fixture", "seed": 11})
            options["exif"] = exif
        if animated:
            options.update(save_all=True, append_images=[Image.new(mode, (16, 16), "white")])
        path = self.root / (mode + ("-animated" if animated else "") + ".avif")
        source.save(path, format="AVIF", **options)
        return path

    def test_real_rgba8_native_components_import_exact_pillow_pixels_and_original_bytes(self):
        path = self.avif(metadata=True)
        original = path.read_bytes()
        with Image.open(path) as image:
            expected = np.asarray(image).copy()
        with av.open(str(path), mode="r") as container:
            self.assertEqual(len(container.streams.video), 2)
            self.assertEqual(
                {stream.metadata.get("title") for stream in container.streams.video},
                {"Color", "Alpha"},
            )
        result = formats.read_image(path)
        np.testing.assert_array_equal(result.pixels, expected)
        self.assertEqual(result.bit_depth, "8-bit")
        self.assertEqual(result.pixels.dtype, np.uint8)
        np.testing.assert_array_equal(result.pixels[..., 3], 37)
        self.assertEqual(json.loads(result.metadata["parameters"])["Prompt"], "alpha codec fixture")
        _, imported = import_selection(str(path), str(self.root / "imports"))
        self.assertEqual(Path(imported).read_bytes(), original)
        self.assertEqual(path.read_bytes(), original)
        copied = formats.read_image(imported)
        np.testing.assert_array_equal(copied.pixels, expected)
        self.assertEqual(copied.metadata, result.metadata)

    def test_rgba8_preview_keeps_alpha_and_parameters(self):
        path = self.avif(metadata=True)
        original = path.read_bytes()
        preview = formats.display_image(path, self.root / "cache")
        with Image.open(preview) as image:
            self.assertEqual((image.mode, image.size), ("RGBA", (16, 16)))
            self.assertEqual(image.getchannel("A").getextrema(), (37, 37))
            self.assertEqual(json.loads(image.info["parameters"])["seed"], 11)
        self.assertEqual(path.read_bytes(), original)

    def test_ordinary_rgb8_keeps_existing_single_stream_path(self):
        path = self.avif("RGB")
        with Image.open(path) as image:
            expected = np.asarray(image).copy()
        with patch.object(
            formats, "_read_avif_auxiliary_alpha", side_effect=AssertionError("not an alpha pair")
        ):
            result = formats.read_image(path)
        self.assertEqual(result.pixels.shape, (16, 16, 3))
        self.assertEqual(result.bit_depth, "8-bit")
        np.testing.assert_allclose(result.pixels.astype(int), expected.astype(int), atol=2)

    def test_true_rgb_and_rgba_animations_are_refused_without_publishing(self):
        for mode in ("RGB", "RGBA"):
            with self.subTest(mode=mode):
                path = self.avif(mode, animated=True)
                original = path.read_bytes()
                with self.assertRaises(ValueError):
                    formats.import_image(path, self.root / (mode + "-imports"))
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(list((self.root / (mode + "-imports")).glob("*")), [])

    def test_unrelated_video_streams_are_not_treated_as_auxiliary_alpha(self):
        path = self.avif()
        unrelated = NS(
            streams=NS(video=[NS(metadata={"title": "Color"}), NS(metadata={"title": "Other"})])
        )
        with (
            patch.object(av, "open", return_value=nullcontext(unrelated)),
            patch.object(Image, "open", side_effect=AssertionError("unrelated tracks")),
            self.assertRaisesRegex(ValueError, "one still image"),
        ):
            formats.read_image(path)

    def test_auxiliary_highbit_frame_is_refused_before_pillow_reduction(self):
        path = self.avif()
        color = NS(index=0, width=16, height=16, metadata={"title": "Color"})
        alpha = NS(index=1, width=16, height=16, metadata={"title": "Alpha"})
        frame = av.VideoFrame(16, 16, "yuv420p10le")
        packet = NS(stream=color, decode=lambda: [frame])
        container = NS(streams=NS(video=[color, alpha]), demux=lambda streams: iter([packet]))
        with (
            patch.object(av, "open", return_value=nullcontext(container)),
            patch.object(Image, "open", side_effect=AssertionError("highbit reduction")),
            self.assertRaisesRegex(ValueError, "requires 8-bit"),
        ):
            formats.read_image(path)

    def test_existing_rgb10_and_rgba_png16_precision_stays_native(self):
        native = native_codecs()
        avif = self.root / "rgb10.avif"
        native._save_avif(
            torch.full((1, 64, 64, 3), 0.34567), str(avif), "10-bit YUV420", "sRGB", 18
        )
        result = formats.read_image(avif)
        self.assertEqual((result.bit_depth, result.pixels.dtype), ("10-bit", np.dtype("uint16")))
        self.assertEqual(result.pixels.shape, (64, 64, 3))
        pixels = torch.tensor([[[0.12345, 0.34567, 0.56789, 0.33333], [1, 0, 0.75, 0]]])
        png = self.root / "rgba16.png"
        png.write_bytes(native._encode_image(pixels, "png", "16-bit", "sRGB"))
        imported = formats.read_image(png)
        self.assertEqual(imported.bit_depth, "16-bit")
        np.testing.assert_array_equal(imported.pixels, (pixels.numpy() * 65535).astype(np.uint16))


if __name__ == "__main__":
    unittest.main()
