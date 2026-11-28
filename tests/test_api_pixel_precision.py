"""Actual encoded precision and public pixel-tool admission, without model work."""

import base64
from contextlib import nullcontext
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import av
from PIL import Image
import torch

import ruinedfooocus.app
from ruinedfooocus.app import api_runtime
from ruinedfooocus.features.image_processing import image_histogram
from ruinedfooocus.features.image_processing import native_image_formats as formats
from ruinedfooocus.integrations.openai_api import create_apps
from test_native_image_formats import native_codecs


def inline(raw):
    return base64.b64encode(raw).decode()


def completed(value):
    events = queue.Queue()
    events.put(("result", value))
    return NS(events=events, cancelled=threading.Event())


class PixelPrecisionAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native = native_codecs()
        values = torch.tensor([[[0.12345, 0.34567, 0.56789, 0.33333], [1, 0, 0.75, 0]]])
        cls.fixtures = {
            "RGB16 PNG": native._encode_image(values[..., :3], "png", "16-bit", "sRGB"),
            "RGBA16 PNG": native._encode_image(values, "png", "16-bit", "sRGB"),
            "EXR32": native._encode_image(values, "exr", "32-bit float", "linear"),
        }
        gray = BytesIO()
        Image.new("I;16", (2, 2), 12345).save(gray, format="PNG")
        cls.fixtures["gray16 PNG"] = gray.getvalue()
        avif = BytesIO()
        native._save_avif(torch.full((1, 64, 64, 3), 0.34567), avif, "10-bit YUV420", "sRGB", 18)
        cls.fixtures["AVIF10"] = avif.getvalue()
        rgba8 = BytesIO()
        Image.new("RGBA", (16, 16), (32, 64, 128, 37)).save(rgba8, format="AVIF")
        cls.rgba8_avif = rgba8.getvalue()
        # Optional frozen real RF artifacts augment the same public regressions.
        # Portable closure tests always exercise all five native encoded fixtures.
        manifest = os.environ.get("RF_CPU_PIXEL_PRECISION_FIXTURES")
        if manifest:
            for entry in json.loads(Path(manifest).read_text(encoding="utf-8")):
                raw = Path(entry["path"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                    raise AssertionError("The frozen RF precision fixture changed")
                cls.fixtures[entry["name"]] = raw

    def setUp(self):
        shared = NS(
            path_manager=NS(get_model_folders=Mock(side_effect=AssertionError("model lookup"))),
            models=NS(offline=True),
            settings=NS(default_settings={}),
        )
        self.shared = shared
        self.enterContext(patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}))
        self.enterContext(patch.object(ruinedfooocus.app, "state", shared, create=True))
        backend = NS(
            histogram=api_runtime.histogram,
            images=api_runtime.images,
            resolve_model=Mock(side_effect=AssertionError("generation model lookup")),
        )
        api, _ = create_apps(backend)
        app = FastAPI()
        app.mount("/v1", api)
        self.client = self.enterContext(TestClient(app))

    def test_native_reader_precision_is_available_before_rgb8_conversion(self):
        for label, raw in self.fixtures.items():
            with self.subTest(fixture=label):
                precision = formats.encoded_precision(raw)
                self.assertGreater(precision.bits, 8)
                self.assertGreater(precision.width, 0)
                self.assertGreater(precision.height, 0)
        self.assertTrue(formats.encoded_precision(self.fixtures["RGBA16 PNG"]).has_alpha)
        self.assertFalse(formats.encoded_precision(self.fixtures["RGB16 PNG"]).has_alpha)
        with self.assertRaises(ValueError):
            formats.encoded_precision(self.fixtures["AVIF10"], max_pixels=64)

    def test_public_highbit_histogram_rejects_before_pillow_or_queue(self):
        with (
            patch.object(api_runtime, "submit") as submit,
            patch.object(Image, "open", side_effect=AssertionError("Pillow reduction")) as pillow,
            patch.object(
                formats.tempfile, "mkstemp", side_effect=AssertionError("temporary image")
            ),
            patch.object(formats, "write_image", side_effect=AssertionError("output image")),
        ):
            for label, raw in self.fixtures.items():
                original = hashlib.sha256(raw).hexdigest()
                with self.subTest(fixture=label):
                    response = self.client.post(
                        "/v1/images/histogram", json={"input_image": inline(raw)}
                    )
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertIn("8-bit", response.json()["error"]["message"])
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), original)
            submit.assert_not_called()
            pillow.assert_not_called()

    def test_public_filters_curves_and_multipart_share_the_precision_guard(self):
        with (
            patch.object(api_runtime, "submit") as submit,
            patch.object(Image, "open", side_effect=AssertionError("Pillow reduction")),
            patch.object(
                formats.tempfile, "mkstemp", side_effect=AssertionError("temporary image")
            ),
        ):
            for label, raw in self.fixtures.items():
                for kind in (
                    "image-blur",
                    "image-sharpen",
                    "adjust-brightness",
                    "adjust-contrast",
                    "image-curves",
                    "glsl-effect",
                ):
                    with self.subTest(fixture=label, kind=kind):
                        body = dict(
                            model="missing-checkpoint",
                            input_image=inline(raw),
                            controlnet={"type": kind},
                        )
                        response = self.client.post("/v1/images/generations", json=body)
                        self.assertEqual(response.status_code, 400, response.text)
                        self.assertIn("8-bit", response.json()["error"]["message"])
                response = self.client.post(
                    "/v1/images/edits",
                    data={"controlnet": json.dumps({"type": "image-curves"})},
                    files={"image": ("source.png", raw, "application/octet-stream")},
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("8-bit", response.json()["error"]["message"])
            submit.assert_not_called()

    def test_rgb8_capture_keeps_alpha_geometry_identity_and_no_model_recipe(self):
        image = Image.new("RGBA", (3, 2))
        image.putdata([(255, 0, 0, 0), (0, 255, 0, 64), (0, 0, 255, 128)] * 2)
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        raw = encoded.getvalue()
        expected = image_histogram.capture(image)
        with patch.object(
            api_runtime, "submit", side_effect=lambda operation: completed([])
        ) as submit:
            for kind in ("image-blur", "adjust-brightness", "image-curves"):
                with self.subTest(kind=kind):
                    response = self.client.post(
                        "/v1/images/generations",
                        json={"input_image": inline(raw), "controlnet": {"type": kind}},
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    operation = submit.call_args.args[0]
                    closure = dict(
                        zip(operation.prepare.__code__.co_freevars, operation.prepare.__closure__)
                    )
                    data = closure["data"].cell_contents
                    captured = data["input_image"]
                    self.assertEqual(captured.size, image.size)
                    self.assertEqual(captured.mode, "RGBA")
                    self.assertEqual(captured.tobytes(), image.tobytes())
                    self.assertEqual(image_histogram.capture(captured), expected)
                    self.assertEqual(data["prompt"], "")

    def test_8bit_inline_formats_keep_the_existing_pillow_color_and_alpha(self):
        for mode, color in (("RGB", (32, 64, 128)), ("RGBA", (32, 64, 128, 37)), ("L", 73)):
            for kind in ("PNG", "JPEG", "WEBP") if mode == "RGB" else ("PNG",):
                with self.subTest(mode=mode, kind=kind):
                    encoded = BytesIO()
                    Image.new(mode, (7, 5), color).save(encoded, format=kind)
                    raw = encoded.getvalue()
                    expected = api_runtime.decode_image(inline(raw), preserve_mode=True)
                    actual = api_runtime.decode_image(
                        inline(raw), preserve_mode=True, require_8bit=True
                    )
                    self.assertEqual((actual.size, actual.mode), (expected.size, expected.mode))
                    self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_other_working_copy_routes_do_not_gain_a_highbit_blacklist(self):
        raw = self.fixtures["RGBA16 PNG"]
        with patch.object(formats, "encoded_precision", side_effect=AssertionError("opt in only")):
            ordinary = api_runtime.decode_image(inline(raw), preserve_alpha=True)
            self.assertEqual(ordinary.mode, "RGBA")
            self.assertEqual(ordinary.size, (2, 1))
        for raw in (b"invalid", b"", b"\x89PNG\r\n\x1a\n"):
            with self.assertRaisesRegex(ValueError, "8-bit"):
                api_runtime.decode_image(inline(raw), require_8bit=True)

    def test_public_malformed_data_uri_is_400_for_default_and_guarded_decoders(self):
        backend = NS(
            images=api_runtime.images,
            upscale=api_runtime.upscale,
            histogram=api_runtime.histogram,
            resolve_model=Mock(
                return_value=("image/local", {"name": "not-loaded", "family": "SDXL"})
            ),
        )
        api, _ = create_apps(backend)
        app = FastAPI()
        app.mount("/v1", api)
        with (
            TestClient(app) as client,
            patch.object(api_runtime, "submit") as submit,
            patch.object(formats, "encoded_precision") as precision,
            patch.object(Image, "open") as pillow,
            # A named image model is first checked against pending catalogue
            # downloads; that local folder check happens before input decoding.
            patch.object(self.shared.path_manager, "get_model_folders", return_value=[]),
        ):
            for malformed in ("data:image/png;base64", "data:image/png", "data:image/"):
                for route, options in (
                    ("generations", {"model": "image/local"}),
                    ("upscale", {"model": "upscaler/local"}),
                    ("generations", {"controlnet": {"type": "image-curves"}}),
                    ("histogram", {}),
                ):
                    with self.subTest(input=malformed, route=route, options=options):
                        response = client.post(
                            "/v1/images/" + route,
                            json={"input_image": malformed, **options},
                        )
                        self.assertEqual(response.status_code, 400, response.text)
                        self.assertIn("valid base64 image", response.json()["error"]["message"])
            submit.assert_not_called()
            precision.assert_not_called()
            pillow.assert_not_called()

    def test_public_rgba8_avif_keeps_auxiliary_alpha_and_source_geometry(self):
        ordinary = api_runtime.decode_image(inline(self.rgba8_avif), preserve_mode=True)
        self.assertEqual((ordinary.mode, ordinary.size), ("RGBA", (16, 16)))
        self.assertEqual(ordinary.getchannel("A").getextrema(), (37, 37))
        precision = formats.encoded_precision(self.rgba8_avif)
        self.assertEqual(precision.bits, 8)
        self.assertTrue(precision.has_alpha)
        with patch.object(
            api_runtime, "submit", side_effect=lambda operation: completed([])
        ) as submit:
            for kind in ("adjust-brightness", "image-curves"):
                response = self.client.post(
                    "/v1/images/generations",
                    json={"input_image": inline(self.rgba8_avif), "controlnet": {"type": kind}},
                )
                self.assertEqual(response.status_code, 200, response.text)
                operation = submit.call_args.args[0]
                closure = dict(
                    zip(operation.prepare.__code__.co_freevars, operation.prepare.__closure__)
                )
                captured = closure["data"].cell_contents["input_image"]
                self.assertEqual((captured.mode, captured.size), (ordinary.mode, ordinary.size))
                self.assertEqual(captured.tobytes(), ordinary.tobytes())
                self.assertEqual(
                    image_histogram.capture(captured), image_histogram.capture(ordinary)
                )
        # Inline inspection does not relax original-file import admission.
        with av.open(BytesIO(self.rgba8_avif), mode="r") as container:
            with self.assertRaisesRegex(ValueError, "one still image"):
                formats._first_frame(container)

    def test_unknown_header_precision_falls_back_to_actual_first_frame(self):
        raw = self.fixtures["RGBA16 PNG"]
        header = NS(
            streams=NS(
                video=[NS(index=0, width=2, height=1, codec_context=NS(format=None), metadata={})]
            )
        )
        actual_open = av.open
        with patch.object(
            av, "open", side_effect=[nullcontext(header), actual_open(BytesIO(raw), mode="r")]
        ):
            self.assertEqual(formats.encoded_precision(raw).bits, 16)
        for components in ((), (NS(bits=0),)):
            with (
                self.subTest(components=components),
                self.assertRaisesRegex(ValueError, "precision"),
            ):
                formats._pixel_precision(2, 1, NS(components=components), formats.MAX_PIXELS)


if __name__ == "__main__":
    unittest.main()
