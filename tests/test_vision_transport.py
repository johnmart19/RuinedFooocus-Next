"""Vision transport is bounded, preserves framing and never changes source pixels."""

import base64
import copy
from io import BytesIO
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from ruinedfooocus.features.chat.llama_vision import image_content, attach_image, review_messages


def decoded(content):
    prefix, encoded = content["image_url"]["url"].split(",", 1)
    with Image.open(BytesIO(base64.b64decode(encoded))) as source:
        return prefix, source.copy()


class VisionTransportTests(unittest.TestCase):
    def test_standard_and_high_are_bounded_without_stretching_or_cropping(self):
        for size in ((1777, 3333), (3301, 1711), (2300, 2300)):
            source = Image.new("RGB", size, (35, 82, 194))
            before = source.tobytes()
            for detail, limit, mime in (("standard", 1024, "jpeg"), ("high", 2048, "png")):
                with self.subTest(size=size, detail=detail):
                    prefix, image = decoded(image_content(source, detail=detail))
                    self.assertIn("image/" + mime, prefix)
                    self.assertLessEqual(max(image.size), limit)
                    self.assertLessEqual(
                        abs(image.width * source.height - image.height * source.width),
                        max(source.size),
                    )
                    self.assertEqual(source.tobytes(), before)

    def test_high_detail_keeps_small_rgb_pixels_exact_and_sources_metadata_intact(self):
        source = Image.frombytes(
            "RGB", (43, 29), bytes((value * 7) % 256 for value in range(43 * 29 * 3))
        )
        source.info["parameters"] = '{"prompt":"literal CAFÉ 42"}'
        before, info = source.tobytes(), copy.deepcopy(source.info)
        _, image = decoded(image_content(source, detail="high"))
        self.assertEqual(image.size, source.size)
        self.assertEqual(image.tobytes(), before)
        self.assertEqual(source.info, info)
        rgba = source.convert("RGBA")
        rgba.putalpha(87)
        before = rgba.tobytes()
        image_content(rgba, detail="high")
        self.assertEqual(rgba.tobytes(), before)

    def test_exif_orientation_is_applied_only_to_the_sent_copy(self):
        source = Image.new("RGB", (40, 20), "blue")
        source.putpixel((0, 0), (255, 0, 0))
        source.getexif()[274] = 6
        before, exif = source.tobytes(), source.getexif().tobytes()
        _, image = decoded(image_content(source, detail="high"))
        self.assertEqual(image.size, (20, 40))
        self.assertEqual(image.getpixel((19, 0)), (255, 0, 0))
        self.assertEqual(source.tobytes(), before)
        self.assertEqual(source.getexif().tobytes(), exif)

    def test_attachment_and_review_share_detail_without_reusing_prior_image(self):
        requests = []
        for color in ("red", "blue"):
            source = Image.new("RGB", (1500, 1000), color)
            messages = [{"role": "user", "content": 'Read "SUN 42".'}]
            attach_image(messages, source, detail="high")
            self.assertEqual(messages[0]["content"][0]["text"], 'Read "SUN 42".')
            self.assertEqual(decoded(messages[0]["content"][1])[1].size, source.size)
            requests.append(messages[0]["content"][1])
            review = review_messages(source, "Read text", "SUN 42", detail="high")
            self.assertEqual(review[-1]["content"][-1], requests[-1])
        self.assertNotEqual(requests[0], requests[1])

    def test_api_review_keeps_png_and_accepts_bounded_large_originals(self):
        from ruinedfooocus.app import api_runtime, state as shared
        from ruinedfooocus.features.chat import llama_models, model_preparation

        original = Image.new("RGB", (4097, 4097), "green")
        buffer = BytesIO()
        original.save(buffer, format="PNG")
        payload = {
            "input_image": base64.b64encode(buffer.getvalue()).decode(),
            "prompt": "Read text",
        }
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "transport-only.gguf"
            projector = Path(directory) / "transport-projector.gguf"
            model.write_bytes(b"inert transport fixture, never loaded")
            projector.write_bytes(b"inert projector fixture, never loaded")
            with (
                patch.object(
                    shared,
                    "settings",
                    SimpleNamespace(default_settings={"llama_mmproj": str(projector)}),
                ),
                patch.object(shared, "path_manager", SimpleNamespace(DOWNLOADABLE_FILES={})),
                patch.object(llama_models, "local_models", return_value={}),
                patch.object(model_preparation, "prepare_chat_request") as prepare,
                patch.object(api_runtime, "submit", side_effect=lambda operation: operation),
            ):
                for detail, limit in (("standard", 1024), ("high", 2048)):
                    operation = api_runtime.review(
                        dict(payload, image_detail=detail), {"path": str(model)}
                    )
                    captured = inspect.getclosurevars(operation).nonlocals
                    self.assertEqual(captured["prepared"].model, model.resolve())
                    self.assertEqual(captured["prepared"].projector, projector.resolve())
                    self.assertTrue(captured["prepared"].vision)
                    content = captured["messages"][-1]["content"][-1]
                    prefix, image = decoded(content)
                    self.assertIn("image/png", prefix)
                    self.assertEqual(image.size, (limit, limit))
                prepare.assert_not_called()

    def test_invalid_detail_fails_before_opening_source(self):
        with patch("PIL.Image.open", side_effect=AssertionError("Must not open source")):
            with self.assertRaisesRegex(ValueError, "image detail"):
                image_content("unused.png", detail="unbounded")


if __name__ == "__main__":
    unittest.main()
