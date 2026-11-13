"""Actual pinned histogram counts and complete analysis UI/API contracts on CPU."""

import asyncio
import base64
from concurrent.futures import CancelledError
from dataclasses import replace
from io import BytesIO
import queue
import sys
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import gradio as gr
from gradio.state_holder import SessionState
import numpy as np
from PIL import Image
import torch

from ruinedfooocus.app import api_runtime
from ruinedfooocus.features.image_processing import image_histogram as histogram
from ruinedfooocus.features.image_processing import ui_image_histogram as ui
from ruinedfooocus.integrations.openai_api import create_apps
from test_native_model_coverage import native_method


class NativeOutput(tuple):
    def __new__(cls, *values):
        return super().__new__(cls, values)

    @property
    def result(self):
        return self


def encoded(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class ImageHistogram(unittest.TestCase):
    def setUp(self):
        self.native = native_method(
            "comfy_extras/nodes_curve.py",
            "ImageHistogram",
            "execute",
            {"np": np, "io": NS(NodeOutput=NativeOutput)},
        )
        self.enterContext(
            patch.dict(sys.modules, {"comfy_extras.nodes_curve": NS(ImageHistogram=self.native)})
        )
        self.image = Image.new("RGBA", (2, 2))
        self.image.putdata(
            [(255, 0, 0, 0), (0, 255, 0, 64), (0, 0, 255, 128), (255, 255, 255, 255)]
        )

    def execute_job(self, operation):
        events = queue.Queue()
        cancelled = threading.Event()
        try:
            events.put(("result", operation(NS(cancelled=cancelled), 17)))
        except ValueError as error:
            events.put(("invalid_request", str(error)))
        return NS(events=events, cancelled=cancelled)

    def test_pinned_native_counts_include_transparent_rgb_and_bt709(self):
        request = histogram.capture(self.image)
        self.image.paste((0, 0, 0, 255), (0, 0, 2, 2))
        result = histogram.analyze(request).as_dict()
        for channel in ("red", "green", "blue"):
            self.assertEqual(result["bins"][channel][0], 2)
            self.assertEqual(result["bins"][channel][255], 2)
        self.assertEqual(result["bins"]["rgb"][0], 2)
        self.assertEqual(result["bins"]["rgb"][255], 2)
        self.assertEqual(
            [index for index, count in enumerate(result["bins"]["luminance"]) if count],
            [18, 54, 182, 255],
        )
        self.assertEqual(sum(result["bins"]["luminance"]), 4)
        self.assertTrue(result["source"]["has_alpha"])
        self.assertEqual(result["source"]["width"], 2)
        self.assertEqual(result["native_node"], histogram.NATIVE_NODE)

    def test_native_rgb_composite_is_integer_average_not_a_normalized_pdf(self):
        result = histogram.analyze(histogram.capture(Image.new("RGB", (1, 1), (255, 0, 0))))
        data = result.as_dict()
        self.assertEqual(sum(data["bins"]["rgb"]), 0)
        self.assertEqual(sum(data["bins"]["luminance"]), 1)
        # Public output can be edited without mutating the frozen node result.
        data["bins"]["red"][255] = 999
        self.assertEqual(result.as_dict()["bins"]["red"][255], 1)

    def test_capture_orients_freezes_and_identifies_color_and_depth(self):
        image = Image.new("RGB", (3, 2), (34, 67, 129))
        image.getexif()[274] = 6
        image.info["exif"] = image.getexif().tobytes()
        image.info["icc_profile"] = b"owned inert profile marker"
        request = histogram.capture(image)
        self.assertEqual(request.size, (2, 3))
        self.assertTrue(request.has_icc_profile)
        self.assertFalse(request.has_alpha)
        self.assertEqual(request.source_sha256, histogram.capture(image.copy()).source_sha256)
        for mode in ("I;16", "I", "F", "CMYK"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "8-bit"):
                histogram.capture(Image.new(mode, (2, 2)))
        for value in (None, "local-file.png", np.zeros((2, 2, 3))):
            with self.assertRaises(ValueError):
                histogram.capture(value)
        with patch.object(histogram, "MAX_PIXELS", 3), self.assertRaises(ValueError):
            histogram.capture(self.image)

    def test_grayscale_palette_and_alpha_do_not_change_channel_count(self):
        for mode in ("1", "L", "LA", "P", "RGB", "RGBA"):
            with self.subTest(mode=mode):
                image = Image.new(mode, (3, 2))
                if mode == "P":
                    image.info["transparency"] = 0
                result = histogram.analyze(histogram.capture(image)).as_dict()
                self.assertEqual(set(result["bins"]), set(histogram.CHANNELS))
                self.assertTrue(all(len(values) == 256 for values in result["bins"].values()))
                self.assertEqual(result["source"]["has_alpha"], mode in ("LA", "P", "RGBA"))

    def test_cancel_before_or_after_native_call_drops_output(self):
        request = histogram.capture(self.image)
        signal = threading.Event()
        signal.set()
        with patch.object(self.native, "execute", wraps=self.native.execute) as native:
            with self.assertRaises(CancelledError):
                histogram.analyze(request, signal)
            native.assert_not_called()
        signal.clear()
        execute = self.native.execute

        def cancel_after(image):
            result = execute(image)
            signal.set()
            return result

        with patch.object(self.native, "execute", side_effect=cancel_after):
            with self.assertRaises(CancelledError):
                histogram.analyze(request, signal)

    def test_malformed_capture_and_native_outputs_are_rejected(self):
        request = histogram.capture(self.image)
        for malformed in (replace(request, rgba=b""), replace(request, size=(0, 2))):
            with self.assertRaises(ValueError):
                histogram.analyze(malformed)
        valid = self.native.execute(torch.zeros(1, 2, 2, 3)).result
        for malformed in (
            valid[:4],
            ([0] * 255,) * 5,
            ([True] * 256,) * 5,
            ([5] * 256,) * 5,
            ([0] * 256,) * 5,
        ):
            with (
                patch.object(self.native, "execute", return_value=malformed),
                self.assertRaises(RuntimeError),
            ):
                histogram.analyze(request)

    def test_api_runtime_freezes_source_before_enqueue_without_generation_resolution(self):
        with patch.object(api_runtime, "submit", side_effect=lambda operation: operation):
            operation = api_runtime.histogram({"input_image": encoded(self.image)})
        self.image.paste((0, 0, 0, 0), (0, 0, 2, 2))
        result = operation(NS(cancelled=threading.Event()), 22)
        self.assertEqual(result["bins"]["red"][255], 2)
        # The histogram decoder keeps source mode so high-bit inputs cannot be
        # silently treated as RGB8 by the analysis contract.
        with self.assertRaisesRegex(ValueError, "8-bit"):
            api_runtime.histogram({"input_image": encoded(Image.new("I;16", (2, 2)))})

    def test_public_api_requires_no_model_and_uses_existing_authorization(self):
        backend = NS(
            histogram=api_runtime.histogram,
            resolve_model=Mock(side_effect=AssertionError("generation model lookup")),
        )
        api, _ = create_apps(backend, key="owned-test-key")
        client = TestClient(api)
        self.assertEqual(
            client.post("/images/histogram", json={"input_image": encoded(self.image)}).status_code,
            401,
        )
        headers = {"Authorization": "Bearer owned-test-key"}
        with patch.object(api_runtime, "submit", side_effect=self.execute_job):
            response = client.post(
                "/images/histogram", json={"input_image": encoded(self.image)}, headers=headers
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["object"], "image.histogram")
            for extra in ({"model": "image/default"}, {"prompt": "hidden"}, {"size": "512x512"}):
                bad = client.post(
                    "/images/histogram",
                    json={"input_image": encoded(self.image), **extra},
                    headers=headers,
                )
                self.assertEqual(bad.status_code, 400)
            for invalid in ("../../owned.png", "http://127.0.0.1/private", "not base64"):
                bad = client.post(
                    "/images/histogram", json={"input_image": invalid}, headers=headers
                )
                self.assertEqual(bad.status_code, 400)
        backend.resolve_model.assert_not_called()
        schema = client.get("/openapi.json").json()
        self.assertIn("/images/histogram", schema["paths"])

    def test_real_gradio_mount_event_result_and_source_change_clear(self):
        with gr.Blocks(analytics_enabled=False) as demo:
            source = gr.Image(type="pil", image_mode="RGBA")
            plot, details = ui.create_ui(source)
        self.assertTrue(
            any(
                isinstance(block, gr.Accordion) and block.open is False
                for block in demo.blocks.values()
            )
        )
        event = next(fn for fn in demo.fns.values() if fn.fn is ui.analyze_source)
        self.assertEqual(event.inputs, [source])
        self.assertNotIn(source, event.outputs)
        with patch.object(api_runtime, "submit", side_effect=self.execute_job):
            chart, data = asyncio.run(ui.analyze_source(self.image))
        self.assertIn("<svg", chart)
        self.assertIn("aria-label", chart)
        self.assertEqual(data["source"]["input_mode"], "RGBA")
        clear_index = next(
            index for index, fn in demo.fns.items() if fn.fn and fn.fn.__name__ == "<lambda>"
        )
        result = asyncio.run(demo.process_api(clear_index, [], state=SessionState(demo)))["data"]
        self.assertEqual(result, ["", None])
        self.assertTrue(any(fn.cancels for fn in demo.fns.values()))

    def test_async_ui_cancellation_sets_serial_worker_signal(self):
        job = NS(events=queue.Queue(), cancelled=threading.Event())

        async def cancel():
            task = asyncio.create_task(ui.analyze_source(self.image))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        with patch.object(api_runtime, "submit", return_value=job):
            asyncio.run(cancel())
        self.assertTrue(job.cancelled.is_set())


if __name__ == "__main__":
    unittest.main()
