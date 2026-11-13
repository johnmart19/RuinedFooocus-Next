"""Actual API capture, serial worker, PNG metadata and capability downloads."""

import base64
from contextlib import nullcontext
from io import BytesIO
import json
import pathlib
from pathlib import Path
import queue
import random
import sys
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from pydantic import ValidationError
import ruinedfooocus.app

from ruinedfooocus.app import api_runtime
from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.image_processing import wevisone
from ruinedfooocus.integrations.openai_api import ImageRequest, create_apps
import test_wevisone as fixtures
from test_utility_sampling_capture import function


class WeVisOneCallers(unittest.TestCase):
    def setUp(self):
        self.harness = fixtures.WeVisOneTests("runTest")
        self.harness.setUp()
        self.addCleanup(self.harness.doCleanups)
        self.harness.manager.model_paths = {"temp_outputs_path": self.harness.root}
        stream = BytesIO()
        self.harness.image.save(stream, format="PNG")
        self.encoded = base64.b64encode(stream.getvalue()).decode()
        self.forbidden = Mock(
            side_effect=AssertionError("generation checkpoint/default/prompt reached")
        )
        shared = NS(
            models=NS(
                offline=True, get_models_by_path=self.forbidden, get_model_base=self.forbidden
            ),
            performance_settings=NS(apply=self.forbidden),
            resolution_settings=object(),
            path_manager=self.harness.manager,
            shared_cache={},
            state={},
        )
        self.shared = shared
        controls = NS(get_settings=lambda data: data.get("controlnet", {}))
        from ruinedfooocus.features.image_processing.depth_utility_pipeline import pipeline

        self.pipe = pipeline()
        route = dict(state={"pipeline": None}, shared=shared, controlnet=controls)
        function("ruinedfooocus/features/generation/pipelines.py", "selected_pipeline", route)
        update = function("ruinedfooocus/features/generation/pipelines.py", "update", route)
        self.enterContext(
            patch(
                "ruinedfooocus.features.image_processing.depth_utility_pipeline.pipeline",
                return_value=self.pipe,
            )
        )
        scope = dict(
            shared=shared,
            settings=NS(default_settings={}),
            control_tools=controls,
            check_interrupt=lambda data: None,
            process_metadata=self.forbidden,
            pipelines=NS(update=update),
            get_checkpoint_hashes=self.forbidden,
            process_prompt=self.forbidden,
            shift_attention=self.forbidden,
            parse_loras=self.forbidden,
            merge_loras=self.forbidden,
            TimeIt=lambda name: nullcontext(),
            resource_path=resource_path,
            random=random,
            Path=Path,
            pathlib=pathlib,
            Image=Image,
            PngInfo=PngInfo,
            json=json,
            generate_temp_filename=lambda **kw: str(
                self.harness.root / ("result." + kw["extension"])
            ),
        )
        execute = function("ruinedfooocus/app/async_worker.py", "_process", scope)
        prepare = function("ruinedfooocus/app/async_worker.py", "prepare_image_data", {})
        self.worker = NS(
            _process=execute,
            prepare_image_data=prepare,
            check_interrupt=lambda data: None,
            add_result=Mock(),
        )
        self.enterContext(
            patch.dict(
                sys.modules,
                {"ruinedfooocus.app.state": shared, "ruinedfooocus.app.async_worker": self.worker},
            )
        )
        self.enterContext(patch.object(ruinedfooocus.app, "state", shared, create=True))
        self.enterContext(patch.object(ruinedfooocus.app, "async_worker", self.worker, create=True))

    def payload(self, **values):
        return dict(
            input_image=self.encoded,
            controlnet={"type": "wevisone-matte"},
            wevisone_model=str(self.harness.model),
            **values,
        )

    def run_operation(self, payload):
        operation = api_runtime._image_operation(payload, {})
        self.assertEqual(operation.resource_group, "image")
        job = NS(cancelled=threading.Event())
        operation.prepare(job)
        return operation(job, "owned-wevisone-job")

    def test_actual_capture_preparation_and_serial_worker_write_owned_native_result(self):
        packet = self.run_operation(self.payload(wevisone_query="person", wevisone_max_length=64))
        self.assertIsInstance(packet, wevisone.WeVisOneArtifacts)
        self.assertEqual(len(packet), 1)
        with Image.open(packet[0]) as image:
            self.assertEqual((image.size, image.mode), ((6, 4), "RGBA"))
            metadata = json.loads(image.info["parameters"])
        self.assertEqual(metadata["controlnet"]["type"], "wevisone-matte")
        self.assertEqual(metadata["image_utility_source"]["native_node"], "WeVisOneDetect")
        self.assertNotIn("base_model_name", metadata)
        self.assertNotIn("steps", metadata)
        self.assertNotIn(str(self.harness.root), json.dumps(metadata))
        self.forbidden.assert_not_called()

    def test_public_api_returns_structured_boxes_answer_and_owned_download(self):
        def images(payload, selected):
            self.assertEqual(selected, {})
            packet = self.run_operation(payload)
            job = NS(cancelled=threading.Event(), events=queue.Queue())
            job.events.put(("result", packet))
            return job

        backend = NS(resolve_model=self.forbidden, images=images)
        api, _ = create_apps(backend)
        app = FastAPI()
        app.mount("/v1", api)
        with TestClient(app) as client:
            response = client.post("/v1/images/generations", json=self.payload())
            self.assertEqual(response.status_code, 200, response.text)
            reply = response.json()
            self.assertEqual(reply["wevisone"]["boxes"][0]["label"], "person")
            self.assertIn("<OBJ>", reply["wevisone"]["answer"])
            self.assertEqual(len(reply["data"]), 1)
            download = client.get(reply["data"][0]["url"])
            self.assertEqual(download.status_code, 200)
            self.assertEqual(Image.open(BytesIO(download.content)).size, (6, 4))
            self.assertEqual(client.get("/v1/images/unowned-path").status_code, 404)
        self.forbidden.assert_not_called()

    def test_typed_request_rejects_cross_operation_geometry_and_generation_inputs(self):
        valid = self.payload()
        self.assertEqual(ImageRequest(**valid).controlnet.type, "wevisone-matte")
        for change in (
            dict(n=2),
            dict(wevisone_query=""),
            dict(wevisone_max_length=True),
            dict(wevisone_output="unknown"),
            dict(mask=self.encoded),
            dict(reference_images=[self.encoded]),
            dict(native_export_conditioning=True),
            dict(controlnet={"type": "edit"}),
        ):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                ImageRequest(**(valid | change))
        with self.assertRaisesRegex(ValueError, "source size"):
            self.run_operation(valid | dict(size="512x512"))

    def test_animated_source_is_refused_before_losing_frame_identity(self):
        stream = BytesIO()
        first = Image.new("RGB", (6, 4), "red")
        first.save(
            stream, format="GIF", save_all=True, append_images=[Image.new("RGB", (6, 4), "blue")]
        )
        encoded = base64.b64encode(stream.getvalue()).decode()
        with self.assertRaises(ValueError):
            self.run_operation(self.payload() | {"input_image": encoded})
        self.harness.loader.assert_not_called()

    def test_public_highbit_uploads_fail_before_pillow_queue_or_native_loader(self):
        import torch
        from test_native_image_formats import native_codecs

        native = native_codecs()
        png = native._encode_image(
            torch.tensor([[[0.12345, 0.34567, 0.56789, 0.33333]]]), "png", "16-bit", "sRGB"
        )
        stream = BytesIO()
        native._save_avif(torch.full((1, 64, 64, 3), 0.34567), stream, "10-bit YUV420", "sRGB", 18)
        api, _ = create_apps(NS(resolve_model=self.forbidden, images=api_runtime.images))
        with (
            TestClient(api) as client,
            patch.object(api_runtime, "submit") as submit,
            patch.object(Image, "open", side_effect=AssertionError("Pillow reduction")) as pillow,
        ):
            for kind in wevisone.KINDS:
                for label, raw in (("RGBA16 PNG", png), ("AVIF10", stream.getvalue())):
                    with self.subTest(kind=kind, source=label):
                        response = client.post(
                            "/images/generations",
                            json=self.payload()
                            | {
                                "input_image": base64.b64encode(raw).decode(),
                                "controlnet": {"type": kind},
                            },
                        )
                        self.assertEqual(response.status_code, 400, response.text)
                        self.assertIn("8-bit", response.text)
            response = client.post(
                "/images/edits",
                files={"image": ("source.png", png, "image/png")},
                data=dict(
                    controlnet=json.dumps({"type": "wevisone-matte"}),
                    wevisone_model=str(self.harness.model),
                ),
            )
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn("8-bit", response.text)
        submit.assert_not_called()
        pillow.assert_not_called()
        self.harness.loader.assert_not_called()
        self.forbidden.assert_not_called()

    def test_public_multipart_native_mask_keeps_scalar_output_and_integer_length(self):
        def images(payload, selected):
            self.assertEqual(payload["wevisone_max_length"], 64)
            packet = self.run_operation(payload)
            job = NS(cancelled=threading.Event(), events=queue.Queue())
            job.events.put(("result", packet))
            return job

        api, _ = create_apps(NS(resolve_model=self.forbidden, images=images))
        with TestClient(api) as client:
            response = client.post(
                "/images/edits",
                files={"image": ("source.png", base64.b64decode(self.encoded), "image/png")},
                data=dict(
                    controlnet=json.dumps({"type": "wevisone-matte"}),
                    wevisone_model=str(self.harness.model),
                    wevisone_max_length="64",
                    wevisone_output="mask",
                    response_format="b64_json",
                ),
            )
        self.assertEqual(response.status_code, 200, response.text)
        reply = response.json()
        self.assertEqual(reply["wevisone"]["output"], "mask")
        with Image.open(BytesIO(base64.b64decode(reply["data"][0]["b64_json"]))) as image:
            self.assertEqual((image.mode, image.size), ("L", (6, 4)))
        self.forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
