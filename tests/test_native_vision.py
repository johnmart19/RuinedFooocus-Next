"""Native RT-DETR detection and SAM 3 point masks: capture, real node calls, API, replay."""

from concurrent.futures import CancelledError
from contextlib import contextmanager
from contextlib import nullcontext
from dataclasses import replace
import json
import pathlib
import random
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
import ruinedfooocus.app  # noqa: F401
from PIL import Image

from ruinedfooocus.features.generation.image_preparation import (
    capture_image_assets,
    prepare_image_assets,
    KEY,
)
from ruinedfooocus.features.image_processing import native_vision as native
from test_image_preparation import Manager
from test_native_model_coverage import native_method
from test_native_variant_defaults import write_header, native_backend, SHAPES
from test_utility_sampling_capture import function

HEADERS = {
    "RT_DETR_v4": {"encoder.pan_blocks.1.cv4.conv.weight": [64, 64, 1, 1]},
    "SAM3": {
        "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight": [3, 1],
        "detector.transformer.decoder.query_embed.weight": [2, 2],
    },
    "SAM31": {
        "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight": [3, 1],
        "detector.transformer.decoder.query_embed.weight": [2, 2],
        "detector.backbone.vision_backbone.propagation_convs.0.conv_1x1.weight": [1, 1, 1, 1],
    },
    "DepthAnything3": {
        "backbone.embeddings.patch_embeddings.projection.weight": [384, 3, 14, 14],
        **{f"head.projects.{index}.weight": [64, 384, 1, 1] for index in range(4)},
        "head.scratch.refinenet1.out_conv.weight": [64, 64, 3, 3],
        "head.scratch.output_conv2.2.weight": [1, 32, 1, 1],
    },
    "wrong": {"bb.layers.1.blocks.0.attn.relative_position_index": [49, 49]},
}
COCO = ["person", "cat"]


class NativeVisionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch.dict(
                SHAPES, {name: shapes for name, shapes in HEADERS.items() if name != "wrong"}
            )
        )
        self.enterContext(native_backend())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        folder = self.root / "diffusion_models"
        folder.mkdir()
        (self.root / "checkpoints").mkdir()
        geometry = self.root / "geometry_estimation"
        geometry.mkdir()
        self.files = {
            name: write_header(
                (geometry if name == "DepthAnything3" else folder) / f"{name}.safetensors", shapes
            )
            for name, shapes in HEADERS.items()
        }
        self.manager = Manager(self.root)
        from ruinedfooocus.features.generation.image_preparation import prepared_asset

        self.real_lease = prepared_asset
        self.image = Image.new("RGBA", (40, 20), (200, 100, 20, 128))
        self.enterContext(patch.object(native, "coco_classes", lambda: list(COCO)))

    def request(self, kind, **values):
        model = self.files[
            {"detect": "RT_DETR_v4", "segment": "SAM3", "depth-map": "DepthAnything3"}[kind]
        ]
        data = dict(input_image=self.image, native_vision_model=str(model))
        if kind == "segment":
            data["segment_positive_points"] = '[{"x":10,"y":5}]'
        data.update(values)
        return capture_image_assets(data, self.manager, {"type": kind})

    def test_header_validation_lists_only_models_of_the_active_tool(self):
        for kind, expected in (("detect", ["RT_DETR_v4"]), ("segment", ["SAM3", "SAM31"])):
            labels = [label for label, _ in native.native_model_choices(self.manager, kind)]
            self.assertEqual([label.split(" · ")[1] for label in labels], expected)
        self.assertEqual(native.validate_model(self.files["SAM31"], "segment"), "SAM31")
        for path, kind in (
            (self.files["SAM3"], "detect"),
            (self.files["RT_DETR_v4"], "segment"),
            (self.files["wrong"], "segment"),
            (self.files["SAM3"].with_suffix(".ckpt"), "segment"),
            (self.files["SAM3"], "track"),
        ):
            with self.assertRaises(ValueError):
                native.validate_model(path, kind)

    def test_capture_copies_the_source_and_needs_no_checkpoint_or_components(self):
        data = self.request("segment", segment_negative_points='[{"x":39.0,"y":19}]')
        self.assertIsNot(data["input_image"], self.image)
        self.image.putpixel((0, 0), (0, 0, 0, 0))
        self.assertEqual(data["input_image"].getpixel((0, 0)), (200, 100, 20, 128))
        (item,) = data[KEY].artifacts
        self.assertEqual((item.role, item.name), ("native-vision", str(self.files["SAM3"])))
        self.assertEqual(data["segment_refine_iterations"], 2)
        self.assertEqual(data["segment_output"], "cutout")
        self.assertEqual(json.loads(data["segment_negative_points"]), [{"x": 39.0, "y": 19}])
        # API callers may name the file below a configured model root.
        named = self.request("detect", native_vision_model="RT_DETR_v4.safetensors")
        self.assertEqual(named["native_vision_model"], str(self.files["RT_DETR_v4"]))
        self.assertEqual((named["detect_class"], named["detect_max_count"]), ("all", 100))

    def test_capture_rejects_unsafe_models_points_and_options_before_queueing(self):
        outside = write_header(self.root / "outside.safetensors", HEADERS["SAM3"])
        huge = Image.new("L", (4097, 4096))
        cases = [
            ("segment", dict(native_vision_model=str(outside))),
            ("segment", dict(native_vision_model="../outside.safetensors")),
            ("segment", dict(native_vision_model=str(self.files["RT_DETR_v4"]))),
            ("segment", dict(native_vision_model=None)),
            ("segment", dict(input_image=None)),
            ("segment", dict(input_image=huge)),
            ("segment", dict(inpaint_toggle=True)),
            ("segment", dict(segment_positive_points="[]")),
            ("segment", dict(segment_positive_points='[{"x":40,"y":5}]')),
            ("segment", dict(segment_positive_points='[{"x":-1,"y":5}]')),
            ("segment", dict(segment_positive_points='[{"x":true,"y":5}]')),
            ("segment", dict(segment_positive_points='[{"x":1,"y":5,"z":0}]')),
            ("segment", dict(segment_positive_points='[{"x":NaN,"y":5}]')),
            ("segment", dict(segment_positive_points='{"x":1,"y":5}')),
            ("segment", dict(segment_positive_points=json.dumps([{"x": 1, "y": 1}] * 257))),
            ("segment", dict(segment_negative_points="not json")),
            ("segment", dict(segment_refine_iterations=6)),
            ("segment", dict(segment_refine_iterations=1.5)),
            ("segment", dict(segment_output="boxes")),
            ("segment", dict(vision_threshold=1.01)),
            ("detect", dict(detect_class="dragon")),
            ("detect", dict(detect_max_count=0)),
            ("detect", dict(detect_max_count=1001)),
        ]
        for kind, values in cases:
            with self.subTest(kind=kind, values=str(values)[:60]), self.assertRaises(ValueError):
                self.request(kind, **values)

    def native_modules(self, kind, config):
        """Pinned node bodies with tiny CPU stand-ins for the model and device services."""
        from comfy import model_detection

        comfy = NS(
            model_detection=model_detection,
            model_management=NS(
                load_model_gpu=Mock(),
                get_torch_device=lambda: torch.device("cpu"),
                intermediate_device=lambda: torch.device("cpu"),
                unload_model_and_clones=Mock(),
            ),
            utils=NS(
                common_upscale=lambda image, w, h, *_, **__: F.interpolate(image, size=(h, w)),
                ProgressBar=lambda total: NS(update=lambda step: None),
            ),
        )
        io = NS(NodeOutput=lambda *values: values)
        scope = dict(comfy=comfy, io=io, torch=torch, F=F, json=json, COCO_CLASSES=COCO)
        self.drawn = []

        def draw(bboxes, image=None):
            self.drawn.append(bboxes)
            return (image * 0.5,)

        detect = native_method("comfy_extras/nodes_rtdetr.py", "RTDETR_detect", "execute", scope)
        segment = native_method("comfy_extras/nodes_sam3.py", "SAM3_Detect", "execute", scope)
        self.segment_calls = []

        def forward_segment(frame, point_inputs=None, mask_inputs=None):
            self.segment_calls.append(point_inputs if mask_inputs is None else "refine")
            logits = torch.full((1, 1, 8, 8), -1.0)
            logits[..., :4] = 1.0  # left half selected
            return logits

        network = Mock(
            side_effect=lambda image, size: [
                dict(
                    scores=torch.tensor([0.9, 0.8, 0.2]),
                    labels=torch.tensor([1, 0, 1]),
                    boxes=torch.tensor([[1.0, 2, 11, 12], [0, 0, 5, 5], [0, 0, 1, 1]]),
                )
            ]
        )
        network.forward_segment = forward_segment
        self.model = NS(
            model=NS(
                model_config=type(config, (), {})(),
                diffusion_model=network,
                get_dtype=lambda: torch.float32,
            )
        )
        self.load = Mock(return_value=self.model)
        self.comfy = comfy
        held = []

        @contextmanager
        def lease(data, role, name, manager, category):
            self.assertEqual((role, category), ("native-vision", "checkpoints"))
            held.append(name)
            try:
                yield Path(name)
            finally:
                held.pop()

        self.held = held
        self.enterContext(
            patch.dict(
                "sys.modules",
                {
                    "comfy": NS(
                        sd=NS(load_diffusion_model=self.load), model_detection=model_detection
                    ),
                    "comfy.sd": NS(load_diffusion_model=self.load),
                    "comfy.model_management": comfy.model_management,
                    "comfy_extras.nodes_rtdetr": NS(
                        RTDETR_detect=detect, DrawBBoxes=NS(execute=draw), COCO_CLASSES=COCO
                    ),
                    "comfy_extras.nodes_sam3": NS(SAM3_Detect=segment),
                },
            )
        )
        self.enterContext(
            patch("ruinedfooocus.features.generation.image_preparation.prepared_asset", lease)
        )

    def test_actual_detect_node_filters_draws_and_keeps_alpha_and_cache(self):
        self.native_modules("detect", "RT_DETR_v4")
        data = self.request("detect", detect_class="cat", vision_threshold=0.5)
        owner = NS()
        result = native.process(owner, data, self.manager, Mock())
        self.assertEqual((result.mode, result.size), ("RGBA", self.image.size))
        self.assertEqual(result.getpixel((0, 0)), (100, 50, 10, 128))
        (boxes,) = self.drawn
        self.assertEqual([box["label"] for box in boxes[0]], ["cat"])
        source = data["native_vision_source"]
        self.assertEqual((source["native"], source["count"]), ("RT_DETR_v4", 1))
        self.assertEqual(source["name"], "RT_DETR_v4.safetensors")
        self.assertNotIn(str(self.root), json.dumps(source))
        native.process(owner, data, self.manager, Mock())
        self.load.assert_called_once()
        self.assertFalse(self.held)
        # A replaced file is a new capture and a new load; the old weights are released.
        self.files["RT_DETR_v4"].write_bytes(self.files["RT_DETR_v4"].read_bytes() + b" ")
        native.process(owner, self.request("detect"), self.manager, Mock())
        self.assertEqual(self.load.call_count, 2)
        self.comfy.model_management.unload_model_and_clones.assert_called_once_with(self.model)

    def test_actual_point_node_builds_union_cutout_or_mask_at_image_size(self):
        self.native_modules("segment", "SAM3")
        data = self.request(
            "segment", segment_negative_points='[{"x":30,"y":5}]', segment_refine_iterations=3
        )
        cutout = native.process(NS(), data, self.manager, Mock())
        points = self.segment_calls[0]
        self.assertEqual(points["point_labels"].tolist(), [[1, 0]])
        self.assertEqual(points["point_coords"][0, 0].tolist(), [10 / 40 * 1008, 5 / 20 * 1008])
        self.assertEqual(self.segment_calls[1:], ["refine", "refine"])
        self.assertEqual((cutout.mode, cutout.size), ("RGBA", self.image.size))
        self.assertEqual(cutout.getpixel((2, 10)), (200, 100, 20, 128))
        self.assertEqual(cutout.getpixel((38, 10))[3], 0)
        data = self.request("segment", segment_output="mask")
        mask = native.process(NS(), data, self.manager, Mock())
        self.assertEqual((mask.mode, mask.size), ("L", self.image.size))
        self.assertEqual((mask.getpixel((2, 10)), mask.getpixel((38, 10))), (255, 0))
        self.assertEqual(data["native_vision_source"]["native"], "SAM3")

    def test_depth_map_captures_oriented_source_and_only_a_compatible_geometry_artifact(self):
        path = self.files["DepthAnything3"]
        self.assertEqual(
            native.native_model_choices(self.manager, "depth-map"),
            [
                ("DepthAnything3.safetensors · DepthAnything3", str(path)),
            ],
        )
        source = Image.new("RGB", (6, 4), (20, 80, 160))
        source.getexif()[274] = 6
        result = capture_image_assets(
            dict(
                input_image=source,
                native_vision_model=path.name,
                prompt="#!not a command",
                image_engine="sdcpp-vulkan",
            ),
            self.manager,
            {"type": "depth-map"},
            offline=True,
        )
        source.putpixel((0, 0), (0, 0, 0))
        self.assertEqual(result["input_image"].size, (4, 6))
        self.assertEqual(result["input_image"].getpixel((3, 0)), (20, 80, 160, 255))
        self.assertEqual(
            (result["prompt"], result["image_number"], result["_sampling_captured"]), ("", 1, False)
        )
        (item,) = result[KEY].artifacts
        self.assertEqual((item.request.destination, item.request.url), (path, ""))
        for selection in (None, "missing.safetensors", str(self.files["SAM3"])):
            with (
                self.subTest(selection=selection),
                self.assertRaisesRegex(ValueError, "installed Depth Anything 3"),
            ):
                self.request("depth-map", native_vision_model=selection)

        path.unlink()
        service = Mock()
        online = dict(result) | {KEY: replace(result[KEY], offline=False)}
        for candidate, message in ((result, "Offline mode"), (online, "was removed")):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex((ValueError, FileNotFoundError), message),
            ):
                prepare_image_assets(candidate, check_cancelled=Mock(), service=service)
        service.submit.assert_not_called()

    def test_depth_map_renders_native_output_as_white_near_grayscale_and_checks_cancellation(self):
        data = self.request("depth-map")
        path = self.files["DepthAnything3"]
        self.native_modules("depth-map", "DepthAnything3")
        infer = Mock(return_value=(object(),))
        low = torch.linspace(1, 0, self.image.width).view(1, 1, -1, 1)
        render = Mock(return_value=(low.expand(1, self.image.height, self.image.width, 3),))

        @contextmanager
        def lease(data, role, name, manager, category):
            self.assertEqual(
                (role, name, category), ("native-vision", str(path), "geometry_estimation")
            )
            with self.real_lease(data, role, name, manager, category) as captured:
                yield captured

        with (
            patch.dict(
                "sys.modules",
                {
                    "comfy_extras.nodes_depth_anything_3": NS(
                        DA3Inference=NS(execute=infer), DA3Render=NS(execute=render)
                    )
                },
            ),
            patch("ruinedfooocus.features.generation.image_preparation.prepared_asset", lease),
        ):
            owner = NS()
            result = native.process(owner, data, self.manager, Mock())
            self.load.assert_called_once_with(str(path))
            self.assertEqual((result.mode, result.size), ("L", self.image.size))
            self.assertEqual((result.getpixel((0, 0)), result.getpixel((39, 0))), (255, 0))
            self.assertEqual(infer.call_args.kwargs["mode"], {"mode": "mono"})
            self.assertEqual(infer.call_args.kwargs["resolution"], 504)
            self.assertEqual(
                render.call_args.kwargs["output"],
                dict(output="depth", normalization="v2_style", apply_sky_clip=False),
            )
            self.assertEqual(data["native_vision_source"]["format"], "png-gray8")
            self.assertNotIn(str(self.root), json.dumps(data["native_vision_source"]))
            precise = dict(
                data,
                output_format="exr",
                output_bit_depth="32-bit float",
                output_color_space="linear",
            )
            result = native.process(owner, precise, self.manager, Mock())
            self.assertEqual(result.dtype.name, "float32")
            torch.testing.assert_close(
                torch.from_numpy(result),
                low.expand(1, self.image.height, self.image.width, 3)[0, :, :, 0],
            )
            self.assertEqual(precise["native_vision_source"]["format"]["output_format"], "exr")
            render.reset_mock()
            cancelled = Mock(side_effect=[None, None, None, CancelledError("stop")])
            with self.assertRaises(CancelledError):
                native.process(owner, data, self.manager, cancelled)
            render.assert_not_called()
            render.return_value = (
                torch.full((1, self.image.height, self.image.width, 3), float("nan")),
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                native.process(owner, data, self.manager, Mock())

    def test_cancellation_and_wrong_loader_result_never_publish_or_leak_the_lease(self):
        self.native_modules("segment", "SAM3")
        data = self.request("segment")
        for calls in (1, 2):
            check = Mock(side_effect=[None] * (calls - 1) + [CancelledError("cancel")])
            with self.assertRaises(CancelledError):
                native.process(NS(), dict(data), self.manager, check)
            self.assertFalse(self.held)
        self.assertEqual(self.load.call_count, 1)
        self.assertFalse(self.segment_calls)
        self.model.model.model_config = type("SDXL", (), {})()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            native.process(NS(), dict(data), self.manager, Mock())
        owner = NS(_native_vision=("signature", "segment", self.model))
        native.clear_model(owner)
        self.assertIsNone(owner._native_vision)
        self.files["SAM3"].write_bytes(self.files["SAM3"].read_bytes() + b" ")
        with (
            patch(
                "ruinedfooocus.features.generation.image_preparation.prepared_asset",
                self.real_lease,
            ),
            self.assertRaisesRegex(ValueError, "changed after submission"),
        ):
            native.process(NS(), data, self.manager, Mock())

    def test_api_and_metadata_transport_the_typed_request_without_sampling_capture(self):
        import ruinedfooocus.app.api_runtime as api
        from ruinedfooocus.features.gallery.image_metadata import render_values
        from ruinedfooocus.integrations.openai_api import ImageRequest
        from pydantic import ValidationError
        from test_api_image_edits import queued_data

        groups = []
        original = api._image_operation

        def capture(*args, **kwargs):
            operation = original(*args, **kwargs)
            groups.append(operation.resource_group)
            return operation

        options = dict(
            native_vision_model=str(self.files["SAM31"]),
            segment_positive_points='[{"x":3,"y":4}]',
            segment_output="mask",
        )
        with (
            patch.object(api, "_image_operation", side_effect=capture),
            patch.object(native, "_roots", lambda manager, kind: (self.root / "diffusion_models",)),
        ):
            data = queued_data(
                "SDXL 1.0",
                controlnet={"type": "segment"},
                payload_options=options,
                source_image=self.image,
                # Points saved for another image must never reach an API request.
                defaults_options=dict(segment_negative_points='[{"x":1,"y":1}]'),
                recipe_resolver=NS(),  # Any generation-recipe lookup would fail.
            )
        self.assertEqual(groups, ["image"])
        self.assertIs(data["_sampling_captured"], False)
        self.assertEqual(data[KEY].artifacts[0].role, "native-vision")
        self.assertEqual(data["segment_output"], "mask")
        self.assertEqual(data["segment_negative_points"], "[]")
        self.assertEqual(data["input_image"].mode, "RGBA")
        request = ImageRequest(controlnet={"type": "detect"}, detect_max_count=5, **options)
        self.assertEqual(request.controlnet.type, "detect")
        for bad in (dict(segment_refine_iterations=6), dict(vision_threshold=2)):
            with self.assertRaises(ValidationError):
                ImageRequest(**bad)
        saved = dict(options, native_vision_source=dict(native="SAM31", name="SAM31.safetensors"))
        self.assertEqual({key: render_values(saved)[key] for key in saved}, saved)

    def test_public_api_selects_the_utility_without_any_generation_model(self):
        import base64
        from io import BytesIO
        from fastapi.testclient import TestClient
        from ruinedfooocus.integrations.openai_api import create_apps
        from test_api_live_contracts import job

        output = self.root / "output.png"
        self.image.save(output)
        encoded = BytesIO()
        self.image.save(encoded, format="PNG")
        backend = NS(
            images=Mock(return_value=job(("result", [str(output)]))),
            resolve_model=Mock(side_effect=AssertionError("generation model resolution")),
        )
        api, _ = create_apps(backend)
        with TestClient(api) as client:
            response = client.post(
                "/images/generations",
                json=dict(
                    model="missing-generation-model",
                    input_image=base64.b64encode(encoded.getvalue()).decode(),
                    controlnet={"type": "segment"},
                    native_vision_model=str(self.files["SAM31"]),
                    segment_positive_points='[{"x":1,"y":2}]',
                    response_format="b64_json",
                ),
            )
        self.assertEqual(response.status_code, 200, response.text)
        backend.resolve_model.assert_not_called()
        self.assertEqual(backend.images.call_args.args[1], {})

    def test_depth_api_captures_the_configured_model_and_oriented_source_before_queueing(self):
        import base64
        from io import BytesIO
        from ruinedfooocus.app import api_runtime as api

        source = Image.new("RGB", (6, 4), (20, 80, 160))
        source.getexif()[274] = 6
        raw = BytesIO()
        source.save(raw, format="PNG", exif=source.getexif())
        model = self.files["DepthAnything3"]
        control = {"type": "depth-map"}
        payload = dict(
            input_image=base64.b64encode(raw.getvalue()).decode(),
            native_vision_model=model.name,
            size="4x6",
            controlnet=control,
            prompt="not a sampling prompt",
            vision_threshold=0.9,
        )
        shared = NS(
            path_manager=self.manager,
            models=NS(offline=True),
            settings=NS(default_settings={"image_engine": "sdcpp-vulkan"}),
            state={},
        )
        worker = NS(
            prepare_image_data=Mock(side_effect=lambda data, **options: dict(data)),
            _process=Mock(return_value=[str(self.root / "result.png")]),
        )
        with (
            patch.dict(
                "sys.modules",
                {"ruinedfooocus.app.state": shared, "ruinedfooocus.app.async_worker": worker},
            ),
            patch.object(sys.modules["ruinedfooocus.app"], "state", shared, create=True),
            patch.object(sys.modules["ruinedfooocus.app"], "async_worker", worker, create=True),
            patch.object(api, "submit", side_effect=lambda run: run),
        ):
            run = api.images(payload, {})
            self.assertEqual(run.resource_group, "image")
            data = dict(
                zip(run.__code__.co_freevars, (cell.cell_contents for cell in run.__closure__))
            )["data"]
            self.assertEqual(data["input_image"].size, (4, 6))
            self.assertEqual(data["input_image"].getpixel((0, 0)), (20, 80, 160, 255))
            self.assertEqual((data["prompt"], data["_sampling_captured"]), ("", False))
            self.assertNotIn("vision_threshold", data)
            self.assertTrue(data[KEY].offline)
            (artifact,) = data[KEY].artifacts
            self.assertEqual(artifact.request.destination, model)
            worker.prepare_image_data.assert_not_called()
            job = object()
            run.prepare(job)
            self.assertEqual(run(job, 7), [str(self.root / "result.png")])
            request = worker._process.call_args.args[0]
            self.assertIs(request[KEY], data[KEY])
            self.assertIs(request["_api_job"], job)
            self.assertEqual(request["task_id"], 7)
            for fields, message in (
                (dict(size="6x4"), "oriented source size 4×6"),
                (dict(native_vision_model=str(self.files["SAM3"])), "installed Depth Anything 3"),
                (dict(mask="not a supported mode"), "do not accept masks"),
            ):
                with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, message):
                    api.images(payload | fields, {})

    def test_public_depth_map_accepts_json_and_multipart_and_returns_grayscale_png(self):
        import base64
        from io import BytesIO
        from fastapi.testclient import TestClient
        from ruinedfooocus.integrations.openai_api import create_apps
        from test_api_live_contracts import job

        output = self.root / "depth.png"
        Image.new("L", self.image.size, 180).save(output)
        raw = BytesIO()
        self.image.save(raw, format="PNG")
        backend = NS(
            images=Mock(side_effect=lambda *args: job(("result", [str(output)]))),
            resolve_model=Mock(side_effect=AssertionError("generation model resolution")),
        )
        api, _ = create_apps(backend)
        fields = dict(
            model="unused",
            native_vision_model="DepthAnything3.safetensors",
            response_format="b64_json",
        )
        payload = dict(
            fields,
            input_image=base64.b64encode(raw.getvalue()).decode(),
            controlnet={"type": "depth-map"},
        )
        with TestClient(api) as client:
            for response in (
                client.post("/images/generations", json=payload),
                client.post(
                    "/images/edits",
                    files={"image": ("source.png", raw.getvalue(), "image/png")},
                    data=dict(fields, controlnet='{"type":"depth-map"}'),
                ),
            ):
                self.assertEqual(response.status_code, 200, response.text)
                result = base64.b64decode(response.json()["data"][0]["b64_json"])
                with Image.open(BytesIO(result)) as decoded:
                    self.assertEqual(
                        (decoded.format, decoded.mode, decoded.size), ("PNG", "L", self.image.size)
                    )
            self.assertEqual(backend.images.call_count, 2)
            self.assertTrue(all(call.args[1] == {} for call in backend.images.call_args_list))
            self.assertTrue(
                all(
                    call.args[0]["controlnet"] == {"type": "depth-map"}
                    for call in backend.images.call_args_list
                )
            )
            backend.images.reset_mock()
            for invalid in (
                dict(n=2),
                dict(input_image=None),
                dict(native_vision_model=None),
                dict(controlnet={"type": "depth-map", "strength": 0.5}),
            ):
                self.assertEqual(
                    client.post("/images/generations", json=payload | invalid).status_code, 400
                )
            self.assertEqual(client.post("/videos/generations", json=payload).status_code, 404)
            backend.images.assert_not_called()
        backend.resolve_model.assert_not_called()

    def test_actual_worker_route_and_png_output_bypass_prompt_checkpoint_and_sampling(self):
        for kind, model in (("segment", "SAM31"), ("depth-map", "DepthAnything3")):
            with self.subTest(kind=kind):
                self._worker_output(kind, model)

    def test_worker_still_path_result_is_exported_in_the_requested_precision_format(self):
        from test_native_image_formats import native_codecs, native_preview
        from ruinedfooocus.features.image_processing import native_image_formats

        with (
            patch.object(native_image_formats, "_native", return_value=native_codecs()),
            patch.dict(
                sys.modules,
                {
                    "comfy_execution.preview_tonemap": native_preview(),
                    native_image_formats.__name__: native_image_formats,
                },
            ),
        ):
            self._worker_output("depth-map", "DepthAnything3", precision_path=True)

    def _worker_output(self, kind, model, precision_path=False):
        from ruinedfooocus.features.image_processing.native_vision_pipeline import pipeline
        from ruinedfooocus.core.paths import resource_path
        from PIL.PngImagePlugin import PngInfo

        pipe = pipeline()

        def process(gen_data, callback):
            gen_data["native_vision_source"] = dict(native=model, name=f"{model}.safetensors")
            if precision_path:
                source = self.root / "owned-pipeline-result.png"
                self.image.convert("L").save(source)
                return [str(source)]
            return [self.image.convert("L") if kind == "depth-map" else self.image.copy()]

        pipe.process = Mock(side_effect=process)
        controls = NS(get_settings=lambda data: data.get("controlnet", {}))
        forbidden = Mock(side_effect=AssertionError("generation dependency accessed"))
        pipe.load_base_model = forbidden
        shared = NS(
            models=NS(get_models_by_path=forbidden, get_model_base=forbidden),
            state={},
            performance_settings=NS(apply=forbidden),
            resolution_settings=object(),
            path_manager=NS(model_paths={"temp_outputs_path": self.root}),
            shared_cache={},
        )
        route = dict(
            state={"pipeline": None},
            shared=shared,
            controlnet=controls,
            native_vision_pipeline=NS(pipeline=lambda: pipe),
        )
        function("ruinedfooocus/features/generation/pipelines.py", "selected_pipeline", route)
        update = function("ruinedfooocus/features/generation/pipelines.py", "update", route)
        self.assertIs(update(dict(prompt="#!ignored", controlnet={"type": kind})), pipe)
        render = function(
            "ruinedfooocus/app/async_worker.py",
            "_process",
            dict(
                shared=shared,
                settings=NS(default_settings={}),
                control_tools=controls,
                check_interrupt=lambda data: None,
                process_metadata=forbidden,
                pipelines=NS(update=update),
                get_checkpoint_hashes=forbidden,
                process_prompt=forbidden,
                shift_attention=forbidden,
                parse_loras=forbidden,
                merge_loras=forbidden,
                TimeIt=lambda name: nullcontext(),
                resource_path=resource_path,
                random=random,
                Path=Path,
                pathlib=pathlib,
                Image=Image,
                PngInfo=PngInfo,
                json=json,
                generate_temp_filename=lambda **kwargs: str(
                    self.root / f"render.{kwargs['extension']}"
                ),
            ),
        )
        data = capture_image_assets(
            dict(
                input_image=self.image,
                native_vision_model=str(self.files[model]),
                segment_positive_points='[{"x":1,"y":2}]',
                prompt="#!ignored",
                style_selection=["Style: enhance"],
                OBP_promptenhance="llama",
                task_id=1,
                task_type="tool_call",
                silent=True,
                index=(0, 1),
            ),
            self.manager,
            {"type": kind},
        )
        if precision_path:
            data.update(
                output_format="exr", output_bit_depth="32-bit float", output_color_space="sRGB"
            )
        result = render(data)
        if precision_path:
            from ruinedfooocus.features.image_processing.native_image_outputs import (
                NativeImageOutput,
            )
            from ruinedfooocus.features.image_processing.native_image_formats import read_image

            self.assertIsInstance(result[0], NativeImageOutput)
            output = read_image(result[0].original)
            self.assertEqual(output.pixels.dtype.name, "float32")
            self.assertEqual(output.pixels.shape, (self.image.height, self.image.width))
            self.assertEqual(shared.state["last_image"], result[0].original)
            self.assertNotEqual(result[0].display, result[0].original)
            self.assertEqual(
                json.loads(output.metadata["parameters"])["output_source_dtype"], "uint8"
            )
            forbidden.assert_not_called()
            return
        self.assertEqual(result, [str(self.root / "render.png")])
        with Image.open(result[0]) as output:
            self.assertEqual(output.mode, "L" if kind == "depth-map" else "RGBA")
            metadata = json.loads(output.info["parameters"])
        self.assertEqual(metadata["controlnet"]["type"], kind)
        self.assertEqual(metadata["native_vision_source"]["native"], model)
        if kind == "depth-map":
            self.assertNotIn(str(self.root), json.dumps(metadata))
        self.assertNotIn("base_model_name", metadata)
        self.assertNotIn("steps", metadata)
        forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
