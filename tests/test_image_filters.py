"""Native filter math, immutable capture and actual UI/API/worker output closure."""

import asyncio
import base64
from concurrent.futures import CancelledError
from contextlib import nullcontext
from io import BytesIO
import json
import pathlib
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import gradio as gr
from gradio.state_holder import SessionState
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import torch
import torch.nn.functional as F

import ruinedfooocus.app
from ruinedfooocus.app import api_runtime
from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.image_processing import image_filters as filters
from ruinedfooocus.features.image_processing import still_image_utilities as dispatch
from ruinedfooocus.features.image_processing import ui_image_filters as ui
from ruinedfooocus.integrations.openai_api import ImageRequest, create_apps
from test_api_live_contracts import job
from test_depth_utilities import NodeOutput as NativeOutput, native_functions
from test_native_model_coverage import native_method
from test_utility_sampling_capture import function


class ImageFilters(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.manager = NS(get_model_folders=Mock(side_effect=AssertionError("model folder lookup")))
        self.memory = NS(
            get_torch_device=Mock(return_value="cpu"),
            intermediate_device=lambda: "cpu",
            unload_model_and_clones=Mock(),
        )
        scope = {
            "torch": torch,
            "F": F,
            "comfy": NS(model_management=self.memory),
            "io": NS(NodeOutput=NativeOutput),
        }
        native_functions("comfy_extras/nodes_post_processing.py", {"gaussian_kernel"}, scope)
        self.blur = native_method("comfy_extras/nodes_post_processing.py", "Blur", "execute", scope)
        self.sharpen = native_method(
            "comfy_extras/nodes_post_processing.py", "Sharpen", "execute", scope
        )
        self.brightness = native_method(
            "comfy_extras/nodes_dataset.py", "AdjustBrightnessNode", "_process", scope
        )
        self.contrast = native_method(
            "comfy_extras/nodes_dataset.py", "AdjustContrastNode", "_process", scope
        )
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "comfy_extras.nodes_post_processing": NS(Blur=self.blur, Sharpen=self.sharpen),
                    "comfy_extras.nodes_dataset": NS(
                        AdjustBrightnessNode=self.brightness, AdjustContrastNode=self.contrast
                    ),
                    "comfy.model_management": self.memory,
                },
            )
        )
        pixels = np.arange(4 * 5 * 4, dtype=np.uint8).reshape(4, 5, 4) * 3
        self.source = Image.fromarray(pixels, "RGBA")

    def capture(self, kind, **options):
        data = dict(
            input_image=self.source,
            controlnet={"type": kind},
            prompt="#!not-a-command",
            image_engine="sdcpp-vulkan",
            **options,
        )
        return assets.capture_image_assets(data, self.manager, data["controlnet"], offline=True)

    def test_all_filters_freeze_pixels_and_need_no_component_or_generation_recipe(self):
        for kind in filters.KINDS:
            data = self.capture(kind)
            captured = data[filters.KEY]
            data["input_image"].paste((0, 0, 0, 0), (0, 0, 5, 4))
            self.assertEqual(captured.source.rgba, self.source.tobytes())
            self.assertEqual(data[assets.KEY].artifacts, ())
            self.assertEqual(
                (
                    data["prompt"],
                    data["image_number"],
                    data["automatic_upscale"],
                    data["_sampling_captured"],
                ),
                ("", 1, "Off", False),
            )
            result = dispatch.process(NS(), data, self.manager, Mock())[0]
            self.assertEqual((result.size, result.mode), (self.source.size, "RGBA"))
            self.assertEqual(
                result.getchannel("A").tobytes(), self.source.getchannel("A").tobytes()
            )
        self.manager.get_model_folders.assert_not_called()

    def test_actual_native_pure_adjustments_have_expected_pixels_and_no_device_lookup(self):
        for kind, factor, expected in (("adjust-brightness", 0, 0), ("adjust-contrast", 0, 128)):
            result = filters.process(
                NS(), self.capture(kind, filter_factor=factor), self.manager, Mock()
            )[0]
            self.assertTrue(np.all(np.asarray(result)[..., :3] == expected))
        for kind in ("adjust-brightness", "adjust-contrast"):
            data = self.capture(kind, filter_factor=1)
            data["filter_factor"] = 0  # A later UI edit cannot change captured intent.
            result = filters.process(NS(), data, self.manager, Mock())[0]
            self.assertEqual(result.tobytes(), self.source.tobytes())
        self.memory.get_torch_device.assert_not_called()

    def test_only_pure_dataset_adjustments_are_in_the_still_image_domain(self):
        from tools.audit_native_model_coverage import node_domain

        for node in ("AdjustBrightness", "AdjustContrast"):
            self.assertEqual(
                node_domain(node, "comfy_extras/nodes_dataset.py", "image/adjustments"), "image"
            )
        self.assertEqual(
            node_domain("ShuffleDataset", "comfy_extras/nodes_dataset.py", "image/batch"),
            "training",
        )

    def test_actual_native_blur_and_sharpen_use_native_cpu_device_standin_and_preserve_alpha(self):
        self.source = Image.new("RGBA", (5, 5), (0, 0, 0, 37))
        self.source.putpixel((2, 2), (255, 255, 255, 233))
        blurred = filters.process(NS(), self.capture("image-blur"), self.manager, Mock())[0]
        self.assertGreater(blurred.getpixel((2, 2))[0], 0)
        self.assertLess(blurred.getpixel((2, 2))[0], 255)
        self.assertGreater(blurred.getpixel((1, 2))[0], 0)
        self.assertEqual(blurred.getchannel("A").tobytes(), self.source.getchannel("A").tobytes())
        sharpened = filters.process(
            NS(), self.capture("image-sharpen", filter_strength=0), self.manager, Mock()
        )[0]
        self.assertEqual(sharpened.tobytes(), self.source.tobytes())
        self.assertEqual(self.memory.get_torch_device.call_count, 2)

    def test_capture_bounds_finite_params_and_reflect_padding_are_explicit(self):
        for options in (
            {"filter_radius": 4},
            {"filter_radius": True},
            {"filter_radius": 0},
            {"filter_sigma": float("inf")},
            {"filter_sigma": 0.01},
            {"filter_radius": 1.5},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.capture("image-blur", **options)
        for field, kind, value in (
            ("filter_strength", "image-sharpen", -1),
            ("filter_factor", "adjust-contrast", 3),
            ("filter_factor", "adjust-brightness", float("nan")),
        ):
            with self.assertRaises(ValueError):
                self.capture(kind, **{field: value})
        for option in (
            {"inpaint_toggle": True},
            {"model_recipe_path": "recipe.yaml"},
            {"reference_adapter": "style"},
        ):
            with self.assertRaises(ValueError):
                filters.capture_assets(
                    dict(
                        input_image=self.source, controlnet={"type": "adjust-brightness"}, **option
                    ),
                    self.manager,
                )
        # The shared utility boundary already omits unrelated hidden generation
        # preferences; an explicit filter needs none of those resources.
        data = self.capture(
            "adjust-brightness", model_recipe_path="missing.yaml", reference_adapter="style"
        )
        self.assertNotIn("model_recipe_path", data)
        self.assertNotIn("reference_adapter", data)

    def test_cancel_and_native_nonfinite_or_size_errors_never_publish_provenance(self):
        data = self.capture("image-blur")
        for output in (torch.zeros(1, 3, 5, 3), torch.full((1, 4, 5, 3), float("nan"))):
            with (
                patch.object(self.blur, "execute", return_value=(output,)),
                self.assertRaises(ValueError),
            ):
                filters.process(NS(), data, self.manager, Mock())
            self.assertNotIn("image_utility_source", data)
        with self.assertRaises(CancelledError):
            filters.process(NS(), data, self.manager, Mock(side_effect=CancelledError()))
        checks = Mock(side_effect=[None, None, CancelledError()])
        with self.assertRaises(CancelledError):
            filters.process(NS(), data, self.manager, checks)
        self.assertNotIn("image_utility_source", data)

    def test_precision_output_metadata_replay_and_previous_native_model_cleanup(self):
        data = self.capture(
            "image-sharpen",
            output_format="exr",
            output_bit_depth="32-bit float",
            output_color_space="sRGB",
            filter_strength=0,
        )
        patcher = object()
        owner = NS(_depth_utility_cache=(None, None, (patcher,)))
        result = dispatch.process(owner, data, self.manager, Mock())[0]
        self.memory.unload_model_and_clones.assert_called_once_with(patcher)
        self.assertEqual((result.shape, result.dtype), ((4, 5, 4), np.dtype("float32")))
        metadata = dispatch.output_metadata(data)
        self.assertEqual(metadata["image_utility_source"]["native_node"], "ImageSharpen")
        self.assertEqual(
            metadata["image_utility_source"]["source_sha256"],
            data[filters.KEY].source.source_sha256,
        )
        self.assertNotIn("base_model_name", metadata)
        replay = dispatch.replay_values(metadata)
        self.assertIsNone(replay["input_image"])
        self.assertIsNone(replay["image_utility_model"])
        self.assertEqual(replay["filter_strength"], 0)
        self.assertEqual(replay["cn_selection"], "Sharpen image")

    def test_public_json_multipart_and_request_bounds_bypass_generation_model(self):
        buffer = BytesIO()
        self.source.save(buffer, format="PNG")
        output = self.root / "output.png"
        self.source.save(output)
        backend = NS(
            images=Mock(side_effect=lambda *args: job(("result", [str(output)]))),
            resolve_model=Mock(side_effect=AssertionError("generation model")),
        )
        api, _ = create_apps(backend)
        with TestClient(api) as client:
            for kind in filters.KINDS:
                body = dict(
                    model="missing-generation-checkpoint",
                    input_image=base64.b64encode(buffer.getvalue()).decode(),
                    controlnet={"type": kind},
                    filter_radius=1,
                    filter_factor=0,
                    response_format="b64_json",
                )
                response = client.post("/images/generations", json=body)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(backend.images.call_args.args[1], {})
                form = {
                    key: str(value)
                    for key, value in body.items()
                    if key not in ("input_image", "controlnet")
                }
                form["controlnet"] = json.dumps(body["controlnet"])
                response = client.post(
                    "/images/edits",
                    data=form,
                    files={"image": ("source.png", buffer.getvalue(), "image/png")},
                )
                self.assertEqual(response.status_code, 200, response.text)
                for override in (
                    {"n": 2},
                    {"input_image": None},
                    {"filter_radius": True},
                    {"filter_factor": 3},
                ):
                    self.assertEqual(
                        client.post("/images/generations", json=body | override).status_code, 400
                    )
        backend.resolve_model.assert_not_called()
        self.assertEqual(
            ImageRequest(
                controlnet={"type": "adjust-contrast"}, input_image="upload", filter_factor=0
            ).filter_factor,
            0,
        )

    def test_actual_api_capture_freezes_before_preparation_and_rejects_resize_or_highbit(self):
        raw = BytesIO()
        self.source.save(raw, format="PNG")
        shared = NS(
            path_manager=self.manager,
            models=NS(offline=True),
            settings=NS(default_settings={}),
            state={},
        )
        worker = NS(
            prepare_image_data=Mock(side_effect=lambda data, **kw: dict(data)),
            _process=Mock(return_value=[str(self.root / "result.png")]),
        )
        with (
            patch.dict(
                sys.modules,
                {"ruinedfooocus.app.state": shared, "ruinedfooocus.app.async_worker": worker},
            ),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(ruinedfooocus.app, "async_worker", worker, create=True),
            patch.object(api_runtime, "submit", side_effect=lambda run: run),
        ):
            for kind in filters.KINDS:
                payload = dict(
                    input_image=base64.b64encode(raw.getvalue()).decode(),
                    controlnet={"type": kind},
                    filter_factor=0,
                )
                operation = api_runtime.images(payload, {})
                operation.prepare(object())
                operation(None, 3)
                data = worker._process.call_args.args[0]
                self.assertIsInstance(data[filters.KEY], filters.FilterRequest)
                self.assertEqual(data["filter_factor"], 0)
                for override in (
                    {"size": "20x20"},
                    {"reference_images": [payload["input_image"]]},
                    {"mask": payload["input_image"]},
                ):
                    with self.assertRaises(ValueError):
                        api_runtime.images(payload | override, {})
            high = BytesIO()
            Image.new("I;16", (2, 2)).save(high, format="PNG")
            with self.assertRaisesRegex(ValueError, "8-bit"):
                api_runtime.images(
                    dict(
                        input_image=base64.b64encode(high.getvalue()).decode(),
                        controlnet={"type": "adjust-brightness"},
                    ),
                    {},
                )

    def test_real_gradio_context_and_library_replay_cannot_overwrite_source_or_resolve_model(self):
        controls = {}
        shared = NS(
            settings=NS(default_settings={"filter_factor": 0}),
            add_ctrl=lambda key, control: controls.update({key: control}),
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
        ):
            with gr.Blocks(analytics_enabled=False) as demo:
                choice = gr.Textbox(value="None")
                tool = gr.Textbox(value="None")
                ui.create_ui(choice, tool, lambda choice, tool: tool)
        event_index = next(
            index for index, event in demo.fns.items() if event.fn.__name__ == "filter_context"
        )
        self.assertEqual(controls["filter_factor"].value, 0)
        self.assertEqual(set(controls), set(filters.FIELDS))
        for kind in filters.KINDS:
            data = asyncio.run(
                demo.process_api(event_index, [kind, kind], state=SessionState(demo))
            )["data"]
            self.assertEqual(
                [item["visible"] for item in data],
                [
                    True,
                    kind in ("image-blur", "image-sharpen"),
                    kind == "image-sharpen",
                    kind.startswith("adjust-"),
                ],
            )
        from ruinedfooocus.features.gallery.image_metadata import read_metadata

        forbidden = Mock(side_effect=AssertionError("generation replay"))
        scope = dict(
            gr=gr,
            json=json,
            get_ctrl=lambda key: key,
            read_metadata=read_metadata,
            resolve_setup=forbidden,
        )
        apply = function("ruinedfooocus/features/gallery/ui.py", "apply_metadata", scope)
        scope["apply_metadata"] = apply
        reuse = function("ruinedfooocus/features/gallery/ui.py", "reuse_setup", scope)
        for kind in filters.KINDS:
            result = reuse(
                json.dumps({"prompt": "", "controlnet": {"type": kind}, "filter_factor": 0})
            )
            self.assertIsNone(result["input_image"]["value"])
            self.assertNotIn("base_model_name", result)
        forbidden.assert_not_called()

    def test_actual_serial_pipeline_writes_source_sized_rgba_png_and_metadata(self):
        from ruinedfooocus.features.image_processing.depth_utility_pipeline import pipeline

        forbidden = Mock(side_effect=AssertionError("generation dependency"))
        shared = NS(
            models=NS(get_models_by_path=forbidden, get_model_base=forbidden),
            state={},
            performance_settings=NS(apply=forbidden),
            resolution_settings=object(),
            path_manager=NS(model_paths={"temp_outputs_path": self.root}),
            shared_cache={},
        )
        control = NS(get_settings=lambda data: data.get("controlnet", {}))
        worker = NS(add_result=Mock(), check_interrupt=lambda data: None)
        with (
            patch.dict(
                sys.modules,
                {"ruinedfooocus.app.state": shared, "ruinedfooocus.app.async_worker": worker},
            ),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(ruinedfooocus.app, "async_worker", worker, create=True),
        ):
            for kind in filters.KINDS:
                data = self.capture(
                    kind, task_id=1, task_type="tool_call", silent=True, index=(0, 1)
                )
                pipe = pipeline()
                route = dict(state={"pipeline": None}, shared=shared, controlnet=control)
                function(
                    "ruinedfooocus/features/generation/pipelines.py", "selected_pipeline", route
                )
                update = function("ruinedfooocus/features/generation/pipelines.py", "update", route)
                with patch(
                    "ruinedfooocus.features.image_processing.depth_utility_pipeline.pipeline",
                    return_value=pipe,
                ):
                    render = function(
                        "ruinedfooocus/app/async_worker.py",
                        "_process",
                        dict(
                            shared=shared,
                            settings=NS(default_settings={}),
                            control_tools=control,
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
                            generate_temp_filename=lambda **kw: str(
                                self.root / (kind + "." + kw["extension"])
                            ),
                        ),
                    )
                    output = render(data)[0]
                with Image.open(output) as image:
                    self.assertEqual((image.size, image.mode), (self.source.size, "RGBA"))
                    self.assertEqual(
                        image.getchannel("A").tobytes(), self.source.getchannel("A").tobytes()
                    )
                    metadata = json.loads(image.info["parameters"])
                self.assertEqual(metadata["controlnet"], {"type": kind})
                self.assertEqual(
                    metadata["image_utility_source"]["native_node"], filters.NODES[kind]
                )
                self.assertNotIn("base_model_name", metadata)
                self.assertNotIn("steps", metadata)
                self.assertNotIn(str(self.root), json.dumps(metadata))
        forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
