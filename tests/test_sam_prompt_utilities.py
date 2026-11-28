"""SAM still-image text/boxes/instances use pinned native calls and one image run."""

import ast
import base64
from concurrent.futures import CancelledError
from contextlib import nullcontext
from io import BytesIO
from itertools import count
import json
import pathlib
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ruinedfooocus.features.generation.image_output_batches import (
    ImageOutputBatch,
    ImageBatchArtifacts,
    PLAN_KEY,
    validate_artifacts,
)
from ruinedfooocus.features.image_processing import native_vision as vision, sam_prompts as sam
from test_native_model_coverage import NATIVE, native_method
from test_native_variant_defaults import write_header
from test_utility_sampling_capture import function
import test_native_vision as fixtures


class SamPromptTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.NativeVisionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.image = self.fixture.image

    def paired_header(self):
        path = self.fixture.files["SAM3"]
        write_header(
            path,
            fixtures.HEADERS["SAM3"]
            | {
                "detector.backbone.language_backbone.encoder.token_embedding.weight": [49408, 1024],
                "detector.backbone.language_backbone.encoder.positional_embedding": [32, 1024],
            },
        )
        return path

    def request(self, **values):
        return self.fixture.request("segment", segment_positive_points="[]", **values)

    def runtime(self, *, scores=(2.0, 1.0)):
        self.fixture.native_modules("segment", "SAM3")
        node = sys.modules["comfy_extras.nodes_sam3"].SAM3_Detect
        scope = node.execute.__func__.__globals__
        tree = ast.parse((NATIVE / "comfy_extras/nodes_sam3.py").read_text(encoding="utf-8"))
        functions = [
            item
            for item in tree.body
            if isinstance(item, ast.FunctionDef)
            and item.name in ("_extract_text_prompts", "_refine_mask")
        ]
        exec(compile(ast.Module(body=functions, type_ignores=[]), "nodes_sam3.py", "exec"), scope)
        outer = self
        self.text_calls, self.box_calls = [], []

        class Network:
            def __call__(self, frame, **kwargs):
                outer.text_calls.append(kwargs)
                masks = torch.full((1, 2, 4, 4), -1.0)
                masks[0, 0, :, :2] = 1
                masks[0, 1, :, 2:] = 1
                return dict(
                    boxes=torch.tensor([[[0, 0, 20, 20], [20, 0, 40, 20]]], dtype=torch.float32),
                    scores=torch.tensor([scores], dtype=torch.float32),
                    masks=masks,
                )

            def forward_segment(self, frame, point_inputs=None, box_inputs=None, mask_inputs=None):
                outer.box_calls.append(box_inputs)
                logits = torch.full((1, 1, 4, 4), -1.0)
                logits[..., :2] = 1
                return logits

        self.fixture.model.model.diffusion_model = Network()
        self.encoded = []
        self.clip = NS(patcher=object())

        def tokenize(text):
            self.encoded.append(text)
            return sam._text_parser()(text)

        def encode(categories):
            return [
                [
                    torch.zeros(1, 32, 1024),
                    {
                        "sam3_multi_cond": [
                            dict(
                                cond=torch.zeros(1, 32, 1024),
                                attention_mask=None,
                                max_detections=count,
                            )
                            for _, count in categories
                        ]
                    },
                ]
            ]

        self.clip.tokenize, self.clip.encode_from_tokens_scheduled = tokenize, encode
        self.load_pair = Mock(return_value=(self.fixture.model, self.clip, None, None))
        sys.modules["comfy.sd"].load_checkpoint_guess_config = self.load_pair
        sys.modules["comfy"].sd.load_checkpoint_guess_config = self.load_pair
        encoder = native_method("nodes.py", "CLIPTextEncode", "encode", {})
        self.enterContext(patch.dict(sys.modules, {"nodes": NS(CLIPTextEncode=encoder)}))

    def test_capture_literal_text_and_boxes_without_generation_or_alternate_encoder(self):
        self.paired_header()
        text = '#!{"prompt":"literal"}:2'
        data = self.request(
            segment_text=text,
            segment_boxes='[{"x":1,"y":2,"width":12,"height":8}]',
            segment_individual_masks=True,
            prompt="/search: must not run",
        )
        self.assertEqual(data["segment_text"], text)
        self.assertEqual(json.loads(data["segment_boxes"]), [dict(x=1, y=2, width=12, height=8)])
        self.assertEqual(
            (data["image_number"], data["prompt"], data["_sampling_captured"]), (1, "", False)
        )
        self.assertEqual(data[PLAN_KEY].kind, "sam-instances")
        self.assertEqual(data[PLAN_KEY].min_per_generation, 0)
        self.assertNotIn("segment_text", vision.worker_data(dict(data))["prompt"])

    def test_text_requires_the_selected_sam_source_language_backbone(self):
        with self.assertRaisesRegex(ValueError, "paired language encoder"):
            self.request(segment_text="cat:2")
        self.paired_header()
        self.assertEqual(self.request(segment_text="cat:2")["segment_text"], "cat:2")

    def test_unsafe_boxes_prompt_counts_and_aggregate_pixels_fail_before_native_loading(self):
        self.paired_header()
        for values in (
            dict(segment_boxes='[{"x":-1,"y":0,"width":1,"height":1}]'),
            dict(segment_boxes='[{"x":39,"y":0,"width":2,"height":1}]'),
            dict(segment_boxes='[{"x":0,"y":0,"width":0,"height":1}]'),
            dict(segment_boxes='[{"x":false,"y":0,"width":1,"height":1}]'),
            dict(segment_boxes='[{"x":NaN,"y":0,"width":1,"height":1}]'),
            dict(segment_boxes='[{"x":0,"y":0,"width":1,"height":1,"score":1}]'),
            dict(segment_text="cat:65"),
            dict(segment_text="cat:2", segment_max_instances=1),
            dict(segment_text="cat", segment_individual_masks="true"),
            dict(segment_text="cat", segment_max_instances=True),
            dict(segment_text="cat\x00"),
            dict(segment_text=","),
            dict(segment_text="cat," * 17),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.request(**values)
        with patch.object(sam, "MAX_MASK_PIXELS", 2 * self.image.width * self.image.height):
            with self.assertRaisesRegex(ValueError, "aggregate pixel"):
                self.request(segment_text="cat:3")
            self.assertEqual(self.request(segment_text="cat:2")["_sam_mask_limit"], 2)
        with patch.object(sam, "MAX_OUTPUT_BYTES", 2 * self.image.width * self.image.height * 16):
            with self.assertRaisesRegex(ValueError, "aggregate pixel"):
                self.request(
                    segment_text="cat:3", output_format="exr", output_bit_depth="32-bit float"
                )

    def test_actual_native_box_path_returns_ordered_instance_cutouts_with_original_alpha(self):
        self.runtime()
        boxes = [dict(x=0, y=0, width=20, height=20), dict(x=20, y=0, width=20, height=20)]
        data = self.request(
            segment_boxes=json.dumps(boxes),
            segment_individual_masks=True,
            segment_refine_iterations=0,
        )
        result = vision.process(NS(), data, self.fixture.manager, Mock())
        self.assertIsInstance(result, ImageOutputBatch)
        self.assertEqual(result.plan, data[PLAN_KEY])
        self.assertEqual(len(result), 2)
        self.assertEqual(data["image_number"], 1)
        self.assertEqual(
            [row["native_vision_source"]["instance_index"] for row in result.metadata], [0, 1]
        )
        self.assertEqual(result[0].getpixel((2, 10))[3], 128)
        self.assertEqual(result[0].getpixel((38, 10))[3], 0)
        self.assertEqual(len(self.box_calls), 2)
        self.assertEqual(self.box_calls[0].tolist(), [[[0, 0], [504, 1008]]])
        self.assertEqual(self.box_calls[1].tolist(), [[[504, 0], [1008, 1008]]])
        self.fixture.load.assert_called_once()
        self.load_pair.assert_not_called()

    def test_actual_native_text_encoder_and_detector_keep_literal_text_and_paired_cache(self):
        self.paired_header()
        self.runtime()
        data = self.request(
            segment_text="#!cat:2", segment_individual_masks=True, segment_refine_iterations=0
        )
        owner = NS()
        result = vision.process(owner, data, self.fixture.manager, Mock())
        self.assertEqual(self.encoded, ["#!cat:2"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].getpixel((2, 10))[3], 128)
        self.assertEqual(result[1].getpixel((38, 10))[3], 128)
        self.load_pair.assert_called_once_with(
            str(self.fixture.files["SAM3"]),
            output_vae=False,
            output_clip=True,
            output_clipvision=False,
        )
        self.fixture.load.assert_not_called()
        union = self.request(segment_text="cat:2", segment_refine_iterations=0)
        self.assertIsInstance(
            vision.process(owner, union, self.fixture.manager, Mock()), Image.Image
        )
        self.load_pair.assert_called_once()
        vision.clear_model(owner)
        self.assertEqual(
            self.fixture.comfy.model_management.unload_model_and_clones.call_args_list,
            [unittest.mock.call(self.fixture.model), unittest.mock.call(self.clip.patcher)],
        )

    def test_no_detections_is_an_empty_typed_batch_and_cancel_is_not_a_result(self):
        self.paired_header()
        self.runtime(scores=(-10, -11))
        data = self.request(
            segment_text="cat:2", segment_individual_masks=True, segment_refine_iterations=0
        )
        result = vision.process(NS(), data, self.fixture.manager, Mock())
        self.assertIsInstance(result, ImageOutputBatch)
        self.assertEqual(len(result), 0)
        self.assertEqual(result.metadata, ())
        with self.assertRaises(CancelledError):
            vision.process(
                NS(), data, self.fixture.manager, Mock(side_effect=CancelledError("stop"))
            )

    def test_missing_paired_encoder_releases_the_new_model_and_never_caches_partial_state(self):
        self.paired_header()
        self.runtime()
        self.load_pair.return_value = (self.fixture.model, None, None, None)
        data = self.request(segment_text="cat:2")
        owner = NS()
        with self.assertRaisesRegex(ValueError, "paired language encoder"):
            vision.process(owner, data, self.fixture.manager, Mock())
        self.assertIsNone(getattr(owner, "_native_vision", None))
        self.fixture.comfy.model_management.unload_model_and_clones.assert_called_once_with(
            self.fixture.model
        )
        self.assertFalse(self.fixture.held)

    def test_native_mask_shape_count_and_nonfinite_results_are_rejected(self):
        self.runtime()
        data = self.request(
            segment_boxes='[{"x":0,"y":0,"width":20,"height":20}]', segment_individual_masks=True
        )
        for masks in (
            torch.zeros(65, 20, 40),
            torch.zeros(1, 19, 40),
            torch.full((1, 20, 40), float("nan")),
        ):
            with patch.object(
                sys.modules["comfy_extras.nodes_sam3"].SAM3_Detect,
                "execute",
                return_value=(masks, [[]]),
            ):
                with self.assertRaisesRegex(ValueError, "invalid or oversized"):
                    vision.process(NS(), dict(data), self.fixture.manager, Mock())

    def worker(self):
        """Actual serial PNG writer with a tiny pinned-native SAM model stand-in."""
        from ruinedfooocus.core.paths import resource_path
        from ruinedfooocus.features.image_processing.native_vision_pipeline import pipeline

        self.paired_header()
        self.runtime()
        pipe = pipeline()

        def process(gen_data, callback=None):
            result = vision.process(pipe, gen_data, self.fixture.manager, Mock())
            return result if isinstance(result, ImageOutputBatch) else [result]

        pipe.process = process
        forbidden = Mock(
            side_effect=AssertionError("generation checkpoint/prompt/sampling accessed")
        )
        pipe.load_base_model = pipe.load_loras = forbidden
        shared = NS(
            models=NS(get_models_by_path=forbidden, get_model_base=forbidden),
            state={},
            performance_settings=NS(apply=forbidden),
            resolution_settings=object(),
            path_manager=NS(model_paths={"temp_outputs_path": self.fixture.root}),
            shared_cache={},
        )
        controls = NS(get_settings=lambda data: data.get("controlnet", {}))
        route = dict(
            state={"pipeline": None},
            shared=shared,
            controlnet=controls,
            native_vision_pipeline=NS(pipeline=lambda: pipe),
        )
        function("ruinedfooocus/features/generation/pipelines.py", "selected_pipeline", route)
        update = function("ruinedfooocus/features/generation/pipelines.py", "update", route)
        filenames = count()
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
                TimeIt=lambda _: nullcontext(),
                resource_path=resource_path,
                random=random,
                Path=Path,
                pathlib=pathlib,
                Image=Image,
                PngInfo=PngInfo,
                json=json,
                generate_temp_filename=lambda **kwargs: str(
                    self.fixture.root / f"instance-{next(filenames)}.{kwargs['extension']}"
                ),
            ),
        )
        return render, forbidden

    def test_actual_worker_exports_each_native_instance_once_and_proves_empty_completion(self):
        render, forbidden = self.worker()
        data = self.request(
            segment_text="cat:2", segment_individual_masks=True, segment_refine_iterations=0
        )
        data.update(task_id=1, task_type="tool_call", silent=True, index=(0, 1))
        result = render(data)
        self.assertIsInstance(result, ImageBatchArtifacts)
        self.assertEqual((len(result), result.completed_generations), (2, 1))
        self.assertIs(validate_artifacts(data, result), result)
        for index, path in enumerate(result):
            with Image.open(path) as saved:
                metadata = json.loads(saved.info["parameters"])
                self.assertEqual(saved.size, self.image.size)
            self.assertEqual(metadata["native_output_group"]["index"], index)
            self.assertEqual(metadata["native_output_group"]["count"], 2)
            self.assertEqual(
                metadata["native_image_output"]["native_vision_source"]["instance_index"], index
            )
            self.assertEqual(metadata["native_vision_model"], self.fixture.files["SAM3"].name)
            self.assertNotIn(str(self.fixture.root), json.dumps(metadata))
        data = self.request(
            segment_text="cat:2", segment_individual_masks=True, segment_refine_iterations=0
        )
        data.update(task_id=2, task_type="tool_call", silent=True, index=(0, 1), vision_threshold=1)
        result = render(data)
        self.assertIsInstance(result, ImageBatchArtifacts)
        self.assertEqual((len(result), result.completed_generations), (0, 1))
        self.assertIs(validate_artifacts(data, result), result)
        forbidden.assert_not_called()

    def test_public_api_and_request_capture_keep_literal_sam_fields_and_zero_output_proof(self):
        from fastapi.testclient import TestClient
        from ruinedfooocus.integrations.openai_api import create_apps, ImageRequest
        from ruinedfooocus.app import api_runtime as api
        from test_api_live_contracts import job
        from ruinedfooocus.features.gallery.image_metadata import render_values

        self.paired_header()
        encoded = BytesIO()
        self.image.save(encoded, format="PNG")
        options = dict(
            segment_text="#!cat:2",
            segment_boxes='[{"x":0,"y":0,"width":20,"height":20}]',
            segment_individual_masks=True,
            segment_max_instances=2,
        )
        payload = dict(
            model="not a generation model",
            input_image=base64.b64encode(encoded.getvalue()).decode(),
            controlnet={"type": "segment"},
            native_vision_model=str(self.fixture.files["SAM3"]),
            response_format="b64_json",
            **options,
        )
        backend = NS(
            images=Mock(return_value=job(("result", []))),
            resolve_model=Mock(side_effect=AssertionError("generation model lookup")),
        )
        application, _ = create_apps(backend)
        with TestClient(application) as client:
            response = client.post("/images/generations", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["data"], [])
        backend.resolve_model.assert_not_called()
        for key, value in options.items():
            self.assertEqual(backend.images.call_args.args[0][key], value)
            self.assertEqual(render_values(options)[key], value)
        ImageRequest(**payload)
        shared = NS(
            path_manager=self.fixture.manager,
            models=NS(offline=True),
            settings=NS(default_settings={}),
            state={},
        )
        worker = NS(
            prepare_image_data=Mock(side_effect=lambda data, **kwargs: dict(data)), _process=Mock()
        )
        with (
            patch.dict(
                sys.modules,
                {"ruinedfooocus.app.state": shared, "ruinedfooocus.app.async_worker": worker},
            ),
            patch.object(sys.modules["ruinedfooocus.app"], "state", shared, create=True),
            patch.object(sys.modules["ruinedfooocus.app"], "async_worker", worker, create=True),
            patch.object(api, "submit", side_effect=lambda run: run),
        ):
            run = api.images(payload, {})
            run.prepare(object())
            captured = worker.prepare_image_data.call_args.args[0]
            self.assertEqual(captured["image_number"], 1)
            self.assertEqual(captured["segment_text"], options["segment_text"])
            worker._process.return_value = ImageBatchArtifacts([], captured[PLAN_KEY], 1)
            self.assertEqual(run(object(), 7), [])
            worker._process.return_value = []
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                run(object(), 8)


if __name__ == "__main__":
    unittest.main()
