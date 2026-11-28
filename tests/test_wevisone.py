"""Real pinned native WeVisOne node bodies with tiny inert CLIP fixtures."""

import ast
from concurrent.futures import CancelledError
from contextlib import nullcontext
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.transforms import ToPILImage, ToTensor

from ruinedfooocus.core.preparation import PreparationBusy, preparation_service
from ruinedfooocus.features.generation.image_preparation import KEY as ASSETS, ImageAssets
from ruinedfooocus.features.image_processing import wevisone as feature
from ruinedfooocus.features.image_processing import still_image_utilities as utilities
from ruinedfooocus.features.models.model_identity import file_signature

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))


def native_definitions(path, names, scope, *, methods=()):
    selected = []
    for original in ast.parse((NATIVE / path).read_text(encoding="utf-8")).body:
        node = deepcopy(original)
        label = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            label = next((n.id for n in node.targets if isinstance(n, ast.Name)), None)
        if label not in names:
            continue
        if isinstance(node, ast.ClassDef):
            node.bases, node.decorator_list = [], []
            node.body = [
                n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods
            ]
        for item in ast.walk(node):
            if isinstance(item, ast.FunctionDef):
                item.returns = None
                for arg in (*item.args.posonlyargs, *item.args.args, *item.args.kwonlyargs):
                    arg.annotation = None
        selected.append(node)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
            str(NATIVE / path),
            "exec",
        ),
        scope,
    )


def native_modules():
    management = NS(
        intermediate_device=lambda: torch.device("cpu"),
        cuda_device_context=lambda _: nullcontext(),
        unload_model_and_clones=Mock(),
    )
    comfy = NS(
        model_management=management,
        ops=NS(use_quantized_matmul=lambda *_: nullcontext()),
        utils=NS(),
    )
    scope = dict(
        torch=torch,
        F=torch.nn.functional,
        json=json,
        re=re,
        math=math,
        io=NS(NodeOutput=lambda *values: values),
        IO=NS(NodeOutput=lambda *values: values),
        comfy=comfy,
        ToPILImage=ToPILImage,
        ToTensor=ToTensor,
        ImageDraw=ImageDraw,
        ImageFont=ImageFont,
    )
    native_definitions("comfy_extras/nodes_rtdetr.py", {"bboxes_per_frame"}, scope)
    native_definitions("comfy/ldm/rt_detr/rtdetr_v4.py", {"COCO_CLASSES"}, scope)
    native_definitions(
        "comfy_extras/nodes_wevisone.py",
        {"PROMPTS", "ITEM_RE", "SHAPE_RE", "build_prompt", "parse_shapes", "WeVisOneDetect"},
        scope,
        methods=("execute",),
    )
    native_definitions(
        "comfy_extras/nodes_rtdetr.py",
        {"DrawBBoxes"},
        scope,
        methods=("execute", "draw_detections"),
    )
    native_definitions("comfy/utils.py", {"repeat_to_batch_size"}, scope)
    comfy.utils.repeat_to_batch_size = scope["repeat_to_batch_size"]
    native_definitions("node_helpers.py", {"image_alpha_fix"}, scope)
    scope["node_helpers"] = NS(image_alpha_fix=scope["image_alpha_fix"])
    native_definitions(
        "comfy_extras/nodes_mask.py",
        {"composite", "ImageCompositeMasked"},
        scope,
        methods=("execute",),
    )
    modules = {
        "comfy": comfy,
        "comfy.model_management": management,
        "comfy_extras.nodes_wevisone": NS(
            WeVisOneDetect=scope["WeVisOneDetect"], parse_shapes=scope["parse_shapes"]
        ),
        "comfy_extras.nodes_rtdetr": NS(DrawBBoxes=scope["DrawBBoxes"]),
        "comfy_extras.nodes_mask": NS(ImageCompositeMasked=scope["ImageCompositeMasked"]),
    }
    return modules, scope


class WeVisOneTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = self.root / "wevisone.safetensors"
        self.model.write_bytes(b"owned model identity fixture, no weights")
        self.manager = NS(get_model_folders=lambda name: [self.root] if name == "clip" else [])
        self.modules, self.native = native_modules()
        self.enterContext(patch.dict(sys.modules, self.modules))
        self.native_signatures = tuple(
            tuple(file_signature(NATIVE / name)) for name in feature.NATIVE_FILES
        )
        self.enterContext(
            patch.object(feature, "native_source_signatures", return_value=self.native_signatures)
        )
        self.validator = self.enterContext(
            patch.object(feature, "validate_model", return_value="WEVISONE_2B")
        )
        self.answer = "<ref>person</ref><box><0><0><1000><1000><OBJ></box>"
        self.logits = torch.tensor([[[0.0, 2.0], [-2.0, 0.0]]], device="cpu")
        self.decode_masks = Mock(side_effect=lambda *args: self.logits)
        self.generate = Mock(return_value=[1, 2])
        self.clip = NS(
            tokenize=Mock(side_effect=lambda text, **kwargs: (text, kwargs)),
            generate=self.generate,
            decode=Mock(side_effect=lambda ids: self.answer),
            patcher=NS(load_device=torch.device("cpu")),
            cond_stage_model=NS(decode_masks=self.decode_masks),
        )
        self.loader = self.enterContext(
            patch.object(feature, "_load_clip", return_value=(self.clip, (self.clip.patcher,)))
        )
        self.owner = NS()
        self.image = Image.new("RGBA", (6, 4), (80, 120, 160, 128))

    def capture(self, kind="wevisone-matte", **values):
        data, assets = feature.capture_assets(
            dict(
                input_image=self.image,
                controlnet={"type": kind},
                wevisone_model=str(self.model),
                **values,
            ),
            self.manager,
        )
        data[ASSETS] = ImageAssets(assets, True)
        return feature.worker_data(data)

    def test_native_matte_freezes_pixels_query_and_multiplies_original_alpha(self):
        data = self.capture(wevisone_query="the person", wevisone_max_length=32)
        self.image.paste((1, 2, 3, 0), (0, 0, 6, 4))
        data["wevisone_query"] = "changed later"
        result = utilities.process(self.owner, data, self.manager, lambda: None)[0]
        expected = torch.nn.functional.interpolate(
            self.logits.sigmoid()[None], size=(4, 6), mode="bilinear", align_corners=False
        )[0, 0].numpy()
        np.testing.assert_array_equal(
            np.asarray(result)[..., :3], np.full((4, 6, 3), (80, 120, 160), dtype=np.uint8)
        )
        np.testing.assert_array_equal(
            np.asarray(result)[..., 3],
            (128 * (expected * 255).round().astype(np.uint16) // 255).astype(np.uint8),
        )
        self.assertEqual(
            self.clip.tokenize.call_args.args[0],
            "Locate the instances that match the following description and provide the alpha mattes: the person.",
        )
        self.assertEqual(
            self.generate.call_args.kwargs,
            dict(do_sample=True, max_length=32, top_k=1, repetition_penalty=1.05, seed=0),
        )
        self.assertEqual(data["image_utility_source"]["boxes"][0]["width"], 6)
        self.assertEqual(feature.output_metadata(data)["wevisone_query"], "the person")
        self.assertNotIn(str(self.root), json.dumps(feature.output_metadata(data)))

    def test_native_detection_accepts_arbitrary_labels_and_preserves_geometry_alpha(self):
        self.answer = "<ref>coffee table</ref><box><0><0><1000><1000></box>"
        data = self.capture("wevisone-detect", wevisone_query="coffee table\nlamp")
        result = utilities.process(self.owner, data, self.manager, lambda: None)[0]
        self.assertEqual(result.size, (6, 4))
        self.assertTrue(np.all(np.asarray(result)[..., 3] == 128))
        self.assertEqual(
            self.clip.tokenize.call_args.args[0],
            "Locate all the instances that match the following description: coffee table. lamp.",
        )
        self.decode_masks.assert_not_called()
        self.assertEqual(data["image_utility_source"]["output"], "annotated")

    def test_native_mask_overlay_and_float_cutout_are_exact_types(self):
        data = self.capture(wevisone_output="mask")
        result = utilities.process(self.owner, data, self.manager, lambda: None)[0]
        self.assertEqual((result.mode, result.size), ("L", (6, 4)))
        data = self.capture(wevisone_output="overlay")
        overlay = utilities.process(self.owner, data, self.manager, lambda: None)[0]
        self.assertEqual((overlay.mode, overlay.size), ("RGBA", (6, 4)))
        self.assertTrue(np.all(np.asarray(overlay)[..., 3] == 128))
        data = self.capture(output_format="exr", output_dtype="float32")
        result = utilities.process(self.owner, data, self.manager, lambda: None)[0]
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 6, 4))
        self.assertEqual(result.dtype, np.float32)
        self.assertGreater(result[0, 0, 3], 0)

    def test_empty_detection_and_matte_are_valid_not_fabricated_objects(self):
        self.answer = "No objects found."
        for kind in feature.KINDS:
            with self.subTest(kind=kind):
                data = self.capture(kind)
                result = utilities.process(self.owner, data, self.manager, lambda: None)[0]
                self.assertEqual(data["image_utility_source"]["boxes"], [])
                if kind == "wevisone-matte":
                    self.assertFalse(np.any(np.asarray(result)[..., 3]))

    def test_allocation_bound_runs_before_native_mask_decoder(self):
        data = self.capture()
        with (
            patch.object(feature, "MAX_MASK_ELEMENTS", 4),
            self.assertRaisesRegex(ValueError, "pixel budget"),
        ):
            utilities.process(self.owner, data, self.manager, lambda: None)
        self.decode_masks.assert_not_called()
        self.assertNotIn("image_utility_source", data)

    def test_native_multiple_object_union_and_outside_boxes_are_bounded(self):
        self.answer = (
            "<ref>cat</ref><box><0><0><1000><1000><OBJ></box>"
            "<ref>dog</ref><box><0><0><500><500><OBJ></box>"
        )
        logits = torch.cat((self.logits, -self.logits), dim=0)
        self.decode_masks.side_effect = lambda *args: logits
        data = self.capture(wevisone_output="mask")
        result = utilities.process(self.owner, data, self.manager, lambda: None)[0]
        expected = (
            torch.nn.functional.interpolate(
                logits.sigmoid()[None], size=(4, 6), mode="bilinear", align_corners=False
            )[0]
            .amax(dim=0)
            .numpy()
        )
        np.testing.assert_array_equal(np.asarray(result), (expected * 255).round().astype(np.uint8))
        self.assertEqual(len(data["image_utility_source"]["boxes"]), 2)
        self.decode_masks.reset_mock()
        self.answer = "<ref>cat</ref><box><0><0><1001><1000><OBJ></box>"
        with self.assertRaisesRegex(ValueError, "outside"):
            utilities.process(self.owner, self.capture(), self.manager, lambda: None)
        self.decode_masks.assert_not_called()

    def test_cancel_after_native_generation_publishes_nothing_and_retry_reuses_encoder(self):
        cancelled = False

        def generate(*args, **kwargs):
            nonlocal cancelled
            cancelled = True
            return [1, 2]

        def check():
            if cancelled:
                raise CancelledError()

        self.generate.side_effect = generate
        data = self.capture()
        with self.assertRaises(CancelledError):
            utilities.process(self.owner, data, self.manager, check)
        self.clip.decode.assert_not_called()
        self.assertNotIn("image_utility_source", data)
        with preparation_service.artifact_lease(self.model, write=True, blocking=False):
            pass
        self.generate.side_effect = None
        utilities.process(self.owner, data, self.manager, lambda: None)
        self.loader.assert_called_once()

    def test_new_native_source_identity_invalidates_cached_encoder(self):
        utilities.process(self.owner, self.capture(), self.manager, lambda: None)
        changed = (self.native_signatures[0][:-1] + (0,), *self.native_signatures[1:])
        with patch.object(feature, "native_source_signatures", return_value=changed):
            utilities.process(self.owner, self.capture(), self.manager, lambda: None)
        self.assertEqual(self.loader.call_count, 2)
        self.modules["comfy.model_management"].unload_model_and_clones.assert_called_once_with(
            self.clip.patcher
        )

    def test_cancel_stale_source_and_cache_retry_contracts(self):
        data = self.capture()
        with self.assertRaises(CancelledError):
            utilities.process(self.owner, data, self.manager, Mock(side_effect=CancelledError()))
        self.loader.assert_not_called()
        utilities.process(self.owner, data, self.manager, lambda: None)
        utilities.process(self.owner, data, self.manager, lambda: None)
        self.loader.assert_called_once()
        self.model.write_bytes(b"replaced file")
        with self.assertRaisesRegex(ValueError, "changed"):
            utilities.process(self.owner, data, self.manager, lambda: None)
        self.assertNotIn("image_utility_source", data)

    def test_read_lease_covers_generation_and_failed_call_releases_it(self):
        def generate(*args, **kwargs):
            with self.assertRaises(PreparationBusy):
                with preparation_service.artifact_lease(self.model, write=True, blocking=False):
                    pass
            raise RuntimeError("native decoder failed")

        self.generate.side_effect = generate
        data = self.capture()
        with self.assertRaisesRegex(RuntimeError, "decoder failed"):
            utilities.process(self.owner, data, self.manager, lambda: None)
        self.assertNotIn("image_utility_source", data)
        with preparation_service.artifact_lease(self.model, write=True, blocking=False):
            pass

    def test_replay_requires_model_and_source_reselection_and_resets_generation_tensors(self):
        data = self.capture(wevisone_output="overlay", wevisone_query="cat")
        utilities.process(self.owner, data, self.manager, lambda: None)
        values = utilities.replay_values(feature.output_metadata(data))
        self.assertEqual(values["cn_selection"], "WeVisOne matting")
        self.assertEqual(values["wevisone_query"], "cat")
        self.assertIsNone(values["wevisone_model"])
        self.assertIsNone(values["input_image"])
        self.assertIsNone(values["native_latent_input"])
        self.assertFalse(values["native_export_conditioning"])


if __name__ == "__main__":
    unittest.main()
