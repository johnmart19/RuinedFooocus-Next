"""Frozen still utilities, actual pinned native bodies and shared API/worker callers."""

import ast
import base64
from dataclasses import replace
from concurrent.futures import CancelledError
from contextlib import nullcontext
import hashlib
import importlib.util
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

import numpy as np
from PIL import Image, ImageColor, ImageDraw
from PIL.PngImagePlugin import PngInfo
import torch
import torch.nn.functional as F
import ruinedfooocus.app

from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.image_processing import depth_utilities as utilities
from ruinedfooocus.features.image_processing import (
    utility_lotus,
    utility_moge,
    utility_face,
    utility_pose,
    utility_marigold,
)
from test_image_preparation import Manager
from test_native_model_coverage import native_method
from test_native_variant_defaults import NATIVE, write_header
from test_utility_sampling_capture import function

SD_VAE = {
    "decoder.conv_in.weight": [512, 4, 3, 3],
    "encoder.conv_in.weight": [128, 3, 3, 3],
    "decoder.conv_out.weight": [3, 128, 3, 3],
    "encoder.down.2.downsample.conv.weight": [512, 512, 3, 3],
    "decoder.up.3.upsample.conv.weight": [512, 512, 3, 3],
}
FACE = {
    "detector_short.stem.weight": [24, 3, 5, 5],
    "detector_full.stem.weight": [32, 3, 3, 3],
    "mesh.stem.weight": [16, 3, 3, 3],
    "canonical_vertices": [468, 3],
    "procrustes_indices": [33],
    "procrustes_weights": [33],
    "blendshapes.output.weight": [52, 2],
    **{
        "topology." + name: [2, 2]
        for name in (
            "face_oval",
            "left_eye",
            "right_eye",
            "left_eyebrow",
            "right_eyebrow",
            "lips",
            "irises",
            "nose",
        )
    },
}
MOGE = {
    "encoder.backbone.embeddings.cls_token": [1, 1, 1024],
    "encoder.output_projections.0.weight": [1024, 1024, 1, 1],
    "neck.input_blocks.0.weight": [256, 1026, 1, 1],
    "points_head.output_blocks.4.weight": [3, 32, 1, 1],
    "mask_head.output_blocks.4.weight": [1, 32, 1, 1],
    "scale_head.0.weight": [1, 1024],
}


class Progress:
    def __init__(self, iterable=None, **kwargs):
        self.iterable, self.current = iterable, 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.iterable)

    def update(self, count):
        self.current += count

    def update_absolute(self, count):
        self.current = count


class NodeOutput(tuple):
    def __new__(cls, *values):
        return super().__new__(cls, values)

    @property
    def result(self):
        return self


def native_functions(filename, names, scope):
    tree = ast.parse((NATIVE / filename).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), scope)


class DepthUtilities(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.manager = Manager(self.root)
        for category in (
            "checkpoints",
            "diffusion_models",
            "geometry_estimation",
            "detection",
            "vae",
            "loras",
            "embeddings",
        ):
            (self.root / category).mkdir()
        self.lotus = write_header(self.root / "diffusion_models/lotus.safetensors", {"lotus": [1]})
        self.vae = write_header(self.root / "vae/sd.safetensors", SD_VAE)
        self.moge = write_header(self.root / "geometry_estimation/moge.safetensors", MOGE)
        self.face = write_header(self.root / "detection/face.safetensors", FACE)
        self.pose = write_header(
            self.root / "checkpoints/pose.safetensors",
            {
                "model.diffusion_model.heatmap_head.conv_layers.0.weight": [640, 640, 3, 3],
                **{"first_stage_model." + key: value for key, value in SD_VAE.items()},
            },
        )
        self.albedo = {
            "image_utility_model": write_header(
                self.root / "diffusion_models/qwen.safetensors", {"qwen": [1]}
            ),
            "image_utility_vae": write_header(
                self.root / "vae/albedo.safetensors",
                {"encoder.conv.weight": [1], "decoder.conv.weight": [1]},
            ),
            "image_utility_lora": write_header(
                self.root / "loras/albedo.safetensors",
                {"layer.lora_down.weight": [2, 32], "layer.lora_up.weight": [32, 2]},
            ),
            "image_utility_conditioning": write_header(
                self.root / "embeddings/albedo.safetensors", {"conditioning": [1, 2, 3584]}
            ),
        }
        self.enterContext(patch.object(utility_lotus.lotus_depth, "validate_model"))
        self.enterContext(
            patch(
                "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                return_value={"native_class": "QwenImage"},
            )
        )
        receipt = json.loads(json.dumps(utility_marigold.recipe()))
        for field, path in self.albedo.items():
            receipt["components"][field].update(
                bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
        self.enterContext(patch.object(utility_marigold, "recipe", return_value=receipt))
        self.source = Image.new("RGBA", (39, 21), (25, 80, 160, 128))
        self.paths = {
            "lotus-depth": self.lotus,
            "moge-depth": self.moge,
            "face-landmarks": self.face,
            "pose-map": self.pose,
            "marigold-albedo": self.albedo["image_utility_model"],
        }

    def data(self, kind, **values):
        fields = {
            "input_image": self.source,
            "image_utility_model": str(self.paths[kind]),
            "controlnet": {"type": kind},
            "prompt": "#!never-execute",
            "image_number": 1,
        }
        if kind == "lotus-depth":
            fields["image_utility_vae"] = str(self.vae)
        if kind == "marigold-albedo":
            fields.update({field: str(path) for field, path in self.albedo.items()})
        fields.update(values)
        return fields

    def capture(self, kind, **values):
        data = self.data(kind, **values)
        return assets.capture_image_assets(data, self.manager, data["controlnet"], offline=True)

    def test_capture_all_operations_freezes_source_roles_and_no_sampling_components(self):
        expected = {
            "lotus-depth": ("lotus-depth", "lotus-vae"),
            "moge-depth": ("moge-model",),
            "face-landmarks": ("face-landmarker",),
            "pose-map": ("sdpose-checkpoint",),
            "marigold-albedo": (
                "marigold-qwen-edit",
                "marigold-albedo-vae",
                "marigold-albedo-lora",
                "marigold-albedo-conditioning",
            ),
        }
        for kind in utilities.KINDS:
            with self.subTest(kind=kind):
                data = self.capture(
                    kind,
                    base_model_name="missing-generation-checkpoint",
                    model_recipe_path="stale.yaml",
                    reference_adapter="redux",
                )
                self.assertEqual(
                    tuple(item.role for item in data[assets.KEY].artifacts), expected[kind]
                )
                self.assertEqual(data["input_image"].size, (39, 21))
                self.assertEqual(data[utilities.KEY].rgba[:4], bytes((25, 80, 160, 128)))
                self.assertEqual(
                    (data["prompt"], data["custom_steps"], data["_sampling_captured"]),
                    ("", 1, False),
                )
                self.assertNotIn("model_recipe_path", data)
                self.assertEqual(data["loras"], [])
        data = self.capture("moge-depth")
        self.source.putpixel((0, 0), (0, 0, 0, 0))
        data["input_image"].putpixel((0, 0), (0, 0, 0, 0))
        self.assertEqual(data[utilities.KEY].rgba[:4], bytes((25, 80, 160, 128)))

    def test_standalone_capture_bypasses_layered_patch_control_and_generation_default_lookups(self):
        from ruinedfooocus.app import api_runtime as api

        forbidden = Mock(
            side_effect=AssertionError("generation route consulted by standalone utility")
        )
        data = self.data(
            "moge-depth",
            base_model_name="absent-layered.safetensors",
            qwen_layer_count=7,
            model_patch_recipe="uso-style",
            control_recipe="alimama-inpaint",
        )
        with (
            patch("ruinedfooocus.features.generation.layered.capture", forbidden),
            patch("ruinedfooocus.features.image_editing.model_patches.capture_assets", forbidden),
            patch("ruinedfooocus.features.image_editing.native_control_recipes.capture", forbidden),
        ):
            result = assets.capture_image_assets(
                data,
                self.manager,
                data["controlnet"],
                offline=True,
                models=NS(get_models_by_path=forbidden),
            )
        for key in ("qwen_layer_count", "model_patch_recipe", "control_recipe"):
            self.assertNotIn(key, result)

        class ForbiddenDefaults(dict):
            def __deepcopy__(self, memo):
                raise AssertionError("standalone API consulted generation defaults")

        raw = BytesIO()
        self.source.save(raw, format="PNG")
        shared = NS(
            path_manager=self.manager,
            models=NS(offline=True),
            settings=NS(default_settings={}),
            state={},
        )
        payload = {
            "controlnet": {"type": "moge-depth"},
            "input_image": base64.b64encode(raw.getvalue()).decode(),
            "image_utility_model": str(self.moge),
        }
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(api, "image_defaults", ForbiddenDefaults()),
            patch.object(api, "submit", side_effect=lambda run: run),
        ):
            self.assertTrue(callable(api.images(payload, {})))
        forbidden.assert_not_called()

    def test_actual_ui_capture_bypasses_generation_defaults_prompt_helpers_and_optional_engines(
        self,
    ):
        forbidden = Mock(side_effect=AssertionError("generation dependency"))
        shared = NS(
            models=NS(
                offline=True,
                get_file=forbidden,
                get_models_by_path=forbidden,
                get_model_base=forbidden,
            ),
            path_manager=self.manager,
            resolution_settings=object(),
            performance_settings=NS(apply=forbidden),
        )
        controls = NS(get_settings=lambda data: data.get("controlnet", {}))
        capture = function(
            "ruinedfooocus/app/async_worker.py",
            "capture_image_data",
            dict(
                shared=shared,
                settings=NS(default_settings={"image_engine": "sdcpp-vulkan"}),
                control_tools=controls,
            ),
        )
        for kind in utilities.KINDS:
            data = self.data(
                kind,
                task_type="process",
                base_model_name="unqualified-generation.safetensors",
                image_engine="sdcpp-vulkan",
                performance_selection="Model Defaults",
                image_number=0,
                style_selection=["Style: enhance"],
                model_recipe_path="stale.yaml",
                reference_adapter="redux",
            )
            result = capture(data)
            self.assertEqual(
                (result["_sampling_captured"], result["image_number"], result["generate_forever"]),
                (False, 1, False),
            )
            self.assertIsInstance(result[utilities.KEY], utilities.UtilityRequest)
            self.assertNotIn("_sdcpp_snapshot", result)
            self.assertNotIn("_prompt_preparation", result)
        forbidden.assert_not_called()

    def test_face_role_has_a_configured_native_root_without_creating_or_loading_models(self):
        from ruinedfooocus.core.path import PathManager
        from ruinedfooocus.core.model_folders import model_folders
        from ruinedfooocus.backends.comfy.model_paths import register_image_model_paths

        calls = []
        self.assertEqual(PathManager.DEFAULT_PATHS["path_detection"], "../models/detection/")
        manager = NS(
            paths=dict(
                PathManager.DEFAULT_PATHS, path_detection=str(self.root / "never-created/detection")
            ),
            get_abspath=lambda path: (self.root / path).resolve(),
        )
        expected = model_folders(manager.paths, "detection", manager.get_abspath)
        self.assertEqual(len(expected), 1)
        with patch.dict(
            sys.modules,
            {
                "folder_paths": NS(
                    add_model_folder_path=lambda kind, path, **kwargs: calls.append((kind, path))
                )
            },
        ):
            register_image_model_paths(manager)
        self.assertIn(("detection", str(expected[0])), calls)
        self.assertFalse(expected[0].exists())

    def test_capture_rejects_wrong_roles_escape_nonfinite_options_and_expired_files(self):
        outside = write_header(self.root / "outside.safetensors", MOGE)
        for kind, changes in (
            ("lotus-depth", {"image_utility_vae": ""}),
            ("moge-depth", {"image_utility_model": str(self.face)}),
            ("moge-depth", {"image_utility_model": str(outside)}),
            ("moge-depth", {"image_utility_model": "../outside.safetensors"}),
            ("moge-depth", {"moge_resolution_level": True}),
            ("moge-depth", {"moge_refine_steps": 9}),
            ("face-landmarks", {"face_min_confidence": float("nan")}),
            ("face-landmarks", {"face_max_count": 0}),
            ("face-landmarks", {"face_detector_variant": "video"}),
            ("pose-map", {"pose_score_threshold": 2}),
            ("marigold-albedo", {"image_utility_lora": str(self.lotus)}),
            ("moge-depth", {"input_image": None}),
            ("moge-depth", {"inpaint_toggle": True}),
        ):
            with (
                self.subTest(kind=kind, changes=changes),
                self.assertRaises((ValueError, FileNotFoundError)),
            ):
                self.capture(kind, **changes)
        self.moge.unlink()
        with self.assertRaises(ValueError):
            self.capture("moge-depth")

    def test_deleted_captured_local_component_never_submits_a_transfer(self):
        data = self.capture("moge-depth")
        data[assets.KEY] = replace(data[assets.KEY], offline=False)
        self.moge.unlink()
        submit = Mock(side_effect=AssertionError("local utility attempted acquisition"))
        with self.assertRaisesRegex(FileNotFoundError, "removed"):
            assets.prepare_image_assets(data, check_cancelled=Mock(), service=NS(submit=submit))
        submit.assert_not_called()

    def test_passive_picker_is_role_specific_and_does_not_hash_or_load_weights(self):
        with patch(
            "ruinedfooocus.features.models.model_identity.cached_file_sha256",
            side_effect=AssertionError("picker hashed weights"),
        ):
            for kind, field, path in (
                ("moge-depth", "image_utility_model", self.moge),
                ("face-landmarks", "image_utility_model", self.face),
                ("lotus-depth", "image_utility_vae", self.vae),
            ):
                values = utilities.component_choices(self.manager, kind, field)
                self.assertIn(str(path), {value for _, value in values})
        self.assertEqual(
            utilities.component_choices(self.manager, "depth-map", "image_utility_model"), []
        )
        self.assertEqual(utility_moge.validate_model(self.moge), "MoGe2")
        self.assertEqual(utility_pose.validate_model(self.pose), "SDPose")
        write_header(self.pose, {"heatmap_head.conv_layers.0.weight": [1]})
        with self.assertRaisesRegex(ValueError, "bundled"):
            utility_pose.validate_model(self.pose)

    def runtime(self):
        self.memory = NS(
            unload_model_and_clones=Mock(),
            intermediate_device=lambda: torch.device("cpu"),
            intermediate_dtype=lambda: torch.float32,
            text_encoder_offload_device=lambda: torch.device("cpu"),
        )
        utils = NS(
            ProgressBar=Progress,
            common_upscale=lambda image, w, h, **kwargs: F.interpolate(image, size=(h, w)),
            load_torch_file=Mock(return_value={}),
        )
        self.folders = {}
        folder_paths = NS(
            add_model_folder_path=lambda kind, path, **kwargs: self.folders.update(
                {kind: Path(path)}
            ),
            get_full_path_or_raise=lambda kind, name: str(self.folders[kind] / name),
        )
        self.comfy = NS(model_management=self.memory, utils=utils, sd=NS())
        self.scope = dict(
            torch=torch,
            F=F,
            np=np,
            comfy=self.comfy,
            io=NS(NodeOutput=NodeOutput),
            tqdm=Progress,
            folder_paths=folder_paths,
            Image=Image,
            ImageColor=ImageColor,
            ImageDraw=ImageDraw,
        )
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "comfy": self.comfy,
                    "comfy.sd": self.comfy.sd,
                    "comfy.model_management": self.memory,
                    "folder_paths": folder_paths,
                },
            )
        )
        return self.scope

    def test_actual_moge_loader_inference_render_handles_native_geometry_and_invalid_depth(self):
        scope = self.runtime()
        seen = []

        def infer(image, **options):
            seen.append((image.shape, options))
            depth = torch.arange(1, 40, dtype=torch.float32)[None, None].expand(1, 21, 39).clone()
            depth[:, :, -1] = float("inf")
            return {"depth": depth, "mask": torch.isfinite(depth)}

        model = NS(patcher=object(), infer=infer)
        scope["MoGeModel"] = lambda state: model
        native_functions("comfy_extras/nodes_moge.py", ("_normalize_disparity",), scope)
        nodes = {
            name: native_method("comfy_extras/nodes_moge.py", name, "execute", scope)
            for name in ("LoadMoGeModel", "MoGeInference", "MoGeRender")
        }
        self.enterContext(patch.dict(sys.modules, {"comfy_extras.nodes_moge": NS(**nodes)}))
        data = self.capture("moge-depth", moge_resolution_level=0, moge_refine_steps=0)
        owner = NS()
        output = utilities.process(owner, data, self.manager, Mock())[0]
        self.assertEqual((output.size, output.mode), (self.source.size, "RGBA"))
        self.assertGreater(output.getpixel((0, 0))[0], output.getpixel((20, 0))[0])
        self.assertEqual(output.getpixel((38, 0)), (0, 0, 0, 128))
        self.assertEqual(seen[0][0], (1, 3, 21, 39))
        self.assertEqual(seen[0][1]["resolution_level"], 0)
        self.assertEqual(seen[0][1]["refine_steps"], 0)
        self.comfy.utils.load_torch_file.assert_called_once()
        utilities.process(owner, data, self.manager, Mock())
        self.comfy.utils.load_torch_file.assert_called_once()
        utilities.clear_model(owner)
        self.memory.unload_model_and_clones.assert_called_once_with(model.patcher)

    def test_actual_face_loader_landmarks_and_overlay_preserve_source_alpha(self):
        scope = self.runtime()
        names = tuple(
            name.removeprefix("topology.") for name in FACE if name.startswith("topology.")
        )
        model = NS(
            canonical_data={},
            load_device="cpu",
            patchers={"short": NS(load_device="cpu"), "full": NS(load_device="cpu")},
            connection_sets={name: frozenset(((0, 1),)) for name in names},
            detect_batch=Mock(
                return_value=[
                    [
                        {
                            "landmarks_xy": np.array([[4, 4], [20, 12]], dtype=np.float32),
                            "bbox_xyxy": [2, 2, 25, 16],
                            "score": 0.9,
                        }
                    ]
                ]
            ),
        )
        scope.update(
            FaceLandmarkerModel=lambda state: model,
            transformation_matrix_from_detection=lambda *args: np.eye(4),
            _ALL_CONNECTION_PARTS=names,
        )
        native_functions(
            "comfy_extras/nodes_mediapipe.py",
            ("_image_to_uint8", "_parse_color", "_fill_missing_frames", "_draw_mesh"),
            scope,
        )
        nodes = {
            name: native_method("comfy_extras/nodes_mediapipe.py", name, "execute", scope)
            for name in (
                "LoadMediaPipeFaceLandmarker",
                "MediaPipeFaceLandmarker",
                "MediaPipeFaceMeshVisualize",
            )
        }
        self.enterContext(patch.dict(sys.modules, {"comfy_extras.nodes_mediapipe": NS(**nodes)}))
        data = self.capture("face-landmarks", face_min_confidence=0, face_max_count=2)
        owner = NS()
        result = utilities.process(owner, data, self.manager, Mock())[0]
        self.assertEqual(result.getpixel((4, 4)), (0, 255, 0, 128))
        self.assertEqual(result.getpixel((35, 18)), (25, 80, 160, 128))
        self.assertEqual(data["image_utility_source"]["face_count"], 1)
        self.assertEqual(
            model.detect_batch.call_args.kwargs,
            {"num_faces": 2, "score_thresh": 0.0, "variant": "short"},
        )
        utilities.clear_model(owner)
        self.assertEqual(self.memory.unload_model_and_clones.call_count, 2)

    def test_lotus_standalone_pad_crop_uses_existing_native_recipe_and_frozen_pixels(self):
        self.runtime()
        data = self.capture("lotus-depth")
        data["input_image"].paste((0, 0, 0, 0), (0, 0, 39, 21))

        def render(owner, pixels, options, captured, manager, **kwargs):
            self.assertEqual(tuple(pixels.shape), (1, 24, 40, 3))
            self.assertEqual(
                options, {"depth_lotus_model": self.lotus.name, "depth_lotus_vae": self.vae.name}
            )
            self.assertAlmostEqual(float(pixels[0, 0, 0, 0]), 25 / 255)
            kwargs["check_cancelled"]()
            return torch.full_like(pixels, 0.5)

        previous_patcher = object()
        owner = NS(_depth_utility_cache=(object(), object(), (previous_patcher,)))
        with patch.object(utility_lotus.lotus_depth, "prepare_lotus", side_effect=render):
            result = utilities.process(owner, data, self.manager, Mock())[0]
        self.memory.unload_model_and_clones.assert_called_once_with(previous_patcher)
        self.assertEqual((result.size, result.getpixel((38, 20))), ((39, 21), (128, 128, 128, 128)))
        self.assertEqual(data["image_utility_source"]["convention"], "white-near")

    def test_actual_sdpose_extractor_and_drawing_use_pose_type_and_bundled_vae(self):
        scope = self.runtime()
        spec = importlib.util.spec_from_file_location(
            "rf_cpu_keypoint_draw", NATIVE / "comfy_extras/pose/keypoint_draw.py"
        )
        drawing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(drawing)
        scope["KeypointDraw"] = drawing.KeypointDraw
        scope["mm"] = NS(get_torch_device=lambda: torch.device("cpu"))
        scope["LotusConditioning"] = native_method(
            "comfy_extras/nodes_lotus.py", "LotusConditioning", "execute", scope
        )
        native_functions(
            "comfy_extras/nodes_sdpose.py", ("_preprocess_keypoints", "_to_openpose_frames"), scope
        )
        head = Mock(
            return_value=(
                np.full((1, 133, 2), 4, dtype=np.float32),
                np.ones((1, 133), dtype=np.float32),
            )
        )
        head.heatmap_size = (2, 2)
        model = NS(model=NS(diffusion_model=NS(heatmap_head=head)))
        clone = NS(model_options={})
        model.clone = lambda: clone

        def sample(model, **kwargs):
            patcher = model.model_options["transformer_options"]["patches"]["output_block_patch"][0]
            patcher(torch.ones(1, 640, 2, 2), None, {})

        self.comfy.sample = NS(sample=Mock(side_effect=sample))
        vae = NS(
            patcher=object(),
            latent_channels=4,
            downscale_ratio=8,
            throw_exception_if_invalid=Mock(),
            encode=Mock(return_value=torch.zeros(1, 4, 1, 1)),
        )
        self.comfy.sd.load_checkpoint_guess_config = Mock(return_value=(model, None, vae, None))
        nodes = {
            name: native_method("comfy_extras/nodes_sdpose.py", name, "execute", scope)
            for name in ("SDPoseKeypointExtractor", "SDPoseDrawKeypoints")
        }
        self.enterContext(patch.dict(sys.modules, {"comfy_extras.nodes_sdpose": NS(**nodes)}))
        data = self.capture("pose-map")
        owner = NS()
        result = utilities.process(owner, data, self.manager, Mock())[0]
        self.assertEqual((result.size, result.mode), ((39, 21), "RGBA"))
        self.assertTrue(np.asarray(result)[..., :3].any())
        self.assertTrue(np.all(np.asarray(result)[..., 3] == 128))
        self.assertEqual(tuple(vae.encode.call_args.args[0].shape), (1, 8, 8, 3))
        self.assertEqual(self.comfy.sample.sample.call_args.kwargs["steps"], 1)
        self.assertEqual(data["image_utility_source"]["native_output"], "POSE_KEYPOINT → IMAGE")
        self.assertEqual(data["image_utility_source"]["people"], 1)
        utilities.clear_model(owner)
        self.assertEqual(self.memory.unload_model_and_clones.call_count, 2)

    def test_marigold_native_conditioning_manual_sigmas_and_srgb_postprocess_match_recipe(self):
        import re

        scope = self.runtime()
        scope.update(json=json, re=re)
        calls = []
        vae = NS(
            patcher=object(),
            latent_channels=16,
            latent_dim=3,
            downscale_ratio=(lambda count: (count + 3) // 4, 8, 8),
            upscale_ratio=(lambda count: count * 4, 8, 8),
            output_channels=3,
            throw_exception_if_invalid=Mock(),
        )
        for method in ("spacial_compression_encode", "spacial_compression_decode"):
            native_vae = native_method("comfy/sd.py", "VAE", method, scope)
            setattr(vae, method, getattr(native_vae, method).__get__(vae))
        self.comfy.sd.load_diffusion_model = Mock(
            return_value=NS(name="base", model=NS(model_config=type("QwenImage", (), {})()))
        )
        self.comfy.utils.load_torch_file = Mock(
            return_value=(
                {"conditioning": torch.ones(1, 2, 3584), "attention_mask": torch.ones(1, 2)},
                {},
            )
        )
        nodes = NS(
            VAELoader=lambda: NS(load_vae=lambda name: (vae,)),
            LoraLoaderModelOnly=lambda: NS(
                load_lora_model_only=lambda model, name, strength: (
                    calls.append(("lora", name, strength)) or model,
                )
            ),
            VAEEncode=lambda: NS(
                encode=lambda codec, image: (
                    calls.append(("encode", tuple(image.shape))) or {"samples": image},
                )
            ),
            VAEDecode=lambda: NS(decode=lambda codec, latent: (latent["samples"] * 0 + 0.5,)),
        )
        condition = native_method(
            "comfy_extras/nodes_cond.py", "ConditioningLoader", "execute", scope
        )
        manual = native_method(
            "comfy_extras/nodes_custom_sampler.py", "ManualSigmas", "execute", scope
        )
        post = native_method(
            "comfy_extras/nodes_marigold.py", "MarigoldV2PostProcess", "execute", scope
        )

        def guider(model, conditioning):
            self.assertEqual(tuple(conditioning[0][0].shape), (1, 2, 3584))
            self.assertIn("attention_mask", conditioning[0][1])
            return (object(),)

        def sample(noise, guide, sampler, sigmas, latent):
            self.assertEqual((noise, sampler, sigmas.tolist()), ("no-noise", "euler", [0.5, 0.0]))
            return (latent,)

        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "nodes": nodes,
                    "comfy_extras.nodes_cond": NS(ConditioningLoader=condition),
                    "comfy_extras.nodes_model_advanced": NS(
                        ModelSamplingAuraFlow=lambda: NS(
                            patch_aura=lambda model, **options: (
                                calls.append(("patch", options)) or model,
                            )
                        )
                    ),
                    "comfy_extras.nodes_custom_sampler": NS(
                        DisableNoise=NS(execute=lambda: ("no-noise",)),
                        BasicGuider=NS(execute=guider),
                        KSamplerSelect=NS(execute=lambda name: (name,)),
                        ManualSigmas=manual,
                        SamplerCustomAdvanced=NS(execute=sample),
                    ),
                    "comfy_extras.nodes_marigold": NS(MarigoldV2PostProcess=post),
                },
            )
        )
        data = self.capture(
            "marigold-albedo",
            output_format="exr",
            output_bit_depth="32-bit float",
            output_color_space="sRGB",
        )
        result = utilities.process(NS(), data, self.manager, Mock())[0]
        self.assertEqual(result.shape, (21, 39, 4))
        self.assertAlmostEqual(float(result[0, 0, 0]), 1.055 * 0.5 ** (1 / 2.4) - 0.055, places=6)
        self.assertAlmostEqual(float(result[0, 0, 3]), 128 / 255, places=6)
        self.assertIn(("encode", (1, 32, 48, 3)), calls)
        self.assertIn(("patch", {"shift": 1.73, "sampling": "img_to_img_velocity"}), calls)
        self.assertEqual(data["image_utility_source"]["result"], "srgb-albedo")

    def test_replacement_cancel_and_nonfinite_or_resized_outputs_cannot_publish(self):
        data = self.capture("moge-depth")
        with patch.object(utility_moge, "render", side_effect=AssertionError("must not run")):
            with self.assertRaises(CancelledError):
                utilities.process(NS(), data, self.manager, Mock(side_effect=CancelledError()))
        with self.moge.open("ab") as target:
            target.write(b"changed")
        with self.assertRaisesRegex(ValueError, "changed after submission|changed"):
            utilities.process(NS(), data, self.manager, Mock())
        data = self.capture("moge-depth")
        for value in (torch.zeros(1, 20, 39, 3), torch.full((1, 21, 39, 3), float("nan"))):
            with (
                patch.object(utility_moge, "render", return_value=(value, {})),
                self.assertRaisesRegex(ValueError, "invalid image"),
            ):
                utilities.process(NS(), data, self.manager, Mock())
        self.assertNotIn("image_utility_source", data)

    def test_cached_loader_cancellation_unloads_new_native_patchers(self):
        self.runtime()
        request = self.capture("moge-depth")[utilities.KEY]
        patcher = object()
        cancel = Mock(side_effect=[None, CancelledError()])
        with self.assertRaises(CancelledError):
            utilities.cached_model(NS(), request, lambda: (object(), (patcher,)), cancel)
        self.memory.unload_model_and_clones.assert_called_once_with(patcher)

    def test_actual_library_apply_and_reuse_restore_intent_without_resolving_generation(self):
        import gradio as gr
        from ruinedfooocus.features.gallery.image_metadata import read_metadata

        forbidden = Mock(side_effect=AssertionError("generation setup resolved"))
        shared = NS(
            models=NS(get_file=forbidden, get_models_by_path=forbidden),
            settings=NS(default_settings={}),
        )
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
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
        ):
            for kind in utilities.KINDS:
                metadata = {
                    "prompt": "",
                    "controlnet": {"type": kind},
                    "width": 39,
                    "height": 21,
                    "image_utility_model": "display-only.safetensors",
                }
                for updates in (apply(metadata, None, None), reuse(json.dumps(metadata))):
                    self.assertIsNone(updates["input_image"]["value"])
                    self.assertIsNone(updates["image_utility_model"]["value"])
                    self.assertNotIn("base_model_name", updates)
                    self.assertIn(kind, utilities.KINDS)
        forbidden.assert_not_called()

    def test_actual_create_submit_skips_prompt_helpers_for_named_utilities(self):
        forbidden = Mock(side_effect=AssertionError("prompt helper called"))
        worker = NS(add_task=Mock(return_value=9))
        scope = dict(
            worker=worker,
            ui_controlnet=NS(tool_kind=lambda choice, kind: kind),
            use_random_prompt=forbidden,
            get_promptlist=forbidden,
        )
        submit = function("ruinedfooocus/ui/app.py", "append_work", scope)
        for kind in utilities.KINDS:
            self.assertEqual(
                submit(
                    {
                        "cn_selection": "tool",
                        "cn_type": kind,
                        "prompt": "#!ignore",
                        "image_number": 0,
                    }
                ),
                9,
            )
            data = worker.add_task.call_args.args[0]
            self.assertEqual((data["prompt"], data["image_total"]), ("", 1))
        forbidden.assert_not_called()

    def test_marigold_task_companions_are_source_verified_but_native_unet_quantization_is_independent(
        self,
    ):
        for field, path in self.albedo.items():
            utility_marigold.verify_component(field, path)
        alternate = write_header(
            self.root / "diffusion_models/alternate.safetensors", {"other-encoding": [1]}
        )
        with self.assertRaisesRegex(ValueError, "Edit2509 backbone provenance"):
            utility_marigold.verify_component("image_utility_model", alternate)
        checksum = hashlib.sha256(alternate.read_bytes()).hexdigest()
        row = {
            "id": "fixture:reviewed-alternate-edit-2509",
            "state": "existing",
            "catalogue_key": "diffusion_models/alternate.safetensors",
            "role": "image_diffusion_checkpoint",
            "native_class": "QwenImage",
            "native_family": "QwenImage",
            "behavioral_variant": "edit-2509",
            "artifact": {
                "bytes": alternate.stat().st_size,
                "sha256": checksum,
                "pinned_url": "https://fixture.invalid/resolve/"
                + "a" * 40
                + "/alternate.safetensors",
            },
        }
        registry = NS(
            entries=lambda **kwargs: [row],
            artifact=lambda identity: row if identity == row["id"] else None,
        )
        with patch(
            "ruinedfooocus.features.models.model_registry.load_registry", return_value=registry
        ):
            proof = utility_marigold.verify_component("image_utility_model", alternate)
            self.assertEqual(proof["sha256"], checksum)
            self.assertEqual(proof["behavioral_variant"], "edit-2509")
            data = self.capture("marigold-albedo", image_utility_model=str(alternate))
            metadata = {
                "_rf_file_signature": list(utilities.file_signature(alternate)),
                "_rf_local_sha256": checksum,
                "_rf_verified_native_source": {
                    "artifact": row["id"],
                    "family": "QwenImage",
                    "sha256": checksum,
                },
            }
            utility_marigold._backbone_proofs.clear()
            with (
                patch(
                    "ruinedfooocus.features.models.model_identity.read_prepared_native_source",
                    return_value=metadata,
                ),
                patch(
                    "ruinedfooocus.features.models.model_identity.cached_file_sha256",
                    side_effect=AssertionError("verified source must not be rehashed"),
                ),
            ):
                self.assertEqual(utility_marigold.backbone_proof(alternate)["sha256"], checksum)
                metadata["_rf_file_signature"] = ["stale", 0, 0, 0]
                utility_marigold._backbone_proofs.clear()
                self.assertIsNone(utility_marigold.backbone_proof(alternate))
            utility_marigold._backbone_proofs.clear()
            utility_marigold.verify_captured_request(data[utilities.KEY])
            renamed = alternate.with_name("renamed-model.safetensors")
            renamed.write_bytes(alternate.read_bytes())
            self.assertEqual(
                utility_marigold.verify_component("image_utility_model", renamed)["sha256"],
                checksum,
            )
            row["behavioral_variant"] = "base"
            with self.assertRaisesRegex(ValueError, "source proof changed"):
                utility_marigold.verify_captured_request(data[utilities.KEY])
            with self.assertRaisesRegex(ValueError, "backbone provenance"):
                utility_marigold.verify_component("image_utility_model", alternate)
        for field in ("image_utility_vae", "image_utility_lora", "image_utility_conditioning"):
            with self.assertRaisesRegex(ValueError, "source-verified"):
                utility_marigold.verify_component(field, alternate)

    def test_marigold_picker_retains_unverified_architecture_without_hashing_or_default_migration(
        self,
    ):
        unverified = write_header(
            self.root / "diffusion_models/unknown-qwen.safetensors", {"unknown-base": [1]}
        )
        with patch(
            "ruinedfooocus.features.models.model_identity.cached_file_sha256",
            side_effect=AssertionError("passive weight hash"),
        ):
            choices = utilities.component_choices(
                self.manager, "marigold-albedo", "image_utility_model"
            )
        self.assertTrue(
            any(value == str(unverified) and "unverified" in label for label, value in choices)
        )
        with self.assertRaisesRegex(ValueError, "backbone provenance"):
            self.capture("marigold-albedo", image_utility_model=str(unverified))

    def test_marigold_frozen_source_changes_fail_before_native_load_or_cache_use(self):
        self.runtime()
        data = self.capture("marigold-albedo")
        facts = utility_marigold.recipe()
        original = facts["source_revision"]
        try:
            facts["source_revision"] = "f" * 40
            with (
                patch.object(
                    utility_marigold,
                    "render",
                    side_effect=AssertionError("native loader/cache must not run"),
                ),
                self.assertRaisesRegex(ValueError, "source proof changed"),
            ):
                utilities.process(
                    NS(_depth_utility_cache=(None, None, ())), data, self.manager, Mock()
                )
        finally:
            facts["source_revision"] = original
        item = next(
            item for item in data[utilities.KEY].files if item.field == "image_utility_model"
        )
        item.path.write_bytes(item.path.read_bytes() + b"changed")
        with (
            patch.object(
                utility_marigold,
                "render",
                side_effect=AssertionError("native loader/cache must not run"),
            ),
            self.assertRaisesRegex(ValueError, "changed"),
        ):
            utilities.process(NS(), data, self.manager, Mock())

    def test_deep_json_candidate_is_normalized_and_skipped_by_real_passive_picker(self):
        from ruinedfooocus.features.models.model_identity import read_safetensors_header

        path = self.root / "geometry_estimation/deep-metadata.safetensors"
        raw = b'{"__metadata__":' + b"[" * 20000 + b"0" + b"]" * 20000 + b"}"
        path.write_bytes(len(raw).to_bytes(8, "little") + raw)
        with self.assertRaisesRegex(ValueError, "nesting limit"):
            read_safetensors_header(path)
        choices = utilities.component_choices(self.manager, "moge-depth", "image_utility_model")
        self.assertIn(str(self.moge), [value for label, value in choices])
        self.assertNotIn(str(path), [value for label, value in choices])

    def test_public_json_and_multipart_api_no_generation_checkpoint_and_strict_options(self):
        from fastapi.testclient import TestClient
        from ruinedfooocus.integrations.openai_api import create_apps
        from test_api_live_contracts import job

        raw = BytesIO()
        self.source.save(raw, format="PNG")
        output = self.root / "result.png"
        self.source.save(output)
        backend = NS(
            images=Mock(side_effect=lambda *args: job(("result", [str(output)]))),
            resolve_model=Mock(side_effect=AssertionError("generation model")),
        )
        api, _ = create_apps(backend)
        with TestClient(api) as client:
            for kind in utilities.KINDS:
                payload = {
                    key: value for key, value in self.data(kind).items() if key in utilities.FIELDS
                }
                payload.update(
                    controlnet={"type": kind},
                    model="missing-generation-checkpoint",
                    input_image=base64.b64encode(raw.getvalue()).decode(),
                    response_format="b64_json",
                )
                result = client.post("/images/generations", json=payload)
                self.assertEqual(result.status_code, 200, result.text)
                self.assertEqual(backend.images.call_args.args[1], {})
                form = {
                    key: str(value)
                    for key, value in payload.items()
                    if key not in ("input_image", "controlnet")
                }
                form["controlnet"] = json.dumps(payload["controlnet"])
                result = client.post(
                    "/images/edits",
                    data=form,
                    files={"image": ("source.png", raw.getvalue(), "image/png")},
                )
                self.assertEqual(result.status_code, 200, result.text)
                for invalid in (
                    {"n": 2},
                    {"image_utility_model": ""},
                    {"moge_refine_steps": True},
                    {"face_min_confidence": 2},
                ):
                    self.assertEqual(
                        client.post("/images/generations", json=payload | invalid).status_code, 400
                    )
        backend.resolve_model.assert_not_called()

    def test_actual_api_runtime_captures_request_before_compute_and_rejects_size_or_extra_sources(
        self,
    ):
        from ruinedfooocus.app import api_runtime as api

        raw = BytesIO()
        source = self.source.copy()
        source.getexif()[274] = 6
        source.save(raw, format="PNG", exif=source.getexif())
        encoded = base64.b64encode(raw.getvalue()).decode()
        shared = NS(
            path_manager=self.manager,
            models=NS(offline=True),
            settings=NS(default_settings={}),
            state={},
        )
        worker = NS(
            prepare_image_data=Mock(side_effect=lambda data, **kwargs: dict(data)),
            _process=Mock(return_value=[str(self.root / "output.png")]),
        )
        with (
            patch.dict(
                sys.modules,
                {"ruinedfooocus.app.state": shared, "ruinedfooocus.app.async_worker": worker},
            ),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(ruinedfooocus.app, "async_worker", worker, create=True),
            patch.object(api, "submit", side_effect=lambda run: run),
        ):
            for kind in utilities.KINDS:
                payload = {
                    key: value for key, value in self.data(kind).items() if key in utilities.FIELDS
                }
                payload.update(input_image=encoded, controlnet={"type": kind}, size="21x39")
                run = api.images(payload, {})
                run.prepare(object())
                self.assertEqual(run(None, 7), [str(self.root / "output.png")])
                data = worker._process.call_args.args[0]
                self.assertEqual(
                    (data[utilities.KEY].size, data["prompt"], data["_sampling_captured"]),
                    ((21, 39), "", False),
                )
                for invalid in (
                    {"size": "39x21"},
                    {"n": 2},
                    {"reference_images": [encoded]},
                    {"mask": encoded},
                    {"model_recipe_path": "recipe.yaml"},
                ):
                    with self.assertRaises(ValueError):
                        api.images(payload | invalid, {})

    def test_actual_worker_writes_safe_utility_png_metadata_without_generation_calls(self):
        from ruinedfooocus.features.image_processing.depth_utility_pipeline import pipeline
        from ruinedfooocus.core.paths import resource_path

        forbidden = Mock(side_effect=AssertionError("generation dependency"))
        shared = NS(
            models=NS(get_models_by_path=forbidden, get_model_base=forbidden),
            state={},
            performance_settings=NS(apply=forbidden),
            resolution_settings=object(),
            path_manager=NS(model_paths={"temp_outputs_path": self.root}),
            shared_cache={},
        )
        controls = NS(get_settings=lambda data: data.get("controlnet", {}))
        for kind in utilities.KINDS:
            with self.subTest(kind=kind):
                data = self.capture(
                    kind, task_id=1, task_type="tool_call", silent=True, index=(0, 1)
                )
                pipe = pipeline()

                def process(gen_data, callback):
                    gen_data["image_utility_source"] = {
                        "operation": kind,
                        "width": 39,
                        "height": 21,
                        "components": [{"name": "local.safetensors", "role": "source", "size": 4}],
                    }
                    return [self.source.copy()]

                pipe.process = process
                route = {"state": {"pipeline": None}, "shared": shared, "controlnet": controls}
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
                    result = render(data)
                self.assertEqual(result, [str(self.root / "render.png")])
                with Image.open(result[0]) as output:
                    self.assertEqual((output.size, output.mode), ((39, 21), "RGBA"))
                    metadata = json.loads(output.info["parameters"])
                self.assertEqual(metadata["controlnet"], {"type": kind})
                self.assertNotIn("base_model_name", metadata)
                self.assertNotIn("steps", metadata)
                self.assertNotIn(str(self.root), json.dumps(metadata))
                replay = utilities.replay_values(metadata)
                self.assertIsNone(replay["input_image"])
                self.assertTrue(all(replay[key] is None for key in utilities.REPLAY_RESETS))
        forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
