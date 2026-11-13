"""Native MediaPipe region-mask bodies on frozen still-image CPU fixtures."""

import ast
from concurrent.futures import CancelledError
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.core.preparation import preparation_service, PreparationBusy, PreparationService
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.image_processing import depth_utilities as utilities
from ruinedfooocus.features.image_processing import utility_face as face
from ruinedfooocus.features.models.model_registry import load_registry
import test_depth_utilities as fixtures
from test_image_preparation import Manager
from test_native_model_coverage import native_method
from test_native_variant_defaults import NATIVE


class MediaPipeFaceMaskTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DepthUtilities()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def runtime(self):
        scope = self.fixture.runtime()
        # Deliberately tiny, independent polygons stand in for topology emitted
        # by a loaded model. Production always receives that model's own sets.
        points = np.array(
            [
                [12, 12],
                [18, 12],
                [18, 15],
                [12, 15],
                [6, 6],
                [10, 6],
                [10, 8],
                [6, 8],
                [25, 6],
                [29, 6],
                [29, 8],
                [25, 8],
                [7, 6],
                [9, 6],
                [9, 7],
                [7, 7],
                [4, 3],
                [33, 3],
                [33, 18],
                [4, 18],
            ],
            dtype=np.float32,
        )
        rings = dict(
            lips=(0, 1, 2, 3),
            left_eye=(4, 5, 6, 7),
            right_eye=(8, 9, 10, 11),
            irises=(12, 13, 14, 15),
            face_oval=(16, 17, 18, 19),
        )
        sets = {
            region: frozenset(zip(ring, (*ring[1:], ring[0]))) for region, ring in rings.items()
        }
        sets.update(left_eyebrow=frozenset(), right_eyebrow=frozenset(), nose=frozenset())
        sets.update(all=frozenset().union(*sets.values()))
        self.face_record = dict(landmarks_xy=points, bbox_xyxy=[4, 3, 33, 18], score=0.9)
        model = NS(
            canonical_data={},
            load_device="cpu",
            patchers={"short": NS(load_device="cpu"), "full": NS(load_device="cpu")},
            connection_sets=sets,
            detect_batch=Mock(return_value=[[self.face_record]]),
        )
        scope.update(
            FaceLandmarkerModel=lambda state: model,
            transformation_matrix_from_detection=lambda *a: np.eye(4),
        )
        tree = ast.parse((NATIVE / "comfy_extras/nodes_mediapipe.py").read_text(encoding="utf-8"))
        for node in tree.body:
            target = (
                node.target
                if isinstance(node, ast.AnnAssign)
                else node.targets[0]
                if isinstance(node, ast.Assign)
                else None
            )
            if getattr(target, "id", "") in (
                "_CONTOUR_PARTS",
                "_ALL_CONNECTION_PARTS",
                "_MASK_REGIONS",
                "_MASK_CUSTOM_FEATURES",
            ):
                exec(
                    compile(ast.Module(body=[node], type_ignores=[]), "native constants", "exec"),
                    scope,
                )
        self.assertEqual(face.MASK_REGIONS, scope["_MASK_REGIONS"])
        fixtures.native_functions(
            "comfy_extras/nodes_mediapipe.py",
            (
                "_image_to_uint8",
                "_parse_color",
                "_fill_missing_frames",
                "_draw_mesh",
                "_ordered_rings",
            ),
            scope,
        )
        nodes = {
            name: native_method("comfy_extras/nodes_mediapipe.py", name, "execute", scope)
            for name in (
                "LoadMediaPipeFaceLandmarker",
                "MediaPipeFaceLandmarker",
                "MediaPipeFaceMeshVisualize",
                "MediaPipeFaceMask",
            )
        }
        nodes["MediaPipeFaceMask"].execute = Mock(wraps=nodes["MediaPipeFaceMask"].execute)
        self.mask_call = nodes["MediaPipeFaceMask"].execute
        scope["IO"] = scope["io"]
        to_image = native_method("comfy_extras/nodes_mask.py", "MaskToImage", "execute", scope)
        to_image.execute = Mock(wraps=to_image.execute)
        self.to_image = to_image.execute
        self.fixture.enterContext(
            patch.dict(
                "sys.modules",
                {
                    "comfy_extras.nodes_mediapipe": NS(**nodes),
                    "comfy_extras.nodes_mask": NS(MaskToImage=to_image),
                },
            )
        )
        return model

    def capture(self, **options):
        return self.fixture.capture("face-landmarks", face_output="region-mask", **options)

    def test_parameters_are_explicit_typed_and_only_active_custom_selection_must_be_nonempty(self):
        self.assertEqual(face.parameters({})["face_output"], "overlay")
        empty = {"face_mask_" + name: False for name in face.MASK_REGIONS}
        self.assertEqual(
            face.parameters(dict(empty, face_mask_regions="custom"))["face_output"], "overlay"
        )
        for values in (
            dict(empty, face_output="region-mask", face_mask_regions="custom"),
            dict(face_output="video"),
            dict(face_mask_regions="guessed-landmarks"),
            dict(face_mask_lips=1),
            dict(face_mask_left_eye="true"),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                face.parameters(values)

    def test_native_custom_regions_use_emitted_topology_frozen_scalars_pixels_and_source_alpha(
        self,
    ):
        model = self.runtime()
        original = self.fixture.source.tobytes()
        self.fixture.source.info["original"] = "retained"
        data = self.capture(
            face_mask_regions="custom", face_mask_face_oval=False, face_mask_lips=True
        )
        digest = hashlib.sha256(data[utilities.KEY].rgba).hexdigest()
        data.update(face_output="overlay", face_mask_lips=False)
        data["input_image"].paste((0, 0, 0, 0), (0, 0, 39, 21))
        result = utilities.process(NS(), data, self.fixture.manager, Mock())[0]
        self.assertEqual((result.mode, result.size), ("RGBA", self.fixture.source.size))
        self.assertEqual(result.getpixel((14, 13)), (255, 255, 255, 128))
        self.assertEqual(result.getpixel((8, 7)), (0, 0, 0, 128))
        self.assertEqual(self.fixture.source.tobytes(), original)
        self.assertEqual(self.fixture.source.info["original"], "retained")
        information = data["image_utility_source"]
        self.assertEqual(information["mask_role"], "inpaint-selection")
        self.assertEqual(information["source_sha256"], digest)
        self.assertFalse(information["mask_empty"])
        self.assertEqual(information["native_output"], "FACE_LANDMARKS → MASK → IMAGE")
        landmarks, regions = self.mask_call.call_args.args
        self.assertIs(landmarks["connection_sets"], model.connection_sets)
        self.assertEqual(
            regions,
            dict(
                regions="custom",
                face_oval=False,
                lips=True,
                left_eye=False,
                right_eye=False,
                irises=False,
            ),
        )
        self.to_image.assert_called_once()
        replay = utilities.replay_values(utilities.output_metadata(data))
        self.assertEqual(replay["face_output"], "region-mask")
        self.assertTrue(replay["face_mask_lips"])
        self.assertIsNone(replay["input_image"])

    def test_native_all_regions_union_multiple_faces_and_no_face_empty_result_is_explicit(self):
        model = self.runtime()
        second = dict(self.face_record, landmarks_xy=self.face_record["landmarks_xy"] + [8, 0])
        model.detect_batch.return_value = [[self.face_record, second]]
        data = self.capture(
            face_max_count=2,
            face_mask_regions="custom",
            face_mask_face_oval=False,
            face_mask_lips=True,
        )
        owner = NS()
        result = utilities.process(owner, data, self.fixture.manager, Mock())[0]
        self.assertEqual(result.getpixel((22, 13)), (255, 255, 255, 128))
        self.assertEqual(data["image_utility_source"]["face_count"], 2)
        all_regions = self.capture()
        result = utilities.process(owner, all_regions, self.fixture.manager, Mock())[0]
        self.assertEqual(result.getpixel((5, 5)), (255, 255, 255, 128))
        self.assertEqual(self.mask_call.call_args.args[1], {"regions": "all"})
        model.detect_batch.return_value = [[]]
        empty = self.capture()
        result = utilities.process(owner, empty, self.fixture.manager, Mock())[0]
        self.assertEqual(result.getextrema(), ((0, 0), (0, 0), (0, 0), (128, 128)))
        self.assertEqual(empty["image_utility_source"]["mask_empty_reason"], "no-face")
        self.assertTrue(empty["image_utility_source"]["mask_empty"])
        self.assertEqual(empty["image_utility_source"]["face_count"], 0)

    def test_overlay_remains_image_only_and_empty_detection_preserves_original_pixels(self):
        model = self.runtime()
        model.detect_batch.return_value = [[]]
        data = self.fixture.capture("face-landmarks")
        result = utilities.process(NS(), data, self.fixture.manager, Mock())[0]
        self.assertEqual(result.tobytes(), self.fixture.source.tobytes())
        self.assertNotIn("mask_role", data["image_utility_source"])
        self.mask_call.assert_not_called()

    def test_precision_mask_retains_exact_source_frame_and_fractional_alpha(self):
        self.runtime()
        self.fixture.source.putalpha(37)
        self.fixture.source.getexif()[274] = 6
        data = self.capture(output_format="png", output_bit_depth="16-bit")
        # The frozen image is oriented before detection and native rasterization.
        self.assertEqual(data[utilities.KEY].size, (21, 39))
        result = utilities.process(NS(), data, self.fixture.manager, Mock())[0]
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual((result.dtype, result.shape), (np.dtype("float32"), (39, 21, 4)))
        np.testing.assert_array_equal(result[..., 0], result[..., 1])
        np.testing.assert_array_equal(result[..., 0], result[..., 2])
        np.testing.assert_array_equal(result[..., 3], np.full((39, 21), 37 / 255, np.float32))

    def test_cancellation_releases_real_component_lease_and_cached_patchers_can_be_unloaded(self):
        model = self.runtime()
        data, owner = self.capture(), NS()
        native, finished = self.mask_call._mock_wraps, []

        def mask_under_lease(*args, **kwargs):
            with self.assertRaises(PreparationBusy):
                with preparation_service.artifact_lease(
                    self.fixture.face, write=True, blocking=False
                ):
                    self.fail("Native face model must retain its component read lease.")
            result = native(*args, **kwargs)
            finished.append(True)
            return result

        def cancelled():
            if finished:
                raise CancelledError()

        self.mask_call.side_effect = mask_under_lease
        with self.assertRaises(CancelledError):
            utilities.process(owner, data, self.fixture.manager, cancelled)
        with preparation_service.artifact_lease(
            self.fixture.face, write=True, blocking=False
        ) as path:
            self.assertEqual(path, self.fixture.face)
        utilities.clear_model(owner)
        self.assertCountEqual(
            [call.args[0] for call in self.fixture.memory.unload_model_and_clones.call_args_list],
            list(model.patchers.values()),
        )
        self.assertNotIn("image_utility_source", data)

    def test_native_device_policy_is_preserved_and_failed_model_cache_is_unloaded(self):
        from ruinedfooocus.features.image_processing import utility_device as device

        model = self.runtime()
        policy = device.UtilityPolicy(torch.device("cuda:0"), torch.device("cpu"), torch.float32)
        data = self.capture()
        owner = NS()
        with patch.object(device, "resolve_policy", return_value=policy):
            utilities.process(owner, data, self.fixture.manager, Mock())
        self.assertEqual(str(model.load_device), "cuda:0")
        self.assertTrue(all(str(p.load_device) == "cuda:0" for p in model.patchers.values()))
        self.assertEqual(data["image_utility_source"]["inference_device"], "cuda:0")
        self.assertEqual(data["image_utility_source"]["inference_dtype"], "torch.float32")
        self.assertEqual(data["image_utility_source"]["raster_device"], "cpu")
        # This is an actual native-node fixture, with detection standing in for
        # model execution. It intentionally never allocates accelerator tensors.
        model.detect_batch.side_effect = [
            RuntimeError("CUDA unsupported device"),
            [[self.face_record]],
        ]
        retry = self.capture()
        with (
            patch.object(device, "resolve_policy", return_value=policy),
            patch.object(device, "warn_cpu_fallback") as warning,
        ):
            utilities.process(owner, retry, self.fixture.manager, Mock())
        record = retry["image_utility_source"]["device_policy"]
        self.assertEqual(record["device"], "cpu")
        self.assertTrue(record["fallback"])
        self.assertIn("CUDA unsupported device", record["fallback_reason"])
        warning.assert_called_once()
        self.assertEqual(self.fixture.memory.unload_model_and_clones.call_count, 2)
        self.assertFalse(torch.cuda.is_initialized())

    def test_explicit_cpu_is_authoritative_for_fresh_native_patchers(self):
        from ruinedfooocus.features.image_processing import utility_device as device

        model = self.runtime()
        model.load_device = "cuda:0"
        for patcher in model.patchers.values():
            patcher.load_device = "cuda:0"
            patcher.offload_device = "cpu"
        data = self.capture()
        with patch.object(sys, "argv", ["face-fixture", "--cpu"]):
            utilities.process(NS(), data, self.fixture.manager, Mock())
        self.assertEqual(str(model.load_device), "cpu")
        self.assertTrue(all(str(p.load_device) == "cpu" for p in model.patchers.values()))
        self.assertTrue(all(p.offload_device == "cpu" for p in model.patchers.values()))
        self.assertEqual(data["image_utility_source"]["device_policy"]["requested"], "cpu")
        self.assertFalse(data["image_utility_source"]["device_policy"]["fallback"])

    def test_host_transfer_failure_is_inside_recovery_and_reloads_native_model_on_cpu(self):
        from ruinedfooocus.features.image_processing import utility_device as device

        class FailedHostTransfer(torch.Tensor):
            def cpu(self, *args, **kwargs):
                raise RuntimeError("CUDA asynchronous host transfer failed")

        model = self.runtime()
        native, calls = self.to_image._mock_wraps, []

        def preview(*args, **kwargs):
            output = native(*args, **kwargs)
            calls.append(True)
            return (
                fixtures.NodeOutput(output[0].as_subclass(FailedHostTransfer))
                if len(calls) == 1
                else output
            )

        self.to_image.side_effect = preview
        policy = device.UtilityPolicy(torch.device("cuda:0"), torch.device("cpu"), torch.float32)
        data = self.capture()
        with (
            patch.object(device, "resolve_policy", return_value=policy),
            patch.object(device, "warn_cpu_fallback") as warning,
        ):
            result = utilities.process(NS(), data, self.fixture.manager, Mock())[0]
        self.assertEqual(result.size, self.fixture.source.size)
        self.assertEqual(str(model.load_device), "cpu")
        self.assertEqual(self.fixture.comfy.utils.load_torch_file.call_count, 2)
        self.assertEqual(data["image_utility_source"]["inference_device"], "cpu")
        self.assertIn(
            "host transfer failed", data["image_utility_source"]["device_policy"]["fallback_reason"]
        )
        warning.assert_called_once()

    def test_gpu_only_offload_rejects_recovery_before_second_constructor_or_success_metadata(self):
        from ruinedfooocus.features.image_processing import utility_device as device

        model = self.runtime()
        self.fixture.memory.text_encoder_offload_device = lambda: torch.device("cuda:0")
        model.detect_batch.side_effect = RuntimeError("CUDA unsupported device")
        policy = device.UtilityPolicy(torch.device("cuda:0"), torch.device("cpu"), torch.float32)
        data, owner = self.capture(), NS()
        with (
            patch.object(device, "resolve_policy", return_value=policy),
            patch.object(device, "warn_cpu_fallback") as warning,
        ):
            with self.assertRaisesRegex(RuntimeError, "Disable GPU-only and use Auto.*--cpu"):
                utilities.process(owner, data, self.fixture.manager, Mock())
        self.assertEqual(self.fixture.comfy.utils.load_torch_file.call_count, 1)
        self.assertIsNone(owner._depth_utility_cache)
        self.assertNotIn("image_utility_source", data)
        warning.assert_not_called()

    def test_bad_native_masks_and_cancelled_masks_never_publish_mask_provenance(self):
        self.runtime()
        for value in (
            torch.zeros(1, 2, 2),
            torch.full((1, 21, 39), float("nan")),
            torch.full((1, 21, 39), 2.0),
        ):
            with self.subTest(shape=value.shape):
                self.mask_call.side_effect = lambda *a, **k: fixtures.NodeOutput(value)
                data = self.capture()
                with self.assertRaisesRegex(ValueError, "invalid selection pixels"):
                    utilities.process(NS(), data, self.fixture.manager, Mock())
                self.assertNotIn("image_utility_source", data)
        self.mask_call.side_effect = None
        finished = []
        native = self.mask_call._mock_wraps

        def cancel_after_native(*args, **kwargs):
            result = native(*args, **kwargs)
            finished.append(True)
            return result

        self.mask_call.side_effect = cancel_after_native
        data = self.capture()

        def cancelled():
            if finished:
                raise CancelledError()

        with self.assertRaises(CancelledError):
            utilities.process(NS(), data, self.fixture.manager, cancelled)
        self.assertNotIn("image_utility_source", data)


class MediaPipePortCompatibilityTests(unittest.TestCase):
    def test_pinned_port_imports_on_installed_cpu_python_without_pip_mediapipe_api(self):
        source = NATIVE / "comfy_extras/mediapipe/face_landmarker.py"
        spec = importlib.util.spec_from_file_location("rf_test_native_face_landmarker", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertIn("detector_variant", inspect.signature(module.FaceLandmarker).parameters)
        self.assertTrue(callable(module.FaceLandmarker.detect_batch))
        self.assertFalse(torch.cuda.is_initialized())


class MediaPipeAcquisitionTests(unittest.TestCase):
    KEY = "detection/mediapipe_face_fp32.safetensors"
    REVISION = "f787efc1ad99492ec7d5b6564c0dd6c48d84ee49"
    SIZE = 5423900
    SHA = "a98c4806081d40eba35102a0f6dc0000c2e1388b72cf24e691703d0605bd888a"

    def setUp(self):
        from PIL import Image

        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.manager = Manager(self.root)
        self.manager.DOWNLOADABLE_FILES = json.loads(
            resource_path("model_catalogue", "native_mediapipe_face.json").read_text(
                encoding="utf-8"
            )
        )
        self.source = Image.new("RGBA", (13, 9), (30, 50, 80, 41))
        self.enterContext(patch.dict(os.environ, {"RF_OFFLINE": "0", "HF_HUB_OFFLINE": "0"}))

    def capture(self, *, offline=False, name=None):
        return assets.capture_image_assets(
            dict(
                input_image=self.source,
                image_utility_model=name or Path(self.KEY).name,
                face_output="region-mask",
            ),
            self.manager,
            {"type": "face-landmarks"},
            offline=offline,
        )

    def test_immutable_publisher_tuple_matches_registry_role_and_native_loader_contract(self):
        source = self.manager.DOWNLOADABLE_FILES[self.KEY]
        entry = load_registry().annotations_for(self.KEY, source)
        self.assertEqual((source["size"], source["sha256"]), (self.SIZE, self.SHA))
        self.assertIn("/resolve/" + self.REVISION + "/detection/", source["url"])
        self.assertEqual((source["path"], entry["role"]), ("path_detection", "image_utility_model"))
        self.assertEqual(
            (entry["native_class"], entry["component_role"]), ("FaceLandmarker", "face-landmarker")
        )
        self.assertEqual(
            (entry["artifact"]["bytes"], entry["artifact"]["sha256"]), (self.SIZE, self.SHA)
        )
        self.assertEqual(
            load_registry().profile(entry["profile_id"])["license"]["value"], "Apache-2.0"
        )
        self.assertFalse(
            entry["chat_weight"] or entry["uses_chat_runtime"] or entry["auto_recipe_supported"]
        )
        source_tree = ast.parse(
            (NATIVE / "comfy_extras/nodes_mediapipe.py").read_text(encoding="utf-8")
        )
        loader = next(
            node
            for node in source_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LoadMediaPipeFaceLandmarker"
        )
        self.assertIn(
            "detection",
            [node.value for node in ast.walk(loader) if isinstance(node, ast.Constant)],
        )

    def test_registered_missing_bundle_freezes_broker_request_and_offline_blocks_before_transfer(
        self,
    ):
        online = self.capture()
        (artifact,) = online[assets.KEY].artifacts
        request = artifact.request
        self.assertEqual((artifact.role, artifact.signature), ("face-landmarker", None))
        self.assertEqual((request.expected_size, request.sha256), (self.SIZE, self.SHA))
        self.assertEqual(request.destination, self.root / self.KEY)
        self.assertEqual(assets.image_readiness(online).state, "preparable")
        self.assertEqual(assets.image_readiness(online).missing_bytes, self.SIZE)
        self.manager.DOWNLOADABLE_FILES[self.KEY]["url"] += "?changed"
        self.manager.root = self.root / "changed"
        self.assertEqual(request.destination, self.root / self.KEY)
        self.assertNotIn("?changed", request.url)
        self.manager.root = self.root
        self.manager.DOWNLOADABLE_FILES[self.KEY]["url"] = request.url
        offline = self.capture(offline=True)
        transport = Mock(side_effect=AssertionError("offline cannot transfer"))
        service = PreparationService(transport=transport, staging_root=self.root / "staging")
        self.addCleanup(service.close)
        with self.assertRaisesRegex(ValueError, "Offline mode.*face-landmarker"):
            assets.prepare_image_assets(offline, check_cancelled=Mock(), service=service)
        transport.assert_not_called()

    def test_changed_catalogue_source_and_missing_custom_bundle_are_rejected_at_capture(self):
        with self.assertRaisesRegex(ValueError, "compatible registered source"):
            self.capture(name="custom-missing.safetensors")
        self.manager.DOWNLOADABLE_FILES[self.KEY]["url"] += "?changed"
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            self.capture()


if __name__ == "__main__":
    unittest.main()
