"""Pinned MoGe normal rendering and panorama depth on controlled CPU geometry."""

from concurrent.futures import CancelledError
from contextlib import contextmanager
import importlib.util
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.image_processing import depth_utilities as utilities
from ruinedfooocus.features.image_processing import utility_moge as moge
from ruinedfooocus.features.image_processing import utility_device as device_policy
import test_depth_utilities as fixtures
from test_native_model_coverage import native_method
from test_native_variant_defaults import NATIVE


_panorama_spec = importlib.util.spec_from_file_location(
    "rf_test_native_moge_panorama", NATIVE / "comfy/ldm/moge/panorama.py"
)
PANORAMA = importlib.util.module_from_spec(_panorama_spec)
_panorama_spec.loader.exec_module(PANORAMA)


class Progress(fixtures.Progress):
    def set_description(self, text):
        self.description = text


class MoGeGeometryOutputTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DepthUtilities()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.geometry = None

    def runtime(self, infer):
        scope = self.fixture.runtime()
        self.model = NS(
            patcher=NS(load_device=torch.device("cpu"), offload_device=torch.device("cpu")),
            infer=Mock(side_effect=infer),
            load_device=torch.device("cpu"),
            dtype=torch.float32,
        )
        scope["MoGeModel"] = lambda state: self.model
        self.fixture.memory.load_model_gpu = Mock()
        self.fixture.memory.text_encoder_offload_device = Mock(return_value=torch.device("cpu"))
        fixtures.native_functions(
            "comfy_extras/nodes_moge.py", ("_normalize_disparity", "_normals_from_points"), scope
        )
        # Read the actual pinned helper module, including its SciPy solver. The
        # replay runner supplies RF_TEST_COMFY_SOURCE; no vendor fixture copy.
        scope["tqdm"] = Progress
        for name in (
            "get_panorama_cameras",
            "split_panorama_image",
            "merge_panorama_depth",
            "spherical_uv_to_directions",
            "_uv_grid",
        ):
            scope[name] = getattr(PANORAMA, name)
        self.merge = scope["merge_panorama_depth"] = Mock(wraps=PANORAMA.merge_panorama_depth)
        nodes = {
            name: native_method("comfy_extras/nodes_moge.py", name, "execute", scope)
            for name in ("LoadMoGeModel", "MoGeInference", "MoGePanoramaInference", "MoGeRender")
        }
        native_render = nodes["MoGeRender"].execute

        def render(geometry, output):
            self.geometry = geometry
            return native_render(geometry, output)

        self.render = nodes["MoGeRender"].execute = Mock(side_effect=render)
        self.fixture.enterContext(
            patch.dict("sys.modules", {"comfy_extras.nodes_moge": NS(**nodes)})
        )
        return scope

    def capture(self, **options):
        return self.fixture.capture("moge-depth", **options)

    @staticmethod
    def sphere(image, **options):
        batch, _, height, width = image.shape
        points = torch.zeros(batch, height, width, 3)
        points[..., 2] = 2
        return {"points": points, "mask": torch.ones(batch, height, width, dtype=torch.bool)}

    def test_parameters_preserve_depth_default_and_validate_bounded_modes(self):
        defaults = moge.parameters({})
        self.assertEqual(
            (
                defaults["moge_output"],
                defaults["moge_projection"],
                defaults["moge_normal_convention"],
            ),
            ("depth", "perspective", "opengl"),
        )
        for invalid in (
            {"moge_output": "mesh"},
            {"moge_projection": "cubemap"},
            {"moge_normal_convention": "camera"},
            {"moge_split_resolution": 255},
            {"moge_merge_resolution": 8193},
            {"moge_panorama_batch_size": 13},
            {"moge_panorama_batch_size": float("nan")},
            {"moge_output": "normal", "moge_projection": "panorama"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                moge.parameters(invalid)

    def test_native_predicted_normals_conventions_frozen_options_and_float_alpha(self):
        def infer(image, **options):
            _, _, height, width = image.shape
            normal = torch.tensor([0.2, 0.6, -0.8]).expand(1, height, width, 3).clone()
            return {"normal": normal}

        self.runtime(infer)
        owner = NS()
        original = self.fixture.source.tobytes()
        for convention, expected in (("opengl", [0.6, 0.2, 0.9]), ("directx", [0.6, 0.8, 0.9])):
            with self.subTest(convention=convention):
                data = self.capture(
                    moge_output="normal",
                    moge_normal_convention=convention,
                    output_format="exr",
                    output_bit_depth="32-bit float",
                )
                data.update(moge_output="depth", moge_normal_convention="directx")
                data["input_image"].paste((0, 0, 0, 0), (0, 0, 39, 21))
                result = utilities.process(owner, data, self.fixture.manager, Mock())[0]
                self.assertEqual(result.shape, (21, 39, 4))
                np.testing.assert_allclose(
                    result[..., :3], np.broadcast_to(expected, (21, 39, 3)), atol=1e-7
                )
                np.testing.assert_allclose(result[..., 3], 128 / 255, atol=1e-7)
                details = data["image_utility_source"]
                self.assertEqual(details["normal_source"], "native-predicted")
                self.assertEqual(details["normal_frame"], "native-camera")
                self.assertEqual(details["native_render_mode"], "normal_" + convention)
        self.assertEqual(self.fixture.source.tobytes(), original)
        self.fixture.comfy.utils.load_torch_file.assert_called_once()
        utilities.clear_model(owner)
        self.fixture.memory.unload_model_and_clones.assert_called_once_with(self.model.patcher)

    def test_native_point_derived_normals_and_tiny_source_rejection(self):
        def infer(image, **options):
            _, _, height, width = image.shape
            y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
            points = torch.stack((x, y, torch.ones_like(x) * 2), dim=-1).float()[None]
            return {"points": points}

        self.runtime(infer)
        data = self.capture(moge_output="normal")
        output = utilities.process(NS(), data, self.fixture.manager, Mock())[0]
        self.assertEqual(output.getpixel((10, 10)), (128, 128, 255, 128))
        self.assertEqual(data["image_utility_source"]["normal_source"], "native-point-derived")
        self.fixture.source = Image.new("RGBA", (1, 2))
        data = self.capture(moge_output="normal")
        with self.assertRaisesRegex(ValueError, "at least 2 × 2"):
            utilities.process(NS(), data, self.fixture.manager, Mock())
        self.assertNotIn("image_utility_source", data)

    def test_actual_panorama_split_merge_and_render_retain_source_canvas(self):
        self.runtime(self.sphere)
        data = self.capture(
            moge_projection="panorama",
            moge_split_resolution=256,
            moge_merge_resolution=256,
            moge_panorama_batch_size=5,
            moge_resolution_level=0,
            moge_refine_steps=0,
            output_format="exr",
            output_bit_depth="32-bit float",
        )
        result = utilities.process(NS(), data, self.fixture.manager, Mock())[0]
        self.assertEqual(result.shape, (21, 39, 4))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[..., 0], result[..., 1])
        np.testing.assert_array_equal(result[..., 1], result[..., 2])
        np.testing.assert_allclose(result[..., 3], 128 / 255, atol=1e-7)
        self.assertEqual(
            [call.args[0].shape[0] for call in self.model.infer.call_args_list], [5, 5, 2]
        )
        for call in self.model.infer.call_args_list:
            self.assertEqual(tuple(call.args[0].shape[1:]), (3, 256, 256))
            self.assertEqual(
                call.kwargs,
                dict(
                    resolution_level=0,
                    fov_x=90.0,
                    force_projection=True,
                    apply_mask=False,
                    apply_metric_scale=False,
                    refine_steps=0,
                ),
            )
        self.merge.assert_called_once()
        self.fixture.memory.load_model_gpu.assert_called_once_with(self.model.patcher)
        self.assertEqual(tuple(self.geometry["points"].shape), (1, 21, 39, 3))
        np.testing.assert_allclose(
            self.geometry["points"].norm(dim=-1), self.geometry["depth"], atol=1e-6
        )
        self.assertNotIn("intrinsics", self.geometry)
        self.assertNotIn("normal", self.geometry)
        details = data["image_utility_source"]
        self.assertEqual(details["native_inference"], "MoGePanoramaInference")
        self.assertEqual(details["depth_measure"], "native-radial-distance")
        self.assertEqual(details["projection"], "equirectangular")

    def test_native_normal_nonfinite_or_wrong_batch_cannot_publish(self):
        self.runtime(lambda image, **options: {})
        for normal in (
            torch.full((1, 21, 39, 3), float("nan")),
            torch.zeros(2, 21, 39, 3),
            torch.zeros(1, 20, 39, 3),
        ):
            with self.subTest(shape=normal.shape):
                self.model.infer.side_effect = lambda *args, **kwargs: {"normal": normal}
                data = self.capture(moge_output="normal")
                with self.assertRaisesRegex(ValueError, "invalid image dimensions or values"):
                    utilities.process(NS(), data, self.fixture.manager, Mock())
                self.assertNotIn("image_utility_source", data)

    def test_native_panorama_uncovered_pixels_use_native_far_fill(self):
        self.runtime(self.sphere)

        def merge(width, height, *args, **kwargs):
            depth = np.broadcast_to(
                np.linspace(1, 4, width, dtype=np.float32), (height, width)
            ).copy()
            mask = np.ones((height, width), dtype=bool)
            mask[:, 0] = False
            return depth, mask

        self.merge.side_effect = merge
        data = self.capture(moge_projection="panorama", moge_split_resolution=256)
        output = utilities.process(NS(), data, self.fixture.manager, Mock())[0]
        depth, mask = self.geometry["depth"], self.geometry["mask"]
        self.assertTrue(
            torch.allclose(
                depth[~mask], (torch.quantile(depth[mask], 0.95) * 5).expand_as(depth[~mask])
            )
        )
        self.assertEqual(output.getpixel((0, 10))[-1], 128)
        self.assertEqual(
            data["image_utility_source"]["convention"], "white-near; uncovered-native-far-fill"
        )

    def test_cancel_between_panorama_view_batches_releases_source_lease_and_publishes_nothing(self):
        self.runtime(self.sphere)
        data = self.capture(moge_projection="panorama", moge_split_resolution=256)
        active = []
        prepared = assets.prepared_asset

        @contextmanager
        def lease(*args, **kwargs):
            with prepared(*args, **kwargs) as path:
                active.append(True)
                try:
                    yield path
                finally:
                    active.pop()

        def infer(image, **options):
            self.assertEqual(active, [True])
            return self.sphere(image, **options)

        self.model.infer.side_effect = infer

        def cancelled():
            if self.model.infer.call_count:
                raise CancelledError()

        with patch.object(assets, "prepared_asset", lease), self.assertRaises(CancelledError):
            utilities.process(NS(), data, self.fixture.manager, cancelled)
        self.assertEqual(active, [])
        self.model.infer.assert_called_once()
        self.merge.assert_not_called()
        self.render.assert_not_called()
        self.assertNotIn("image_utility_source", data)

    def test_cancel_after_panorama_merge_prevents_render_and_provenance(self):
        self.runtime(self.sphere)
        data = self.capture(moge_projection="panorama", moge_split_resolution=256)

        def cancelled():
            if self.merge.call_count:
                raise CancelledError()

        with self.assertRaises(CancelledError):
            utilities.process(NS(), data, self.fixture.manager, cancelled)
        self.merge.assert_called_once()
        self.render.assert_not_called()
        self.assertNotIn("image_utility_source", data)

    def test_actual_host_copy_device_failure_rebuilds_cpu_model_and_reports_fallback(self):
        def infer(image, **options):
            return {"normal": torch.zeros(1, 21, 39, 3)}

        scope = self.runtime(infer)
        first = self.model
        second = NS(
            patcher=NS(load_device=torch.device("cpu"), offload_device=torch.device("cpu")),
            infer=Mock(side_effect=infer),
            load_device=torch.device("cpu"),
            dtype=torch.float32,
        )
        scope["MoGeModel"] = Mock(side_effect=[first, second])
        native_render = self.render.side_effect

        class HostCopyFailure(torch.Tensor):
            def cpu(self, *args, **kwargs):
                raise RuntimeError("HIP error during native output host transfer")

        def render(*args, **kwargs):
            output = native_render(*args, **kwargs)[0]
            if self.render.call_count == 1:
                output = output.as_subclass(HostCopyFailure)
            return fixtures.NodeOutput(output)

        self.render.side_effect = render
        selected = device_policy.UtilityPolicy(
            torch.device("cuda:0"), torch.device("cpu"), torch.float32
        )
        data = self.capture(moge_output="normal")
        owner = NS()
        with (
            patch.object(device_policy, "resolve_policy", return_value=selected),
            patch.object(device_policy, "warn_cpu_fallback") as warning,
        ):
            output = utilities.process(owner, data, self.fixture.manager, Mock())[0]
        self.assertEqual(output.size, self.fixture.source.size)
        self.assertEqual(str(first.load_device), "cuda:0")
        self.assertEqual(str(first.patcher.load_device), "cuda:0")
        self.assertEqual(str(first.patcher.offload_device), "cpu")
        self.assertEqual(str(second.load_device), "cpu")
        self.assertEqual(second.dtype, torch.float32)
        self.fixture.memory.unload_model_and_clones.assert_called_once_with(first.patcher)
        self.assertIs(owner._depth_utility_cache[1], second)
        self.assertEqual(self.fixture.comfy.utils.load_torch_file.call_count, 2)
        metadata = data["image_utility_source"]
        self.assertEqual(metadata["inference_device"], "cpu")
        self.assertTrue(metadata["device_policy"]["fallback"])
        self.assertIn("host transfer", metadata["device_policy"]["fallback_reason"])
        warning.assert_called_once()

    def test_explicit_cpu_retargets_fresh_patcher_and_gpu_only_offload_fails_before_load(self):
        self.runtime(lambda image, **options: {"normal": torch.zeros(1, 21, 39, 3)})
        self.model.load_device = self.model.patcher.load_device = torch.device("cuda:0")
        data = self.capture(moge_output="normal")
        selected = device_policy.UtilityPolicy(
            torch.device("cpu"), torch.device("cpu"), torch.float32, requested="cpu"
        )
        with patch.object(device_policy, "resolve_policy", return_value=selected):
            utilities.process(NS(), data, self.fixture.manager, Mock())
        self.assertEqual(str(self.model.load_device), "cpu")
        self.assertEqual(str(self.model.patcher.load_device), "cpu")
        self.assertEqual(str(self.model.patcher.offload_device), "cpu")
        self.fixture.comfy.utils.load_torch_file.reset_mock()
        self.fixture.memory.text_encoder_offload_device.return_value = torch.device("cuda:0")
        data = self.capture(moge_output="normal")
        with (
            patch.object(device_policy, "resolve_policy", return_value=selected),
            self.assertRaisesRegex(RuntimeError, "Disable GPU-only and use Auto"),
        ):
            utilities.process(NS(), data, self.fixture.manager, Mock())
        self.fixture.comfy.utils.load_torch_file.assert_not_called()
        self.assertNotIn("image_utility_source", data)


if __name__ == "__main__":
    unittest.main()
