"""Consumer bounds and exact native CPU formulas across the E transport seam."""

from concurrent.futures import CancelledError
from contextlib import contextmanager
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch

from ruinedfooocus.features import generation
from ruinedfooocus.features.image_processing import latent_composition as composition


class Artifacts(list):
    def __init__(self, images, files):
        super().__init__(images)
        self.tensor_files = tuple(files)


class LatentCompositionTests(unittest.TestCase):
    @contextmanager
    def transport(self, *, scale=8):
        # Mock only E's owned file boundary; execute the real consumer and pinned
        # native formulas. E's serializer/receipt tests cover the actual files.
        context = (("version", 2), ("layout", dict(kind="native-still", scale=scale)))
        self.primary = NS(shape=(1, 4, 4, 4), context=context, dtype="F16", sha256="a" * 64)
        self.secondary = NS(shape=(1, 4, 4, 4), context=context, dtype="F16", sha256="b" * 64)
        self.base = {"samples": torch.zeros((1, 4, 4, 4), device="cpu")}
        self.overlay = {"samples": torch.ones((1, 4, 4, 4), device="cpu")}
        exported = NS(path="owned-result.latent")
        self.result = None

        def capture(primary, secondary, manager, *, data):
            self.assertEqual((primary, secondary), (b"primary", b"secondary"))
            return dict(
                data,
                _native_tensor_plan=NS(inputs=(self.primary, self.secondary)),
                _native_tensor_exports=[],
                native_latent_composition_source={},
            )

        def save(data, primary, result, *, composition):
            self.assertIs(primary, self.primary)
            self.result = result
            self.operation = composition
            return exported

        api = NS(
            KEY="_native_tensor_plan",
            EXPORT_KEY="_native_tensor_exports",
            capture_composition_inputs=Mock(side_effect=capture),
            assert_composable=Mock(),
            load_composition_latent=Mock(
                side_effect=lambda data, item: self.base if item is self.primary else self.overlay
            ),
            save_composed_latent=Mock(side_effect=save),
            TensorGenerationArtifacts=Artifacts,
        )
        with (
            patch.object(generation, "native_tensor_io", api, create=True),
            patch.dict(sys.modules, {"ruinedfooocus.features.generation.native_tensor_io": api}),
        ):
            yield api

    def data(self, **values):
        return dict(
            latent_composition_primary=b"primary",
            latent_composition_secondary=b"secondary",
            **values,
        )

    def test_capture_keeps_only_operation_and_lifecycle_without_model_or_prompt_state(self):
        with self.transport() as api:
            data = composition.capture(
                self.data(
                    prompt=object(), loras=object(), base_model_name="missing", task_id="owned"
                ),
                object(),
            )
        self.assertNotIn("base_model_name", data)
        self.assertNotIn("prompt", data)
        self.assertNotIn("loras", data)
        self.assertNotIn(composition.INPUTS[0], data)
        self.assertEqual(
            api.capture_composition_inputs.call_args.kwargs["data"], {"task_id": "owned"}
        )
        self.assertEqual(data["controlnet"], {"type": "latent-compose"})

    def test_mask_is_owned_and_native_selection_keeps_original_tensor_inputs(self):
        mask = Image.new("L", (32, 32), 255)
        with self.transport():
            data = composition.capture(
                self.data(latent_composition_mode="masked", latent_composition_mask=mask), object()
            )
            mask.paste(0, (0, 0, 32, 32))
            result = composition.process(data, lambda: None)
            self.assertEqual(result, [])
            self.assertEqual(len(result.tensor_files), 1)
            self.assertTrue(torch.all(self.result["samples"] == 1).item())
            self.assertEqual(torch.count_nonzero(self.base["samples"]).item(), 0)
            self.assertEqual(self.operation["native_node"], "LatentCompositeMasked")
            self.assertEqual(self.operation["preserved_shape"], [1, 4, 4, 4])

    def test_codec16_coordinates_convert_pixels_to_native_eight_units(self):
        with self.transport(scale=16):
            data = composition.capture(self.data(latent_composition_x=32), object())
            composition.process(data, lambda: None)
            self.assertEqual(self.operation["native_x"], 16)
            self.assertEqual(self.operation["x_pixels"], 32)
            self.assertEqual(self.operation["destination_pixels"], [64, 64])
            self.assertEqual(torch.count_nonzero(self.result["samples"][:, :, :, :2]).item(), 0)
            self.assertTrue(torch.all(self.result["samples"][:, :, :, 2:] == 1).item())

    def test_bad_grid_offsets_feather_and_mask_frames_fail_before_native_load(self):
        for changed in (
            dict(latent_composition_x=1),
            dict(latent_composition_y=32),
            dict(latent_composition_x=24, latent_composition_feather=16),
            dict(
                latent_composition_mode="masked", latent_composition_mask=Image.new("L", (31, 32))
            ),
        ):
            with self.subTest(changed=changed), self.transport() as api:
                with self.assertRaises(ValueError):
                    composition.capture(self.data(**changed), object())
                api.load_composition_latent.assert_not_called()
                api.save_composed_latent.assert_not_called()

    def test_nonfinite_fractional_boolean_or_unknown_settings_fail_before_transport(self):
        for changed in (
            dict(latent_composition_x=True),
            dict(latent_composition_y=float("inf")),
            dict(latent_composition_x=0.5),
            dict(latent_composition_mode="blend"),
            dict(latent_composition_mode="masked", latent_composition_feather=8),
            dict(latent_composition_mask=Image.new("L", (32, 32))),
        ):
            with self.subTest(changed=changed), self.transport() as api:
                with self.assertRaises(ValueError):
                    composition.capture(self.data(**changed), object())
                api.capture_composition_inputs.assert_not_called()

    def test_cancellation_and_stale_native_source_never_export_or_publish_metadata(self):
        with self.transport() as api:
            data = composition.capture(self.data(), object())
            with self.assertRaises(CancelledError):
                composition.process(data, Mock(side_effect=CancelledError()))
            api.load_composition_latent.assert_not_called()
            api.save_composed_latent.assert_not_called()
            with patch.object(composition.native, "source_signatures", return_value=()):
                with self.assertRaisesRegex(ValueError, "changed"):
                    composition.process(data, lambda: None)
            api.save_composed_latent.assert_not_called()
            self.assertEqual(data["native_latent_composition_source"], {})

    def test_api_operation_bounds_require_two_uploads_one_result_and_typed_control(self):
        good = dict(
            controlnet={"type": "latent-compose"},
            latent_composition_primary="cHJpbWFyeQ==",
            latent_composition_secondary="c2Vjb25kYXJ5",
        )
        self.assertEqual(composition.request_values(good), composition.DEFAULTS)
        for changed in (
            dict(n=2),
            dict(latent_composition_primary=None),
            dict(controlnet={"type": "latent-compose", "model": "missing"}),
            dict(native_latent_input="generation"),
            dict(controlnet={"type": "image-grid"}),
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                composition.request_values(dict(good, **changed))


if __name__ == "__main__":
    unittest.main()
