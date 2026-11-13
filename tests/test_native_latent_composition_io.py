"""The standalone composition transport reads no model and keeps native provenance."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import safetensors.torch
import torch

from ruinedfooocus.features.generation import native_tensor_io as native
from ruinedfooocus.features.generation import native_tensor_context as context
import test_native_tensor_context as fixtures
from test_qwen_image21_conditioning import native_class


class CompositionTransportTests(unittest.TestCase):
    def setUp(self):
        fixtures.SavedNativeWorkflowTests.setUp(self)

    def sources(self, *, family="SD15", channels=4, scale=8):
        data = fixtures.SavedNativeWorkflowTests.capture(
            self, width=scale * 3, height=scale * 2, native_export_latent=True
        )
        outputs = []
        for value in (0.0, 0.75):
            outputs.append(
                native.export_tensors(
                    self.owner,
                    data,
                    {"samples": torch.full((1, channels, 2, 3), value, dtype=torch.float16)},
                    None,
                    None,
                )[0]
            )
        return [Path(item.path).read_bytes() for item in outputs]

    def test_owned_two_file_capture_native_math_and_save_without_inventory_or_model(self):
        first, second = self.sources()
        self.models.get_file.reset_mock()
        with patch(
            "ruinedfooocus.features.models.model_architecture.inspect_model_file",
            side_effect=AssertionError("model inspection reached"),
        ):
            data = native.capture_composition_inputs(first, second, self.manager)
            primary, source = data[native.KEY].inputs
            destination = native.load_composition_latent(data, primary)
            overlay = native.load_composition_latent(data, source)
        self.models.get_file.assert_not_called()
        self.assertEqual(primary.path.parent, source.path.parent)
        self.assertEqual(primary.path.read_bytes(), first)
        self.assertEqual(source.path.read_bytes(), second)
        self.assertEqual(primary.dtype, "F16")
        self.assertEqual(destination["samples"].dtype, torch.float32)
        node = native_class("nodes.py", "LatentComposite", {"torch": torch}, methods=("composite",))
        result = node().composite(destination, overlay, 0, 0)[0]
        artifact = native.save_composed_latent(
            data,
            primary,
            result,
            composition={
                "native_node": "LatentComposite",
                "x_pixels": 0,
                "y_pixels": 0,
                "codec_scale": 8,
            },
        )
        raw = Path(artifact.path).read_bytes()
        values, saved = context.header(raw, "latent", "SD15")
        self.assertEqual(artifact.kind, "latent-composite")
        self.assertEqual(values["latent_tensor"]["dtype"], "F32")
        self.assertEqual(saved["layout"]["composition"]["original_dtype"], "F16")
        self.assertNotIn(str(self.root), str(saved))
        np.testing.assert_array_equal(
            safetensors.torch.load(raw)["latent_tensor"].numpy(), result["samples"].numpy()
        )
        self.assertEqual(data[native.EXPORT_KEY], [artifact])
        # The derived output remains locally receipted and can be composed again.
        repeated = native.capture_composition_inputs(raw, raw, self.manager)
        self.assertEqual(repeated[native.KEY].inputs[0].dtype, "F32")

    def test_codec_sixteen_packed_rank_four_is_accepted_without_axis_conversion(self):
        self.owner.xl_base_patched.unet.model.model_config = type("Flux2", (), {})()
        self.owner.xl_base_patched.vae.latent_channels = 64
        self.owner.xl_base_patched.vae.spacial_compression_decode = lambda: 16
        with patch(
            "ruinedfooocus.features.models.model_architecture.inspect_model_file",
            return_value={"native_class": "Flux2"},
        ):
            first, second = self.sources(family="Flux2", channels=64, scale=16)
        data = native.capture_composition_inputs(first, second, self.manager)
        primary, source = data[native.KEY].inputs
        native.assert_composable(primary, source)
        self.assertEqual(primary.shape, (1, 64, 2, 3))
        self.assertEqual(dict(primary.context)["layout"]["scale"], 16)

    def test_external_files_mismatched_dtype_and_unowned_inputs_are_refused(self):
        from test_native_tensor_io import latent_file

        first, second = self.sources()
        with self.assertRaisesRegex(ValueError, "locally receipted"):
            native.capture_composition_inputs(latent_file(), second, self.manager)
        data = native.capture_composition_inputs(first, second, self.manager)
        primary, source = data[native.KEY].inputs
        with self.assertRaisesRegex(ValueError, "dtype"):
            native.assert_composable(primary, replace(source, dtype="BF16"))
        with self.assertRaisesRegex(ValueError, "owned"):
            native.load_composition_latent({}, primary)
        with self.assertRaises(ValueError):
            native.capture_composition_inputs("C:/other-owner/private.latent", second, self.manager)

    def test_native_save_refuses_reshape_cast_or_executable_metadata_before_write(self):
        data = native.capture_composition_inputs(*self.sources(), self.manager)
        primary = data[native.KEY].inputs[0]
        value = native.load_composition_latent(data, primary)
        for sample in (value["samples"].unsqueeze(2), value["samples"].half()):
            with patch.object(
                self.modules["nodes"].SaveLatent,
                "save",
                side_effect=AssertionError("native save reached"),
            ) as save:
                with self.assertRaisesRegex(ValueError, "before saving"):
                    native.save_composed_latent(
                        data,
                        primary,
                        {"samples": sample},
                        composition={"native_node": "LatentComposite"},
                    )
            save.assert_not_called()
        with self.assertRaisesRegex(ValueError, "JSON primitives"):
            native.save_composed_latent(
                data,
                primary,
                value,
                composition={"native_node": "LatentComposite", "hook": object()},
            )


if __name__ == "__main__":
    unittest.main()
