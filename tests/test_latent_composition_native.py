"""Execute the actual pinned latent bodies on bounded CPU tensors."""

import unittest

import numpy as np
import torch

from ruinedfooocus.features.image_processing import latent_composition_native as native


class NativeLatentCompositionTests(unittest.TestCase):
    def setUp(self):
        self.operations = native.operations(native.source_signatures())
        self.base = {"samples": torch.zeros((1, 4, 4, 4), device="cpu"), "batch_index": [7]}
        self.source = {"samples": torch.ones((1, 4, 4, 4), device="cpu")}

    def test_plain_uses_native_pixel_offsets_and_preserves_destination_canvas(self):
        result = self.operations.plain(None, self.base, self.source, 8, 16, feather=0)[0]
        expected = torch.zeros_like(self.base["samples"])
        expected[:, :, 2:, 1:] = 1
        np.testing.assert_array_equal(result["samples"].numpy(), expected.numpy())
        self.assertEqual(result["batch_index"], [7])
        self.assertEqual(result["samples"].shape, self.base["samples"].shape)
        self.assertEqual(torch.count_nonzero(self.base["samples"]).item(), 0)
        self.assertTrue(torch.all(self.source["samples"] == 1).item())

    def test_plain_feather_retains_actual_native_edge_ramp(self):
        result = self.operations.plain(None, self.base, self.source, 8, 8, feather=16)[0]
        self.assertAlmostEqual(result["samples"][0, 0, 1, 1].item(), 0.25)
        self.assertAlmostEqual(result["samples"][0, 0, 2, 2].item(), 1.0)
        self.assertEqual(result["samples"][0, 0, 0, 0].item(), 0)

    def test_masked_keeps_zero_one_and_fractional_selection_without_changing_inputs(self):
        mask = torch.tensor([[[0, 0.25, 0.5, 1]] * 4], device="cpu")
        result = self.operations.masked(None, self.base, self.source, 0, 0, False, mask)[0]
        expected = mask.unsqueeze(1).expand_as(self.base["samples"])
        np.testing.assert_array_equal(result["samples"].numpy(), expected.numpy())
        self.assertEqual(result["batch_index"], [7])
        self.assertEqual(torch.count_nonzero(self.base["samples"]).item(), 0)
        np.testing.assert_array_equal(mask[0, 0].numpy(), np.array([0, 0.25, 0.5, 1]))

    def test_masked_full_selection_is_native_plain_replacement(self):
        plain = self.operations.plain(None, self.base, self.source, 16, 8, feather=0)[0]
        masked = self.operations.masked(None, self.base, self.source, 16, 8, False, None)[0]
        np.testing.assert_array_equal(plain["samples"].numpy(), masked["samples"].numpy())

    def test_explicit_codec16_grid_conversion_preserves_pixel_meaning(self):
        # A pixel offset32 at scale16 is two grid cells, passed as native offset16.
        node_offset = (32 // 16) * 8
        result = self.operations.masked(None, self.base, self.source, node_offset, 0, False)[0]
        self.assertEqual(torch.count_nonzero(result["samples"][:, :, :, :2]).item(), 0)
        self.assertTrue(torch.all(result["samples"][:, :, :, 2:] == 1).item())

    def test_changed_sources_are_rejected_before_creating_operation_namespaces(self):
        with self.assertRaisesRegex(ValueError, "changed"):
            native.operations(())


if __name__ == "__main__":
    unittest.main()
