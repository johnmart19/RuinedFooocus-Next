"""Focused Torch Cascade vision admission, using the existing native CPU fixture."""

from concurrent.futures import CancelledError
from pathlib import Path
import unittest
from unittest.mock import patch

from ruinedfooocus.features.generation import cascade_guidance as guide
import test_cascade_guidance as cascade_fixtures


class NativeTorchReferenceCandidateTests(unittest.TestCase):
    def setUp(self):
        # Compose only the existing setup; do not inherit or collect its other
        # workflow tests, whose independent contracts have their own receipts.
        self.fixture = cascade_fixtures.CascadeGuidanceTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def test_torch_vision_is_an_explicit_unknown_choice_without_browsing_weights(self):
        from ruinedfooocus.core.model_formats import NATIVE_TORCH_SUFFIXES
        from ruinedfooocus.features.generation.ui_cascade_guidance import component_choices

        fixture = self.fixture
        for suffix in NATIVE_TORCH_SUFFIXES:
            path = fixture.root / "clip_vision" / ("unknown" + suffix.upper())
            path.write_bytes(b"runtime validation required")
            with patch.object(Path, "open", side_effect=AssertionError("picker read weights")):
                self.assertTrue(guide.compatible_file("vision", path))
                self.assertFalse(guide.compatible_file("control", path))
        choices = component_choices(fixture.manager, "vision", "unknown.PT")
        for suffix in NATIVE_TORCH_SUFFIXES:
            name = "unknown" + suffix.upper()
            self.assertIn((name + " · Compatibility unknown", name), choices)
        data = fixture.captured(cascade_super_resolution=False, cascade_vision_model="unknown.PT")
        self.assertEqual(data[guide.KEY].vision_model, "unknown.PT")
        fixture.vision_load.assert_not_called()
        fixture.vision_load.side_effect = CancelledError()
        with self.assertRaises(CancelledError):
            guide.apply_guidance(
                fixture.owner, [["x", {}]], [["", {}]], data, fixture.manager, None, lambda: None
            )
        (fixture.root / "clip_vision/unknown.PT").write_bytes(b"replaced")
        with self.assertRaisesRegex(ValueError, "changed"):
            guide.apply_guidance(
                fixture.owner, [["x", {}]], [["", {}]], data, fixture.manager, None, lambda: None
            )


if __name__ == "__main__":
    unittest.main()
