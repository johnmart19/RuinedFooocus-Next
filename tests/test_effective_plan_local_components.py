"""Explicit local component plans retain file identity without acquiring a publisher file."""

from dataclasses import FrozenInstanceError
import json
import unittest
from unittest.mock import Mock

from ruinedfooocus.features.models import effective_image_plan as plans
from ruinedfooocus.features.models.model_identity import file_signature
import test_effective_image_plan as plan_fixtures
import test_effective_plan_transport as transport_fixtures


class LocalPlanComponentTests(unittest.TestCase):
    def setUp(self):
        self.base = transport_fixtures.EffectivePlanTransportTests()
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.base.register()
        self.fixture, self.paths = self.base.fixture, self.base.paths
        self.local = self.base.root / "clip/clip_l.safetensors"
        self.local.parent.mkdir(parents=True)
        plan_fixtures.header(self.local)
        self.fixture.values = {"clip_l": self.local.name}

    def test_explicit_local_request_has_no_download_or_unrelated_publisher_integrity(self):
        self.paths.preparation_request = Mock(side_effect=AssertionError("local cannot acquire"))
        captured = self.fixture.capture()
        row = self.base.row(captured)
        self.assertIsNone(row.download_key)
        self.assertIsNone(row.catalogue_identity)
        self.assertEqual(row.source, self.base.url)
        self.assertEqual(row.acquisition.destination, self.local)
        self.assertEqual(
            (row.acquisition.url, row.acquisition.expected_size, row.acquisition.sha256),
            ("", None, None),
        )
        self.assertEqual(row.signature, tuple(file_signature(self.local)))
        self.paths.preparation_request.assert_not_called()
        with self.assertRaises(FrozenInstanceError):
            row.acquisition.url = self.base.pin
        # Catalogue provenance is independent from the explicitly selected file.
        self.paths.DOWNLOADABLE_FILES[self.base.key]["url"] += "?changed"
        captured[plans.KEY].verify_request(captured)

    def test_replaced_or_removed_local_file_cannot_reuse_its_captured_signature(self):
        for replace_file in (True, False):
            with self.subTest(replace=replace_file):
                plan_fixtures.header(self.local)
                captured = self.fixture.capture()
                if replace_file:
                    plan_fixtures.header(self.local, "BF16")
                else:
                    self.local.unlink()
                with self.assertRaisesRegex(ValueError, "component clip_l changed after capture"):
                    captured[plans.KEY].verify_files()

    def test_local_request_and_separate_provenance_serialize_in_plan_metadata(self):
        captured = self.fixture.capture()
        row = next(
            row
            for row in json.loads(json.dumps(captured[plans.KEY].metadata()))["components"]
            if row["key"] == "clip_l"
        )
        self.assertEqual(row["source"], self.base.url)
        self.assertIsNone(row["download_key"])
        self.assertEqual(
            row["acquisition"],
            dict(url="", destination=str(self.local), expected_size=None, sha256=None),
        )
        self.assertEqual(row["signature"], file_signature(self.local))


if __name__ == "__main__":
    unittest.main()
