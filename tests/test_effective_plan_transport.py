"""Submission plans bind actual immutable transport without changing passive inventory."""

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ruinedfooocus.core.path import PathManager
from ruinedfooocus.features.gallery.editor_session import _mapping_json
from ruinedfooocus.features.models import component_profiles, effective_image_plan as plans
from ruinedfooocus.features.models.model_registry import ModelRegistry, download_identity
import test_effective_image_plan as fixtures


class EffectivePlanTransportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EffectiveImagePlanTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.paths = PathManager.__new__(PathManager)
        self.paths.paths = {
            "path_clip": str(self.root / "clip"),
            "path_vae": str(self.root / "vae"),
            "path_loras": [str(self.root)],
        }
        self.paths.model_paths = {"cache_path": self.root / "cache"}
        self.paths.DOWNLOADABLE_FILES = {}
        self.fixture.paths, self.fixture.values = self.paths, {}
        self.key = "clip/clip_l.safetensors"
        self.url = "https://huggingface.co/Example/image-components/resolve/main/text_encoders/clip_l.safetensors"
        self.pin = self.url.replace("/main/", "/" + "1" * 40 + "/")
        self.sha = hashlib.sha256(b"fixture component payload").hexdigest()
        self.registry = None
        self.enterContext(
            patch(
                "ruinedfooocus.features.models.model_registry.load_registry",
                side_effect=lambda: self.registry,
            )
        )

    def register(self, *, pinned=True, annotated=True):
        source = dict(
            path="path_clip", filename="clip_l.safetensors", url=self.url, size=25, sha256=self.sha
        )
        self.paths.DOWNLOADABLE_FILES[self.key] = source
        rows = []
        if annotated:
            rows.append(
                dict(
                    id="existing:" + self.key,
                    catalogue_key=self.key,
                    state="existing",
                    profile_id="components",
                    role="text_conditioning_encoder",
                    chat_weight=False,
                    download_identity=download_identity(source),
                    artifact=dict(
                        bytes=25, sha256=self.sha, pinned_url=self.pin if pinned else self.url
                    ),
                    companions=dict(text=[], image=[]),
                )
            )
        self.registry = ModelRegistry(
            dict(schema_version=1, profiles={"components": {}}, artifacts=rows, benchmarks=[])
        )

    def row(self, data):
        return next(row for row in data[plans.KEY].components if row.key == "clip_l")

    def test_pinned_transport_is_actual_producer_request_and_canonical_identity_is_retained(self):
        self.register()
        captured = self.fixture.capture()
        row = self.row(captured)
        request = self.paths.preparation_request(self.key, destination=Path(row.path))
        self.assertEqual(row.acquisition, request)
        self.assertEqual((row.source, row.acquisition.url), (self.url, self.pin))
        self.assertEqual((row.acquisition.expected_size, row.acquisition.sha256), (25, self.sha))
        self.assertEqual(
            row.catalogue_identity, download_identity(self.paths.DOWNLOADABLE_FILES[self.key])
        )
        self.assertIs(captured[plans.COMPONENTS_KEY], captured[plans.KEY].components)
        with self.assertRaises(FrozenInstanceError):
            row.acquisition.url = self.url

    def test_legacy_registered_transport_keeps_its_original_url_and_integrity(self):
        self.register(annotated=False)
        row = self.row(self.fixture.capture())
        self.assertEqual((row.source, row.acquisition.url), (self.url, self.url))
        self.assertEqual((row.acquisition.expected_size, row.acquisition.sha256), (25, self.sha))

    def test_bound_acquisition_serializes_in_existing_plan_metadata(self):
        self.register()
        captured = self.fixture.capture()
        metadata = json.loads(json.dumps(captured[plans.KEY].metadata()))
        row = next(item for item in metadata["components"] if item["key"] == "clip_l")
        self.assertEqual(
            row["acquisition"]["destination"], str(self.root / "clip/clip_l.safetensors")
        )
        self.assertEqual(row["acquisition"]["url"], self.pin)
        self.assertEqual(row["source"], self.url)
        ticket_json = _mapping_json(captured[plans.KEY].metadata(), "effective plan")
        self.assertEqual(json.loads(ticket_json), metadata)
        png = PngInfo()
        png.add_text("parameters", ticket_json)
        output = self.root / "plan-metadata.png"
        Image.new("RGBA", (2, 3), (12, 23, 34, 45)).save(output, pnginfo=png)
        with Image.open(output) as image:
            self.assertEqual(json.loads(image.info["parameters"]), metadata)

    def test_same_submission_reuses_the_once_bound_request(self):
        self.register()
        producer = self.paths.preparation_request
        self.paths.preparation_request = Mock(wraps=producer)
        captured = self.fixture.capture()
        again = plans.capture_effective_image_plan(
            captured, self.fixture.models, self.fixture.values, self.paths
        )
        self.assertIs(again[plans.KEY], captured[plans.KEY])
        self.paths.preparation_request.assert_called_once()

    def test_passive_component_inventory_does_not_bind_or_call_preparation(self):
        self.register()
        self.paths.preparation_request = Mock(side_effect=AssertionError("passive inventory"))
        state = component_profiles.component_inventory(
            self.fixture.path, self.fixture.family, {}, self.paths
        )
        row = next(item for item in state["rows"] if item["key"] == "clip_l")
        self.assertEqual(row["source"], self.url)
        self.paths.preparation_request.assert_not_called()

    def test_source_drift_between_inventory_and_plan_binding_is_rejected(self):
        self.register()
        snapshot = component_profiles.snapshot_request(
            self.fixture.request, self.fixture.models, {}, self.paths
        )
        self.paths.DOWNLOADABLE_FILES[self.key]["url"] += "?changed"
        with self.assertRaisesRegex(ValueError, "after inventory capture"):
            plans.capture_effective_image_plan(snapshot, self.fixture.models, {}, self.paths)


if __name__ == "__main__":
    unittest.main()
