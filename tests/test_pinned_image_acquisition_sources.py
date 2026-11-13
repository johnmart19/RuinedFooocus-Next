"""Production requests retain immutable publisher URLs and registry integrity."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from ruinedfooocus.core.path import PathManager
from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models.model_registry import load_registry, RegistryError


class PinnedImageAcquisitionSources(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.manager = PathManager.__new__(PathManager)
        self.manager.paths = {
            "path_clip": str(self.root / "text_encoders"),
            "path_detection": str(self.root / "detection"),
        }
        self.manager.DOWNLOADABLE_FILES = {}
        for name in ("clip.json", "native_mediapipe_face.json"):
            self.manager.DOWNLOADABLE_FILES.update(
                json.loads(resource_path("model_catalogue", name).read_text(encoding="utf-8"))
            )

    def test_qwen3vl_8b_production_request_uses_pinned_registry_source_without_driver_enrichment(
        self,
    ):
        key = "clip/qwen3vl_8b_fp8_scaled.safetensors"
        source = self.manager.DOWNLOADABLE_FILES[key]
        artifact = load_registry().annotations_for(key, source)["artifact"]
        request = self.manager.preparation_request(key)
        self.assertEqual(request.url, artifact["pinned_url"])
        self.assertIn("/resolve/8606e7df2f5389282617b8b534db290e1f5d7811/", request.url)
        self.assertEqual(request.expected_size, 10588637512)
        self.assertEqual(
            request.sha256, "4ba424cf62e51392e4d1a39933e803706f4e823c1065f36aaf149c6453f66bcd"
        )
        self.assertEqual(request.destination, self.root / "text_encoders" / Path(key).name)

    def test_face_production_request_uses_the_same_immutable_first_use_source(self):
        key = "detection/mediapipe_face_fp32.safetensors"
        request = self.manager.preparation_request(key)
        self.assertIn("/resolve/f787efc1ad99492ec7d5b6564c0dd6c48d84ee49/", request.url)
        self.assertEqual(
            (request.expected_size, request.sha256),
            (5423900, "a98c4806081d40eba35102a0f6dc0000c2e1388b72cf24e691703d0605bd888a"),
        )
        self.assertEqual(request.destination, self.root / key)

    def test_source_drift_cannot_keep_the_old_immutable_registry_annotation(self):
        key = "clip/qwen3vl_8b_fp8_scaled.safetensors"
        source = self.manager.DOWNLOADABLE_FILES[key]
        source["url"] += "?changed"
        with self.assertRaisesRegex(RegistryError, "no longer matches"):
            self.manager.preparation_request(key)

    def test_every_current_annotated_source_keeps_its_compatible_production_request(self):
        catalogue = {}
        for path in resource_path("model_catalogue").glob("*.json"):
            catalogue.update(json.loads(path.read_text(encoding="utf-8")))
        self.manager.DOWNLOADABLE_FILES = catalogue
        registry = load_registry()
        changed = 0
        for key, source in catalogue.items():
            annotation = registry.annotations_for(key, source)
            if annotation is None:
                continue
            with self.subTest(key=key):
                request = self.manager.preparation_request(
                    key, destination=self.root / "captured" / source["filename"]
                )
                pin = annotation["artifact"].get("pinned_url")
                self.assertEqual(request.url, pin or source["url"])
                changed += request.url != source["url"]
        self.assertGreater(changed, 0)

    def test_distinct_invalid_or_mismatched_pins_fail_before_a_request_is_returned(self):
        key = "clip/qwen3vl_8b_fp8_scaled.safetensors"
        pinned = "https://huggingface.co/Comfy-Org/Boogu-Image/resolve/8606e7df2f5389282617b8b534db290e1f5d7811/text_encoders/qwen3vl_8b_fp8_scaled.safetensors"
        malformed = (
            pinned.replace("https:", "http:"),
            pinned.replace("huggingface.co", "huggingface.co.evil.invalid"),
            pinned.replace("Comfy-Org/Boogu-Image", "other/Boogu-Image"),
            pinned.replace("text_encoders/", "other_directory/"),
            pinned.replace("/8606e7df2f5389282617b8b534db290e1f5d7811/", "/other-branch/"),
            pinned.replace("huggingface.co", "user@huggingface.co"),
            pinned + "#fragment",
            pinned + "?download=true",
            pinned + "\n",
            pinned.replace("text_encoders/", "text_encoders/%2e%2e/"),
            123,
        )
        for pin in malformed:
            with self.subTest(pin=pin):
                annotation = {"artifact": {"bytes": 10, "sha256": "a" * 64, "pinned_url": pin}}
                registry = NS(annotations_for=lambda *args: annotation)
                with patch(
                    "ruinedfooocus.features.models.model_registry.load_registry",
                    return_value=registry,
                ):
                    with self.assertRaisesRegex(
                        ValueError, "pinned publisher source is incompatible"
                    ):
                        self.manager.preparation_request(key)

    def test_distinct_pin_requires_integrity_but_absent_or_unchanged_pin_preserves_legacy(self):
        key = "clip/qwen3vl_8b_fp8_scaled.safetensors"
        original = self.manager.DOWNLOADABLE_FILES[key]["url"]
        pin = original.replace("/main/", "/8606e7df2f5389282617b8b534db290e1f5d7811/")
        for size, checksum in ((None, None), (10, None), (True, "a" * 64), (10, "bad")):
            annotation = {"artifact": {"bytes": size, "sha256": checksum, "pinned_url": pin}}
            registry = NS(annotations_for=lambda *args: annotation)
            with patch(
                "ruinedfooocus.features.models.model_registry.load_registry", return_value=registry
            ):
                with self.assertRaises(ValueError):
                    self.manager.preparation_request(key)
        for pin in (None, original):
            annotation = {"artifact": {"bytes": None, "sha256": None, "pinned_url": pin}}
            registry = NS(annotations_for=lambda *args: annotation)
            with patch(
                "ruinedfooocus.features.models.model_registry.load_registry", return_value=registry
            ):
                self.assertEqual(self.manager.preparation_request(key).url, original)


if __name__ == "__main__":
    unittest.main()
