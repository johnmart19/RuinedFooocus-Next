"""Download evidence follows the public indexed registry, without its history."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.features.models.model_registry import download_identity
from tools.audit_download_links import inventory
from tools.split_model_registry import split


class RegistryDownloadAuditTests(unittest.TestCase):
    def fixture(self, root):
        catalogue = root / "ruinedfooocus/resources/model_catalogue"
        catalogue.mkdir(parents=True)
        downloads = {
            "weight": {
                "url": "https://example.org/pinned/model.gguf",
                "path": "path_llm",
                "filename": "model.gguf",
            },
            "projector": {
                "url": "https://example.org/pinned/mmproj.gguf",
                "path": "path_llm",
                "filename": "mmproj.gguf",
            },
        }
        (catalogue / "llm.json").write_text(json.dumps(downloads), encoding="utf-8")
        entries = []
        for key, download in downloads.items():
            entries.append(
                {
                    "id": f"existing:{key}",
                    "catalogue_key": key,
                    "state": "existing",
                    "profile_id": "family",
                    "role": "language_model" if key == "weight" else "vision_projector",
                    "chat_weight": key == "weight",
                    "download_identity": download_identity(download),
                    "artifact": {"bytes": 123, "sha256": "a" * 64},
                    "behavioral_variant": "instruction",
                    "native_family": "fixture",
                    "runtime": {"comfy": {"pin": "b" * 40}},
                    "companions": {
                        "text": [],
                        "image": ["existing:projector"] if key == "weight" else [],
                    },
                }
            )
        proposed = deepcopy(entries[-1])
        proposed.update(id="proposed:future", catalogue_key="future", state="proposed")
        proposed["proposed_download"] = {**downloads["projector"], "filename": "future.gguf"}
        proposed["download_identity"] = download_identity(proposed["proposed_download"])
        entries.append(proposed)
        document = {
            "schema_version": 1,
            "profiles": {
                "family": {
                    "label": "Fixture",
                    "sources": ["https://example.org/source-card"],
                    "provenance": {"source_revision": "c" * 40},
                    "dependencies": ["required projector"],
                    "qualification": {"must_not_read": True},
                }
            },
            "artifacts": entries,
            "benchmarks": [],
        }
        split(document, root / "ruinedfooocus/resources/model_metadata")
        return catalogue

    def test_exact_tuple_variants_companions_sources_and_proposals_without_history_or_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            original_read = Path.read_bytes

            def guarded_read(path):
                if "qualification" in path.parts:
                    raise AssertionError(
                        "Link inventory must not materialize benchmark/qualification history"
                    )
                return original_read(path)

            with (
                patch.object(Path, "read_bytes", guarded_read),
                patch(
                    "tools.audit_download_links.build_opener",
                    side_effect=AssertionError("No network"),
                ),
            ):
                rows, sources = inventory(root)
            weight = next(row for row in rows if row["key"] == "weight")
            self.assertTrue(weight["registry"]["download_tuple_matches"])
            self.assertEqual(weight["expected_sha256"], "a" * 64)
            self.assertEqual(weight["registry"]["behavioral_variant"], "instruction")
            self.assertEqual(weight["registry"]["companions"]["image"], ["existing:projector"])
            profile = next(row for row in rows if row["key"] == "family")
            self.assertEqual(profile["profile_evidence"]["provenance"]["source_revision"], "c" * 40)
            reference = next(
                row for row in rows if row.get("url") == "https://example.org/source-card"
            )
            self.assertEqual(reference["status"], "reference_only")
            proposal = next(row for row in rows if row["key"] == "proposed:future")
            self.assertEqual(proposal["status"], "unresolved")
            self.assertFalse(proposal["registry"]["download_tuple_matches"])
            self.assertTrue(any("/artifacts/" in path for path in sources))
            self.assertFalse(any("/qualification/" in path for path in sources))

    def test_changed_tuple_keeps_download_and_records_an_unresolved_annotation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalogue = self.fixture(root)
            path = catalogue / "llm.json"
            data = json.loads(path.read_text())
            data["weight"]["url"] = "https://example.org/changed/model.gguf"
            path.write_text(json.dumps(data))
            rows, _ = inventory(root)
            weight = next(row for row in rows if row["key"] == "weight")
            self.assertEqual(weight["url"], data["weight"]["url"])
            self.assertNotIn("registry", weight)
            error = next(row for row in rows if row["key"] == "registry-validation")
            self.assertEqual(error["status"], "unresolved")
            self.assertIn("no longer matches", error["reason"])


if __name__ == "__main__":
    unittest.main()
