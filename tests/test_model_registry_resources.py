"""Resource migration preserves tuples and avoids history reads on hot paths."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models.model_registry import (
    RegistryError,
    load_registry,
    registry_document,
    metadata_digest,
)
from tools.split_model_registry import baseline_manifest, document_digest, split, verify


class RegistryResourceTests(unittest.TestCase):
    def test_split_can_preserve_existing_custom_artifact_shards(self):
        document = registry_document()
        identity = document["artifacts"][0]["id"]
        profile = document["artifacts"][0]["profile_id"]
        with tempfile.TemporaryDirectory() as folder:
            index = split(
                document,
                folder,
                artifact_paths={identity: "artifacts/custom.json"},
                profile_paths={profile: "families/custom.json"},
            )
            self.assertEqual(index["artifacts"][identity]["path"], "artifacts/custom.json")
            self.assertEqual(index["profiles"][profile]["path"], "families/custom.json")
            self.assertEqual(document_digest(registry_document(folder)), document_digest(document))
            self.assertEqual(verify(folder)["artifacts"], len(document["artifacts"]))

    def test_invalid_existing_artifact_shard_mappings_write_nothing(self):
        document = registry_document()
        identity = document["artifacts"][0]["id"]
        invalid = [
            {identity: "../outside.json"},
            {identity: "artifacts/../outside.json"},
            {identity: "families/custom.json"},
            {identity: "artifacts/nested/custom.json"},
            {"existing:unknown": "artifacts/custom.json"},
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for paths in invalid:
                with (
                    self.subTest(paths=paths),
                    self.assertRaisesRegex(ValueError, "Invalid existing artifact shard"),
                ):
                    split(document, root, artifact_paths=paths)
                self.assertEqual(list(root.iterdir()), [])
            profile = document["artifacts"][0]["profile_id"]
            for paths in ({profile: "../outside.json"}, {"unknown": "families/custom.json"}):
                with (
                    self.subTest(paths=paths),
                    self.assertRaisesRegex(ValueError, "Invalid existing profile shard"),
                ):
                    split(document, root, profile_paths=paths)
                self.assertEqual(list(root.iterdir()), [])

    def test_split_roundtrips_multiple_profiles_in_a_preserved_shared_shard(self):
        document = registry_document()
        identities = list(document["profiles"])[:2]
        with tempfile.TemporaryDirectory() as folder:
            index = split(
                document,
                folder,
                profile_paths={identity: "families/shared.json" for identity in identities},
            )
            shared = json.loads((Path(folder) / "families/shared.json").read_text())
            self.assertEqual(set(shared["profiles"]), set(identities))
            self.assertTrue(
                all(
                    index["profiles"][identity]["path"] == "families/shared.json"
                    for identity in identities
                )
            )
            self.assertEqual(document_digest(registry_document(folder)), document_digest(document))
            self.assertEqual(verify(folder)["profiles"], len(document["profiles"]))

    def test_new_resources_cannot_hide_changes_to_original_migration_annotations(self):
        original = registry_document()
        manifest = baseline_manifest(original)
        with tempfile.TemporaryDirectory() as folder:
            document = copy.deepcopy(original)
            document["artifacts"][0]["label"] = "changed original annotation"
            split(document, folder, migration_baseline=manifest)
            with self.assertRaisesRegex(ValueError, "Original migration annotations"):
                verify(folder)

    def test_unsafe_role_cannot_write_outside_split_destination(self):
        original = registry_document()
        original["artifacts"][0]["role"] = "../../outside"
        original["artifacts"][0]["chat_weight"] = False
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, "Unsafe model role"):
                split(original, root / "split")
            self.assertEqual(list(root.iterdir()), [])

    def test_valid_digest_cannot_bypass_lazy_semantic_validation(self):
        original = registry_document()
        mutations = [
            lambda row: row.update(companions=None),
            lambda row: row.update(companions={"text": "not-a-list"}),
            lambda row: row.update(companions={"text": ["missing"]}),
            lambda row: row.update(companions={"image": [row["id"]]}),
            lambda row: row.update(chat_weight="true"),
            lambda row: row.update(chat_weight=True, role="latent_autoencoder"),
            lambda row: row.update(download_identity="bad"),
            lambda row: row["artifact"].update(sha256=3),
            lambda row: row["artifact"].update(bytes=True),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                index = split(original, root)
                identity, descriptor = next(iter(index["artifacts"].items()))
                path = root / descriptor["path"]
                document = json.loads(path.read_text(encoding="utf-8"))
                mutate(document["artifacts"][identity])
                path.write_text(json.dumps(document), encoding="utf-8")
                index["shards"][descriptor["path"]]["sha256"] = metadata_digest(document)
                (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
                with self.assertRaises(RegistryError):
                    load_registry(root).artifact(identity)

    def test_missing_indexed_profile_key_and_wrong_qualification_profile_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index = split(registry_document(), root)
            identity, descriptor = next(iter(index["profiles"].items()))
            path = root / descriptor["path"]
            document = json.loads(path.read_text(encoding="utf-8"))
            document["profiles"][identity].pop("label")
            path.write_text(json.dumps(document), encoding="utf-8")
            index["shards"][descriptor["path"]]["sha256"] = metadata_digest(document)
            (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(RegistryError, "profile fields"):
                load_registry(root).profile(identity)
            path = root / index["qualification"][identity]
            document = json.loads(path.read_text(encoding="utf-8"))
            document["profile_id"] = "wrong"
            path.write_text(json.dumps(document), encoding="utf-8")
            index["shards"][index["qualification"][identity]]["sha256"] = metadata_digest(document)
            (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(RegistryError, "qualification profile"):
                load_registry(root).benchmarks(identity)

    def test_split_materialization_matches_all_original_annotations_and_tuples(self):
        report = verify(resource_path("model_metadata"))
        historical = report["historical_migration_baseline"]
        self.assertEqual(
            (historical["profiles"], historical["artifacts"], historical["benchmarks"]),
            (167, 324, 135),
        )
        self.assertEqual(
            historical["source_document_sha256"],
            "3c9953484f4571596323629082b21be525f5500f4b9493f8ba9106edc92347f5",
        )
        self.assertEqual(
            historical["download_identities_sha256"],
            "39b1460f216f0e29a361d34c31f3168756e50f3fae82824f1b471c2a5796cdd4",
        )
        report = report["migration_baseline"]
        self.assertEqual(
            (report["profiles"], report["artifacts"], report["benchmarks"]), (142, 282, 132)
        )
        self.assertEqual(
            report["source_document_sha256"],
            "78fe28c0da9ac3592e74f6259db1c217e06de9e419836ef56798b6ddf9bd3e28",
        )
        self.assertEqual(
            report["download_identities_sha256"],
            "c2d454cd743cf7f338506a70d4d4230fdcde0b3324ebd73c4c571730d5f77455",
        )

    def test_scope_retirement_cannot_hide_a_changed_retained_annotation_or_unknown_key(self):
        index = json.loads(
            resource_path("model_metadata", "index.json").read_text(encoding="utf-8")
        )
        baseline = index["migration_baseline"]
        with tempfile.TemporaryDirectory() as folder:
            document = registry_document()
            document["profiles"]["generation:wan21_vae"]["title"] = "changed retained image codec"
            split(document, folder, migration_baseline=baseline)
            with self.assertRaisesRegex(ValueError, "Original migration annotations"):
                verify(folder)
            invalid = copy.deepcopy(baseline)
            invalid["scope_retirement"]["profiles"].append("not-an-original-profile")
            split(registry_document(), folder, migration_baseline=invalid)
            with self.assertRaisesRegex(ValueError, "Invalid migration scope retirement"):
                verify(folder)

    def test_index_artifact_profile_reads_never_open_qualification_until_requested(self):
        registry = load_registry()
        paths = []
        read = registry._shard
        with patch.object(
            registry, "_shard", side_effect=lambda name: (paths.append(name), read(name))[1]
        ):
            row = registry.artifact("existing:llm/Qwen3-1.7B-Q8_0.gguf")
            profile = registry.profile(row["profile_id"])
            self.assertIn("label", profile)
            self.assertTrue(profile["label"])
            self.assertFalse(any(path.startswith("qualification/") for path in paths))
            self.assertEqual(profile["qualification"]["status"], "candidate_not_default")
            self.assertTrue(any(path.startswith("qualification/") for path in paths))

    def test_new_resource_root_roundtrips_and_changed_shards_fail_explicitly(self):
        document = registry_document()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index = split(document, root)
            self.assertEqual(document_digest(registry_document(root)), document_digest(document))
            registry = load_registry(root)
            entry = next(iter(index["artifacts"].values()))
            path = root / entry["path"]
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RegistryError, "Changed model metadata shard"):
                registry.entries()

    def test_duplicate_index_keys_unknown_schema_and_path_escape_are_rejected(self):
        original = json.loads(
            resource_path("model_metadata", "index.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "index.json"
            mutations = [
                lambda doc: doc.update(schema_version=99),
                lambda doc: doc["shards"].update({"../outside.json": {"sha256": "a" * 64}}),
                lambda doc: doc["artifacts"].update(
                    {"existing:duplicate": next(iter(doc["artifacts"].values()))}
                ),
            ]
            for mutate in mutations:
                document = copy.deepcopy(original)
                mutate(document)
                path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaises(RegistryError):
                    load_registry(path)
            path.write_text('{"schema_version":2,"schema_version":2}', encoding="utf-8")
            with self.assertRaises(RegistryError):
                load_registry(path)


if __name__ == "__main__":
    unittest.main()
