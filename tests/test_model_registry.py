"""Registry annotations cannot mutate downloader identities or certify execution."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models.model_registry import (
    ModelRegistry,
    RegistryError,
    download_identity,
    load_registry,
    unique_payload_bytes,
    registry_document,
)

BIREFNET_KEY = "background_removal/birefnet.safetensors"


class ModelRegistryTests(unittest.TestCase):
    def test_warning_artwork_is_packaged_with_preview_recognition(self):
        from ruinedfooocus.core.paths import asset_path
        from ruinedfooocus.features.models.model_previews import is_warning

        for filename in ("warning.jpeg", "warning.png"):
            with self.subTest(filename=filename):
                self.assertTrue(asset_path(filename).is_file())
                self.assertTrue(is_warning(asset_path(filename)))

    def test_payload_totals_deduplicate_hashes_without_hiding_unknown_or_conflicting_sizes(self):
        first = {"id": "one", "artifact": {"sha256": "same", "bytes": 4}}
        alias = {"id": "two", "artifact": {"sha256": "same", "bytes": 4}}
        unknown = {"id": "three", "artifact": {"bytes": None}}
        self.assertEqual(unique_payload_bytes([first, alias]), 4)
        self.assertEqual(unique_payload_bytes([]), 0)
        self.assertIsNone(unique_payload_bytes([first, unknown]))
        alias["artifact"]["bytes"] = 8
        self.assertIsNone(unique_payload_bytes([first, alias]))

    def test_live_downloads_join_exactly_and_future_proposals_stay_separate(self):
        registry = load_registry()
        downloads = {}
        for path in resource_path("model_catalogue").glob("*.json"):
            for key, value in json.loads(path.read_text(encoding="utf-8")).items():
                self.assertNotIn(key, downloads)
                downloads[key] = value
        self.assertEqual({row["catalogue_key"] for row in registry.entries()}, set(downloads))
        for key, download in downloads.items():
            self.assertEqual(registry.annotations_for(key, download)["catalogue_key"], key)
        document = registry_document()
        candidate = next(row for row in document["artifacts"] if "Nemotron-3.5-Bolt" in row["id"])
        candidate.update(state="proposed", id="proposed:" + candidate["catalogue_key"])
        candidate["proposed_download"] = downloads[candidate["catalogue_key"]]
        staged = ModelRegistry(document)
        self.assertNotIn(
            candidate["catalogue_key"], {row["catalogue_key"] for row in staged.entries()}
        )
        self.assertIn(candidate["id"], {row["id"] for row in staged.entries(include_proposed=True)})

    def test_reviewed_additions_preserve_previous_downloads_and_exact_companion_pairs(self):
        registry = load_registry()
        baseline = {}
        migration = json.loads(
            resource_path("model_metadata", "index.json").read_text(encoding="utf-8")
        ).get("migration_baseline")
        original_keys = (
            {identity.partition(":")[2] for identity in migration["artifact_ids"]}
            if migration
            else None
        )
        for path in resource_path("model_catalogue").glob("*.json"):
            if path.name not in (
                "llm_reviewed.json",
                "native_image_components.json",
                "qwen_image21.json",
            ):
                baseline.update(
                    {
                        key: value
                        for key, value in json.loads(path.read_text(encoding="utf-8")).items()
                        if original_keys is None or key in original_keys
                    }
                )
        identities = {key: download_identity(value) for key, value in baseline.items()}
        # Exact candidates have their own identity tests; keep the
        # retained catalogue identity gate unchanged for every previous tuple.
        # Exact retired identities and the original digest remain in the
        # migration baseline; split_model_registry verifies its retained projection.
        identities.pop("llm/LFM2.5-1.2B-Instruct-Q8_0.gguf", None)
        identities.pop("llm/Qwen3-1.7B-Q8_0.gguf", None)
        identities.pop(BIREFNET_KEY, None)
        digest = hashlib.sha256(
            json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(digest, "7c0953a41451e6a9ce801778e10724e62ece99a38d28cedd15419b5d33532b5e")
        additions = json.loads(resource_path("model_catalogue", "llm_reviewed.json").read_text())
        self.assertFalse(set(additions) & set(baseline))
        weights = []
        for key, download in additions.items():
            row = registry.annotations_for(key, download)
            self.assertEqual(row["state"], "existing")
            self.assertNotIn("proposed_download", row)
            self.assertNotIn("/resolve/main/", download["url"])
            self.assertEqual(row["artifact"]["pinned_url"], download["url"])
            self.assertEqual(row["artifact"]["availability"], "payload_reachable_bounded")
            self.assertIsNotNone(row["artifact"]["sha256"])
            self.assertGreater(row["artifact"]["bytes"], 0)
            if row["chat_weight"]:
                weights.append(row)
                self.assertEqual(row["runtime"]["native_llama_cpp"]["chat"], "source_supported")
                paired = [registry.artifact(identity) for identity in row["companions"]["image"]]
                self.assertEqual(
                    [additions[item["catalogue_key"]]["filename"] for item in paired],
                    [download["mmproj"]] if download.get("mmproj") else [],
                )
            else:
                self.assertEqual(row["role"], "vision_projector")
                self.assertEqual(download["path"], "path_vision")
        self.assertEqual(len(weights), 25)
        self.assertEqual(len(additions) - len(weights), 19)

    def test_qwen_image21_companions_and_specialized_roles(self):
        registry = load_registry()
        catalogue = resource_path("model_catalogue", "qwen_image21.json")
        downloads = json.loads(catalogue.read_text(encoding="utf-8"))
        variants = {}
        for key, download in downloads.items():
            row = registry.annotations_for(key, download)
            self.assertFalse(row["chat_weight"])
            self.assertEqual(row["artifact"]["pinned_url"], download["url"])
            self.assertNotIn("/resolve/main/", download["url"])
            if row["role"] == "image_diffusion_checkpoint":
                variants[row["behavioral_variant"]] = row["format"]
                self.assertEqual(
                    row["companions"]["text"],
                    (
                        "existing:clip/qwen3vl_8b_int8_convrot.safetensors",
                        "existing:vae/qwen_image_2.1_vae_bf16.safetensors",
                    ),
                )
            elif row["role"] == "prompt_enhancer":
                self.assertEqual(download["path"], "path_clip")
                self.assertEqual(row["companions"]["text"], ())
            elif row["role"] == "control_conditioning":
                self.assertEqual(download["path"], "path_model_patches")
        self.assertEqual(variants, {"turbo": "safetensors", "base": "gguf"})

    def test_birefnet_annotation_preserves_the_exact_published_tuple_and_payload(self):
        registry = load_registry()
        catalogue = resource_path("model_catalogue", "background_removal.json")
        download = json.loads(catalogue.read_text(encoding="utf-8"))[BIREFNET_KEY]
        self.assertEqual(
            download_identity(download),
            "2af729e609a87f88a63a7ec9959b80ad4d8910ff9eb62a8cf4ded6f773e8dfca",
        )
        self.assertEqual(download["size"], 444473596)
        self.assertEqual(
            download["sha256"],
            "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154",
        )
        row = registry.annotations_for(BIREFNET_KEY, download)
        self.assertEqual(
            (row["role"], row["architecture"], row["format"]),
            ("background_removal", "BiRefNet", "safetensors"),
        )
        self.assertEqual(row["artifact"]["pinned_url"], download["url"])
        self.assertEqual(row["artifact"]["bytes"], download["size"])
        self.assertEqual(row["artifact"]["sha256"], download["sha256"])
        self.assertEqual(row["companions"]["image"], ())
        self.assertEqual(registry.profile(row["profile_id"])["license"]["value"], "MIT")

    def test_metadata_cannot_change_downloads_and_unknown_custom_models_stay_unknown(self):
        registry = load_registry()
        key = "llm/LFM3-1.7B-Q4_K_M.gguf"
        download = json.loads(resource_path("model_catalogue", "llm.json").read_text())[key]
        with self.assertRaisesRegex(RegistryError, "no longer matches"):
            registry.annotations_for(key, dict(download, url="https://example.invalid/other.gguf"))
        self.assertIsNone(registry.annotations_for("llm/custom.gguf", download))
        with self.assertRaises(TypeError):
            registry.artifact("existing:" + key)["runtime"]["native_llama_cpp"]["chat"] = "verified"

    def test_chat_weights_and_components_are_distinct_with_valid_image_companions(self):
        registry = load_registry()
        for row in registry.entries(include_proposed=True):
            if row["chat_weight"]:
                self.assertEqual(row["role"], "language_model")
                self.assertTrue(row["uses_chat_runtime"])
            if row["role"] == "vision_projector":
                self.assertFalse(row["chat_weight"])
        smol = registry.artifact("existing:vision/SmolVLM-256M-Instruct-Q8_0.gguf")
        self.assertFalse(smol["companions"]["text"])
        self.assertTrue(smol["companions"]["image"])
        for companion in smol["companions"]["image"]:
            self.assertEqual(registry.artifact(companion)["role"], "vision_projector")

    def test_repaired_sources_keep_exact_weight_and_projector_identity(self):
        from urllib.parse import urlsplit

        registry = load_registry()
        weight_key = "llm/Qwen3.5-9B-ultra-uncensored-heretic-v2-Q4_K_M.gguf"
        projector_key = "vision/mmproj-Qwen3.5-9B-BF16.gguf"
        catalogue = {}
        for name in ("clip.json", "llm.json", "llm_vision.json"):
            catalogue.update(json.loads(resource_path("model_catalogue", name).read_text()))
        for key in ("clip/qwen_3_4b.safetensors", weight_key, projector_key):
            row = registry.annotations_for(key, catalogue[key])
            path = urlsplit(catalogue[key]["url"]).path
            self.assertIn("/resolve/", path)
            self.assertNotIn("/main/", path)
            self.assertEqual(row["artifact"]["pinned_url"], catalogue[key]["url"])
            self.assertIsNotNone(row["artifact"]["sha256"])
        weight, projector = (catalogue[key] for key in (weight_key, projector_key))
        self.assertIn("heretic-v2", weight["filename"])
        self.assertEqual(weight["mmproj"], projector["filename"])
        self.assertEqual(
            urlsplit(weight["url"]).path.rsplit("/", 1)[0],
            urlsplit(projector["url"]).path.rsplit("/", 1)[0],
        )
        self.assertIn(
            "existing:" + projector_key,
            registry.artifact("existing:" + weight_key)["companions"]["image"],
        )

    def test_benchmarks_preserve_nulls_units_protocol_and_attribution(self):
        registry = load_registry()
        rows = registry.benchmarks("language:lfm3_1_7b")
        self.assertTrue(rows)
        self.assertTrue(all(not row["local_reproduction"] for row in rows))
        self.assertEqual(rows[0]["precision"], "BF16")
        self.assertIsNone(rows[0]["quantization"])
        self.assertIn("source_url", rows[0]["evidence"])
        self.assertTrue(rows[0]["protocol"]["protocol_id"])
        proposed = registry.benchmarks("proposed:ministral3_8b")
        self.assertTrue(any(row["unit"] == "fraction" for row in proposed))
        self.assertTrue(all(row["evidence"] for row in proposed))

    def test_import_and_metadata_read_do_not_load_tensor_or_application_runtimes(self):
        script = """
import sys
from ruinedfooocus.features.models.model_registry import load_registry
load_registry()
assert not any(name in sys.modules for name in ('torch', 'comfy', 'xllamacpp', 'ruinedfooocus.app.state'))
"""
        subprocess.run(
            [sys.executable, "-c", script], check=True, cwd=Path(__file__).resolve().parents[1]
        )

    def test_invalid_documents_fail_with_explicit_errors(self):
        original = registry_document()
        mutations = [
            lambda doc: doc["artifacts"].append(doc["artifacts"][0]),
            lambda doc: doc["artifacts"][0].update(profile_id=[]),
            lambda doc: doc["artifacts"][0]["artifact"].update(sha256=42),
            lambda doc: doc["artifacts"][0]["artifact"].update(bytes=True),
            lambda doc: doc["artifacts"][0]["companions"].update(image=["missing:projector"]),
            lambda doc: doc["artifacts"][0].update(role="vision_projector", chat_weight=True),
            lambda doc: doc["benchmarks"][0].update(value=float("nan")),
            lambda doc: doc["payload_alias_groups"][0].update(sha256=None),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                document = copy.deepcopy(original)
                mutate(document)
                with self.assertRaises(RegistryError):
                    ModelRegistry(document)

    def test_missing_corrupt_duplicate_keys_and_file_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "catalogue.json"
            with self.assertRaises(RegistryError):
                load_registry(path)
            for content in ("{", '{"schema_version":1,"schema_version":1}'):
                path.write_text(content)
                with self.assertRaises(RegistryError):
                    load_registry(path)
            document = {
                "schema_version": 1,
                "profiles": {},
                "artifacts": [],
                "benchmarks": [],
                "provenance": {"revision": "before"},
            }
            path.write_text(json.dumps(document))
            first = load_registry(path)
            self.assertIs(first, load_registry(path))
            document["provenance"]["revision"] = "after-change"
            path.write_text(json.dumps(document))
            self.assertEqual(load_registry(path).provenance["revision"], "after-change")
            self.assertEqual(first.provenance["revision"], "before")


if __name__ == "__main__":
    unittest.main()
