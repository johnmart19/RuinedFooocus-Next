"""Bernini resources bind only the pinned one-frame image component graph."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.core.path import PathManager
from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models.acquisition import Catalogue
from ruinedfooocus.features.models.model_registry import load_registry


EXPECTED = {
    "high": (15574833216, "9ff3d7369da98f8eaf71045f7d0e99d5e344eea5e7dc934a930608786fe73f52"),
    "low": (15574833216, "ce35c070f345c91abe43d6bf29b0eccb8687e4026fadc0e0a70a73c9afb7bbb3"),
    "clip": (6735906897, "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68"),
    "vae": (253806278, "1ab9a32cc2c740f6e39d80d367ce5dcc28db8c71b79b28670546b8973e9d75f9"),
    "lora": (630697104, "37d49218544b9e0bfb8e831d1399f451fbc5068aff6474f42a90c928363c3573"),
}


class BerniniImageResourceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.sources = json.loads(resource_path("bernini_image_components.json").read_text())[
            "components"
        ]
        self.downloads = json.loads(
            resource_path("model_catalogue", "bernini_image.json").read_text()
        )
        self.manager = PathManager.__new__(PathManager)
        self.manager.paths = {
            "path_" + category: str(self.root / category)
            for category in ("diffusion_models", "clip", "vae", "loras")
        }
        self.manager.model_paths = {"cache_path": self.root / "cache"}
        self.manager.DOWNLOADABLE_FILES = copy.deepcopy(self.downloads)
        self.registry = load_registry()

    def test_exact_manifest_tuples_keep_shared_physical_filenames_and_source_only_evidence(self):
        self.assertEqual(set(self.sources), set(EXPECTED))
        self.assertEqual(len(self.downloads), 5)
        roles = dict(
            high="image_diffusion_checkpoint",
            low="image_stage_companion",
            clip="text_conditioning_encoder",
            vae="latent_autoencoder",
            lora="image_lora",
        )
        for role, source in self.sources.items():
            with self.subTest(role=role):
                self.assertEqual((source["bytes"], source["sha256"]), EXPECTED[role])
                self.assertEqual(source["filename"], source["source_filename"])
                self.assertIn("/resolve/" + source["revision"] + "/", source["url"])
                self.assertEqual(len(source["revision"]), 40)
                download = self.downloads[source["catalogue_key"]]
                self.assertEqual(download["filename"], source["filename"])
                self.assertEqual(download["url"], source["url"])
                self.assertEqual((download["size"], download["sha256"]), EXPECTED[role])
                row = self.registry.annotations_for(source["catalogue_key"], download)
                self.assertEqual(row["profile_id"], "generation:bernini_image")
                self.assertEqual(row["role"], roles[role])
                self.assertFalse(row["chat_weight"])
                self.assertFalse(row["uses_chat_runtime"])
                self.assertFalse(row["artifact"]["full_file_hash_recomputed"])
                self.assertEqual(row["artifact"]["header_sha256"], source["header_sha256"])
                self.assertEqual(row["artifact"]["pinned_url"], source["url"])
                self.assertEqual(
                    row["hardware"]["standalone_vram_status"],
                    "unqualified_workload_and_components_dependent",
                )
        high = self.registry.artifact("existing:" + self.sources["high"]["catalogue_key"])
        expected = [
            "existing:" + self.sources[role]["catalogue_key"] for role in ("low", "clip", "vae")
        ]
        for mode in ("text", "image"):
            self.assertEqual(list(high["companions"][mode]), expected)
        self.assertEqual(high["native_family"], "BerniniImage")
        self.assertEqual(high["evidence"]["default_recipe_id"], "native-bernini-image")
        for component in high["component_contract"]["roles"]:
            key = self.registry.artifact(component["artifact_id"])["catalogue_key"]
            self.assertEqual(component["filename"], self.downloads[key]["filename"])
        retired = json.loads(resource_path("model_metadata", "index.json").read_text())[
            "migration_baseline"
        ]["scope_retirement"]["artifact_ids"]
        self.assertFalse({"existing:" + key for key in self.downloads} & set(retired))

    def test_passive_capture_collects_required_pair_components_and_excludes_optional_lora(self):
        from ruinedfooocus.core.preparation import preparation_service

        catalogue = Catalogue(self.manager, self.registry)
        with (
            patch.object(
                preparation_service, "submit", side_effect=AssertionError("No acquisition")
            ),
            patch.object(self.manager, "download_file", side_effect=AssertionError("No download")),
            patch.object(self.manager, "get_file_path", side_effect=AssertionError("No resolver")),
            patch("urllib.request.urlopen", side_effect=AssertionError("No network")),
            patch("ruinedfooocus.features.models.acquisition._space", return_value=10**12),
        ):
            high_key = self.sources["high"]["catalogue_key"]
            low = catalogue.record(self.sources["low"]["catalogue_key"])
            self.assertEqual(low["task"], "Component")
            self.assertEqual(low["category_label"], "Component")
            self.assertFalse(low["can_activate"])
            for mode in ("text", "image"):
                plan = catalogue.capture(high_key, mode=mode, reserve_gib=0)
                self.assertEqual(len(plan.artifacts), 4)
                expected = [self.sources[role] for role in ("high", "low", "clip", "vae")]
                self.assertEqual(
                    [request.url for request in plan.artifacts], [row["url"] for row in expected]
                )
                self.assertEqual(
                    sum(request.expected_size for request in plan.artifacts),
                    sum(row["bytes"] for row in expected),
                )
                for request, source in zip(plan.artifacts, expected):
                    self.assertEqual(request.sha256, source["sha256"])
                    self.assertEqual(request.destination.name, source["source_filename"])
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
