"""Pinned Krea style source stays passive until explicit image preparation."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.core.path import PathManager
from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.models.acquisition import Catalogue
from ruinedfooocus.features.models.model_registry import RegistryError, load_registry


KEY = "loras/krea2_style_reference.safetensors"
REVISION = "588feaf955b08b9ac4723eeb751752ab045dba71"
SHA256 = "f50df5a9e62e4be8aa926a63dd5bb1a64770c4004f763c1208007ae13daa82b8"
SIZE = 457111760
URL = f"https://huggingface.co/Comfy-Org/Krea-2/resolve/{REVISION}/{KEY}"


class KreaStyleResourceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.downloads = json.loads(
            resource_path("model_catalogue", "native_krea_style.json").read_text(encoding="utf-8")
        )
        self.manager = PathManager.__new__(PathManager)
        self.manager.paths = {"path_loras": str(self.root / "loras")}
        self.manager.model_paths = {"cache_path": self.root / "cache"}
        self.manager.DOWNLOADABLE_FILES = copy.deepcopy(self.downloads)

    def test_exact_publisher_tuple_has_source_only_image_adapter_annotations(self):
        self.assertEqual(set(self.downloads), {KEY})
        source = self.downloads[KEY]
        self.assertEqual(
            source,
            dict(url=URL, path="path_loras", filename=Path(KEY).name, size=SIZE, sha256=SHA256),
        )
        row = load_registry().annotations_for(KEY, source)
        self.assertEqual(row["role"], "conditioning_adapter")
        self.assertEqual(row["profile_id"], "generation:reviewed_krea2_turbo")
        self.assertFalse(row["chat_weight"])
        self.assertFalse(row["uses_chat_runtime"])
        self.assertEqual(row["artifact"]["bytes"], SIZE)
        self.assertEqual(row["artifact"]["sha256"], SHA256)
        self.assertEqual(row["artifact"]["pinned_url"], URL)
        self.assertEqual(row["artifact"]["availability"], "publisher_metadata_only")
        self.assertFalse(row["artifact"]["full_file_hash_recomputed"])
        self.assertEqual(row["hardware"]["standalone_vram_status"], "profile_or_workflow_dependent")
        changed = dict(source, url=source["url"].replace(REVISION, "main"))
        with self.assertRaisesRegex(RegistryError, "no longer matches"):
            load_registry().annotations_for(KEY, changed)

    def test_catalogue_browsing_and_capture_do_not_download_or_claim_image_activation(self):
        from ruinedfooocus.core.preparation import preparation_service

        with (
            patch.object(
                preparation_service, "submit", side_effect=AssertionError("No acquisition")
            ),
            patch.object(self.manager, "download_file", side_effect=AssertionError("No download")),
            patch.object(
                self.manager, "prepare_file", side_effect=AssertionError("No preparation")
            ),
            patch.object(self.manager, "get_file_path", side_effect=AssertionError("No resolver")),
            patch("urllib.request.urlopen", side_effect=AssertionError("No network")),
            patch("ruinedfooocus.features.models.acquisition._space", return_value=SIZE + 1),
        ):
            catalogue = Catalogue(self.manager, load_registry())
            row = catalogue.record(KEY)
            self.assertEqual(row["category_label"], "LoRAs")
            self.assertEqual(row["task"], "Component")
            self.assertEqual(row["status"], "Available")
            self.assertFalse(row["can_activate"])
            plan = catalogue.capture(KEY, reserve_gib=0)
        self.assertEqual(len(plan.artifacts), 1)
        self.assert_request(plan.artifacts[0])
        self.assertEqual(list(self.root.iterdir()), [])

    def assert_request(self, request):
        self.assertEqual(request.url, URL)
        self.assertEqual(request.expected_size, SIZE)
        self.assertEqual(request.sha256, SHA256)
        self.assertEqual(request.destination, (self.root / "loras" / Path(KEY).name).resolve())


if __name__ == "__main__":
    unittest.main()
