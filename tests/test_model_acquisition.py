"""Catalogue browsing and shared preparation without real transfers or compute."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.path import PathManager
from ruinedfooocus.core.preparation import PreparationService
from ruinedfooocus.features.models.acquisition import (
    Catalogue,
    CATEGORIES,
    AcquisitionSession,
    checkpoint_name,
    DownloadPlan,
)
from ruinedfooocus.features.models.model_registry import (
    ModelRegistry,
    RegistryError,
    download_identity,
    load_registry as packaged_registry,
)


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name).resolve()
        self.manager = PathManager.__new__(PathManager)
        self.manager.paths = {"path_" + kind: str(self.root / kind) for kind in CATEGORIES}
        self.manager.model_paths = {"cache_path": self.root / "cache"}
        self.manager.DOWNLOADABLE_FILES = {
            "checkpoints/a.safetensors": {
                "url": "https://example.test/a",
                "path": "path_checkpoints",
                "filename": "a.safetensors",
            },
            "clip/encoder.safetensors": {
                "url": "https://example.test/clip",
                "path": "path_clip",
                "filename": "encoder.safetensors",
            },
        }
        rows = []
        for key, source in self.manager.DOWNLOADABLE_FILES.items():
            rows.append(
                dict(
                    id="existing:" + key,
                    catalogue_key=key,
                    state="existing",
                    profile_id="image",
                    label=key,
                    role="image_diffusion_checkpoint"
                    if key.startswith("checkpoints")
                    else "text_conditioning_encoder",
                    chat_weight=False,
                    download_identity=download_identity(source),
                    quantization={"label": "FP16"},
                    artifact={"bytes": 4, "sha256": hashlib.sha256(b"unit").hexdigest()},
                    companions={"text": [], "image": []},
                )
            )
        rows[0]["companions"]["text"] = [rows[1]["id"]]
        self.document = dict(
            schema_version=1,
            profiles={"image": {"consumers": ["SDXL"], "license": {"value": "Test license"}}},
            artifacts=rows,
            benchmarks=[],
        )
        self.registry = ModelRegistry(self.document)
        loaded = patch(
            "ruinedfooocus.features.models.model_registry.load_registry", return_value=self.registry
        )
        loaded.start()
        self.addCleanup(loaded.stop)
        self.catalogue = Catalogue(self.manager, self.registry)

    def installed(self, category, name, data=b"unit"):
        path = self.root / category / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_browsing_is_cached_until_explicit_refresh_and_retains_custom_local_files(self):
        custom = self.installed("checkpoints", "custom.safetensors", b"custom")
        with patch.object(self.catalogue, "_scan", wraps=self.catalogue._scan) as scan:
            all_rows = self.catalogue.records()
            self.catalogue.records(query="encoder")
            self.catalogue.records(category="Image models", status="Installed")
            self.assertEqual(scan.call_count, 1)
            self.catalogue.invalidate()
            self.catalogue.records()
            self.assertEqual(scan.call_count, 2)
        row = next(row for row in all_rows if row["local"] == str(custom))
        self.assertEqual(row["status"], "Installed · custom")
        self.assertEqual(row["quantization"], "Unknown")
        self.assertFalse(row["allowed"])
        self.assertFalse(row["can_activate"])
        self.assertFalse((self.root / "vae").exists())
        copy.deepcopy(all_rows)  # Gradio's per-session state may copy the snapshot.

    def test_same_name_wrong_size_is_visible_conflict_and_cannot_activate_or_replace(self):
        path = self.installed("checkpoints", "a.safetensors", b"different file")
        row = self.catalogue.record("checkpoints/a.safetensors")
        self.assertEqual(row["local"], str(path))
        self.assertIn("size mismatch", row["status"])
        self.assertFalse(row["can_activate"])
        self.assertEqual(len(self.catalogue.records(status="Installed", include_local=False)), 1)
        with self.assertRaisesRegex(ValueError, "different size"):
            self.catalogue.capture(row["key"], reserve_gib=0)
        with self.assertRaises(ValueError):
            checkpoint_name(row, Mock())
        self.assertEqual(path.read_bytes(), b"different file")

    def test_capture_uses_identical_legacy_request_and_freezes_profile_destinations(self):
        legacy = self.manager.preparation_request("checkpoints/a.safetensors")
        plan = self.catalogue.capture("checkpoints/a.safetensors", reserve_gib=0)
        self.assertEqual(plan.artifacts[0], legacy)
        self.assertEqual(legacy.expected_size, 4)
        self.assertEqual(legacy.sha256, hashlib.sha256(b"unit").hexdigest())
        self.assertEqual(len(plan.artifacts), 2)
        self.manager.paths["path_checkpoints"] = str(self.root / "changed")
        self.manager.DOWNLOADABLE_FILES["checkpoints/a.safetensors"]["url"] = (
            "https://example.test/replaced"
        )
        self.assertEqual(plan.artifacts[0], legacy)
        self.assertEqual(plan.artifacts[0].destination.parent, self.root / "checkpoints")

    def test_conflicting_or_wrong_tuple_metadata_is_rejected_before_subscription(self):
        key = "checkpoints/a.safetensors"
        self.manager.DOWNLOADABLE_FILES[key]["size"] = 5
        with self.assertRaisesRegex(ValueError, "Conflicting size"):
            self.manager.preparation_request(key)
        self.manager.DOWNLOADABLE_FILES[key].pop("size")
        self.manager.DOWNLOADABLE_FILES[key]["url"] = "https://example.test/another-source"
        with self.assertRaises(RegistryError):
            self.manager.preparation_request(key)
        row = self.catalogue.record(key)
        self.assertFalse(row["allowed"])

    def test_retired_media_unknown_size_and_disk_reserve_fail_before_subscription(self):
        document = copy.deepcopy(self.document)
        document["artifacts"][0]["role"] = "audio_video_diffusion_checkpoint"
        video = Catalogue(self.manager, ModelRegistry(document))
        with self.assertRaisesRegex(ValueError, "still images and local Chat"):
            video.record("checkpoints/a.safetensors")
        with self.assertRaisesRegex(ValueError, "still images and local Chat"):
            video.capture("checkpoints/a.safetensors", reserve_gib=0)
        with patch("ruinedfooocus.features.models.acquisition._space", return_value=7):
            with self.assertRaisesRegex(ValueError, "free-space reserve"):
                self.catalogue.capture("checkpoints/a.safetensors", reserve_gib=0)
        document = copy.deepcopy(self.document)
        document["artifacts"][0]["artifact"].update(bytes=None, sha256=None)
        unknown = ModelRegistry(document)
        with patch(
            "ruinedfooocus.features.models.model_registry.load_registry", return_value=unknown
        ):
            catalogue = Catalogue(self.manager, unknown)
            with self.assertRaisesRegex(ValueError, "size is unknown"):
                catalogue.capture("checkpoints/a.safetensors", reserve_gib=0)
            self.assertIsNone(
                catalogue.capture("checkpoints/a.safetensors", reserve_gib=0, allow_unknown=True)
                .artifacts[0]
                .expected_size
            )

    def test_path_traversal_and_unrecorded_workflow_are_not_ready_activation(self):
        key = "checkpoints/a.safetensors"
        self.manager.DOWNLOADABLE_FILES[key]["filename"] = "../outside.safetensors"
        with self.assertRaisesRegex(ValueError, "inside"):
            self.catalogue.record(key)
        self.manager.DOWNLOADABLE_FILES[key]["filename"] = "a.safetensors"
        self.installed("checkpoints", "a.safetensors")
        document = copy.deepcopy(self.document)
        document["profiles"]["image"]["consumers"] = []
        row = Catalogue(self.manager, ModelRegistry(document)).record(key)
        self.assertIn("workflow not recorded", row["support"])
        self.assertFalse(row["can_activate"])

    def test_native_qwen_identity_activates_safetensors_and_gguf_from_diffusion_models(self):
        for filename in ("qwen_turbo_int8.safetensors", "qwen_Q4_K_M.gguf"):
            with self.subTest(filename=filename):
                self.manager.DOWNLOADABLE_FILES = {
                    "diffusion_models/" + filename: {
                        "url": "https://example.test/qwen",
                        "path": "path_diffusion_models",
                        "filename": filename,
                    }
                }
                document = copy.deepcopy(self.document)
                source = self.manager.DOWNLOADABLE_FILES["diffusion_models/" + filename]
                annotation = document["artifacts"][0]
                annotation.update(
                    id="existing:diffusion_models/" + filename,
                    catalogue_key="diffusion_models/" + filename,
                    download_identity=download_identity(source),
                    native_family="QwenImage21",
                    architecture="QwenImage21",
                    companions={"text": [], "image": []},
                )
                document["profiles"]["image"]["consumers"] = ["Qwen Image 2.1"]
                document["artifacts"] = [annotation]
                local = self.installed("diffusion_models", filename)
                catalogue = Catalogue(self.manager, ModelRegistry(document))
                row = catalogue.record("diffusion_models/" + filename)
                self.assertTrue(row["can_activate"])
                models = SimpleNamespace(get_names=lambda _: [filename], get_file=lambda *_: local)
                self.assertEqual(checkpoint_name(row, models), filename)
                self.assertEqual(
                    catalogue.records(category="Image models", include_local=False), [row]
                )

    def test_image_category_excludes_typed_video_without_guessing_from_filename(self):
        document = copy.deepcopy(self.document)
        document["artifacts"][0]["role"] = "audio_video_diffusion_checkpoint"
        catalogue = Catalogue(self.manager, ModelRegistry(document))
        self.assertEqual(catalogue.records(category="Image models"), [])
        self.assertFalse(
            any(row["key"] == "checkpoints/a.safetensors" for row in catalogue.records())
        )

    def test_companion_only_and_missing_native_recipe_are_not_advertised_as_usable_images(self):
        for role in ("image_diffusion_companion", "image_diffusion_checkpoint"):
            with self.subTest(role=role):
                document = copy.deepcopy(self.document)
                document["artifacts"][0].update(
                    role=role, native_family="SDXL", auto_recipe_supported=False
                )
                self.installed("checkpoints", "a.safetensors")
                catalogue = Catalogue(self.manager, ModelRegistry(document))
                row = catalogue.record("checkpoints/a.safetensors")
                self.assertTrue(row["allowed"])
                self.assertFalse(row["can_activate"])
                if role == "image_diffusion_companion":
                    self.assertEqual(row["task"], "Component")
                    self.assertEqual(catalogue.records(category="Image models"), [])
                else:
                    self.assertIn("not implemented", row["support"])

    def test_two_sessions_share_legacy_tuple_and_cancel_only_their_own_subscription(self):
        ready, release = threading.Event(), threading.Event()
        calls = []

        def transport(request, staging, progress, cancelled):
            calls.append(request)
            ready.set()
            self.assertTrue(release.wait(3))
            path = staging / "inert.part"
            path.write_bytes(b"unit")
            progress(4, 4, 0)
            return path

        service = PreparationService(transport=transport, staging_root=self.root / "staging")
        self.addCleanup(lambda: (release.set(), service.close()))
        plan = self.catalogue.capture(
            "checkpoints/a.safetensors", include_companions=False, reserve_gib=0
        )
        first, second = AcquisitionSession(), AcquisitionSession()
        first.start(plan, service=service)
        self.assertTrue(ready.wait(3))
        second.start(plan, service=service)
        first.cancel()
        self.assertTrue(second.poll()[1])
        release.set()
        second.handles[0].result(timeout=3)
        self.assertEqual(len(calls), 1)
        message, running, complete = second.poll()
        self.assertTrue(complete)
        self.assertFalse(running)
        self.assertIn("Ready", message)
        self.assertIn("cancelled", first.poll()[0])
        self.assertEqual(plan.artifacts[0].destination.read_bytes(), b"unit")

    def test_offline_never_subscribes_and_explicit_activation_only_returns_local_identity(self):
        session, service = AcquisitionSession(), Mock()
        plan = self.catalogue.capture("checkpoints/a.safetensors", reserve_gib=0)
        with self.assertRaisesRegex(ValueError, "Offline"):
            session.start(plan, offline=True, service=service)
        service.submit.assert_not_called()
        local = self.installed("checkpoints", "a.safetensors")
        row = self.catalogue.record("checkpoints/a.safetensors")
        models = SimpleNamespace(get_names=lambda _: ["a.safetensors"], get_file=lambda *_: local)
        self.assertEqual(checkpoint_name(row, models), "a.safetensors")

    def test_entire_authoritative_catalogue_projects_without_inventing_workflow_or_memory_support(
        self,
    ):
        from ruinedfooocus.core.paths import resource_path
        from ruinedfooocus.features.models.ui_acquisition import detail_rows

        downloads = {}
        for path in resource_path("model_catalogue").glob("*.json"):
            downloads.update(json.loads(path.read_text(encoding="utf-8")))
        self.manager.DOWNLOADABLE_FILES = downloads
        rows = Catalogue(self.manager, packaged_registry()).records(include_local=False)
        self.assertEqual({row["key"] for row in rows}, set(downloads))
        self.assertFalse(any(row["can_activate"] for row in rows))
        for row in rows:
            with self.subTest(key=row["key"]):
                detail_rows(row)
                self.assertNotEqual(row["task"], "Video (excluded)")

    def test_background_removal_projection_uses_registered_provenance_and_local_roots(self):
        from ruinedfooocus.core.paths import resource_path

        path = resource_path("model_catalogue", "background_removal.json")
        self.manager.DOWNLOADABLE_FILES = json.loads(path.read_text(encoding="utf-8"))
        registry = packaged_registry()
        catalogue = Catalogue(self.manager, registry)
        custom = self.installed("background_removal", "local.sft", b"custom file")
        key = "background_removal/birefnet.safetensors"
        (row,) = catalogue.records(category="Background removal", task="Image tools")
        self.assertEqual(
            (row["key"], row["status"], row["quantization"]), (key, "Available", "FP16")
        )
        self.assertTrue(row["allowed"])
        self.assertFalse(row["can_activate"])
        (local,) = catalogue.records(category="Background removal", status="Installed")
        self.assertEqual(local["key"], "local:" + str(custom))
        with patch(
            "ruinedfooocus.features.models.model_registry.load_registry", return_value=registry
        ):
            request = self.manager.preparation_request(key)
        self.assertEqual(request.expected_size, row["published_bytes"])
        self.assertEqual(request.sha256, row["annotation"]["artifact"]["sha256"])
        self.assertEqual(request.destination, self.root / "background_removal/birefnet.safetensors")

    def test_first_failure_cancels_only_this_sessions_remaining_companion(self):
        ready, release = threading.Event(), threading.Event()

        def transport(request, staging, progress, cancelled):
            if request.destination.name == "a.safetensors":
                raise OSError("synthetic transfer failure")
            ready.set()
            self.assertTrue(release.wait(3))
            path = staging / "inert.part"
            path.write_bytes(b"unit")
            return path

        service = PreparationService(transport=transport, staging_root=self.root / "staging")
        self.addCleanup(lambda: (release.set(), service.close()))
        plan = self.catalogue.capture("checkpoints/a.safetensors", reserve_gib=0)
        first, other = AcquisitionSession(), AcquisitionSession()
        first.start(plan, service=service)
        self.assertTrue(ready.wait(3))
        other.start(DownloadPlan("Shared encoder", (plan.artifacts[1],)), service=service)
        with self.assertRaises(OSError):
            first.handles[0].result(timeout=3)
        message, running, completed = first.poll()
        self.assertIn("synthetic transfer failure", message)
        self.assertFalse(running)
        self.assertFalse(completed)
        self.assertTrue(other.poll()[1])
        release.set()
        other.handles[0].result(timeout=3)
        self.assertTrue(other.poll()[2])
        with self.assertRaisesRegex(ValueError, "no missing files"):
            AcquisitionSession().start(DownloadPlan("Empty", ()), service=service)


if __name__ == "__main__":
    unittest.main()
