"""Explicit, previewed maintenance destroys only private fixture targets."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.app.resource_lifecycle import FairAdmission, JobRegistry
from ruinedfooocus.core import maintenance_reset as reset
from ruinedfooocus.core.path import PathManager
from ruinedfooocus.core.preparation import PreparationBusy, PreparationService
from ruinedfooocus.core.settings import SettingsManager
from ruinedfooocus.features.gallery.index_store import connect_index


class MaintenanceResetTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(patch.dict(os.environ, RF_DATA_DIR=str(self.root)))
        self.settings = SettingsManager("current", defaults={"prompt": "factory", "style": []})
        self.settings.default_settings["prompt"] = "saved"
        self.settings.save_settings()
        self.manager = PathManager.__new__(PathManager)
        self.manager.name = "current"
        self.manager.set_settings_path("current")
        self.manager.paths = self.manager.load_paths()
        self.manager.model_paths = self.manager.get_model_paths()
        self.service = PreparationService(staging_root=self.root / "staging")
        self.addCleanup(self.service.close)
        self.registry, self.admission = JobRegistry(), FairAdmission()

    def file(self, name, value=b"private fixture"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return path

    def thumbnail(self, letter="a"):
        return self.file("cache/library-thumbnails/" + letter * 64 + ".webp")

    def execute(self, plan, **options):
        return reset.execute_reset(
            plan,
            self.manager,
            self.settings,
            service=self.service,
            registry=self.registry,
            admission=self.admission,
            **options,
        )

    def test_cache_only_retains_legacy_payloads_inputs_resumes_models_and_outputs(self):
        thumbnail = self.thumbnail()
        protected = [
            self.file(name)
            for name in [
                "cache/llama.cpp/server.exe",
                "cache/rag/weights.bin",
                "cache/preparation/download.part",
                "cache/uploads/input.png",
                "cache/manual.txt",
                "models/model.gguf",
                "repositories/native.py",
                "outputs/original.png",
            ]
        ]
        plan = reset.preview_reset("cache", self.manager, self.settings)
        self.assertEqual([target.path for target in plan.targets], [str(thumbnail)])
        result = self.execute(plan)
        self.assertTrue(result["complete"])
        self.assertEqual(result["removed"], 1)
        self.assertFalse(thumbnail.exists())
        self.assertTrue(all(path.read_bytes() == b"private fixture" for path in protected))

    def test_unknown_index_payload_is_retained_instead_of_deleted(self):
        unknown = self.file("cache/images.db", b"private unknown non-SQLite payload")
        plan = reset.preview_reset("cache", self.manager, self.settings)
        self.assertIn(str(unknown), plan.retained)
        self.assertFalse(any(target.path == str(unknown) for target in plan.targets))
        self.assertTrue(self.execute(plan)["complete"])
        self.assertEqual(unknown.read_bytes(), b"private unknown non-SQLite payload")

    def test_outputs_require_confirmation_and_clear_exact_default_targets(self):
        original = self.file("outputs/day/original.png")
        other = self.file("custom outputs/untouched.png")
        plan = reset.preview_reset("cache-outputs", self.manager, self.settings)
        self.assertEqual(sum(target.kind == "output-file" for target in plan.targets), 1)
        with self.assertRaisesRegex(ValueError, "Confirm"):
            self.execute(plan)
        self.assertTrue(original.exists())
        result = self.execute(plan, confirmed=True)
        self.assertTrue(result["complete"])
        self.assertFalse(original.exists())
        self.assertTrue((self.root / "outputs").is_dir())
        self.assertTrue(other.exists())

    def test_factory_reset_uses_paired_current_profile_save_and_preserves_other_profile(self):
        other = SettingsManager("other", defaults={"prompt": "other"})
        before = other.settings_path.read_bytes()
        captured = copy.deepcopy(self.settings.default_settings)
        plan = reset.preview_reset("factory-profile", self.manager, self.settings)
        self.assertEqual(plan.profile, "current")
        self.assertEqual(
            {Path(target.path).name for target in plan.targets}, {"settings.json", "paths.json"}
        )
        with self.assertRaisesRegex(ValueError, "Confirm"):
            self.execute(plan)
        result = self.execute(replace(plan, profile="other"), confirmed=True)
        self.assertTrue(result["complete"])
        self.assertTrue(result["profile_reset"])
        self.assertEqual(result["profile"], "current")
        self.assertEqual(self.settings.default_settings["prompt"], "factory")
        self.assertEqual(captured["prompt"], "saved")
        self.assertEqual(other.settings_path.read_bytes(), before)
        self.assertEqual(SettingsManager("current").default_settings["prompt"], "factory")

    def test_factory_save_failure_rolls_back_both_original_documents(self):
        from ruinedfooocus.core import profile_store

        plan = reset.preview_reset("factory-profile", self.manager, self.settings)
        original = {target.path: Path(target.path).read_bytes() for target in plan.targets}
        write = profile_store.save_json

        def fail(path, value):
            if Path(path).name == "paths.json":
                raise PermissionError("private injected save failure")
            return write(path, value)

        with (
            patch.object(profile_store, "save_json", side_effect=fail),
            self.assertRaises(PermissionError),
        ):
            self.execute(plan, confirmed=True)
        self.assertEqual(self.settings._default_settings["prompt"], "saved")
        self.assertTrue(all(Path(path).read_bytes() == value for path, value in original.items()))

    def test_changed_preview_rejects_before_any_deletion(self):
        thumbnail = self.thumbnail()
        plan = reset.preview_reset("cache-outputs", self.manager, self.settings)
        output = self.file("outputs/new.png")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.execute(plan, confirmed=True)
        self.assertTrue(thumbnail.exists())
        self.assertTrue(output.exists())

    def test_compute_job_or_artifact_use_rejects_before_changes(self):
        thumbnail = self.thumbnail()
        plan = reset.preview_reset("cache", self.manager, self.settings)
        item = self.registry.create()
        with self.assertRaises(PreparationBusy):
            self.execute(plan)
        self.registry.finish(item)
        with self.service.artifact_lease(thumbnail), self.assertRaises(PreparationBusy):
            self.execute(plan)
        self.assertTrue(thumbnail.exists())

    def test_unrelated_resident_model_lease_does_not_require_unloading_to_clear_cache(self):
        model = self.file("models/loaded.gguf")
        self.thumbnail()
        plan = reset.preview_reset("cache", self.manager, self.settings)
        with self.service.artifact_lease(model):
            self.assertTrue(self.execute(plan)["complete"])
        self.assertEqual(model.read_bytes(), b"private fixture")

    def test_profile_publication_in_progress_rejects_factory_reset(self):
        plan = reset.preview_reset("factory-profile", self.manager, self.settings)
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def publish():
            entered.set()
            self.assertTrue(release.wait(5))

        with ThreadPoolExecutor(max_workers=1) as pool:
            saving = pool.submit(
                self.settings.save_profile,
                {"prompt": "another explicit save"},
                self.manager.paths,
                profile="current",
                on_publish=publish,
            )
            self.assertTrue(entered.wait(5))
            with self.assertRaises(PreparationBusy):
                self.execute(plan, confirmed=True)
            release.set()
            saving.result(5)
        self.assertEqual(self.settings.default_settings["prompt"], "another explicit save")

    def test_maintenance_excludes_concurrent_new_compute_until_it_finishes(self):
        output = self.file("outputs/original.png")
        plan = reset.preview_reset("cache-outputs", self.manager, self.settings)
        entered, release, submitted = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(release.set)
        unlink = Path.unlink

        def remove(path, *args, **kwargs):
            if path == output:
                entered.set()
                self.assertTrue(release.wait(5))
            return unlink(path, *args, **kwargs)

        def submit():
            submitted.set()
            return self.registry.create()

        with ThreadPoolExecutor(max_workers=2) as pool, patch.object(Path, "unlink", remove):
            action = pool.submit(self.execute, plan, confirmed=True)
            self.assertTrue(entered.wait(5))
            queued = pool.submit(submit)
            self.assertTrue(submitted.wait(5))
            self.assertFalse(queued.done())
            release.set()
            self.assertTrue(action.result(5)["complete"])
            self.registry.finish(queued.result(5))

    def test_cancel_before_or_between_deletions_never_reports_success(self):
        first, second = self.file("outputs/a.png"), self.file("outputs/b.png")
        plan = reset.preview_reset("cache-outputs", self.manager, self.settings)
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(CancelledError):
            self.execute(plan, confirmed=True, cancelled=cancelled)
        self.assertTrue(first.exists() and second.exists())
        cancelled.clear()
        unlink = Path.unlink

        def remove(path, *args, **kwargs):
            result = unlink(path, *args, **kwargs)
            if path == first:
                cancelled.set()
            return result

        with patch.object(Path, "unlink", remove):
            result = self.execute(plan, confirmed=True, cancelled=cancelled)
        self.assertFalse(result["complete"])
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["removed"], 1)
        self.assertTrue(second.exists())

    def test_permission_failure_is_partial_failure_not_false_success(self):
        output = self.file("outputs/a.png")
        plan = reset.preview_reset("cache-outputs", self.manager, self.settings)
        with patch.object(Path, "unlink", side_effect=PermissionError("private denied fixture")):
            result = self.execute(plan, confirmed=True)
        self.assertFalse(result["complete"])
        self.assertIn("private denied fixture", result["errors"][0])
        self.assertTrue(output.exists())

    def test_output_link_blocks_plan_without_following_or_deleting_it(self):
        output = self.file("outputs/link.png")
        lstat = Path.lstat

        def status(path):
            if path == output:
                return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
            return lstat(path)

        with patch.object(Path, "lstat", status), self.assertRaisesRegex(ValueError, "Linked"):
            reset.preview_reset("cache-outputs", self.manager, self.settings)
        self.assertTrue(output.exists())

    def test_custom_roots_are_retained_and_cannot_be_supplied_as_targets(self):
        custom = self.file("custom mount/original.png")
        self.manager.model_paths = {**self.manager.model_paths, "temp_outputs_path": custom.parent}
        plan = reset.preview_reset("cache-outputs", self.manager, self.settings)
        self.assertIn(str(custom.parent), plan.retained)
        self.assertTrue(self.execute(plan, confirmed=True)["complete"])
        self.assertTrue(custom.exists())
        with self.assertRaises(ValueError):
            reset.preview_reset("../../models", self.manager, self.settings)

    def test_recheck_recovers_missing_owned_folders_with_no_original_deletion(self):
        cache = self.root / "cache"
        self.assertTrue(cache.is_relative_to(self.root) and cache != self.root)
        shutil.rmtree(cache)
        original = self.file("models/keep.gguf")
        folders = reset.recheck_folders(self.manager, self.settings)
        self.assertTrue(all(Path(path).is_dir() for path in folders))
        self.assertTrue(original.exists())

    def test_index_reset_is_reinitialized_and_unpreviewed_cache_entries_are_retained(self):
        database = self.root / "cache/images.db"
        connection = connect_index(database)
        self.addCleanup(connection.close)
        connection.execute(
            "INSERT INTO images(fullpath,path,json) VALUES('private','private','{}')"
        )
        connection.commit()
        first = self.thumbnail()
        plan = reset.preview_reset("cache", self.manager, self.settings)
        unlink = Path.unlink
        new = self.root / "cache/library-thumbnails" / ("b" * 64 + ".webp")

        def remove(path, *args, **kwargs):
            if path == first:
                new.write_bytes(b"not in approved preview")
            return unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", remove):
            result = self.execute(plan)
        self.assertTrue(result["complete"])
        self.assertTrue(new.exists())
        self.assertEqual(connection.execute("SELECT count(*) FROM images").fetchone()[0], 0)
        self.assertEqual(
            connection.execute("SELECT revision FROM library_snapshot").fetchone()[0], 0
        )


if __name__ == "__main__":
    unittest.main()
