"""Deleted owned storage recovers in private fixtures without reviving custom roots."""

import copy
import errno
import os
from pathlib import Path
import shutil
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core import paths
from ruinedfooocus.core.path import PathManager
from ruinedfooocus.core.settings import SettingsManager
from ruinedfooocus.core.profile_store import read_profile_data, save_profile
from ruinedfooocus.features.gallery.index_store import (
    connect_index,
    incremental_scan,
    index_connection,
    page_snapshot,
)


class StorageRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(patch.dict(os.environ, RF_DATA_DIR=str(self.root)))

    def remove(self, path):
        target = Path(path).absolute()
        self.assertTrue(target.is_relative_to(self.root))
        self.assertNotEqual(target, self.root)
        shutil.rmtree(target)

    def manager(self, **overrides):
        manager = PathManager.__new__(PathManager)
        manager.set_settings_path()
        manager.paths = manager.load_paths()
        manager.paths.update(overrides)
        manager.save_paths()
        manager.model_paths = manager.get_model_paths()
        return manager

    def test_missing_settings_before_boot_creates_factory_profile(self):
        owner = SettingsManager(defaults={"prompt": "factory", "style": []})
        self.assertEqual(owner.default_settings, {"prompt": "factory", "style": []})
        self.assertTrue(owner.settings_path.is_file())

    def test_deleted_profile_replaces_memory_with_factory_defaults_and_preserves_capture(self):
        changed = Mock()
        owner = SettingsManager(
            "running", defaults={"prompt": "factory", "style": []}, on_change=changed
        )
        owner.default_settings.update(prompt="saved", custom="remove with profile")
        owner.default_settings["style"].append("saved style")
        owner.save_settings()
        captured = copy.deepcopy(owner.default_settings)
        previous_memory = owner.default_settings
        changed.reset_mock()
        self.remove(self.root / "settings")
        self.assertEqual(owner.default_settings, {"prompt": "factory", "style": []})
        self.assertTrue(owner.settings_path.is_file())
        self.assertEqual(captured["prompt"], "saved")
        self.assertEqual(previous_memory["style"], ["saved style"])
        changed.assert_called_once_with()

    def test_empty_factory_profile_does_not_lose_unsaved_memory(self):
        owner = SettingsManager()
        owner.default_settings["custom"] = "keep until explicit save"
        self.assertEqual(owner.default_settings["custom"], "keep until explicit save")

    def test_deleted_default_outputs_and_cache_recover_on_next_path_use(self):
        manager = self.manager()
        outputs, cache = self.root / "outputs", self.root / "cache"
        self.remove(outputs)
        self.remove(cache)
        self.assertTrue(manager.model_paths["temp_outputs_path"].is_dir())
        self.assertTrue(manager.model_paths["cache_path"].is_dir())

    def test_deleted_path_profile_resets_paths_without_mutating_captured_map(self):
        custom = self.root / "custom outputs"
        manager = self.manager(path_outputs=str(custom))
        captured = manager.model_paths
        (custom / "original.txt").write_text("untouched", encoding="utf-8")
        self.remove(self.root / "settings")
        self.assertEqual(manager.paths["path_outputs"], manager.DEFAULT_PATHS["path_outputs"])
        self.assertEqual(manager.model_paths["temp_outputs_path"], self.root / "outputs")
        self.assertEqual(captured["temp_outputs_path"], custom)
        self.assertEqual((custom / "original.txt").read_text(), "untouched")

    def test_runtime_recovery_does_not_recreate_missing_custom_source_or_cache(self):
        custom = self.root / "custom mount"
        manager = self.manager(
            path_outputs=str(custom / "outputs"), path_cache=str(custom / "cache")
        )
        self.remove(custom)
        manager.ensure_default_storage()
        self.assertFalse(custom.exists())

    def test_factory_path_lists_are_not_mutated_by_saved_profile_choices(self):
        with patch.object(PathManager, "DEFAULT_PATHS", copy.deepcopy(PathManager.DEFAULT_PATHS)):
            manager = self.manager()
            factory = list(manager.DEFAULT_PATHS["path_loras"])
            manager.paths["path_loras"].append(str(self.root / "custom loras"))
            manager.save_paths()
            self.remove(self.root / "settings")
            self.assertEqual(manager.paths["path_loras"], factory)
            self.assertEqual(manager.DEFAULT_PATHS["path_loras"], factory)

    def test_pending_restart_paths_do_not_prevent_active_default_output_recovery(self):
        manager = self.manager()
        manager.paths["path_outputs"] = str(self.root / "next launch custom outputs")
        manager.save_paths()
        self.remove(self.root / "outputs")
        self.assertEqual(manager.model_paths["temp_outputs_path"], self.root / "outputs")
        self.assertTrue((self.root / "outputs").is_dir())
        self.assertFalse((self.root / "next launch custom outputs").exists())

    def test_disk_profile_save_does_not_switch_active_worker_folders_before_publish(self):
        manager = self.manager()
        pending = {**manager.paths, "path_outputs": str(self.root / "pending outputs")}
        save_profile(manager.settings_path.parent, {}, pending)
        self.assertEqual(manager.model_paths["temp_outputs_path"], self.root / "outputs")
        self.assertEqual(manager.paths["path_outputs"], manager.DEFAULT_PATHS["path_outputs"])
        self.assertFalse((self.root / "pending outputs").exists())

    def test_linked_owned_folder_is_rejected_before_creation(self):
        real_stat = Path.lstat

        def status(path):
            if path == self.root / "outputs":
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return real_stat(path)

        with patch.object(Path, "lstat", status), self.assertRaisesRegex(OSError, "link|reparse"):
            paths.ensure_app_directory("outputs")
        self.assertFalse((self.root / "outputs").exists())

    def test_real_directory_permission_failure_propagates(self):
        real_mkdir = Path.mkdir

        def mkdir(path, *args, **kwargs):
            if path == self.root / "outputs":
                raise PermissionError(errno.EACCES, "private denied fixture", str(path))
            return real_mkdir(path, *args, **kwargs)

        with patch.object(Path, "mkdir", mkdir), self.assertRaises(PermissionError):
            paths.ensure_app_directory("outputs")

    def test_missing_custom_data_root_is_not_recreated_as_a_mount(self):
        missing = self.root / "missing mount/application"
        with patch.dict(os.environ, RF_DATA_DIR=str(missing)), self.assertRaises(FileNotFoundError):
            paths.ensure_app_directory("cache")
        self.assertFalse(missing.parent.exists())

    def test_gallery_reopens_deleted_default_index_and_cold_scans_existing_sources(self):
        manager = self.manager()
        source = self.root / "outputs/original.png"
        source.write_text("original private fixture", encoding="utf-8")
        database = self.root / "cache/images.db"
        connection = connect_index(database)
        connection.close()
        browser = SimpleNamespace(_database_path=database, path_manager=manager)
        reader = Mock(side_effect=lambda path: {"prompt": path.read_text(encoding="utf-8")})
        with index_connection(browser) as connection:
            self.assertEqual(incremental_scan(connection, [source.parent], reader).count, 1)
        self.remove(database.parent)
        with index_connection(browser) as connection:
            self.assertEqual(page_snapshot(connection, 1, "", 10)[0], 0)
            self.assertEqual(incremental_scan(connection, [source.parent], reader).parsed, 1)
        self.assertEqual(source.read_text(), "original private fixture")

    def test_gallery_releases_legacy_persistent_handle_and_reopens_new_database(self):
        manager = self.manager()
        database = self.root / "cache/images.db"
        browser = SimpleNamespace(
            _database_path=database, path_manager=manager, sql_conn=connect_index(database)
        )
        self.addCleanup(browser.sql_conn.close)
        with index_connection(browser):
            pass
        self.assertIsNone(browser.sql_conn)
        self.remove(database.parent)
        with index_connection(browser) as connection:
            self.assertEqual(page_snapshot(connection, 1, "", 10)[0], 0)

    def test_gallery_uses_factory_cache_after_profile_removal_without_changing_old_cache(self):
        manager = self.manager(path_cache=str(self.root / "custom cache"))
        original = self.root / "outputs/old.png"
        original.write_text("private original")
        old_database = self.root / "custom cache/images.db"
        browser = SimpleNamespace(_database_path=old_database, path_manager=manager)
        with index_connection(browser) as connection:
            incremental_scan(
                connection, [original.parent], lambda path: {"prompt": path.read_text()}
            )
        self.remove(self.root / "settings")
        with index_connection(browser) as connection:
            self.assertEqual(browser._database_path, self.root / "cache/images.db")
            self.assertEqual(page_snapshot(connection, 1, "", 10)[0], 0)
        old = connect_index(old_database)
        self.addCleanup(old.close)
        self.assertEqual(page_snapshot(old, 1, "", 10)[0], 1)
        self.assertEqual(original.read_text(), "private original")

    def test_recreated_default_outputs_remove_stale_rows_but_keep_archive_sources(self):
        manager = self.manager()
        outputs, archive = self.root / "outputs", self.root / "archive"
        archive.mkdir()
        (outputs / "removed.png").write_text("private removed fixture")
        original = archive / "retained.png"
        original.write_text("retained archive fixture")
        browser = SimpleNamespace(
            _database_path=self.root / "cache/images.db", path_manager=manager
        )
        with index_connection(browser) as connection:
            incremental_scan(
                connection, [outputs, archive], lambda path: {"prompt": path.read_text()}
            )
        self.remove(outputs)
        with index_connection(browser) as connection:
            self.assertEqual(page_snapshot(connection, 1, "", 10)[0], 1)
            self.assertEqual(
                connection.execute("SELECT revision FROM library_snapshot").fetchone()[0], 0
            )
        self.assertEqual(original.read_text(), "retained archive fixture")

    def test_missing_custom_index_parent_is_reported_without_recreating_it(self):
        database = self.root / "custom mount/cache/images.db"
        browser = SimpleNamespace(_database_path=database)
        with self.assertRaises(FileNotFoundError):
            with index_connection(browser):
                pass
        self.assertFalse(database.parent.exists())

    def test_gallery_index_file_does_not_follow_a_link(self):
        database = self.root / "cache/images.db"
        real_stat = Path.lstat

        def status(path):
            if path == database:
                return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
            return real_stat(path)

        with patch.object(Path, "lstat", status), self.assertRaisesRegex(OSError, "link|reparse"):
            connect_index(database)
        self.assertFalse(database.exists())

    def test_owned_profile_document_link_is_rejected_without_reading_its_target(self):
        owner = SettingsManager(defaults={"prompt": "factory"})
        before = owner.settings_path.read_bytes()
        real_stat = Path.lstat

        def status(path):
            if path == owner.settings_path:
                return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
            return real_stat(path)

        with patch.object(Path, "lstat", status):
            with self.assertRaisesRegex(OSError, "link|reparse"):
                owner.default_settings
            with self.assertRaisesRegex(OSError, "link|reparse"):
                read_profile_data(owner.settings_path)
        self.assertEqual(owner.settings_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
