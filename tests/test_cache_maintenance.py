"""Maintenance may remove only recognized, inactive derived fixture records."""

from contextlib import closing
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.core import cache_maintenance as maintenance
from ruinedfooocus.core.cache_contract import record_key, write_record
from ruinedfooocus.core.preparation import PreparationRequest, PreparationService
from ruinedfooocus.features.gallery.index_store import connect_index
from ruinedfooocus.features.models.model_identity import cached_file_sha256


class CacheMaintenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rf-maintenance-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        environment = patch.dict(os.environ, RF_DATA_DIR=str(self.root))
        environment.start()
        self.addCleanup(environment.stop)
        self.cache = self.root / "окремий cache"
        self.manager = SimpleNamespace(
            model_paths={
                "cache_path": self.cache,
                "modelfile_path": [self.root / "models"],
                "temp_outputs_path": self.root / "outputs",
            }
        )
        self.source = self.root / "models/source.safetensors"
        self.source.parent.mkdir()
        self.source.write_bytes(b"owned input, not weights")

    def record(self):
        checksum = cached_file_sha256(self.source, self.cache / "hashes")
        return next((self.cache / "hashes").glob("*.json")), checksum

    def test_targeted_clear_reuses_and_regenerates_hash_without_touching_source_or_legacy(self):
        record, expected = self.record()
        legacy = record.parent / ("a" * 64 + ".json")
        legacy.write_text(json.dumps({"signature": ["old", 1, 2, 3], "sha256": "B" * 64}))
        malformed = record.parent / "manual.json"
        malformed.write_text("{")
        original = self.source.read_bytes()
        result = maintenance.clear_category("model-hashes", self.manager)
        self.assertEqual((result["removed"], result["skipped"], result["errors"]), (1, 2, []))
        self.assertFalse(record.exists())
        self.assertTrue(legacy.exists())
        self.assertTrue(malformed.exists())
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(cached_file_sha256(self.source, self.cache / "hashes"), expected)

    def test_active_and_queued_artifact_consumers_are_skipped_without_waiting(self):
        record, _ = self.record()
        entered, release = threading.Event(), threading.Event()

        def transfer(request, staging, report, cancelled):
            entered.set()
            self.assertTrue(release.wait(5))
            path = staging / "artifact.part"
            path.write_bytes(b"fixture")
            return path

        service = PreparationService(
            max_workers=1, max_jobs=2, transport=transfer, staging_root=self.root / "staging"
        )
        self.addCleanup(service.close)
        with service.artifact_lease(record):
            self.assertEqual(
                maintenance.clear_category("model-hashes", self.manager, service=service)[
                    "skipped"
                ],
                1,
            )
        active = service.submit(
            PreparationRequest("https://fixture.test/first", self.root / "first.bin")
        )
        self.assertTrue(entered.wait(5))
        pending = service.submit(PreparationRequest("https://fixture.test/existing", record))
        try:
            self.assertEqual(
                maintenance.clear_category("model-hashes", self.manager, service=service)[
                    "skipped"
                ],
                1,
            )
            self.assertTrue(record.exists())
        finally:
            release.set()
        active.result(5), pending.result(5)

    def test_index_reset_is_transactional_and_skips_an_active_scan_lease(self):
        path = self.cache / "images.db"
        connection = connect_index(path)
        self.addCleanup(connection.close)
        connection.execute(
            "INSERT INTO images(fullpath,path,json) VALUES(?,?,?)",
            (str(self.source), "source", "{}"),
        )
        connection.commit()
        service = PreparationService(staging_root=self.root / "staging")
        self.addCleanup(service.close)
        with service.artifact_lease(path):
            result = maintenance.clear_category("library-index", self.manager, service=service)
            self.assertEqual((result["removed"], result["skipped"]), (0, 1))
        result = maintenance.clear_category("library-index", self.manager, service=service)
        self.assertEqual((result["removed"], result["bytes"], result["errors"]), (1, 0, []))
        self.assertEqual(connection.execute("SELECT count(*) FROM images").fetchone()[0], 0)
        self.assertTrue(path.is_file())
        self.assertTrue(self.source.is_file())

    def test_thumbnail_inventory_and_clear_preserve_unowned_nested_and_source_files(self):
        directory = self.cache / "library-thumbnails"
        directory.mkdir(parents=True)
        thumbnail = directory / ("a" * 64 + ".webp")
        thumbnail.write_bytes(b"derived thumbnail")
        retained = [
            directory / "manual.webp",
            directory / ("b" * 64 + ".webp.123.tmp"),
            directory / "nested" / ("c" * 64 + ".webp"),
        ]
        for path in retained:
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"retained fixture")
        row = next(
            row
            for row in maintenance.maintenance_categories(self.manager)
            if row["id"] == "library-thumbnails"
        )
        self.assertEqual((row["files"], row["protected"]), (4, False))
        self.assertEqual(row["bytes"], len(b"derived thumbnail") + 3 * len(b"retained fixture"))
        result = maintenance.clear_category("library-thumbnails", self.manager)
        self.assertEqual((result["removed"], result["skipped"], result["errors"]), (1, 3, []))
        self.assertEqual(result["bytes"], len(b"derived thumbnail"))
        self.assertFalse(thumbnail.exists())
        self.assertTrue(all(path.is_file() for path in retained))
        self.assertEqual(self.source.read_bytes(), b"owned input, not weights")

    def test_thumbnail_clear_skips_active_copy_and_protected_cache_alias(self):
        directory = self.cache / "library-thumbnails"
        directory.mkdir(parents=True)
        thumbnail = directory / ("a" * 64 + ".webp")
        thumbnail.write_bytes(b"derived thumbnail")
        service = PreparationService(staging_root=self.root / "staging")
        self.addCleanup(service.close)
        with service.artifact_lease(directory):
            result = maintenance.clear_category("library-thumbnails", self.manager, service=service)
        self.assertEqual((result["removed"], result["skipped"], result["errors"]), (0, 1, []))
        self.assertTrue(thumbnail.is_file())
        self.manager.model_paths["cache_path"] = self.source.parent
        protected = self.source.parent / "library-thumbnails" / thumbnail.name
        protected.parent.mkdir()
        protected.write_bytes(b"retained in models")
        result = maintenance.clear_category("library-thumbnails", self.manager)
        self.assertEqual((result["removed"], result["skipped"]), (0, 1))
        self.assertTrue(protected.is_file())

    def test_thumbnail_clear_is_bounded_and_reports_failed_deletions(self):
        directory = self.cache / "library-thumbnails"
        directory.mkdir(parents=True)
        for number in range(3):
            (directory / (f"{number:064x}" + ".webp")).write_bytes(b"derived thumbnail")
        with patch.object(maintenance, "MAX_ENTRIES", 2):
            result = maintenance.clear_category("library-thumbnails", self.manager)
        self.assertEqual((result["removed"], result["errors"]), (2, []))
        with patch.object(Path, "unlink", side_effect=OSError("fixture permission denied")):
            result = maintenance.clear_category("library-thumbnails", self.manager)
        self.assertEqual(result["removed"], 0)
        self.assertIn("permission denied", result["errors"][0])
        self.assertEqual(len(list(directory.iterdir())), 1)

    def test_future_schema_and_protected_categories_are_retained(self):
        path = self.cache / "images.db"
        connection = connect_index(path)
        connection.execute("PRAGMA user_version=99")
        connection.close()
        self.assertEqual(maintenance.clear_category("library-index", self.manager)["skipped"], 1)
        with closing(sqlite3.connect(path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 99)
        for category in ("inputs", "pending", "artwork", "runtimes", "../models", "missing"):
            with self.assertRaises(ValueError):
                maintenance.clear_category(category, self.manager)

    def test_workflow_caches_are_counted_without_opening_files_and_cannot_be_cleared(self):
        categories = (
            "model-sources",
            "catalogue-previews",
            "native-image-previews",
            "native-image-exports",
        )
        original = self.source.read_bytes()
        fixtures = {}
        for number, category in enumerate(categories):
            for root in (self.cache, self.root / "cache"):
                path = root / category / "nested" / "retained.data"
                path.parent.mkdir(parents=True)
                payload = f"protected workflow result {number}".encode()
                path.write_bytes(payload)
                fixtures[path] = payload
        with (
            patch("builtins.open", side_effect=AssertionError("Inventory must only stat files")),
            patch.object(
                Path, "open", side_effect=AssertionError("Inventory must only stat files")
            ),
        ):
            rows = {row["id"]: row for row in maintenance.maintenance_categories(self.manager)}
            for category in categories:
                row = rows[category]
                self.assertEqual(row["files"], 2)
                self.assertEqual(
                    row["bytes"],
                    sum(len(value) for path, value in fixtures.items() if category in path.parts),
                )
                self.assertTrue(row["protected"])
                self.assertTrue(row["bounded"])
                with self.assertRaisesRegex(ValueError, "derived cache category"):
                    maintenance.clear_category(category, self.manager)
        for path, payload in fixtures.items():
            self.assertEqual(path.read_bytes(), payload)
        self.assertEqual(self.source.read_bytes(), original)

    def test_workflow_cache_inventory_deduplicates_configured_and_data_root(self):
        self.manager.model_paths["cache_path"] = self.root / "cache"
        path = self.root / "cache/native-image-exports/current-export.exr"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"user export result")
        row = next(
            row
            for row in maintenance.maintenance_categories(self.manager)
            if row["id"] == "native-image-exports"
        )
        self.assertEqual(
            (row["files"], row["bytes"], row["protected"]), (1, path.stat().st_size, True)
        )

    def test_cache_root_alias_into_models_is_not_a_cleanup_target(self):
        self.manager.model_paths["cache_path"] = self.source.parent
        path = self.source.parent / "hashes"
        cached_file_sha256(self.source, path)
        result = maintenance.clear_category("model-hashes", self.manager)
        self.assertEqual((result["removed"], result["skipped"]), (0, 1))
        self.assertTrue(list(path.glob("*.json")))

    def test_reparse_file_and_internal_symlink_are_rejected(self):
        record, _ = self.record()
        linked = maintenance._linked
        with patch.object(
            maintenance, "_linked", side_effect=lambda path: Path(path) == record or linked(path)
        ):
            result = maintenance.clear_category("model-hashes", self.manager)
        self.assertEqual((result["removed"], result["skipped"]), (0, 1))
        self.assertTrue(record.exists())
        alias = record.parent / ("b" * 64 + ".json")
        try:
            alias.symlink_to(record)
        except OSError:
            return  # Windows without symlink privilege still executes the reparse guard above.
        self.assertFalse(maintenance._safe_path(alias, self.cache, []))
        self.assertTrue(record.exists())

    def test_symlink_directory_escape_is_not_traversed_or_cleared(self):
        other = self.root / "outside"
        other.mkdir()
        value = {
            "canonical_path": str(self.source),
            "size": 1,
            "mtime_ns": 1,
            "ctime_ns": 1,
            "parser_version": 1,
        }
        record = other / (record_key("model-sha256", value) + ".json")
        write_record(record, "model-sha256", value, {"sha256": "A" * 64})
        self.cache.mkdir()
        try:
            (self.cache / "hashes").symlink_to(other, target_is_directory=True)
        except OSError:
            self.skipTest("Host does not permit owned symlink fixtures")
        result = maintenance.clear_category("model-hashes", self.manager)
        self.assertEqual(result["removed"], 0)
        self.assertTrue(record.exists())

    def test_startup_marker_reset_and_lazy_sizes_preserve_inputs_and_deduplicate_roots(self):
        marker = self.root / "cache/dependency-checks.json"
        marker.parent.mkdir()
        marker.write_text(json.dumps({"packages": "a" * 64}))
        upload = self.root / "cache/uploads/original.png"
        upload.parent.mkdir()
        upload.write_bytes(b"private input")
        self.assertEqual(maintenance.clear_category("startup-checks", self.manager)["removed"], 1)
        self.assertTrue(upload.exists())
        self.manager.model_paths["cache_path"] = marker.parent
        self.assertEqual(len(maintenance.cache_roots(self.manager)), 1)
        rows = maintenance.maintenance_categories(self.manager)
        inputs = next(row for row in rows if row["id"] == "inputs")
        self.assertEqual(
            (inputs["files"], inputs["bytes"], inputs["protected"]),
            (1, len(b"private input"), True),
        )

    def test_failed_delete_reports_error_without_changing_other_entries(self):
        record, _ = self.record()
        with patch.object(Path, "unlink", side_effect=OSError("fixture permission denied")):
            result = maintenance.clear_category("model-hashes", self.manager)
        self.assertEqual(result["removed"], 0)
        self.assertIn("permission denied", result["errors"][0])
        self.assertTrue(record.exists())

    def test_real_gradio_maintenance_is_lazy_bounded_and_only_offers_derived_categories(self):
        import gradio as gr
        from ruinedfooocus.app.preparation_admission import PreparationAdmission
        from ruinedfooocus.ui import cache_maintenance as cache_ui

        self.record()
        with patch.object(
            cache_ui, "maintenance_categories", wraps=maintenance.maintenance_categories
        ) as inventory:
            with gr.Blocks() as app:
                cache_ui.create_cache_maintenance(self.manager)
            inventory.assert_not_called()
            functions = {
                function.fn.__name__: function for function in app.fns.values() if function.fn
            }
            dropdown = next(
                block for block in app.blocks.values() if isinstance(block, gr.Dropdown)
            )
            self.assertEqual(
                {value for _, value in dropdown.choices},
                {"model-hashes", "library-index", "library-thumbnails", "startup-checks"},
            )
            for function in functions.values():
                self.assertEqual(
                    (function.concurrency_id, function.concurrency_limit), ("cache_maintenance", 1)
                )
                self.assertEqual(len(function.show_progress_on), 1)
            pool = PreparationAdmission()
            try:
                with patch.object(cache_ui, "preparation_admission", pool):
                    rows = asyncio.run(functions["sizes"].fn())
                    self.assertEqual(len(rows), 12)
                    status = asyncio.run(functions["clear_selected"].fn("model-hashes"))
                self.assertIn("Cleared 1", status)
                self.assertEqual(list((self.cache / "hashes").glob("*.json")), [])
                self.assertTrue(self.source.is_file())
            finally:
                pool.close()


if __name__ == "__main__":
    unittest.main()
