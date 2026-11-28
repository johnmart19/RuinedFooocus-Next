"""Real SQLite transactions and deterministic races, without app imports or media inference."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.features.gallery.index_store import (
    connect_index,
    ensure_schema,
    incremental_scan,
    page_snapshot,
)


class GalleryIndexTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rf-incremental-library-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.outputs = self.root / "власні outputs"
        self.outputs.mkdir()
        self.database = self.root / "custom cache/images.db"
        self.connection = connect_index(self.database)
        self.addCleanup(self.connection.close)
        self.metadata = Mock(side_effect=lambda path: {"prompt": path.read_text()})

    def source(self, name, value):
        path = self.outputs / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(value)
        return path

    def scan(self, connection=None, **kwargs):
        return incremental_scan(
            connection or self.connection, [self.outputs], self.metadata, **kwargs
        )

    def rows(self, connection=None):
        return (
            (connection or self.connection)
            .execute("SELECT fullpath,path,json FROM images ORDER BY fullpath")
            .fetchall()
        )

    def test_warm_scan_reuses_metadata_and_parses_only_new_or_replaced_sources(self):
        first = self.source("a.png", "first")
        second = self.source("b.png", "second")
        cold = self.scan()
        warm = self.scan()
        self.assertEqual((cold.count, cold.parsed, cold.reused), (2, 2, 0))
        self.assertEqual((warm.count, warm.parsed, warm.reused), (2, 0, 2))
        self.assertEqual(self.metadata.call_count, 2)
        first.write_text("replacement with a different identity")
        second.unlink()
        self.source("child/c.png", "new source")
        changed = self.scan()
        self.assertEqual(
            (changed.count, changed.parsed, changed.reused, changed.removed), (2, 2, 0, 1)
        )
        self.assertEqual(self.metadata.call_count, 4)
        self.assertEqual(first.read_text(), "replacement with a different identity")

    def test_initial_scan_populates_once_and_retains_empty_completed_snapshots(self):
        result = self.scan(only_if_uninitialized=True)
        self.assertIsNotNone(result)
        self.assertEqual(result.count, 0)
        self.source("first.png", "first")
        with patch(
            "ruinedfooocus.features.gallery.index_store.os.walk",
            side_effect=AssertionError("initialized picker scanned output folders"),
        ):
            self.assertIsNone(self.scan(only_if_uninitialized=True))
        # Explicit Refresh still scans after an initial snapshot exists.
        self.assertEqual(self.scan().count, 1)

    def test_cancelled_initial_scan_does_not_mark_initialized_and_retry_populates(self):
        self.source("first.png", "first")
        cancelled = threading.Event()
        with self.assertRaises(CancelledError):
            self.scan(
                only_if_uninitialized=True,
                cancelled=cancelled,
                progress=lambda *args: cancelled.set(),
            )
        self.assertEqual(self.rows(), [])
        self.assertEqual(
            self.connection.execute("SELECT revision FROM library_snapshot").fetchone()[0], 0
        )
        self.assertEqual(self.scan(only_if_uninitialized=True).count, 1)

    def test_concurrent_initial_scans_publish_one_snapshot(self):
        self.source("first.png", "first")
        entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def metadata(path):
            entered.set()
            self.assertTrue(release.wait(5))
            return {"prompt": path.read_text()}

        self.metadata.side_effect = metadata

        def initialize(started=None):
            connection = sqlite3.connect(self.database)
            try:
                if started is not None:
                    started.set()
                return self.scan(connection, only_if_uninitialized=True)
            finally:
                connection.close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(initialize)
            self.assertTrue(entered.wait(5))
            second = executor.submit(initialize, second_started)
            try:
                self.assertTrue(second_started.wait(5))
                self.assertFalse(second.done())
            finally:
                release.set()
            self.assertEqual(first.result(5).count, 1)
            self.assertIsNone(second.result(5))
        self.assertEqual(self.metadata.call_count, 1)
        self.assertEqual(
            self.connection.execute("SELECT revision FROM library_snapshot").fetchone()[0], 1
        )

    def test_readers_keep_complete_previous_snapshot_while_scan_is_blocked(self):
        self.source("a.png", "old prompt")
        self.scan()
        self.source("b.png", "new prompt")
        entered, release = threading.Event(), threading.Event()

        def blocked(path):
            entered.set()
            self.assertTrue(release.wait(5))
            return {"prompt": "new prompt"}

        self.metadata.side_effect = blocked
        with ThreadPoolExecutor(max_workers=1) as executor:

            def write():
                connection = sqlite3.connect(self.database)
                try:
                    return self.scan(connection)
                finally:
                    connection.close()

            future = executor.submit(write)
            self.assertTrue(entered.wait(5))
            try:
                count, _, _, rows = page_snapshot(self.connection, 1, "", 10)
                self.assertEqual((count, len(rows)), (1, 1))
                self.assertEqual(page_snapshot(self.connection, 1, "new prompt", 10)[0], 0)
            finally:
                release.set()
            self.assertEqual(future.result(5).count, 2)
        self.assertEqual(page_snapshot(self.connection, 1, "new prompt", 10)[0], 1)

    def test_cancellation_rolls_back_changed_added_and_removed_rows_and_retry_works(self):
        old = self.source("old.png", "old")
        self.scan()
        before = self.rows()
        old.unlink()
        self.source("new.png", "new")
        cancelled = threading.Event()
        with self.assertRaises(CancelledError):
            self.scan(cancelled=cancelled, progress=lambda *args: cancelled.set())
        self.assertEqual(self.rows(), before)
        result = self.scan()
        self.assertEqual((result.count, result.parsed, result.removed), (1, 1, 1))

    def test_parse_failure_missing_root_changed_source_and_failed_commit_preserve_snapshot(self):
        first = self.source("first.png", "previous")
        self.scan()
        before = self.rows()
        second = self.source("second.png", "new")
        self.metadata.side_effect = OSError("unreadable source")
        with self.assertRaisesRegex(OSError, "unreadable"):
            self.scan()
        self.assertEqual(self.rows(), before)
        self.metadata.side_effect = lambda path: {"prompt": path.read_text()}
        with self.assertRaisesRegex(OSError, "unavailable"):
            incremental_scan(self.connection, [self.root / "missing"], self.metadata)
        self.assertEqual(self.rows(), before)
        with self.assertRaisesRegex(OSError, "disk full"):
            self.scan(publish=Mock(side_effect=OSError("disk full")))
        self.assertEqual(self.rows(), before)

        def changing(path):
            path.write_text("changed during metadata read")
            return {"prompt": "stale"}

        self.metadata.side_effect = changing
        with self.assertRaisesRegex(OSError, "changed while"):
            self.scan()
        self.assertEqual(self.rows(), before)
        self.assertEqual(first.read_text(), "previous")
        self.assertEqual(second.read_text(), "changed during metadata read")

    def test_legacy_schema_keeps_old_metadata_and_unknown_newer_schema_is_untouched(self):
        connection = sqlite3.connect(":memory:")
        self.addCleanup(connection.close)
        connection.execute("CREATE TABLE images(fullpath text,path text,json text)")
        connection.execute("INSERT INTO images VALUES('old.png','old.png','{}')")
        connection.commit()
        ensure_schema(connection)
        self.assertEqual(
            connection.execute("SELECT fullpath FROM images").fetchall(), [("old.png",)]
        )
        connection.execute("PRAGMA user_version=99")
        with self.assertRaisesRegex(ValueError, "matching application"):
            ensure_schema(connection)
        self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 99)

    def test_page_snapshot_never_stats_all_indexed_paths(self):
        for index in range(100):
            self.source(f"{index:04d}.png", f"prompt {index}")
        self.scan()
        with patch.object(
            Path, "stat", side_effect=AssertionError("paging performed filesystem I/O")
        ):
            count, pages, page, rows = page_snapshot(self.connection, 8, "", 7)
        self.assertEqual((count, pages, page, len(rows)), (100, 15, 8, 7))
        self.assertTrue(all(len(row) == 5 for row in rows))

    def test_page_size_three_boundaries_typed_entry_and_rescan_deletion(self):
        for total in (0, 1, 3, 4, 7):
            with self.subTest(total=total):
                for path in self.outputs.glob("*.png") if self.outputs.exists() else ():
                    path.unlink()
                for index in range(total):
                    self.source(f"{index}.png", f"prompt {index}")
                if not self.outputs.exists():
                    self.outputs.mkdir()
                self.scan()
                expected = max(1, (total + 2) // 3)
                for requested, page in (
                    ("1", 1),
                    ("99", expected),
                    (-1, 1),
                    (None, 1),
                    (float("nan"), 1),
                    ("invalid", 1),
                ):
                    count, pages, actual, rows = page_snapshot(self.connection, requested, "", 3)
                    self.assertEqual((count, pages, actual), (total, expected, page))
                    self.assertLessEqual(len(rows), 3)
                if total:
                    (self.outputs / f"{total - 1}.png").unlink()
                    self.scan()
                    self.assertEqual(page_snapshot(self.connection, "99", "", 3)[0], total - 1)

    def test_aliases_and_overlapping_roots_keep_first_root_relative_identity(self):
        first = self.source("child/one.png", "one")
        self.source("two.png", "two")
        result = incremental_scan(
            self.connection, [first.parent, self.outputs, first.parent.resolve()], self.metadata
        )
        self.assertEqual((result.count, result.parsed), (2, 2))
        child = next(row for row in self.rows() if Path(row[0]).name == "one.png")
        self.assertEqual((child[1], json.loads(child[2])["file_path"]), ("one.png", "one.png"))


if __name__ == "__main__":
    unittest.main()
