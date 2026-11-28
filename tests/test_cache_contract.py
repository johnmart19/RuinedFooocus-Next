"""Versioned hash caches use disposable local inputs and no native runtimes."""

import builtins
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from ruinedfooocus.core.cache_contract import (
    canonical_path_key,
    configured_cache_root,
    file_identity,
    library_thumbnail_root,
    owned_library_thumbnail,
    read_record,
    record_key,
    write_record,
)
from ruinedfooocus.features.models.model_identity import cached_file_sha256, file_signature


class CacheContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rf-cache-contract-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "джерело.safetensors"
        self.source.write_bytes(b"owned fixture, not a real model")
        self.cache = self.root / "custom cache"

    def test_keys_bind_parser_version_and_complete_file_identity(self):
        identity = file_identity(self.source)
        self.assertEqual(set(identity), {"canonical_path", "size", "mtime_ns", "ctime_ns"})
        first = record_key("model-sha256", identity | {"parser_version": 1})
        self.assertEqual(
            first,
            record_key(
                "model-sha256", dict(reversed(list((identity | {"parser_version": 1}).items())))
            ),
        )
        self.assertNotEqual(first, record_key("model-sha256", identity | {"parser_version": 2}))
        self.assertNotEqual(first, record_key("other", identity | {"parser_version": 1}))

    @unittest.skipUnless(os.name == "nt", "Windows final-path prefix equivalence")
    def test_windows_extended_and_dos_paths_share_one_lease_key(self):
        from ruinedfooocus.core.preparation import PreparationService

        for plain, extended in (
            (r"C:\cache\queued.json", r"\\?\C:\cache\queued.json"),
            (r"\\server\share\cache.json", r"\\?\UNC\server\share\cache.json"),
        ):
            with patch.object(Path, "resolve", return_value=Path(plain)):
                first = PreparationService._key(plain)
            with patch.object(Path, "resolve", return_value=Path(extended)):
                second = PreparationService._key(extended)
            self.assertEqual(first, second)
            with patch.object(Path, "resolve", return_value=Path(extended)):
                self.assertEqual(canonical_path_key(extended), first)

    def test_custom_root_and_versioned_unicode_record(self):
        manager = type("Manager", (), {"model_paths": {"cache_path": self.cache}})()
        self.assertEqual(configured_cache_root(manager), self.cache.resolve())
        identity = file_identity(self.source)
        path = self.cache / "record.json"
        write_record(path, "test", identity, {"text": "власні дані"})
        self.assertEqual(read_record(path, "test", identity), {"text": "власні дані"})
        self.assertIsNone(read_record(path, "test", identity | {"size": 1}))
        self.assertIsNone(read_record(path, "other", identity))
        value = json.loads(path.read_bytes())
        value["schema_version"] = True
        path.write_text(json.dumps(value), encoding="utf-8")
        self.assertIsNone(read_record(path, "test", identity))

    def test_malformed_partial_unknown_and_oversized_records_are_cold_misses(self):
        path = self.root / "record.json"
        for data in (b"{", b"\xff", b"[]", b'{"schema_version":2}', b" " * 65537):
            path.write_bytes(data)
            self.assertIsNone(read_record(path, "test", {}))

    def test_library_thumbnail_ownership_is_limited_to_direct_digest_webp_files(self):
        directory = library_thumbnail_root(self.cache)
        self.assertEqual(directory, self.cache / "library-thumbnails")
        directory.mkdir(parents=True)
        thumbnail = directory / ("a" * 64 + ".webp")
        thumbnail.write_bytes(b"derived fixture")
        self.assertTrue(owned_library_thumbnail(thumbnail))
        self.assertFalse(owned_library_thumbnail(directory / "manual.webp"))
        self.assertFalse(owned_library_thumbnail(directory / ("b" * 64 + ".webp")))
        with patch.object(
            Path,
            "lstat",
            return_value=type("Reparse", (), {"st_mode": 0o100644, "st_file_attributes": 0x400})(),
        ):
            self.assertFalse(owned_library_thumbnail(thumbnail))

    def test_failed_atomic_replacement_retains_previous_record(self):
        path = self.root / "record.json"
        write_record(path, "test", {}, {"old": True})
        previous = path.read_bytes()
        with patch.object(Path, "replace", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                write_record(path, "test", {}, {"new": True})
        self.assertEqual(path.read_bytes(), previous)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_cold_warm_and_replacement_hashing_are_local_and_source_preserving(self):
        original = self.source.read_bytes()
        expected = hashlib.sha256(original).hexdigest().upper()
        self.assertEqual(cached_file_sha256(self.source, self.cache), expected)
        with patch("builtins.open", side_effect=AssertionError("warm hit reread weights")):
            self.assertEqual(cached_file_sha256(self.source, self.cache), expected)
        records = list(self.cache.glob("*.json"))
        self.assertEqual(len(records), 1)
        self.assertEqual(json.loads(records[0].read_bytes())["namespace"], "model-sha256")
        self.assertEqual(self.source.read_bytes(), original)
        self.source.write_bytes(b"replacement source fixture")
        changed = cached_file_sha256(self.source, self.cache)
        self.assertEqual(changed, hashlib.sha256(self.source.read_bytes()).hexdigest().upper())
        self.assertNotEqual(changed, expected)

    def test_legacy_hash_is_read_compatibly_and_retained_for_rollback(self):
        signature = file_signature(self.source)
        legacy = self.cache / (hashlib.sha256(json.dumps(signature).encode()).hexdigest() + ".json")
        legacy.parent.mkdir(parents=True)
        checksum = hashlib.sha256(self.source.read_bytes()).hexdigest().upper()
        legacy.write_text(json.dumps({"signature": signature, "sha256": checksum}))
        before = legacy.read_bytes()
        with patch("builtins.open", side_effect=AssertionError("valid legacy hit reread weights")):
            self.assertEqual(cached_file_sha256(self.source, self.cache), checksum)
        self.assertEqual(legacy.read_bytes(), before)
        self.assertEqual(len(list(self.cache.glob("*.json"))), 2)

    def test_two_hash_callers_share_one_writer_and_cache_failure_keeps_result(self):
        entered, release = threading.Event(), threading.Event()
        original_open = builtins.open
        reads = []

        @contextmanager
        def source_reader(path, mode="r", *args, **kwargs):
            if Path(path) == self.source and mode == "rb":
                reads.append(str(path))
                entered.set()
                self.assertTrue(release.wait(5))
            with original_open(path, mode, *args, **kwargs) as stream:
                yield stream

        with ThreadPoolExecutor(max_workers=2) as executor, patch("builtins.open", source_reader):
            first = executor.submit(cached_file_sha256, self.source, self.cache)
            self.assertTrue(entered.wait(5))
            second = executor.submit(cached_file_sha256, self.source, self.cache)
            release.set()
            self.assertEqual(first.result(5), second.result(5))
        self.assertEqual(len(reads), 1)
        other = self.root / "unwritable cache"
        with patch(
            "ruinedfooocus.features.models.model_identity.write_record", side_effect=OSError("full")
        ):
            result = cached_file_sha256(self.source, other)
        self.assertEqual(result, hashlib.sha256(self.source.read_bytes()).hexdigest().upper())


if __name__ == "__main__":
    unittest.main()
