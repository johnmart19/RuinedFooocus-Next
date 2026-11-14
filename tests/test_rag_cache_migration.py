"""Private pinned fixtures prove cache purging cannot remove the canonical model."""

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.features.chat import rag_preparation as rag
from tools import migrate_rag_model as migration


class RagCacheMigrationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(patch.dict(os.environ, {"RF_DATA_DIR": str(self.root)}))
        self.source = self.root / "cache/rag" / rag.MODEL_REVISION
        self.target = self.root / "models/rag_embeddings" / rag.MODEL_REVISION
        self.files = []
        for name, _, _ in rag.MODEL_FILES:
            payload = ("private-fixture:" + name).encode()
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            self.files.append((name, len(payload), hashlib.sha256(payload).hexdigest()))
        self.enterContext(patch.object(migration, "MODEL_FILES", tuple(self.files)))
        self.enterContext(patch.object(rag, "MODEL_FILES", tuple(self.files)))
        self.snapshot = self.enterContext(
            patch.object(
                rag, "snapshot_download", side_effect=rag.LocalEntryNotFoundError("absent")
            )
        )

    def test_default_and_explicit_destination_and_dry_run_do_not_copy(self):
        report = migration.migrate(self.root)
        self.assertEqual(report["status"], "ready")
        self.assertFalse(self.target.exists())
        self.assertEqual(rag.capture_rag([["text", "source"]]).model, self.target)
        explicit = self.root / "explicit-model"
        with patch.object(
            rag, "snapshot_download", side_effect=rag.LocalEntryNotFoundError("absent")
        ):
            self.assertEqual(
                rag.capture_rag([["text", "source"]], cache_root=explicit).model, explicit
            )

    def test_copy_then_private_cache_purge_retains_complete_offline_model(self):
        with self.assertRaisesRegex(ValueError, "idle"):
            migration.migrate(self.root, apply=True)
        originals = {name: (self.source / name).read_bytes() for name, _, _ in self.files}
        report = migration.migrate(self.root, apply=True, idle=True)
        self.assertEqual(len(report["copied"]), 8)
        for name, payload in originals.items():
            self.assertEqual((self.source / name).read_bytes(), payload)
            self.assertEqual((self.target / name).read_bytes(), payload)
        self.assertEqual(
            migration.migrate(self.root, apply=True, idle=True)["status"], "already-present"
        )
        shutil.rmtree(self.root / "cache")  # Only this test's private fixture.
        prepared = rag.capture_rag([["text", "source"]], offline=True)
        self.assertTrue(prepared.present)
        self.assertEqual(prepared.model, self.target)
        self.assertIs(rag.prepare_rag(prepared), prepared)
        self.snapshot.assert_not_called()

    def test_missing_bad_or_conflicting_files_never_overwrite(self):
        missing = self.source / self.files[-1][0]
        missing.unlink()
        self.assertEqual(
            migration.migrate(self.root, apply=True, idle=True)["status"], "missing-source"
        )
        self.assertFalse(self.target.exists())
        missing.write_bytes(("private-fixture:" + self.files[-1][0]).encode())
        conflict = self.target / self.files[0][0]
        conflict.parent.mkdir(parents=True)
        conflict.write_bytes(b"user data")
        with self.assertRaisesRegex(OSError, "conflicting"):
            migration.migrate(self.root, apply=True, idle=True)
        self.assertEqual(conflict.read_bytes(), b"user data")

    def test_bad_same_size_source_digest_is_rejected_before_publication(self):
        source = self.source / self.files[-1][0]
        source.write_bytes(b"x" * source.stat().st_size)
        with self.assertRaisesRegex(OSError, "SHA256"):
            migration.migrate(self.root, apply=True, idle=True)
        self.assertFalse(self.target.exists())

    def test_publication_race_preserves_the_other_writer(self):
        original_link = os.link

        def race(source, target):
            Path(target).write_bytes(b"other writer")
            return original_link(source, target)

        with patch.object(migration.os, "link", side_effect=race):
            with self.assertRaises(FileExistsError):
                migration.migrate(self.root, apply=True, idle=True)
        self.assertEqual((self.target / self.files[0][0]).read_bytes(), b"other writer")
        self.assertTrue(all((self.source / name).is_file() for name, _, _ in self.files))
        self.assertFalse(list(self.target.rglob(".rag-copy-*")))

    def test_parent_reparse_is_rejected_before_any_write(self):
        from types import SimpleNamespace

        original = Path.lstat

        def lstat(path, *args, **kwargs):
            if path == self.root / "cache":
                return SimpleNamespace(st_mode=0, st_file_attributes=0x400)
            return original(path, *args, **kwargs)

        with patch.object(Path, "lstat", lstat):
            with self.assertRaisesRegex(ValueError, "reparse"):
                migration.migrate(self.root, apply=True, idle=True)
        self.assertFalse(self.target.exists())
