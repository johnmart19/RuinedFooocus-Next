"""Read-only transfer state while workers own writes; no network/model reads."""

from concurrent.futures import CancelledError
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from ruinedfooocus.core.preparation import PreparationRequest, PreparationService
from ruinedfooocus.core.hub_download import _resume_offset, _validated_resume_offset


class TransferStatusTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.service = PreparationService(staging_root=self.root / "cache")
        self.addCleanup(self.service.close)
        self.request = PreparationRequest(
            "https://huggingface.co/owner/model/resolve/" + "a" * 40 + "/model.gguf",
            self.root / "models/Publisher.GGUF",
            expected_size=8,
            sha256="b" * 64,
        )

    def partial(self, *, receipt=None, data=b"part"):
        staging = self.service._staging_directory(self.request)
        staging.mkdir(parents=True)
        part, record = staging / "hub-http.part", staging / "hub-http.json"
        part.write_bytes(data)
        record.write_text(
            json.dumps(receipt or dict(url=self.request.url, etag='"' + "b" * 64 + '"', size=8)),
            encoding="utf-8",
        )
        return part, record

    def test_absent_offline_status_does_not_create_files_or_contact_network(self):
        with patch("requests.get", side_effect=AssertionError("no network")):
            status = self.service.transfer_status(self.request)
        self.assertEqual(status["state"], "absent")
        self.assertFalse(status["resumable"])
        self.assertFalse((self.root / "cache").exists())
        self.assertFalse((self.root / "models").exists())

    def test_exact_http_partial_is_conditional_resume_and_aliases_share_status(self):
        part, record = self.partial()
        before = record.read_bytes(), part.read_bytes()
        alias = PreparationRequest(
            self.request.url,
            self.root / "models/child/../Publisher.GGUF",
            expected_size=8,
            sha256="b" * 64,
        )
        status = self.service.transfer_status(alias)
        self.assertEqual(
            (status["state"], status["received_bytes"], status["resumable"]), ("partial", 4, True)
        )
        self.assertEqual(self.service.transfer_status(self.request), status)
        self.assertEqual(before, (record.read_bytes(), part.read_bytes()))

    def test_stale_digest_size_source_and_oversized_receipts_remain_untouched(self):
        part, record = self.partial()
        valid = json.loads(record.read_text())
        for changes in (
            {"url": "https://other.test/model"},
            {"etag": '"' + "c" * 64 + '"'},
            {"size": 9},
        ):
            with self.subTest(changes=changes):
                record.write_text(json.dumps(valid | changes))
                before = record.read_bytes()
                status = self.service.transfer_status(self.request)
                self.assertEqual(status["state"], "stale_partial")
                self.assertFalse(status["resumable"])
                self.assertEqual(record.read_bytes(), before)
                self.assertTrue(part.exists())
        record.write_bytes(b" " * (16 * 1024 + 1))
        self.assertIsNone(_validated_resume_offset(part, record, valid))
        self.assertTrue(record.exists())
        # The actual transfer retains its old cleanup behavior on invalid receipts.
        self.assertEqual(_resume_offset(part, record, valid), 0)
        self.assertFalse(record.exists())
        self.assertFalse(part.exists())

    def test_sdk_partial_is_visible_but_never_claimed_resumable(self):
        staging = self.service._staging_directory(self.request)
        partial = staging / ".cache/huggingface/download/model.fixture.incomplete"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"sdk")
        status = self.service.transfer_status(self.request)
        self.assertEqual((status["state"], status["received_bytes"]), ("partial_unverified", 3))
        self.assertIsNone(status["resumable"])

    def test_active_and_cancelled_job_snapshot_never_waits_for_write_lease(self):
        started, release = threading.Event(), threading.Event()

        def transfer(request, staging, report, cancelled):
            (staging / "hub-http.part").write_bytes(b"part")
            (staging / "hub-http.json").write_text(
                json.dumps(dict(url=request.url, etag='"' + request.sha256 + '"', size=8))
            )
            report(4, 8, 2)
            started.set()
            self.assertTrue(release.wait(5))
            raise CancelledError()

        self.service._transport = transfer
        self.addCleanup(release.set)
        subscription = self.service.submit(self.request)
        self.assertTrue(started.wait(5))
        with patch.object(
            self.service,
            "_staging_directory",
            side_effect=AssertionError("no filesystem while active"),
        ):
            status = self.service.transfer_status(self.request)
            self.assertEqual((status["state"], status["received_bytes"]), ("running", 4))
            conflict = PreparationRequest(
                self.request.url, self.request.destination, expected_size=9
            )
            self.assertEqual(self.service.transfer_status(conflict)["state"], "conflict")
            subscription.cancel()
            self.assertEqual(self.service.transfer_status(self.request)["state"], "cancelling")
        release.set()
        self.service.close()
        self.assertEqual(self.service.transfer_status(self.request)["state"], "partial")

    def test_staging_escape_is_rejected_before_cache_read_or_transfer(self):
        staging = self.service._staging_directory(self.request)
        from ruinedfooocus.core.cache_contract import resolved_path

        def resolved(path):
            return self.root / "outside" if Path(path) == staging else resolved_path(path)

        with patch("ruinedfooocus.core.preparation.resolved_path", side_effect=resolved):
            self.assertEqual(self.service.transfer_status(self.request)["state"], "unknown")
            with self.assertRaisesRegex(ValueError, "escapes"):
                self.service._staging_directory(self.request)
        self.assertFalse((self.root / "outside").exists())


if __name__ == "__main__":
    unittest.main()
