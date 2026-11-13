"""A preparation receipt proves verified bytes at one captured file identity."""

from concurrent.futures import CancelledError
import hashlib
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from ruinedfooocus.core.cache_contract import file_identity
from ruinedfooocus.core.preparation import PreparationRequest, PreparationService


class PreparationReceiptTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.payload = b"verified"
        self.sha256 = hashlib.sha256(self.payload).hexdigest()

    def service(self, transport=None):
        def complete(request, staging, progress, cancelled):
            result = staging / "complete.part"
            result.write_bytes(self.payload)
            return result

        service = PreparationService(
            transport=transport or complete, staging_root=self.root / "cache"
        )
        self.addCleanup(service.close)
        return service

    def request(self, *, sha256=True):
        return PreparationRequest(
            "https://fixture.test/weight",
            self.root / "model.gguf",
            expected_size=len(self.payload),
            sha256=self.sha256 if sha256 else None,
        )

    def test_new_and_existing_verified_files_expose_immutable_identity_without_caller_reads(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                if existing:
                    self.request().destination.write_bytes(self.payload)
                subscription = self.service().submit(self.request())
                result = subscription.result(5)
                expected = file_identity(result)
                with patch.object(
                    Path, "open", side_effect=AssertionError("receipt must not read model bytes")
                ):
                    receipt = subscription.verified_signature
                self.assertEqual(dict(receipt), expected)
                with self.assertRaises(TypeError):
                    receipt["size"] = 999

    def test_file_changed_after_lease_does_not_change_the_captured_certificate(self):
        subscription = self.service().submit(self.request())
        result = subscription.result(5)
        certificate = subscription.verified_signature
        result.write_bytes(b"changed after verification")
        self.assertNotEqual(dict(certificate), file_identity(result))
        self.assertIs(certificate, subscription.verified_signature)

    def test_size_only_and_failed_hash_never_receive_a_verified_certificate(self):
        subscription = self.service().submit(self.request(sha256=False))
        subscription.result(5)
        self.assertIsNone(subscription.verified_signature)
        request = PreparationRequest(
            self.request().url,
            self.request().destination,
            expected_size=len(self.payload),
            sha256="f" * 64,
        )
        failed = self.service().submit(request)
        with self.assertRaisesRegex(IOError, "SHA256"):
            failed.result(5)
        self.assertIsNone(failed.verified_signature)

    def test_pending_and_cancelled_subscribers_never_receive_a_certificate(self):
        started, release = threading.Event(), threading.Event()

        def transfer(request, staging, progress, cancelled):
            started.set()
            self.assertTrue(release.wait(5))
            raise CancelledError()

        service = self.service(transfer)
        self.addCleanup(release.set)
        subscription = service.submit(self.request())
        self.assertTrue(started.wait(5))
        self.assertIsNone(subscription.verified_signature)
        subscription.cancel()
        self.assertIsNone(subscription.verified_signature)
        release.set()


if __name__ == "__main__":
    unittest.main()
