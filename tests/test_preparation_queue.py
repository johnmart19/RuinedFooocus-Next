"""Waiting transfers release bounded ownership without touching active work."""

from concurrent.futures import CancelledError
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from ruinedfooocus.core.preparation import PreparationBusy, PreparationRequest, PreparationService


class PreparationQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def request(self, name):
        return PreparationRequest("https://fixture.invalid/" + name, self.root / name)

    def service(self, transport):
        service = PreparationService(
            max_workers=1, max_jobs=2, transport=transport, staging_root=self.root / "staging"
        )
        self.addCleanup(service.close)
        return service

    def test_last_waiting_subscriber_returns_capacity_and_artifact_ownership_before_active_finishes(
        self,
    ):
        entered, release = threading.Event(), threading.Event()
        transferred = []

        def transfer(request, staging, report, cancelled):
            transferred.append(request.destination.name)
            if request.destination.name == "active.bin":
                entered.set()
                self.assertTrue(release.wait(5))
                self.assertFalse(cancelled.is_set())
            path = staging / "payload.part"
            path.write_bytes(b"fixture")
            return path

        service = self.service(transfer)
        active = service.submit(self.request("active.bin"))
        self.assertTrue(entered.wait(5))
        try:
            queued = service.submit(self.request("queued.bin"))
            same = service.submit(self.request("queued.bin"))
            self.assertTrue(queued.cancel())
            with self.assertRaises(PreparationBusy):
                service.submit(self.request("retry.bin"))
            self.assertTrue(same.cancel())
            with service.artifact_lease(self.root / "queued.bin", write=True, blocking=False):
                pass
            retry = service.submit(self.request("queued.bin"))
            self.assertFalse(active.done())
            self.assertFalse(active._job.cancelled.is_set())
        finally:
            release.set()
        self.assertEqual(active.result(5).read_bytes(), b"fixture")
        self.assertEqual(retry.result(5).read_bytes(), b"fixture")
        self.assertEqual(transferred, ["active.bin", "queued.bin"])
        for subscriber in (queued, same):
            with self.assertRaises(CancelledError):
                subscriber.result(5)
        self.assertFalse(service._jobs)
        self.assertFalse(service._artifacts)

    def test_executor_submission_failure_rolls_back_destination_and_queue_capacity(self):
        def transfer(request, staging, report, cancelled):
            path = staging / "payload.part"
            path.write_bytes(b"retry")
            return path

        service = self.service(transfer)
        request = self.request("retry.bin")
        with patch.object(service._executor, "submit", side_effect=RuntimeError("executor failed")):
            with self.assertRaisesRegex(RuntimeError, "executor failed"):
                service.submit(request)
        self.assertFalse(service._jobs)
        with service.artifact_lease(request.destination, write=True, blocking=False):
            pass
        self.assertEqual(service.submit(request).result(5).read_bytes(), b"retry")


if __name__ == "__main__":
    unittest.main()
