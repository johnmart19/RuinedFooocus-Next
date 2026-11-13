"""Deterministic preparation ownership tests; no real network or compute."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from ruinedfooocus.core.preparation import (
    PreparationBusy,
    PreparationConflict,
    PreparationRequest,
    PreparationService,
    transfer_file,
)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def service(self, transport, **options):
        service = PreparationService(
            transport=transport, staging_root=self.root / "cache", **options
        )
        self.addCleanup(service.close)
        return service

    def request(self, name="model.gguf", **options):
        return PreparationRequest("https://fixture.test/model", self.root / name, **options)

    def test_operational_destination_preserves_requested_filename_case(self):
        request = self.request("Publisher-MiXeD.GGUF")
        self.assertEqual(request.destination.name, "Publisher-MiXeD.GGUF")
        service = self.service(self.complete)
        result = service.submit(request).result(5)
        self.assertEqual(result.name, "Publisher-MiXeD.GGUF")
        self.assertIn("Publisher-MiXeD.GGUF", [path.name for path in self.root.iterdir()])

    def test_nonblocking_cleanup_skips_active_readers_and_queued_consumers(self):
        entered, release = threading.Event(), threading.Event()

        def transfer(*arguments):
            entered.set()
            self.assertTrue(release.wait(5))
            return self.complete(*arguments)

        service = self.service(transfer, max_workers=1, max_jobs=2)
        first = service.submit(self.request("active.gguf"))
        self.assertTrue(entered.wait(5))
        second = service.submit(self.request("queued.gguf"))
        try:
            for name in ("active.gguf", "queued.gguf"):
                with self.assertRaises(PreparationBusy):
                    with service.artifact_lease(self.root / name, write=True, blocking=False):
                        self.fail("Cleanup must not acquire a pending artifact")
            with service.artifact_lease(self.root / "derived.json"):
                with self.assertRaises(PreparationBusy):
                    with service.artifact_lease(
                        self.root / "derived.json", write=True, blocking=False
                    ):
                        self.fail("Cleanup must not acquire an active reader")
            with service.artifact_lease(self.root / "unrelated.json", write=True, blocking=False):
                pass
        finally:
            release.set()
        self.assertTrue(first.result(5).is_file())
        self.assertTrue(second.result(5).is_file())
        # Failed acquisition must leave no queued-writer state behind.
        with service.artifact_lease(self.root / "derived.json", write=True, blocking=False):
            pass

    @staticmethod
    def complete(request, staging, report, cancelled):
        path = staging / "artifact.part"
        path.write_bytes(b"complete")
        return path

    def test_alias_subscribers_share_transfer_and_cancellation_is_scoped(self):
        started, release = threading.Event(), threading.Event()
        calls, first_progress, second_progress = [], [], []

        def transfer(request, staging, report, cancelled):
            calls.append(request)
            report(0, 8, 0)
            started.set()
            self.assertTrue(release.wait(5))
            self.assertFalse(cancelled.is_set())
            report(4, 8, 1)
            return self.complete(request, staging, report, cancelled)

        service = self.service(transfer)
        first = service.submit(self.request(), lambda *event: first_progress.append(event))
        self.assertTrue(started.wait(5))
        alias = PreparationRequest(first.request.url, self.root / "child/../model.gguf")
        second = service.submit(alias, lambda *event: second_progress.append(event))
        self.assertTrue(first.cancel())
        release.set()
        self.assertEqual(second.result(5).read_bytes(), b"complete")
        with self.assertRaises(CancelledError):
            first.result(5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first_progress, [(0, 8, 0)])
        self.assertEqual(second_progress[-1], (8, 8, 0))

    def test_independent_destinations_overlap_and_queue_is_bounded(self):
        barrier = threading.Barrier(3)
        release = threading.Event()

        def transfer(*arguments):
            barrier.wait(5)
            self.assertTrue(release.wait(5))
            return self.complete(*arguments)

        service = self.service(transfer, max_workers=2, max_jobs=2)
        first = service.submit(self.request("a.gguf"))
        second = service.submit(self.request("b.gguf"))
        barrier.wait(5)
        with self.assertRaises(PreparationBusy):
            service.submit(self.request("c.gguf"))
        with self.assertRaises(PreparationConflict):
            service.submit(PreparationRequest("https://other.test", first.request.destination))
        release.set()
        self.assertTrue(first.result(5).is_file())
        self.assertTrue(second.result(5).is_file())

    def test_transfer_failure_reaches_both_subscribers_and_retry_releases_capacity(self):
        started, release = threading.Event(), threading.Event()

        def transfer(*arguments):
            started.set()
            self.assertTrue(release.wait(5))
            raise OSError("network failed")

        service = self.service(transfer, max_workers=1, max_jobs=1)
        first = service.submit(self.request())
        self.assertTrue(started.wait(5))
        second = service.submit(self.request())
        release.set()
        for subscriber in (first, second):
            with self.assertRaisesRegex(OSError, "network failed"):
                subscriber.result(5)
        self.assertFalse(first.request.destination.exists())
        service._transport = self.complete
        self.assertEqual(service.submit(self.request()).result(5).read_bytes(), b"complete")

    def test_last_subscriber_cancels_transfer_without_publishing_partial_file(self):
        started, release = threading.Event(), threading.Event()

        def transfer(request, staging, report, cancelled):
            path = staging / "artifact.part"
            path.write_bytes(b"partial")
            started.set()
            self.assertTrue(release.wait(5))
            report(7, 8, 1)
            return path

        service = self.service(transfer)
        subscriber = service.submit(self.request())
        self.assertTrue(started.wait(5))
        self.assertFalse(subscriber.request.destination.exists())
        subscriber.cancel()
        with self.assertRaises(PreparationConflict):
            service.submit(self.request())
        release.set()
        service.close()
        self.assertFalse(subscriber.request.destination.exists())

    def test_hash_size_and_empty_failures_never_publish(self):
        service = self.service(self.complete)
        for options in ({"sha256": "0" * 64}, {"expected_size": 20}):
            with self.subTest(options=options), self.assertRaises(OSError):
                service.submit(self.request(**options)).result(5)
            self.assertFalse(self.request().destination.exists())
        good = self.request(sha256=hashlib.sha256(b"complete").hexdigest(), expected_size=8)
        self.assertEqual(service.submit(good).result(5), good.destination)

        def empty(request, staging, report, cancelled):
            path = staging / "empty.part"
            path.touch()
            return path

        service._transport = empty
        with self.assertRaises(OSError):
            service.submit(self.request("empty.gguf")).result(5)
        self.assertFalse((self.root / "empty.gguf").exists())

    def test_preexisting_user_file_is_verified_and_never_replaced(self):
        request = self.request(sha256="0" * 64)
        request.destination.write_bytes(b"user model")

        def forbidden(*arguments):
            self.fail("existing model must not download")

        service = self.service(forbidden)
        with self.assertRaises(OSError):
            service.submit(request).result(5)
        self.assertEqual(request.destination.read_bytes(), b"user model")

    def test_disk_publication_failure_has_no_final_and_can_retry(self):
        import os

        service = self.service(self.complete)
        publication = "os.rename" if os.name == "nt" else "os.link"
        with patch(publication, side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                service.submit(self.request()).result(5)
        self.assertFalse(self.request().destination.exists())
        self.assertEqual(service.submit(self.request()).result(5).read_bytes(), b"complete")

    def test_cancelling_a_job_waiting_for_loaded_artifact_does_not_transfer(self):
        def forbidden(*arguments):
            self.fail("cancelled job must not transfer")

        service = self.service(forbidden)
        request = self.request()
        with service.artifact_lease(request.destination):
            subscriber = service.submit(request)
            with service._condition:
                self.assertTrue(
                    service._condition.wait_for(
                        lambda: (
                            service._artifacts[service._key(request.destination)].waiting_writers
                        ),
                        timeout=5,
                    )
                )
            subscriber.cancel()
            service.close()
            with self.assertRaises(CancelledError):
                subscriber.result(5)
        self.assertFalse(request.destination.exists())
        self.assertFalse(service._artifacts)

    def test_noncooperating_writer_is_preserved_at_atomic_publication(self):
        def transfer(request, staging, report, cancelled):
            path = self.complete(request, staging, report, cancelled)
            request.destination.write_bytes(b"imported model")
            return path

        service = self.service(transfer)
        with self.assertRaises(FileExistsError):
            service.submit(self.request()).result(5)
        self.assertEqual(self.request().destination.read_bytes(), b"imported model")

    def test_read_lease_blocks_transfer_and_write_lease_waits_for_transfer(self):
        started, release, mutation_started = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def transfer(*arguments):
            started.set()
            self.assertTrue(release.wait(5))
            return self.complete(*arguments)

        service = self.service(transfer)
        request = self.request()
        with service.artifact_lease(request.destination):
            subscriber = service.submit(request)
            with service._condition:
                self.assertTrue(
                    service._condition.wait_for(
                        lambda: (
                            service._artifacts[service._key(request.destination)].waiting_writers
                        ),
                        timeout=5,
                    )
                )
            self.assertFalse(started.is_set())
        self.assertTrue(started.wait(5))

        def remove():
            mutation_started.set()
            with service.artifact_lease(request.destination, write=True):
                self.assertEqual(request.destination.read_bytes(), b"complete")

        with ThreadPoolExecutor(1) as executor:
            mutation = executor.submit(remove)
            self.assertTrue(mutation_started.wait(5))
            with service._condition:
                self.assertTrue(
                    service._condition.wait_for(
                        lambda: (
                            service._artifacts[service._key(request.destination)].waiting_writers
                        ),
                        timeout=5,
                    )
                )
            self.assertFalse(mutation.done())
            release.set()
            self.assertEqual(subscriber.result(5), request.destination)
            mutation.result(5)
        self.assertFalse(service._artifacts)

    def test_cancelled_lease_waiter_releases_ownership(self):
        service = self.service(self.complete)
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(CancelledError):
            with service.artifact_lease(self.root / "model", write=True, cancelled=cancelled):
                self.fail("cancelled lease acquired")
        self.assertFalse(service._artifacts)

    def test_progress_callback_failure_does_not_fail_preparation(self):
        service = self.service(self.complete)

        def disconnected(*arguments):
            raise RuntimeError("UI disconnected")

        self.assertTrue(service.submit(self.request(), disconnected).result(5).is_file())

    def test_subscribers_receive_their_own_gradio_context_after_submit_returns(self):
        from gradio.context import LocalContext

        started, release = threading.Event(), threading.Event()
        events = {"first": [], "second": []}

        def transfer(*arguments):
            started.set()
            self.assertTrue(release.wait(5))
            arguments[2](4, 8, 1)
            return self.complete(*arguments)

        def subscribe(service, name):
            event_token = LocalContext.event_id.set(name)
            request_token = LocalContext.request.set(name + " request")
            try:
                return service.submit(
                    self.request(),
                    lambda *event: events[name].append(
                        (LocalContext.event_id.get(), LocalContext.request.get(), event)
                    ),
                )
            finally:
                LocalContext.request.reset(request_token)
                LocalContext.event_id.reset(event_token)

        service = self.service(transfer)
        first = subscribe(service, "first")
        self.assertTrue(started.wait(5))
        second = subscribe(service, "second")
        release.set()
        first.result(5)
        second.result(5)
        for name in events:
            self.assertEqual(
                events[name],
                [
                    (name, name + " request", (4, 8, 1)),
                    (name, name + " request", (8, 8, 0)),
                ],
            )

    def test_external_sdk_staging_is_invisible_to_rf_and_native_scanners(self):
        import runpy
        from types import SimpleNamespace
        from ruinedfooocus.core.path import PathManager
        from ruinedfooocus.features.models.embedding_browser import scan_embeddings
        from ui_test_helpers import function_from

        root = self.root / "models"
        root.mkdir()
        for name in ("ready.gguf", "ready.safetensors"):
            (root / name).write_bytes(b"ready")
        barrier, release = threading.Barrier(3), threading.Event()

        def transfer(request, staging, report, cancelled):
            path = staging / request.destination.name
            path.write_bytes(b"unverified SDK final")
            barrier.wait(5)
            self.assertTrue(release.wait(5))
            return path

        service = self.service(transfer)
        jobs = [
            service.submit(
                PreparationRequest("https://fixture.test/" + name, root / name, sha256="0" * 64)
            )
            for name in ("pending.gguf", "pending.safetensors")
        ]
        barrier.wait(5)
        try:
            manager = PathManager.__new__(PathManager)
            manager.paths = {"path_llm": str(root)}
            manager.DOWNLOADABLE_FILES = {}
            state = SimpleNamespace(
                path_manager=manager, settings=SimpleNamespace(default_settings={})
            )
            project = Path(__file__).resolve().parents[1]
            with patch.dict("sys.modules", {"ruinedfooocus.app.state": state}):
                chat = runpy.run_path(str(project / "ruinedfooocus/features/chat/llama_models.py"))
                self.assertEqual(set(chat["local_models"]().values()), {str(root / "ready.gguf")})
                owner = SimpleNamespace(
                    ready={},
                    names={},
                    model_dirs={"checkpoints": [root]},
                    revision=0,
                    EXTENSIONS=PathManager.EXTENSIONS,
                )
                update = function_from(
                    "ruinedfooocus/features/models/model_handler.py",
                    "update_all_models",
                    {"Path": Path},
                )
                update(owner)
                self.assertEqual(owner.names["checkpoints"], ["ready.gguf", "ready.safetensors"])
            self.assertEqual(
                [entry.name for entry in scan_embeddings([root])[0]], ["ready.safetensors"]
            )
            self.assertEqual(manager.get_model_filenames(root), ["ready.gguf", "ready.safetensors"])
            with patch.dict(
                "sys.modules",
                {
                    "comfy.cli_args": SimpleNamespace(
                        args=SimpleNamespace(
                            base_directory=str(self.root / "native"), models_directory=None
                        )
                    )
                },
            ):
                native = runpy.run_path(str(project / "repositories/ComfyUI/folder_paths.py"))
            native["folder_names_and_paths"]["preparation_test"] = (
                [str(root)],
                {".gguf", ".safetensors"},
            )
            self.assertEqual(
                native["get_filename_list"]("preparation_test"), ["ready.gguf", "ready.safetensors"]
            )
        finally:
            release.set()
        for job in jobs:
            with self.assertRaisesRegex(OSError, "SHA256"):
                job.result(5)
        self.assertEqual(
            sorted(path.name for path in root.iterdir()), ["ready.gguf", "ready.safetensors"]
        )
        self.assertFalse(list((self.root / "cache").rglob("*.gguf")))
        self.assertFalse(list((self.root / "cache").rglob("*.safetensors")))

    def test_cross_volume_publication_copies_to_part_and_cleans_failed_copy(self):
        import errno
        from ruinedfooocus.core.preparation import _publish_no_clobber

        service = self.service(self.complete)
        sources = []

        def publish(source, destination):
            sources.append(source)
            if len(sources) == 1:
                raise OSError(errno.EXDEV, "different filesystem")
            self.assertEqual(source.suffix, ".part")
            self.assertEqual(source.parent, destination.parent)
            self.assertEqual(source.read_bytes(), b"complete")
            self.assertFalse(destination.exists())
            _publish_no_clobber(source, destination)

        with patch("ruinedfooocus.core.preparation._publish_no_clobber", side_effect=publish):
            self.assertEqual(service.submit(self.request()).result(5).read_bytes(), b"complete")
        self.assertFalse(list(self.root.glob("*.part")))
        with (
            patch(
                "ruinedfooocus.core.preparation._publish_no_clobber",
                side_effect=OSError(errno.EXDEV, "different filesystem"),
            ),
            patch("ruinedfooocus.core.preparation.os.fsync", side_effect=OSError("disk full")),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            service.submit(self.request("failed.gguf")).result(5)
        self.assertFalse((self.root / "failed.gguf").exists())
        self.assertFalse(list(self.root.glob("*.part")))

    def test_cross_volume_copy_is_reverified_before_publication(self):
        import errno
        import os

        service = self.service(self.complete)
        fsync = os.fsync

        def corrupt_copy(descriptor):
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, b"tampered")
            fsync(descriptor)

        request = self.request(sha256=hashlib.sha256(b"complete").hexdigest())
        with (
            patch(
                "ruinedfooocus.core.preparation._publish_no_clobber",
                side_effect=OSError(errno.EXDEV, "different filesystem"),
            ) as publish,
            patch("ruinedfooocus.core.preparation.os.fsync", side_effect=corrupt_copy),
            self.assertRaisesRegex(OSError, "SHA256"),
        ):
            service.submit(request).result(5)
        publish.assert_called_once()
        self.assertFalse(request.destination.exists())
        self.assertFalse(list(self.root.glob("*.part")))

    def test_catalog_destination_is_immutable_and_staging_is_not_discovered(self):
        from ruinedfooocus.core.path import PathManager

        manager = PathManager.__new__(PathManager)
        manager.paths = {"path_llm": str(self.root / "old")}
        manager.DOWNLOADABLE_FILES = {
            "alias": {
                "url": "https://fixture.test/model",
                "path": "path_llm",
                "filename": "model.gguf",
            }
        }
        started, release = threading.Event(), threading.Event()

        def transfer(*arguments):
            started.set()
            self.assertTrue(release.wait(5))
            return self.complete(*arguments)

        service = self.service(transfer)
        with patch("ruinedfooocus.core.preparation.preparation_service", service):
            subscriber = manager.prepare_file("alias")
            self.assertTrue(started.wait(5))
            manager.paths["path_llm"] = str(self.root / "new")
            manager.DOWNLOADABLE_FILES["alias"]["url"] = "https://changed.test"
            with self.assertRaises(FrozenInstanceError):
                subscriber.request.url = "changed"
            release.set()
            self.assertEqual(subscriber.result(5), self.root / "old/model.gguf")
        hidden = self.root / "cache/test/partial.gguf"
        hidden.parent.mkdir(parents=True)
        hidden.write_bytes(b"partial")
        self.assertEqual(manager.get_model_filenames(self.root / "old"), ["model.gguf"])
        manager.paths["path_llm"] = str(self.root / "old")
        self.assertEqual(manager.get_folder_list("llm"), ["model.gguf"])

    def test_http_short_response_leaves_no_partial_and_can_retry(self):
        class Response:
            headers = {"content-length": "10"}

            def __enter__(self):
                return self

            def __exit__(self, *arguments):
                pass

            def raise_for_status(self):
                pass

            def iter_content(self, **options):
                yield b"short"

        service = self.service(transfer_file)
        with patch("requests.get", return_value=Response()) as get:
            with self.assertRaisesRegex(OSError, "Incomplete"):
                service.submit(self.request()).result(5)
            self.assertEqual(get.call_count, 1)
        self.assertFalse(self.request().destination.exists())
        self.assertFalse(list(self.root.rglob("*.part")))


if __name__ == "__main__":
    unittest.main()
