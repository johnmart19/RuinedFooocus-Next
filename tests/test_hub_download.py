"""Hub transport contracts using fakes and an isolated loopback range server."""

from concurrent.futures import CancelledError
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

from ruinedfooocus.core.hub_download import download_hub_file
from ruinedfooocus.core.preparation import PreparationRequest, PreparationService, transfer_file


class HubDownloadTests(unittest.TestCase):
    url = "https://huggingface.co/fixture/model/resolve/release%2F1/sub/model.gguf"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.service = PreparationService(
            max_workers=1, transport=transfer_file, staging_root=self.root / "cache"
        )
        self.addCleanup(self.service.close)
        self.destination = self.root / "renamed.gguf"

    def test_sdk_uses_stable_resume_staging_and_publishes_only_after_verification(self):
        directories, events = [], []

        def sdk(**options):
            self.assertEqual(options["repo_id"], "fixture/model")
            self.assertEqual(options["revision"], "release/1")
            self.assertEqual(options["filename"], "sub/model.gguf")
            staging = Path(options["local_dir"])
            directories.append(staging)
            self.assertNotEqual(staging, self.destination.parent)
            self.assertFalse(self.destination.exists())
            partial = staging / "resume.incomplete"
            if len(directories) == 1:
                partial.write_bytes(b"resume state")
                raise OSError("interrupted")
            self.assertEqual(partial.read_bytes(), b"resume state")
            completed = staging / "sub/model.gguf"
            completed.parent.mkdir()
            completed.write_bytes(b"complete")
            with options["tqdm_class"](total=8, initial=3, disable=True) as progress:
                progress.last_update = 0
                progress.update(5)
                self.assertEqual(progress.n, 8)
            return str(completed)

        with (
            patch("ruinedfooocus.core.preparation.preparation_service", self.service),
            patch("ruinedfooocus.core.hub_download.hf_hub_download", side_effect=sdk),
        ):
            with self.assertRaisesRegex(OSError, "interrupted"):
                download_hub_file(self.url, self.destination)
            self.assertFalse(self.destination.exists())
            self.assertEqual(
                download_hub_file(self.url, self.destination, lambda *event: events.append(event)),
                self.destination,
            )
        self.assertEqual(directories[0], directories[1])
        self.assertEqual(self.destination.read_bytes(), b"complete")
        self.assertEqual(events[0], (3, 8, 0))
        self.assertEqual(events[-1], (8, 8, 0))

    def test_cancelled_sdk_transfer_retains_resume_and_never_publishes(self):
        from ruinedfooocus.core.preparation import PreparationRequest

        started, release = threading.Event(), threading.Event()
        staging_paths = []

        def sdk(**options):
            staging = Path(options["local_dir"])
            staging_paths.append(staging)
            (staging / "owned.incomplete").write_bytes(b"partial")
            with options["tqdm_class"](total=8, disable=True) as progress:
                started.set()
                self.assertTrue(release.wait(5))
                progress.update(4)
            self.fail("cancel must interrupt SDK progress")

        with patch("ruinedfooocus.core.hub_download.hf_hub_download", side_effect=sdk):
            subscription = self.service.submit(PreparationRequest(self.url, self.destination))
            self.assertTrue(started.wait(5))
            subscription.cancel()
            release.set()
            self.service.close()
            with self.assertRaises(CancelledError):
                subscription.result(5)
        self.assertFalse(self.destination.exists())
        self.assertEqual((staging_paths[0] / "owned.incomplete").read_bytes(), b"partial")

    def test_invalid_hub_relative_paths_never_reach_sdk(self):
        from ruinedfooocus.core.preparation import PreparationRequest

        with patch("ruinedfooocus.core.hub_download.hf_hub_download") as sdk:
            for filename in ("%2E%2E/outside.gguf", "%2Fabsolute.gguf", "C%3Aevil.gguf"):
                url = "https://huggingface.co/fixture/model/resolve/main/" + filename
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    self.service.submit(PreparationRequest(url, self.destination)).result(5)
            sdk.assert_not_called()

    def test_gated_access_error_is_actionable_and_final_stays_absent(self):
        from huggingface_hub.errors import HfHubHTTPError
        from types import SimpleNamespace

        error = HfHubHTTPError(
            "Access denied", response=SimpleNamespace(status_code=403, headers={}, request=None)
        )
        with (
            patch("ruinedfooocus.core.preparation.preparation_service", self.service),
            patch("ruinedfooocus.core.hub_download.hf_hub_download", side_effect=error),
            self.assertRaisesRegex(RuntimeError, "HF_TOKEN"),
        ):
            download_hub_file(self.url, self.destination)
        self.assertFalse(self.destination.exists())


class VerifiedHubHttpTests(unittest.TestCase):
    revision = "a" * 40
    url = f"https://huggingface.co/fixture/model/resolve/{revision}/model.gguf"
    content = bytes(range(256)) * (3 * 1024 * 1024 // 256)
    sha256 = hashlib.sha256(content).hexdigest()
    etag = f'"{sha256}"'
    partial_size = 2 * 1024 * 1024

    def setUp(self):
        from huggingface_hub.utils import get_session

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.destination = self.root / "model.gguf"
        self.phase = "interrupt"
        self.requests = []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                phase = fixture.phase
                if phase == "gated":
                    self.send_response(403)
                    self.end_headers()
                    return
                self.send_response(302 if phase == "cdn" else 200)
                self.send_header(
                    "X-Repo-Commit", "b" * 40 if phase == "revision" else fixture.revision
                )
                self.send_header("ETag", '"other"' if phase in ("etag", "cdn") else fixture.etag)
                self.send_header(
                    "Content-Length", str(0 if phase == "cdn" else len(fixture.content))
                )
                if phase == "xet":
                    self.send_header("X-Xet-Hash", "fixture")
                if phase == "cdn":
                    self.send_header("Location", "https://cdn-lfs.hf.co/fixture")
                    self.send_header("X-Linked-ETag", fixture.etag)
                    self.send_header("X-Linked-Size", str(len(fixture.content)))
                self.end_headers()

            def do_GET(self):
                phase = fixture.phase
                requested_range = self.headers.get("Range")
                fixture.requests.append(
                    (
                        self.path,
                        requested_range,
                        self.headers.get("If-Range"),
                        self.headers.get("Authorization"),
                    )
                )
                offset = int(requested_range[6:-1]) if requested_range and phase != "restart" else 0
                total = len(fixture.content)
                body = fixture.content[offset:]
                if phase == "corrupt":
                    body = b"c" * len(body)
                self.send_response(206 if offset else 200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("ETag", fixture.etag)
                if offset:
                    start = offset - 1 if phase == "invalid_range" else offset
                    self.send_header("Content-Range", f"bytes {start}-{total - 1}/{total}")
                self.end_headers()
                if phase == "interrupt":
                    self.wfile.write(body[: fixture.partial_size])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_WR)
                else:
                    try:
                        self.wfile.write(body)
                    except OSError:
                        pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.addCleanup(server.server_close)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)

        def localize(request):
            self.assertIn(request.url.host, {"huggingface.co", "cdn-lfs.hf.co"})
            request.url = request.url.copy_with(
                scheme="http", host="127.0.0.1", port=server.server_port
            )

        client = type(get_session())(trust_env=False, event_hooks={"request": [localize]})
        self.addCleanup(client.close)
        owner = "ruinedfooocus.core.hub_download"
        self.enterContext(patch(owner + ".get_session", return_value=client))
        self.enterContext(
            patch(
                owner + ".build_hf_headers", side_effect=lambda: {"authorization": "Bearer fixture"}
            )
        )
        self.enterContext(patch(owner + ".try_to_load_from_cache", return_value=None))
        self.enterContext(patch(owner + ".constants.HF_HUB_OFFLINE", False))
        self.enterContext(patch(owner + ".constants.ENDPOINT", "https://huggingface.co"))
        self.sdk = self.enterContext(
            patch(owner + ".hf_hub_download", side_effect=RuntimeError("SDK fallback"))
        )

    def transfer(self, progress=None):
        service = PreparationService(staging_root=self.root / "cache")
        request = PreparationRequest(self.url, self.destination, len(self.content), self.sha256)
        try:
            return service.submit(request, progress).result(8)
        finally:
            service.close()

    def interrupt(self):
        self.phase = "interrupt"
        with self.assertRaisesRegex(Exception, "closed|incomplete|received"):
            self.transfer()
        self.assertFalse(self.destination.exists())
        parts = list(self.root.rglob("hub-http.part"))
        self.assertEqual([part.stat().st_size for part in parts], [self.partial_size])
        self.assertEqual(len(list(self.root.rglob("hub-http.json"))), 1)
        return parts[0]

    def test_resume_requests_only_remaining_bytes_after_service_restart(self):
        self.interrupt()
        self.phase = "resume"
        events = []
        self.assertEqual(self.transfer(lambda *values: events.append(values)), self.destination)
        self.assertEqual(self.requests[-1][1:3], (f"bytes={self.partial_size}-", self.etag))
        self.assertEqual(events[0], (self.partial_size, len(self.content), 0))
        self.assertEqual(self.destination.read_bytes(), self.content)
        self.sdk.assert_not_called()

    def test_full_response_discards_the_previous_prefix(self):
        part = self.interrupt()
        part.write_bytes(b"x" * self.partial_size)
        self.phase = "restart"
        events = []
        self.transfer(lambda *values: events.append(values))
        self.assertEqual(self.requests[-1][1], f"bytes={self.partial_size}-")
        self.assertEqual(events[0], (0, len(self.content), 0))
        self.assertEqual(self.destination.read_bytes(), self.content)

    def test_invalid_content_range_preserves_the_original_partial(self):
        part = self.interrupt()
        self.phase = "invalid_range"
        with self.assertRaisesRegex(OSError, "Invalid Content-Range"):
            self.transfer()
        self.assertEqual(part.read_bytes(), self.content[: self.partial_size])
        self.assertFalse(self.destination.exists())
        self.sdk.assert_not_called()

    def test_changed_revision_or_etag_discards_the_previous_prefix(self):
        for phase in ("revision", "etag"):
            with self.subTest(phase=phase):
                part = self.interrupt()
                self.phase = phase
                with self.assertRaisesRegex(RuntimeError, "SDK fallback"):
                    self.transfer()
                self.assertFalse(part.exists())
                self.assertFalse(list(self.root.rglob("hub-http.json")))
                self.assertIsNone(self.requests[-1][1])
                self.assertFalse(self.destination.exists())
        self.assertEqual(self.sdk.call_count, 2)

    def test_corrupted_resume_does_not_publish_or_retry_via_sdk(self):
        self.interrupt()
        self.phase = "corrupt"
        with self.assertRaisesRegex(OSError, "SHA256"):
            self.transfer()
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.root.rglob("hub-http-*.complete")))
        self.assertFalse(list(self.root.rglob("hub-http.part")))
        self.assertFalse(list(self.root.rglob("hub-http.json")))
        self.sdk.assert_not_called()

    def test_gated_metadata_reuses_the_actionable_error(self):
        self.phase = "gated"
        with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
            self.transfer()
        self.assertEqual(self.requests, [])
        self.sdk.assert_not_called()

    def test_xet_metadata_retains_sdk_transport(self):
        self.phase = "xet"
        with self.assertRaisesRegex(RuntimeError, "SDK fallback"):
            self.transfer()
        self.assertEqual(self.requests, [])
        self.sdk.assert_called_once()

    def test_cdn_requests_do_not_forward_the_hub_token(self):
        self.phase = "cdn"
        self.transfer()
        self.assertEqual(self.requests, [("/fixture", None, None, None)])
        self.assertEqual(self.destination.read_bytes(), self.content)


if __name__ == "__main__":
    unittest.main()
