"""Network work has a real process deadline and never installs application code."""

from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import pygit2

from ruinedfooocus.packaging.git_transport import bounded_git_operation


class GitTransportTests(unittest.TestCase):
    def test_fetch_uses_this_interpreter_and_a_finite_deadline(self):
        with patch("ruinedfooocus.packaging.git_transport.subprocess.run") as run:
            run.return_value.returncode = 0
            bounded_git_operation("fetch", Path("checkout with spaces"), "origin")
        arguments, options = run.call_args
        self.assertEqual(arguments[0][0], sys.executable)
        self.assertEqual(arguments[0][2:], ["fetch", "checkout with spaces", "origin"])
        self.assertEqual(options["timeout"], 10)
        self.assertNotIn("shell", options)

    def test_failed_worker_reports_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = pygit2.init_repository(folder)
            repo.free()
            # The deadline is not under test here: it only has to outlast a slow
            # interpreter start on a busy machine. The worker exits by itself.
            with self.assertRaisesRegex(RuntimeError, "Git fetch failed"):
                bounded_git_operation("fetch", folder, "missing-remote", timeout=120)

    def test_blocked_http_clone_is_terminated_at_its_deadline(self):
        accepted = threading.Event()
        release = threading.Event()
        workers = []

        class WorkerBlockedBeforeDeadline(subprocess.Popen):
            """Interpreter start-up and the pygit2 import are not network time.

            The production deadline, expiry and termination are unchanged; its
            clock only starts once the worker is blocked on the server (or has
            exited), so a slow process start cannot consume the deadline.
            """

            def __init__(self, *arguments, **options):
                super().__init__(*arguments, **options)
                workers.append(self)

            def communicate(self, input=None, timeout=None):
                limit = time.monotonic() + 120
                while (
                    timeout is not None
                    and not accepted.wait(0.05)
                    and self.poll() is None
                    and time.monotonic() < limit
                ):
                    pass
                return super().communicate(input, timeout)

        with socket.socket() as server, tempfile.TemporaryDirectory() as folder:
            server.bind(("127.0.0.1", 0))
            server.listen()
            server.settimeout(120)

            def block_request():
                try:
                    connection, _ = server.accept()
                    with connection:
                        accepted.set()
                        release.wait(120)
                except OSError:
                    pass

            thread = threading.Thread(target=block_request, daemon=True)
            thread.start()
            try:
                with (
                    patch(
                        "ruinedfooocus.packaging.git_transport.subprocess.Popen",
                        WorkerBlockedBeforeDeadline,
                    ),
                    self.assertRaisesRegex(TimeoutError, "timed out after 1 seconds"),
                ):
                    bounded_git_operation(
                        "clone",
                        Path(folder) / "staged checkout",
                        f"http://127.0.0.1:{server.getsockname()[1]}/blocked.git",
                        timeout=1,
                    )
                self.assertTrue(accepted.is_set(), "The worker never reached the slow server")
                self.assertFalse(release.is_set(), "The server stopped blocking the worker")
                self.assertEqual(len(workers), 1)
                self.assertIsNotNone(workers[0].poll(), "The blocked worker is still running")
            finally:
                release.set()
                server.close()
                thread.join(timeout=4)

    def test_invalid_operation_is_rejected_before_launching_a_process(self):
        with patch("ruinedfooocus.packaging.git_transport.subprocess.run") as run:
            with self.assertRaises(ValueError):
                bounded_git_operation("checkout", ".", "origin")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
