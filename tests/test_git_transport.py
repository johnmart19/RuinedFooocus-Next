"""Network work has a real process deadline and never installs application code."""

from pathlib import Path
import socket
import sys
import tempfile
import threading
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
            with self.assertRaisesRegex(RuntimeError, "Git fetch failed"):
                bounded_git_operation("fetch", folder, "missing-remote", timeout=5)

    def test_blocked_http_clone_is_terminated_at_its_deadline(self):
        accepted = threading.Event()
        release = threading.Event()
        with socket.socket() as server, tempfile.TemporaryDirectory() as folder:
            server.bind(("127.0.0.1", 0))
            server.listen()
            server.settimeout(3)

            def block_request():
                try:
                    connection, _ = server.accept()
                    with connection:
                        accepted.set()
                        release.wait(5)
                except OSError:
                    pass

            thread = threading.Thread(target=block_request, daemon=True)
            thread.start()
            try:
                with self.assertRaisesRegex(TimeoutError, "timed out after 1 seconds"):
                    bounded_git_operation(
                        "clone",
                        Path(folder) / "staged checkout",
                        f"http://127.0.0.1:{server.getsockname()[1]}/blocked.git",
                        timeout=1,
                    )
                self.assertTrue(accepted.is_set(), "The worker never reached the slow server")
            finally:
                release.set()
                thread.join(timeout=4)

    def test_invalid_operation_is_rejected_before_launching_a_process(self):
        with patch("ruinedfooocus.packaging.git_transport.subprocess.run") as run:
            with self.assertRaises(ValueError):
                bounded_git_operation("checkout", ".", "origin")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
