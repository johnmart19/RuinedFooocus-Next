"""Real inert Python children exercise operation bounds without runtime/device probes."""

from concurrent.futures import CancelledError
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

from ruinedfooocus.core import bounded_subprocess


class BoundedSubprocessTests(unittest.TestCase):
    def run_child(self, code, **kwargs):
        return bounded_subprocess.run_bounded([sys.executable, "-I", "-c", code], **kwargs)

    def test_binary_input_and_separate_outputs_are_drained(self):
        stdout, stderr = self.run_child(
            "import sys; data=sys.stdin.buffer.read(); sys.stderr.buffer.write(b'notice'); "
            "sys.stdout.buffer.write(data)",
            input_data="Café 0007".encode(),
        )
        self.assertEqual(stdout.decode(), "Café 0007")
        self.assertEqual(stderr, b"notice")

    def test_overflow_and_timeout_reap_owned_child(self):
        for code, options, error in (
            ("import sys; sys.stderr.buffer.write(b'x'*1000000)", {"max_output": 4096}, ValueError),
            ("import time; time.sleep(30)", {"timeout": 0.15}, subprocess.TimeoutExpired),
        ):
            original = bounded_subprocess.start_process
            children = []

            def start(*args, **kwargs):
                result = original(*args, **kwargs)
                children.append(result[0])
                return result

            began = time.monotonic()
            with patch.object(bounded_subprocess, "start_process", side_effect=start):
                with self.assertRaises(error):
                    self.run_child(code, **options)
            self.assertLess(time.monotonic() - began, 5)
            self.assertIsNotNone(children[0].poll())

    def test_cancellation_and_shared_deadline_stop_running_work(self):
        stopped = threading.Event()

        def check():
            if stopped.is_set():
                raise CancelledError("Stopped")

        timer = threading.Timer(0.15, stopped.set)
        timer.start()
        try:
            with self.assertRaises(CancelledError):
                self.run_child("import time; time.sleep(30)", check_cancelled=check)
        finally:
            timer.cancel()
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_child("raise AssertionError('must not run')", deadline=time.monotonic() - 1)

    def test_failures_never_echo_untrusted_child_stderr(self):
        with self.assertRaisesRegex(
            bounded_subprocess.BoundedProcessError, "could not complete"
        ) as caught:
            self.run_child(
                "import sys; sys.stdout.write('before'); sys.stderr.write('private document'); sys.exit(2)"
            )
        self.assertNotIn("private", str(caught.exception))
        self.assertNotIn("private", repr(caught.exception))
        self.assertEqual(caught.exception.returncode, 2)
        self.assertEqual(caught.exception.stdout, b"before")
        self.assertEqual(caught.exception.stderr, b"private document")

    def test_exited_parent_with_descendant_inheriting_pipes_cannot_outlive_budget(self):
        began = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_child(
                "import subprocess,sys; subprocess.Popen([sys.executable, '-I', '-c', "
                "'import time; time.sleep(30)']); print('inert child started')",
                timeout=0.25,
            )
        self.assertLess(time.monotonic() - began, 5)


if __name__ == "__main__":
    unittest.main()
