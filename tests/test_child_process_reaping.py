"""Owned children are killed and their handles released when polite termination stalls."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.backends.llama import llama_build
from ruinedfooocus.backends.onnx import onnx_runtime


class StalledProcess:
    """A child that ignores ``terminate`` and exits only after ``kill``."""

    stdin = stdout = None

    def __init__(self, *, dies_on_kill=True):
        self.dies_on_kill = dies_on_kill
        self.terminated = self.killed = False
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        if self.dies_on_kill:
            self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("child", timeout)
        return self.returncode


class ChildProcessReapingTests(unittest.TestCase):
    def test_abandoned_build_stage_kills_a_child_that_ignores_terminate(self):
        process = StalledProcess()
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "build.log"
            log.write_text("", encoding="utf-8")
            with (
                patch(
                    "ruinedfooocus.core.child_process.start_process", return_value=(process, None)
                ),
                patch.object(llama_build.time, "sleep"),
            ):
                stage = llama_build._run(["compiler"], "Compiling", log, {})
                next(stage)
                # The interface abandons the generator when the build is cancelled.
                stage.close()
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(process.returncode, -9)

    def test_mask_worker_close_releases_everything_when_the_child_cannot_be_reaped(self):
        session = onnx_runtime.IsolatedMaskSession()
        process = StalledProcess(dies_on_kill=False)
        log, job, scratch, reader = Mock(), Mock(), Mock(), Mock()
        session.process, session.log, session.job = process, log, job
        session.scratch, session.reader = scratch, reader
        # A worker stuck in a driver call must not block CPU recovery or keep
        # the job object, log handle and scratch directory.
        session.close()
        self.assertTrue(process.killed)
        job.close.assert_called_once_with()
        log.close.assert_called_once_with()
        scratch.cleanup.assert_called_once_with()
        self.assertIsNone(session.process)
        self.assertIsNone(session.job)
        self.assertIsNone(session.log)
        self.assertIsNone(session.scratch)


if __name__ == "__main__":
    unittest.main()
