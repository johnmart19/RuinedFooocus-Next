"""Independent installer child policy without starting native processes."""

import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.backends.llama import llama_installer
from ruinedfooocus.backends.onnx import acquisition


class InstallerEnvironmentTests(unittest.TestCase):
    def setUp(self):
        report = {"requested": {"HIP_VISIBLE_DEVICES": "1"}, "inherited": {}}
        arguments = SimpleNamespace(args=SimpleNamespace(_device_environment_status=report))
        for context in (
            patch.dict(sys.modules, {"ruinedfooocus.app.arguments": arguments}),
            patch.dict(
                os.environ, {"HIP_VISIBLE_DEVICES": "1", "GGML_VK_VISIBLE_DEVICES": "0"}, clear=True
            ),
        ):
            context.start()
            self.addCleanup(context.stop)

    def test_chat_child_restores_only_image_override(self):
        environment = llama_installer.runtime_environment(False, backend="CPU")
        self.assertNotIn("HIP_VISIBLE_DEVICES", environment)
        self.assertEqual(environment["GGML_VK_VISIBLE_DEVICES"], "0")
        self.assertEqual(os.environ["HIP_VISIBLE_DEVICES"], "1")

    def test_onnx_cancellable_child_preserves_explicit_options(self):
        process, job = Mock(), Mock()
        process.poll.return_value = 0
        process.returncode = 0
        with patch(
            "ruinedfooocus.core.child_process.start_process", return_value=(process, job)
        ) as start:
            result = acquisition.run(
                ["inert"], check_cancelled=lambda: None, cwd="private", capture_output=True
            )
        environment = start.call_args.kwargs["env"]
        self.assertNotIn("HIP_VISIBLE_DEVICES", environment)
        self.assertEqual(environment["GGML_VK_VISIBLE_DEVICES"], "0")
        self.assertEqual(start.call_args.kwargs["cwd"], "private")
        self.assertEqual(result.returncode, 0)
        job.close.assert_called_once()

    def test_explicit_child_environment_is_not_replaced(self):
        with patch.object(acquisition.subprocess, "run") as run:
            acquisition.run(["inert"], env={"OWNED": "1"})
        self.assertEqual(run.call_args.kwargs["env"], {"OWNED": "1"})

    def test_cancel_closes_job_even_when_process_reaping_fails(self):
        process, job = Mock(), Mock()
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired("inert", 5)
        cancelled = Mock(side_effect=[None, RuntimeError("cancelled")])
        with patch("ruinedfooocus.core.child_process.start_process", return_value=(process, job)):
            with self.assertRaises(subprocess.TimeoutExpired):
                acquisition.run(["inert"], check_cancelled=cancelled)
        process.kill.assert_called_once()
        job.close.assert_called_once()

    def test_interactive_pipe_request_fails_before_process_creation(self):
        with patch("ruinedfooocus.core.child_process.start_process") as start:
            with self.assertRaisesRegex(ValueError, "interactive pipes"):
                acquisition.run(["inert"], check_cancelled=lambda: None, stdin=subprocess.PIPE)
        start.assert_not_called()
