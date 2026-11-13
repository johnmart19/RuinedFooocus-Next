"""Worker diagnostics preserve readiness, fixed deadlines and cancellation."""

from contextlib import redirect_stdout
import io
import json
import queue
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.backends.onnx import onnx_runtime as runtime
from ruinedfooocus.backends.onnx import onnx_worker as worker


def progress(phase="creating native model session", elapsed=0.25):
    return {"event": "initialization_progress", "phase": phase, "elapsed_seconds": elapsed}


class OnnxDiagnosticTests(unittest.TestCase):
    def test_worker_emits_phase_and_elapsed_without_loading_a_model(self):
        output = io.StringIO()
        with redirect_stdout(output), patch.object(worker.time, "monotonic", return_value=10.5):
            worker.initialization_progress("creating native model session", 10)
        self.assertEqual(json.loads(output.getvalue()), progress(elapsed=0.5))

    def test_progress_is_visible_but_does_not_count_as_ready(self):
        session = runtime.IsolatedMaskSession(provider="Windows ML")
        session.messages.put(progress())
        ready = {"ready": True, "provider": "MIGraphXExecutionProvider"}
        session.messages.put(ready)
        with patch.object(runtime, "report_status") as status:
            self.assertEqual(session._wait(180, phase="initialization"), ready)
        self.assertEqual(
            session.initialization_progress,
            {"phase": "creating native model session", "elapsed_seconds": 0.25},
        )
        self.assertIn("phase started at 0.2s", status.call_args.args[0])

    def test_progress_never_extends_the_original_initialization_deadline(self):
        session = runtime.IsolatedMaskSession()
        session.messages.put(progress())
        session.messages.put(progress("publishing readiness", 179))
        with patch.object(runtime.time, "monotonic", side_effect=[100, 100.1, 280.1]):
            with self.assertRaisesRegex(
                runtime.GPUWorkerError,
                "initialization timed out after 180s.*creating native model session",
            ):
                session._wait(180, phase="initialization")
        self.assertEqual(session.messages.qsize(), 1)

    def test_inference_timeout_is_distinct_and_does_not_reuse_initialization_phase(self):
        session = runtime.IsolatedMaskSession()
        session.initialization_progress = {"phase": "publishing readiness", "elapsed_seconds": 1}
        with patch.object(runtime.time, "monotonic", side_effect=[0, 121]):
            with self.assertRaisesRegex(
                runtime.GPUWorkerError, "inference timed out after 120s"
            ) as error:
                session._wait(120, phase="inference")
        self.assertNotIn("publishing readiness", str(error.exception))

    def test_old_worker_ready_remains_accepted(self):
        session = runtime.IsolatedMaskSession()
        session.messages.put({"ready": True})
        self.assertEqual(session._wait(180, phase="initialization"), {"ready": True})

    def test_cancellation_after_progress_closes_without_cpu_recovery(self):
        session = runtime.IsolatedMaskSession()
        session.messages.put(progress())
        session.check_cancelled = Mock(side_effect=[None, RuntimeError("cancelled")])
        with patch.object(session, "close") as close, patch.object(session, "_cpu_predict") as cpu:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                session._wait(180, phase="initialization")
        close.assert_called_once_with()
        cpu.assert_not_called()
        self.assertFalse(session.disabled)

    def test_progress_payload_validation_and_wrong_phase_refuse(self):
        for message, phase in (
            (progress(elapsed=float("nan")), "initialization"),
            (progress(phase="x" * 161), "initialization"),
            (progress(), "inference"),
        ):
            with self.subTest(message=message, phase=phase):
                session = runtime.IsolatedMaskSession()
                session.messages.put(message)
                with self.assertRaisesRegex(
                    runtime.GPUWorkerError, "Invalid initialization progress"
                ):
                    session._wait(180, phase=phase)

    def test_reader_bounds_progress_flood_without_dropping_ready(self):
        session = runtime.IsolatedMaskSession()
        stream = io.StringIO((json.dumps(progress()) + "\n") * 1000 + '{"ready":true}\n')
        messages = queue.Queue()
        session._read_messages(SimpleNamespace(stdout=stream), messages)
        received = list(messages.queue)
        self.assertEqual(
            sum(value.get("event") == "initialization_progress" for value in received), 32
        )
        self.assertEqual(sum(value.get("ready") is True for value in received), 1)

    def test_reader_refuses_an_oversized_protocol_line(self):
        session = runtime.IsolatedMaskSession()
        messages = queue.Queue()
        session._read_messages(SimpleNamespace(stdout=io.StringIO("x" * 65537 + "\n")), messages)
        error = messages.get_nowait()
        self.assertEqual(error["phase"], "protocol")
        self.assertIn("exceeded 65536", error["error"])


if __name__ == "__main__":
    unittest.main()
