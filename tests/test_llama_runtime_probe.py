"""Independent device-probe provider; all native work stays mocked."""

from concurrent.futures import CancelledError
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.backends.llama import runtime_probe


class RuntimeProbeTests(unittest.TestCase):
    def test_reports_only_initialized_expected_backend_with_bounded_owned_child(self):
        check = Mock()
        events = []
        environment = {"RF_FIXTURE": "inert"}
        output = b"Available devices:\nVulkan0: fixture\nVulkan1: fixture\nCUDA0: fixture\n"
        with patch.object(runtime_probe, "run_bounded", return_value=(output, b"")) as run:
            self.assertEqual(
                runtime_probe.probe_runtime(
                    Path("inert"),
                    environment,
                    "Vulkan",
                    check_cancelled=check,
                    progress_callback=events.append,
                ),
                "Vulkan/CUDA",
            )
            run.assert_called_with(
                ["inert", "--list-devices"],
                env=environment,
                timeout=30,
                max_output=64 * 1024,
                check_cancelled=check,
            )
            self.assertEqual(
                events,
                [
                    {
                        "phase": "probing_device",
                        "message": "Checking the selected runtime's devices.",
                    }
                ],
            )
            self.assertIsNone(runtime_probe.probe_runtime(Path("inert"), environment, "ROCm"))
            run.return_value = (b"No usable devices found", b"")
            self.assertIsNone(runtime_probe.probe_runtime(Path("inert"), environment, "Vulkan"))

    def test_cancel_and_failure_classes_never_publish_a_device(self):
        for failure, expected in (
            (CancelledError(), CancelledError),
            (FileNotFoundError(2, "fixture missing dependency"), runtime_probe.RuntimeUnavailable),
            (RuntimeError("fixture failed device"), runtime_probe.RuntimeVerificationError),
        ):
            with (
                self.subTest(failure=type(failure).__name__),
                patch.object(runtime_probe, "run_bounded", side_effect=failure),
                self.assertRaises(expected),
            ):
                runtime_probe.probe_runtime(
                    Path("inert"), {}, "Vulkan", check_cancelled=Mock(), progress_callback=Mock()
                )

    def test_known_missing_loader_patterns_remain_distinct_from_device_crashes(self):
        for output in (
            b"private path: error while loading shared libraries: missing.so",
            b"private path: cannot open shared object file",
            b"private path: Vulkan loader library not found",
        ):
            for returncode, expected in (
                (1, runtime_probe.RuntimeUnavailable),
                (-11, runtime_probe.RuntimeVerificationError),
            ):
                failure = runtime_probe.BoundedProcessError(returncode, b"", output)
                with (
                    self.subTest(output=output, returncode=returncode),
                    patch.object(runtime_probe, "run_bounded", side_effect=failure),
                    self.assertRaises(expected) as caught,
                ):
                    runtime_probe.probe_runtime(Path("inert"), {}, "Vulkan")
                self.assertNotIn("private", str(caught.exception))
                self.assertNotIn("private", repr(caught.exception))
        failure = runtime_probe.BoundedProcessError(1, b"", b"private device failure")
        with (
            patch.object(runtime_probe, "run_bounded", side_effect=failure),
            self.assertRaises(runtime_probe.RuntimeVerificationError),
        ):
            runtime_probe.probe_runtime(Path("inert"), {}, "Vulkan")
