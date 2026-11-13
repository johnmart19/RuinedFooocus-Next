"""Legacy DirectML installation has bounded owned subprocesses, not GPU tests."""

from concurrent.futures import CancelledError
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.backends.onnx import onnx_installer as installer


class LegacyOnnxInstallCancellationTests(unittest.TestCase):
    def test_cancel_before_acquisition_does_not_create_a_runtime_folder(self):
        check = Mock(side_effect=CancelledError())
        with (
            patch.object(installer, "skip_dependency_install", return_value=False),
            patch.object(installer, "runtime_path") as path,
        ):
            with self.assertRaises(CancelledError):
                installer.install_runtime(check_cancelled=check)
        path.assert_not_called()

    def fixture(self, folder):
        binary = Path(folder) / "onnxruntime/capi/onnxruntime_pybind11_state.pyd"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"private fixture, never loaded")
        return binary

    def test_install_and_canary_use_the_owned_checker_and_separate_timeouts(self):
        with tempfile.TemporaryDirectory() as folder:
            self.fixture(folder)
            check = Mock()
            with (
                patch.object(installer, "skip_dependency_install", return_value=False),
                patch.object(installer, "runtime_path", return_value=Path(folder)),
                patch.object(installer, "runtime_available", return_value=False),
                patch.object(installer, "install_plan", return_value=["fixture-install"]),
                patch("importlib.metadata.distributions", return_value=[]),
                patch(
                    "ruinedfooocus.backends.onnx.acquisition.run",
                    side_effect=[
                        SimpleNamespace(returncode=0),
                        SimpleNamespace(
                            stdout=json.dumps(
                                {"ok": True, "version": installer.VERSION.split(".2026")[0]}
                            )
                        ),
                    ],
                ) as run,
            ):
                result = installer.install_runtime(check_cancelled=check)
            self.assertEqual([call.kwargs["timeout"] for call in run.call_args_list], [300, 60])
            self.assertTrue(
                all(call.kwargs["check_cancelled"] is check for call in run.call_args_list)
            )
            self.assertIn("model workload unqualified", result)
            self.assertTrue((Path(folder) / "verified.json").is_file())

    def test_cancellation_after_canary_preserves_the_previous_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            self.fixture(folder)
            marker = Path(folder) / "verified.json"
            marker.write_bytes(b"prior marker")
            with (
                patch.object(installer, "skip_dependency_install", return_value=False),
                patch.object(installer, "runtime_path", return_value=Path(folder)),
                patch.object(installer, "runtime_available", return_value=True),
                patch("importlib.metadata.distributions", return_value=[]),
                patch(
                    "ruinedfooocus.backends.onnx.acquisition.run",
                    return_value=SimpleNamespace(
                        stdout=json.dumps(
                            {"ok": True, "version": installer.VERSION.split(".2026")[0]}
                        )
                    ),
                ),
            ):
                with self.assertRaises(CancelledError):
                    installer.install_runtime(
                        check_cancelled=Mock(side_effect=[None, CancelledError()])
                    )
            self.assertEqual(marker.read_bytes(), b"prior marker")
