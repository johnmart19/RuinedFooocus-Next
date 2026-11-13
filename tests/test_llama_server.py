"""Native chat lifecycle checks with inert subprocess and HTTP replacements."""

import os
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import CancelledError
from unittest.mock import Mock, patch

from ruinedfooocus.backends.llama import llama_server, runtime_probe


class LlamaServerTests(unittest.TestCase):
    def test_auto_acquires_missing_dependencies_for_a_cached_vendor_binary(self):
        def environment(gpu, backend, *, acquire):
            if backend == "ROCm" and not acquire:
                raise RuntimeError("WSL bridge is missing")
            return {}

        with (
            patch.dict(os.environ, LLAMA_SERVER=""),
            patch.object(
                llama_server, "server_path", side_effect=lambda gpu, backend, **kw: Path(backend)
            ),
            patch.object(llama_server, "runtime_environment", side_effect=environment) as prepare,
            patch.object(llama_server, "_probe_runtime", side_effect=[None, "ROCm"]),
            patch(
                "ruinedfooocus.packaging.hardware_inventory.detect_gpus",
                return_value=[{"vendor": "amd"}],
            ),
            patch(
                "ruinedfooocus.packaging.hardware_inventory.llama_installed", return_value=["ROCm"]
            ),
        ):
            result = llama_server.resolve_runtime(True, "Auto")
        self.assertTrue(result[2])
        self.assertEqual(result[3], "ROCm")
        self.assertEqual(
            [(call.kwargs["backend"], call.kwargs["acquire"]) for call in prepare.call_args_list],
            [("ROCm", False), ("Vulkan", True), ("ROCm", True)],
        )

    def test_auto_acquires_native_rocm_before_cpu_when_wsl_vulkan_has_no_gpu(self):
        with (
            patch.dict(os.environ, LLAMA_SERVER=""),
            patch.object(
                llama_server, "server_path", side_effect=lambda gpu, backend, **kw: Path(backend)
            ) as select,
            patch.object(llama_server, "runtime_environment", return_value={}),
            patch.object(llama_server, "_probe_runtime", side_effect=[None, "ROCm"]),
            patch(
                "ruinedfooocus.packaging.hardware_inventory.detect_gpus",
                return_value=[{"vendor": "amd"}],
            ),
            patch("ruinedfooocus.packaging.hardware_inventory.llama_installed", return_value=[]),
        ):
            result = llama_server.resolve_runtime(True, "Auto")
        self.assertTrue(result[2])
        self.assertEqual(result[3], "ROCm")
        self.assertEqual(
            [call.kwargs["backend"] for call in select.call_args_list], ["Vulkan", "ROCm"]
        )
        self.assertTrue(all(call.kwargs["acquire"] for call in select.call_args_list))

    def test_auto_prefers_cached_vendor_gpu_before_acquiring_vulkan(self):
        with (
            patch.dict(os.environ, LLAMA_SERVER=""),
            patch.object(
                llama_server,
                "server_path",
                side_effect=lambda gpu, backend, **kwargs: Path(backend),
            ) as select,
            patch.object(llama_server, "runtime_environment", return_value={}),
            patch.object(
                runtime_probe,
                "run_bounded",
                return_value=(b"CUDA0: GPU", b""),
            ),
            patch(
                "ruinedfooocus.packaging.hardware_inventory.detect_gpus",
                return_value=[{"vendor": "nvidia"}],
            ),
            patch(
                "ruinedfooocus.packaging.hardware_inventory.llama_installed",
                return_value=[llama_server.CUDA_BACKEND],
            ),
        ):
            result = llama_server.resolve_runtime(True, "Auto")
        self.assertTrue(result[2])
        self.assertEqual(result[3], "CUDA")
        self.assertEqual(result[4], "")
        select.assert_called_once_with(True, backend=llama_server.CUDA_BACKEND, acquire=False)

    def test_auto_uses_vulkan_when_the_cached_rocm_runtime_has_no_device(self):
        with (
            patch.dict(os.environ, LLAMA_SERVER=""),
            patch.object(
                llama_server, "server_path", side_effect=lambda gpu, backend, **kw: Path(backend)
            ) as select,
            patch.object(llama_server, "runtime_environment", return_value={}),
            patch.object(llama_server, "_probe_runtime", side_effect=[None, "Vulkan"]),
            patch(
                "ruinedfooocus.packaging.hardware_inventory.detect_gpus",
                return_value=[{"vendor": "amd"}],
            ),
            patch(
                "ruinedfooocus.packaging.hardware_inventory.llama_installed", return_value=["ROCm"]
            ),
        ):
            result = llama_server.resolve_runtime(True, "Auto")
        self.assertEqual(result[3], "Vulkan")
        self.assertIn("ROCm: no visible GPU", result[4])
        self.assertEqual(
            [call.kwargs["backend"] for call in select.call_args_list], ["ROCm", "Vulkan"]
        )

    def test_auto_cpu_reuses_verified_binary_and_explains_absent_gpu(self):
        with (
            patch.dict(os.environ, LLAMA_SERVER=""),
            patch.object(llama_server, "server_path", return_value=Path("vulkan")) as select,
            patch.object(llama_server, "runtime_environment", return_value={}),
            patch.object(
                runtime_probe,
                "run_bounded",
                return_value=(b"", b""),
            ),
            patch("ruinedfooocus.packaging.hardware_inventory.detect_gpus", return_value=[]),
            patch("ruinedfooocus.packaging.hardware_inventory.llama_installed", return_value=[]),
            self.assertLogs(level="WARNING") as logs,
        ):
            result = llama_server.resolve_runtime(True, "Auto")
        self.assertFalse(result[2])
        self.assertEqual(result[3], "CPU")
        self.assertIn("no visible GPU", result[4])
        self.assertIn("Chat (llama.cpp)", logs.output[0])
        self.assertIn("Using CPU fallback", logs.output[0])
        select.assert_called_once()

    def test_device_error_stops_auto_without_cpu_retry(self):
        with (
            patch("ruinedfooocus.packaging.hardware_inventory.detect_gpus", return_value=[]),
            patch("ruinedfooocus.packaging.hardware_inventory.llama_installed", return_value=[]),
            patch.object(llama_server, "server_path", return_value=Path("vulkan")),
            patch.object(llama_server, "runtime_environment", return_value={}),
            patch.object(
                runtime_probe,
                "run_bounded",
                side_effect=RuntimeError("device lost"),
            ) as probe,
        ):
            with self.assertRaises(llama_server.RuntimeVerificationError):
                llama_server.resolve_runtime(True, "Auto")
        probe.assert_called_once()

    def test_cancel_before_runtime_selection_closes_session_and_never_starts_process(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "fixture.gguf"
            model.write_bytes(b"fixture")
            session = Mock(headers={})
            with (
                patch.object(llama_server.requests, "Session", return_value=session),
                patch.object(llama_server, "server_path") as select,
                patch.object(llama_server, "start_process") as start,
            ):
                with self.assertRaises(CancelledError):
                    llama_server.Server(model, {}, check_cancelled=Mock(side_effect=CancelledError))
            select.assert_not_called()
            start.assert_not_called()
            session.close.assert_called_once()

    def test_server_logs_follow_writable_data_from_a_different_working_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            model = root / "LFM3-1.7B-Q4_K_M.gguf"
            model.write_bytes(b"fixture only; no model weights")
            application, data = root / "application", root / "data"
            application.mkdir()
            process = Mock()
            process.poll.return_value = None
            session = Mock()
            session.headers = {}
            session.get.return_value.ok = True
            current = Path.cwd()
            server = None
            try:
                os.chdir(application)
                with (
                    patch.dict(os.environ, RF_DATA_DIR=str(data)),
                    patch.object(llama_server.requests, "Session", return_value=session),
                    patch.object(llama_server.socket, "socket") as socket,
                    patch.object(llama_server, "server_path", return_value=Path("llama-server")),
                    patch.object(llama_server, "runtime_environment", return_value={}),
                    patch.object(llama_server, "projector_path", return_value=None),
                    patch.object(
                        llama_server.subprocess,
                        "run",
                        side_effect=AssertionError(
                            "CPU mode must not enumerate accelerator devices"
                        ),
                    ),
                    patch.object(
                        llama_server, "start_process", return_value=(process, None)
                    ) as start,
                ):
                    socket.return_value.__enter__.return_value.getsockname.return_value = (
                        "127.0.0.1",
                        61234,
                    )
                    server = llama_server.Server(
                        model, {"llama_server_args": "--temp 0.7", "llm_n_predict": 0}, gpu=False
                    )
                    self.assertEqual(server.backend, "CPU")
                    self.assertNotIn("--reasoning-format", start.call_args.args[0])
                    self.assertIn("--jinja", start.call_args.args[0])
                    command = start.call_args.args[0]
                    self.assertEqual(command[command.index("--n-predict") + 1], "0")
                    temperatures = [
                        command[i + 1] for i, option in enumerate(command) if option == "--temp"
                    ]
                    self.assertEqual(temperatures, ["0.7"])
                    # This fixture has a familiar filename, not verified LFM3 content.
                    self.assertNotIn("--top-k", command)
                    self.assertNotIn("--repeat-penalty", command)
                    self.assertNotIn("--reasoning-budget", command)
                    self.assertTrue((data / "cache/llama.cpp/server-61234.log").is_file())
                    self.assertFalse((application / "cache").exists())
                    self.assertEqual(model.read_bytes(), b"fixture only; no model weights")
            finally:
                if server is not None:
                    server.close()
                os.chdir(current)
            process.terminate.assert_called_once()
            session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
