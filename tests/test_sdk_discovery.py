"""SDK path discovery and explicit diagnostics using fake SDKs, never devices."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.backends.llama import llama_installer, llama_runtime_info


class SdkDiscoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rf-sdk-metadata-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.wsl = False
        exists = Path.exists
        device_path = patch.object(
            Path, "exists", lambda path: self.wsl if path.as_posix() == "/dev/dxg" else exists(path)
        )
        device_path.start()
        self.addCleanup(device_path.stop)
        environment = patch.dict(
            os.environ,
            RF_DATA_DIR=str(self.root),
            LLAMA_SERVER="",
            LD_LIBRARY_PATH="inherited-library-path",
        )
        environment.start()
        self.addCleanup(environment.stop)

    def sdk(self, suffix=""):
        directory = self.root / "cache/llama.cpp" / ("rocm-10.0" + suffix)
        for package in ("_rocm_sdk_core", "_rocm_sdk_libraries"):
            (directory / "lib/python3.12/site-packages" / package / "lib").mkdir(parents=True)
        (directory / "gfx1100.ready").touch()
        return directory

    def runtime_binary(self):
        binary = self.root / "cache/llama.cpp/b1234/rocm-gfx1100/llama-server"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"private metadata fixture; never executed")
        return binary

    def diagnostic(self, *, name="posix", path_error=None):
        calls = []

        def run(command, env=None):
            calls.append((command, dict(env or {})))
            if path_error and "--version" in command:
                raise path_error
            return (
                "RF_RUNTIME_INFO=" + json.dumps({"version": "fixture", "devices": []})
                if "-c" in command
                else "fixture"
            )

        scope_os = SimpleNamespace(name=name, environ=dict(os.environ))
        with (
            patch.object(llama_runtime_info, "os", scope_os),
            patch.object(llama_runtime_info, "_run", side_effect=run),
            patch.object(
                llama_installer,
                "ensure_rocm_libraries",
                side_effect=AssertionError("SDK acquisition"),
            ),
            patch(
                "ruinedfooocus.packaging.rocm_installer.detect_rocm_targets",
                side_effect=AssertionError("device query"),
            ),
        ):
            result = llama_runtime_info._inspect_runtime_info()
        return result, calls

    def test_linux_refresh_uses_same_published_sdk_as_runtime_without_detection(self):
        base = self.sdk()
        active = self.sdk("-new")
        pointer = base.parent / (base.name + ".active.json")
        pointer.write_text(json.dumps({"directory": str(active)}), encoding="utf-8")
        self.runtime_binary()
        result, calls = self.diagnostic()
        environments = [env for command, env in calls if "--list-devices" in command]
        self.assertEqual(len(environments), 1)
        library_path = environments[0]["LD_LIBRARY_PATH"]
        self.assertIn(str(active / "lib/python3.12/site-packages/_rocm_sdk_core/lib"), library_path)
        self.assertNotIn(
            str(base / "lib/python3.12/site-packages/_rocm_sdk_core/lib"), library_path
        )
        with (
            patch.object(llama_installer.platform, "system", return_value="Linux"),
            patch.object(llama_installer, "rocm_target", return_value="gfx1100"),
        ):
            actual = llama_installer.runtime_environment(True, backend="ROCm", acquire=False)
        self.assertEqual(environments[0]["LD_LIBRARY_PATH"], actual["LD_LIBRARY_PATH"])
        self.assertIn("rocm-gfx1100", result)

    def test_wsl_diagnostic_reuses_the_serving_bridge_without_acquisition(self):
        self.wsl = True
        self.sdk()
        self.runtime_binary()
        bridge = self.root / "bridge"
        bridge.mkdir()
        with (
            patch(
                "ruinedfooocus.backends.llama.wsl_bridge.library_directory", return_value=bridge
            ) as select,
            patch.object(llama_installer.platform, "system", return_value="Linux"),
            patch.object(llama_installer, "rocm_target", return_value="gfx1100"),
        ):
            _, calls = self.diagnostic()
            actual = llama_installer.runtime_environment(True, backend="ROCm", acquire=False)
        environments = [env for command, env in calls if "--list-devices" in command]
        self.assertEqual(environments[0]["LD_LIBRARY_PATH"], actual["LD_LIBRARY_PATH"])
        self.assertIn(str(bridge), actual["LD_LIBRARY_PATH"])
        self.assertLess(
            actual["LD_LIBRARY_PATH"].replace("\\", "/").index("_rocm_sdk_core/lib"),
            actual["LD_LIBRARY_PATH"].index(str(bridge)),
        )
        self.assertEqual(select.call_count, 2)
        self.assertTrue(all(call.kwargs == {"acquire": False} for call in select.call_args_list))

    def test_diagnostic_falls_back_to_legacy_sdk_when_active_pointer_is_unusable(self):
        base = self.sdk()
        self.runtime_binary()
        pointer = base.parent / (base.name + ".active.json")
        for value in (
            None,
            "[]",
            json.dumps({"directory": None}),
            json.dumps({"directory": str(self.root / "outside-cache")}),
        ):
            if value is not None:
                pointer.write_text(value, encoding="utf-8")
            _, calls = self.diagnostic()
            environment = next(env for command, env in calls if "--version" in command)
            self.assertIn(
                str(base / "lib/python3.12/site-packages/_rocm_sdk_core/lib"),
                environment["LD_LIBRARY_PATH"],
            )
            self.assertNotIn("outside-cache", environment["LD_LIBRARY_PATH"])

    def test_query_failure_is_visible_and_does_not_trigger_installation(self):
        self.runtime_binary()
        for error in (
            subprocess.TimeoutExpired("fixture", 15),
            ValueError("malformed paths"),
            TypeError("malformed library list"),
        ):
            with self.subTest(error=type(error).__name__):
                result, _ = self.diagnostic(path_error=error)
                self.assertIn("could not query this build's devices", result)

    def test_windows_native_sdk_api_precedes_stale_system_paths_without_global_mutation(self):
        directory = self.root / "wheel-owned/bin"
        directory.mkdir(parents=True)
        inherited = {"PATH": "old-system-sdk/bin", "HIP_PATH": "old-system-sdk"}
        snapshot = dict(os.environ)
        with (
            patch.object(
                llama_installer.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout=json.dumps([str(directory)])),
            ) as execute,
            patch.object(llama_installer.subprocess, "CREATE_NO_WINDOW", 0, create=True),
            patch.object(
                llama_installer,
                "ensure_rocm_libraries",
                side_effect=AssertionError("SDK acquisition"),
            ),
        ):
            result = llama_installer.windows_rocm_environment(dict(inherited), install=False)
        self.assertEqual(result["PATH"], os.pathsep.join((str(directory), inherited["PATH"])))
        self.assertEqual(result["HIP_PATH"], inherited["HIP_PATH"])
        self.assertEqual(dict(os.environ), snapshot)
        command = execute.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertIn("find_libraries('amdhip64', 'hipblas', 'hipblaslt')", command[-1])
        self.assertNotIn("initialize_process", command[-1])
        self.assertNotIn("rocm-sdk test", command[-1])
        self.assertEqual(execute.call_args.kwargs["timeout"], 30)

    def test_missing_windows_sdk_is_unknown_without_acquisition(self):
        environment = {"PATH": "unchanged", "HIP_PATH": "system-sdk"}
        with (
            patch.object(
                llama_installer.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=1, stdout=""),
            ),
            patch.object(llama_installer.subprocess, "CREATE_NO_WINDOW", 0, create=True),
            patch.object(
                llama_installer,
                "ensure_rocm_libraries",
                side_effect=AssertionError("SDK acquisition"),
            ),
            patch.object(
                llama_installer, "rocm_target", side_effect=AssertionError("device query")
            ),
        ):
            self.assertEqual(
                llama_installer.windows_rocm_environment(dict(environment), install=False),
                environment,
            )

    def test_cpu_and_other_vendor_native_routes_never_discover_amd_sdk(self):
        for backend, gpu in (
            ("ROCm", False),
            ("CPU", False),
            ("Vulkan", True),
            ("CUDA 12.8", True),
        ):
            with (
                self.subTest(backend=backend, gpu=gpu),
                patch.object(llama_installer.platform, "system", return_value="Windows"),
                patch.object(
                    llama_installer,
                    "windows_rocm_environment",
                    side_effect=AssertionError("AMD SDK accessed"),
                ),
                patch(
                    "ruinedfooocus.packaging.rocm_installer.detect_rocm_targets",
                    side_effect=AssertionError("AMD target query"),
                ),
                patch(
                    "ruinedfooocus.packaging.vendor_admission.linux_amd_adapters",
                    side_effect=AssertionError("AMD adapter query"),
                ),
            ):
                self.assertEqual(
                    llama_installer.runtime_environment(gpu, backend=backend), os.environ.copy()
                )

    def test_native_query_timeout_is_bounded(self):
        with patch.object(
            llama_runtime_info,
            "run_bounded",
            side_effect=subprocess.TimeoutExpired("fixture", 15),
        ) as execute:
            with self.assertRaises(subprocess.TimeoutExpired):
                llama_runtime_info._run(["owned-fixture", "--list-devices"])
        self.assertEqual(execute.call_args.kwargs["timeout"], 15)
        self.assertEqual(execute.call_args.kwargs["max_output"], 16 * 1024)


if __name__ == "__main__":
    unittest.main()
