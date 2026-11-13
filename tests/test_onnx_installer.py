"""CPU-only plans and isolated runtime integrity; no packages or GPU work."""

import hashlib, json, os, sys, tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import ruinedfooocus.backends.onnx.onnx_installer as installer
from ruinedfooocus.backends.onnx import acquisition
from types import SimpleNamespace
from unittest.mock import Mock
from contextlib import contextmanager


class OnnxInstallerTests(unittest.TestCase):
    def test_default_migraphx_install_produces_evidence_and_binds_discovered_ep_device(self):
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / "plugin"
            prepared = SimpleNamespace(
                target=SimpleNamespace(
                    system="Linux", machine="x86_64", python_abi="cp312", environment="native"
                ),
                adapter=dict(adapter_id="physical-pci", vendor="amd", gfx="gfx1100"),
                evidence=dict(artifacts=(), closure_complete=True, native_payload=True),
            )

            @contextmanager
            def prepare(*args, **kwargs):
                yield prepared

            def run(command, **kwargs):
                if command[0] == "pip":
                    root = Path(command[1])
                    (root / "onnxruntime/capi").mkdir(parents=True)
                    (root / "onnxruntime/capi/onnxruntime_pybind11_state.so").write_bytes(
                        b"fixture"
                    )
                    (root / "plugin.so").write_bytes(b"plugin")
                    return SimpleNamespace(returncode=0)
                if "--list-devices" in command:
                    return SimpleNamespace(
                        returncode=0, stdout=json.dumps({"devices": [{"id": "a" * 24}]})
                    )
                root = Path(command[command.index("--runtime") + 1])
                self.assertEqual(command[command.index("--ep-device") + 1], "a" * 24)
                report = dict(
                    ok=True,
                    version="1.29.0",
                    device_identity="a" * 24,
                    plugin_libraries=[
                        dict(
                            name="MIGraphXExecutionProvider",
                            path="plugin.so",
                            sha256=hashlib.sha256((root / "plugin.so").read_bytes()).hexdigest(),
                        )
                    ],
                )
                return SimpleNamespace(returncode=0, stdout=json.dumps(report), stderr="")

            with (
                patch.object(acquisition, "prepare_migraphx", side_effect=prepare) as producer,
                patch.object(acquisition, "run", side_effect=run),
                patch.object(installer, "skip_dependency_install", return_value=False),
                patch.object(installer, "native_runtime_path", return_value=final),
                patch.object(installer, "_verified", return_value=False),
                patch.object(
                    installer,
                    "native_install_plan",
                    side_effect=lambda n, root, *a, **k: ["pip", str(root)],
                ),
                patch.object(installer, "migraphx_decision", return_value={"eligible": True}),
                patch.object(installer, "missing_requirements", return_value=[]),
            ):
                installer.install_native_runtime("MIGraphX")
            producer.assert_called_once_with(None, check_cancelled=None)
            marker = json.loads((final / "verified.json").read_text())
            self.assertEqual(marker["adapter"]["onnx_device_id"], "a" * 24)
            self.assertEqual(marker["provider_device_id"], "a" * 24)
            self.assertEqual(marker["qualification"], "canary_only_not_model_qualified")

    def test_cancellable_acquisition_terminates_only_its_owned_child(self):
        class Cancelled(Exception):
            pass

        process, job = Mock(), Mock()
        process.poll.return_value = None
        with (
            patch("ruinedfooocus.core.child_process.start_process", return_value=(process, job)),
            self.assertRaises(Cancelled),
        ):
            acquisition.run(["owned-child"], check_cancelled=Mock(side_effect=[None, Cancelled()]))
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)
        job.close.assert_called_once_with()

    def test_native_acquisition_publishes_only_after_probe_and_keeps_previous_folder(self):
        for failure in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                final = Path(folder) / "runtime"
                final.mkdir()
                (final / "previous.txt").write_text("retained")

                def run(command, **kwargs):
                    if command[0] == "pip":
                        root = Path(command[1])
                        capi = root / "onnxruntime/capi"
                        capi.mkdir(parents=True)
                        (capi / "onnxruntime_pybind11_state.pyd").write_bytes(b"fixture binary")
                        return SimpleNamespace(returncode=0)
                    return SimpleNamespace(
                        returncode=1 if failure else 0,
                        stdout=json.dumps(
                            dict(
                                ok=not failure,
                                version=installer.NATIVE_RUNTIMES["CUDA"].version,
                                error="probe rejected",
                            )
                        ),
                        stderr="",
                    )

                with (
                    patch.object(installer, "skip_dependency_install", return_value=False),
                    patch.object(installer, "native_runtime_path", return_value=final),
                    patch.object(installer, "_verified", return_value=False),
                    patch.object(
                        installer,
                        "native_install_plan",
                        side_effect=lambda n, root, *a, **k: ["pip", str(root)],
                    ),
                    patch.object(installer, "missing_requirements", return_value=[]),
                    patch.object(acquisition, "run", side_effect=run),
                ):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, "probe rejected"):
                            installer.install_native_runtime("CUDA")
                        self.assertEqual((final / "previous.txt").read_text(), "retained")
                    else:
                        installer.install_native_runtime("CUDA")
                        self.assertTrue((final / "verified.json").is_file())
                        previous = list(Path(folder).glob("runtime.previous-*"))
                        self.assertEqual(len(previous), 1)
                        self.assertEqual((previous[0] / "previous.txt").read_text(), "retained")
                self.assertFalse(list(Path(folder).glob(".onnx-prepare-*")))

    def test_install_skip_policy_prevents_cache_writes_and_provider_probes(self):
        for argv, offline in [([], "1"), (["--offline"], ""), (["--skip-dependency-install"], "")]:
            with (
                self.subTest(argv=argv, offline=offline),
                patch.object(sys, "argv", ["test", *argv]),
                patch.dict(os.environ, RF_OFFLINE=offline),
                patch.object(installer, "runtime_path") as location,
                patch.object(installer.subprocess, "run") as execute,
            ):
                with self.assertRaisesRegex(RuntimeError, "installation is disabled"):
                    installer.install_runtime()
                location.assert_not_called()
                execute.assert_not_called()

    def test_optional_install_is_isolated_pinned_and_does_not_replace_cpu_ort(self):
        with (
            patch.object(installer.platform, "system", return_value="Windows"),
            patch.object(installer.platform, "machine", return_value="AMD64"),
        ):
            plan = installer.install_plan("isolated-folder")
        self.assertIn("--target", plan)
        self.assertIn("--no-deps", plan)
        self.assertIn("onnxruntime-windowsml==" + installer.VERSION, plan)
        self.assertNotIn("onnxruntime", plan)
        with (
            patch.object(installer.platform, "system", return_value="Windows"),
            patch.object(installer.platform, "machine", return_value="AMD64"),
            patch.object(installer.sys, "version_info", (3, 14)),
        ):
            self.assertIn(
                "onnxruntime-windowsml==" + installer.VERSION,
                installer.install_plan("isolated-directml"),
            )
            cuda_plan = installer.native_install_plan("CUDA", "isolated-cuda")
            self.assertTrue(
                any(
                    value.startswith("onnxruntime-gpu") and "1.32.0" in value for value in cuda_plan
                )
            )
            with self.assertRaisesRegex(RuntimeError, "Python"):
                installer.native_install_plan("OpenVINO", "isolated-openvino")
        with patch.object(installer.platform, "system", return_value="Linux"):
            with self.assertRaises(RuntimeError):
                installer.install_plan()

    def test_marker_rejects_changed_binary_or_python_abi(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "binary.pyd").write_bytes(b"original")
            marker = dict(
                version=installer.VERSION,
                python=list(sys.version_info[:2]),
                binary="binary.pyd",
                sha256=hashlib.sha256(b"original").hexdigest(),
            )
            (root / "verified.json").write_text(json.dumps(marker))
            with patch.object(installer, "runtime_path", return_value=root):
                self.assertTrue(installer.runtime_available())
                (root / "binary.pyd").write_bytes(b"changed")
                self.assertFalse(installer.runtime_available())
