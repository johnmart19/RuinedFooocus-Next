"""Portable provider policy and real child-process lifecycle without GPU models."""

from pathlib import Path
from contextlib import ExitStack
import json
import os
import subprocess
import sysconfig
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from weakref import WeakSet

from PIL import Image
import ruinedfooocus.features.image_processing.auxiliary_resources as resources
import ruinedfooocus.backends.onnx.onnx_installer as installer
import ruinedfooocus.backends.onnx.onnx_runtime as runtime
from ruinedfooocus.app import resource_lifecycle as life


class OnnxRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.ownership = ExitStack()
        self.addCleanup(self.ownership.close)
        item = life.JobRegistry().create()
        self.ownership.enter_context(life.job_scope(item))
        self.ownership.enter_context(life.admission.lease(item))

    def test_unmanaged_gpu_prediction_refuses_without_loading_or_cpu_retry(self):
        factory = Mock()
        session = runtime.NativeMaskSession(factory)
        token = life._CURRENT.set(None)
        try:
            with self.assertRaises(life.OwnershipError):
                session.predict(Image.new("RGB", (2, 2)))
        finally:
            life._CURRENT.reset(token)
        factory.assert_not_called()
        self.assertFalse(session.disabled)

    def test_real_worker_reports_initialization_error_before_device_registration(self):
        (installer.ROOT / "tmp").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=installer.ROOT / "tmp") as folder:
            command = installer.worker_command(sysconfig.get_path("purelib"))
            command += [
                "--provider",
                "Windows ML",
                "--manifest",
                str(Path(folder) / "missing.json"),
            ]
            result = subprocess.run(
                command,
                env=dict(os.environ, RF_DATA_DIR=folder),
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(result.returncode, 1)
        error = json.loads(result.stdout.strip().splitlines()[-1])["error"]
        self.assertIn("Windows ML installation record is missing or changed", error)
        self.assertIn("RuntimeError", error)
        progress = [json.loads(line) for line in result.stdout.splitlines()[:-1]]
        self.assertEqual(
            [message["phase"] for message in progress],
            ["loading runtime", "selecting provider and device"],
        )

    def test_native_gpu_session_reuses_then_releases_and_recreates(self):
        factory = Mock()
        session = runtime.NativeMaskSession(factory)
        image = Image.new("RGB", (2, 2))
        with patch.object(runtime, "release_torch_weights") as release:
            session.predict(image)
            session.predict(image)
            factory.assert_called_once_with()
            release.assert_called_once_with()
            session.close()
            session.predict(image)
            self.assertEqual(release.call_count, 2)
        self.assertEqual(factory.call_count, 2)

    def test_native_provider_failures_retry_and_cache_cpu_without_losing_options(self):
        from onnxruntime.capi.onnxruntime_pybind11_state import (
            EPFail,
            EngineError,
            Fail,
            RuntimeException,
        )

        image = Image.new("RGB", (2, 2))
        for error in (EPFail, EngineError, Fail, RuntimeException):
            for stage in ("load", "predict"):
                with self.subTest(error=error, stage=stage):
                    factory = Mock()
                    if stage == "load":
                        factory.side_effect = error("GPU unavailable")
                    else:
                        factory.return_value.predict.side_effect = error("GPU allocation failed")
                    cpu = Mock()
                    cpu.predict.return_value = ["CPU mask"]
                    rembg = SimpleNamespace(new_session=Mock(return_value=cpu))
                    session = runtime.NativeMaskSession(factory)
                    with (
                        patch.object(runtime, "release_torch_weights") as release,
                        patch.dict("sys.modules", {"rembg": rembg}),
                    ):
                        self.assertEqual(
                            session.predict(image, "option", variant=True), ["CPU mask"]
                        )
                        session.predict(image, "option", variant=True)
                    rembg.new_session.assert_called_once_with(providers=["CPUExecutionProvider"])
                    cpu.predict.assert_called_with(image, "option", variant=True)
                    factory.assert_called_once_with()
                    release.assert_called_once_with()

    def test_native_input_errors_and_cancellation_do_not_retry(self):
        from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

        for error in (InvalidArgument, ValueError):
            session = runtime.NativeMaskSession(Mock(side_effect=error("invalid input")))
            with patch.object(runtime, "release_torch_weights"), self.assertRaises(error):
                session.predict(Image.new("RGB", (2, 2)))
            self.assertFalse(session.disabled)
        session = runtime.NativeMaskSession(Mock())
        session.check_cancelled = Mock(side_effect=RuntimeError("cancelled"))
        with (
            patch.object(runtime, "release_torch_weights") as release,
            self.assertRaisesRegex(RuntimeError, "cancelled"),
        ):
            session.predict(Image.new("RGB", (2, 2)))
        release.assert_not_called()
        session.factory.assert_not_called()

    def test_cpu_cuda_and_unavailable_windows_runtime_keep_native_fallbacks(self):
        import onnxruntime as ort

        rembg = SimpleNamespace(new_session=Mock())
        rembg.new_session.return_value.inner_session.get_providers.return_value = [
            "CUDAExecutionProvider"
        ]
        with (
            patch.object(
                ort, "get_available_providers", return_value=["CUDAExecutionProvider"]
            ) as providers,
            patch.dict(
                "sys.modules",
                {
                    "rembg": rembg,
                    "ruinedfooocus.app.arguments": SimpleNamespace(
                        args=SimpleNamespace(directml=None)
                    ),
                },
            ),
            patch.object(runtime, "runtime_available", return_value=False),
            patch.object(
                runtime, "create_audited_mask_session", return_value=rembg.new_session.return_value
            ) as audited,
            patch.object(runtime, "release_torch_weights"),
        ):
            runtime.new_session(cpu=True)
            rembg.new_session.assert_called_with(providers=["CPUExecutionProvider"])
            native = runtime.new_session(preference="Auto")
            native.predict(Image.new("RGB", (2, 2)))
            audited.assert_called_once_with(
                "CUDAExecutionProvider", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            providers.return_value = ["CPUExecutionProvider"]
            runtime.new_session(preference="Windows GPU")
            rembg.new_session.assert_called_with(providers=["CPUExecutionProvider"])

    def test_explicit_directml_does_not_use_an_installed_cuda_provider(self):
        import onnxruntime as ort

        rembg = SimpleNamespace(new_session=Mock())
        args = SimpleNamespace(directml=2)
        with (
            patch.object(ort, "get_available_providers", return_value=["CUDAExecutionProvider"]),
            patch.dict(
                "sys.modules",
                {"rembg": rembg, "ruinedfooocus.app.arguments": SimpleNamespace(args=args)},
            ),
            patch.object(runtime.platform, "system", return_value="Windows"),
            patch.object(runtime, "runtime_available", return_value=True),
        ):
            selected = runtime.new_session(preference="Auto")
            self.assertIsInstance(selected, runtime.IsolatedMaskSession)
            self.assertEqual(selected.device_id, 2)
            rembg.new_session.assert_not_called()
            with patch.object(runtime, "runtime_available", return_value=False):
                runtime.new_session(preference="Auto")
            rembg.new_session.assert_called_once_with(providers=["CPUExecutionProvider"])

    def test_gpu_worker_failure_retries_cpu_once_and_stays_on_cpu(self):
        session = runtime.IsolatedMaskSession()
        rembg = SimpleNamespace(new_session=Mock())
        rembg.new_session.return_value.predict.return_value = ["CPU mask"]
        with (
            patch.object(session, "_start", side_effect=runtime.GPUWorkerError("unavailable")),
            patch.dict("sys.modules", {"rembg": rembg}),
        ):
            self.assertEqual(session.predict(Image.new("RGB", (4, 4))), ["CPU mask"])
            self.assertEqual(session.predict(Image.new("RGB", (4, 4))), ["CPU mask"])
        self.assertTrue(session.disabled)
        rembg.new_session.assert_called_once_with(providers=["CPUExecutionProvider"])

    def test_cancellation_propagates_without_cpu_retry(self):
        class Cancelled(Exception):
            pass

        session = runtime.IsolatedMaskSession()
        session.check_cancelled = Mock(side_effect=Cancelled())
        with patch.object(session, "close") as close, self.assertRaises(Cancelled):
            session._wait(1)
        close.assert_called_once()
        self.assertFalse(session.disabled)

    def test_dead_worker_pipe_cleanup_does_not_block_fallback(self):
        session = runtime.IsolatedMaskSession()
        process = Mock()
        job = session.job = Mock()
        process.poll.return_value = 1
        process.stdin.close.side_effect = OSError(22, "Invalid argument")
        session.process = process
        session.close()
        self.assertIsNone(session.process)
        process.stdout.close.assert_called_once()
        job.close.assert_called_once_with()
        self.assertIsNone(session.job)

    def test_resource_groups_release_gpu_before_other_work(self):
        resource = Mock(resource_group="rembg")
        with patch.object(resources, "_resources", WeakSet()):
            resources.register(resource)
            resources.release_except("rembg")
            resource.close.assert_not_called()
            resources.release_except()
            resource.close.assert_called_once()

    def test_real_ipc_reuses_child_and_close_releases_it_then_allows_restart(self):
        script = """import sys,json,site
site.addsitedir(sys.argv[sys.argv.index('--site-dir')+1])
from PIL import Image
from pathlib import Path
print(json.dumps({'event':'initialization_progress','phase':'creating native model session','elapsed_seconds':0.01}),flush=True)
print(json.dumps({'ready':True}),flush=True)
for line in sys.stdin:
 request=json.loads(line)
 with Image.open(request['input']) as image:
  mask=Image.new('L',image.size,127)
 path=Path(request['output'])/'mask.png'
 mask.save(path)
 print(json.dumps({'masks':[str(path)]}),flush=True)
"""
        (installer.ROOT / "tmp").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=installer.ROOT / "tmp") as folder:
            root = Path(folder)
            worker = root / "fake_worker.py"
            worker.write_text(script)
            memory = SimpleNamespace(unload_all_models=Mock(), soft_empty_cache=Mock())
            with (
                patch.dict("sys.modules", {"comfy": SimpleNamespace(model_management=memory)}),
                patch.object(runtime, "ROOT", root),
                patch.object(runtime, "WORKER", worker),
            ):
                session = runtime.IsolatedMaskSession()
                try:
                    self.assertEqual(
                        session.predict(Image.new("RGB", (4, 3)))[0].getpixel((0, 0)), 127
                    )
                    process = session.process
                    self.assertEqual(
                        session.initialization_progress["phase"], "creating native model session"
                    )
                    session.predict(Image.new("RGB", (5, 2)))
                    self.assertIs(session.process, process)
                    session.close()
                    self.assertIsNotNone(process.poll())
                    self.assertEqual(session.predict(Image.new("RGB", (2, 2)))[0].size, (2, 2))
                    self.assertNotEqual(session.process.pid, process.pid)
                finally:
                    session.close()
                self.assertFalse(list((root / "cache/onnx/jobs").iterdir()))
                self.assertFalse(list((root / "cache/onnx").glob("worker-*")))


if __name__ == "__main__":
    unittest.main()
