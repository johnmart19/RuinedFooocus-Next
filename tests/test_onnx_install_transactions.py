"""Real per-target exclusion with deterministic held acquisitions; no installs."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.backends.onnx import acquisition, onnx_installer as native, windows_ml as winml
from ruinedfooocus.packaging.install_lock import installation_lock


class TransactionTests(unittest.TestCase):
    def test_target_lock_and_acquisition_use_one_frozen_runtime_choice(self):
        runtime = native.native_runtime("CUDA", "2.14.0+cu130")
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(native, "native_runtime", return_value=runtime) as choose,
                patch.object(
                    native, "native_runtime_path", return_value=Path(directory) / "runtime"
                ),
                patch.object(
                    native, "_install_native_runtime_locked", return_value="verified"
                ) as install,
            ):
                self.assertEqual(native.install_native_runtime("CUDA"), "verified")
        choose.assert_called_once_with("CUDA")
        self.assertIs(install.call_args.kwargs["runtime"], runtime)

    def exercise(self, module, invoke, entered, release):
        second_attempt = threading.Event()
        counter_lock = threading.Lock()
        attempts = 0

        @contextmanager
        def observed(target):
            nonlocal attempts
            with counter_lock:
                attempts += 1
                if attempts == 2:
                    second_attempt.set()
            with installation_lock(target):
                yield

        with (
            patch.object(module, "installation_lock", side_effect=observed),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            first = pool.submit(invoke, 0)
            try:
                self.assertTrue(entered.wait(5), "First acquisition did not reach held work")
                second = pool.submit(invoke, 1)
                self.assertTrue(
                    second_attempt.wait(5), "Second acquisition did not reach the real lock"
                )
                self.assertFalse(second.done())
            finally:
                release.set()
            first.result(timeout=10)
            second.result(timeout=10)

    def test_two_native_installers_publish_once_and_second_rechecks_verified_state(self):
        with tempfile.TemporaryDirectory() as folder:
            final = Path(folder) / "runtime"
            entered, release = threading.Event(), threading.Event()
            installs = []

            def run(command, **kwargs):
                if command[0] == "pip":
                    installs.append(command)
                    root = Path(command[1])
                    capi = root / "onnxruntime/capi"
                    capi.mkdir(parents=True)
                    (capi / "onnxruntime_pybind11_state.pyd").write_bytes(b"owned fixture")
                    entered.set()
                    self.assertTrue(release.wait(5))
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        dict(ok=True, version=native.NATIVE_RUNTIMES["CUDA"].version)
                    ),
                    stderr="",
                )

            with (
                patch.object(native, "skip_dependency_install", return_value=False),
                patch.object(native, "native_runtime_path", return_value=final),
                patch.object(native, "native_runtime", return_value=native.NATIVE_RUNTIMES["CUDA"]),
                patch.object(
                    native,
                    "native_install_plan",
                    side_effect=lambda n, root, *a, **k: ["pip", str(root)],
                ),
                patch.object(native, "missing_requirements", return_value=[]),
                patch.object(acquisition, "run", side_effect=run),
            ):
                self.exercise(
                    native, lambda _: native.install_native_runtime("CUDA"), entered, release
                )
            self.assertEqual(len(installs), 1)
            self.assertTrue(native._verified(final, native.NATIVE_RUNTIMES["CUDA"].version))
            self.assertFalse(list(Path(folder).glob(".onnx-prepare-*")))
            self.assertFalse(list(Path(folder).glob("*.previous-*")))

    def test_winml_catalogue_and_manifest_merge_serialize_across_vendors_and_recheck_duplicates(
        self,
    ):
        for names in (
            ("MIGraphXExecutionProvider", "OpenVINOExecutionProvider"),
            ("MIGraphXExecutionProvider",) * 2,
        ):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                manifest = root / "provider.json"
                entered, release = threading.Event(), threading.Event()
                installs = []
                records = {}
                for name in set(names):
                    library = root / (name + ".dll")
                    library.write_bytes(name.encode())
                    records[name] = dict(
                        name=name,
                        library_path=str(library),
                        sha256=hashlib.sha256(name.encode()).hexdigest(),
                    )

                def run(command, **kwargs):
                    if command[0] == "pip":
                        installs.append(command)
                        if len(installs) == 1:
                            entered.set()
                            self.assertTrue(release.wait(5))
                        return SimpleNamespace(returncode=0)
                    name = command[command.index("--ep-name") + 1]
                    return SimpleNamespace(returncode=0, stdout=json.dumps(records[name]))

                with (
                    patch.object(winml, "manifest_path", return_value=manifest),
                    patch.object(winml.platform, "system", return_value="Windows"),
                    patch(
                        "ruinedfooocus.packaging.startup_options.skip_dependency_install",
                        return_value=False,
                    ),
                    patch.object(winml, "install_plan", return_value=["pip"]),
                    patch.object(acquisition, "run", side_effect=run),
                ):
                    self.exercise(
                        winml,
                        lambda index: winml.install_provider(names[index], "GPU"),
                        entered,
                        release,
                    )
                document = json.loads(manifest.read_text())
                self.assertEqual(set(document["providers"]), set(names))
                self.assertEqual(len(installs), len(set(names)))
                self.assertFalse(list(root.glob("provider-*.tmp")))


if __name__ == "__main__":
    unittest.main()
