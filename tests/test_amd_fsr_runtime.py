"""AMD FSR runtime acquisition: pinned SDK files, refusal reasons and the one-time build.

Downloads and the compiler are replaced by stand-ins, so nothing is fetched or
compiled here. These checks do not show that AMD's files download or that the
helper compiles on a real machine.
"""

from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.backends.amd_fsr import runtime
from ruinedfooocus.features.image_processing import amd_fsr4


class Finished:
    def __init__(self, error=None):
        self.error = error

    def done(self):
        return True

    def result(self, timeout=None):
        if self.error is not None:
            raise self.error
        return None


class Service:
    def __init__(self, error=None):
        self.requests, self.error = [], error

    def submit(self, request):
        self.requests.append(request)
        return Finished(self.error)


class AmdFsrRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()

    def sdk(self):
        for relative in runtime.SDK_FILES:
            path = self.root / "sdk" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"stand-in")
        return self.root / "sdk"

    def test_every_sdk_file_is_pinned_to_one_commit_of_the_amd_repository(self):
        self.assertRegex(runtime.SDK_COMMIT, r"^[0-9a-f]{40}$")
        self.assertTrue(runtime.RAW.startswith("https://raw.githubusercontent.com/GPUOpen-"))
        self.assertTrue(runtime.RAW.endswith("/FidelityFX-SDK/" + runtime.SDK_COMMIT))
        for relative, (size, digest) in runtime.SDK_FILES.items():
            self.assertFalse(Path(relative).is_absolute() or ".." in relative, relative)
            self.assertGreater(size, 0)
            self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", digest), relative)
        names = {Path(relative).name for relative in runtime.SDK_FILES}
        self.assertTrue(set(runtime.DLLS) <= names)
        # The adapter and the builder must agree on what a complete runtime is.
        self.assertEqual(set(amd_fsr4.RUNTIME_FILES), {runtime.HELPER, *runtime.DLLS})
        self.assertTrue(runtime.SOURCE.is_file())

    def test_acquisition_is_refused_with_the_reason_it_cannot_happen_here(self):
        script = self.root / "vcvars64.bat"
        with patch.object(runtime.os, "name", "posix"):
            self.assertIn("Windows", runtime.acquisition_blocker())
        with patch.object(runtime.os, "name", "nt"):
            with patch.object(runtime, "offline", return_value=True):
                self.assertIn("Offline mode", runtime.acquisition_blocker())
            with (
                patch.object(runtime, "offline", return_value=False),
                patch.object(runtime, "compiler", return_value=None),
            ):
                self.assertIn("C++ build tools", runtime.acquisition_blocker())
            with (
                patch.object(runtime, "offline", return_value=False),
                patch.object(runtime, "compiler", return_value=script),
            ):
                self.assertIsNone(runtime.acquisition_blocker())

    def test_each_file_is_requested_from_the_pin_with_its_size_and_digest(self):
        service = Service()
        self.assertEqual(runtime.fetch_sdk(self.root / "sdk", service=service), self.root / "sdk")
        self.assertEqual(len(service.requests), len(runtime.SDK_FILES))
        for request, (relative, (size, digest)) in zip(
            service.requests, runtime.SDK_FILES.items(), strict=True
        ):
            self.assertEqual(request.url, runtime.RAW + "/" + relative)
            self.assertEqual(Path(request.destination), self.root / "sdk" / relative)
            self.assertEqual((request.expected_size, request.sha256), (size, digest))

        def cancelled():
            raise KeyboardInterrupt

        stopped = Service()
        with self.assertRaises(KeyboardInterrupt):
            runtime.fetch_sdk(self.root / "sdk", service=stopped, check_cancelled=cancelled)
        self.assertEqual(stopped.requests, [])
        with self.assertRaisesRegex(OSError, "connection reset"):
            runtime.fetch_sdk(self.root / "sdk", service=Service(OSError("connection reset")))

    def test_build_publishes_the_helper_last_and_keeps_the_licence_with_the_dlls(self):
        with self.assertRaisesRegex(RuntimeError, "not a complete AMD FSR SDK"):
            runtime.build(self.root / "absent", self.root / "out", script=self.root / "vc.bat")
        sdk, output = self.sdk(), self.root / "out"
        with (
            patch.object(runtime, "compiler", return_value=None),
            self.assertRaisesRegex(RuntimeError, "build tools were not found"),
        ):
            runtime.build(sdk, output)

        def failed(command, **options):
            return type("Result", (), {"returncode": 2, "stdout": "error C1083", "stderr": ""})()

        with (
            patch.object(runtime.subprocess, "run", failed),
            self.assertRaisesRegex(RuntimeError, "did not compile. error C1083"),
        ):
            runtime.build(sdk, output, script=self.root / "vc.bat")
        # A failed compile leaves no partial runtime that would look usable.
        self.assertEqual(list(output.iterdir()), [])

        def compiled(command, **options):
            batch = Path(command[-1]).read_text(encoding="utf-8")
            self.assertIn(str(runtime.SOURCE), batch)
            self.assertIn("amd_fidelityfx_loader_dx12.lib", batch)
            (Path(options["cwd"]) / runtime.HELPER).write_bytes(b"helper")
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch.object(runtime.subprocess, "run", compiled):
            self.assertEqual(runtime.build(sdk, output, script=self.root / "vc.bat"), output)
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {runtime.HELPER, runtime.LICENCE, *runtime.DLLS},
        )
        with patch.object(amd_fsr4.os, "name", "nt"):
            self.assertIsNone(amd_fsr4.unavailable_reason(output))

    def test_ensure_acquires_once_and_reports_failures_as_a_usable_message(self):
        output, steps = self.root / "out", []
        with (
            patch.object(runtime, "fetch_sdk", lambda **options: steps.append("fetch") or "sdk"),
            patch.object(runtime, "build", lambda sdk, target: steps.append("build") or target),
        ):
            self.assertEqual(runtime.ensure(output, ready=lambda: True), output)
            self.assertEqual(steps, [])
            with (
                patch.object(runtime, "acquisition_blocker", return_value="Offline mode: no."),
                self.assertRaisesRegex(ValueError, "Offline mode"),
            ):
                runtime.ensure(output, ready=lambda: False)
            self.assertEqual(steps, [])
            with patch.object(runtime, "acquisition_blocker", return_value=None):
                self.assertEqual(runtime.ensure(output, ready=lambda: False), output)
            self.assertEqual(steps, ["fetch", "build"])

        def broken(**options):
            raise OSError("connection reset")

        with (
            patch.object(runtime, "acquisition_blocker", return_value=None),
            patch.object(runtime, "fetch_sdk", broken),
            self.assertRaisesRegex(
                ValueError, "could not be prepared: connection reset FSR 1 needs no runtime"
            ),
        ):
            runtime.ensure(output)


if __name__ == "__main__":
    unittest.main()
