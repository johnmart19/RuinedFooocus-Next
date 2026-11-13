"""Runtime acquisition contracts with inert archives and mocked native work."""

from concurrent.futures import CancelledError
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from ruinedfooocus.backends.llama import llama_installer as installer, llama_update


class Response:
    status_code = 200
    is_redirect = False

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload

    def iter_content(self, chunk_size):
        yield self.payload[:10]
        yield self.payload[10:]


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(installer.platform, "system", return_value="Windows"))
        self.enterContext(patch.object(installer, "skip_dependency_install", return_value=False))
        self.get = self.enterContext(patch.object(installer.requests, "get"))
        self.package = "llama-b12400-bin-win-cpu-x64.zip"
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as bundle:
            bundle.writestr("llama-server.exe", b"inert executable; never started")
        self.body = stream.getvalue()

    def release(self):
        return {
            "tag_name": "b12400",
            "draft": False,
            "assets": [
                {
                    "name": self.package,
                    "state": "uploaded",
                    "size": len(self.body),
                    "digest": "sha256:" + hashlib.sha256(self.body).hexdigest(),
                    "browser_download_url": f"{llama_update._DOWNLOAD}/b12400/{self.package}",
                }
            ],
        }

    def prepare(self, release=None):
        self.get.side_effect = [Response(release or self.release()), Response(self.body)]

    def install(self, **kwargs):
        return installer._install_server(
            self.root / "new", self.package, version="b12400", backend="CPU", **kwargs
        )

    def test_reviewed_manifest_covers_every_advertised_prebuilt_and_cuda_companion(self):
        manifest = json.loads(
            (
                Path(installer.__file__).with_name("releases") / f"{installer.VERSION}.json"
            ).read_text()
        )
        assets = {item["name"]: item for item in manifest["assets"]}
        self.assertEqual(manifest["target_commitish"], "31953bcf74ba489a8bbb0c936c5fcf24fcf9485c")
        for system in ("Windows", "Linux"):
            for machine in ("x86_64", "aarch64"):
                with patch(
                    "ruinedfooocus.backends.llama.llama_build.built_server", return_value=None
                ):
                    choices = installer.backend_choices(system, machine)
                for backend in choices[1:]:
                    package = installer.package_name(
                        system, machine, backend != "CPU", backend=backend
                    )
                    self.assertIn(package, assets)
                    dependencies = [package]
                    if "-win-cuda-" in package:
                        dependencies.append(
                            "cudart-llama-bin-win-" + package.split("-bin-win-", 1)[1]
                        )
                    elif "-ubuntu-cuda-" in package:
                        dependencies.append("cudart-" + package)
                    for name in dependencies:
                        self.assertIn(name, assets)
                        self.assertLessEqual(assets[name]["size"], llama_update._MAX_ARCHIVE)
                        self.assertRegex(assets[name]["digest"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(
            installer.package_name("Windows", "AMD64", True, backend="CUDA 13.4", version="b12110"),
            "llama-b12110-bin-win-cuda-13.4-x64.zip",
        )

    def test_progress_exact_transfer_and_verified_cache_reuse(self):
        self.prepare()
        events = []
        executable = self.install(progress_callback=events.append)
        self.assertEqual(executable.read_bytes(), b"inert executable; never started")
        phases = [event["phase"] for event in events]
        self.assertIn("resolving_runtime", phases)
        self.assertIn("downloading_runtime", phases)
        self.assertEqual(phases[-1], "staging_runtime")
        final_transfer = next(event for event in events if event.get("received") == len(self.body))
        self.assertEqual(final_transfer["total"], len(self.body))
        self.assertEqual(
            final_transfer["runtime"], {"tag": "b12400", "backend": "CPU", "package": self.package}
        )
        self.get.reset_mock(side_effect=True)
        self.assertEqual(self.install(acquire=False), executable)
        self.get.assert_not_called()

    def test_cancel_transfer_and_publication_leave_old_verified_runtime_usable(self):
        old = self.root / "old"
        old.mkdir()
        binary = old / "llama-server.exe"
        binary.write_bytes(b"old verified fixture")
        (old / "rf-runtime.json").write_text(json.dumps(installer._runtime_manifest(old, "old")))
        for phase in ("downloading_runtime", "staging_runtime"):
            self.prepare()
            cancelled = False

            def progress(event):
                nonlocal cancelled
                if event["phase"] == phase:
                    cancelled = True

            def check():
                if cancelled:
                    raise CancelledError()

            with self.subTest(phase=phase), self.assertRaises(CancelledError):
                self.install(progress_callback=progress, check_cancelled=check)
            self.assertFalse((self.root / "new").exists())
            self.assertEqual(installer._install_server(old, "old", acquire=False), binary.resolve())
        self.prepare()
        self.assertTrue(self.install().is_file())

    def test_first_use_checks_digest_size_and_zip_paths_before_publishing(self):
        original = self.body
        for failure in ("digest", "size", "unsafe path"):
            self.body = original
            release = self.release()
            if failure == "digest":
                release["assets"][0]["digest"] = "sha256:" + "0" * 64
            elif failure == "size":
                release["assets"][0]["size"] -= 1
            else:
                stream = io.BytesIO()
                with zipfile.ZipFile(stream, "w") as bundle:
                    bundle.writestr("../escape.exe", b"inert")
                self.body = stream.getvalue()
                release = self.release()
            self.prepare(release)
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                self.install()
            self.assertFalse((self.root / "new").exists())
            self.assertFalse((self.root / "escape.exe").exists())

    def test_cancel_hash_verification_does_not_republish_or_acquire(self):
        self.prepare()
        self.install()
        self.get.reset_mock(side_effect=True)
        count = 0

        def check():
            nonlocal count
            count += 1
            if count == 2:
                raise CancelledError()

        with self.assertRaises(CancelledError):
            self.install(check_cancelled=check)
        self.get.assert_not_called()
        self.assertTrue(self.install(acquire=False).is_file())

    def test_pinned_cache_requires_exact_publisher_archive_identities(self):
        package = installer.package_name("Windows", "x86_64", False, backend="CPU")
        directory = self.root / "pinned"
        directory.mkdir()
        executable = directory / "llama-server.exe"
        executable.write_bytes(b"inert trusted offline fixture")
        manifest = installer._runtime_manifest(directory, package)
        marker = directory / "rf-runtime.json"
        marker.write_text(json.dumps(manifest | {"archives": []}))
        with self.assertRaisesRegex(RuntimeError, "identity/integrity"):
            installer._install_server(directory, package, acquire=False)
        manifest["archives"] = installer._pinned_archives(package, installer.VERSION)
        marker.write_text(json.dumps(manifest))
        self.assertEqual(
            installer._install_server(directory, package, acquire=False), executable.resolve()
        )
        self.get.assert_not_called()

    def test_early_cancellation_prevents_runtime_and_sdk_acquisition(self):
        check = Mock(side_effect=CancelledError())
        with self.assertRaises(CancelledError):
            installer.server_path(check_cancelled=check)
        with patch("ruinedfooocus.packaging.rocm_installer.rocm_metadata_layer") as selector:
            with self.assertRaises(CancelledError):
                installer.ensure_rocm_libraries(self.root / "sdk", "gfx1100", check_cancelled=check)
        selector.assert_not_called()
        self.get.assert_not_called()

    def test_cancel_child_uses_existing_bounded_owner(self):
        check = Mock()
        with (
            patch(
                "ruinedfooocus.core.bounded_subprocess.run_bounded", side_effect=CancelledError()
            ) as run,
            self.assertRaises(CancelledError),
        ):
            installer._run_cancellable(["inert"], check_cancelled=check, capture_output=True)
        self.assertIs(run.call_args.kwargs["check_cancelled"], check)
        self.assertEqual(run.call_args.args[0], ["inert"])
