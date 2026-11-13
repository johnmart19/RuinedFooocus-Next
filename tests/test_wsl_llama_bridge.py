"""Native WSL bridge acquisition with inert payloads; no runtime/device calls."""

import hashlib
from concurrent.futures import CancelledError
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from ruinedfooocus.backends.llama import wsl_bridge


class WslBridgeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "bridge"
        self.library = b"inert library fixture, never loaded"
        self.archive = b"inert deb fixture, never extracted by dpkg"
        for name, value in (
            ("data_path", lambda *args: self.directory),
            ("LIBRARY_SHA256", hashlib.sha256(self.library).hexdigest()),
            ("ARCHIVE_SHA256", hashlib.sha256(self.archive).hexdigest()),
        ):
            replacement = patch.object(wsl_bridge, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def payload(self, name=None, content=None):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as bundle:
            data = self.library if content is None else content
            member = tarfile.TarInfo(name or f"./opt/rocm/lib/librocdxg.so.{wsl_bridge.VERSION}")
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
        return stream.getvalue()

    def test_verified_cache_works_offline_without_downloading_or_extracting(self):
        self.directory.mkdir()
        (self.directory / "librocdxg.so").write_bytes(self.library)
        with (
            patch.object(wsl_bridge, "skip_dependency_install", return_value=True),
            patch.object(wsl_bridge.requests, "get") as get,
            patch.object(wsl_bridge.subprocess, "run") as run,
        ):
            self.assertEqual(wsl_bridge.library_directory(), self.directory)
            get.assert_not_called()
            run.assert_not_called()

    def test_missing_cache_respects_offline_and_no_acquisition(self):
        for acquire, offline in ((False, False), (True, True)):
            with (
                self.subTest(acquire=acquire, offline=offline),
                patch.object(wsl_bridge, "skip_dependency_install", return_value=offline),
                patch.object(wsl_bridge.requests, "get") as get,
            ):
                with self.assertRaisesRegex(RuntimeError, "WSL bridge"):
                    wsl_bridge.library_directory(acquire=acquire)
                get.assert_not_called()

    def test_offline_reuses_verified_system_bridge_without_installing_a_duplicate(self):
        system = self.directory.parent / "system" / "librocdxg.so"
        system.parent.mkdir()
        system.write_bytes(self.library)
        with (
            patch.object(wsl_bridge, "SYSTEM_LIBRARY", system),
            patch.object(wsl_bridge, "skip_dependency_install", return_value=True),
            patch.object(wsl_bridge.requests, "get") as get,
        ):
            self.assertEqual(wsl_bridge.library_directory(), system.parent)
            self.assertFalse((self.directory / "librocdxg.so").exists())
            get.assert_not_called()

    def acquire(self, payload, *, archive=None, **callbacks):
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [self.archive if archive is None else archive]
        with (
            patch.object(wsl_bridge, "skip_dependency_install", return_value=False),
            patch.object(wsl_bridge.shutil, "which", return_value="/usr/bin/dpkg-deb"),
            patch.object(wsl_bridge.requests, "get", return_value=response),
            patch.object(
                wsl_bridge.installer,
                "_run_cancellable",
                return_value=subprocess.CompletedProcess([], 0, payload),
            ) as run,
            patch.object(wsl_bridge.subprocess, "run", side_effect=AssertionError("native call")),
        ):
            # Both callback and compatibility paths use the inert native fixture.
            with patch.object(wsl_bridge.subprocess, "run", run):
                result = wsl_bridge.library_directory(acquire=True, **callbacks)
            self.assertEqual(run.call_args.args[0][:2], ["dpkg-deb", "--fsys-tarfile"])
            return result

    def test_acquisition_verifies_archive_and_extracts_only_the_expected_library(self):
        self.assertEqual(self.acquire(self.payload()), self.directory)
        self.assertEqual((self.directory / "librocdxg.so").read_bytes(), self.library)

    def test_bad_archive_never_replaces_an_existing_library(self):
        self.directory.mkdir()
        target = self.directory / "librocdxg.so"
        target.write_bytes(b"retain invalid cache for diagnosis")
        with self.assertRaisesRegex(RuntimeError, "download checksum"):
            self.acquire(self.payload(), archive=b"corrupt archive")
        self.assertEqual(target.read_bytes(), b"retain invalid cache for diagnosis")

    def test_rejects_unexpected_paths_and_modified_library(self):
        for payload, error in (
            (self.payload(name="../../librocdxg.so"), "no supported userspace library"),
            (self.payload(content=b"corrupt library"), "library checksum"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(RuntimeError, error):
                self.acquire(payload)
            self.assertFalse((self.directory / "librocdxg.so").exists())

    def test_cancellation_transfer_verification_and_publication_keep_old_library(self):
        self.directory.mkdir()
        target = self.directory / "librocdxg.so"
        target.write_bytes(b"retain old bridge for diagnosis")
        for phase in ("downloading_runtime", "verifying_runtime", "staging_runtime"):
            cancelled = False
            events = []

            def progress(event):
                nonlocal cancelled
                events.append(event)
                if event["phase"] == phase:
                    cancelled = True

            def check():
                if cancelled:
                    raise CancelledError()

            with self.subTest(phase=phase), self.assertRaises(CancelledError):
                self.acquire(self.payload(), progress_callback=progress, check_cancelled=check)
            self.assertEqual(target.read_bytes(), b"retain old bridge for diagnosis")
            self.assertFalse((self.directory / "librocdxg.so.tmp").exists())
            self.assertTrue(events)
            self.assertTrue(
                all(event["runtime"]["tag"] == wsl_bridge.installer.VERSION for event in events)
            )
            self.assertTrue(
                all(
                    event["runtime"]["package"] == f"rocdxg-{wsl_bridge.VERSION}"
                    for event in events
                )
            )
