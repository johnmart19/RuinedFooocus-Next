"""Installed native build reuse with inert receipts; no native execution."""

from concurrent.futures import CancelledError
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.backends.llama import llama_installer as installer
from ruinedfooocus.core.native_runtime_paths import migrate_legacy_runtime_storage
from ruinedfooocus.packaging import hardware_inventory


class CompatibleCachedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict(os.environ, LLAMA_SERVER=""))
        self.enterContext(patch.object(installer.platform, "system", return_value="Windows"))
        self.enterContext(patch.object(installer.platform, "machine", return_value="AMD64"))
        self.skip = self.enterContext(
            patch.object(installer, "skip_dependency_install", return_value=True)
        )
        self.enterContext(
            patch.object(
                installer,
                "package_directory",
                side_effect=lambda package, version=installer.VERSION: (
                    self.root
                    / "repositories"
                    / "llama.cpp"
                    / version
                    / package.removesuffix(".zip").removesuffix(".tar.gz")
                ),
            )
        )
        self.network = self.enterContext(patch.object(installer.requests, "get"))
        self.native = self.enterContext(patch.object(installer.subprocess, "run"))
        self.addCleanup(self.network.assert_not_called)
        self.addCleanup(self.native.assert_not_called)

    def bundle(self, version="b12110", backend="Vulkan", system="Windows", machine="AMD64"):
        package = installer.package_name(
            system, machine, backend != "CPU", backend=backend, version=version
        )
        directory = installer.package_directory(package, version=version)
        directory.mkdir(parents=True)
        executable = directory / ("llama-server.exe" if system == "Windows" else "llama-server")
        executable.write_bytes(b"inert native fixture")
        saved = installer._runtime_manifest(directory, package, version=version)
        saved["archives"] = installer._pinned_archives(package, version) or [
            {
                "name": package,
                "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{version}/{package}",
                "sha256": "a" * 64,
            }
        ]
        self.receipt(directory, saved)
        return directory, executable, saved

    @staticmethod
    def receipt(directory, saved):
        (directory / "rf-runtime.json").write_text(json.dumps(saved), encoding="utf-8")

    def load(self, backend="Vulkan", **kwargs):
        return installer.server_path(backend != "CPU", backend, **kwargs)

    def inventory(self):
        return hardware_inventory.llama_cached_servers("Windows", "AMD64")

    def test_migrated_previous_build_loads_and_inventory_agrees(self):
        directory, executable, _ = self.bundle()
        old_family = self.root / "cache" / "llama.cpp"
        old_family.parent.mkdir()
        directory.parents[1].rename(old_family)
        results = migrate_legacy_runtime_storage(self.root)
        self.assertEqual(results[0]["status"], "copied-source-retained")
        self.assertTrue(old_family.is_dir())
        self.assertEqual(self.load(), executable.resolve())
        self.assertEqual(self.inventory()["Vulkan"], str(executable.resolve()))

    def test_both_supported_previous_builds_and_cpu_are_reusable(self):
        for version in reversed(installer.COMPATIBLE_CACHED_VERSIONS):
            with self.subTest(version=version):
                _, executable, _ = self.bundle(version, "CPU")
                self.assertEqual(self.load("CPU"), executable.resolve())
                self.assertEqual(self.inventory()["CPU"], str(executable.resolve()))

    def test_explicit_no_acquisition_reuses_previous_build(self):
        _, executable, _ = self.bundle()
        self.skip.return_value = False
        self.assertEqual(self.load(acquire=False), executable.resolve())

    def test_online_acquisition_keeps_current_pin(self):
        self.bundle()
        self.skip.return_value = False
        with patch.object(installer, "_install_server", return_value=Path("new-pin")) as install:
            self.assertEqual(self.load(), Path("new-pin"))
        self.assertEqual(install.call_args.kwargs["version"], installer.VERSION)

    def test_current_pin_wins_over_older_installed_build(self):
        self.bundle()
        _, executable, _ = self.bundle(installer.VERSION)
        self.assertEqual(self.load(), executable.resolve())
        self.assertEqual(self.inventory()["Vulkan"], str(executable.resolve()))

    def test_corrupt_or_incomplete_current_pin_does_not_fall_back(self):
        self.bundle()
        directory, executable, _ = self.bundle(installer.VERSION)
        executable.write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "unverified"):
            self.load()
        executable.unlink()
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self.load()
        self.assertNotIn("Vulkan", self.inventory())
        self.assertTrue(directory.is_dir())

    def test_tampered_older_payload_is_rejected_on_load(self):
        _, executable, _ = self.bundle()
        executable.write_bytes(b"tampered")
        # Passive inventory reports presence; full hashing belongs to activation.
        self.assertIn("Vulkan", self.inventory())
        with self.assertRaisesRegex(RuntimeError, "unverified"):
            self.load()

    def test_missing_and_incomplete_older_runtime_never_acquire(self):
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self.load()
        self.assertNotIn("Vulkan", self.inventory())
        directory, executable, _ = self.bundle()
        executable.unlink()
        before = (directory / "rf-runtime.json").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self.load()
        self.assertNotIn("Vulkan", self.inventory())
        self.assertEqual((directory / "rf-runtime.json").read_bytes(), before)

    def test_vendor_sdk_backend_never_reuses_an_older_package(self):
        self.bundle(backend="ROCm")
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self.load("ROCm")
        self.assertNotIn("ROCm", self.inventory())

    def test_invalid_or_missing_receipt_is_not_discovered(self):
        directory, _, saved = self.bundle()
        marker = directory / "rf-runtime.json"
        invalid = [
            None,
            [],
            {},
            {**saved, "version": "b12400"},
            {**saved, "package": "other.zip"},
            {**saved, "files": {}},
            {**saved, "archives": []},
        ]
        for value in invalid:
            with self.subTest(value=value):
                self.receipt(directory, value)
                with self.assertRaisesRegex(RuntimeError, "missing"):
                    self.load()
                self.assertNotIn("Vulkan", self.inventory())
        marker.unlink()
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self.load()

    def test_wrong_archive_identity_is_not_discovered(self):
        directory, _, saved = self.bundle()
        for field, value in (
            ("name", "other.zip"),
            ("url", "https://example.invalid/archive"),
            ("sha256", "not-a-digest"),
        ):
            with self.subTest(field=field):
                invalid = {**saved, "archives": [{**saved["archives"][0], field: value}]}
                self.receipt(directory, invalid)
                with self.assertRaisesRegex(RuntimeError, "missing"):
                    self.load()
                self.assertNotIn("Vulkan", self.inventory())

    def test_source_pinned_previous_archive_digest_must_match(self):
        directory, _, saved = self.bundle("b12327")
        saved["archives"][0]["sha256"] = "0" * 64
        self.receipt(directory, saved)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self.load()
        self.assertNotIn("Vulkan", self.inventory())

    def test_other_backend_platform_architecture_and_unknown_version_are_not_reused(self):
        for version, backend, system, machine in (
            ("b12110", "CPU", "Windows", "AMD64"),
            ("b12110", "Vulkan", "Linux", "x86_64"),
            ("b12110", "Vulkan", "Windows", "arm64"),
            ("b12400", "Vulkan", "Windows", "AMD64"),
        ):
            self.bundle(version, backend, system, machine)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self.load()
        self.assertNotIn("Vulkan", self.inventory())

    def test_custom_runtime_keeps_precedence(self):
        self.bundle()
        custom = self.root / "custom.exe"
        custom.write_bytes(b"explicit external runtime")
        with patch.dict(os.environ, LLAMA_SERVER=str(custom)):
            self.assertEqual(self.load(), custom.resolve())

    def test_cancelled_integrity_verification_propagates(self):
        self.bundle()
        checks = 0

        def cancelled():
            nonlocal checks
            checks += 1
            if checks == 3:
                raise CancelledError()

        with self.assertRaises(CancelledError):
            self.load(check_cancelled=cancelled)
        self.assertEqual(checks, 3)


if __name__ == "__main__":
    unittest.main()
