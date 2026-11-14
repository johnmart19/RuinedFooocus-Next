"""Persistent runtime paths and inert, source-preserving migration fixtures."""

from concurrent.futures import CancelledError
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from ruinedfooocus.core import native_runtime_paths as storage
from ruinedfooocus.core import paths


class NativeRuntimeStorageTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(patch.dict(os.environ, RF_DATA_DIR=str(self.root), LLAMA_SERVER=""))

    def bundle(self, *, package="package", version="b12366"):
        from ruinedfooocus.backends.llama import llama_installer as installer

        folder = self.root / "cache/llama.cpp" / version / package
        folder.mkdir(parents=True)
        binary = folder / ("llama-server.exe" if os.name == "nt" else "llama-server")
        binary.write_bytes(b"inert executable; never run")
        (folder / "backend.dll").write_bytes(b"inert library")
        receipt = installer._runtime_manifest(folder, package, version=version)
        receipt["archives"] = [{"name": package, "sha256": "a" * 64, "url": "fixture"}]
        (folder / "rf-runtime.json").write_text(json.dumps(receipt), encoding="utf-8")
        return folder, binary, receipt

    def test_default_archive_and_external_data_roots_do_not_depend_on_cwd(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(paths, "APP_ROOT", self.root / "archive"),
        ):
            self.assertEqual(
                storage.native_runtime_root("llama.cpp"),
                self.root / "archive/repositories/llama.cpp",
            )
        with (
            patch.dict(os.environ, RF_DATA_DIR="portable-data"),
            patch.object(paths, "APP_ROOT", self.root / "archive"),
        ):
            self.assertEqual(
                storage.native_runtime_root("onnx"),
                self.root / "archive/portable-data/repositories/onnx",
            )
        self.assertEqual(storage.native_runtime_root("sdcpp"), self.root / "repositories/sdcpp")
        self.assertFalse((self.root / "repositories").exists())
        with self.assertRaises(ValueError):
            storage.native_runtime_root("llama.cpp", "../../escape")

    def test_all_managed_callers_use_persistent_paths(self):
        from ruinedfooocus.backends.llama import llama_installer, llama_update, llama_build
        from ruinedfooocus.backends.onnx import onnx_installer, windows_ml
        from ruinedfooocus.backends.sdcpp import activation

        repository = self.root / "repositories"
        self.assertTrue(
            llama_installer.package_directory("fixture.zip").is_relative_to(
                repository / "llama.cpp"
            )
        )
        self.assertTrue(llama_build.build_root().is_relative_to(repository / "llama.cpp"))
        with patch.object(llama_update, "_family", return_value=("Windows", "amd64")):
            self.assertTrue(llama_update._pointer("CPU").is_relative_to(repository / "llama.cpp"))
        with patch.object(onnx_installer, "ROOT", self.root):
            self.assertTrue(onnx_installer.runtime_path().is_relative_to(repository / "onnx"))
            self.assertTrue(
                onnx_installer.native_runtime_path("OpenVINO").is_relative_to(repository / "onnx")
            )
        self.assertEqual(windows_ml.manifest_path(), repository / "onnx/windows-ml/provider.json")
        self.assertEqual(activation.root_path(), repository / "sdcpp")

    def test_plan_only_counts_and_does_not_hash_or_create_paths(self):
        folder, _, _ = self.bundle()
        with patch.object(storage, "_hash", side_effect=AssertionError("dry run hashed payload")):
            plans = storage.plan_legacy_runtime_migration(self.root)
        self.assertEqual(plans[0]["files"], 3)
        self.assertEqual(plans[0]["bytes"], sum(p.stat().st_size for p in folder.iterdir()))
        self.assertEqual([p["status"] for p in plans], ["ready", "missing", "missing"])
        self.assertFalse((self.root / "repositories").exists())

    def test_verified_bundle_receipts_selection_survive_cache_removal_offline(self):
        from ruinedfooocus.backends.llama import llama_installer as installer
        from ruinedfooocus.backends.llama import llama_update

        folder, binary, receipt = self.bundle()
        selection = self.root / "cache/llama.cpp/selected/Windows-amd64-CPU.json"
        selection.parent.mkdir()
        selection.write_text(json.dumps({"tag": "b12366", "sha256": "a" * 64}))
        provenance = self.root / "cache/llama.cpp/xllamacpp-backend.json"
        provenance.write_text('{"index":"fixture-index"}')
        before = {
            str(p.relative_to(self.root / "cache")): p.read_bytes()
            for p in (folder / "rf-runtime.json", binary, selection, provenance)
        }
        result = storage.migrate_legacy_runtime_storage(self.root)
        self.assertEqual(result[0]["status"], "copied-source-retained")
        self.assertEqual(
            {name: (self.root / "cache" / name).read_bytes() for name in before}, before
        )
        # Only the private fixture's disposable cache is removed here.
        shutil.rmtree(self.root / "cache")
        target = installer.package_directory("package", version="b12366")
        with (
            patch.object(installer.requests, "get", side_effect=AssertionError("network")),
            patch.dict(os.environ, RF_OFFLINE="1"),
        ):
            server = installer._install_server(target, "package", acquire=False, version="b12366")
        self.assertEqual(server.read_bytes(), b"inert executable; never run")
        self.assertEqual(json.loads((target / "rf-runtime.json").read_text()), receipt)
        with patch.object(llama_update, "_family", return_value=("Windows", "amd64")):
            self.assertEqual(llama_update._read_selected("CPU"), ("b12366", "a" * 64))
        self.assertEqual(
            (self.root / "repositories/llama.cpp/xllamacpp-backend.json").read_text(),
            '{"index":"fixture-index"}',
        )

    def test_custom_server_is_neither_relocated_nor_replaced(self):
        from ruinedfooocus.backends.llama import llama_installer as installer

        custom = self.root / "external/custom-server"
        custom.parent.mkdir()
        custom.write_bytes(b"custom")
        with patch.dict(os.environ, LLAMA_SERVER=str(custom)):
            self.assertEqual(installer.server_path(False), custom.resolve())
        self.assertEqual(storage.migrate_legacy_runtime_storage(self.root)[0]["status"], "missing")
        self.assertEqual(custom.read_bytes(), b"custom")

    def test_sdk_pointer_remaps_only_owned_directory(self):
        from ruinedfooocus.backends.llama.llama_installer import active_rocm_directory

        family = self.root / "cache/llama.cpp"
        active = family / "rocm-10.0-windows-published"
        active.mkdir(parents=True)
        (active / "gfx1100.ready").write_bytes(b"")
        pointer = family / "rocm-10.0-windows.active.json"
        pointer.write_text(json.dumps({"directory": str(active)}))
        external = family / "rocm-external.active.json"
        external.write_text(json.dumps({"directory": str(self.root / "custom-sdk")}))
        storage.migrate_legacy_runtime_storage(self.root, families=("llama.cpp",))
        target = self.root / "repositories/llama.cpp"
        self.assertEqual(active_rocm_directory(target / "rocm-10.0-windows"), target / active.name)
        self.assertEqual(json.loads(pointer.read_text())["directory"], str(active))
        self.assertEqual((target / external.name).read_bytes(), external.read_bytes())

    def test_incomplete_or_tampered_legacy_bundle_is_not_published(self):
        folder, binary, _ = self.bundle()
        binary.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "integrity"):
            storage.migrate_legacy_runtime_storage(self.root)
        self.assertFalse((self.root / "repositories/llama.cpp").exists())
        (folder / "rf-runtime.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "Unverified"):
            storage.migrate_legacy_runtime_storage(self.root)
        self.assertTrue(binary.exists())

    def test_destination_collision_preserves_both_and_unrelated_clone(self):
        self.bundle()
        target = self.root / "repositories/llama.cpp"
        target.mkdir(parents=True)
        clone = self.root / "repositories/llama_cpp"
        clone.mkdir()
        (clone / "manual.txt").write_bytes(b"unrelated source")
        self.assertEqual(storage.plan_legacy_runtime_migration(self.root)[0]["status"], "collision")
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            storage.migrate_legacy_runtime_storage(self.root)
        self.assertEqual(list(target.iterdir()), [])
        self.assertEqual((clone / "manual.txt").read_bytes(), b"unrelated source")

    def test_symlink_or_reparse_point_cannot_escape(self):
        folder, _, _ = self.bundle()
        real_lstat = Path.lstat

        def reparse(path, *args, **kwargs):
            if path == folder:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return real_lstat(path, *args, **kwargs)

        with patch.object(Path, "lstat", reparse), self.assertRaisesRegex(RuntimeError, "reparse"):
            storage.plan_legacy_runtime_migration(self.root)
        self.assertFalse((self.root / "repositories").exists())

    def test_changed_source_and_incomplete_onnx_never_publish(self):
        folder, binary, _ = self.bundle()
        original_copy = shutil.copy2

        def changing(source, destination):
            result = original_copy(source, destination)
            if source == binary:
                binary.write_bytes(b"changed during copy")
            return result

        with (
            patch.object(storage.shutil, "copy2", side_effect=changing),
            self.assertRaisesRegex(RuntimeError, "changed"),
        ):
            storage.migrate_legacy_runtime_storage(self.root)
        self.assertFalse((self.root / "repositories/llama.cpp").exists())
        incomplete = self.root / "cache/onnx/cp312/incomplete-runtime"
        incomplete.mkdir(parents=True)
        (incomplete / "provider.pyd").write_bytes(b"unverified")
        with self.assertRaisesRegex(RuntimeError, "Incomplete legacy ONNX"):
            storage.migrate_legacy_runtime_storage(self.root, families=("onnx",))
        self.assertTrue((incomplete / "provider.pyd").exists())
        self.assertFalse((self.root / "repositories/onnx").exists())

    def test_missing_legacy_source_preserves_existing_destination_without_blocking_other_families(
        self,
    ):
        target = self.root / "repositories/onnx"
        target.mkdir(parents=True)
        (target / "custom.txt").write_bytes(b"retain")
        self.bundle()
        results = storage.migrate_legacy_runtime_storage(self.root)
        self.assertEqual(
            [row["status"] for row in results], ["copied-source-retained", "existing", "missing"]
        )
        self.assertEqual((target / "custom.txt").read_bytes(), b"retain")

    def test_copy_failure_or_cancellation_preserves_source_and_no_partial_destination(self):
        folder, binary, _ = self.bundle()
        with (
            patch.object(storage.shutil, "copy2", side_effect=OSError("disk full")),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            storage.migrate_legacy_runtime_storage(self.root)
        self.assertTrue(binary.exists())
        self.assertFalse((self.root / "repositories/llama.cpp").exists())
        with self.assertRaises(CancelledError):
            storage.migrate_legacy_runtime_storage(
                self.root, check_cancelled=lambda: (_ for _ in ()).throw(CancelledError())
            )
        self.assertEqual(len(list(folder.iterdir())), 3)

    def test_onnx_verified_provider_and_windowsml_receipt_survive(self):
        from ruinedfooocus.backends.onnx import onnx_installer as installer

        with patch.object(installer, "ROOT", self.root):
            target = installer.native_runtime_path("OpenVINO")
            source = self.root / "cache/onnx" / target.relative_to(self.root / "repositories/onnx")
            source.mkdir(parents=True)
            payload = source / "provider.pyd"
            payload.write_bytes(b"inert provider")
            marker = {
                "version": installer.native_runtime("OpenVINO").version,
                "python": list(sys.version_info[:2]),
                "binary": payload.name,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
            (source / "verified.json").write_text(json.dumps(marker))
            catalog = self.root / "cache/onnx/windows-ml/provider.json"
            catalog.parent.mkdir()
            catalog.write_text('{"external_provider":"C:/custom/provider.dll"}')
            storage.migrate_legacy_runtime_storage(self.root, families=("onnx",))
            shutil.rmtree(self.root / "cache")
            self.assertTrue(installer.native_runtime_available("OpenVINO"))
            self.assertEqual(json.loads((target / "verified.json").read_text()), marker)
            self.assertEqual(
                (self.root / "repositories/onnx/windows-ml/provider.json").read_bytes(),
                b'{"external_provider":"C:/custom/provider.dll"}',
            )

    def test_sdcpp_payload_and_acceptance_survive_disposable_cache_removal(self):
        from ruinedfooocus.backends.sdcpp import activation, artifacts

        item = artifacts.ARTIFACTS[0]
        target = activation.runtime_directory(item)
        source = self.root / "cache/sdcpp" / target.relative_to(self.root / "repositories/sdcpp")
        source.mkdir(parents=True)
        executable = "sd-cli.exe" if item.system == "Windows" else "sd-cli"
        for name in (executable, "stable-diffusion.cpp.txt", "ggml.txt"):
            (source / name).write_bytes(b"inert fixture")
        manifest = {
            "revision": artifacts.SOURCE_REVISION,
            "archive_sha256": item.sha256,
            "files": artifacts._payload(source),
        }
        (source / "rf-runtime.json").write_text(json.dumps(manifest))
        receipt = self.root / "cache/sdcpp/acceptance.json"
        receipt.write_text('{"schema":1,"receipts":[]}')
        storage.migrate_legacy_runtime_storage(self.root, families=("sdcpp",))
        shutil.rmtree(self.root / "cache")
        candidate = artifacts.open_runtime(target, item)
        self.assertEqual(candidate.executable, target.resolve() / executable)
        self.assertEqual(activation._read_receipts(), [])


if __name__ == "__main__":
    unittest.main()
