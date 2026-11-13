"""Startup receipts track dependency contracts without rechecking each UI revision."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.core import dependency_cache as cache


class DependencyCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rf-dependency-cache-")
        self.root = Path(self.temporary.name).resolve()
        assert self.root.parent == Path(tempfile.gettempdir()).resolve()
        self.addCleanup(self.temporary.cleanup)
        for name, value in {
            "requirements_versions.txt": "packaging==26.3\n",
            "pip/modules.txt": "gradio==6.31.0\n",
            "ruinedfooocus/app/bootstrap.py": "# bootstrap\n",
            "entry_with_update.py": "# updater\n",
        }.items():
            self.write(name, value)
        self.library = self.root / "site-packages"
        self.library.mkdir()
        paths = patch.object(cache.sysconfig, "get_path", return_value=str(self.library))
        paths.start()
        self.addCleanup(paths.stop)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return path

    def seed(self, module=cache):
        for stage in ("bootstrap", "maintenance"):
            module.record(self.root, stage)
        self.assert_stages(False, module)

    def assert_stages(self, expected, module=cache):
        for stage in ("bootstrap", "maintenance"):
            with self.subTest(stage=stage):
                self.assertEqual(module.needed(self.root, stage), expected)

    def fresh_module(self, environment, arguments=()):
        spec = importlib.util.spec_from_file_location("cache_fixture", cache.__file__)
        module = importlib.util.module_from_spec(spec)
        with (
            patch.dict(os.environ, environment),
            patch.object(sys, "argv", ["entry_with_update.py", *arguments]),
        ):
            spec.loader.exec_module(module)
        return module

    def git(self, *arguments):
        return subprocess.check_output(
            [
                "git",
                "-c",
                "user.name=Cache Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "commit.gpgSign=false",
                "-c",
                "core.hooksPath=" + str(self.root / "no-hooks"),
                "-C",
                str(self.root),
                *arguments,
            ],
            stderr=subprocess.STDOUT,
        )

    def test_docs_and_ui_only_commit_reuses_both_stages(self):
        self.git("init", "--quiet")
        self.git("add", "requirements_versions.txt", "pip", "ruinedfooocus", "entry_with_update.py")
        self.git("commit", "--quiet", "-m", "Dependency contract")
        before = self.git("rev-parse", "HEAD")
        self.seed()
        self.write("README.md", "Documentation update.\n")
        self.write("ruinedfooocus/ui/app.py", "# UI-only change\n")
        self.write("ruinedfooocus/features/chat/ui.py", "# Chat UI-only change\n")
        self.git("add", "README.md", "ruinedfooocus/ui", "ruinedfooocus/features")
        self.git("commit", "--quiet", "-m", "Documentation and UI")
        self.assertNotEqual(before, self.git("rev-parse", "HEAD"))
        self.assert_stages(False)

    def test_archive_revision_only_change_reuses_receipts_but_pins_and_policy_invalidate(self):
        manifest = {
            "source_commit": "a" * 40,
            "backend_pins": {"ComfyUI": "pin-a"},
            "runtime_bundled": False,
        }
        self.write("distribution.json", json.dumps(manifest))
        self.seed()
        manifest["source_commit"] = "b" * 40
        self.write("distribution.json", json.dumps(manifest, indent=2))
        self.assert_stages(False)
        for key, value in (("backend_pins", {"ComfyUI": "pin-b"}), ("runtime_bundled", True)):
            with self.subTest(field=key):
                self.seed()
                manifest[key] = value
                self.write("distribution.json", json.dumps(manifest))
                self.assert_stages(True)

    def test_contract_requirements_helpers_additions_and_removals_invalidate(self):
        for name in (
            "requirements_versions.txt",
            "pip/modules.txt",
            "entry_with_update.py",
            "ruinedfooocus/app/bootstrap.py",
            "ruinedfooocus/packaging/runtime_bundles.py",
            "ruinedfooocus/core/settings_schema.py",
            "ruinedfooocus/backends/llama/llama_installer.py",
        ):
            with self.subTest(path=name):
                path = self.write(name, "# dependency contract v1\n")
                self.seed()
                path.write_text("# dependency contract v2\n", encoding="utf-8")
                self.assert_stages(True)
                self.seed()
                path.unlink()
                self.assert_stages(True)

    def test_python_abi_and_installed_package_metadata_changes_invalidate(self):
        self.seed()
        original = cache.sysconfig.get_config_var
        with patch.object(
            cache.sysconfig,
            "get_config_var",
            side_effect=lambda name: "changed-abi" if name == "SOABI" else original(name),
        ):
            self.assert_stages(True)
        metadata = self.write("site-packages/fixture-1.0.dist-info/METADATA", "Name: fixture\n")
        self.seed()
        stat = metadata.stat()
        os.utime(metadata, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        self.assert_stages(True)

    def test_next_launch_device_and_cli_context_changes_invalidate(self):
        first = self.fresh_module(
            {"TORCH_PLATFORM": "cpu", "CUDA_VISIBLE_DEVICES": "0"}, ("--cpu",)
        )
        self.seed(first)
        changed_device = self.fresh_module(
            {"TORCH_PLATFORM": "cpu", "CUDA_VISIBLE_DEVICES": "1"}, ("--cpu",)
        )
        self.assert_stages(True, changed_device)
        changed_cli = self.fresh_module(
            {"TORCH_PLATFORM": "cpu", "CUDA_VISIBLE_DEVICES": "0"}, ("--directml=0",)
        )
        self.assert_stages(True, changed_cli)

    def test_startup_normalization_is_stable_and_next_matching_launch_reuses(self):
        environment = {"TORCH_PLATFORM": "", "CUDA_VISIBLE_DEVICES": "0"}
        arguments = ("--cuda-device", "1", "--port", "7860")
        first = self.fresh_module(environment, arguments)
        self.seed(first)
        with patch.dict(os.environ, TORCH_PLATFORM="cu132", CUDA_VISIBLE_DEVICES="1"):
            self.assert_stages(False, first)
            self.seed(first)
        next_launch = self.fresh_module(environment, ("--cuda-device", "1", "--port", "7861"))
        self.assert_stages(False, next_launch)

    def test_missing_legacy_and_forced_receipts_revalidate_without_changing_other_stages(self):
        self.assert_stages(True)
        self.write(
            "cache/dependency-checks.json",
            json.dumps({"bootstrap": "legacy-v1", "maintenance": "legacy-v1", "unrelated": "keep"}),
        )
        self.assert_stages(True)
        self.seed()
        self.assertTrue(cache.needed(self.root, "bootstrap", force=True))
        self.assertEqual(json.loads(cache.cache_path(self.root).read_text())["unrelated"], "keep")


if __name__ == "__main__":
    unittest.main()
