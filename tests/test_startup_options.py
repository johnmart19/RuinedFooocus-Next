"""Package-skip policy must work before either entrypoint imports GPU libraries."""

import ast
import os
from pathlib import Path
import sys
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.packaging.startup_options import LEGACY_SKIP_FLAG, skip_dependency_install

ROOT = Path(__file__).resolve().parents[1]


class StartupOptionsTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt" or not shutil.which("bash"), "Requires a native Linux shell")
    def test_shell_prerequisites_preserve_offline_choices(self):
        source = (ROOT / "RuinedFooocus.sh").read_text()
        function = (
            "install_system_prerequisites() {"
            + source.split("install_system_prerequisites() {", 1)[1].split("\n}\n", 1)[0]
            + "\n}\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "called"
            apt = Path(folder) / "apt-get"
            apt.write_text(f"#!/bin/sh\n: > '{marker}'\n")
            apt.chmod(0o755)
            for options, values in (
                ("--offline", {}),
                ("--skip-dependency-install", {}),
                ("", {"RF_OFFLINE": "1"}),
            ):
                with self.subTest(options=options, values=values):
                    environment = dict(os.environ, PATH=folder, RF_OFFLINE="0", PYTHON="") | values
                    result = subprocess.run(
                        [
                            shutil.which("bash"),
                            "-c",
                            "startup_arguments=("
                            + options
                            + ")\n"
                            + function
                            + "install_system_prerequisites libgomp1",
                        ],
                        env=environment,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertFalse(marker.exists())

    def test_skip_packages_and_offline_are_distinct_with_a_hidden_legacy_alias(self):
        source = (ROOT / "ruinedfooocus/app/arguments.py").read_text(encoding="utf-8")
        scope = {}
        exec(source.split("# ComfyUI compute arguments")[0], scope)
        parser = scope["parser"]
        with patch.dict(os.environ, RF_OFFLINE="0"):
            self.assertFalse(skip_dependency_install([]))
            for flag in ("--skip-dependency-install", LEGACY_SKIP_FLAG):
                args = parser.parse_args([flag])
                self.assertTrue(args.skip_dependency_install)
                self.assertFalse(args.offline)
                self.assertTrue(skip_dependency_install([flag]))
            self.assertTrue(skip_dependency_install(["--offline"]))
        with patch.dict(os.environ, RF_OFFLINE="1"):
            self.assertTrue(skip_dependency_install([]))
        self.assertIn("--skip-dependency-install", parser.format_help())
        self.assertNotIn(LEGACY_SKIP_FLAG, parser.format_help())

    def test_updater_never_installs_bootstrap_packages_when_installation_is_disabled(self):
        source = (ROOT / "entry_with_update.py").read_text(encoding="utf-8")
        prefix = source.split("if not offline and source_checkout(root):")[0]
        for flag in ("--skip-dependency-install", LEGACY_SKIP_FLAG, "--offline"):
            with (
                self.subTest(flag=flag),
                patch.object(sys, "argv", ["entry_with_update.py", flag]),
                patch.dict(os.environ, RF_OFFLINE="0"),
                patch("os.chdir"),
                patch("subprocess.run", side_effect=AssertionError("Unexpected installation")),
            ):
                exec(
                    compile(prefix, "entry_with_update.py", "exec"),
                    {"__file__": str(ROOT / "entry_with_update.py")},
                )

    def test_bootstrap_skip_path_does_not_probe_or_install_runtimes(self):
        source = ast.parse((ROOT / "ruinedfooocus/app/bootstrap.py").read_text(encoding="utf-8"))
        function = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == "prepare_environment"
        )
        scope = dict(
            sys=sys, version=SimpleNamespace(version="test"), args=SimpleNamespace(api=False)
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), "bootstrap", "exec"), scope)
        scope["prepare_environment"](True)

    def test_bootstrap_does_not_repeat_a_failed_source_check_during_startup(self):
        source = ast.parse((ROOT / "ruinedfooocus/app/bootstrap.py").read_text(encoding="utf-8"))
        function = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == "clone_git_repos"
        )
        scope = dict(
            Path=Path,
            sys=SimpleNamespace(path=[]),
            script_path=str(ROOT),
            dir_repos="repositories",
            repo_dir=lambda name: str(ROOT / "repositories" / name),
            git_repos=[
                {
                    "path": "Fixture",
                    "hash": "pin",
                    "url": "unused",
                    "name": "Fixture",
                    "add_path": "Fixture",
                }
            ],
            _checked_git_repos=set(),
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), "bootstrap", "exec"), scope)
        with patch("ruinedfooocus.packaging.launch_util.git_clone", return_value=False) as clone:
            scope["clone_git_repos"](offline=False)
            scope["clone_git_repos"](offline=False)
        clone.assert_called_once()
        self.assertEqual(len(scope["sys"].path), 1)


if __name__ == "__main__":
    unittest.main()
