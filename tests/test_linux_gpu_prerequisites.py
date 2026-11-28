"""GPU compiler acquisition is CPU-only, Python-aware and confined to installers."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.packaging import linux_gpu_prerequisites as prerequisites


class LinuxGpuPrerequisiteTests(unittest.TestCase):
    def test_disabled_and_non_linux_never_check_or_install(self):
        with (
            patch.object(prerequisites.sys, "platform", "win32"),
            patch.object(prerequisites.subprocess, "run") as run,
        ):
            prerequisites.ensure_gpu_jit_toolchain(enabled=True)
            run.assert_not_called()
        with patch("importlib.util.find_spec") as find:
            prerequisites.ensure_gpu_jit_toolchain(enabled=False)
            find.assert_not_called()

    def test_complete_toolchain_needs_no_install(self):
        with (
            patch.object(prerequisites.sys, "platform", "linux"),
            patch("importlib.util.find_spec", return_value=object()),
            patch.dict(prerequisites.os.environ, CC=""),
            patch.object(prerequisites.shutil, "which", return_value="/usr/bin/gcc"),
            patch.object(Path, "is_file", return_value=True),
            patch.object(prerequisites.subprocess, "run") as run,
        ):
            prerequisites.ensure_gpu_jit_toolchain(enabled=True)
            run.assert_not_called()

    def test_acquisition_selects_the_running_python_branch_and_verifies_it(self):
        ready = False

        def run(command, **kwargs):
            nonlocal ready
            if "install" in command:
                ready = True

        def which(name):
            return "/usr/bin/" + name if name == "apt-get" or ready else None

        with (
            patch.object(prerequisites.sys, "platform", "linux"),
            patch.object(prerequisites.sys, "version_info", SimpleNamespace(major=3, minor=13)),
            patch("importlib.util.find_spec", return_value=object()),
            patch.dict(prerequisites.os.environ, CC=""),
            patch.object(prerequisites.os, "geteuid", return_value=1000, create=True),
            patch.object(prerequisites.shutil, "which", side_effect=which),
            patch.object(Path, "is_file", side_effect=lambda: ready),
            patch.object(prerequisites.subprocess, "run", side_effect=run) as install,
        ):
            prerequisites.ensure_gpu_jit_toolchain(enabled=True)
        self.assertEqual(
            install.call_args_list[1].args[0],
            ["sudo", "apt-get", "install", "-y", "gcc", "libc6-dev", "python3.13-dev"],
        )

    def test_invalid_explicit_compiler_is_not_silently_replaced(self):
        with (
            patch.object(prerequisites.sys, "platform", "linux"),
            patch("importlib.util.find_spec", return_value=object()),
            patch.dict(prerequisites.os.environ, CC="custom-missing-compiler"),
            patch.object(prerequisites.shutil, "which", return_value=None),
            patch.object(prerequisites.subprocess, "run") as install,
        ):
            with self.assertRaisesRegex(RuntimeError, "configured GPU JIT"):
                prerequisites.ensure_gpu_jit_toolchain(enabled=True)
            install.assert_not_called()
