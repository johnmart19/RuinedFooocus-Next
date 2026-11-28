"""Actual launcher fixtures keep paths, interpreter choice, arguments and errors."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = """import json, sys
print(json.dumps({'python': sys.executable, 'argv': sys.argv[1:], 'unbuffered': sys.stdout.write_through}), flush=True)
raise SystemExit(7)
"""


class LauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rf-launcher-")
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name) / "Launcher проба and spaces"
        self.folder.mkdir()
        (self.folder / "entry_with_update.py").write_text(ENTRY, encoding="utf-8")
        self.arguments = [
            "--offline",
            "--settings",
            "profile with spaces",
            "--prompt",
            "hello & bye",
        ]
        self.environment = dict(os.environ, PYTHON=sys.executable, RF_OFFLINE="1")

    @unittest.skipUnless(os.name == "nt", "Requires Windows cmd.exe")
    def test_batch_keeps_explicit_python_arguments_and_failure_exit(self):
        batch = self.folder / "RuinedFooocus.bat"
        shutil.copyfile(ROOT / batch.name, batch)
        command = subprocess.list2cmdline([str(batch), *self.arguments])
        command = f'"{os.environ.get("COMSPEC", "cmd.exe")}" /d /s /c "{command}"'
        result = subprocess.run(
            command,
            cwd=self.folder.parent,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[0], "Starting RuinedFooocus...")
        self.assert_entry(json.loads(lines[1]))

    @unittest.skipIf(os.name == "nt" or not shutil.which("bash"), "Requires a native Linux shell")
    def test_shell_keeps_explicit_and_active_environments_without_installing(self):
        shell = self.folder / "RuinedFooocus.sh"
        shutil.copyfile(ROOT / shell.name, shell)
        # A stale local environment must not override either explicit selection.
        local = self.folder / "venv" / "bin"
        local.mkdir(parents=True)
        (local / "python").write_text("#!/bin/sh\nexit 88\n", encoding="utf-8")
        (local / "python").chmod(0o755)
        active = self.folder.parent / "active environment" / "bin"
        active.mkdir(parents=True)
        (active / "python").write_text(
            "#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n', encoding="utf-8"
        )
        (active / "python").chmod(0o755)
        for explicit in (True, False):
            with self.subTest(explicit=explicit):
                environment = dict(self.environment, VIRTUAL_ENV=str(active.parent))
                if not explicit:
                    environment.pop("PYTHON")
                result = subprocess.run(
                    [shutil.which("bash"), str(shell), *self.arguments],
                    cwd=self.folder.parent,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertEqual(result.stdout.splitlines()[0], "Starting RuinedFooocus...")
                self.assert_entry(json.loads(result.stdout.splitlines()[1]))

    def assert_entry(self, values):
        self.assertEqual(Path(values["python"]).resolve(), Path(sys.executable).resolve())
        self.assertEqual(values["argv"], self.arguments)
        self.assertTrue(values["unbuffered"])


if __name__ == "__main__":
    unittest.main()
