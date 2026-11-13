"""Keep tool help/imports inert and the maintained inventory complete."""

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    ROOT / "tools/README.md",
    ROOT / "docs/integrations/tools.md",
    ROOT / "docs/integrations/README.md",
    ROOT / "docs/download-audit.md",
    ROOT / "docs/development.md",
)

# This runs in a separate process; its audit hook cannot affect other suites.
HELP_GUARD = r"""
import builtins, os, runpy, sys
from pathlib import Path
sys.dont_write_bytecode = True
path, cli = Path(sys.argv[1]), sys.argv[2] == "cli"
sys.path[:0] = [str(path.parent), str(path.parent.parent)]
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "comfy", "comfy_extras", "gradio", "gradio_client"}:
        raise AssertionError("Help/import initialized a runtime: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
def audit(event, args):
    if event == "open":
        mode, flags = args[1:3]
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        ):
            raise AssertionError("Help/import attempted a file write")
    if event in {
        "os.mkdir", "os.remove", "os.rmdir", "os.rename", "os.symlink", "os.link",
        "os.chmod", "os.truncate", "subprocess.Popen", "os.system", "os.exec",
        "os.posix_spawn", "socket.connect", "socket.bind", "socket.getaddrinfo",
    }:
        raise AssertionError("Help/import attempted a side effect: " + event)
sys.addaudithook(audit)
sys.argv = [str(path), "--help"]
try:
    runpy.run_path(str(path), run_name="__main__" if cli else "__rf_tool_import__")
except SystemExit as error:
    if error.code not in (0, None):
        raise
"""


class ToolContractTests(unittest.TestCase):
    def test_every_current_python_and_shell_tool_has_an_inventory_row(self):
        text = DOCS[0].read_text(encoding="utf-8")
        documented = set(re.findall(r"\]\(([^()]+\.(?:py|sh))\)", text))
        current = {
            path.name for path in (ROOT / "tools").iterdir() if path.suffix in {".py", ".sh"}
        }
        self.assertFalse(
            current - documented, f"Missing tool contracts: {sorted(current - documented)}"
        )
        for name in ("feature_commits.json", "validation_assets.json", "format/package-lock.json"):
            self.assertIn(f"]({name})", text)

    def test_python_help_and_helper_imports_have_no_writes_network_processes_or_runtime(self):
        with tempfile.TemporaryDirectory(prefix="rf-tool-help-") as folder:
            data = Path(folder) / "private-data-must-not-be-created"
            environment = dict(os.environ)
            environment.update(
                RF_DATA_DIR=str(data),
                PYTHONDONTWRITEBYTECODE="1",
                GRADIO_ANALYTICS_ENABLED="False",
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
            )
            for path in sorted((ROOT / "tools").glob("*.py")):
                with self.subTest(tool=path.name):
                    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
                    cli = path.name == "api_search.py" or any(
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "ArgumentParser"
                        for node in ast.walk(tree)
                    )
                    result = subprocess.run(
                        [sys.executable, "-c", HELP_GUARD, str(path), "cli" if cli else "helper"],
                        cwd=folder,
                        env=environment,
                        text=True,
                        capture_output=True,
                        timeout=25,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    if cli:
                        self.assertIn("usage:", result.stdout.lower())
            self.assertFalse(data.exists())

    def test_owned_documentation_local_links_resolve(self):
        for path in DOCS:
            for link in re.findall(r"\]\(([^\s()]+)\)", path.read_text(encoding="utf-8")):
                parsed = urlsplit(link)
                if parsed.scheme or not parsed.path:
                    continue
                with self.subTest(document=path.name, link=link):
                    self.assertTrue((path.parent / unquote(parsed.path)).resolve().exists())

    def test_formatting_lock_and_configuration_agree(self):
        package = json.loads((ROOT / "tools/format/package.json").read_text())
        lock = json.loads((ROOT / "tools/format/package-lock.json").read_text())
        expected = package["devDependencies"]["prettier"]
        self.assertEqual(lock["packages"][""]["devDependencies"]["prettier"], expected)
        self.assertEqual(lock["packages"]["node_modules/prettier"]["version"], expected)
        config = (ROOT / "pyproject.toml").read_text()
        requirement = (ROOT / "tools/format/requirements.txt").read_text().strip()
        self.assertIn(f'required-version = "{requirement.removeprefix("ruff")}"', config)


if __name__ == "__main__":
    unittest.main()
