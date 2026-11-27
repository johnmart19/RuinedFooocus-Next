"""Developer HTML formatting must preserve executable and user-visible literals."""

from pathlib import Path
import json
import re
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import format_code
from tools.format_code import canonical_html, command_batches


class FormattingTests(unittest.TestCase):
    def test_normalizes_html_tokens_but_preserves_script_strings_and_svg(self):
        text = '<!doctype html>\n<meta charset="utf-8" />\n<script>const text = "<img />";</script>\n<svg><path d="M 0 1" /></svg>\n'
        result = canonical_html(text)
        self.assertTrue(result.startswith('<!DOCTYPE html>\n<meta charset="utf-8">'))
        self.assertIn('const text = "<img />";', result)
        self.assertIn('<path d="M 0 1" />', result)
        self.assertEqual(canonical_html(result), result)


class FormatterDependencyTests(unittest.TestCase):
    def test_ruff_install_pin_matches_required_version(self):
        requirements = (format_code.ROOT / "tools/format/requirements.txt").read_text(
            encoding="utf-8"
        )
        pins = re.findall(r"^ruff==([^\s;]+)$", requirements, re.MULTILINE)
        self.assertEqual(len(pins), 1)
        config = (format_code.ROOT / "pyproject.toml").read_text(encoding="utf-8")
        required = re.findall(r'^required-version = "==([^\"]+)"$', config, re.MULTILINE)
        self.assertEqual(required, pins)

    def test_prettier_manifest_and_lock_describe_the_same_tool(self):
        folder = format_code.ROOT / "tools/format"
        package = json.loads((folder / "package.json").read_text(encoding="utf-8"))
        lock = json.loads((folder / "package-lock.json").read_text(encoding="utf-8"))
        self.assertTrue(package["private"])
        self.assertEqual(lock["name"], package["name"])
        self.assertEqual(lock["packages"][""]["devDependencies"], package["devDependencies"])
        self.assertEqual(lock["packages"][""]["engines"], package["engines"])
        prettier = lock["packages"]["node_modules/prettier"]
        self.assertRegex(package["devDependencies"]["prettier"], r"^\d+\.\d+\.\d+$")
        self.assertEqual(prettier["version"], package["devDependencies"]["prettier"])

        def minimum_node_version(specification):
            match = re.fullmatch(r">=(\d+)(?:\.(\d+))?(?:\.(\d+))?", specification)
            self.assertIsNotNone(match, "Review the Node engine range: " + specification)
            return tuple(int(part or 0) for part in match.groups())

        self.assertGreaterEqual(
            minimum_node_version(package["engines"]["node"]),
            minimum_node_version(prettier["engines"]["node"]),
        )


class FormatterBatchTests(unittest.TestCase):
    @staticmethod
    def units(command):
        return len(subprocess.list2cmdline(command).encode("utf-16-le")) // 2 + 1

    def test_long_path_list_keeps_every_file_once_in_order(self):
        prefix = ["formatter", "format", "--check"]
        files = [f"owned/{number:04d}-{'x' * 200}.py" for number in range(400)]
        batches = list(command_batches(prefix, files))
        self.assertGreater(len(batches), 1)
        self.assertEqual([path for batch in batches for path in batch[len(prefix) :]], files)
        self.assertTrue(all(batch[: len(prefix)] == prefix for batch in batches))
        self.assertTrue(
            all(self.units(batch) <= format_code.COMMAND_LINE_UNITS for batch in batches)
        )

    def test_budget_accounts_for_windows_quotes_backslashes_and_utf16(self):
        prefix = [r"C:\tools with spaces\formatter.exe", "--check"]
        files = [r"a folder\source.py", 'quote" and space.py', "folder ending \\", "emoji-😀.py"]
        limit = max(self.units([*prefix, path]) for path in files)
        batches = list(command_batches(prefix, files, max_units=limit))
        self.assertGreater(len(batches), 1)
        self.assertEqual([path for batch in batches for path in batch[len(prefix) :]], files)
        self.assertTrue(all(self.units(batch) <= limit for batch in batches))
        emoji = [*prefix, files[-1]]
        with self.assertRaisesRegex(RuntimeError, "command budget"):
            list(command_batches(prefix, files[-1:], max_units=self.units(emoji) - 1))

    def test_exact_limit_fits_and_unbatchable_path_fails_clearly(self):
        prefix, files = ["formatter", "check"], ["one file.py"]
        limit = self.units([*prefix, *files])
        self.assertEqual(list(command_batches(prefix, files, max_units=limit)), [[*prefix, *files]])
        with self.assertRaisesRegex(RuntimeError, "one file.py"):
            list(command_batches(prefix, files, max_units=limit - 1))

    def test_empty_group_never_invokes_implicit_current_directory(self):
        self.assertEqual(list(command_batches(["formatter", "format"], [])), [])

    def test_runner_preserves_stages_and_collects_failures_after_a_failed_batch(self):
        python = [str(Path(f"owned python/{number:04d}-{'x' * 200}.py")) for number in range(200)]
        web = [str(Path(f"owned web/{number:04d}-{'y' * 200}.json")) for number in range(150)]
        excluded = [
            "ruinedfooocus/backends/vendor/untouched.py",
            "ruinedfooocus/backends/vendor/untouched.json",
        ]
        calls = []

        def execute(command, **kwargs):
            calls.append(command)
            self.assertEqual(kwargs, {"cwd": format_code.ROOT})
            self.assertLessEqual(self.units(command), format_code.COMMAND_LINE_UNITS)
            return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)

        with (
            patch.object(
                format_code,
                "tracked_files",
                return_value=[Path(path) for path in [*python, *web, *excluded]],
            ),
            patch.object(Path, "is_file", return_value=True),
            patch.object(format_code.shutil, "which", return_value="node"),
            patch.object(format_code.subprocess, "run", side_effect=execute),
        ):
            self.assertEqual(format_code.run(check=True), 1)
        format_calls = [command for command in calls if command[1:3] == ["format", "--check"]]
        check_calls = [command for command in calls if command[1] == "check"]
        web_calls = [
            command for command in calls if command[2] == "--check" and command not in format_calls
        ]
        self.assertGreater(len(format_calls), 1)
        self.assertGreater(len(check_calls), 1)
        self.assertGreater(len(web_calls), 1)
        self.assertEqual([path for command in format_calls for path in command[3:]], python)
        self.assertEqual([path for command in check_calls for path in command[2:]], python)
        self.assertEqual([path for command in web_calls for path in command[3:]], web)
        self.assertEqual(calls, [*format_calls, *check_calls, *web_calls])


if __name__ == "__main__":
    unittest.main()
