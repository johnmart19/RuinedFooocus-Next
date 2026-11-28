"""Exercise subprocess failure and stale evidence handling without inference."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.run_comfy_validation import ROOT, run_cases


class ValidationRunnerTests(unittest.TestCase):
    def test_crash_clears_old_pass_and_records_failure_without_report(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            case = root / "case.json"
            case.write_text("{}")
            report, process = root / "case-comparison.json", root / "case-process.json"
            report.write_text('{"comparison":{"passed":true}}')
            process.write_text('{"exit_code":0,"passed":true}')

            def crash(*args, **kwargs):
                self.assertFalse(report.exists())
                self.assertFalse(process.exists())
                return SimpleNamespace(returncode=-1)

            with patch("tools.run_comfy_validation.subprocess.run", side_effect=crash):
                result = run_cases([case], root)[0]
            self.assertFalse(result["passed"])
            self.assertEqual(result["exit_code"], -1)
            self.assertIsNone(result["comparison_sha256"])
            self.assertEqual(json.loads(process.read_text()), result)

    def test_report_pass_needs_zero_exit_and_each_report_gets_current_digest(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            cases = [root / (name + ".json") for name in ("failed", "passed")]
            for case in cases:
                case.write_text("{}")

            def complete(command, **kwargs):
                report = Path(command[-1])
                report.write_text('{"comparison":{"passed":true}}')
                return SimpleNamespace(returncode=1 if report.name.startswith("failed") else 0)

            with patch("tools.run_comfy_validation.subprocess.run", side_effect=complete) as run:
                results = run_cases(cases, root)
            self.assertEqual([r["passed"] for r in results], [False, True])
            self.assertEqual(run.call_count, 2)
            for case, result in zip(cases, results):
                report = root / (case.stem + "-comparison.json")
                self.assertEqual(
                    result["comparison_sha256"], hashlib.sha256(report.read_bytes()).hexdigest()
                )

    def test_spawn_error_is_recorded(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            case = root / "case.json"
            case.write_text("{}")
            with patch(
                "tools.run_comfy_validation.subprocess.run", side_effect=OSError("cannot spawn")
            ):
                result = run_cases([case], root)[0]
            self.assertFalse(result["passed"])
            self.assertIsNone(result["exit_code"])
            self.assertIn("cannot spawn", result["error"])


if __name__ == "__main__":
    unittest.main()
