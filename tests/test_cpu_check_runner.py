"""CPU acceptance must isolate writable data and exclude video before importing tests."""

import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.check_cpu import run


class CpuRunnerTests(unittest.TestCase):
    def test_real_explicit_unittest_skips_match_only_their_discovered_leaves(self):
        fixtures = (
            (
                "import unittest\nclass Decorated(unittest.TestCase):\n"
                "    @unittest.skip('explicit method skip')\n"
                "    def test_skip(self): raise AssertionError('not run')\n"
                "    def test_good(self): self.assertTrue(True)\n"
                "@unittest.skip('explicit class skip')\nclass Whole(unittest.TestCase):\n"
                "    def test_skip(self): raise AssertionError('not run')\n",
                0,
                "executed=1, skipped=2, discovered=3",
            ),
            (
                "import unittest\nclass Method(unittest.TestCase):\n"
                "    def test_skip(self): raise unittest.SkipTest('explicit test skip')\n",
                0,
                "executed=0, skipped=1, discovered=1",
            ),
            (
                "import unittest\nclass ClassFixture(unittest.TestCase):\n"
                "    @classmethod\n"
                "    def setUpClass(cls): raise unittest.SkipTest('explicit class fixture skip')\n"
                "    def test_one(self): raise AssertionError('not run')\n"
                "    def test_two(self): raise AssertionError('not run')\n",
                0,
                "executed=0, skipped=2, discovered=2",
            ),
            (
                "import unittest\ndef setUpModule(): raise unittest.SkipTest('explicit module skip')\n"
                "class One(unittest.TestCase):\n"
                "    def test_one(self): raise AssertionError('not run')\n"
                "class Two(unittest.TestCase):\n"
                "    def test_two(self): raise AssertionError('not run')\n",
                0,
                "executed=0, skipped=2, discovered=2",
            ),
            (
                "import unittest\nclass Explicit(unittest.TestCase):\n"
                "    @classmethod\n"
                "    def setUpClass(cls): raise unittest.SkipTest('skip this class only')\n"
                "    def test_skip(self): raise AssertionError('not run')\n"
                "class Suppressed(unittest.TestCase):\n"
                "    def test_case(self): raise AssertionError('must execute')\n"
                "    def run(self, result=None): pass\n",
                1,
                "mismatched leaf dispositions",
            ),
            (
                "import unittest\nclass Subtests(unittest.TestCase):\n"
                "    def test_case(self):\n"
                "        with self.subTest('optional'): raise unittest.SkipTest('one subtest')\n"
                "        self.assertTrue(True)\n",
                0,
                "executed=1, skipped=0, discovered=1, fixture_skip_events=0, subtest_skips=1",
            ),
        )
        real_run = subprocess.run
        for source, expected, summary in fixtures:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "tests").mkdir()
                (root / "tests/test_inert.py").write_text(source, encoding="utf-8")
                captured = []

                def execute(*args, **kwargs):
                    result = real_run(*args, **kwargs, capture_output=True, text=True)
                    captured.append(result.stdout + result.stderr)
                    return result

                with patch("tools.check_cpu.subprocess.run", side_effect=execute):
                    self.assertEqual(run(["test_inert.py"], root=root, timeout=30), expected)
                self.assertIn(summary, captured[0])

    def test_real_inert_suites_must_execute_every_discovered_case(self):
        fixtures = (
            (
                "import unittest\nclass Normal(unittest.TestCase):\n"
                "    def test_case(self): self.assertTrue(True)\n",
                0,
            ),
            (
                "import unittest\nclass Suppressed(unittest.TestCase):\n"
                "    def test_case(self): raise AssertionError('must execute')\n"
                "    def run(self, result=None): pass\n",
                1,
            ),
            (
                "import unittest\nclass ForgedCount(unittest.TestCase):\n"
                "    def test_case(self): raise AssertionError('must execute')\n"
                "    def run(self, result=None): result.testsRun += 1\n",
                1,
            ),
            (
                "import unittest\nclass Partial(unittest.TestCase):\n"
                "    def test_good(self): self.assertTrue(True)\n"
                "    def test_suppressed(self): raise AssertionError('must execute')\n"
                "    def run(self, result=None):\n"
                "        if self._testMethodName == 'test_suppressed': return\n"
                "        return super().run(result)\n",
                1,
            ),
        )
        for source, expected in fixtures:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "tests").mkdir()
                (root / "tests/test_inert.py").write_text(source, encoding="utf-8")
                self.assertEqual(run(["test_inert.py"], root=root, timeout=30), expected)

    def test_private_data_guard_and_failure_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tests").mkdir()
            for name in ("test_prompt.py", "test_video_models.py"):
                (root / "tests" / name).touch()
            observed = {}

            def execute(command, **options):
                observed.update(options)
                self.assertTrue(Path(options["env"]["RF_DATA_DIR"]).is_dir())
                self.assertEqual(options["env"]["HF_HUB_OFFLINE"], "1")
                self.assertIn("test_prompt.py", command[2])
                self.assertNotIn("test_video_models.py", command[2])
                self.assertIn("torch.cuda._lazy_init = prohibit_gpu", command[2])
                return SimpleNamespace(returncode=1)

            with patch("tools.check_cpu.subprocess.run", side_effect=execute):
                self.assertEqual(run(["test_*.py"], root=root, timeout=15), 1)
            self.assertEqual(observed["timeout"], 15)
            self.assertFalse(Path(observed["env"]["RF_DATA_DIR"]).exists())
            self.assertNotEqual(os.environ.get("RF_DATA_DIR"), observed["env"]["RF_DATA_DIR"])

    def test_absent_selection_does_not_start_python(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("tools.check_cpu.subprocess.run") as execute:
                with self.assertRaises(ValueError):
                    run(["test_missing.py"], root=Path(folder))
                execute.assert_not_called()
