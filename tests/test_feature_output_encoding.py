"""Feature CPU output is lossless UTF-8 even under a legacy Windows locale."""

import os
from pathlib import Path
import tempfile
import unittest

from tools.validate_feature_cherry_picks import _run_cpu_test_process


class FeatureOutputEncodingTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.environment = dict(os.environ, PYTHONUTF8="0", PYTHONIOENCODING="cp1251")

    def test_unicode_stdout_and_stderr_override_legacy_environment_without_mutating_it(self):
        message = "Опис café 猫 🧪"
        code = (
            "import sys; assert sys.flags.utf8_mode == 1; "
            f"sys.stdout.write({message!a}); sys.stderr.write({message[::-1]!a})"
        )
        result = _run_cpu_test_process(code, self.folder, self.environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, message)
        self.assertEqual(result.stderr, message[::-1])
        self.assertEqual(self.environment["PYTHONUTF8"], "0")
        self.assertEqual(self.environment["PYTHONIOENCODING"], "cp1251")

    def test_failed_child_retains_exact_unicode_and_exit_status(self):
        message = "Помилка café 猫"
        result = _run_cpu_test_process(
            f"import sys; sys.stdout.write({message!a}); sys.stderr.write({message!a}); sys.exit(7)",
            self.folder,
            self.environment,
        )
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout + result.stderr, message * 2)

    def test_invalid_wire_bytes_fail_strictly_in_the_caller(self):
        with self.assertRaises(UnicodeDecodeError) as raised:
            _run_cpu_test_process("import os; os.write(2, b'\\xce')", self.folder, self.environment)
        self.assertEqual(raised.exception.object, b"\xce")


if __name__ == "__main__":
    unittest.main()
