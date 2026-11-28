"""The real UI builds on CPU and every registered event is wired consistently."""

from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class UiWiringTests(unittest.TestCase):
    def test_every_ui_event_has_present_components_and_a_matching_callback(self):
        # A separate process builds the whole app exactly as startup does, but
        # stops before serving; this process keeps its own test isolation.
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/check_ui_wiring.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        output = result.stdout + result.stderr
        summary = [line for line in output.splitlines() if line.startswith("Checked ")]
        self.assertTrue(summary, output[-4000:])
        self.assertEqual(result.returncode, 0, output[-4000:])
        self.assertIn(" 0 wiring problems", summary[-1])
        self.assertNotIn("WARNING", output)
        events = int(summary[-1].split()[1])
        self.assertGreater(events, 1000)


if __name__ == "__main__":
    unittest.main()
