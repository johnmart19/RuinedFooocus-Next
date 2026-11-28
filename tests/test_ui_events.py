"""Every reachable UI event runs on the real app; side effects stay contained."""

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

GUARD_PROBE = r"""
import os, socket, subprocess, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tools.check_ui_events import BlockedSideEffect, _guard

inside, outside = Path(sys.argv[2]), Path(sys.argv[3])
_guard([inside])
(inside / "allowed.txt").write_text("ok")
os.remove(inside / "allowed.txt")
attempts = {
    "installer": lambda: subprocess.Popen([sys.executable, "-m", "pip", "--version"]),
    "network": lambda: socket.create_connection(("192.0.2.1", 80), timeout=1),
    "write": lambda: open(outside / "blocked.txt", "w"),
    "delete": lambda: os.remove(outside / "keep.txt"),
    "restart": lambda: os.execv(sys.executable, [sys.executable]),
}
for name, attempt in attempts.items():
    try:
        attempt()
    except BlockedSideEffect:
        print("blocked", name)
print("kept", (outside / "keep.txt").exists(), (outside / "blocked.txt").exists())
"""


class UiEventTests(unittest.TestCase):
    def test_every_reachable_event_runs_and_actions_refuse_or_stay_private(self):
        # A separate process builds the app as startup does and fires each event
        # with its initial values; the private RF_DATA_DIR from this runner is
        # the only folder actions may write to.
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/check_ui_events.py"), "--actions"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
            env=dict(os.environ, RF_OFFLINE="1"),
        )
        output = result.stdout + result.stderr
        summary = [line for line in output.splitlines() if line.startswith("Fired ")]
        self.assertTrue(summary, output[-4000:])
        self.assertEqual(result.returncode, 0, output[-4000:])
        counts = dict(
            (key, int(value))
            for value, key in re.findall(
                r"(\d+) (events cleanly|with user messages|problems|timeouts)", summary[-1]
            )
        )
        self.assertEqual(counts["problems"], 0, output[-4000:])
        self.assertEqual(counts["timeouts"], 0, output[-4000:])
        self.assertGreater(counts["events cleanly"] + counts["with user messages"], 900)

    def test_action_guards_block_installers_network_restarts_and_outside_files(self):
        with tempfile.TemporaryDirectory() as inside, tempfile.TemporaryDirectory() as outside:
            Path(outside, "keep.txt").write_text("user file")
            result = subprocess.run(
                [sys.executable, "-c", GUARD_PROBE, str(ROOT), inside, outside],
                capture_output=True,
                text=True,
                timeout=120,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in ("installer", "network", "write", "delete", "restart"):
            self.assertIn(f"blocked {name}", result.stdout)
        self.assertIn("kept True False", result.stdout)


if __name__ == "__main__":
    unittest.main()
