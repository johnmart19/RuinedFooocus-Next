"""Settings validation at startup must not load packages the installer may replace."""

from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = """
import sys
sys.path.insert(0, {root!r})
from ruinedfooocus.features.image_editing import inpaint_utils
assert inpaint_utils.outpaint_margins(
    dict(outpaint_left=0, outpaint_top=8, outpaint_right=0, outpaint_bottom=0)
)
loaded = sorted(name for name in ("numpy", "torch") if name in sys.modules)
print("LOADED", loaded)
"""


class StartupNativeImportTests(unittest.TestCase):
    def test_validating_saved_outpaint_margins_does_not_import_numpy_or_torch(self):
        # The package plan refuses to replace an imported native package, so an early
        # numpy import blocked every online start that had a numpy update to apply.
        result = subprocess.run(
            [sys.executable, "-c", SCRIPT.format(root=str(ROOT))],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertIn("LOADED []", result.stdout, result.stderr[-600:])


if __name__ == "__main__":
    unittest.main()
