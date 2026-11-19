"""The layered-image path must import its ComfyUI nodes in a fresh interpreter."""

from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = """
import sys
sys.argv = ["test", "--cpu"]
sys.path[:0] = [{comfy!r}, {root!r}]
import comfy.cli_args
comfy.cli_args.args.cpu = True
import torch
from types import SimpleNamespace
from ruinedfooocus.features.generation import layered
assert "comfy_extras.nodes_latent" not in sys.modules
try:
    layered.layer_latent(SimpleNamespace(layers=2), {{"samples": torch.zeros(1, 16, 2, 4, 4)}})
except ValueError as error:
    print("REACHED", error)
"""


class LayeredNativeImportTests(unittest.TestCase):
    def test_layer_cutting_nodes_load_without_another_feature_importing_them_first(self):
        comfy = ROOT / "repositories" / "ComfyUI"
        required = ("comfy/options.py", "comfy_extras/nodes_latent.py", "nodes.py")
        if not all((comfy / name).is_file() for name in required):
            self.skipTest("The pinned ComfyUI checkout is not present.")
        result = subprocess.run(
            [sys.executable, "-c", SCRIPT.format(comfy=str(comfy), root=str(ROOT))],
            capture_output=True,
            text=True,
            timeout=300,
            env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        # The shape check after the imports is reached; a circular import is not.
        self.assertIn("REACHED", result.stdout, result.stderr[-600:])
        self.assertNotIn("circular import", result.stderr)


if __name__ == "__main__":
    unittest.main()
