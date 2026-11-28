from ruinedfooocus.core.paths import asset_path, resource_path

"""Shared bounded header inspection for metadata and component UI callers."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

from ruinedfooocus.features.models.model_identity import read_safetensors_header

ROOT = Path(__file__).resolve().parents[1]


class ModelHeaderTests(unittest.TestCase):
    def model_browser_reader(self):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/models/model_handler.py").read_text(encoding="utf-8")
        )
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "read_safetensors_header"
        )
        node.decorator_list = []
        scope = {
            "asset_path": asset_path,
            "resource_path": resource_path,
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), "metadata_reader", "exec"), scope)
        return scope["read_safetensors_header"]

    def test_metadata_and_component_inspection_share_header_shapes(self):
        header = {"weight": {"shape": [2, 3]}, "__metadata__": {"modelspec.title": "fixture"}}
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            path = Path(directory) / "model.safetensors"
            raw = json.dumps(header).encode()
            path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"tensor payload is not read")
            self.assertEqual(read_safetensors_header(path), header)
            self.assertEqual(self.model_browser_reader()(path), header)

    def test_malformed_truncated_and_oversized_headers_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            path = Path(directory) / "model.safetensors"
            for raw in (
                b"",
                b"1234",
                (16 * 1024**2 + 1).to_bytes(8, "little"),
                (50).to_bytes(8, "little") + b"{}",
                (2).to_bytes(8, "little") + b"[]",
                (13).to_bytes(8, "little") + b'{"weight": 1}',
            ):
                with self.subTest(raw=raw):
                    path.write_bytes(raw)
                    with self.assertRaises(ValueError):
                        read_safetensors_header(path)
                    self.assertEqual(self.model_browser_reader()(path), {})
            missing = Path(directory) / "missing.safetensors"
            with self.assertRaises(OSError):
                read_safetensors_header(missing)
            self.assertEqual(self.model_browser_reader()(missing), {})
            self.assertEqual(self.model_browser_reader()(Path(directory) / "model.gguf"), {})


if __name__ == "__main__":
    unittest.main()
