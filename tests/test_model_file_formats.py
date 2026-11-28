"""Safe .sft aliases preserve inspection, discovery and original file bytes."""

import ast
import json
from pathlib import Path
import tempfile
import unittest

from safetensors.numpy import save_file
import numpy as np

from ruinedfooocus.core.model_formats import (
    NATIVE_TORCH_SUFFIXES,
    SAFETENSORS_SUFFIXES,
    is_safetensors,
)
from ruinedfooocus.core.path import PathManager
from ruinedfooocus.features.models.model_identity import (
    cached_safetensors_header,
    read_safetensors_header,
)
from ruinedfooocus.features.models.component_candidates import header_shapes
from ruinedfooocus.features.models.checkpoint_tools import inspect_checkpoint
from ruinedfooocus.features.models.lora_names import resolve_lora_name

ROOT = Path(__file__).resolve().parents[1]


class ModelFileFormatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_safe_alias_keeps_metadata_dtype_shapes_and_validated_inspection(self):
        for suffix in (".safetensors", ".sft", ".SFT"):
            with self.subTest(suffix=suffix):
                path = self.root / ("fixture" + suffix)
                save_file(
                    {"layer.weight": np.arange(6, dtype=np.float32).reshape(2, 3)},
                    path,
                    metadata={"_quantization_metadata": json.dumps({"layers": {}})},
                )
                original = path.read_bytes()
                header = read_safetensors_header(path)
                self.assertEqual(header["layer.weight"]["dtype"], "F32")
                self.assertEqual(header_shapes(path), {"layer.weight": (2, 3)})
                self.assertIn("_quantization_metadata", header["__metadata__"])
                self.assertEqual(cached_safetensors_header(path), header)
                self.assertEqual(inspect_checkpoint(path)["dtypes"], {"F32": 24})
                self.assertEqual(path.read_bytes(), original)

    def test_replaced_sft_invalidates_header_cache_and_bad_headers_stay_bounded(self):
        path = self.root / "same.sft"
        save_file({"one": np.zeros(1, dtype=np.float32)}, path)
        self.assertIn("one", cached_safetensors_header(path))
        save_file({"replacement": np.zeros(3, dtype=np.float32)}, path)
        self.assertNotIn("one", cached_safetensors_header(path))
        self.assertIn("replacement", cached_safetensors_header(path))
        path.write_bytes((17 * 1024**2).to_bytes(8, "little") + b"{}")
        with self.assertRaisesRegex(ValueError, "oversized"):
            read_safetensors_header(path)
        for suffix in (".ckpt", ".pth", ".bin", ".pt2", ".pkl", ".gguf"):
            self.assertFalse(is_safetensors(self.root / ("untrusted" + suffix)))

    def test_discovery_and_lora_normalization_keep_native_container_candidates(self):
        for name in ("first.safetensors", "second.SFT", "legacy.ckpt", "no.pt2", "no.pkl"):
            (self.root / name).write_bytes(b"local fixture")
        manager = object.__new__(PathManager)
        manager.DOWNLOADABLE_FILES = {}
        manager.get_model_folders = lambda category: [self.root]
        self.assertEqual(
            set(manager.get_folder_list("checkpoints")),
            {"first.safetensors", "second.SFT", "legacy.ckpt", "no.pt2", "no.pkl"},
        )
        # Registry discovery uses the same safe aliases as the path catalogue.
        tree = ast.parse((ROOT / "ruinedfooocus/features/models/model_handler.py").read_text())
        assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute) and target.attr == "EXTENSIONS"
                for target in node.targets
            )
        )
        extensions = eval(
            compile(ast.Expression(assignment.value), "registry suffixes", "eval"),
            {
                "SAFETENSORS_SUFFIXES": SAFETENSORS_SUFFIXES,
                "NATIVE_TORCH_SUFFIXES": NATIVE_TORCH_SUFFIXES,
            },
        )
        self.assertEqual(extensions, manager.EXTENSIONS)
        self.assertEqual(resolve_lora_name("missing.sft", []), "missing.sft")
        self.assertEqual(resolve_lora_name("style", ["nested/style.SFT"]), "nested/style.SFT")
