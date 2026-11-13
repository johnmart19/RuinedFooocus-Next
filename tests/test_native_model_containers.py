"""Proven native container readers stay weights-only; discovery never unpickles."""

import ast
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import safetensors.torch
import torch

from ruinedfooocus.core.model_formats import NATIVE_TORCH_SUFFIXES, SAFETENSORS_SUFFIXES
from ruinedfooocus.core.path import PathManager
from ruinedfooocus.features.models.component_candidates import installed_candidates
from ruinedfooocus.features.models.lora_names import resolve_lora_name

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))


class UnsupportedObject:
    pass


def native_file_reader(mmap=False):
    source = COMFY / "comfy/utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_torch_file"
    )
    scope = {
        "torch": torch,
        "safetensors": safetensors,
        "MMAP_TORCH_FILES": mmap,
        "DISABLE_MMAP": False,
        "comfy": SimpleNamespace(
            memory_management=SimpleNamespace(aimdo_enabled=False),
            storage=SimpleNamespace(annotate_state_dict=Mock()),
        ),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope["load_torch_file"]


class NativeModelContainerTests(unittest.TestCase):
    def test_native_advertised_container_suffixes_have_rf_discovery_candidates(self):
        tree = ast.parse((COMFY / "folder_paths.py").read_text(encoding="utf-8"))
        declaration = next(
            n
            for n in tree.body
            if isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and n.target.id == "supported_pt_extensions"
        )
        self.assertEqual(
            set(PathManager.EXTENSIONS) - {".gguf"}, ast.literal_eval(declaration.value)
        )
        for suffix in NATIVE_TORCH_SUFFIXES:
            self.assertEqual(resolve_lora_name("missing" + suffix, []), "missing" + suffix)

    def test_actual_native_reader_accepts_only_proven_cpu_state_dict_containers(self):
        value = torch.arange(6, device="cpu", dtype=torch.float32).reshape(2, 3)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for suffix in NATIVE_TORCH_SUFFIXES:
                path = root / ("state" + suffix)
                torch.save({"state_dict": {"layer.weight": value}}, path)
                for mmap in (False, True):
                    with (
                        self.subTest(suffix=suffix, mmap=mmap),
                        patch.object(
                            torch,
                            "load",
                            wraps=torch.load,
                        ) as load,
                    ):
                        state, metadata = native_file_reader(mmap)(str(path), return_metadata=True)
                        torch.testing.assert_close(state["layer.weight"], value)
                        self.assertEqual(state["layer.weight"].device.type, "cpu")
                        self.assertIsNone(metadata)
                        self.assertIs(load.call_args.kwargs["weights_only"], True)
                        self.assertEqual(load.call_args.kwargs["map_location"], torch.device("cpu"))
                        self.assertEqual(load.call_args.kwargs.get("mmap", False), mmap)
            for suffix in SAFETENSORS_SUFFIXES:
                path = root / ("state" + suffix)
                safetensors.torch.save_file(
                    {"layer.weight": value}, path, metadata={"source": "owned CPU fixture"}
                )
                state, metadata = native_file_reader()(str(path), return_metadata=True)
                torch.testing.assert_close(state["layer.weight"], value)
                self.assertEqual(metadata, {"source": "owned CPU fixture"})
        self.assertFalse(torch.cuda.is_initialized())

    def test_arbitrary_pickle_and_unallowlisted_objects_stay_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "object.pkl"
            torch.save(UnsupportedObject(), path)
            with self.assertRaises(pickle.UnpicklingError):
                native_file_reader()(str(path))
            # A pickle stream is not a native torch.save state_dict container.
            path.write_bytes(pickle.dumps({"integer": 1}, protocol=2))
            with self.assertRaises((pickle.UnpicklingError, RuntimeError)):
                native_file_reader()(str(path))

    def test_legacy_candidates_are_unknown_without_loading_weights_on_browse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for suffix in NATIVE_TORCH_SUFFIXES:
                (root / ("local" + suffix)).write_bytes(b"candidate requires runtime validation")
            paths = SimpleNamespace(
                paths={"path_clip": [str(root)]},
                get_abspath=Path,
                EXTENSIONS=PathManager.EXTENSIONS,
            )
            with patch.object(torch, "load", side_effect=AssertionError("Browse loaded weights")):
                rows = installed_candidates("clip_l", paths)
            self.assertEqual(len(rows), len(NATIVE_TORCH_SUFFIXES))
            self.assertEqual({row["compatibility"] for row in rows}, {"unknown"})
