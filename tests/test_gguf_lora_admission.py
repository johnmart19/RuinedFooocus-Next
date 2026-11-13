"""Typed local LoRA admission agrees with the pinned GGUF adapter reader."""

import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gguf
import numpy as np

from ruinedfooocus.features.models import gguf_metadata
from ruinedfooocus.features.models.gguf_adapters import discover_image_loras, inspect_gguf_lora
from ruinedfooocus.features.models.lora_names import capture_gguf_lora

ROOT = Path(__file__).resolve().parents[1]
GGUF = Path(os.environ.get("RF_TEST_GGUF_SOURCE", ROOT / "repositories/molbal_comfyui_gguf"))


def write_adapter(path, *, kind=gguf.GGMLQuantizationType.F32, factors=None, **metadata):
    writer = gguf.GGUFWriter(path, metadata.pop("architecture", "sdxl"))
    for key, value in {"general.type": "adapter", "adapter.type": "lora", **metadata}.items():
        if isinstance(value, str):
            writer.add_string(key, value)
        else:
            writer.add_float32(key, value)
    factors = (
        factors
        if factors is not None
        else {
            "linear.weight.lora_a": (32, 32),
            "linear.weight.lora_b": (64, 32),
        }
    )
    for name, shape in factors.items():
        values = np.zeros(shape, dtype=np.float32)
        writer.add_tensor(name, gguf.quants.quantize(values, kind), raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class GgufLoraAdmissionTests(unittest.TestCase):
    def test_all_native_factor_types_keep_default_header_shape_and_match_actual_reader(self):
        native = runpy.run_path(str(GGUF / "lora.py"))["load_gguf_lora"]
        with tempfile.TemporaryDirectory() as directory:
            for kind in (
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.BF16,
                gguf.GGMLQuantizationType.Q8_0,
            ):
                with self.subTest(kind=kind):
                    path = Path(directory) / (kind.name + ".GGUF")
                    write_adapter(path, kind=kind)
                    before = path.read_bytes()
                    header, _ = gguf_metadata.read_gguf_header(path)
                    self.assertEqual(
                        header,
                        {
                            "linear.weight.lora_a": {"shape": [32, 32]},
                            "linear.weight.lora_b": {"shape": [64, 32]},
                        },
                    )
                    with patch("gguf.GGUFReader", side_effect=AssertionError("No mapped reader")):
                        admitted = inspect_gguf_lora(path)
                    _, targets, declarations = native(path)
                    self.assertEqual(admitted.target_count, len(targets))
                    self.assertEqual(admitted.factor_types, (int(kind),))
                    self.assertEqual(admitted.architecture, declarations["architecture"])
                    self.assertEqual(tuple(targets["linear"]["down"].shape), (32, 32))
                    self.assertEqual(path.read_bytes(), before)

    def test_wrong_roles_layouts_quantizations_and_nonfinite_alpha_never_enter_inventory(self):
        cases = (
            {"general.type": "model"},
            {"adapter.type": "other"},
            {"adapter.lora.alpha": float("nan")},
            {"kind": gguf.GGMLQuantizationType.Q4_0},
            {"factors": {}},
            {"factors": {"linear.lora_a": (32, 32)}},
            {"factors": {"linear.lora_a": (32, 32), "linear.lora_b": (64, 16)}},
            {"factors": {"linear.lora_a": (1, 32, 32), "linear.lora_b": (64, 32)}},
            {"factors": {"unrelated.weight": (32, 32)}},
            {"factors": {"linear.lora_a": (32, 32), "linear.weight.lora_b": (64, 32)}},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, options in enumerate(cases):
                path = root / f"bad-{index}.gguf"
                write_adapter(path, **options)
                with self.subTest(options=options), self.assertRaises(ValueError):
                    inspect_gguf_lora(path)
            self.assertEqual(discover_image_loras([root]), [])
            with self.assertRaisesRegex(ValueError, "adapter"):
                capture_gguf_lora("bad-0.gguf", [root])

    def test_discovery_keeps_root_priority_native_candidates_and_unknown_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "one", root / "two"
            first.mkdir()
            (second / "nested").mkdir(parents=True)
            (first / "ordinary.SFT").write_bytes(b"native candidate")
            (first / "shadow.GGUF").write_bytes(b"wrong role in preferred root")
            write_adapter(second / "shadow.GGUF")
            path = second / "nested/custom.gGuF"
            write_adapter(path, architecture="custom-or-text-encoder")
            names = discover_image_loras([first, second])
            self.assertEqual(set(names), {"ordinary.SFT", str(Path("nested/custom.gGuF"))})
            self.assertEqual(
                capture_gguf_lora("nested/custom.gGuF", [first, second])[0], path.resolve()
            )

    def test_storage_opt_in_remains_bounded_and_cached_source_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bounded.gguf"
            write_adapter(path)
            sizes = []
            take = gguf_metadata._HeaderStream.take

            def capture(reader, size):
                sizes.append(size)
                return take(reader, size)

            with patch.object(gguf_metadata._HeaderStream, "take", capture):
                inspected = inspect_gguf_lora(path)
            self.assertLess(sum(sizes), 4096)
            with (
                patch(
                    "ruinedfooocus.features.models.gguf_adapters.file_signature",
                    side_effect=[list(inspected.signature), ["changed", 1, 2, 3]],
                ),
                self.assertRaisesRegex(OSError, "changed"),
            ):
                inspect_gguf_lora(path)
            write_adapter(path, **{"general.type": "model"})
            with self.assertRaisesRegex(ValueError, "adapter"):
                inspect_gguf_lora(path)

    def test_recursive_quantization_json_is_rejected_without_breaking_scan_or_api(self):
        from ruinedfooocus.app import api_runtime

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid, malformed = root / "valid.GGUF", root / "malformed.GGUF"
            write_adapter(valid)
            write_adapter(
                malformed,
                **{
                    "comfy.gguf.quant.linear.weight.lora_a": "[" * 20_000 + "0" + "]" * 20_000,
                },
            )
            with self.assertRaisesRegex(ValueError, "Invalid GGUF quantization metadata"):
                gguf_metadata.read_gguf_header(malformed, include_storage=True)
            self.assertEqual(discover_image_loras([root]), [valid.name])
            state = SimpleNamespace(
                models=SimpleNamespace(
                    get_names=lambda _: [valid.name, malformed.name],
                    model_dirs={"loras": [root]},
                )
            )
            with (
                patch.dict(sys.modules, {"ruinedfooocus.app.state": state}),
                patch("ruinedfooocus.app.state", state, create=True),
            ):
                self.assertEqual(
                    api_runtime.resolve_image_loras([{"name": valid.name, "strength": 0.5}]),
                    [[valid.name, "0.5 - valid.GGUF"]],
                )
                with self.assertRaisesRegex(ValueError, "Invalid GGUF quantization metadata"):
                    api_runtime.resolve_image_loras([{"name": malformed.name, "strength": 1}])
            write_adapter(
                malformed,
                **{"comfy.gguf.quant.linear.weight.lora_a": '{"format":[],"orig_shape":[32,32]}'},
            )
            with self.assertRaisesRegex(ValueError, "metadata format"):
                inspect_gguf_lora(malformed)
            self.assertEqual(discover_image_loras([root]), [valid.name])
