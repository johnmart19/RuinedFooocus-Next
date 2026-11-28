"""Logical GGUF shape inspection never reads payloads or initializes Torch."""

import ast
import builtins
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import sys
from unittest.mock import Mock, patch

from gguf import GGUFValueType as T
from gguf import GGUFWriter
import numpy as np

from ruinedfooocus.features.models.gguf_metadata import read_gguf_header
from ruinedfooocus.features.models.gguf_metadata import read_scalar_strings
from ruinedfooocus.features.models.image_components import IMAGE_COMPONENTS, inventory

ROOT = Path(__file__).resolve().parents[1]


def field(value, *types):
    return SimpleNamespace(types=types, parts=[], contents=lambda: value)


class GgufMetadataTests(unittest.TestCase):
    def test_scalar_inventory_skips_vocabulary_in_both_byte_orders(self):
        from gguf import GGUFEndian

        with tempfile.TemporaryDirectory() as folder:
            for endian in (GGUFEndian.LITTLE, GGUFEndian.BIG):
                with self.subTest(endian=endian):
                    path = Path(folder) / f"{endian.name}.gguf"
                    writer = GGUFWriter(path, "qwen3", endianess=endian)
                    writer.add_array("tokenizer.ggml.tokens", ["one", "два", "三"])
                    writer.add_array("tokenizer.ggml.scores", [1.0, 2.0, 3.0])
                    writer.add_bool("unused.flag", True)
                    writer.add_string("general.type", "model")
                    writer.write_header_to_file()
                    writer.write_kv_data_to_file()
                    writer.write_tensors_to_file()
                    writer.close()
                    self.assertEqual(
                        read_scalar_strings(path, ("general.architecture", "general.type")),
                        {"general.architecture": "qwen3", "general.type": "model"},
                    )
                    path.unlink()

    def test_scalar_inventory_rejects_truncated_and_unbounded_metadata(self):
        import struct

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.gguf"
            for payload in (b"GGUF", b"GGUF" + struct.pack("<IQQ", 3, 0, 100_001)):
                path.write_bytes(payload)
                with self.assertRaises(ValueError):
                    read_scalar_strings(path, ("general.architecture",))

    def read(self, fields, shape=None):
        shape = [2, 3] if shape is None else shape
        pack = lambda kind, value: struct.pack("<" + kind, value)
        string = lambda value: pack("Q", len(value.encode())) + value.encode()
        payload = b"GGUF" + pack("I", 3) + pack("Q", 1) + pack("Q", len(fields))
        formats = {T.UINT32: "I", T.INT32: "i", T.BOOL: "?"}
        for key, entry in fields.items():
            types, value = entry.types, entry.contents()
            payload += string(key) + pack("I", types[0])
            if types[0] == T.STRING:
                payload += string(value)
            elif types[0] == T.ARRAY:
                payload += pack("I", types[1]) + pack("Q", len(value))
                payload += b"".join(pack(formats[types[1]], item) for item in value)
            else:
                payload += pack(formats[types[0]], value)
        payload += string("layer.weight") + pack("I", len(shape))
        payload += b"".join(pack("Q", value) for value in reversed(shape))
        payload += pack("I", 0) + pack("Q", 0)  # F32, relative tensor offset.
        payload = payload.ljust((len(payload) + 31) // 32 * 32, b"\0")
        tensor_start = len(payload)
        payload += b"\0" * (math.prod(shape) * 4)
        self.closed = Mock()
        original_import = builtins.__import__
        original_open = builtins.open
        closed = self.closed

        class HeaderFile:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()
                closed()

            def __getattr__(self, key):
                return getattr(self.stream, key)

            def read(self, size):
                if self.stream.tell() + size > tensor_start:
                    raise AssertionError("Tensor payload was accessed")
                return self.stream.read(size)

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "comfy", "molbal_comfyui_gguf"):
                raise AssertionError("Metadata inspection initialized a tensor runtime")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.gguf"
            path.write_bytes(payload)
            with (
                patch("gguf.GGUFReader", side_effect=AssertionError("No full mapped reader")),
                patch(
                    "builtins.open",
                    side_effect=lambda *a, **kw: HeaderFile(original_open(*a, **kw)),
                ),
                patch("builtins.__import__", side_effect=guarded),
            ):
                return read_gguf_header(path)

    def test_original_nd_shape_and_native_metadata_survive_without_payload_access(self):
        header, architecture = self.read(
            {
                "general.architecture": field("sdxl", T.STRING),
                "config": field('{"transformer":{"pdd_steps":4}}', T.STRING),
                "is_512": field(True, T.BOOL),
                "comfy.gguf.orig_shape.layer.weight": field([2, 3, 1, 1, 1], T.ARRAY, T.INT32),
            }
        )
        self.assertEqual(architecture, "sdxl")
        self.assertEqual(header["layer.weight"]["shape"], [2, 3, 1, 1, 1])
        self.assertTrue(header["__metadata__"]["is_512"])
        self.closed.assert_called_once()

    def test_packed_shapes_use_declared_logical_dimensions_without_unpacking(self):
        for fmt, divisor in (("int4_cr", 2), ("int1_cr", 8)):
            with self.subTest(format=fmt):
                header, _ = self.read(
                    {
                        "comfy.gguf.quant.layer.weight": field(
                            json.dumps({"format": fmt, "orig_shape": [2, 256]}),
                            T.STRING,
                        )
                    },
                    [2, 256 // divisor],
                )
                self.assertEqual(header["layer.weight"]["shape"], [2, 256])
                self.closed.assert_called_once()

    def test_scalar_metadata_tensor_shape_is_preserved(self):
        header, _ = self.read({}, [])
        self.assertEqual(header["layer.weight"]["shape"], [])
        self.closed.assert_called_once()

    def test_real_small_gguf_round_trip_uses_typed_metadata_and_releases_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "small.gguf"
            writer = GGUFWriter(str(path), "sdxl")
            writer.add_tensor("layer.weight", np.zeros((2, 3, 1, 1), dtype=np.float32))
            writer.add_key_value(
                "comfy.gguf.orig_shape.layer.weight", [2, 3, 1, 1, 1], T.ARRAY, T.INT32
            )
            writer.add_string("config", '{"transformer":{"pdd_steps":4}}')
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            header, arch = read_gguf_header(path)
            self.assertEqual(arch, "sdxl")
            self.assertEqual(header["layer.weight"]["shape"], [2, 3, 1, 1, 1])
            self.assertIn("pdd_steps", header["__metadata__"]["config"])
            path.unlink()

    def test_malformed_conflicting_and_retired_metadata_fail_closed_and_release_file(self):
        cases = (
            {"comfy.gguf.orig_shape.layer.weight": field([2, 4], T.ARRAY, T.INT32)},
            {"comfy.gguf.orig_shape.layer.weight": field([0, 6], T.ARRAY, T.INT32)},
            {"comfy.gguf.orig_shape.layer.weight": field([2, 3], T.ARRAY, T.UINT32)},
            {"comfy.gguf.quant.layer.weight": field("not JSON", T.STRING)},
            {
                "comfy.gguf.quant.layer.weight": field(
                    json.dumps(
                        {
                            "format": "int4_cr",
                            "orig_shape": [2, 256],
                        }
                    ),
                    T.STRING,
                )
            },
            {
                "comfy.gguf.quant.layer.weight": field(
                    json.dumps(
                        {
                            "format": "unknown",
                            "orig_shape": [2, 3],
                        }
                    ),
                    T.STRING,
                )
            },
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.read(values)
            self.closed.assert_called_once()
        for fmt in ("int4_compact_gemm", "int4_pytorch"):
            with self.assertRaisesRegex(ValueError, "retired.*Q8_CR"):
                self.read(
                    {
                        "comfy.gguf.quant.layer.weight": field(
                            json.dumps({"format": fmt}),
                            T.STRING,
                        )
                    }
                )
            self.closed.assert_called_once()

    def test_registry_consumes_the_same_header_as_inventory(self):
        source = ROOT / "ruinedfooocus/features/models/model_handler.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_detect_checkpoint_base"
        )
        node.decorator_list = []
        scope = {"Path": Path}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        header = {"weight": {"shape": [2, 256]}, "__metadata__": {"config": "{}"}}
        with (
            patch(
                "ruinedfooocus.features.models.gguf_metadata.read_gguf_header",
                return_value=(header, "flux"),
            ),
            patch(
                "ruinedfooocus.features.models.model_architecture.detect_tensor_architecture",
                return_value="Flux.1 D",
            ) as detect,
        ):
            self.assertEqual(scope["_detect_checkpoint_base"](Path("fixture.gguf")), "Flux.1 D")
        detect.assert_called_once_with(header)

    def test_shared_hyvid_tag_defers_to_exact_native_image_classification(self):
        source = ROOT / "ruinedfooocus/features/models/model_handler.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_detect_checkpoint_base"
        )
        node.decorator_list = []
        scope = {"Path": Path}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        header = {"weight": {"shape": [2, 256]}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "shared-tag.gguf"
            path.write_bytes(b"fixture")
            with (
                patch(
                    "ruinedfooocus.features.models.gguf_metadata.read_gguf_header",
                    return_value=(header, "hyvid"),
                ),
                patch(
                    "ruinedfooocus.features.models.model_architecture.detect_tensor_architecture",
                ) as detect,
            ):
                for label in ("HunyuanImage21", "HunyuanImage21Refiner"):
                    detect.return_value = label
                    self.assertEqual(scope["_detect_checkpoint_base"](path), label)
                    # The helper admits classification, not an unimplemented recipe.
                    with (
                        patch.dict(
                            IMAGE_COMPONENTS, {label: {"clip_names": [], "vae_name": "pixel_space"}}
                        ),
                        patch.dict(
                            sys.modules,
                            {"comfy.model_detection": SimpleNamespace()},
                        ),
                        patch(
                            "ruinedfooocus.features.models.model_architecture.inspect_tensor_model",
                            return_value=dict(
                                architecture=label, vae_prefixes=(), text_encoder_prefixes=()
                            ),
                        ),
                    ):
                        self.assertEqual(
                            inventory(path, "Unknown", {}, SimpleNamespace())["architecture"], label
                        )
                for label in (None, "HunyuanVideo15", "Hunyuan Video"):
                    detect.return_value = label
                    self.assertEqual(scope["_detect_checkpoint_base"](path), "Hunyuan Video")
                with patch.dict(sys.modules):
                    sys.modules.pop("comfy.model_detection", None)
                    with self.assertRaisesRegex(ValueError, "image checkpoints only"):
                        inventory(path, "Unknown", {}, SimpleNamespace())
