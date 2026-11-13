"""Header browsing remains bounded, detached and safe when errors retain frames."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import gguf
import numpy as np

from ruinedfooocus.features.models import gguf_metadata as metadata

ROOT = Path(__file__).resolve().parents[1]


def binary(*, fields=(), shape=(2, 32), dtype=0, offset=0, body=256, tensor_count=1):
    def integer(kind, value):
        return struct.pack("<" + kind, value)

    def string(value):
        raw = value.encode()
        return integer("Q", len(raw)) + raw

    result = b"GGUF" + integer("I", 3) + integer("Q", tensor_count) + integer("Q", len(fields))
    for key, kind, value in fields:
        result += string(key) + integer("I", kind)
        if kind == 8:
            result += string(value)
        elif kind == 9:
            element, values = value
            result += integer("I", element) + integer("Q", len(values))
            result += b"".join(
                string(item) if element == 8 else integer("i", item) for item in values
            )
        else:
            result += integer("I", value)
    result += string("weight") + integer("I", len(shape))
    result += b"".join(integer("Q", size) for size in reversed(shape))
    result += integer("I", dtype) + integer("Q", offset)
    return result.ljust((len(result) + 31) // 32 * 32, b"\0") + b"\0" * body


class StreamingGgufTests(unittest.TestCase):
    def test_native_f16_q8_q4_shapes_and_typed_metadata_match_in_both_byte_orders(self):
        with tempfile.TemporaryDirectory() as folder:
            for endian in (gguf.GGUFEndian.LITTLE, gguf.GGUFEndian.BIG):
                for kind in (
                    gguf.GGMLQuantizationType.F16,
                    gguf.GGMLQuantizationType.Q8_0,
                    gguf.GGMLQuantizationType.Q4_K,
                ):
                    with self.subTest(endian=endian, kind=kind):
                        path = Path(folder) / f"{endian.name}-{kind.name}.gguf"
                        writer = gguf.GGUFWriter(path, "sdxl", endianess=endian)
                        values = np.zeros((2, 256), dtype=np.float32)
                        # The native Python library reads Q4_K but does not encode it.
                        # Zero blocks provide valid physical storage for header parity.
                        data = (
                            np.zeros((2, gguf.constants.GGML_QUANT_SIZES[kind][1]), dtype=np.uint8)
                            if kind == gguf.GGMLQuantizationType.Q4_K
                            else gguf.quantize(values, kind)
                        )
                        writer.add_tensor("weight", data, raw_dtype=kind)
                        writer.add_key_value(
                            "comfy.gguf.orig_shape.weight",
                            [2, 1, 16, 16],
                            gguf.GGUFValueType.ARRAY,
                            gguf.GGUFValueType.INT32,
                        )
                        writer.add_string("modelspec.variant", "base")
                        writer.add_string("config", '{"transformer":{"pdd_steps":4}}')
                        writer.add_bool("is_512", True)
                        writer.write_header_to_file()
                        writer.write_kv_data_to_file()
                        writer.write_tensors_to_file()
                        writer.close()

                        def native():
                            reader = gguf.GGUFReader(path)
                            return reader.tensors[0].shape.tolist(), {
                                key: reader.get_field(key).contents()
                                for key in ("config", "modelspec.variant", "is_512")
                            }

                        shape, declarations = native()
                        header, architecture = metadata.read_gguf_header(path)
                        self.assertEqual(shape, [256, 2])
                        self.assertEqual(architecture, "sdxl")
                        self.assertEqual(header["weight"]["shape"], [2, 1, 16, 16])
                        self.assertEqual(header["__metadata__"], declarations)
                        path.unlink()

    def test_tokenizer_is_skipped_and_task_architecture_never_constructs_a_mapped_reader(self):
        from ruinedfooocus.features.models.model_capabilities import local_architecture

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "arbitrary.gguf"
            path.write_bytes(
                binary(
                    fields=(
                        ("tokenizer.ggml.tokens", 9, (8, ["x" * 256] * 5000)),
                        ("general.architecture", 8, "qwen3"),
                    )
                )
            )
            with (
                patch("gguf.GGUFReader", side_effect=AssertionError("No mapped reader")),
                patch("numpy.memmap", side_effect=AssertionError("No mapped tensor views")),
                patch.object(metadata, "MAX_METADATA_BYTES", 256),
            ):
                self.assertEqual(metadata.read_gguf_header(path)[1], "qwen3")
                self.assertEqual(local_architecture(path), "qwen3")

    def test_counts_strings_dimensions_types_alignment_ranges_and_truncation_are_bounded(self):
        cases = [
            binary(tensor_count=100_001),
            binary(shape=(1,) * 9),
            binary(shape=(2**31 + 1,)),
            binary(dtype=999),
            binary(offset=2**64 - 1),
            binary(offset=1),
            binary()[:-1],
            binary(fields=(("general.alignment", 4, 3),)),
            binary(fields=(("modelspec.variant", 8, "x" * 257),)),
            binary(
                fields=(("general.architecture", 8, "sdxl"), ("general.architecture", 8, "other"))
            ),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "malformed.gguf"
            for payload in cases:
                path.write_bytes(payload)
                with self.subTest(payload_length=len(payload)), self.assertRaises(ValueError):
                    metadata.read_gguf_header(path)
            path.write_bytes(binary())
            with patch.object(metadata, "MAX_HEADER_BYTES", 32), self.assertRaises(ValueError):
                metadata.read_gguf_header(path)

    def test_changed_source_is_rejected_and_parallel_results_are_plain_python_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.gguf"
            payload = binary(fields=(("general.architecture", 8, "sdxl"),))
            path.write_bytes(payload)
            with ThreadPoolExecutor(max_workers=4) as executor:
                values = list(executor.map(metadata.read_gguf_header, [path] * 16))
            self.assertTrue(
                all(value == ({"weight": {"shape": [2, 32]}}, "sdxl") for value in values)
            )
            self.assertEqual(json.loads(json.dumps(values))[0][0]["weight"]["shape"], [2, 32])
            unchanged = metadata._HeaderStream.unchanged

            def mutate(reader, selected):
                with path.open("ab") as output:
                    output.write(b"x")
                unchanged(reader, selected)

            with (
                patch.object(metadata._HeaderStream, "unchanged", mutate),
                self.assertRaises(OSError),
            ):
                metadata.read_gguf_header(path)

    def test_retained_error_frame_formats_safely_in_a_cpu_subprocess(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad-shape.gguf"
            path.write_bytes(binary(fields=(("comfy.gguf.orig_shape.weight", 9, (5, [2, 33])),)))
            code = """
import sys,traceback
from ruinedfooocus.features.models.gguf_metadata import read_gguf_header
try:
    read_gguf_header(sys.argv[1])
except ValueError as error:
    traceback.TracebackException.from_exception(error,capture_locals=True)
    print('safe retained frame')
else:
    raise AssertionError('Expected corrupt shape rejection')
assert 'torch' not in sys.modules
"""
            result = subprocess.run(
                [sys.executable, "-X", "faulthandler", "-c", code, str(path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("safe retained frame", result.stdout)


if __name__ == "__main__":
    unittest.main()
