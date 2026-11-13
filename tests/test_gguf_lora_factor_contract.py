"""The native adapter reader, rather than checkpoint dequantization, owns factors."""

import os
from pathlib import Path
import runpy
import unittest

import gguf

ROOT = Path(__file__).resolve().parents[1]
GGUF = Path(os.environ.get("RF_TEST_GGUF_SOURCE", ROOT / "repositories/molbal_comfyui_gguf"))


class UnreadFactor:
    name = "linear.weight.lora_a"

    def __init__(self, kind):
        self.tensor_type = kind

    @property
    def shape(self):
        raise AssertionError(
            "Unsupported native factor types must fail before shape/materialization"
        )

    @property
    def data(self):
        raise AssertionError("Unsupported native factor types must fail before reading bytes")


class GgufLoraFactorContractTests(unittest.TestCase):
    def test_native_importer_explicitly_limits_factor_types_and_refuses_every_other_enum(self):
        native = runpy.run_path(str(GGUF / "lora.py"))
        expected = {
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.BF16,
            gguf.GGMLQuantizationType.Q8_0,
        }
        self.assertEqual(native["_SUPPORTED_FACTOR_TYPES"], expected)
        refused = set(gguf.GGMLQuantizationType) - expected
        # Guard the specific Q4/Q5/K/IQ question as well as future enum additions.
        for name in ("Q4_0", "Q5_0", "Q4_K", "Q5_K", "IQ4_XS"):
            self.assertIn(getattr(gguf.GGMLQuantizationType, name), refused)
        for kind in refused:
            with (
                self.subTest(kind=kind.name),
                self.assertRaisesRegex(
                    ValueError, "Only F32, F16, BF16, and Q8_0 factors are supported"
                ),
            ):
                native["_tensor_to_float"](UnreadFactor(kind))


if __name__ == "__main__":
    unittest.main()
