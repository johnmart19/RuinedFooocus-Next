from ruinedfooocus.core.paths import asset_path, resource_path

"""Compare bounded prefill against the installed upstream math on CPU."""
import ast
from pathlib import Path
import unittest
from types import ModuleType
from unittest.mock import patch

import torch
import torch.nn.functional as F

from ruinedfooocus.backends.comfy.sequence_prefill import (
    install_qwen35_prefill,
    windowed_delta_rule,
)


def native_delta_rule():
    source = (
        Path(__file__).resolve().parents[1] / "repositories/ComfyUI/comfy/text_encoders/qwen35.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "torch_chunk_gated_delta_rule"
    )
    namespace = {"asset_path": asset_path, "resource_path": resource_path, "torch": torch, "F": F}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[node.name]


class SequencePrefillTests(unittest.TestCase):
    def test_install_is_shared_and_idempotent(self):
        encoders = ModuleType("comfy.text_encoders")
        encoders.qwen35 = ModuleType("comfy.text_encoders.qwen35")
        encoders.qwen35.torch_chunk_gated_delta_rule = native_delta_rule()
        with patch.dict("sys.modules", {"comfy.text_encoders": encoders}):
            install_qwen35_prefill()
            first = encoders.qwen35.torch_chunk_gated_delta_rule
            install_qwen35_prefill()
            self.assertIs(encoders.qwen35.torch_chunk_gated_delta_rule, first)

    def test_matches_native_with_tail_state_and_precision(self):
        native = native_delta_rule()
        for length, chunk, window in [(31, 64, 128), (321, 64, 128), (1025, 64, 512), (97, 16, 33)]:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(length=length, dtype=dtype, chunk=chunk):
                    generator = torch.Generator().manual_seed(42)
                    q, k, v = [
                        torch.randn(1, length, 2, 8, generator=generator).to(dtype)
                        for _ in range(3)
                    ]
                    g = -torch.rand(1, length, 2, generator=generator).to(dtype)
                    beta = torch.rand(1, length, 2, generator=generator).to(dtype)
                    state = torch.randn(1, 2, 8, 8, generator=generator)
                    expected, expected_state = native(q, k, v, g, beta, chunk, state, True)
                    actual, actual_state = windowed_delta_rule(native, window)(
                        q, k, v, g, beta, chunk, state, True
                    )
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(actual_state, expected_state, rtol=1e-5, atol=1e-6)
                    self.assertTrue(torch.isfinite(actual).all())

    def test_state_is_not_returned_when_not_requested(self):
        native = native_delta_rule()
        q = torch.ones(1, 129, 1, 4)
        g = -torch.ones(1, 129, 1)
        _, state = windowed_delta_rule(native, 64)(q, q, q, g, -g)
        self.assertIsNone(state)


if __name__ == "__main__":
    unittest.main()
