"""Native text classification and codec contracts survive unrelated filenames/precision."""

import ast
from enum import Enum
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.features.models.component_candidates import inspect_local_component
from ruinedfooocus.features.models.component_profiles import component_contract
from test_component_candidates import qwen_wan21_shapes

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))


def text_detector():
    tree = ast.parse((NATIVE / "comfy/sd.py").read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in ("TEModel", "detect_te_model")
    ]
    scope = {"Enum": Enum}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(NATIVE / "comfy/sd.py"), "exec"), scope
    )
    return scope["detect_te_model"]


def header(path, shapes, dtype):
    value = {
        key: dict(dtype=dtype, shape=list(shape), data_offsets=[0, 0])
        for key, shape in shapes.items()
    }
    data = json.dumps(value).encode()
    path.write_bytes(len(data).to_bytes(8, "little") + data)
    return path


class ComponentFamilyContractTests(unittest.TestCase):
    def test_anima_encoder_alternate_precision_uses_native_logical_family_not_name(self):
        detector = text_detector()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(sys.modules, {"comfy.sd": SimpleNamespace(detect_te_model=detector)}),
        ):
            root = Path(folder)
            for dtype in ("F16", "BF16", "F8_E4M3"):
                path = header(
                    root / f"unrelated-{dtype}.safetensors",
                    {
                        "model.layers.0.post_attention_layernorm.weight": (1024,),
                        "model.layers.0.self_attn.q_norm.weight": (128,),
                    },
                    dtype,
                )
                self.assertEqual(
                    inspect_local_component("clip_qwen3_06b", path)["compatibility"], "compatible"
                )
                self.assertEqual(
                    inspect_local_component("clip_qwen3_4b", path)["compatibility"], "incompatible"
                )
            contract = component_contract("Anima")
            self.assertEqual(contract["roles"][0]["logical_types"], ["QWEN3_06B"])

    def test_native_tokenizer_requirement_and_rgb_codec_cannot_be_replaced_by_size_label(self):
        detector = text_detector()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(sys.modules, {"comfy.sd": SimpleNamespace(detect_te_model=detector)}),
        ):
            root = Path(folder)
            encoder = header(
                root / "official-tokenizer-name.safetensors",
                {
                    "layers.0.self_attn.sinks": (8,),
                    "layers.0.mlp.experts.gate_up_proj.weight": (1, 1),
                },
                "F16",
            )
            self.assertEqual(
                inspect_local_component("clip_lens_gpt_oss", encoder)["compatibility"],
                "incompatible",
            )
            vae = header(root / "qwen_image21_64channel.safetensors", qwen_wan21_shapes(), "BF16")
            self.assertEqual(
                inspect_local_component("vae_qwen_image", vae)["compatibility"], "compatible"
            )
            self.assertEqual(
                inspect_local_component("vae_qwen_image21", vae)["compatibility"], "incompatible"
            )
            self.assertFalse(
                next(
                    row
                    for row in component_contract("QwenImage21")["roles"]
                    if row["role"] == "vae"
                )["gguf_image_supported"]
            )


if __name__ == "__main__":
    unittest.main()
