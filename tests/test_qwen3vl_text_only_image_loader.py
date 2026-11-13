"""Metadata-only admission and actual pinned single-encoder Q8_CR routing on CPU."""

import ast
from contextlib import nullcontext
from functools import partial
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

from ruinedfooocus.backends.comfy.text_encoder_loader import _GGUFReaderPath
from ruinedfooocus.features.models.gguf_metadata import qwen3vl_text_only_variant

ROOT = Path(__file__).resolve().parents[1]
NATIVE = (
    Path(os.environ.get("RF_TEST_GGUF_SOURCE", ROOT / "repositories/molbal_comfyui_gguf"))
    / "nodes.py"
)


def metadata_file(path, *, arch="qwen3vl", variant="4b", text_only=True, order="<"):
    def pack(kind, value):
        return struct.pack(order + kind, value)

    def string(value):
        value = value.encode()
        return pack("Q", len(value)) + value

    fields = [("general.architecture", 8, arch), ("comfy.gguf.qwen3vl.variant", 8, variant)]
    if text_only is not None:
        fields.append(("comfy.gguf.qwen3vl.text_only", 7, text_only))
    data = b"GGUF" + pack("I", 3) + pack("Q", 0) + pack("Q", len(fields))
    for key, kind, value in fields:
        data += string(key) + pack("I", kind)
        data += string(value) if kind == 8 else pack("?", value)
    path.write_bytes(data)
    return path


def native_functions(scope):
    tree = ast.parse(NATIVE.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_load_dynamic_gguf_clip", "_prune_qwen3vl_vision")
    ]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(NATIVE), "exec"), scope)
    return scope["_load_dynamic_gguf_clip"]


class Qwen3VLTextOnlyImageLoaderTests(unittest.TestCase):
    def test_exact_typed_marker_in_both_byte_orders_never_reads_tensor_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "encoder.GGUF"
            for order in ("<", ">"):
                for variant in ("4b", "8b"):
                    metadata_file(path, variant=variant, order=order)
                    self.assertEqual(qwen3vl_text_only_variant(path), variant)
            for marker in (None, False):
                metadata_file(path, text_only=marker)
                self.assertIsNone(qwen3vl_text_only_variant(path))
            for values in ({"arch": "qwen3"}, {"variant": "32b"}):
                metadata_file(path, **values)
                with self.assertRaisesRegex(ValueError, "4b or 8b"):
                    qwen3vl_text_only_variant(path)
            metadata_file(path)
            raw = path.read_bytes()
            path.write_bytes(raw[:-1])
            with self.assertRaises(ValueError):
                qwen3vl_text_only_variant(path)

    def fixture(self):
        transformer = NS(model_type="qwen3vl_4b", visual=object())
        clip = NS(cond_stage_model=NS(modules=lambda: [transformer]), patcher=NS())
        state = {"owned-tiny-state": object()}
        read = Mock(
            return_value=(
                state,
                dict(
                    arch_str="qwen3vl",
                    qwen3vl_text_only=True,
                    gguf_quant_formats=("int8_tensorwise",),
                ),
            )
        )
        progress = NS(callback_for=Mock(return_value="progress"), complete_file=Mock())
        load = Mock(return_value=clip)
        restore = Mock(return_value=1)
        required = Mock()
        scope = dict(
            _require_dynamic_vram=required,
            GGUFLoadProgress=lambda paths: progress,
            gguf_clip_loader=read,
            _qwen3vl_q8_layout_available=lambda: False,
            materialize_int8_source_weights=restore,
            logging=Mock(),
            comfy=NS(
                model_management=NS(text_encoder_offload_device=lambda: "offload"),
                sd=NS(load_text_encoder_state_dicts=load),
            ),
            GGMLOps=object(),
            _suppress_expected_qwen3vl_vision_warnings=nullcontext,
            _clone_as_dynamic_gguf_patcher=lambda patcher: NS(),
            partial=partial,
            _load_dynamic_gguf_clip_patcher=lambda *a, **k: None,
            folder_paths=NS(get_folder_paths=lambda role: []),
        )
        return NS(
            load=native_functions(scope),
            clip=clip,
            transformer=transformer,
            read=read,
            state=state,
            construct=load,
            restore=restore,
            required=required,
        )

    def test_actual_pin_retains_eligibility_fallback_reload_and_image_input_refusal(self):
        native = self.fixture()
        path = _GGUFReaderPath("Original/Qwen.GGUF")
        self.assertIs(native.load([path], "FLUX", enable_q8_native=True), native.clip)
        native.required.assert_called_once_with()
        native.read.assert_called_once_with(
            path, dynamic=True, progress_callback="progress", return_extra=True
        )
        native.restore.assert_called_once_with(native.state)
        self.assertEqual(native.construct.call_args.kwargs["state_dicts"], [native.state])
        self.assertFalse(native.construct.call_args.kwargs["disable_dynamic"])
        self.assertIsNone(native.transformer.visual)
        with self.assertRaisesRegex(ValueError, "text-only.*image inputs"):
            native.transformer.preprocess_embed({"type": "image"}, "cpu")
        self.assertEqual(native.transformer.preprocess_embed({"type": "text"}, "cpu"), (None, None))
        factory, arguments = native.clip.patcher.cached_patcher_init
        self.assertTrue(factory.keywords["enable_q8_native"])
        self.assertEqual(arguments, ([path], "FLUX"))

    def test_actual_pin_refuses_disabled_dynamic_vram_before_any_weight_reader(self):
        native = self.fixture()
        native.required.side_effect = RuntimeError("Dynamic VRAM is not enabled")
        with self.assertRaisesRegex(RuntimeError, "Dynamic VRAM"):
            native.load([_GGUFReaderPath("qwen.gguf")], "FLUX", enable_q8_native=True)
        native.read.assert_not_called()
        native.construct.assert_not_called()


if __name__ == "__main__":
    unittest.main()
