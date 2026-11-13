"""Native Qwen Image 2.1 T2I flags/metadata and request geometry on CPU."""

import ast
from concurrent.futures import CancelledError
import math
import numbers
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from ruinedfooocus.features.image_editing.qwen_image21_conditioning import (
    text_to_image_conditioning,
)
from test_native_model_coverage import NATIVE

MODULE = "ruinedfooocus.features.image_editing.qwen_image21_conditioning"


def native_class(filename, name, scope, *, methods, bases=()):
    tree = ast.parse((NATIVE / filename).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    node.bases = [ast.Name(id=base, ctx=ast.Load()) for base in bases]
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    for method in node.body:
        method.returns = None
        for argument in method.args.args + method.args.kwonlyargs:
            argument.annotation = None
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), filename, "exec"
        ),
        scope,
    )
    return scope[name]


def native_function(filename, name, scope):
    tree = ast.parse((NATIVE / filename).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), scope)
    return scope[name]


def runtime():
    upscale = Mock(wraps=native_function("comfy/utils.py", "common_upscale", dict(torch=torch)))
    native = SimpleNamespace(
        utils=SimpleNamespace(common_upscale=upscale),
        model_management=SimpleNamespace(intermediate_device=lambda: "cpu"),
    )
    output = native_class(
        "comfy_api/latest/_io.py",
        "NodeOutput",
        dict(Any=object, _UIOutput=object),
        methods=("__init__", "__getitem__"),
    )
    scope = dict(torch=torch, math=math, comfy=native, io=SimpleNamespace(NodeOutput=output))
    node = native_class(
        "comfy_extras/nodes_qwen.py", "TextEncodeQwenImage21", scope, methods=("execute",)
    )
    modules = {
        "comfy": native,
        "comfy.utils": native.utils,
        "comfy_extras.nodes_qwen": SimpleNamespace(TextEncodeQwenImage21=node),
    }
    return modules, upscale


def clip_fixture():
    mask = torch.tensor([[1, 1, 0]])
    clip = SimpleNamespace(
        tokenize=Mock(side_effect=lambda text, **kwargs: dict(text=text, **kwargs)),
        encode_from_tokens_scheduled=Mock(
            side_effect=lambda tokens: [
                [tokens["text"], dict(attention_mask=mask, native_marker=object())]
            ]
        ),
    )
    return clip, mask


class QwenImage21ConditioningTests(unittest.TestCase):
    def test_actual_native_paired_node_keeps_flags_empty_negative_and_nonsquare_target(self):
        for width, height in ((96, 64), (64, 96), (64, 64)):
            modules, upscale = runtime()
            clip, mask = clip_fixture()
            with (
                patch.dict("sys.modules", modules),
                patch(MODULE + ".warm_patched_text_encoder") as warm,
            ):
                positive, negative, latent = text_to_image_conditioning(
                    clip, "a cat", "", width, height
                )
            warm.assert_called_once_with(clip)
            self.assertEqual([call.args[0] for call in clip.tokenize.call_args_list], ["a cat", ""])
            for call in clip.tokenize.call_args_list:
                self.assertEqual(
                    call.kwargs, dict(images=[], keep_vision=True, prevent_empty_text=True)
                )
            self.assertEqual((positive[0][0], negative[0][0]), ("a cat", ""))
            self.assertIs(positive[0][1]["attention_mask"], mask)
            self.assertIs(negative[0][1]["attention_mask"], mask)
            self.assertEqual(tuple(latent["samples"].shape), (1, 64, height // 16, width // 16))
            self.assertEqual(latent["samples"].device.type, "cpu")
            self.assertTrue((latent["samples"] == 0).all())
            if width != height:
                self.assertEqual(
                    upscale.call_args.args[1:],
                    (width // 16, height // 16, "nearest-exact", "disabled"),
                )
            else:
                upscale.assert_not_called()

    def test_bad_dimensions_fail_before_encoding_and_never_round_request_silently(self):
        for width, height in ((0, 64), (-16, 64), (24, 64), (64, 65), (True, 64), (64.0, 64)):
            clip, _ = clip_fixture()
            with (
                patch(MODULE + ".warm_patched_text_encoder") as warm,
                self.assertRaisesRegex(ValueError, "multiples of 16"),
            ):
                text_to_image_conditioning(clip, "cat", "", width, height)
            warm.assert_not_called()
            clip.tokenize.assert_not_called()

    def test_cancellation_and_native_errors_propagate_without_retry_or_negative_encode(self):
        for error in (CancelledError("cancelled"), RuntimeError("encoder failed")):
            modules, _ = runtime()
            clip, _ = clip_fixture()
            clip.encode_from_tokens_scheduled.side_effect = error
            with (
                patch.dict("sys.modules", modules),
                patch(MODULE + ".warm_patched_text_encoder"),
                self.assertRaises(type(error)),
            ):
                text_to_image_conditioning(clip, "cat", "blur", 64, 64)
            clip.encode_from_tokens_scheduled.assert_called_once()
            clip.tokenize.assert_called_once()

    def test_a_later_request_does_not_change_earlier_conditions_or_latent(self):
        modules, _ = runtime()
        clip, _ = clip_fixture()
        with patch.dict("sys.modules", modules), patch(MODULE + ".warm_patched_text_encoder"):
            first = text_to_image_conditioning(clip, "first", "blur", 96, 64)
            marker = first[0][0][1]["native_marker"]
            second = text_to_image_conditioning(clip, "second", "noise", 64, 96)
        self.assertEqual((first[0][0][0], first[1][0][0]), ("first", "blur"))
        self.assertIs(first[0][0][1]["native_marker"], marker)
        self.assertEqual(tuple(first[2]["samples"].shape), (1, 64, 4, 6))
        self.assertEqual(tuple(second[2]["samples"].shape), (1, 64, 6, 4))
        self.assertIsNot(first[2]["samples"], second[2]["samples"])

    def test_actual_native_encoder_system_prefix_and_attention_mask_remain_aligned(self):
        context = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])

        class BaseEncoder:
            def encode_token_weights(self, pairs):
                return context, "pooled", dict(attention_mask=mask)

        scope = dict(torch=torch, numbers=numbers, BaseEncoder=BaseEncoder)
        encoder = native_class(
            "comfy/text_encoders/qwen_image21.py",
            "QwenImage21TEModel",
            scope,
            methods=("encode_token_weights",),
            bases=("BaseEncoder",),
        )()
        encoder.clip = "qwen3vl_8b"
        encoder.qwen3vl_8b = SimpleNamespace(image_spans=[])
        pairs = {
            "qwen3vl_8b": [[(token, 1) for token in (151644, 10, 11, 151644, 20, 21, 22, 23)]],
            "keep_vision": True,
        }
        encoded, pooled, extra = encoder.encode_token_weights(pairs)
        torch.testing.assert_close(encoded, context[:, 3:])
        torch.testing.assert_close(extra["attention_mask"], mask[:, 3:])
        self.assertEqual(pooled, "pooled")
        self.assertNotIn("image_slots", extra)


if __name__ == "__main__":
    unittest.main()
