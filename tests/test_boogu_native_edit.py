"""Execute pinned Boogu conditioning with weightless CPU image/encoder fixtures."""

import ast
from contextlib import contextmanager
import math
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

from ruinedfooocus.features.image_editing.ordered_native_references import (
    OrderedNativeReferences,
    recipe_for,
)
from ruinedfooocus.features.models.image_components import native_edit_supported

# The shared native fixtures restore sys.modules snapshots. Load these real caller
# dataclass owners before those snapshots so their identities survive API capture.
from ruinedfooocus.features.models import lora_sources as _lora_sources
from ruinedfooocus.features.generation import image_output_batches as _image_output_batches

from test_ordered_native_references import process_nodes, runtime, vae_fixture
from test_qwen_image21_conditioning import native_class, native_function


@contextmanager
def native_modules(modules):
    missing = object()
    original = {key: sys.modules.get(key, missing) for key in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is missing:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def native_runtime():
    modules = runtime()
    scope = dict(
        math=math,
        comfy=NS(
            utils=NS(
                common_upscale=native_function(
                    "comfy/utils.py", "common_upscale", dict(torch=torch)
                )
            )
        ),
        node_helpers=modules["node_helpers"],
        io=NS(NodeOutput=lambda *values: values),
    )
    boogu = native_class(
        "comfy_extras/nodes_boogu.py", "TextEncodeBooguEdit", scope, methods=("execute",)
    )
    modules["comfy_extras.nodes_boogu"] = NS(TextEncodeBooguEdit=boogu)
    return modules


def clip_fixture():
    return NS(
        tokenize=Mock(side_effect=lambda text, **kwargs: dict(text=text, **kwargs)),
        encode_from_tokens_scheduled=Mock(
            side_effect=lambda tokens: [
                [torch.full((1, 2, 4), float(len(tokens["text"]))), {"text": tokens["text"]}]
            ]
        ),
    )


class BooguNativeEditTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(native_modules(native_runtime()))
        self.enterContext(
            patch("ruinedfooocus.backends.comfy.text_encoder_warmup.warm_patched_text_encoder")
        )

    def test_auto_needs_current_edit_variant_and_explicit_recipe_does_not_admit_turbo(self):
        self.assertTrue(native_edit_supported("Boogu"))
        self.assertEqual(
            recipe_for("Boogu", source=dict(native_family="Boogu", behavioral_variant="edit")),
            "boogu-edit",
        )
        self.assertEqual(recipe_for("Boogu", "boogu-edit"), "boogu-edit")
        for source in (None, dict(native_family="Boogu", behavioral_variant="turbo")):
            with self.assertRaises(ValueError):
                recipe_for("Boogu", source=source)
        with self.assertRaises(ValueError):
            recipe_for(
                "Boogu", "boogu-edit", dict(native_family="Boogu", behavioral_variant="turbo")
            )

        with self.assertRaisesRegex(ValueError, "eightfold"):
            OrderedNativeReferences(
                "boogu-edit", vae_fixture(scale=16), (Image.new("RGB", (32, 32)),)
            )

    def test_native_pair_preserves_negative_and_positive_only_vision_with_shared_latents(self):
        clip, vae = clip_fixture(), vae_fixture()
        images = (torch.full((1, 24, 40, 3), 0.2), torch.full((1, 40, 24, 3), 0.7))
        refs = OrderedNativeReferences("boogu-edit", vae, images)
        images[0].zero_()
        positive, negative = refs.conditioning(clip, "red coat", "blur")
        self.assertEqual(positive[0][1]["text"], "red coat")
        self.assertEqual(negative[0][1]["text"], "blur")
        calls = clip.tokenize.call_args_list
        self.assertEqual(len(calls[0].kwargs["images"]), 2)
        self.assertNotIn("images", calls[1].kwargs)
        self.assertEqual(vae.encode.call_count, 2)
        for first, second in zip(
            positive[0][1]["reference_latents"], negative[0][1]["reference_latents"]
        ):
            self.assertIs(first, second)
        self.assertAlmostEqual(float(positive[0][1]["reference_latents"][0].mean()), 0.2, places=5)

    def test_scheduled_native_positives_reuse_reference_encoding_and_keep_original_inputs(self):
        clip, vae = clip_fixture(), vae_fixture()
        refs = OrderedNativeReferences("boogu-edit", vae, (Image.new("RGB", (48, 32), "red"),))
        _, negative, schedule = refs.scheduled_conditioning(clip, "[red|blue]", "blur", 4)
        self.assertEqual(vae.encode.call_count, 1)
        self.assertEqual(negative[0][1]["text"], "blur")
        self.assertEqual([row[1]["text"] for row in schedule], ["red", "blue", "red", "blue"])
        self.assertTrue(
            all(
                row[1]["reference_latents"][0] is negative[0][1]["reference_latents"][0]
                for row in schedule
            )
        )
        refs.images[0].add_(0.1)
        with self.assertRaisesRegex(ValueError, "changed after submission"):
            refs.conditioning(clip, "again", "")

    def test_actual_production_ordered_caller_uses_boogu_recipe_and_keeps_extra_reference_aspect(
        self,
    ):
        nodes = process_nodes()
        prepare = next(
            n
            for n in nodes
            if isinstance(n, ast.If) and ast.unparse(n.test) == "ordered_recipe is not None"
        )
        encode = next(
            n
            for n in nodes
            if isinstance(n, ast.If)
            and "ordered_references is not None and ordered_recipe !=" in ast.unparse(n.test)
        )
        vae = vae_fixture()
        scope = dict(
            ordered_recipe="boogu-edit",
            ordered_source=None,
            ordered_references=None,
            OrderedNativeReferences=OrderedNativeReferences,
            self=NS(xl_base_patched=NS(vae=vae, clip=clip_fixture())),
            input_image=torch.full((1, 32, 48, 3), 0.3),
            additional_references=[Image.new("RGB", (16, 40), "blue")],
            reference_tensor=lambda image, *size: torch.zeros((1, image.height, image.width, 3)),
            gen_data=dict(width=64, height=64, steps=2),
            clip_skip=1,
            positive_prompt="red",
            negative_prompt="blur",
            clean_prompt_cond_caches=lambda: {
                key: dict(cache=None, text=None) for key in ("+", "-", "switch")
            },
        )
        exec(
            compile(
                ast.Module(body=[prepare, encode], type_ignores=[]),
                "production Boogu caller",
                "exec",
            ),
            scope,
        )
        self.assertEqual(tuple(scope["ordered_references"].images[1].shape), (1, 40, 16, 3))
        self.assertEqual(scope["negative_conditioning"][0][1]["text"], "blur")
        self.assertEqual(scope["gen_data"]["native_edit_recipe"], "boogu-edit")

    def test_typed_api_and_actual_image_admission_select_native_edit(self):
        from test_api_image_edits import queued_data
        from ruinedfooocus.integrations.openai_api import ImageRequest

        self.assertEqual(
            ImageRequest(
                input_image="fixture", native_edit_recipe="boogu-edit", reference_images=["r"] * 15
            ).native_edit_recipe,
            "boogu-edit",
        )
        captured = queued_data(
            "Boogu", payload_options=dict(native_edit_recipe="boogu-edit", reference_images=[])
        )
        self.assertEqual(captured["controlnet"], {"type": "edit"})
        self.assertEqual(captured["native_edit_recipe"], "boogu-edit")
