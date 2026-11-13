from ruinedfooocus.core.paths import asset_path, resource_path

"""Native O1 reference conditioning and request isolation; CPU, no weights."""
import ast
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image
import torch

from ruinedfooocus.features.image_editing.image_geometry import (
    prepare_reference,
    restore_reference_frame,
)
from ruinedfooocus.features.generation.size_plan import SizePlan
from ruinedfooocus.features.image_editing.image_guidance import (
    capabilities,
    normalize_reference_images,
    reference_image_map,
)

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))


def native_reference():
    """Use the pinned node, conditioning copier and NodeOutput sequence protocol."""
    output_tree = ast.parse((NATIVE / "comfy_api/latest/_io.py").read_text(encoding="utf-8"))
    output = next(
        n for n in output_tree.body if isinstance(n, ast.ClassDef) and n.name == "NodeOutput"
    )
    output.bases = []
    output.body = [
        n
        for n in output.body
        if isinstance(n, ast.FunctionDef) and n.name in ("__init__", "__getitem__", "result")
    ]
    namespace = dict(
        asset_path=asset_path,
        resource_path=resource_path,
        Any=Any,
        _UIOutput=type("UIOutput", (), {}),
    )
    exec(compile(ast.Module(body=[output], type_ignores=[]), "native_output", "exec"), namespace)
    helper_tree = ast.parse((NATIVE / "node_helpers.py").read_text(encoding="utf-8"))
    helper = next(
        n
        for n in helper_tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "conditioning_set_values"
    )
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), "native_conditioning", "exec"),
        namespace,
    )
    tree = ast.parse((NATIVE / "comfy_extras/nodes_hidream_o1.py").read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HiDreamO1ReferenceImages"
    )
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "execute"]
    namespace.update(
        REFERENCE_IMAGE_INPUT_SLOTS=100,
        node_helpers=SimpleNamespace(conditioning_set_values=namespace["conditioning_set_values"]),
        io=SimpleNamespace(NodeOutput=namespace["NodeOutput"], Autogrow=SimpleNamespace(Type=dict)),
    )
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), "native_o1_references", "exec"), namespace
    )
    return namespace[cls.name]


def process_tree():
    tree = ast.parse(
        (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(encoding="utf-8")
    )
    return tree, next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process"
    )


def reference_mode(process):
    return next(
        n
        for n in process.body
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "hidream_edit and input_image is not None and (not img2img_mode)"
    )


class HiDreamReferenceTests(unittest.TestCase):
    def test_reference_prompt_switching_has_no_false_controlnet_warning(self):
        _, process = process_tree()
        branch = next(
            n
            for n in ast.walk(process)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "image_edit and input_images > 0"
        )
        helper_tree = ast.parse((NATIVE / "node_helpers.py").read_text(encoding="utf-8"))
        copier = next(
            n
            for n in helper_tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "conditioning_set_values"
        )
        owner = SimpleNamespace(
            conditions={key: {"cache": [[key, {}]]} for key in ("+", "-", "switch")},
            textencode=Mock(),
        )
        owner.textencode.side_effect = lambda sign, text, skip: owner.conditions[sign].update(
            cache=[[text, {}]]
        )
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            self=owner,
            callback=None,
            positive_prompt="[first|second]",
            negative_prompt="",
            clip_skip=1,
            input_image=object(),
            hidream_edit=True,
            architecture="HiDreamO1",
            img2img_mode=False,
            control_type="None",
            controlnet={"type": "None"},
            switched_prompt=[],
            gen_data={"steps": 2},
            pp=SimpleNamespace(prompt_switch_per_step=lambda *args: ["first", "second"]),
        )
        exec(
            compile(ast.Module(body=[copier], type_ignores=[]), "native_conditioning", "exec"),
            scope,
        )
        output = StringIO()
        with redirect_stdout(output):
            exec(
                compile(
                    ast.Module(body=[reference_mode(process), *branch.orelse], type_ignores=[]),
                    "prompt_switching",
                    "exec",
                ),
                scope,
            )
        self.assertIsNone(scope["controlnet"])
        self.assertNotIn("ControlNet", output.getvalue())
        self.assertEqual(
            [(row[1]["start_percent"], row[1]["end_percent"]) for row in scope["switched_prompt"]],
            [(0, 0.5), (0.5, 1)],
        )

    def test_native_socket_order_and_negative_metadata_leave_text_cache_unchanged(self):
        node = native_reference()
        positive = [[torch.zeros(1, device="cpu"), {"text_input_ids": "positive"}]]
        negative = [[torch.ones(1, device="cpu"), {"text_input_ids": "negative"}]]
        refs = {
            f"image_{i}": torch.full((1, 2, 3, 3), i / 16, device="cpu")
            for i in reversed(range(1, 17))
        }
        output = node.execute(positive=positive, negative=negative, images=refs)
        pos, neg = output
        for branch in (pos, neg):
            self.assertEqual(
                [float(x[0, 0, 0, 0]) for x in branch[0][1]["reference_latents"]],
                [i / 16 for i in range(1, 17)],
            )
        self.assertEqual(neg[0][1]["prompt_type"], "negative")
        self.assertEqual(positive[0][1], {"text_input_ids": "positive"})
        self.assertEqual(negative[0][1], {"text_input_ids": "negative"})
        self.assertIs(node.execute(positive=positive, negative=negative)[0], positive)

    def test_route_attaches_current_refs_and_switched_prompts_without_cache_accumulation(self):
        tree, process = process_tree()
        branch = next(
            n
            for n in process.body
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "hidream_edit and input_image is not None"
        )
        transform = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reference_tensor"
        )
        scope = dict(asset_path=asset_path, resource_path=resource_path, torch=torch, np=np)
        exec(
            compile(ast.Module(body=[transform], type_ignores=[]), "reference_tensor", "exec"),
            scope,
        )
        positive = [[torch.zeros(1, device="cpu"), {"text_input_ids": "positive"}]]
        negative = [[torch.ones(1, device="cpu"), {"text_input_ids": "negative"}]]
        scope.update(
            HiDreamO1ReferenceImages=native_reference(),
            reference_image_map=reference_image_map,
            self=SimpleNamespace(conditions={"+": {"cache": positive}, "-": {"cache": negative}}),
            hidream_edit=True,
            additional_references=[Image.new("RGB", (5, 9), (13, 26, 39))],
        )
        for mode, level in ((False, 0.2), (True, 0.8)):
            image = torch.full((1, 32, 64, 3), level, device="cpu")
            switched = [[torch.zeros(1, device="cpu"), {"start_percent": 0.5, "end_percent": 1}]]
            control = {"type": "img2img", "denoise": 0.25} if mode else {"type": "None"}
            scope.update(
                input_image=image,
                positive_conditioning=positive,
                negative_conditioning=negative,
                switched_prompt=switched,
                img2img_mode=mode,
                controlnet=control,
                input_images=1,
            )
            exec(
                compile(
                    ast.Module(body=[reference_mode(process), branch], type_ignores=[]),
                    "native_edit_route",
                    "exec",
                ),
                scope,
            )
            for key in ("positive_conditioning", "negative_conditioning", "switched_prompt"):
                refs = scope[key][0][1]["reference_latents"]
                self.assertEqual(len(refs), 2)
                self.assertIs(refs[0], image)
                self.assertEqual(tuple(refs[1].shape), (1, 9, 5, 3))
            self.assertEqual(scope["switched_prompt"][0][1]["start_percent"], 0.5)
            self.assertEqual(scope["input_images"], 0)
            self.assertIs(scope["controlnet"], control if mode else None)
            self.assertNotIn("reference_latents", positive[0][1])
            self.assertNotIn("reference_latents", negative[0][1])
            self.assertNotIn("reference_latents", switched[0][1])
        scope.update(
            input_image=None,
            positive_conditioning=positive,
            negative_conditioning=negative,
            switched_prompt=[],
            input_images=0,
        )
        exec(
            compile(
                ast.Module(body=[reference_mode(process), branch], type_ignores=[]),
                "text_only_route",
                "exec",
            ),
            scope,
        )
        self.assertIs(scope["positive_conditioning"], positive)
        self.assertIs(scope["negative_conditioning"], negative)
        self.assertEqual(scope["input_images"], 0)

    def test_full_reference_mode_preserves_geometry_and_explicit_img2img_keeps_source_latent(self):
        _, process = process_tree()
        preprocess = next(
            n
            for n in ast.walk(process)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "input_image"
        )
        image = Image.fromarray(np.arange(117 * 239 * 3, dtype=np.uint8).reshape(239, 117, 3))
        data = dict(width=2048, height=2048)
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            input_image=image,
            control_type="edit",
            controlnet={"type": "edit"},
            image_edit=False,
            ordered_recipe=None,
            pid_model=False,
            sd4x_model=False,
            refiner_model=False,
            instruct_model=False,
            size_plan=None,
            SizePlan=SizePlan,
            hidream_edit=True,
            architecture="HiDreamO1",
            gen_data=data,
            prepare_reference=prepare_reference,
            guidance={"image_resize_mode": "preserve"},
        )
        exec(
            compile(ast.Module(body=[preprocess], type_ignores=[]), "reference_preprocess", "exec"),
            scope,
        )
        self.assertEqual(scope["control_type"], "edit")
        self.assertEqual(tuple(scope["input_image"].shape), (1, 256, 128, 3))
        restored = restore_reference_frame(scope["input_image"], scope["reference_frame"])
        self.assertTrue(torch.equal(restored, torch.from_numpy(np.array(image)).unsqueeze(0) / 255))
        branch = next(
            n
            for n in process.body
            if isinstance(n, ast.If) and ast.unparse(n.test).startswith("edit_latent is not None")
        )
        source = {"samples": torch.zeros(1, 3, 256, 128, device="cpu")}
        encoder = Mock(return_value=[source])
        empty = Mock(return_value=[{"samples": torch.zeros(1, 3, 256, 128, device="cpu")}])
        owner = SimpleNamespace(
            xl_base_patched=SimpleNamespace(
                vae=SimpleNamespace(spacial_compression_decode=lambda: 1)
            ),
            model_info={"latent": "HiDreamO1"},
        )
        scope.update(
            self=owner,
            edit_latent=None,
            VAEEncode=lambda: SimpleNamespace(encode=encoder),
            EmptyHiDreamO1LatentImage=lambda: SimpleNamespace(execute=empty),
        )
        for mode in (False, True):
            scope.update(img2img_mode=mode, controlnet={"type": "img2img", "denoise": 0.25})
            exec(
                compile(
                    ast.Module(body=[branch], type_ignores=[]), "reference_initial_latent", "exec"
                ),
                scope,
            )
            self.assertEqual(scope["denoise"], 0.25 if mode else None)
            if mode:
                self.assertIs(scope["latent"], source)
                self.assertIs(encoder.call_args.kwargs["pixels"], scope["input_image"])
        self.assertEqual((empty.call_count, encoder.call_count), (1, 1))

    def test_normalization_and_ui_allow_native_o1_but_reject_invalid_refs(self):
        from ruinedfooocus.features.image_editing.ui_guidance import references_visible

        primary = Image.new("RGB", (8, 8))
        extra = Image.new("RGBA", (5, 7))
        self.assertTrue(capabilities("HiDream O1")["multiple_refs"])
        self.assertTrue(references_visible("HiDreamO1", "Disabled"))
        self.assertEqual(
            normalize_reference_images([(extra, "caption")], primary, "HiDream O1")[0].mode, "RGB"
        )
        for refs, image in (
            ([extra], None),
            (["C:/private.png"], primary),
            ([extra] * 16, primary),
        ):
            with self.assertRaises(ValueError):
                normalize_reference_images(refs, image, "HiDreamO1")


if __name__ == "__main__":
    unittest.main()
