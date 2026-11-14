"""JoyImage uses the pinned paired reference encodes and separate sampler latent."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch

from ruinedfooocus.features.image_editing.image_geometry import prepare_reference
from ruinedfooocus.features.image_editing.image_guidance import normalize_reference_images
from ruinedfooocus.features.models.image_components import edit_control
from test_native_model_coverage import NATIVE, route_info
from test_qwen_image21_conditioning import clip_fixture, native_class, native_function
from ui_test_helpers import function_from


PIPELINE = "ruinedfooocus/features/generation/image_pipeline.py"


def runtime():
    comfy = SimpleNamespace(
        utils=SimpleNamespace(
            common_upscale=native_function("comfy/utils.py", "common_upscale", {"torch": torch})
        ),
        model_management=SimpleNamespace(
            intermediate_device=lambda: "cpu", intermediate_dtype=lambda: torch.float32
        ),
    )
    output = native_class(
        "comfy_api/latest/_io.py",
        "NodeOutput",
        dict(Any=object, _UIOutput=object),
        methods=("__init__", "__getitem__"),
    )
    scope = dict(
        torch=torch,
        comfy=comfy,
        io=SimpleNamespace(NodeOutput=output),
        node_helpers=SimpleNamespace(
            conditioning_set_values=native_function(
                "node_helpers.py", "conditioning_set_values", {}
            )
        ),
    )
    owner = "comfy_extras/nodes_joyimage.py"
    tree = ast.parse((NATIVE / owner).read_text(encoding="utf-8"))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign))
    scope["BUCKETS_1024"] = ast.literal_eval(assignment.value)
    for name in ("_find_best_bucket", "_resize_reference", "_encode"):
        native_function(owner, name, scope)
    joy = native_class(owner, "TextEncodeJoyImageEdit", scope, methods=("execute",))
    empty = native_class(
        "comfy_extras/nodes_sd3.py", "EmptySD3LatentImage", scope, methods=("execute",)
    )
    fix_latent = native_function("comfy/sample.py", "fix_empty_latent_channels", scope)
    return joy, empty, fix_latent


class JoyImageReferenceTests(unittest.TestCase):
    def test_reference_branches_share_source_and_keep_a_separate_empty_wan_grid(self):
        joy, empty, fix_latent = runtime()
        encode = function_from(PIPELINE, "joy_reference_conditioning", {})
        clip, mask = clip_fixture()
        pixels, frame = prepare_reference(Image.new("RGB", (96, 64), "red"), (96, 64))
        vae = SimpleNamespace(
            encode=Mock(
                side_effect=lambda image: torch.ones(
                    (1, 16, 1, image.shape[1] // 8, image.shape[2] // 8)
                )
            )
        )
        with patch.dict(
            sys.modules,
            {"comfy_extras.nodes_joyimage": SimpleNamespace(TextEncodeJoyImageEdit=joy)},
        ):
            with self.assertRaisesRegex(ValueError, "Wan 2.1 VAE"):
                encode(clip, None, pixels, "change the color", "")
            positive, negative = encode(clip, vae, pixels, "change the color", "")

        self.assertEqual(
            [call.args[0] for call in clip.tokenize.call_args_list], ["change the color", ""]
        )
        self.assertEqual(vae.encode.call_count, 2)
        for result, token, encoded in zip(
            (positive, negative), clip.tokenize.call_args_list, vae.encode.call_args_list
        ):
            source = token.kwargs["images"][0]
            self.assertIs(source, encoded.args[0])
            self.assertEqual(tuple(source.shape), (1, 832, 1216, 3))
            self.assertIs(result[0][1]["attention_mask"], mask)
            self.assertEqual(
                tuple(result[0][1]["reference_latents"][0].shape), (1, 16, 1, 104, 152)
            )

        self.assertEqual(route_info()["JoyImage"]["latent"], "SD3")
        latent = empty.execute(frame["canvas_width"], frame["canvas_height"])[0]
        wan = SimpleNamespace(
            latent_channels=16,
            latent_dimensions=3,
            spacial_downscale_ratio=8,
            fix_empty_latent=lambda value: value,
        )
        native_model = SimpleNamespace(get_model_object=lambda key: wan)
        actual = fix_latent(native_model, latent["samples"], latent["downscale_ratio_spacial"])
        self.assertEqual(tuple(actual.shape), (1, 16, 1, 8, 12))
        self.assertEqual(torch.count_nonzero(actual).item(), 0)

    def test_same_prompt_with_new_source_keeps_reference_latents_out_of_pipeline_cache(self):
        source = Path(__file__).resolve().parents[1] / PIPELINE
        tree = ast.parse(source.read_text(encoding="utf-8"))
        process = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "process"
        )
        # Some of these tests are elif branches, so collect every if in process.
        branches = [node for node in ast.walk(process) if isinstance(node, ast.If)]
        encode_branch = next(
            node for node in branches if ast.unparse(node.test) == "image_edit and input_images > 0"
        )
        handoff = next(
            node
            for node in branches
            if isinstance(node.body[0], ast.Assign)
            and ast.unparse(node.body[0].value) == "self.conditions['+']['cache']"
        )
        code = compile(
            ast.Module(body=[encode_branch, handoff], type_ignores=[]), str(source), "exec"
        )
        joy, _, _ = runtime()
        encode = function_from(PIPELINE, "joy_reference_conditioning", {})
        clip, _ = clip_fixture()
        vae = SimpleNamespace(encode=Mock(side_effect=lambda image: image[0, 0, 0].clone()))
        cached_text = object()
        cache = {"+": {"text": "same prompt", "cache": cached_text}, "-": {"cache": None}}
        owner = SimpleNamespace(
            conditions=cache,
            xl_base_patched=SimpleNamespace(
                clip=clip, vae=vae, unet=SimpleNamespace(model=type("JoyImage", (), {})())
            ),
        )
        warm = Mock()
        with patch.dict(
            sys.modules,
            {"comfy_extras.nodes_joyimage": SimpleNamespace(TextEncodeJoyImageEdit=joy)},
        ):
            for color, expected in (("red", [1.0, 0.0, 0.0]), ("blue", [0.0, 0.0, 1.0])):
                pixels, _ = prepare_reference(Image.new("RGB", (96, 64), color), (96, 64))
                scope = dict(
                    self=owner,
                    architecture="JoyImage",
                    joy_reference_edit=True,
                    image_edit=True,
                    input_images=1,
                    img2img_mode=False,
                    callback=None,
                    input_image=pixels,
                    positive_prompt="same prompt",
                    negative_prompt="",
                    clean_prompt_cond_caches=lambda: {"+": {}, "-": {}},
                    warm_patched_text_encoder=warm,
                    joy_reference_conditioning=encode,
                )
                exec(code, scope)
                self.assertIs(owner.conditions, cache)
                self.assertIs(owner.conditions["+"]["cache"], cached_text)
                self.assertIsNone(owner.conditions["-"]["cache"])
                for name in ("positive_conditioning", "negative_conditioning"):
                    latent = scope[name][0][1]["reference_latents"][0]
                    self.assertEqual(latent.tolist(), expected)
        self.assertEqual(vae.encode.call_count, 4)

    def test_edit_admits_one_joy_source_without_advertising_longcat_edit(self):
        self.assertEqual(edit_control("JoyImage"), {"type": "edit"})
        self.assertEqual(edit_control("joyimage"), {"type": "edit"})
        # LongCat Image edits natively through its LongCat Image Edit recipe.
        self.assertEqual(edit_control("LongCatImage")["type"], "edit")
        primary = Image.new("RGB", (32, 32))
        with self.assertRaisesRegex(ValueError, "Additional references require"):
            normalize_reference_images([primary], primary, "JoyImage")


if __name__ == "__main__":
    unittest.main()
