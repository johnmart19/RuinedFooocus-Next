from ruinedfooocus.core.paths import asset_path, resource_path

"""Exercise pinned native guidance callbacks on CPU, without loading models."""
import ast
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from ruinedfooocus.features.image_editing.image_guidance import (
    request_model,
    resolve_guidance,
    restore_guidance_metadata,
    validate_guidance,
)

ROOT = Path(__file__).resolve().parents[1]


class Patcher:
    def __init__(self):
        self.cfg = self.post_cfg = self.cond_batch = None

    def clone(self):
        other = Patcher()
        other.__dict__.update(self.__dict__)
        return other

    def get_model_object(self, name):
        return SimpleNamespace(percent_to_sigma=lambda fraction: 1 - fraction)

    def set_model_sampler_cfg_function(self, fn):
        self.cfg = fn

    def set_model_sampler_post_cfg_function(self, fn):
        self.post_cfg = fn

    def set_model_sampler_calc_cond_batch_function(self, fn):
        self.cond_batch = fn


def native_execute(filename, cls, **scope):
    tree = ast.parse(
        (ROOT / "repositories/ComfyUI/comfy_extras" / filename).read_text(encoding="utf-8")
    )
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    function = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
    function.decorator_list = []
    namespace = dict(
        asset_path=asset_path,
        resource_path=resource_path,
        torch=torch,
        re=re,
        io=SimpleNamespace(NodeOutput=lambda *values: values),
        **scope,
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
    return lambda **kwargs: namespace["execute"](None, **kwargs)


class NativeGuidanceOptionsTests(unittest.TestCase):
    def test_pre_and_post_cfg_follow_native_callbacks_and_leave_the_cache_unmodified(self):
        native = native_execute("nodes_cfg.py", "CFGNorm")
        base = Patcher()
        cond = torch.tensor([3.0, 4.0], device="cpu").reshape(1, 2, 1, 1)
        uncond = torch.tensor([2.0, 3.0], device="cpu").reshape_as(cond)
        combined = uncond + 0.1 * (cond - uncond)
        before = request_model(
            base,
            resolve_guidance({"cfg_norm_strength": 1, "cfg_norm_pre_cfg": True}, {}, "SDXL"),
            "SDXL",
            {"cfg_norm": native},
        )
        after = request_model(
            base,
            resolve_guidance({"cfg_norm_strength": 1}, {}, "SDXL"),
            "SDXL",
            {"cfg_norm": native},
        )
        pre_result = before.cfg(dict(cond=cond, uncond=uncond, cond_scale=0.1))
        post_result = after.post_cfg(dict(cond_denoised=cond, denoised=combined))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(pre_result), torch.tensor(5.0)))
        self.assertTrue(torch.allclose(post_result, combined))
        self.assertGreater(torch.linalg.vector_norm(pre_result), torch.linalg.vector_norm(combined))
        self.assertIsNone(base.cfg)
        self.assertIsNone(base.post_cfg)
        self.assertIsNone(before.post_cfg)
        self.assertIsNone(after.cfg)
        unused = request_model(
            base,
            resolve_guidance({"cfg_norm_pre_cfg": True}, {}, "SDXL"),
            "SDXL",
            {"cfg_norm": native},
        )
        self.assertIsNone(unused.cfg)

    def test_native_slg_uses_selected_blocks_only_within_the_sampling_range(self):
        batch = Mock(return_value=("positive", "negative"))
        replace = Mock(
            side_effect=lambda options, fn, kind, block, index: {
                **options,
                "blocks": options.get("blocks", ()) + ((block, index),),
            }
        )
        native = native_execute(
            "nodes_slg.py",
            "SkipLayerGuidanceDiTSimple",
            comfy=SimpleNamespace(
                samplers=SimpleNamespace(calc_cond_batch=batch),
                model_patcher=SimpleNamespace(set_model_options_patch_replace=replace),
            ),
        )
        base = Patcher()
        values = resolve_guidance(
            dict(
                moss_image_slg=True,
                moss_image_slg_double_layers="1, 3",
                moss_image_slg_single_layers="8, 9",
                moss_image_slg_start=0.2,
                moss_image_slg_end=0.8,
            ),
            {},
            "MOSSImage",
        )
        model = request_model(base, values, "MOSSImage", {"slg": native})
        options = {"keep": "existing"}
        args = dict(
            input=torch.zeros(1, device="cpu"),
            model=object(),
            conds=["+", "-"],
            sigma=torch.tensor([0.5], device="cpu"),
            model_options=options,
        )
        self.assertEqual(model.cond_batch(args), ["positive", "negative"])
        self.assertEqual(batch.call_count, 2)
        self.assertEqual(batch.call_args_list[0].args[-1], options)
        self.assertEqual(
            batch.call_args_list[1].args[-1]["blocks"],
            (("double_block", 1), ("double_block", 3), ("single_block", 8), ("single_block", 9)),
        )
        self.assertEqual(options, {"keep": "existing"})
        self.assertIsNone(base.cond_batch)
        batch.reset_mock()
        model.cond_batch({**args, "sigma": torch.tensor([0.95], device="cpu")})
        batch.assert_called_once()
        self.assertEqual(batch.call_args.args[-1], options)

    def test_invalid_layer_lists_ranges_and_boolean_modes_are_rejected(self):
        for payload in (
            {"moss_image_slg_single_layers": "-1"},
            {"moss_image_slg_double_layers": "1.5"},
            {"moss_image_slg_single_layers": "word"},
            {"moss_image_slg_single_layers": "1,"},
            {"moss_image_slg_single_layers": "1" * 1025},
            {"moss_image_slg_start": True},
            {"moss_image_slg_start": 0.7, "moss_image_slg_end": 0.3},
            {"moss_image_slg_end": float("nan")},
            {"cfg_norm_pre_cfg": "false"},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_guidance(payload)
        values = resolve_guidance(
            {"moss_image_slg": True, "moss_image_slg_single_layers": "7, 8"}, {}, "SDXL"
        )
        self.assertFalse(values["moss_image_slg"])
        self.assertEqual(values["moss_image_slg_single_layers"], "22")
        self.assertEqual(
            validate_guidance({"moss_image_slg_double_layers": "  "}),
            {"moss_image_slg_double_layers": ""},
        )
        from ruinedfooocus.integrations.openai_api import ImageRequest
        from pydantic import ValidationError

        for values in (
            {"moss_image_slg_start": -0.1},
            {"moss_image_slg_end": 1.1},
            {"moss_image_slg_single_layers": "1" * 1025},
        ):
            with self.assertRaises(ValidationError):
                ImageRequest(**values)

    def test_invalid_recorded_slg_range_does_not_break_metadata_replay(self):
        for interval in (
            {"moss_image_slg_start": 0.7, "moss_image_slg_end": 0.3},
            {"moss_image_slg_start": 0.8},
        ):
            restored = restore_guidance_metadata(
                {"image_guidance": {**interval, "cfg_norm_pre_cfg": True}}
            )
            self.assertEqual(restored, {"cfg_norm_pre_cfg": True})
            resolve_guidance(restored, {}, "MOSSImage")


if __name__ == "__main__":
    unittest.main()
