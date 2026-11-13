from ruinedfooocus.core.paths import asset_path, resource_path

"""Keep native and override shift units distinct when importing image metadata."""
import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from ruinedfooocus.features.image_editing.image_guidance import (
    effective_guidance,
    request_model,
    resolve_guidance,
    restore_guidance_metadata,
)

ROOT = Path(__file__).resolve().parents[1]


def native_samplers():
    path = (
        Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))
        / "comfy/model_sampling.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"ModelSamplingFlux", "ModelSamplingDiscreteFlow", "time_snr_shift", "flux_time_shift"}
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    scope = dict(
        asset_path=asset_path,
        resource_path=resource_path,
        torch=torch,
        math=math,
        __name__="comfy.model_sampling",
    )
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope


class Patcher:
    def __init__(self, sampling):
        self.sampling = sampling

    def get_model_object(self, name):
        return self.sampling

    def clone(self):
        return Patcher(self.sampling)


class ImageSamplingReplayTests(unittest.TestCase):
    def setUp(self):
        self.native = native_samplers()

    def metadata(self, kind, shift, requested=None):
        sampler = self.native[kind](
            SimpleNamespace(sampling_settings={"shift": shift, "multiplier": 1})
        )
        base = Patcher(sampler)
        values = resolve_guidance({"sampling_shift": requested}, {}, "QwenImage")
        return base, effective_guidance(values, base, "QwenImage")

    def test_native_qwen_auto_replay_keeps_the_exact_native_sigma_table(self):
        base, metadata = self.metadata("ModelSamplingFlux", 1.15)
        self.assertEqual(metadata.get("effective_sampling_shift_mode"), "flux")
        self.assertEqual(metadata["effective_sampling_shift"], 1.15)
        for payload in (metadata, {"image_guidance": metadata}):
            restored = restore_guidance_metadata(payload, family="QwenImage")
            self.assertIsNone(restored["sampling_shift"])
            values = resolve_guidance(restored, {}, "QwenImage")
            patch = Mock(side_effect=AssertionError("Native Auto must not apply an AuraFlow shift"))
            replay = request_model(base, values, "QwenImage", nodes={"AuraFlow": patch})
            patch.assert_not_called()
            self.assertIs(replay.sampling, base.sampling)
            self.assertTrue(torch.equal(replay.sampling.sigmas, base.sampling.sigmas))
        # The same raw number is a different schedule in the override node.
        wrong = self.native["ModelSamplingDiscreteFlow"](
            SimpleNamespace(sampling_settings={"shift": 1.15, "multiplier": 1})
        )
        point = torch.tensor(0.5)
        self.assertGreater(float(base.sampling.sigma(point) - wrong.sigma(point)), 0.2)

    def test_typed_flow_shift_still_replays_its_recorded_value(self):
        base, metadata = self.metadata("ModelSamplingDiscreteFlow", 3.1)
        self.assertEqual(metadata.get("effective_sampling_shift_mode"), "flow")
        for payload in (metadata, {"image_guidance": metadata}):
            restored = restore_guidance_metadata(payload, family="QwenImage")
            self.assertEqual(restored["sampling_shift"], 3.1)
            actual = self.native["ModelSamplingDiscreteFlow"](
                SimpleNamespace(
                    sampling_settings={"shift": restored["sampling_shift"], "multiplier": 1}
                )
            )
            self.assertTrue(torch.equal(actual.sigmas, base.sampling.sigmas))

    def test_legacy_qwen_auto_does_not_guess_units_but_explicit_overrides_win(self):
        for family in ("QwenImage", "Qwen-Image"):
            for value in (1.15, 3.1):
                metadata = dict(sampling_shift=None, effective_sampling_shift=value)
                self.assertIsNone(
                    restore_guidance_metadata(metadata, family=family)["sampling_shift"]
                )
                metadata["sampling_shift"] = 2.2
                self.assertEqual(
                    restore_guidance_metadata(metadata, family=family)["sampling_shift"], 2.2
                )
        for mode in ("flux", "unknown", False, {}):
            self.assertIsNone(
                restore_guidance_metadata(
                    dict(
                        sampling_shift=None,
                        effective_sampling_shift=1.15,
                        effective_sampling_shift_mode=mode,
                    ),
                    family="QwenImage",
                )["sampling_shift"]
            )
        self.assertEqual(
            restore_guidance_metadata(
                dict(sampling_shift=None, effective_sampling_shift=5), family="MOSSImage"
            )["sampling_shift"],
            5,
        )

    def test_qwen_png_preset_keeps_auto_in_the_visible_shift_control(self):
        from ruinedfooocus.features.image_editing.ui_guidance import preset_control_state

        _, metadata = self.metadata("ModelSamplingFlux", 1.15)
        result = preset_control_state(
            {},
            {},
            dict(software="RuinedFooocus", image_guidance=metadata),
            7,
            "QwenImage",
            "qwen",
            "qwen",
            None,
        )
        self.assertTrue(result["locked"])
        self.assertIsNone(result["values"]["sampling_shift"])


if __name__ == "__main__":
    unittest.main()
