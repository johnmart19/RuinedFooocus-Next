"""Image-refiner headers defer to native classification before video filtering."""

import ast
import logging
import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from ruinedfooocus.features.models.model_architecture import detect_tensor_architecture
from ruinedfooocus.features.models.image_scope import unsupported_tensor_family
from test_native_model_coverage import NATIVE


class NativeRefinerDiscoveryTests(unittest.TestCase):
    def native(self, header):
        tree = ast.parse((NATIVE / "comfy/model_detection.py").read_text(encoding="utf-8"))
        scope = dict(torch=torch, math=math, logging=logging)
        for name in (
            "count_blocks",
            "any_suffix_in",
            "detect_unet_config",
            "model_config_from_unet_config",
            "model_config_from_unet",
        ):
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(NATIVE), "exec"), scope)
        base = ast.parse((NATIVE / "comfy/supported_models_base.py").read_text(encoding="utf-8"))
        match = next(
            n for n in ast.walk(base) if isinstance(n, ast.FunctionDef) and n.name == "matches"
        )
        exec(compile(ast.Module(body=[match], type_ignores=[]), str(NATIVE), "exec"), scope)
        supported = ast.parse((NATIVE / "comfy/supported_models.py").read_text(encoding="utf-8"))
        names = ("HunyuanImage21Refiner", "HunyuanImage21", "HunyuanVideoI2V", "HunyuanVideo")
        classes = []
        for name in names:
            node = next(n for n in supported.body if isinstance(n, ast.ClassDef) and n.name == name)
            config = next(
                n.value
                for n in node.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "unet_config" for t in n.targets)
            )
            classes.append(
                type(
                    name,
                    (),
                    dict(
                        unet_config=ast.literal_eval(config),
                        required_keys={},
                        matches=scope["matches"],
                        __init__=lambda self, config: setattr(self, "unet_config", config),
                    ),
                )
            )
        scope["comfy"] = NS(
            supported_models=NS(models=classes),
            utils=NS(detect_layer_quantization=lambda *args: {}),
        )
        detection = NS(
            unet_prefix_from_state_dict=lambda _: "",
            model_config_from_unet=scope["model_config_from_unet"],
        )
        with patch.dict("sys.modules", {"comfy": NS(model_detection=detection)}):
            return detect_tensor_architecture(header)

    def header(self, channels, patch, vector=False):
        shapes = {
            "txt_in.individual_token_refiner.blocks.0.norm1.weight": [3072],
            "img_in.proj.weight": [3072, channels, *patch],
            "final_layer.linear.weight": [64 * math.prod(patch), 3072],
            "txt_in.input_embedder.weight": [3072, 3584],
        }
        if vector:
            shapes["vector_in.in_layer.weight"] = [3072, 768]
        return {key: {"shape": value} for key, value in shapes.items()}

    def test_native_refiner32_and128_input_channels_are_not_filtered_as_video(self):
        for channels in (32, 128):
            header = self.header(channels, [1, 1, 1])
            self.assertIsNone(unsupported_tensor_family(header))
            self.assertEqual(self.native(header), "HunyuanImage21Refiner")

    def test_actual_video_discriminators_still_exclude_skyreels_and_vector_conditioning(self):
        for header in (
            self.header(32, [1, 2, 2]),
            self.header(33, [1, 2, 2]),
            self.header(32, [1, 1, 1], vector=True),
        ):
            self.assertEqual(unsupported_tensor_family(header), "Hunyuan Video")


if __name__ == "__main__":
    unittest.main()
