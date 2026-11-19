"""INT8 checkpoint copies: layer choice, file layout, refusals and the pair benchmark.

A small synthetic checkpoint is quantized on the CPU with the installed kernel
library. These checks do not show that a real model loads, runs faster or keeps
its image quality; that was measured separately on a GPU.
"""

import json
from pathlib import Path
import tempfile
import unittest
import unittest.mock

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from ruinedfooocus.features.models import gpu_optimizer


class GpuOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.cache = self.root / "cache"
        # The kernel library asks whether a GPU exists before choosing its CPU path.
        self.enterContext(unittest.mock.patch.object(torch.cuda, "is_available", lambda: False))
        torch.manual_seed(7)
        self.tensors = {
            "model.diffusion_model.blocks.0.self_attn.q_proj.weight": torch.randn(1024, 1024)
            * 0.02,
            "model.diffusion_model.blocks.0.self_attn.q_proj.bias": torch.zeros(1024),
            "model.diffusion_model.blocks.0.mlp.layer1.weight": (
                torch.randn(2048, 1024) * 0.02
            ).half(),
            "model.diffusion_model.blocks.0.adaln_modulation.1.weight": torch.randn(1024, 1024),
            "model.diffusion_model.x_embedder.proj.1.weight": torch.randn(1024, 68),
            "model.diffusion_model.blocks.0.small.weight": torch.randn(256, 1024),
            "text_encoders.t5.blocks.0.attn.q.weight": torch.randn(1024, 1024),
        }
        self.source = self.root / "model.safetensors"
        save_file(self.tensors, self.source, metadata={"note": "kept"})
        self.chosen = [
            "model.diffusion_model.blocks.0.self_attn.q_proj.weight",
            "model.diffusion_model.blocks.0.mlp.layer1.weight",
        ]

    def test_only_large_transformer_linear_weights_are_chosen(self):
        planned = gpu_optimizer.plan(self.source)
        self.assertEqual(sorted(planned["layers"]), sorted(self.chosen))
        self.assertLess(planned["bytes_after"], planned["bytes_before"])
        for name, shape, wanted in (
            ("single_blocks.3.linear1.weight", [9216, 3072], True),
            ("content_mode_blocks.image.0.qkv.weight", [9216, 3072], True),
            ("layers.12.attention.to_q.weight", [3840, 3840], True),
            ("blocks.2.attn.qkv.weight", [1024, 3000], False),  # not a multiple of the group
            ("final_layer.linear.weight", [1024, 2048], False),
            ("blocks.1.norm1.linear.weight", [2048, 2048], False),
            ("first_stage_model.blocks.0.proj.weight", [1024, 1024], False),
        ):
            tensor = dict(dtype="F16", shape=shape)
            self.assertEqual(gpu_optimizer.eligible(name, tensor), wanted, name)
        self.assertFalse(
            gpu_optimizer.eligible("blocks.0.attn.q.weight", dict(dtype="I8", shape=[1024, 1024]))
        )

    def test_copy_stores_chosen_layers_as_int8_and_keeps_everything_else(self):
        progress = []
        result = gpu_optimizer.create_int8_copy(
            self.source,
            self.cache,
            {"baseModel": "Anima"},
            progress=lambda fraction, text: progress.append(fraction),
        )
        target = result["path"]
        self.assertEqual(target.name, "model.int8.safetensors")
        self.assertEqual(result["layers"], 2)
        self.assertIn("Original retained", result["message"])
        self.assertTrue(progress and max(progress) <= 1.0)
        with safe_open(target, "pt") as handle:
            keys = set(handle.keys())
            metadata = handle.metadata()
            for name in self.chosen:
                prefix = name[: -len("weight")]
                weight = handle.get_tensor(name)
                self.assertEqual(weight.dtype, torch.int8)
                self.assertEqual(tuple(weight.shape), tuple(self.tensors[name].shape))
                scale = handle.get_tensor(prefix + "weight_scale")
                self.assertEqual(
                    (scale.dtype, tuple(scale.shape)), (torch.float32, (weight.shape[0], 1))
                )
                record = json.loads(bytes(handle.get_tensor(prefix + "comfy_quant").tolist()))
                self.assertEqual(record, gpu_optimizer.RECORD)
                # The stored layer is the source layer within quantization error.
                from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

                params = TensorWiseINT8Layout.Params(
                    scale=scale,
                    orig_dtype=torch.float32,
                    orig_shape=tuple(weight.shape),
                    convrot=True,
                    convrot_groupsize=gpu_optimizer.GROUP_SIZE,
                )
                restored = TensorWiseINT8Layout.dequantize(weight, params).float()
                original = self.tensors[name].float()
                error = float((restored - original).abs().mean() / original.abs().mean())
                self.assertLess(error, 0.05, name)
            for name, tensor in self.tensors.items():
                if name not in self.chosen:
                    self.assertTrue(torch.equal(handle.get_tensor(name), tensor), name)
        self.assertEqual(len(keys), len(self.tensors) + 4)
        self.assertEqual(metadata["note"], "kept")
        self.assertEqual(metadata["rf.source_name"], "model.safetensors")
        self.assertIn("INT8 ConvRot", metadata["rf.conversion"])
        info = json.loads((self.cache / "model.int8.json").read_text(encoding="utf-8"))
        self.assertEqual(info["rf_source"]["name"], "model.safetensors")
        self.assertEqual(info["baseModel"], "Anima")
        # The source is untouched and nothing temporary is left beside it.
        with safe_open(self.source, "pt") as handle:
            self.assertEqual(set(handle.keys()), set(self.tensors))
        self.assertEqual(
            sorted(path.name for path in self.root.iterdir()),
            ["cache", "model.int8.safetensors", "model.safetensors"],
        )
        with self.assertRaisesRegex(FileExistsError, "Already exists"):
            gpu_optimizer.create_int8_copy(self.source, self.cache)

    def test_quantized_unet_and_plain_checkpoints_are_refused_before_writing(self):
        copy = gpu_optimizer.create_int8_copy(self.source, self.cache)["path"]
        with self.assertRaisesRegex(ValueError, "already quantized"):
            gpu_optimizer.plan(copy)
        unet = self.root / "unet.safetensors"
        save_file(
            {
                "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight": torch.randn(
                    1280, 1280
                )
            },
            unet,
        )
        with self.assertRaisesRegex(ValueError, "UNet checkpoint"):
            gpu_optimizer.plan(unet)
        plain = self.root / "plain.safetensors"
        save_file({"embedding.weight": torch.randn(8, 8)}, plain)
        with self.assertRaisesRegex(ValueError, "No transformer linear layers"):
            gpu_optimizer.create_int8_copy(plain, self.cache)
        self.assertFalse((self.root / "plain.int8.safetensors").exists())

    def test_pair_is_found_from_either_file_and_the_last_run_is_compared(self):
        names = ["a.safetensors", "a.int8.safetensors", "sub/b.safetensors", "c.int8.safetensors"]
        self.assertEqual(
            gpu_optimizer.pair("a.safetensors", names), ("a.safetensors", "a.int8.safetensors")
        )
        self.assertEqual(
            gpu_optimizer.pair("a.int8.safetensors", names),
            ("a.safetensors", "a.int8.safetensors"),
        )
        self.assertIsNone(gpu_optimizer.pair("sub/b.safetensors", names))
        self.assertIsNone(gpu_optimizer.pair("c.int8.safetensors", names))
        calls = []
        clock = iter([0, 40, 40, 70, 70, 100, 100, 127])

        def generate(name):
            calls.append(name)

        with unittest.mock.patch.object(gpu_optimizer.time, "perf_counter", lambda: next(clock)):
            result = gpu_optimizer.benchmark_pair(
                "a.safetensors", "a.int8.safetensors", generate=generate
            )
        self.assertEqual(calls, ["a.safetensors"] * 2 + ["a.int8.safetensors"] * 2)
        self.assertEqual(result["seconds"], {"a.safetensors": 30, "a.int8.safetensors": 27})
        self.assertIn("10% faster", result["message"])


if __name__ == "__main__":
    unittest.main()
