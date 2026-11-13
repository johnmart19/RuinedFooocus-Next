from ruinedfooocus.core.paths import asset_path, resource_path

"""Native image decoding contracts with simulated VRAM, never GPU execution."""
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from ruinedfooocus.features.image_editing.image_guidance import resolve_guidance, validate_guidance
from ruinedfooocus.backends.comfy.vae_decode import decode_image_vae

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def native_method(filename, cls, name):
    tree = ast.parse((ROOT / "repositories/ComfyUI" / filename).read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
    function = next(
        node for node in node.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    namespace = {
        "asset_path": asset_path,
        "resource_path": resource_path,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


class VaeDecodeTests(unittest.TestCase):
    def setup_decoder(self, device="cuda", capacity=24):
        full, tiles = Mock(return_value=["pixels"]), Mock(return_value=["pixels"])
        memory = Mock()
        memory.get_total_memory.return_value = capacity * GIB
        memory.extra_reserved_memory.return_value = GIB
        vae = SimpleNamespace(
            device=SimpleNamespace(type=device),
            vae_dtype=torch.float16,
            patcher=SimpleNamespace(model_size=lambda: GIB),
            extra_1d_channel=None,
            spacial_compression_decode=lambda: 8,
            memory_used_decode=lambda shape, dtype: shape[-2] * shape[-1] * 4 * 1024**2,
        )
        vae._tile_bounded_shape = MethodType(
            native_method("comfy/sd.py", "VAE", "_tile_bounded_shape"), vae
        )
        models = SimpleNamespace(vae=vae, unet=object())
        context = patch.dict(
            "sys.modules",
            {
                "comfy": SimpleNamespace(model_management=memory),
                "nodes": SimpleNamespace(
                    VAEDecode=lambda: SimpleNamespace(decode=full),
                    VAEDecodeTiled=lambda: SimpleNamespace(decode=tiles),
                ),
            },
        )
        context.start()
        self.addCleanup(context.stop)
        return models, full, tiles, memory

    def test_cpu_keeps_native_decode_and_explicit_tiling_is_available(self):
        models, full, tiles, memory = self.setup_decoder("cpu")
        samples = {"samples": torch.zeros(1, 4, 16, 16)}
        self.assertEqual(decode_image_vae(models, samples), "pixels")
        full.assert_called_once_with(vae=models.vae, samples=samples)
        tiles.assert_not_called()
        memory.get_total_memory.assert_not_called()
        decode_image_vae(models, samples, tiled=True, tile_size=256, overlap=0)
        tiles.assert_called_once_with(vae=models.vae, samples=samples, tile_size=256, overlap=0)

    def test_auto_fits_2d_and_single_frame_image_latents_across_vram_capacities(self):
        for shape in ((1, 16, 192, 128), (1, 16, 1, 192, 128)):
            sizes = []
            for capacity in (24, 16, 8, 4, 3):
                with self.subTest(shape=shape, capacity=capacity):
                    models, full, tiles, _ = self.setup_decoder(capacity=capacity)
                    decode_image_vae(models, {"samples": torch.zeros(shape)})
                    full.assert_not_called()
                    size = tiles.call_args.kwargs["tile_size"]
                    sizes.append(size)
                    bounded = models.vae._tile_bounded_shape(shape, size // 8, size // 8, 1)
                    self.assertLessEqual(
                        models.vae.memory_used_decode(bounded, torch.float16), (capacity - 2) * GIB
                    )
            self.assertEqual(sizes, [512, 256, 256, 128, 128])

    def test_a_small_gpu_image_keeps_native_full_decode(self):
        models, full, tiles, _ = self.setup_decoder()
        decode_image_vae(models, {"samples": torch.zeros(1, 16, 16, 16)})
        full.assert_called_once()
        tiles.assert_not_called()

    def test_fatal_device_failure_is_not_retried_as_an_oom(self):
        models, full, tiles, memory = self.setup_decoder("cpu")
        full.side_effect = RuntimeError("CUDA error: unspecified launch failure")

        def raise_error(error):
            raise error

        memory.raise_non_oom.side_effect = raise_error
        with self.assertRaisesRegex(RuntimeError, "unspecified launch failure"):
            decode_image_vae(models, {"samples": torch.zeros(1, 4, 8, 8)})
        self.assertEqual(full.call_count, 1)
        tiles.assert_not_called()
        memory.unload_model_and_clones.assert_not_called()

    def test_tile_overlap_is_capped_by_the_native_node(self):
        decode = native_method("nodes.py", "VAEDecodeTiled", "decode")
        vae = SimpleNamespace(
            temporal_compression_decode=lambda: None,
            spacial_compression_decode=lambda: 8,
            decode_tiled=Mock(return_value=torch.zeros(1, 64, 64, 3)),
        )
        samples = {"samples": torch.zeros(1, 4, 8, 8)}
        decode(object(), vae, samples, tile_size=64, overlap=64)
        self.assertEqual(vae.decode_tiled.call_args.kwargs["overlap"], 2)

    def test_request_defaults_validation_and_explicit_native_parameters(self):
        from ruinedfooocus.integrations.openai_api import ImageRequest
        from pydantic import ValidationError

        defaults = resolve_guidance({}, {}, "SDXL")
        self.assertEqual(
            {key: defaults[key] for key in ("vae_tiled", "vae_tile_size", "vae_overlap")},
            dict(vae_tiled=False, vae_tile_size=None, vae_overlap=None),
        )
        request = ImageRequest(vae_tiled=True, vae_tile_size=256, vae_overlap=0)
        self.assertEqual(
            validate_guidance(
                request.model_dump(include={"vae_tiled", "vae_tile_size", "vae_overlap"})
            ),
            dict(vae_tiled=True, vae_tile_size=256, vae_overlap=0),
        )
        self.assertEqual(
            validate_guidance({"vae_tile_size": "", "vae_overlap": None}),
            dict(vae_tile_size=None, vae_overlap=None),
        )
        for values in (
            {"vae_tiled": "true"},
            {"vae_tile_size": 63},
            {"vae_tile_size": 100},
            {"vae_overlap": float("nan")},
            {"vae_overlap": -32},
            {"vae_overlap": True},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_guidance(values)
        for size in (0, 63, 100, 4128):
            with self.assertRaises(ValidationError):
                ImageRequest(vae_tile_size=size)


if __name__ == "__main__":
    unittest.main()
