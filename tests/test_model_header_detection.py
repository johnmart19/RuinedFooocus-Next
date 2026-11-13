"""Native detection owns Diffusers conversion; only shape metadata is materialized."""

import json
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.features.models.model_architecture import detect_tensor_architecture
from ruinedfooocus.features.models.image_scope import unsupported_tensor_family
from ruinedfooocus.features.models.model_identity import file_signature

ROOT = Path(__file__).resolve().parents[1]


class ModelHeaderDetectionTests(unittest.TestCase):
    def native(self, detection, header):
        with patch.dict(
            "sys.modules",
            {
                "torch": SimpleNamespace(empty=lambda shape, device: SimpleNamespace(shape=shape)),
                "comfy": SimpleNamespace(model_detection=detection),
            },
        ):
            return detect_tensor_architecture(header)

    def test_conversion_uses_native_mapping_and_preserves_input_and_metadata(self):
        metadata = {"config": '{"transformer":{"pdd_steps":4}}'}
        header = {"diffusers.weight": {"shape": [2, 3]}, "__metadata__": metadata}
        original = json.dumps(header, sort_keys=True)
        supported = type("Flux2", (), {"unet_config": {"depth_single_blocks": 20}})()

        def convert(values):
            return {"converted.weight": values.pop("diffusers.weight")}

        direct = Mock(
            side_effect=lambda values, prefix, metadata=None: (
                supported if "converted.weight" in values else None
            )
        )
        detection = SimpleNamespace(
            unet_prefix_from_state_dict=lambda _: "",
            model_config_from_unet=direct,
            convert_diffusers_mmdit=Mock(side_effect=convert),
            model_config_from_diffusers_unet=Mock(),
        )
        self.assertEqual(self.native(detection, header), "Flux2Klein4B")
        self.assertEqual(json.dumps(header, sort_keys=True), original)
        self.assertIs(direct.call_args.kwargs["metadata"], metadata)
        detection.convert_diffusers_mmdit.assert_called_once()
        detection.model_config_from_diffusers_unet.assert_not_called()

    def test_diffusers_unet_fallback_and_unknown_headers_use_native_refusal(self):
        fallback = Mock(return_value=type("SDXL", (), {"unet_config": {}})())
        detection = SimpleNamespace(
            unet_prefix_from_state_dict=lambda _: "model.diffusion_model.",
            model_config_from_unet=Mock(return_value=None),
            convert_diffusers_mmdit=Mock(return_value=None),
            model_config_from_diffusers_unet=fallback,
        )
        self.assertEqual(
            self.native(
                detection,
                {
                    "model.diffusion_model.conv_in.weight": {"shape": [320, 4, 3, 3]},
                },
            ),
            "SDXL 1.0",
        )
        self.assertIn("conv_in.weight", fallback.call_args.args[0])
        fallback.return_value = None
        self.assertIsNone(self.native(detection, {"unrecognized": {"shape": [1]}}))
        self.assertIsNone(self.native(detection, {"bad": {"shape": [True]}}))

    def test_passive_shared_hunyuan_marker_needs_positive_video_evidence(self):
        for prefix in ("", "model.diffusion_model.", "diffusion_model."):
            marker = {
                prefix + "txt_in.individual_token_refiner.blocks.0.norm1.weight": {"shape": [3072]}
            }
            for image_projection in ([3072, 64, 2, 2], [3072, 64, 1, 1, 1]):
                image = {**marker, prefix + "img_in.proj.weight": {"shape": image_projection}}
                self.assertIsNone(unsupported_tensor_family(image))
            self.assertIsNone(unsupported_tensor_family(marker))
            for discriminator in (
                {"vector_in.in_layer.weight": {"shape": [3072, 768]}},
                {"img_in.proj.weight": {"shape": [3072, 32, 1, 2, 2]}},
                {"img_in.proj.weight": {"shape": [3072, 33, 1, 2, 2]}},
                {"vision_in.proj.0.weight": {"shape": [1152, 1152]}},
            ):
                self.assertEqual(
                    unsupported_tensor_family(
                        {
                            **marker,
                            **{prefix + name: entry for name, entry in discriminator.items()},
                        }
                    ),
                    "Hunyuan Video",
                )

    def test_authoritative_hunyuan_cache_correction_preserves_provenance_and_file_identity(self):
        source = ROOT / "ruinedfooocus/features/models/model_handler.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "get_models_by_path"
        )
        scope = {"Path": Path, "json": json, "file_signature": file_signature}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "model.safetensors"
            path.write_bytes(b"source identity fixture")
            cache = root / "cache"
            cache.mkdir()
            metadata = {
                "baseModel": "Hunyuan Video",
                "name": "Publisher title",
                "modelId": 42,
                "rf_source": "original provenance",
                "_rf_file_signature": file_signature(path),
            }
            cached = cache / "model.json"
            owner = SimpleNamespace(
                cache_paths={"checkpoints": cache}, offline=True, detect_checkpoint_base=Mock()
            )
            for detected in ("HunyuanImage21", "HunyuanImage21Refiner", None):
                cached.write_text(json.dumps(metadata), encoding="utf-8")
                before = cached.read_bytes()
                owner.detect_checkpoint_base.return_value = detected
                result = scope["get_models_by_path"](owner, "checkpoints", path)
                self.assertEqual(result, {**metadata, "baseModel": detected or "Hunyuan Video"})
                self.assertEqual(cached.read_bytes(), before)
            other = {**metadata, "baseModel": "SDXL 1.0"}
            cached.write_text(json.dumps(other), encoding="utf-8")
            before = cached.read_bytes()
            owner.detect_checkpoint_base.return_value = "HunyuanImage21"
            self.assertEqual(
                scope["get_models_by_path"](owner, "checkpoints", path),
                {**other, "baseModel": "HunyuanImage21"},
            )
            self.assertEqual(cached.read_bytes(), before)
            cached.write_text(json.dumps(metadata), encoding="utf-8")

            def replace(_):
                path.write_bytes(b"replacement with new file identity")
                return "HunyuanImage21"

            owner.detect_checkpoint_base.side_effect = replace
            self.assertEqual(scope["get_models_by_path"](owner, "checkpoints", path), metadata)

    def test_real_pinned_diffusers_conversion_and_unet_detection_on_meta_tensors(self):
        script = r"""
import copy, json, os, sys
from functools import wraps
from pathlib import Path
from unittest.mock import patch
source = Path(os.environ.get('RF_TEST_COMFY_SOURCE', Path.cwd() / 'repositories/ComfyUI'))
sys.path.insert(0, str(source))
sys.argv = [sys.argv[0], '--cpu']
import torch
def forbidden(*args, **kwargs):
    raise AssertionError('Accelerator probe during shape-only classification')
for backend in (torch.cuda, torch.xpu):
    for name in ('_lazy_init', 'init', 'is_available', 'device_count', 'current_device',
                 'get_device_properties', 'get_device_name', 'get_device_capability', 'mem_get_info'):
        if hasattr(backend, name):
            setattr(backend, name, forbidden)
@wraps(torch.backends.mps.is_available)
def no_mps():
    return forbidden()
torch.backends.mps.is_available = no_mps
from comfy.cli_args import args
args.cpu = True
from ruinedfooocus.backends.comfy.cpu_startup import initialize_cpu_model_management
initialize_cpu_model_management()
from comfy import model_detection
from ruinedfooocus.features.models.model_architecture import detect_tensor_architecture
from ruinedfooocus.features.models.image_scope import unsupported_tensor_family

sdxl = {'conv_in.weight': [320, 4, 3, 3], 'add_embedding.linear_1.weight': [1280, 2816]}
for block, depth in enumerate((0, 2, 10)):
    for index in range(2):
        sdxl[f'down_blocks.{block}.resnets.{index}.conv1.weight'] = [1]
        for layer in range(depth):
            sdxl[f'down_blocks.{block}.attentions.{index}.transformer_blocks.{layer}.attn2.to_k.weight'] = [64, 2048]
sd3 = {'pos_embed.proj.weight': [1536, 16, 2, 2]}
for name in ('add_q_proj', 'add_k_proj', 'add_v_proj'):
    sd3[f'transformer_blocks.0.attn.{name}.weight'] = [1536, 1536]
with patch.object(model_detection, 'convert_diffusers_mmdit', wraps=model_detection.convert_diffusers_mmdit) as convert:
    for shapes, expected in ((sdxl, 'SDXL 1.0'), (sd3, 'SD3')):
        header = {name: {'shape': shape} for name, shape in shapes.items()}
        before = copy.deepcopy(header)
        assert detect_tensor_architecture(header) == expected, expected
        assert header == before
    assert convert.call_count >= 2
for patch_shape, vector, vision, expected, passive in (
    ([2, 2], False, False, 'HunyuanImage21', None),
    ([1, 1, 1], False, False, 'HunyuanImage21Refiner', None),
    ([1, 2, 2], True, False, 'Hunyuan Video', 'Hunyuan Video'),
    ([1, 2, 2], False, True, 'HunyuanVideo15', 'Hunyuan Video'),
):
    shapes = {'txt_in.individual_token_refiner.blocks.0.norm1.weight': [3072],
              'img_in.proj.weight': [3072, 64, *patch_shape],
              'final_layer.linear.weight': [64 * __import__('math').prod(patch_shape), 3072],
              'txt_in.input_embedder.weight': [3072, 4096]}
    if vector:
        shapes['vector_in.in_layer.weight'] = [3072, 768]
    if vision:
        shapes['vision_in.proj.0.weight'] = [1152, 1152]
    header = {name: {'shape': shape} for name, shape in shapes.items()}
    assert detect_tensor_architecture(header) == expected, expected
    assert unsupported_tensor_family(header) == passive, expected
assert not torch.cuda.is_initialized()
print(json.dumps({'native_source': str(source), 'cases': ['Diffusers SDXL UNet', 'Diffusers SD3 MMDiT', 'Hunyuan image/refiner/video/video15'], 'tensor_device': 'meta', 'gpu_initialized': False}))
"""
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"tensor_device": "meta"', result.stdout)
