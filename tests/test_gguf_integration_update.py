"""Focused CPU tests for GGUF diffusion loader tuning and modern GGUF quant headers."""

import ast
import importlib
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gguf

from ruinedfooocus.core.settings_schema import (
    BOOLEAN_FIELDS,
    ENUM_FIELDS,
    GGUF_DTYPE_CHOICES,
    validate_known,
)
from ruinedfooocus.features.models.gguf_metadata import read_gguf_header

ROOT = Path(__file__).resolve().parents[1]


def _write_minimal_gguf(path, tensors, *, architecture="flux", alignment=32):
    """Write a minimal valid GGUF v3 file with given (name, gguf_type, shape) tensors."""

    def _pack_string(text):
        encoded = text.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    fields = [("general.architecture", 8, _pack_string(architecture))]
    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<I", 3)
    header += struct.pack("<QQ", len(tensors), len(fields))
    for key, kind, payload in fields:
        header += _pack_string(key) + struct.pack("<I", kind) + payload

    offset = 0
    tensor_blobs = bytearray()
    for name, qtype, shape in tensors:
        block, width = gguf.GGML_QUANT_SIZES[qtype]
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = (numel // block) * width
        # GGUF stores dimensions reversed (ne[0] is innermost).
        dims = list(reversed(shape))
        header += _pack_string(name)
        header += struct.pack("<I", len(dims))
        for dim in dims:
            header += struct.pack("<Q", dim)
        header += struct.pack("<IQ", int(qtype), offset)
        padded = ((nbytes + alignment - 1) // alignment) * alignment
        tensor_blobs += b"\x00" * padded
        offset += padded

    pad = (-len(header)) % alignment
    path.write_bytes(bytes(header) + (b"\x00" * pad) + bytes(tensor_blobs))


class GGUFSettingsSchemaTests(unittest.TestCase):
    def test_schema_exposes_gguf_loader_controls(self):
        self.assertEqual(
            GGUF_DTYPE_CHOICES, ("default", "target", "float32", "float16", "bfloat16")
        )
        self.assertEqual(ENUM_FIELDS["gguf_dequant_dtype"], GGUF_DTYPE_CHOICES)
        self.assertEqual(ENUM_FIELDS["gguf_patch_dtype"], GGUF_DTYPE_CHOICES)
        self.assertIn("gguf_patch_on_device", BOOLEAN_FIELDS)

    def test_validate_known_accepts_and_rejects_gguf_settings(self):
        validated = validate_known(
            {
                "gguf_dequant_dtype": "bfloat16",
                "gguf_patch_dtype": "target",
                "gguf_patch_on_device": False,
            }
        )
        self.assertEqual(validated["gguf_dequant_dtype"], "bfloat16")
        self.assertEqual(validated["gguf_patch_dtype"], "target")
        self.assertIs(validated["gguf_patch_on_device"], False)

        with self.assertRaises(ValueError):
            validate_known({"gguf_dequant_dtype": "int4"})
        with self.assertRaises(ValueError):
            validate_known({"gguf_patch_dtype": "fp8"})
        with self.assertRaises(ValueError):
            validate_known({"gguf_patch_on_device": "yes"})


class GGUFLoaderWiringTests(unittest.TestCase):
    def _import_loader(self, advanced_cls):
        folder_paths = SimpleNamespace(add_model_folder_path=Mock())
        nodes = SimpleNamespace(UnetLoaderGGUFAdvanced=advanced_cls)
        stubs = {
            "folder_paths": folder_paths,
            "molbal_comfyui_gguf": SimpleNamespace(nodes=nodes),
            "molbal_comfyui_gguf.nodes": nodes,
        }
        sys.modules.pop("ruinedfooocus.backends.comfy.gguf_loader", None)
        with patch.dict(sys.modules, stubs):
            module = importlib.import_module("ruinedfooocus.backends.comfy.gguf_loader")
        return module, folder_paths

    def test_loader_options_defaults_and_overrides(self):
        module, _ = self._import_loader(Mock())
        self.assertEqual(
            module.loader_options({}),
            {
                "dequant_dtype": "default",
                "patch_dtype": "default",
                "patch_on_device": True,
            },
        )
        self.assertEqual(
            module.loader_options(
                {
                    "gguf_dequant_dtype": "float16",
                    "gguf_patch_dtype": "bfloat16",
                    "gguf_patch_on_device": False,
                }
            ),
            {
                "dequant_dtype": "float16",
                "patch_dtype": "bfloat16",
                "patch_on_device": False,
            },
        )
        self.assertEqual(
            module.loader_options(
                {
                    "gguf_dequant_dtype": "float16",
                    "gguf_patch_dtype": "bfloat16",
                    "gguf_patch_on_device": False,
                },
                dequant_dtype="target",
                patch_on_device=True,
            ),
            {
                "dequant_dtype": "target",
                "patch_dtype": "bfloat16",
                "patch_on_device": True,
            },
        )
        with self.assertRaises(ValueError):
            module.loader_options({"gguf_dequant_dtype": "invalid"})
        with self.assertRaises(ValueError):
            module.loader_options({"gguf_patch_dtype": "invalid"})
        with self.assertRaises(ValueError):
            module.loader_options({"gguf_patch_on_device": "true"})

    def test_load_diffusion_model_uses_advanced_loader_and_handles_model_name_tuple(self):
        sentinel_model = object()
        loader_instance = Mock()
        # Matches upstream molbal/ComfyUI-GGUF 5bf9d29ac9 returning (model, model_name).
        loader_instance.load_unet.return_value = (sentinel_model, "flux-dev-q4.gguf")
        advanced_cls = Mock(return_value=loader_instance)
        module, folder_paths = self._import_loader(advanced_cls)

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "flux-dev-q4.gguf"
            model_path.write_bytes(b"GGUF")
            with (
                patch(
                    "ruinedfooocus.features.models.image_scope.require_image_checkpoint",
                    return_value=model_path,
                ),
                patch.object(
                    module,
                    "_saved_settings",
                    return_value={
                        "gguf_dequant_dtype": "bfloat16",
                        "gguf_patch_dtype": "float32",
                        "gguf_patch_on_device": False,
                    },
                ),
            ):
                loaded = module.load_diffusion_model(str(model_path))

        self.assertIs(loaded, sentinel_model)
        folder_paths.add_model_folder_path.assert_called_once_with(
            "diffusion_models", str(model_path.resolve().parent), is_default=True
        )
        loader_instance.load_unet.assert_called_once_with(
            "flux-dev-q4.gguf",
            dequant_dtype="bfloat16",
            patch_dtype="float32",
            patch_on_device=False,
        )

    def test_load_diffusion_model_preserves_legacy_defaults_when_unset(self):
        sentinel_model = object()
        loader_instance = Mock()
        loader_instance.load_unet.return_value = (sentinel_model,)
        advanced_cls = Mock(return_value=loader_instance)
        module, _ = self._import_loader(advanced_cls)

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "sdxl-q8.gguf"
            model_path.write_bytes(b"GGUF")
            with (
                patch(
                    "ruinedfooocus.features.models.image_scope.require_image_checkpoint",
                    return_value=model_path,
                ),
                patch.object(module, "_saved_settings", return_value={}),
            ):
                loaded = module.load_diffusion_model(str(model_path))

        self.assertIs(loaded, sentinel_model)
        loader_instance.load_unet.assert_called_once_with(
            "sdxl-q8.gguf",
            dequant_dtype="default",
            patch_dtype="default",
            patch_on_device=True,
        )


class GGUFModernQuantHeaderTests(unittest.TestCase):
    def test_read_gguf_header_supports_modern_quant_types(self):
        modern_types = [
            ("w_mxfp4", gguf.GGMLQuantizationType.MXFP4, [4, 32]),
            ("w_nvfp4", gguf.GGMLQuantizationType.NVFP4, [4, 64]),
            ("w_tq1_0", gguf.GGMLQuantizationType.TQ1_0, [2, 256]),
            ("w_tq2_0", gguf.GGMLQuantizationType.TQ2_0, [2, 256]),
            ("w_iq4_xs", gguf.GGMLQuantizationType.IQ4_XS, [2, 256]),
            ("w_iq2_xxs", gguf.GGMLQuantizationType.IQ2_XXS, [2, 256]),
            ("w_q4_k", gguf.GGMLQuantizationType.Q4_K, [2, 256]),
            ("w_q8_0", gguf.GGMLQuantizationType.Q8_0, [4, 32]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "modern_quants.gguf"
            _write_minimal_gguf(path, modern_types, architecture="flux")
            header, arch = read_gguf_header(path, include_storage=True)

        self.assertEqual(arch, "flux")
        for name, qtype, shape in modern_types:
            self.assertIn(name, header)
            self.assertEqual(header[name]["shape"], shape)
            self.assertEqual(header[name]["gguf_type"], int(qtype))


class GGUFSettingsUIWiringTests(unittest.TestCase):
    def test_ui_settings_registers_gguf_loader_controls(self):
        tree = ast.parse((ROOT / "ruinedfooocus/ui/ui_settings.py").read_text(encoding="utf-8"))
        keys = [
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "add_setting"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ]
        for key in ("gguf_dequant_dtype", "gguf_patch_dtype", "gguf_patch_on_device"):
            self.assertIn(key, keys)


if __name__ == "__main__":
    unittest.main()
