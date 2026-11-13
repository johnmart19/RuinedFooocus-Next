"""Exercise explicit image GGUF LoRA dispatch with pinned native CPU patches."""

import ast
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gguf
import numpy as np

from ruinedfooocus.features.models.lora_names import (
    capture_gguf_lora,
    merge_loras,
    resolve_lora_name,
)
from ruinedfooocus.features.models.model_identity import file_signature


ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))
GGUF = Path(os.environ.get("RF_TEST_GGUF_SOURCE", ROOT / "repositories/molbal_comfyui_gguf"))


def functions(source, names, scope):
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"Expected native functions {names} in {source}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope


class GgufLoraDispatchTests(unittest.TestCase):
    def test_explicit_name_keeps_its_format_and_prompt_override_weight(self):
        installed = ["styles/detail.safetensors"]
        self.assertEqual(resolve_lora_name("detail.GGUF", installed), "detail.GGUF")
        self.assertEqual(resolve_lora_name("styles/detail.GGUF", installed), "styles/detail.GGUF")
        self.assertEqual(
            merge_loras(
                [{"name": "styles/detail.GGUF", "weight": 0.1}],
                [{"name": r"styles\detail.GGUF", "weight": -0.75}],
                installed,
            ),
            [{"name": r"styles\detail.GGUF", "weight": -0.75}],
        )

    def test_file_capture_is_local_and_uses_configured_root_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            roots = [root / "first", root / "second"]
            for folder in roots:
                (folder / "nested").mkdir(parents=True)
                writer = gguf.GGUFWriter(folder / "nested/local.GGUF", "sdxl")
                writer.add_string("general.type", "adapter")
                writer.add_string("adapter.type", "lora")
                writer.add_tensor("linear.weight.lora_a", np.zeros((1, 4), dtype=np.float32))
                writer.add_tensor("linear.weight.lora_b", np.zeros((3, 1), dtype=np.float32))
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
            source, stamp = capture_gguf_lora(r"nested\local.GGUF", roots)
            self.assertEqual(source, roots[0] / "nested/local.GGUF")
            self.assertEqual(stamp, file_signature(source))
            with (
                patch(
                    "ruinedfooocus.features.models.lora_names.file_signature",
                    return_value=[str(root / "outside.gguf"), 1, 1, 1],
                ),
                self.assertRaisesRegex(ValueError, "outside its configured"),
            ):
                capture_gguf_lora("nested/local.GGUF", roots)
            for invalid in (
                str(source),
                "../private.GGUF",
                r"C:\private.GGUF",
                "local:stream.gguf",
            ):
                with (
                    self.subTest(invalid=invalid),
                    self.assertRaisesRegex(ValueError, "relative filename"),
                ):
                    capture_gguf_lora(invalid, roots)
            with self.assertRaises(FileNotFoundError):
                capture_gguf_lora("local.GGUF", roots)

    def test_pinned_adapter_load_and_native_patch_weights(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy, sys; runpy.run_path(sys.argv[1])['check_native_patches']()",
                str(Path(__file__).resolve()),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def check_native_patches():
    # Own the native import environment and fail if any accelerator initializes.
    sys.path.insert(0, str(COMFY))
    sys.argv = [sys.argv[0], "--cpu"]
    import gguf
    import torch
    from safetensors.torch import save_file
    from comfy.cli_args import args
    from ruinedfooocus.backends.comfy.cpu_startup import initialize_cpu_model_management

    def reject_accelerator(*_, **__):
        raise AssertionError("Unexpected accelerator initialization during GGUF LoRA CPU test")

    for backend in (torch.cuda, torch.xpu):
        for name in (
            "_lazy_init",
            "init",
            "is_available",
            "device_count",
            "current_device",
            "get_device_properties",
            "get_device_name",
            "get_device_capability",
            "mem_get_info",
        ):
            if hasattr(backend, name):
                setattr(backend, name, reject_accelerator)
    args.cpu = True
    initialize_cpu_model_management()
    import comfy.lora
    import comfy.model_patcher
    import comfy.sd
    from ruinedfooocus.backends.comfy.gguf_lora import apply_gguf_lora

    loader = runpy.run_path(str(GGUF / "lora.py"))["load_gguf_lora"]
    aliases = functions(
        GGUF / "nodes.py",
        ("_gguf_lora_key_map", "_remap_gguf_lora_for_comfy"),
        dict(comfy=comfy, torch=torch),
    )
    vendor = {
        "molbal_comfyui_gguf.lora": SimpleNamespace(load_gguf_lora=loader),
        "molbal_comfyui_gguf.nodes": SimpleNamespace(**aliases),
    }
    cpu = torch.device("cpu")
    target = "diffusion_model.linear.weight"
    clip_target = "clip_l.transformer.text_projection.weight"
    down = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    up = torch.tensor([[1.0], [2.0], [3.0]])
    delta = up @ down
    model = torch.nn.Module()
    model.model_config = SimpleNamespace(unet_config={})
    model.diffusion_model = torch.nn.Module()
    model.diffusion_model.linear = torch.nn.Linear(4, 3, bias=False, device=cpu)
    unet = comfy.model_patcher.ModelPatcher(model, cpu, cpu)
    clip = comfy.sd.CLIP(no_init=True)
    clip.cond_stage_model = torch.nn.Module()
    clip.cond_stage_model.clip_l = torch.nn.Module()
    clip.cond_stage_model.clip_l.transformer = torch.nn.Module()
    clip.cond_stage_model.clip_l.transformer.text_projection = torch.nn.Linear(
        4, 3, bias=False, device=cpu
    )
    clip.patcher = comfy.model_patcher.ModelPatcher(clip.cond_stage_model, cpu, cpu)
    clip.tokenizer, clip.layer_idx, clip.tokenizer_options = None, None, {}
    clip.use_clip_schedule, clip.apply_hooks_to_conds = False, None
    check = unittest.TestCase()

    def write(path, pairs, *, general_type="adapter", alpha=2.0, quantized_down=False):
        writer = gguf.GGUFWriter(path, "sdxl")
        writer.add_string("general.type", general_type)
        writer.add_string("adapter.type", "lora")
        writer.add_float32("adapter.lora.alpha", alpha)
        for name, first, second in pairs:
            if quantized_down:
                dtype = gguf.GGMLQuantizationType.Q8_0
                writer.add_tensor(
                    f"{name}.lora_a", gguf.quants.quantize(first.numpy(), dtype), raw_dtype=dtype
                )
            else:
                writer.add_tensor(f"{name}.lora_a", first.numpy())
            writer.add_tensor(f"{name}.lora_b", second.numpy())
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, vendor):
        root = Path(directory).resolve()
        native_gguf = root / "a.GGUF"
        ordinary = root / "z.safetensors"
        bad = root / "bad.gguf"
        write(native_gguf, [(target, down, up), ("lora_te1_text_projection.weight", down, up)])
        save_file(
            {
                "diffusion_model.linear.lora_A.weight": down,
                "diffusion_model.linear.lora_B.weight": up,
            },
            ordinary,
        )
        native_gguf_signature = file_signature(native_gguf)
        lookup = Mock(side_effect=lambda _, name, **__: root / name)
        scope = functions(
            ROOT / "ruinedfooocus/features/generation/image_pipeline.py",
            ("load_loras",),
            dict(
                comfy=comfy,
                file_signature=file_signature,
                shared=SimpleNamespace(
                    models=SimpleNamespace(
                        model_dirs={"loras": [root]},
                        get_model_path=lookup,
                    )
                ),
            ),
        )
        run = scope["load_loras"]
        base = SimpleNamespace(unet=unet, clip=clip, vae=None, clip_vision=None)
        owner = SimpleNamespace(
            xl_base=base, xl_base_patched_hash=None, StableDiffusionModel=SimpleNamespace
        )
        request = [
            {"name": ordinary.name, "weight": 0.75},
            {"name": native_gguf.name, "weight": -0.5},
        ]
        run(owner, request)
        first = owner.xl_base_patched
        lookup.assert_called_once_with("loras", ordinary.name, hash=None)
        check.assertFalse(unet.patches)
        check.assertFalse(clip.patcher.patches)
        check.assertEqual([row[0] for row in first.unet.patches[target]], [-0.5, 0.75])
        check.assertEqual([row[0] for row in first.clip.patcher.patches[clip_target]], [-0.5])
        torch.testing.assert_close(
            comfy.lora.calculate_weight(
                first.unet.patches[target], torch.zeros_like(delta), target
            ),
            -0.25 * delta,
        )
        torch.testing.assert_close(
            comfy.lora.calculate_weight(
                first.clip.patcher.patches[clip_target], torch.zeros_like(delta), clip_target
            ),
            -delta,
        )
        check.assertEqual(
            first.unet.get_attachment("lora_metadata")["gguf_lora"]["target_count"], 2
        )
        run(owner, request)
        check.assertIs(owner.xl_base_patched, first)

        for pairs, options, expected in (
            ([(target, down, up)], {"general_type": "model"}, "not an adapter"),
            ([("unknown.weight", down, up)], {}, "targets do not match"),
            ([(target, down[:, :3].contiguous(), up)], {}, "factor dimensions"),
            (
                [(target, down, up), ("lora_unet_linear.weight", down, up)],
                {},
                "duplicate native targets",
            ),
            ([(target, down, up)], {"alpha": float("nan")}, "alpha must be finite"),
        ):
            write(bad, pairs, **options)
            expected_error = (
                ValueError if "general_type" in options or "alpha" in options else RuntimeError
            )
            with check.assertRaisesRegex(expected_error, expected):
                run(owner, request + [{"name": bad.name, "weight": 1.0}])
            check.assertIs(owner.xl_base_patched, first)
        for strength in (True, float("nan"), float("inf"), 100.1):
            with check.assertRaisesRegex(ValueError, "strength"):
                run(owner, [{"name": native_gguf.name, "weight": strength}])
        with check.assertRaisesRegex(ValueError, "more than once"):
            run(owner, request + [{"name": "./a.GGUF", "weight": 0.4}])

        write(native_gguf, [(target, down, up)], alpha=1.0)
        with check.assertRaisesRegex(ValueError, "changed before loading"):
            apply_gguf_lora(native_gguf, native_gguf_signature, unet, clip, -0.5)

        def changing_loader(path):
            factors = loader(path)
            write(Path(path), [(target, down, up)], alpha=3.0)
            return factors

        with (
            patch.object(vendor["molbal_comfyui_gguf.lora"], "load_gguf_lora", changing_loader),
            check.assertRaisesRegex(ValueError, "changed during loading"),
        ):
            apply_gguf_lora(native_gguf, file_signature(native_gguf), unet, clip, -0.5)
        run(owner, request)
        check.assertIsNot(owner.xl_base_patched, first)
        check.assertFalse(owner.xl_base_patched.clip.patcher.patches)
        run(owner, [{"name": native_gguf.name, "weight": 0}])
        check.assertIs(owner.xl_base_patched, base)

        flux = comfy.model_base.Flux.__new__(comfy.model_base.Flux)
        torch.nn.Module.__init__(flux)
        flux.model_config = SimpleNamespace(unet_config={"depth": 1, "hidden_size": 2})
        qkv = torch.nn.Linear(32, 6, bias=False, device=cpu)
        flux.diffusion_model = torch.nn.Module()
        block = torch.nn.Module()
        block.img_attn = torch.nn.Module()
        block.img_attn.qkv = qkv
        flux.diffusion_model.double_blocks = torch.nn.ModuleList([block])
        flux_patcher = comfy.model_patcher.ModelPatcher(flux, cpu, cpu)
        split = root / "split.gguf"
        split_down, split_up = torch.ones(1, 32), torch.ones(2, 1, dtype=torch.float16)
        write(
            split,
            [("transformer.transformer_blocks.0.attn.to_k.weight", split_down, split_up)],
            alpha=1.0,
            quantized_down=True,
        )
        patched, _ = apply_gguf_lora(split, file_signature(split), flux_patcher, None, 0.5)
        qkv_key = "diffusion_model.double_blocks.0.img_attn.qkv.weight"
        expected = torch.zeros(6, 32)
        expected[2:4] = 0.5
        torch.testing.assert_close(
            comfy.lora.calculate_weight(
                patched.patches[qkv_key], torch.zeros_like(expected), qkv_key
            ),
            expected,
            atol=0.001,
            rtol=0,
        )
        check.assertFalse(torch.cuda.is_initialized())
