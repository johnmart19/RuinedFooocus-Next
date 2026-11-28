"""Focused tests for xllamacpp 2026.11.12361 packaging matrix and CommonParams mapping."""

import ast
from enum import IntEnum
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from ruinedfooocus.backends.llama.runtime_policy import configure_xllamacpp_params
from ruinedfooocus.packaging import gpu_installer, hardware_inventory, support_matrix

ROOT = Path(__file__).resolve().parents[1]


def _load_current_model_settings():
    tree = ast.parse(
        (ROOT / "ruinedfooocus/features/chat/llama_pipeline.py").read_text(encoding="utf-8")
    )
    pipeline_cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pipeline"
    )
    method = next(
        node
        for node in pipeline_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "current_model_settings"
    )
    scope = {
        "DEFAULT_MODEL": "default.gguf",
        "json": json,
        "settings": SimpleNamespace(default_settings={}),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), "llama_pipeline.py", "exec"), scope)
    return scope["current_model_settings"]


class _FakeGgmlType(IntEnum):
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_BF16 = 30


class _FakeFlashAttnType(IntEnum):
    LLAMA_FLASH_ATTN_TYPE_AUTO = -1
    LLAMA_FLASH_ATTN_TYPE_DISABLED = 0
    LLAMA_FLASH_ATTN_TYPE_ENABLED = 1


class _FakeLoadMode(IntEnum):
    LLAMA_LOAD_MODE_AUTO = 0
    LLAMA_LOAD_MODE_NONE = 1
    LLAMA_LOAD_MODE_MMAP = 2


class _FakeReasoningFormat(IntEnum):
    COMMON_REASONING_FORMAT_NONE = 0
    COMMON_REASONING_FORMAT_AUTO = 1
    COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY = 2
    COMMON_REASONING_FORMAT_DEEPSEEK = 3


def _fake_engine(*, module_level_enums=False):
    if module_level_enums:
        attrs = {}
        for enum_cls in (_FakeGgmlType, _FakeFlashAttnType, _FakeLoadMode, _FakeReasoningFormat):
            for member in enum_cls:
                attrs[member.name] = member
        return SimpleNamespace(**attrs)
    return SimpleNamespace(
        ggml_type=_FakeGgmlType,
        llama_flash_attn_type=_FakeFlashAttnType,
        llama_load_mode=_FakeLoadMode,
        common_reasoning_format=_FakeReasoningFormat,
    )


def _fake_params():
    return SimpleNamespace(
        n_predict=-1,
        n_ctx=8192,
        n_batch=2048,
        n_ubatch=512,
        n_parallel=-1,
        n_gpu_layers=-1,
        no_kv_offload=False,
        no_op_offload=False,
        mmproj=SimpleNamespace(path=""),
        mmproj_use_gpu=True,
        no_mmproj=False,
        flash_attn_type=_FakeFlashAttnType.LLAMA_FLASH_ATTN_TYPE_AUTO,
        cache_type_k=_FakeGgmlType.GGML_TYPE_F16,
        cache_type_v=_FakeGgmlType.GGML_TYPE_F16,
        load_mode=_FakeLoadMode.LLAMA_LOAD_MODE_AUTO,
        moe_cache_size=0,
        moe_cache_cpu_share=0.0,
        reasoning_format=_FakeReasoningFormat.COMMON_REASONING_FORMAT_DEEPSEEK,
        enable_reasoning=-1,
        sampling=SimpleNamespace(reasoning_budget_tokens=-1),
        ctx_shift=False,
        endpoint_metrics=True,
        use_jinja=False,
    )


class XllamacppPackagingMatrixTests(unittest.TestCase):
    def test_pinned_release_matches_latest_2026_11_12361(self):
        self.assertEqual(gpu_installer.XLLAMACPP_VERSION, "2026.11.12361")
        self.assertEqual(gpu_installer.XLLAMACPP_LLAMA_TAG, "b12361")
        self.assertEqual(
            hardware_inventory.XLLAMACPP_INDEXES,
            {
                "CPU": None,
                "Vulkan": "vulkan",
                "CUDA 12.8": "cu128",
                "CUDA 13.2": "cu132",
                "ROCm 6.4": "rocm-6.4.1",
                "ROCm 7.2": "rocm-7.2.4",
            },
        )

    def test_wheel_index_routing_across_platforms_and_architectures(self):
        self.assertEqual(
            gpu_installer.xllamacpp_index("cu132", "Windows", choice="CUDA 13.2", machine="AMD64"),
            "https://xorbitsai.github.io/xllamacpp/whl/cu132",
        )
        self.assertEqual(
            gpu_installer.xllamacpp_index("cu128", "Linux", choice="CUDA 12.8", machine="aarch64"),
            "https://xorbitsai.github.io/xllamacpp/whl/cu128",
        )
        self.assertEqual(
            gpu_installer.xllamacpp_index("cu134", "Linux", choice="Auto", machine="aarch64"),
            "https://xorbitsai.github.io/xllamacpp/whl/cu132",
        )
        self.assertEqual(
            gpu_installer.xllamacpp_index("cpu", "Windows", choice="Vulkan", machine="ARM64"),
            "https://xorbitsai.github.io/xllamacpp/whl/vulkan",
        )
        self.assertEqual(
            gpu_installer.xllamacpp_index("rocm7.2", "Linux", choice="ROCm 7.2", machine="x86_64"),
            "https://xorbitsai.github.io/xllamacpp/whl/rocm-7.2.4",
        )
        self.assertEqual(
            gpu_installer.xllamacpp_index("mps", "Darwin", choice="Auto", machine="arm64"),
            "https://pypi.org/simple",
        )
        with self.assertRaisesRegex(RuntimeError, "ROCm 10 PyTorch"):
            gpu_installer.xllamacpp_index("rocm10.2", "Linux", choice="ROCm 7.2", machine="x86_64")
        with self.assertRaisesRegex(RuntimeError, "Linux/WSL x86_64"):
            gpu_installer.xllamacpp_index("rocm7.2", "Windows", choice="ROCm 7.2", machine="AMD64")
        with self.assertRaisesRegex(RuntimeError, "Windows x86_64 and Linux x86_64/ARM64"):
            gpu_installer.xllamacpp_index("cu132", "Windows", choice="CUDA 13.2", machine="ARM64")

    def test_hardware_inventory_and_support_matrix_reflect_wheel_eligibility(self):
        nvidia = [{"vendor": "nvidia", "name": "RTX 5090"}]
        amd = [{"vendor": "amd", "name": "Radeon RX 9070"}]
        self.assertEqual(
            hardware_inventory.xllamacpp_backend_choices(nvidia, "Windows", "AMD64"),
            ["Auto", "CPU", "Vulkan", "CUDA 12.8", "CUDA 13.2"],
        )
        self.assertEqual(
            hardware_inventory.xllamacpp_backend_choices(nvidia, "Linux", "aarch64"),
            ["Auto", "CPU", "CUDA 12.8", "CUDA 13.2"],
        )
        self.assertEqual(
            hardware_inventory.xllamacpp_backend_choices(
                amd, "Linux", "x86_64", torch_version="2.15.0+rocm10.2.0"
            ),
            ["Auto", "CPU", "Vulkan"],
        )
        rows = {
            row["option"]: row
            for row in support_matrix.xllama_rows(amd, "Linux", "Linux", "x86_64")
        }
        self.assertIn("ROCm 7.2", rows)
        self.assertIn("Vulkan", rows)


class XllamacppCommonParamsMappingTests(unittest.TestCase):
    def test_kv_cache_quantization_auto_enables_flash_attn_on_gpu_only(self):
        engine = _fake_engine()
        params = _fake_params()
        configure_xllamacpp_params(
            params,
            {"llm_kv_cache_type": "q8_0", "llm_flash_attn": "Auto", "llm_n_ctx": 16384},
            engine=engine,
            cpu=False,
        )
        self.assertEqual(params.cache_type_k, _FakeGgmlType.GGML_TYPE_Q8_0)
        self.assertEqual(params.cache_type_v, _FakeGgmlType.GGML_TYPE_Q8_0)
        self.assertEqual(params.flash_attn_type, _FakeFlashAttnType.LLAMA_FLASH_ATTN_TYPE_ENABLED)
        self.assertEqual(params.n_ctx, 16384)
        self.assertEqual(params.n_parallel, 1)
        self.assertTrue(params.ctx_shift)
        self.assertFalse(params.endpoint_metrics)
        self.assertTrue(params.use_jinja)

        cpu_params = _fake_params()
        configure_xllamacpp_params(
            cpu_params,
            {"llm_kv_cache_type": "q4_0", "llm_flash_attn": "Auto"},
            engine=engine,
            cpu=True,
        )
        self.assertEqual(cpu_params.cache_type_k, _FakeGgmlType.GGML_TYPE_Q4_0)
        self.assertEqual(cpu_params.cache_type_v, _FakeGgmlType.GGML_TYPE_Q4_0)
        self.assertEqual(cpu_params.flash_attn_type, _FakeFlashAttnType.LLAMA_FLASH_ATTN_TYPE_AUTO)
        self.assertEqual(cpu_params.n_gpu_layers, 0)
        self.assertTrue(cpu_params.no_kv_offload)
        self.assertTrue(cpu_params.no_op_offload)
        self.assertFalse(cpu_params.mmproj_use_gpu)

    def test_explicit_flash_attn_off_overrides_kv_auto_promotion(self):
        engine = _fake_engine(module_level_enums=True)
        params = _fake_params()
        configure_xllamacpp_params(
            params,
            {"llm_kv_cache_type": "q8_0", "llm_flash_attn": "Off"},
            engine=engine,
            cpu=False,
        )
        self.assertEqual(params.flash_attn_type, _FakeFlashAttnType.LLAMA_FLASH_ATTN_TYPE_DISABLED)

    def test_batch_size_clamps_ubatch_and_load_mode_replaces_use_mmap(self):
        engine = _fake_engine()
        params = _fake_params()
        configure_xllamacpp_params(
            params,
            {"llm_batch_size": 256, "llm_mmap": "Disabled"},
            engine=engine,
            cpu=False,
        )
        self.assertEqual(params.n_batch, 256)
        self.assertEqual(params.n_ubatch, 256)
        self.assertEqual(params.load_mode, _FakeLoadMode.LLAMA_LOAD_MODE_NONE)
        self.assertFalse(hasattr(params, "use_mmap"))

        enabled_params = _fake_params()
        configure_xllamacpp_params(
            enabled_params,
            {"llm_mmap": "Enabled"},
            engine=engine,
            cpu=False,
        )
        self.assertEqual(enabled_params.load_mode, _FakeLoadMode.LLAMA_LOAD_MODE_MMAP)

        legacy_params = SimpleNamespace(
            n_predict=-1,
            n_ctx=8192,
            n_gpu_layers=-1,
            no_kv_offload=False,
            no_op_offload=False,
            mmproj=SimpleNamespace(path=""),
            mmproj_use_gpu=True,
            use_mmap=True,
            ctx_shift=False,
            endpoint_metrics=True,
            use_jinja=False,
        )
        configure_xllamacpp_params(
            legacy_params, {"llm_mmap": "Disabled"}, engine=engine, cpu=False
        )
        self.assertFalse(legacy_params.use_mmap)

    def test_moe_cache_share_mmproj_and_reasoning_budget(self):
        engine = _fake_engine()
        params = _fake_params()
        configure_xllamacpp_params(
            params,
            {
                "llama_moe_cache_cpu_share_enabled": True,
                "llama_moe_cache_cpu_share": 0.35,
            },
            engine=engine,
            cpu=False,
            projector="/models/mmproj.gguf",
            native_defaults=("--reasoning-budget", "0"),
        )
        self.assertAlmostEqual(params.moe_cache_cpu_share, 0.35)
        self.assertEqual(params.mmproj.path, "/models/mmproj.gguf")
        self.assertFalse(params.no_mmproj)
        self.assertTrue(params.mmproj_use_gpu)
        self.assertEqual(params.reasoning_format, _FakeReasoningFormat.COMMON_REASONING_FORMAT_AUTO)
        self.assertEqual(params.enable_reasoning, -1)
        self.assertEqual(params.sampling.reasoning_budget_tokens, 0)

        no_proj = _fake_params()
        configure_xllamacpp_params(
            no_proj,
            {"llama_moe_cache_cpu_share_enabled": True, "llama_moe_cache_cpu_share": 0.5},
            engine=engine,
            cpu=True,
            projector=None,
        )
        self.assertTrue(no_proj.no_mmproj)
        self.assertEqual(no_proj.moe_cache_cpu_share, 0.0)

    def test_pipeline_settings_fingerprint_tracks_xllamacpp_backend_and_perf_controls(self):
        current_model_settings = _load_current_model_settings()
        pipe = SimpleNamespace(loaded_paths=(), model_annotation=None)
        base = {
            "llm_runtime": "xllamacpp",
            "xllamacpp_backend": "Vulkan",
            "llm_flash_attn": "Auto",
            "llm_kv_cache_type": "Auto",
            "llm_batch_size": 0,
            "llm_mmap": "Auto",
        }
        fp_base = current_model_settings(pipe, values=base)
        for key, value in (
            ("xllamacpp_backend", "CUDA 13.2"),
            ("llm_flash_attn", "On"),
            ("llm_kv_cache_type", "q8_0"),
            ("llm_batch_size", 512),
            ("llm_mmap", "Disabled"),
        ):
            with self.subTest(key=key):
                mutated = dict(base, **{key: value})
                self.assertNotEqual(fp_base, current_model_settings(pipe, values=mutated))
