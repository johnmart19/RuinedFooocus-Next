"""Verify ComfyUI model coverage aliases, image compute optimizations, and LLM runtime flags."""

import argparse
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from ruinedfooocus.backends.comfy.comfy_args import PerformanceFeature, add_comfy_arguments
from ruinedfooocus.backends.comfy.image_compute import apply_compute, finalize_compute
from ruinedfooocus.core.settings_schema import COMPUTE_CHOICES, ENUM_FIELDS, INTEGER_RANGES
from ruinedfooocus.features.models.image_components import ALIASES, IMAGE_COMPONENTS


class ModelCoverageTests(unittest.TestCase):
    def test_comfy_model_classes_resolve_to_component_specs(self):
        for family, expected in (
            ("SD15", "BaseModel"),
            ("FluxSchnell", "Flux"),
            ("PixArtAlpha", "PixArt"),
            ("PixArtSigma", "PixArt"),
        ):
            with self.subTest(family=family):
                canonical = ALIASES.get(family, family)
                self.assertEqual(canonical, expected)
                self.assertIn(canonical, IMAGE_COMPONENTS)


class ImageComputeOptimizationTests(unittest.TestCase):
    def _parser(self):
        parser = argparse.ArgumentParser()
        add_comfy_arguments(parser)
        return parser

    def test_fp8_precision_attention_and_fast_mode_apply_and_respect_cpu(self):
        parser = self._parser()
        args = parser.parse_args([])
        apply_compute(
            args,
            {
                "image_unet_precision": "FP8_E4M3FN",
                "image_text_precision": "FP8_E5M2",
                "image_attention": "CK",
                "image_fast_mode": "FP16 accumulation",
            },
        )
        self.assertTrue(args.fp8_e4m3fn_unet)
        self.assertTrue(args.fp8_e5m2_text_enc)
        self.assertTrue(args.use_ck_attention)
        self.assertEqual(args.fast, [PerformanceFeature.Fp16Accumulation])

        # CPU finalization suppresses saved FP8 precision flags cleanly.
        suppressed = finalize_compute(args, cpu=True)
        self.assertFalse(args.fp8_e4m3fn_unet)
        self.assertFalse(args.fp8_e5m2_text_enc)
        self.assertIn("fp8_e4m3fn_unet", suppressed)

    def test_fast_mode_all_sets_empty_list_and_cli_takes_priority(self):
        parser = self._parser()
        args = parser.parse_args([])
        apply_compute(args, {"image_fast_mode": "All", "image_attention": "Flash"})
        self.assertEqual(args.fast, [])
        self.assertTrue(args.use_flash_attention)

        args_cli = parser.parse_args(["--fast", "fp8_matrix_mult", "--use-sage-attention"])
        apply_compute(
            args_cli,
            {"image_fast_mode": "All", "image_attention": "Flash"},
            options={"--fast", "--use-sage-attention"},
        )
        self.assertEqual(args_cli.fast, [PerformanceFeature.Fp8MatrixMultiplication])
        self.assertTrue(args_cli.use_sage_attention)
        self.assertFalse(args_cli.use_flash_attention)


class LlmOptimizationSchemaTests(unittest.TestCase):
    def test_llm_optimization_settings_are_registered_in_schema(self):
        self.assertEqual(ENUM_FIELDS["llm_flash_attn"], ("Auto", "On", "Off"))
        self.assertEqual(ENUM_FIELDS["llm_kv_cache_type"], ("Auto", "f16", "q8_0", "q4_0"))
        self.assertEqual(ENUM_FIELDS["llm_mmap"], ("Auto", "Enabled", "Disabled"))
        self.assertEqual(INTEGER_RANGES["llm_batch_size"], (0, 16384))
        self.assertIn("FP8_E4M3FN", COMPUTE_CHOICES["image_unet_precision"])
        self.assertIn("FP8_E5M2", COMPUTE_CHOICES["image_text_precision"])
        self.assertIn("CK", COMPUTE_CHOICES["image_attention"])
        self.assertIn("All", COMPUTE_CHOICES["image_fast_mode"])

    def test_llama_server_passes_flash_attn_kv_cache_batch_and_mmap_flags(self):
        from ruinedfooocus.backends.llama import llama_server

        captured = {}

        class FakeProc:
            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                pass

        def fake_start(cmd, **kwargs):
            captured["cmd"] = list(cmd)
            return FakeProc(), None

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "test.gguf"
            model_path.write_bytes(b"GGUF" + b"\x00" * 64)
            exe_path = Path(tmp) / "llama-server.exe"
            exe_path.write_bytes(b"MZ")
            with (
                patch.object(
                    llama_server,
                    "resolve_runtime",
                    return_value=(exe_path, {}, True, "ROCm", ""),
                ),
                patch.object(
                    llama_server,
                    "runtime_identity",
                    return_value={"signature": None, "manifest_verified": False},
                ),
                patch.object(
                    llama_server,
                    "model_header_evidence",
                    return_value={"moe": False},
                ),
                patch.object(
                    llama_server,
                    "moe_cache_cpu_share_capability",
                    return_value={
                        "arguments": [],
                        "extra_arguments": [],
                        "supported": False,
                        "source": "app_setting",
                    },
                ),
                patch.object(llama_server, "public_moe_cache_state", return_value={}),
                patch.object(llama_server, "warn_unsupported"),
                patch.object(llama_server, "projector_path", return_value=None),
                patch.object(llama_server, "start_process", side_effect=fake_start),
                patch.object(
                    llama_server.requests.Session,
                    "get",
                    return_value=SimpleNamespace(ok=True),
                ),
                patch.object(
                    llama_server,
                    "startup_evidence",
                    return_value={
                        "quantization": "Q4_K_M",
                        "architecture": "llama",
                        "moe_cache_buffers": [],
                        "host_experts": 0,
                        "state": "full",
                        "context": 8192,
                    },
                ),
            ):
                srv = llama_server.Server(
                    model_path,
                    {
                        "llm_n_ctx": 8192,
                        "llm_n_gpu_layers": -1,
                        "llm_kv_cache_type": "q8_0",
                        "llm_flash_attn": "Auto",
                        "llm_batch_size": 1024,
                        "llm_mmap": "Disabled",
                    },
                    gpu=True,
                )
                srv.close()
        cmd = captured["cmd"]
        self.assertIn("--cache-type-k", cmd)
        self.assertEqual(cmd[cmd.index("--cache-type-k") + 1], "q8_0")
        self.assertIn("--cache-type-v", cmd)
        self.assertEqual(cmd[cmd.index("--cache-type-v") + 1], "q8_0")
        # Auto flash-attn turns on automatically for quantized KV caches on GPU.
        self.assertIn("--flash-attn", cmd)
        self.assertEqual(cmd[cmd.index("--flash-attn") + 1], "on")
        self.assertIn("--batch-size", cmd)
        self.assertEqual(cmd[cmd.index("--batch-size") + 1], "1024")
        self.assertIn("--no-mmap", cmd)


if __name__ == "__main__":
    unittest.main()
