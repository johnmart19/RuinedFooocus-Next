"""Native attention must retain workspace protection and portable fallbacks."""

from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.backends.comfy.model_attention import configure_qwen_attention


class QwenAttentionPolicyTests(unittest.TestCase):
    def configure(self, native=True, os_name="Windows", hip="test", **flags):
        comfy = ModuleType("comfy")
        comfy.model_management = SimpleNamespace(
            pytorch_attention_enabled=lambda: native,
            pytorch_attention_flash_attention=lambda: native,
            xformers_enabled=lambda: False,
            dtype_size=lambda dtype: 2,
        )
        attention = ModuleType("comfy.ldm.modules.attention")
        attention.attention_sub_quad = Mock(return_value="bounded")
        model = type(
            "QwenImage",
            (),
            {"memory_usage_factor": 1.0, "get_dtype_inference": lambda self: "fp16"},
        )()
        patcher = SimpleNamespace(model=model, model_options={})
        modules = {
            "torch": SimpleNamespace(version=SimpleNamespace(hip=hip)),
            "comfy": comfy,
            "comfy.cli_args": SimpleNamespace(args=SimpleNamespace(**({"cpu": False} | flags))),
            "comfy.ldm.modules.attention": attention,
        }
        with (
            patch.dict("sys.modules", modules),
            patch(
                "ruinedfooocus.backends.comfy.model_attention.platform.system", return_value=os_name
            ),
        ):
            result = configure_qwen_attention(patcher)
            factor = model.memory_usage_factor
            configure_qwen_attention(patcher)
            self.assertEqual(
                model.memory_usage_factor, factor, "Workspace reserve must be idempotent"
            )
        return result, patcher, attention

    def test_verified_native_attention_retains_the_existing_workspace_allowance(self):
        result, patcher, attention = self.configure()
        self.assertTrue(result)
        self.assertNotIn(
            "optimized_attention_override", patcher.model_options["transformer_options"]
        )
        self.assertEqual(patcher.model.memory_usage_factor, 9.375)
        attention.attention_sub_quad.assert_not_called()

    def test_no_native_kernel_keeps_memory_bounded_fallback(self):
        result, patcher, attention = self.configure(native=False)
        self.assertTrue(result)
        override = patcher.model_options["transformer_options"]["optimized_attention_override"]
        self.assertEqual(override(None, "q", "k", "v", 4), "bounded")
        attention.attention_sub_quad.assert_called_once_with("q", "k", "v", 4)

    def test_cpu_cuda_directml_and_explicit_choices_are_unchanged(self):
        for options in (
            {"os_name": "Linux"},
            {"hip": None},
            {"cpu": True},
            {"use_quad_cross_attention": True},
            {"use_pytorch_cross_attention": True},
        ):
            with self.subTest(options=options):
                result, patcher, _ = self.configure(**options)
                self.assertFalse(result)
                self.assertEqual(patcher.model_options, {})
                self.assertEqual(patcher.model.memory_usage_factor, 1.0)


if __name__ == "__main__":
    unittest.main()
