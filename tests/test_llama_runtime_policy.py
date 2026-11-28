"""Pure acquisition preference and cached-GPU-wheel CPU-offload checks."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ruinedfooocus.backends.llama.runtime_policy import (
    admit_xllamacpp_devices,
    configure_xllamacpp_offload,
    preferred_backend,
)


class LlamaRuntimePolicyTests(unittest.TestCase):
    def test_actual_device_admission_explains_auto_cpu_and_rejects_explicit_gpu(self):
        engine = Mock()
        engine.get_device_info.return_value = [{"name": "CPU"}]
        label, cpu, reason = admit_xllamacpp_devices(engine)
        self.assertEqual(label, "CPU")
        self.assertTrue(cpu)
        self.assertIn("No usable GPU", reason)
        with self.assertRaisesRegex(RuntimeError, "no usable Vulkan"):
            admit_xllamacpp_devices(engine, "Vulkan")
        engine.get_device_info.side_effect = RuntimeError("device failure")
        with self.assertRaisesRegex(RuntimeError, "device failure"):
            admit_xllamacpp_devices(engine)

    def test_explicit_cpu_never_queries_gpu_and_cached_cuda_is_real_alternative(self):
        engine = Mock()
        self.assertEqual(admit_xllamacpp_devices(engine, cpu=True), ("CPU", True, ""))
        engine.get_device_info.assert_not_called()
        engine.get_device_info.return_value = [{"name": "CUDA0"}, {"name": "CPU"}]
        self.assertEqual(admit_xllamacpp_devices(engine), ("CUDA", False, ""))

    def test_auto_is_vulkan_first_for_both_engines_independent_of_image_torch(self):
        for system in ("Windows", "Linux"):
            self.assertEqual(preferred_backend(system), "Vulkan")
            self.assertEqual(preferred_backend(system, gpu=False), "CPU")
            self.assertEqual(preferred_backend(system, "ROCm"), "ROCm")
        self.assertEqual(preferred_backend("Darwin"), "Auto")

    def test_saved_cpu_disables_every_offload_control_with_cached_vulkan(self):
        params = SimpleNamespace(
            n_gpu_layers=-1, no_kv_offload=False, no_op_offload=False, mmproj_use_gpu=True
        )
        configure_xllamacpp_offload(params, {"llm_n_gpu_layers": -1}, cpu=True)
        self.assertEqual(params.n_gpu_layers, 0)
        self.assertTrue(params.no_kv_offload)
        self.assertTrue(params.no_op_offload)
        self.assertFalse(params.mmproj_use_gpu)
