"""Runtime details describe initialized state without hardware probes."""

from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ruinedfooocus.backends.comfy.runtime_info import runtime_details


class ImageRuntimeInfoTests(unittest.TestCase):
    def details(
        self, *, device="cpu", hip="7.16", cuda=None, directml=None, sdk=None, preview_method=None
    ):
        torch = SimpleNamespace(
            __version__="test-build", version=SimpleNamespace(hip=hip, cuda=cuda)
        )
        args = SimpleNamespace(
            directml=directml,
            disable_async_offload=False,
            async_offload=None,
            cpu_vae=False,
            fast=set(),
            preview_method=preview_method,
        )
        return dict(
            runtime_details(
                torch,
                SimpleNamespace(type=device),
                "Selected test device",
                args,
                SimpleNamespace(aimdo_enabled=False),
                package_version=sdk or Mock(return_value="10.1"),
            )
        )

    def test_cpu_on_amd_wheel_distinguishes_active_backend_and_compiled_components(self):
        rows = self.details()
        self.assertEqual(rows["Active image backend"], "CPU")
        self.assertEqual(rows["Compiled HIP component"], "7.16")
        self.assertEqual(rows["ROCm wheel SDK"], "10.1")
        self.assertEqual(rows["Dynamic VRAM active"], "No")

    def test_cuda_and_legacy_directml_do_not_infer_rocm_from_gpu_api_spelling(self):
        no_sdk = Mock(side_effect=AssertionError("No AMD package query on CUDA"))
        rows = self.details(device="cuda", hip=None, cuda="13.2", sdk=no_sdk)
        self.assertEqual(rows["Active image backend"], "CUDA")
        self.assertEqual(rows["Compiled CUDA runtime"], "13.2")
        rows = self.details(device="privateuseone", hip=None, directml=0, sdk=no_sdk)
        self.assertEqual(rows["Active image backend"], "DirectML (legacy PyTorch)")
        no_sdk.assert_not_called()

    def test_non_wheel_rocm_does_not_invent_a_matching_sdk_release(self):
        rows = self.details(device="cuda", sdk=Mock(side_effect=PackageNotFoundError("rocm")))
        self.assertEqual(rows["Active image backend"], "ROCm / HIP")
        self.assertIn("Not declared", rows["ROCm wheel SDK"])

    def test_sampling_preview_reports_initialized_policy_and_conditional_fallback(self):
        for value, expected in (
            ("auto", "Auto (Fast RGB when the model supports it)"),
            ("none", "Off"),
            ("latent2rgb", "Fast RGB when the model supports it"),
            ("taesd", "Approximate VAE when available; otherwise Fast RGB when supported"),
            (None, "Unknown"),
        ):
            with self.subTest(value=value):
                rows = self.details(preview_method=SimpleNamespace(value=value))
                self.assertEqual(rows["Sampling preview at launch"], expected)

    def test_explicit_zero_and_disabled_streams_remain_distinct(self):
        torch = SimpleNamespace(__version__="cpu", version=SimpleNamespace(hip=None, cuda=None))
        args = SimpleNamespace(
            directml=None,
            disable_async_offload=False,
            async_offload=0,
            cpu_vae=True,
            fast={"fp16_accumulation"},
        )
        for disabled, expected in ((False, "0"), (True, "Disabled")):
            args.disable_async_offload = disabled
            rows = dict(
                runtime_details(
                    torch,
                    SimpleNamespace(type="cpu"),
                    "CPU",
                    args,
                    SimpleNamespace(aimdo_enabled=False),
                )
            )
            self.assertEqual(rows["Weight transfer streams requested"], expected)
            self.assertEqual(rows["VAE target requested"], "CPU")
            self.assertEqual(rows["Fast optimizations requested"], "fp16_accumulation")


if __name__ == "__main__":
    unittest.main()
