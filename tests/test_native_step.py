"""ComfyUI's per-node cleanup runs after every job and before a model is replaced.

The native modules are stand-ins; these checks do not allocate GPU memory. The
leak they guard against was measured on a GPU (see native_step.py).
"""

from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from ruinedfooocus.backends.comfy import native_step

ROOT = Path(__file__).resolve().parents[1]


class NativeStepTests(unittest.TestCase):
    def modules(self, enabled, calls):
        return {
            "comfy.memory_management": NS(aimdo_enabled=enabled),
            "comfy.model_management": NS(reset_cast_buffers=lambda: calls.append("reset")),
            "comfy.model_prefetch": NS(cleanup_prefetch_queues=lambda: calls.append("prefetch")),
            "comfy_aimdo.model_vbar": NS(
                vbars_reset_watermark_limits=lambda: calls.append("watermarks")
            ),
        }

    def test_upstream_order_with_dynamic_vram_and_nothing_without_it(self):
        calls = []
        stand_ins = self.modules(True, calls)
        with patch.dict(sys.modules, stand_ins):
            # `import a.b as c` resolves through the parent package attribute.
            with (
                patch.dict(
                    sys.modules,
                    {
                        "comfy": NS(
                            **{
                                k.split(".")[1]: v
                                for k, v in stand_ins.items()
                                if k.startswith("comfy.")
                            }
                        )
                    },
                ),
                patch.dict(
                    sys.modules, {"comfy_aimdo": NS(model_vbar=stand_ins["comfy_aimdo.model_vbar"])}
                ),
            ):
                self.assertTrue(native_step.end_native_step())
        # Same order as ComfyUI's execution.py after each node.
        self.assertEqual(calls, ["prefetch", "reset", "watermarks"])
        calls.clear()
        disabled = self.modules(False, calls)
        with (
            patch.dict(sys.modules, disabled),
            patch.dict(
                sys.modules, {"comfy": NS(memory_management=disabled["comfy.memory_management"])}
            ),
        ):
            self.assertFalse(native_step.end_native_step())
        self.assertEqual(calls, [])

    def test_model_replacement_clears_the_reference_before_collecting(self):
        from ruinedfooocus.features.generation import image_model_lifecycle

        order = []
        memory = NS(
            cleanup_models=lambda: order.append("cleanup_models"),
            soft_empty_cache=lambda: order.append("soft_empty_cache"),
        )
        with (
            patch.object(native_step, "end_native_step", lambda: order.append("end_native_step")),
            patch("gc.collect", lambda generation=2: order.append("gc")),
        ):
            image_model_lifecycle.collect_image_models(memory)
        self.assertEqual(order, ["end_native_step", "gc", "cleanup_models", "soft_empty_cache"])

    def test_every_worker_job_ends_with_the_cleanup(self):
        source = (ROOT / "ruinedfooocus/app/async_worker.py").read_text(encoding="utf-8")
        worker_source = source[source.index("def worker():") :]
        block = worker_source[worker_source.index("cleanup_interrupt(item)") :][:400]
        self.assertIn("end_native_step()", block)

    def test_release_native_runs_cleanup_before_unloading_models(self):
        from ruinedfooocus.app import resource_lifecycle

        order = []
        memory = NS(
            unload_all_models=lambda: order.append("unload_all_models"),
            soft_empty_cache=lambda: order.append("soft_empty_cache"),
        )
        with (
            patch.dict(sys.modules, {"comfy.model_management": memory}),
            patch.object(native_step, "end_native_step", lambda: order.append("end_native_step")),
        ):
            resource_lifecycle._release_native()
        self.assertEqual(order, ["end_native_step", "unload_all_models", "soft_empty_cache"])

    def test_vae_decode_oom_retry_runs_cleanup_before_unloading_unet(self):
        from ruinedfooocus.backends.comfy import vae_decode

        order = []
        attempts = [0]

        class OOMError(RuntimeError):
            pass

        class FakeVAEDecode:
            def decode(self, vae, samples):
                attempts[0] += 1
                if attempts[0] == 1:
                    raise OOMError("out of memory")
                return ("decoded",)

        memory = NS(
            raise_non_oom=lambda error: None,
            unload_model_and_clones=lambda unet: order.append(("unload", unet)),
            soft_empty_cache=lambda: order.append("soft_empty_cache"),
        )
        fake_nodes = NS(VAEDecode=FakeVAEDecode, VAEDecodeTiled=FakeVAEDecode)
        models = NS(
            unet="unet-model",
            vae=NS(device=NS(type="cpu")),
        )
        samples = {"samples": NS(is_nested=False, ndim=4)}
        with (
            patch.dict(
                sys.modules,
                {
                    "comfy": NS(model_management=memory),
                    "comfy.model_management": memory,
                    "nodes": fake_nodes,
                },
            ),
            patch.object(native_step, "end_native_step", lambda: order.append("end_native_step")),
        ):
            result = vae_decode.decode_image_vae(models, samples)
        self.assertEqual(result, "decoded")
        self.assertEqual(order, ["end_native_step", ("unload", "unet-model"), "soft_empty_cache"])

    def test_reset_image_models_clears_powerup_and_depth_caches(self):
        from ruinedfooocus.features.generation import image_model_lifecycle

        owner = NS(
            depth_model="da3",
            depth_model_signature=("sig", "fp16"),
            depth_model_name="depth_anything_3_base.safetensors",
            _lotus_depth=("sig", "lotus", "vae"),
            native_model_patch_cache=("id", ("patch", "vision")),
        )
        image_model_lifecycle.reset_image_models(owner)
        for attr in (
            "depth_model",
            "depth_model_signature",
            "depth_model_name",
            "_lotus_depth",
            "native_model_patch_cache",
        ):
            self.assertIsNone(getattr(owner, attr))

    def test_qwen_prompt_enhancement_runs_cleanup_before_unloading(self):
        source = (ROOT / "ruinedfooocus/features/prompting/qwen_image21_enhancement.py").read_text(
            encoding="utf-8"
        )
        block = source[source.index("            finally:") :]
        self.assertIn("end_native_step()", block)
        self.assertLess(block.index("end_native_step()"), block.index("unload_all_models()"))


if __name__ == "__main__":
    unittest.main()
