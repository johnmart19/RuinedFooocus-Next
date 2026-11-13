"""Device routing, memory cleanup, matching processors and serial UI jobs."""

from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import torch
import ruinedfooocus.features.image_processing.caption_runtime as runtime
import ruinedfooocus.features.image_processing.interrogate as interrogate


class CaptionRuntimeTests(unittest.TestCase):
    def backend(self, device="cuda", directml=False):
        management = SimpleNamespace(
            is_directml_enabled=lambda: directml,
            get_torch_device=lambda: torch.device(device),
            free_memory=Mock(),
        )
        return management, patch.dict(
            "sys.modules", {"comfy": SimpleNamespace(model_management=management)}
        )

    def test_gpu_inference_reserves_memory_and_offloads_after_result(self):
        memory, scope = self.backend()
        model = Mock(parameters=lambda: [torch.ones(2, 2)])

        def infer(device):
            self.assertTrue(torch.is_inference_mode_enabled())
            return str(device)

        with scope:
            self.assertEqual(runtime.run_caption(model, infer), "cuda")
        memory.free_memory.assert_called_once()
        model.to.assert_any_call(torch.device("cuda"))
        self.assertEqual(model.to.call_args.args, ("cpu",))

    def test_generation_oom_retries_once_on_cpu_and_offloads(self):
        _, scope = self.backend()
        model = Mock(parameters=lambda: [])
        infer = Mock(side_effect=[torch.OutOfMemoryError("GPU full"), "CPU caption"])
        with scope:
            self.assertEqual(runtime.run_caption(model, infer), "CPU caption")
        self.assertEqual([call.args[0].type for call in infer.call_args_list], ["cuda", "cpu"])
        self.assertEqual(model.to.call_args.args, ("cpu",))

    def test_cpu_and_directml_use_cpu_without_gpu_memory_calls(self):
        for device, directml in [("cpu", False), ("privateuseone", True)]:
            with self.subTest(device=device):
                memory, scope = self.backend(device, directml)
                model = Mock()
                with scope:
                    result = runtime.run_caption(model, lambda actual: actual.type)
                self.assertEqual(result, "cpu")
                memory.free_memory.assert_not_called()

    def test_other_errors_propagate_and_always_offload(self):
        _, scope = self.backend()
        model = Mock(parameters=lambda: [])
        with scope, self.assertRaisesRegex(ValueError, "bad input"):
            runtime.run_caption(model, Mock(side_effect=ValueError("bad input")))
        self.assertEqual(model.to.call_args.args, ("cpu",))

    def test_florence_processor_matches_native_checkpoint_and_is_cached_on_cpu(self):
        interrogate.caption_model.cache_clear()
        self.addCleanup(interrogate.caption_model.cache_clear)
        api = SimpleNamespace(
            AutoProcessor=Mock(),
            BlipForConditionalGeneration=Mock(),
            Florence2ForConditionalGeneration=Mock(),
        )
        with patch.dict(
            "sys.modules",
            {
                "transformers": api,
                "ruinedfooocus.app.arguments": SimpleNamespace(args=SimpleNamespace(offline=True)),
            },
        ):
            first = interrogate.caption_model("florence")
            self.assertIs(first, interrogate.caption_model("florence"))
        api.AutoProcessor.from_pretrained.assert_called_once_with(
            "florence-community/Florence-2-base", local_files_only=True
        )
        api.Florence2ForConditionalGeneration.from_pretrained.assert_called_once_with(
            "florence-community/Florence-2-base", dtype=torch.float32, local_files_only=True
        )

    def test_clip_uses_cpu_for_directml_and_releases_both_models(self):
        memory, scope = self.backend("privateuseone", True)
        config = Mock(return_value=SimpleNamespace(apply_low_vram_defaults=Mock()))
        model = SimpleNamespace(
            interrogate=Mock(return_value="CLIP caption"), caption_model=Mock(), clip_model=Mock()
        )
        api = SimpleNamespace(Config=config, Interrogator=Mock(return_value=model))
        shared = SimpleNamespace(
            path_manager=SimpleNamespace(model_paths={"cache_path": "caption-cache"})
        )
        with (
            scope,
            patch.dict(
                "sys.modules", {"clip_interrogator": api, "ruinedfooocus.app.state": shared}
            ),
        ):
            self.assertEqual(interrogate.clip_look(Mock(), "", None), "CLIP caption")
        self.assertEqual(config.call_args.kwargs["device"].type, "cpu")
        self.assertEqual(
            config.call_args.kwargs["cache_path"], str(Path("caption-cache") / "clip-interrogator")
        )
        model.caption_model.to.assert_called_once_with("cpu")
        model.clip_model.to.assert_called_once_with("cpu")
        memory.free_memory.assert_not_called()

    def test_clip_gpu_oom_retries_on_cpu_after_releasing_models(self):
        memory, scope = self.backend()
        failed = SimpleNamespace(
            interrogate=Mock(side_effect=torch.OutOfMemoryError("full")),
            caption_model=Mock(),
            clip_model=Mock(),
        )
        recovered = SimpleNamespace(
            interrogate=Mock(return_value="CPU CLIP caption"),
            caption_model=Mock(),
            clip_model=Mock(),
        )
        config = Mock(side_effect=lambda **kw: SimpleNamespace(apply_low_vram_defaults=Mock()))
        api = SimpleNamespace(Config=config, Interrogator=Mock(side_effect=[failed, recovered]))
        shared = SimpleNamespace(
            path_manager=SimpleNamespace(model_paths={"cache_path": "caption-cache"})
        )
        with (
            scope,
            patch.dict(
                "sys.modules", {"clip_interrogator": api, "ruinedfooocus.app.state": shared}
            ),
        ):
            self.assertEqual(interrogate.clip_look(Mock(), "", None), "CPU CLIP caption")
        self.assertEqual(
            [call.kwargs["device"].type for call in config.call_args_list], ["cuda", "cpu"]
        )
        failed.caption_model.to.assert_called_once_with("cpu")
        failed.clip_model.to.assert_called_once_with("cpu")
        memory.free_memory.assert_called_once()

    def test_ui_caption_is_queued_but_metadata_and_ordinary_prompts_are_immediate(self):
        api = SimpleNamespace(
            submit=Mock(return_value="job"), wait_for_result=Mock(return_value="caption")
        )
        image = SimpleNamespace(info={"parameters": '{"Prompt": "fox"}'})
        gr = SimpleNamespace(Info=Mock())
        captioner = Mock(return_value="worker caption")
        with (
            patch.dict("sys.modules", {"ruinedfooocus.app.api_runtime": api}),
            patch.dict(interrogate.looks, {"florence": captioner}),
        ):
            self.assertEqual(interrogate.look(image, "florence:", gr), "caption")
            captioner.assert_not_called()
            operation = api.submit.call_args.args[0]
            self.assertEqual(operation(None, 1), "worker caption")
            self.assertEqual(interrogate.look(image, "my prompt", gr), "my prompt")
            self.assertEqual(interrogate.look(image, "", gr), "fox")
        api.submit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
