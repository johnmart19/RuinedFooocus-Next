"""Reader selection preserves actual paths and pinned native format refusals."""

from contextlib import contextmanager
import os
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.backends.comfy.text_encoder_loader import has_gguf_encoder, load_mixed_clip
import test_model_cache as cache


@contextmanager
def native_loaders(extra=None, torch_state=None, *, text_only=None):
    gguf = Mock(return_value=({"encoder": "gguf"}, extra or {}))
    torch = Mock(return_value={"encoder": "torch"} if torch_state is None else torch_state)
    patcher = Mock(return_value="native-clip-patcher")
    progress = SimpleNamespace(
        callback_for=Mock(return_value="progress-callback"), complete_file=Mock()
    )
    progress_factory = Mock(return_value=progress)
    dynamic = Mock(return_value="dynamic-clip")
    modules = {
        name: ModuleType(name)
        for name in (
            "comfy",
            "comfy.utils",
            "molbal_comfyui_gguf",
            "molbal_comfyui_gguf.loader",
            "molbal_comfyui_gguf.nodes",
        )
    }
    modules["comfy.utils"].load_torch_file = torch
    modules["molbal_comfyui_gguf.loader"].gguf_clip_loader = gguf
    modules["molbal_comfyui_gguf.nodes"].DualCLIPLoaderGGUF = Mock(
        return_value=SimpleNamespace(load_patcher=patcher)
    )
    modules["molbal_comfyui_gguf.nodes"].GGUFLoadProgress = progress_factory
    modules["molbal_comfyui_gguf.nodes"]._load_dynamic_gguf_clip = dynamic
    with (
        patch.dict("sys.modules", modules),
        patch(
            "ruinedfooocus.features.models.gguf_metadata.qwen3vl_text_only_variant",
            return_value=text_only,
        ),
    ):
        yield SimpleNamespace(
            gguf=gguf,
            torch=torch,
            patcher=patcher,
            progress=progress,
            progress_factory=progress_factory,
            dynamic=dynamic,
        )


class TextEncoderFormatTests(unittest.TestCase):
    def test_production_loader_uses_native_state_dict_or_mixed_reader_for_the_selected_container(
        self,
    ):
        for filename in ("text.PT", "text.SFT", "text.GGUF", "text.gGuF"):
            with self.subTest(filename=filename):
                fixture = cache.ModelCacheTests()
                fixture.setUp()
                self.addCleanup(fixture.doCleanups)
                checkpoint = fixture.file("checkpoint.safetensors")
                source = fixture.file(filename)
                if filename.casefold().endswith(".gguf"):
                    from test_gguf_projector_dependencies import tiny_gguf

                    tiny_gguf(source, "t5")
                fixture.file("vae.safetensors")
                load, owner, _, native, _ = fixture.base_loader(checkpoint, bare=True)
                owner.get_clip_and_vae.return_value["clip_names"] = [filename]
                with patch(
                    "ruinedfooocus.backends.comfy.text_encoder_loader.load_mixed_clip",
                    return_value=cache.Clip(),
                ) as mixed:
                    load()
                self.assertIsNone(owner.load_error)
                if filename.casefold().endswith(".gguf"):
                    mixed.assert_called_once_with([str(source)], "native", projectors=())
                    native.sd.load_clip.assert_not_called()
                else:
                    mixed.assert_not_called()
                    native.sd.load_clip.assert_called_once()
                    self.assertEqual(
                        native.sd.load_clip.call_args.kwargs["ckpt_paths"], [str(source)]
                    )

    def test_case_insensitive_gguf_is_a_reader_decision_without_renaming_files(self):
        for name in ("Qwen.GGUF", "Nested/Encoder.gGuF", "lower.gguf"):
            with self.subTest(name=name), native_loaders() as native:
                self.assertTrue(has_gguf_encoder([name]))
                self.assertEqual(load_mixed_clip([name], "native-clip-type"), "native-clip-patcher")
                native.gguf.assert_called_once_with(
                    name, progress_callback="progress-callback", return_extra=True
                )
                native.torch.assert_not_called()
                native.patcher.assert_called_once_with(
                    [name], "native-clip-type", [{"encoder": "gguf"}]
                )
                native.progress.complete_file.assert_called_once_with(name)

    def test_mixed_encoder_order_and_native_weights_only_reader_are_retained(self):
        paths = ["first.PT", "second.gGuF", "third.SFT"]
        with native_loaders() as native:
            load_mixed_clip(paths, "native-type")
            self.assertEqual(
                [call.args[0] for call in native.torch.call_args_list], [paths[0], paths[2]]
            )
            self.assertTrue(
                all(call.kwargs == {"safe_load": True} for call in native.torch.call_args_list)
            )
            self.assertEqual(
                native.patcher.call_args.args,
                (
                    paths,
                    "native-type",
                    [{"encoder": "torch"}, {"encoder": "gguf"}, {"encoder": "torch"}],
                ),
            )
            self.assertEqual(
                [call.args[0] for call in native.progress.complete_file.call_args_list], paths
            )
        self.assertFalse(
            has_gguf_encoder(["encoder.PT", "encoder.SFT", "encoder.gguf.safetensors"])
        )

    def test_qwen3vl_text_only_tensorwise_int8_retains_native_dynamic_loader_refusal(self):
        extra = {
            "arch_str": "qwen3vl",
            "qwen3vl_text_only": True,
            "gguf_quant_formats": ("int8_tensorwise",),
        }
        for name in ("qwen.gguf", "Qwen.GGUF"):
            with self.subTest(name=name), native_loaders(extra) as native:
                with self.assertRaisesRegex(ValueError, "Dynamic VRAM"):
                    load_mixed_clip([name], "QWEN_IMAGE")
                native.patcher.assert_not_called()
        with native_loaders({**extra, "qwen3vl_text_only": False}) as native:
            self.assertEqual(load_mixed_clip(["qwen.GGUF"], "QWEN_IMAGE"), "native-clip-patcher")

    def test_scaled_fp8_and_parse_errors_do_not_install_a_partial_patcher(self):
        with native_loaders(torch_state={"scaled_fp8": object()}) as native:
            with self.assertRaisesRegex(ValueError, "scaled FP8"):
                load_mixed_clip(["text.GGUF", "clip.SFT"], "FLUX")
            native.patcher.assert_not_called()
        with native_loaders() as native:
            native.gguf.side_effect = ValueError("invalid encoder architecture")
            with self.assertRaisesRegex(ValueError, "invalid encoder"):
                load_mixed_clip(["text.GGUF"], "FLUX")
            native.patcher.assert_not_called()

    def test_explicit_single_text_only_encoder_uses_native_dynamic_route_with_original_path(self):
        for name in ("qwen.gguf", "Nested/Qwen.GGUF", "Qwen.gGuF"):
            for variant in ("4b", "8b"):
                with (
                    self.subTest(name=name, variant=variant),
                    native_loaders(text_only=variant) as native,
                ):
                    self.assertEqual(load_mixed_clip([name], "FLUX"), "dynamic-clip")
                    paths, clip_type = native.dynamic.call_args.args
                    self.assertEqual(clip_type, "FLUX")
                    self.assertEqual(paths, [name])
                    self.assertEqual(os.fspath(paths[0]), name)
                    self.assertTrue(paths[0].endswith(".gguf"))
                    self.assertEqual(str(paths[0]), name)
                    self.assertEqual(native.dynamic.call_args.kwargs, {"enable_q8_native": True})
                    native.gguf.assert_not_called()
                    native.patcher.assert_not_called()
        with native_loaders(text_only="4b") as native:
            with self.assertRaisesRegex(ValueError, "by itself"):
                load_mixed_clip(["qwen.GGUF", "clip.safetensors"], "FLUX")
            native.dynamic.assert_not_called()
            native.gguf.assert_not_called()
            native.torch.assert_not_called()

    def test_dynamic_loader_refusal_is_not_retried_through_legacy_reader(self):
        with native_loaders(text_only="4b") as native:
            native.dynamic.side_effect = RuntimeError("Dynamic VRAM is not enabled")
            with self.assertRaisesRegex(RuntimeError, "Dynamic VRAM"):
                load_mixed_clip(["qwen.GGUF"], "FLUX")
            native.gguf.assert_not_called()
            native.patcher.assert_not_called()


if __name__ == "__main__":
    unittest.main()
