"""Official Lotus depth node ordering, source capture and native model guards."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from ruinedfooocus.features.image_editing import lotus_depth as lotus
from ruinedfooocus.features.image_editing.depth_control import (
    DEPTH_DEFAULTS,
    depth_settings,
    prepare_depth,
)
from ruinedfooocus.features.generation.image_preparation import capture_image_assets, KEY
from test_image_preparation import Manager
from test_native_variant_defaults import write_header, native_backend, SHAPES, NATIVE
from test_native_model_coverage import native_method


class LotusDepthTests(unittest.TestCase):
    def test_pinned_node_signatures_and_first_sigma_preserve_native_tail(self):
        tree = ast.parse(
            (NATIVE / "comfy_extras/nodes_custom_sampler.py").read_text(encoding="utf-8")
        )
        expected = {
            "BasicScheduler": ["cls", "model", "scheduler", "steps", "denoise"],
            "SetFirstSigma": ["cls", "sigmas", "sigma"],
            "BasicGuider": ["cls", "model", "conditioning"],
            "DisableNoise": ["cls"],
            "SamplerCustomAdvanced": [
                "cls",
                "noise",
                "guider",
                "sampler",
                "sigmas",
                "latent_image",
            ],
        }
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in expected:
                execute = next(
                    n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "execute"
                )
                self.assertEqual([arg.arg for arg in execute.args.args], expected.pop(node.name))
        self.assertFalse(expected)
        native = native_method(
            "comfy_extras/nodes_custom_sampler.py",
            "SetFirstSigma",
            "execute",
            dict(torch=torch, io=NS(NodeOutput=lambda *items: items)),
        )
        original = torch.tensor([14.0, 0.0])
        result = native.execute(original, 999.0)[0]
        self.assertEqual(result.tolist(), [999.0, 0.0])
        self.assertEqual(original.tolist(), [14.0, 0.0])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manager = Manager(self.root)
        for category in ("checkpoints", "vae", "controlnet"):
            (self.root / category).mkdir()
        self.model = write_header(
            self.root / "checkpoints" / "renamed.safetensors", {"unknown": [1]}
        )
        self.vae = write_header(
            self.root / "vae" / "codec.safetensors",
            {
                "decoder.conv_in.weight": [512, 4, 3, 3],
                "encoder.conv_in.weight": [128, 3, 3, 3],
                "decoder.conv_out.weight": [3, 128, 3, 3],
                "encoder.down.2.downsample.conv.weight": [512, 512, 3, 3],
                "decoder.up.3.upsample.conv.weight": [512, 512, 3, 3],
            },
        )
        (self.root / "controlnet" / "control-lora-depth-rank128.safetensors").write_bytes(b"local")
        self.options = dict(
            DEPTH_DEFAULTS,
            depth_model="lotus",
            depth_lotus_model=self.model.name,
            depth_lotus_vae=self.vae.name,
        )
        self.image = torch.zeros(1, 16, 24, 3)

    def capture(self):
        return capture_image_assets(
            dict(input_image="source"), self.manager, dict(type="depth", **self.options)
        )

    def test_settings_and_api_require_explicit_components_without_changing_old_maps(self):
        from ruinedfooocus.integrations.openai_api import ControlNetRequest

        self.assertEqual(depth_settings({})["depth_preprocess"], "none")
        self.assertEqual(depth_settings(self.options), self.options)
        request = ControlNetRequest(type="depth", **self.options)
        self.assertEqual(request.depth_lotus_model, self.model.name)
        with self.assertRaisesRegex(ValueError, "Select both"):
            depth_settings(dict(self.options, depth_lotus_vae=""))
        with self.assertRaises(ValueError):
            depth_settings(dict(self.options, depth_lotus_model=True))

    def test_native_classifier_accepts_lotus_and_rejects_ordinary_sd2_same_name(self):
        shapes = {
            "input_blocks.0.0.weight": [320, 4, 3, 3],
            "input_blocks.1.0.in_layers.0.weight": [320],
            "input_blocks.1.0.out_layers.3.weight": [320, 320, 3, 3],
            "input_blocks.1.1.proj_in.weight": [320, 320],
            "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": [320, 1024],
            "label_emb.0.0.weight": [1280, 4],
            "out.2.weight": [4, 320, 3, 3],
        }
        with patch.dict(SHAPES, LotusD=shapes), native_backend():
            from comfy import model_detection

            # Existing shared fixture covers DiTs; this UNet additionally needs
            # the pin's transformer-depth helper, without executing backend init.
            tree = ast.parse((NATIVE / "comfy/model_detection.py").read_text(encoding="utf-8"))
            helper = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "calculate_transformer_depth"
            )
            exec(
                compile(ast.Module(body=[helper], type_ignores=[]), str(NATIVE), "exec"),
                model_detection.detect_unet_config.__globals__,
            )
            write_header(self.model, shapes)
            lotus.validate_model(self.model)
            shapes.pop("label_emb.0.0.weight")
            write_header(self.model, shapes)
            with self.assertLogs(level="ERROR"), self.assertRaisesRegex(ValueError, "LotusD"):
                lotus.validate_model(self.model)
        lotus.validate_vae(self.vae)
        write_header(self.vae, {"decoder.conv_in.weight": [512, 16, 3, 3]})
        with self.assertRaisesRegex(ValueError, "compatible SD VAE"):
            lotus.validate_vae(self.vae)

    def test_capture_freezes_both_local_components_and_rejects_missing_or_wrong_codec(self):
        with patch.object(lotus, "validate_model"):
            data = self.capture()
            self.assertEqual(
                [a.role for a in data[KEY].artifacts], ["controlnet", "lotus-depth", "lotus-vae"]
            )
            self.assertIsNone(data[KEY].depth_name)
            self.vae.unlink()
            with self.assertRaises(FileNotFoundError):
                self.capture()
            write_header(self.vae, {"unknown": [1]})
            with self.assertRaisesRegex(ValueError, "compatible SD VAE"):
                self.capture()

    def test_actual_depth_dispatch_uses_lotus_not_da3(self):
        with (
            patch.dict("sys.modules", {"ruinedfooocus.app.state": NS(path_manager=self.manager)}),
            patch.object(lotus, "prepare_lotus", return_value=self.image) as call,
        ):
            self.assertIs(prepare_depth(NS(), self.image, self.options, {}), self.image)
            self.assertIs(call.call_args.args[-1], self.manager)

    def test_local_roots_include_diffusion_models_and_reject_absolute_traversal_and_escape(self):
        root = self.root / "diffusion_models"
        root.mkdir()
        second = write_header(root / "other.safetensors", {"unknown": [1]})
        with patch.object(lotus, "validate_model"):
            choices = lotus.local_models(self.manager)
        self.assertEqual(
            choices,
            {self.model.name: str(self.model.resolve()), second.name: str(second.resolve())},
        )
        self.assertEqual(lotus.selected_file(self.manager, second.name, "model"), second.resolve())
        for name in (str(self.model.resolve()), "../outside.safetensors"):
            with self.assertRaisesRegex(ValueError, "configured local folders"):
                lotus.selected_file(self.manager, name, "model")
        original = Path.resolve
        canonical = self.model.resolve()
        with patch.object(
            Path,
            "resolve",
            lambda path: (
                self.root / "outside.safetensors" if original(path) == canonical else original(path)
            ),
        ):
            with self.assertRaisesRegex(ValueError, "outside its configured folder"):
                lotus.selected_file(self.manager, self.model.name, "model")
        write_header(root / self.model.name, {"other": [1]})

        def valid(path):
            if path == self.model.resolve():
                raise ValueError("first root has incompatible same name")

        with patch.object(lotus, "validate_model", side_effect=valid):
            self.assertNotIn(self.model.name, lotus.local_models(self.manager))

    def test_other_engine_cannot_silently_ignore_selected_lotus(self):
        with self.assertRaisesRegex(ValueError, "Lotus depth preprocessing requires"):
            capture_image_assets(
                dict(input_image="source", image_engine="sdcpp-cpu"),
                self.manager,
                dict(type="depth", **self.options),
            )

    def test_native_recipe_cache_and_late_replacement_guard(self):
        calls = []

        def node(name, result):
            return NS(
                execute=Mock(side_effect=lambda *args: (calls.append((name, args)), [result])[1])
            )

        model = NS(model=NS(model_config=type("LotusD", (), {})()))
        vae = NS(
            throw_exception_if_invalid=Mock(),
            patcher=object(),
            latent_channels=4,
            downscale_ratio=8,
        )
        sd = NS(load_diffusion_model=Mock(return_value=model), VAE=Mock(return_value=vae))
        utils = NS(load_torch_file=Mock(return_value=({}, {})))
        cancel = Mock()
        memory = NS(
            throw_exception_if_processing_interrupted=cancel, unload_model_and_clones=Mock()
        )
        sampler = NS(
            **{
                name: node(name, value)
                for name, value in {
                    "BasicScheduler": "initial-sigmas",
                    "SetFirstSigma": "fixed-sigmas",
                    "KSamplerSelect": "euler",
                    "BasicGuider": "guider",
                    "DisableNoise": "no-noise",
                    "SamplerCustomAdvanced": "sampled-latent",
                }.items()
            }
        )
        nodes = NS(
            VAEEncode=lambda: NS(encode=Mock(return_value=["encoded"])),
            VAEDecode=lambda: NS(decode=Mock(return_value=[torch.ones_like(self.image)])),
            ImageInvert=lambda: NS(invert=lambda pixels: [1 - pixels]),
        )
        modules = {
            "comfy": NS(sd=sd, utils=utils),
            "comfy.sd": sd,
            "comfy.utils": utils,
            "comfy.model_management": memory,
            "nodes": nodes,
            "comfy_extras.nodes_lotus": NS(
                LotusConditioning=node("LotusConditioning", "condition")
            ),
            "comfy_extras.nodes_custom_sampler": sampler,
        }
        owner = NS()
        with patch.dict("sys.modules", modules), patch.object(lotus, "validate_model"):
            data = self.capture()
            result = lotus.prepare_lotus(owner, self.image, self.options, data, self.manager)
            self.assertTrue(torch.equal(result, self.image))
            lotus.prepare_lotus(owner, self.image, self.options, data, self.manager)
            sd.load_diffusion_model.assert_called_once()
            sampler.BasicScheduler.execute.assert_called_with(model, "normal", 1, 1.0)
            sampler.SetFirstSigma.execute.assert_called_with("initial-sigmas", 999.0)
            sampler.SamplerCustomAdvanced.execute.assert_called_with(
                "no-noise", "guider", "euler", "fixed-sigmas", "encoded"
            )
            state = self.model.stat()
            os.utime(self.model, ns=(state.st_atime_ns, state.st_mtime_ns + 1000000))
            with self.assertRaisesRegex(ValueError, "changed after submission"):
                lotus.prepare_lotus(owner, self.image, self.options, data, self.manager)
            self.assertEqual(sd.load_diffusion_model.call_count, 1)
            data = self.capture()
            lotus.prepare_lotus(owner, self.image, self.options, data, self.manager)
            self.assertEqual(sd.load_diffusion_model.call_count, 2)
            self.assertEqual(memory.unload_model_and_clones.call_count, 2)
            cancel.side_effect = ValueError("cancelled")
            with self.assertRaisesRegex(ValueError, "cancelled"):
                lotus.prepare_lotus(owner, self.image, self.options, data, self.manager)


if __name__ == "__main__":
    unittest.main()
