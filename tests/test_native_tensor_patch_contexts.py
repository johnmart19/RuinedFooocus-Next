"""Real native patch rebuilds coexist with bounded numeric workflow files."""

import ast
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from ruinedfooocus.features.generation import native_tensor_io as native
from ruinedfooocus.features.generation import native_tensor_context as context
from ruinedfooocus.features.generation import image_preparation as prep
from ruinedfooocus.features.image_editing import gligen_layout as gligen
from ruinedfooocus.features.image_editing import native_hypernetwork as hyper
from ruinedfooocus.features.image_editing import lora_hooks
import test_native_tensor_context as workflow_fixtures
import test_native_auxiliary_guidance as guidance_fixtures
import test_native_lora_hooks as hook_fixtures
import test_api_native_inpaint as api_fixtures
from test_native_variant_defaults import NATIVE


class NumericPatchContextTests(unittest.TestCase):
    def setUp(self):
        workflow_fixtures.SavedNativeWorkflowTests.setUp(self)

    def fixture(self, cls):
        value = cls(methodName="runTest")
        value.setUp()
        self.addCleanup(value.doCleanups)
        return value

    def guidance(self):
        files = self.fixture(guidance_fixtures.AuxiliaryTests)
        owner, backend, clip = files.runtime()
        native_nodes = self.modules["nodes"]
        native_nodes.GLIGENTextBoxApply = sys.modules["nodes"].GLIGENTextBoxApply
        self.enterContext(patch.dict(sys.modules, {"nodes": native_nodes}))
        vae = self.owner.xl_base_patched.vae
        vae.downscale_ratio = vae.upscale_ratio = 8
        vae.output_channels = 3
        owner.xl_base_patched.vae = vae
        owner.xl_base_patched.unet.model = NS(
            model_config=type("SD15", (), {"unet_config": {"in_channels": 4}})()
        )
        return files, owner, backend, clip

    def capture_guidance(self, files, **values):
        data = (
            files.data
            | dict(
                base_model_name="owned-model.safetensors",
                base_model_hash="a" * 64,
                native_tensor_family="auto",
            )
            | values
        )
        data, first = gligen.capture_assets(data, files.manager, self.models, "BaseModel")
        data, second = hyper.capture_assets(data, files.manager, self.models, "BaseModel")
        data[prep.KEY] = prep.ImageAssets(first + second, True)
        return native.capture_request(data, self.manager, self.models)

    def test_hypernetwork_callback_is_rebuilt_and_numeric_conditioning_roundtrips(self):
        files, owner, _, _ = self.guidance()
        data = self.capture_guidance(
            files, gligen_model=None, native_export_latent=True, native_export_conditioning=True
        )
        base = owner.xl_base_patched.unet
        patched = hyper.patch(owner, base, data, files.manager, lambda: None)
        callbacks = patched.model_options["transformer_options"]["patches"]
        self.assertNotIn("patches", base.model_options["transformer_options"])
        q, k, v = (torch.ones(1, 2, 64) * n for n in (1, 2, 3))
        actual = callbacks["attn2_patch"][0](q, k, v, {})
        self.assertTrue(torch.equal(actual[1], torch.zeros_like(k)))
        self.assertTrue(torch.equal(actual[2], v * -0.5))
        self.assertEqual(data["hypernetwork_guidance"]["sampling_target"], "primary-model")
        condition = [[torch.ones(1, 2, 768), {}]]
        outputs = native.export_tensors(
            owner, data, {"samples": torch.zeros(1, 4, 6, 8)}, condition, condition
        )
        raw = {item.kind: Path(item.path).read_bytes() for item in outputs}
        replay = self.capture_guidance(
            files,
            gligen_model=None,
            native_latent_input=raw["latent"],
            native_positive_input=raw["positive"],
        )
        restored = native.latent_input(owner, replay)
        positive, _ = native.apply_conditioning_inputs(owner, replay, [], [])
        self.assertEqual(tuple(restored["samples"].shape), (1, 4, 6, 8))
        self.assertEqual(tuple(positive[0][0].shape), (1, 2, 768))
        self.assertNotIn("hooks", positive[0][1])
        with self.assertRaisesRegex(ValueError, "original saved"):
            self.capture_guidance(
                files,
                gligen_model=None,
                hypernetwork_strength=0.25,
                native_latent_input=raw["latent"],
            )

    def test_gligen_only_rebuilds_live_layout_for_latent_files_and_refuses_object_conditioning(
        self,
    ):
        files, owner, backend, clip = self.guidance()
        data = self.capture_guidance(files, hypernetwork_model=None, native_export_latent=True)
        base = [[torch.ones(1, 2, 64), {}]]
        positive = gligen.apply(owner, base, data, files.manager, lambda: None)
        self.assertIn("gligen", positive[0][1])
        output = native.export_tensors(
            owner, data, {"samples": torch.zeros(1, 4, 6, 8)}, positive, base
        )[0]
        raw = Path(output.path).read_bytes()
        self.assertNotIn(b"CoreModelPatcher", raw)
        replay = self.capture_guidance(files, hypernetwork_model=None, native_latent_input=raw)
        native.latent_input(owner, replay)
        rebuilt = gligen.apply(owner, base, replay, files.manager, lambda: None)
        self.assertEqual(len(rebuilt[0][1]["gligen"][2]), len(files.boxes))
        for values in (
            dict(native_export_conditioning=True),
            dict(native_positive_input="untrusted"),
        ):
            with (
                self.subTest(values=values),
                self.assertRaisesRegex(ValueError, "executable position model"),
            ):
                self.capture_guidance(files, hypernetwork_model=None, **values)
        self.assertTrue(clip.tokenize.called)
        self.assertTrue(backend.sd.load_gligen.called)

    def test_real_asgi_capture_admits_numeric_hypernetwork_and_latent_only_gligen_cold(self):
        files = self.fixture(guidance_fixtures.AuxiliaryTests)
        api = self.fixture(api_fixtures.NativeInpaintApiTests)
        state = sys.modules["ruinedfooocus.app.state"]
        state.path_manager.get_model_folders = lambda category: [files.root]
        for fields in (
            dict(hypernetwork_model=files.hyper_file.name, native_export_conditioning=True),
            dict(
                gligen_model=files.gligen_file.name,
                gligen_layout=json.dumps(files.boxes),
                native_export_latent=True,
            ),
        ):
            with self.subTest(fields=fields):
                response = api.client.post(
                    "/images/generations",
                    json=dict(prompt="actual request", size="512x512", **fields),
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(api.prepared[-1]["native_tensor_workflow"])
                self.assertTrue(api.prepared[-1][native.KEY].context)
        refused = api.client.post(
            "/images/generations",
            json=dict(
                prompt="actual request",
                size="512x512",
                gligen_model=files.gligen_file.name,
                native_export_conditioning=True,
            ),
        )
        self.assertEqual(refused.status_code, 400)
        self.assertIn("executable position model", refused.text)

    def test_hypernetwork_staged_sampling_patches_only_primary_and_keeps_refiner_schedule(self):
        from test_sdxl_handoff import SDXLHandoffTests

        files, attention_owner, _, _ = self.guidance()
        harness = self.fixture(SDXLHandoffTests)
        original_base = harness.base
        original_base.model.model_config = type("SDXL", (), {"unet_config": {"in_channels": 4}})()

        def configure(value):
            value.model, value.load_device = original_base.model, "cpu"
            value.get_model_object = lambda name: (
                value.diffusion
                if name == "diffusion_model"
                else original_base.get_model_object(name)
            )
            value.clone = lambda: configure(guidance_fixtures.Model.clone(value))
            return value

        base = configure(attention_owner.xl_base_patched.unet)
        harness.owner.xl_base_patched.unet = base
        harness.owner.xl_base_patched.vae = attention_owner.xl_base_patched.vae
        data = harness.captured() | dict(
            hypernetwork_model=files.hyper_file.name,
            hypernetwork_strength=-0.5,
            native_export_latent=True,
        )
        data, extra = hyper.capture_assets(data, files.manager, harness.models, "SDXL")
        data[prep.KEY] = prep.ImageAssets((*data[prep.KEY].artifacts, *extra), True)
        with patch(
            "ruinedfooocus.features.models.model_architecture.inspect_model_file",
            return_value={"native_class": "SDXL"},
        ):
            data = native.capture_request(data, harness.manager, harness.models)
        patched = hyper.patch(harness.owner, base, data, files.manager, Mock())
        harness.base = patched
        output = harness.sample(data)
        self.assertEqual(len(harness.records), 2)
        self.assertIn(
            "attn1_patch",
            harness.records[0]["model"].model_options["transformer_options"]["patches"],
        )
        self.assertNotIn("patches", harness.records[1]["model"].model_options)
        self.assertEqual(harness.records[1]["sigmas"].tolist(), [0.5, 0.25, 0.0])
        nodes = self.modules["nodes"]
        with patch.dict(sys.modules, {"nodes": nodes}):
            artifact = native.export_tensors(harness.owner, data, {"samples": output}, None, None)[
                0
            ]
        self.assertEqual(artifact.family, "SDXL")
        self.assertEqual(data["hypernetwork_guidance"]["sampling_target"], "primary-model")

    def clip_encoder(self, clip):
        tree = ast.parse((NATIVE / "comfy/sd.py").read_text(encoding="utf-8"))
        native_class = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CLIP"
        )
        methods = [
            n
            for n in native_class.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("encode_from_tokens_scheduled", "add_hooks_to_dict")
        ]
        scope = dict(
            model_management=NS(
                cuda_device_context=lambda *_: nullcontext(),
                throw_exception_if_processing_interrupted=Mock(),
            ),
            ProgressBar=lambda *_: NS(update=Mock()),
        )
        exec(
            compile(
                ast.Module(body=methods, type_ignores=[]), "pinned numeric scheduled CLIP", "exec"
            ),
            scope,
        )
        clip.layer_idx = None
        clip.patcher.load_device = "cpu"
        clip.patcher.patch_hooks = Mock()
        clip.cond_stage_model = NS(
            reset_clip_options=Mock(),
            set_clip_options=Mock(),
            encode_token_weights=lambda *_: (
                torch.ones(1, 2, 2048, dtype=torch.float16),
                torch.ones(1, 1280, dtype=torch.bfloat16),
            ),
        )
        clip.load_model = Mock()
        clip.apply_hooks_to_conds = None
        clip.add_hooks_to_dict = MethodType(scope["add_hooks_to_dict"], clip)
        return lambda: scope["encode_from_tokens_scheduled"](
            clip, {"literal": [1]}, add_dict={"width": 64}, show_pbar=False
        )

    def test_native_clip_only_hook_intervals_remain_numeric_with_exact_source_context(self):
        files = self.fixture(hook_fixtures.NativeLoraHookTests)
        files.manager.model_paths = {"cache_path": files.root}
        data = files.capture(
            lora_hook_mode="interpolated",
            lora_hook_model_strength=0,
            lora_hook_clip_strength=0.4,
            lora_hook_schedule_clip=True,
            lora_hook_start=0.2,
            lora_hook_end=0.8,
            lora_hook_keyframe_count=3,
        )
        owner, original = files.owner()
        original.vae = self.owner.xl_base_patched.vae
        original.unet.model = NS(
            model_config=type("SDXL", (), {"unet_config": {"in_channels": 4}})()
        )
        values = dict(
            width=64,
            height=48,
            base_model_hash="a" * 64,
            native_tensor_family="auto",
            native_export_conditioning=True,
        )
        with patch(
            "ruinedfooocus.features.models.model_architecture.inspect_model_file",
            return_value={"native_class": "SDXL"},
        ):
            data = native.capture_request(data | values, files.manager, files.models)
            with lora_hooks.request_scope(owner, data, files.manager, Mock()):
                encoded = self.clip_encoder(owner.xl_base_patched.clip)()
                self.assertIsNone(owner._request_lora_hooks)
                self.assertFalse(any("hooks" in row[1] for row in encoded))
                output = native.export_tensors(
                    owner, data, {"samples": torch.zeros(1, 4, 6, 8)}, encoded, encoded
                )[0]
            self.assertGreater(len(encoded), 1)
            raw = Path(output.path).read_bytes()
            replay = files.capture(
                lora_hook_mode="interpolated",
                lora_hook_model_strength=0,
                lora_hook_clip_strength=0.4,
                lora_hook_schedule_clip=True,
                lora_hook_start=0.2,
                lora_hook_end=0.8,
                lora_hook_keyframe_count=3,
            )
            replay = native.capture_request(
                replay | values | dict(native_positive_input=raw), files.manager, files.models
            )
            with lora_hooks.request_scope(owner, replay, files.manager, Mock()):
                self.clip_encoder(owner.xl_base_patched.clip)
                actual, _ = native.apply_conditioning_inputs(owner, replay, [], [])
                self.assertEqual(
                    [(row[1]["clip_start_percent"], row[1]["clip_end_percent"]) for row in actual],
                    [(row[1]["clip_start_percent"], row[1]["clip_end_percent"]) for row in encoded],
                )
                self.assertEqual(actual[0][0].dtype, torch.float16)
                self.assertEqual(actual[0][1]["pooled_output"].dtype, torch.bfloat16)
                self.assertIs(lora_hooks.conditioning(owner, actual, [])[0], actual)

    def test_model_hook_objects_remain_refused_while_latent_receipts_bind_resolved_sources(self):
        files = self.fixture(hook_fixtures.NativeLoraHookTests)
        files.manager.model_paths = {"cache_path": files.root}
        data = files.capture()
        with patch(
            "ruinedfooocus.features.models.model_architecture.inspect_model_file",
            return_value={"native_class": "SDXL"},
        ):
            with self.assertRaisesRegex(ValueError, "executable GLIGEN or hook"):
                native.capture_request(
                    data | dict(native_export_conditioning=True), files.manager, files.models
                )
            captured = native.capture_request(
                data | dict(width=64, height=48, native_export_latent=True),
                files.manager,
                files.models,
            )
        saved = json.loads(captured[native.KEY].context)["settings"]["resolved_hook_sources"][0]
        self.assertEqual(saved["settings"]["lora_hook_model_strength"], 1.0)
        self.assertNotIn(str(files.root), json.dumps(saved))
        stale = context.capture_context(
            dict(
                base_model_name="checkpoint.safetensors",
                lora_hook_mode="off",
                lora_hook_sources="invalid stale source",
            ),
            files.models,
        )
        self.assertNotIn("resolved_hook_sources", stale["settings"])
        owner, original = files.owner()
        original.vae = self.owner.xl_base_patched.vae
        original.unet.model = NS(
            model_config=type("SDXL", (), {"unet_config": {"in_channels": 4}})()
        )
        with lora_hooks.request_scope(owner, captured, files.manager, Mock()):
            condition = [[torch.ones(1, 2, 2048), {"pooled_output": torch.ones(1, 1280)}]]
            hooked, negative = lora_hooks.conditioning(owner, condition, condition)
            self.assertIn("hooks", hooked[0][1])
            original_group = owner._request_lora_hooks
            output = native.export_tensors(
                owner, captured, {"samples": torch.zeros(1, 4, 6, 8)}, hooked, negative
            )[0]
        raw = Path(output.path).read_bytes()
        self.assertNotIn(b"HookGroup", raw)
        replay = files.capture()
        with patch(
            "ruinedfooocus.features.models.model_architecture.inspect_model_file",
            return_value={"native_class": "SDXL"},
        ):
            replay = native.capture_request(
                replay | dict(native_latent_input=raw), files.manager, files.models
            )
        with lora_hooks.request_scope(owner, replay, files.manager, Mock()):
            loaded = native.latent_input(owner, replay)
            self.assertEqual(tuple(loaded["samples"].shape), (1, 4, 6, 8))
            self.assertIsNot(owner._request_lora_hooks, original_group)

    def test_runtime_identity_keeps_loaded_stamps_and_disambiguates_equal_basename_roots(self):
        component = self.root / "encoder.safetensors"
        component.write_bytes(b"original loaded component")
        loaded = context.file_signature(component)
        self.owner.component_file_signatures = [loaded]
        self.owner.xl_base_patched_hash = [(loaded, 0.5)]
        data = native.capture_request(
            dict(
                base_model_name="owned-model.safetensors",
                native_tensor_family="auto",
                native_tensor_workflow=True,
                native_export_latent=True,
                width=24,
                height=16,
            ),
            self.manager,
            self.models,
        )
        before = context.runtime_context(self.owner, data)
        component.write_bytes(b"changed disk file is not the loaded model")
        self.assertEqual(context.runtime_context(self.owner, data), before)
        self.owner.component_file_signatures = [context.file_signature(component)]
        self.owner.xl_base_patched_hash = [(context.file_signature(component), 0.5)]
        after = context.runtime_context(self.owner, data)
        self.assertNotEqual(after["components"], before["components"])
        self.assertNotEqual(after["loras"], before["loras"])
        sibling = (self.root / "other" / component.name).resolve()
        other = [str(sibling), *loaded[1:]]
        self.assertEqual(context._signature_stamp(loaded)[:4], context._signature_stamp(other)[:4])
        self.assertNotEqual(context._signature_stamp(loaded)[4], context._signature_stamp(other)[4])


if __name__ == "__main__":
    unittest.main()
