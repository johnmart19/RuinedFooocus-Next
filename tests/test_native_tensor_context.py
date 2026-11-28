"""Saved native workflow layouts and serializer payloads, using real node bodies."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import safetensors.torch
import torch

from ruinedfooocus.features.generation import native_tensor_io as native
from ruinedfooocus.features.generation import native_tensor_context as context
import test_native_tensor_io as fixtures


class SavedNativeWorkflowTests(unittest.TestCase):
    def setUp(self):
        fixtures.NativeTensorIOTests.setUp(self)
        (self.root / "owned-model.safetensors").write_bytes(b"owned CPU model identity fixture")

    def capture(self, **values):
        return fixtures.NativeTensorIOTests.capture(
            self, native_tensor_workflow=True, **({"native_tensor_family": "auto"} | values)
        )

    def ordinary(self, **values):
        return self.capture(width=24, height=16, **values)

    def test_multiple_safe_entries_masks_and_reference_lists_roundtrip_with_original_precision(
        self,
    ):
        source = Image.new("RGBA", (24, 16), (30, 60, 90, 127))
        options = dict(
            controlnet={"type": "edit"},
            input_image=source,
            native_export_latent=True,
            native_export_conditioning=True,
        )
        data = self.ordinary(**options)
        mask = torch.linspace(0, 1, 16 * 24).reshape(1, 16, 24)
        references = [torch.arange(24, dtype=torch.bfloat16).reshape(1, 4, 2, 3)]
        positive = [
            [
                torch.ones(1, 2, 768, dtype=torch.float16),
                dict(mask=mask, reference_latents=references, start_percent=0.0, end_percent=0.5),
            ],
            [
                torch.zeros(1, 3, 768, dtype=torch.bfloat16),
                dict(start_percent=0.5, end_percent=1.0),
            ],
        ]
        negative = [[torch.zeros(1, 2, 768), {}]]
        latent = dict(samples=torch.ones(1, 4, 2, 3, dtype=torch.float16), noise_mask=mask)
        outputs = native.export_tensors(self.owner, data, latent, positive, negative)
        raw = {artifact.kind: Path(artifact.path).read_bytes() for artifact in outputs}
        self.assertNotIn(str(self.root), raw["positive"].decode("utf-8", errors="ignore"))
        replay = self.ordinary(
            **options, native_latent_input=raw["latent"], native_positive_input=raw["positive"]
        )
        self.assertEqual(native.conditioning_inputs(self.owner, replay), {})
        loaded = native.latent_input(self.owner, replay)
        np.testing.assert_array_equal(loaded["noise_mask"].numpy(), mask.numpy())
        actual, counterpart = native.apply_conditioning_inputs(self.owner, replay, None, negative)
        self.assertIs(counterpart, negative)
        self.assertEqual(len(actual), 2)
        self.assertEqual(actual[0][0].dtype, torch.float16)
        self.assertEqual(actual[1][0].dtype, torch.bfloat16)
        self.assertEqual(actual[0][1]["reference_latents"][0].dtype, torch.bfloat16)
        self.assertEqual(actual[1][1]["start_percent"], 0.5)
        np.testing.assert_array_equal(actual[0][1]["mask"].numpy(), mask.numpy())

    def test_complex_external_or_tampered_files_need_an_original_server_owned_receipt(self):
        data = self.ordinary(native_export_latent=True)
        artifact = native.export_tensors(
            self.owner, data, {"samples": torch.zeros(1, 4, 2, 3)}, None, None
        )[0]
        raw = Path(artifact.path).read_bytes()
        receipt = context.receipt_root(data[native.KEY].directory) / (artifact.sha256 + ".json")
        receipt.unlink()
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.ordinary(native_latent_input=raw)
        with self.assertRaisesRegex(ValueError, "context"):
            self.ordinary(native_latent_input=fixtures.latent_file())

    def test_saved_source_and_companion_selection_are_explicit_runtime_boundaries(self):
        source = Image.new("RGB", (24, 16), "red")
        options = dict(input_image=source, controlnet={"type": "img2img"})
        data = self.ordinary(**options, native_export_latent=True)
        artifact = native.export_tensors(
            self.owner, data, {"samples": torch.zeros(1, 4, 2, 3)}, None, None
        )[0]
        raw = Path(artifact.path).read_bytes()
        with self.assertRaisesRegex(ValueError, "original saved"):
            self.ordinary(
                native_latent_input=raw,
                input_image=Image.new("RGB", (24, 16), "blue"),
                controlnet={"type": "img2img"},
            )
        replay = self.ordinary(**options, native_latent_input=raw)
        self.owner.xl_base_patched.vae.first_stage_model = type("DifferentCodec", (), {})()
        with self.assertRaisesRegex(ValueError, "codec"):
            native.latent_input(self.owner, replay)

    def layered_owner(self, layers=2):
        from ruinedfooocus.features.generation.layered import LayeredPlan
        from ruinedfooocus.features.generation.image_output_batches import OutputPlan, PLAN_KEY

        self.owner.xl_base_patched.unet.model.model_config = type("QwenImage", (), {})()
        self.owner.xl_base_patched.vae.latent_channels = 16
        self.owner.xl_base_patched.vae.latent_dim = 3
        plan = LayeredPlan(layers, "qwen-layers", "owned source")
        return {
            "_native_layered_plan": plan,
            "qwen_layer_count": layers,
            PLAN_KEY: OutputPlan("qwen-layers", layers, layers),
        }

    def test_layered_native_axis_is_preserved_until_native_output_plan_splits_layers(self):
        from ruinedfooocus.features.generation import layered
        from test_layered_generation import runtime as native_nodes

        options = self.layered_owner()
        with patch(
            "ruinedfooocus.features.models.model_architecture.inspect_model_file",
            return_value={"native_class": "QwenImage"},
        ):
            data = self.capture(width=32, height=16, native_export_latent=True, **options)
            sample = torch.arange(1 * 16 * 3 * 2 * 4, dtype=torch.float32).reshape(1, 16, 3, 2, 4)
            artifact = native.export_tensors(self.owner, data, {"samples": sample}, None, None)[0]
            raw = Path(artifact.path).read_bytes()
            replay = self.capture(native_latent_input=raw, **options)
            loaded = native.latent_input(self.owner, replay)
        self.assertEqual(tuple(loaded["samples"].shape), (1, 16, 3, 2, 4))
        np.testing.assert_array_equal(loaded["samples"].numpy(), sample.numpy())
        with patch.dict("sys.modules", native_nodes()):
            split = layered.layer_latent(options["_native_layered_plan"], loaded)
        self.assertEqual(tuple(split["samples"].shape), (2, 16, 1, 2, 4))
        self.assertEqual(replay["_native_image_output_plan"].kind, "qwen-layers")

    def test_refiner_single_frame_and_standalone_sdxl_layouts_are_native_not_reshaped(self):
        for family, shape, scale, rank, channels in (
            ("HunyuanImage21Refiner", (1, 32, 1, 2, 3), 16, 3, 32),
            ("SDXLRefiner", (1, 4, 2, 3), 8, 2, 4),
        ):
            with (
                self.subTest(family=family),
                patch(
                    "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                    return_value={"native_class": family},
                ),
            ):
                self.owner.xl_base_patched.unet.model.model_config = type(family, (), {})()
                vae = self.owner.xl_base_patched.vae
                vae.latent_channels, vae.latent_dim = channels, rank
                vae.spacial_compression_decode = lambda scale=scale: scale
                data = self.capture(width=3 * scale, height=2 * scale, native_export_latent=True)
                tensor = torch.zeros(shape)
                artifact = native.export_tensors(self.owner, data, {"samples": tensor}, None, None)[
                    0
                ]
                replay = self.capture(native_latent_input=Path(artifact.path).read_bytes())
                actual = native.latent_input(self.owner, replay)["samples"]
                self.assertEqual(tuple(actual.shape), shape)

    def test_executable_graph_objects_are_refused_and_numeric_exports_are_bounded_before_copy(self):
        data = self.ordinary(native_export_conditioning=True)
        for options in (
            {"hooks": object()},
            {"gligen": ("position", object(), [])},
            {"control": object()},
        ):
            with (
                self.subTest(options=list(options)),
                self.assertRaisesRegex(ValueError, "executable"),
            ):
                native.export_tensors(
                    self.owner,
                    data,
                    {"samples": torch.zeros(1, 4, 2, 3)},
                    [[torch.zeros(1, 1, 768), options]],
                    [],
                )
        self.assertEqual(data[native.EXPORT_KEY], [])
        data = self.ordinary(native_export_latent=True)
        with patch.object(
            self.modules["nodes"].SaveLatent,
            "save",
            side_effect=AssertionError("native copy reached"),
        ) as saver:
            with self.assertRaises(ValueError):
                native.export_tensors(
                    self.owner,
                    data,
                    {"samples": torch.empty(1, 4, 8192, 8192, device="meta")},
                    None,
                    None,
                )
        saver.assert_not_called()

    def test_context_mode_alone_is_off_and_new_protocol_supports_composition_gate(self):
        self.assertFalse(native.enabled({"native_tensor_workflow": True}))
        data = self.ordinary(native_export_latent=True)
        artifact = native.export_tensors(
            self.owner, data, {"samples": torch.zeros(1, 4, 2, 3)}, None, None
        )[0]
        raw = Path(artifact.path).read_bytes()
        replay = self.ordinary(native_latent_input=raw)
        companion = native.capture_latent_companion(raw, replay, self.manager, self.models)
        primary = replay[native.KEY].inputs[0]
        native.assert_composable(primary, companion)
        self.assertEqual(
            tuple(native.load_latent_item(self.owner, replay, companion)["samples"].shape),
            (1, 4, 2, 3),
        )

    def test_production_final_conditioning_hook_replaces_mask_and_references_once(self):
        data = self.ordinary(native_export_conditioning=True)
        condition = [
            [
                torch.ones(1, 2, 768),
                {"mask": torch.ones(1, 16, 24), "reference_latents": [torch.ones(1, 4, 2, 3)]},
            ]
        ]
        artifact = native.export_tensors(
            self.owner, data, {"samples": torch.zeros(1, 4, 2, 3)}, condition, condition
        )[0]
        replay = self.ordinary(native_positive_input=Path(artifact.path).read_bytes())
        tree = ast.parse(
            (fixtures.ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        process = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "process"
        )
        hook = next(
            node
            for node in process.body
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.ImportFrom)
                and any(alias.name == "apply_conditioning_inputs" for alias in child.names)
                for child in node.body
            )
        )
        original = [[torch.zeros(1, 1, 768), {"reference_latents": [torch.zeros(1, 4, 2, 3)]}]]
        scope = dict(
            self=self.owner,
            gen_data=replay,
            positive_cond=original,
            negative_conditioning=original,
            reference_frame=None,
        )
        exec(
            compile(
                ast.Module(body=[hook], type_ignores=[]),
                "actual final numeric conditioning hook",
                "exec",
            ),
            scope,
        )
        self.assertEqual(len(scope["positive_cond"][0][1]["reference_latents"]), 1)
        self.assertTrue(scope["positive_cond"][0][1]["mask"].all())
        self.assertIs(scope["negative_conditioning"], original)

    def test_production_worker_preserves_layer_output_plan_with_tensor_downloads(self):
        from ruinedfooocus.features.generation.image_output_batches import (
            ImageBatchArtifacts,
            OutputPlan,
            PLAN_KEY,
            validate_artifacts,
        )

        tree = ast.parse(
            (fixtures.ROOT / "ruinedfooocus/app/async_worker.py").read_text(encoding="utf-8")
        )
        process = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_process"
        )
        packet_start = next(
            index
            for index, node in enumerate(process.body)
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "ImageBatchArtifacts"
                for child in ast.walk(node)
            )
        )
        function = ast.FunctionDef(
            name="packet",
            args=ast.parse(
                "def packet(gen_data, res, output_plan, completed_groups, image_number, stop_batch): pass"
            )
            .body[0]
            .args,
            body=process.body[packet_start:],
            decorator_list=[],
        )
        scope = dict(ImageBatchArtifacts=ImageBatchArtifacts)
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                "actual worker output-plan packet",
                "exec",
            ),
            scope,
        )
        plan = OutputPlan("qwen-layers", 2, 2)
        data = {
            PLAN_KEY: plan,
            "image_number": 1,
            "_native_tensor_exports": ("owned tensor receipt",),
        }
        packet = scope["packet"](data, ["layer-one", "layer-two"], plan, 1, 1, False)
        self.assertIs(validate_artifacts(data, packet), packet)
        self.assertEqual(packet.tensor_files, ("owned tensor receipt",))

    def test_typed_api_allows_source_workflow_context_and_replay_tracks_its_original_bytes(self):
        from ruinedfooocus.integrations.openai_api import ImageRequest

        body = ImageRequest(
            native_tensor_workflow=True,
            native_export_latent=True,
            controlnet={"type": "edit"},
            native_tensor_family="QwenImage",
        )
        self.assertTrue(body.native_tensor_workflow)
        data = self.ordinary(native_export_latent=True)
        artifact = native.export_tensors(
            self.owner, data, {"samples": torch.zeros(1, 4, 2, 3)}, None, None
        )[0]
        raw = Path(artifact.path).read_bytes()
        replay = self.ordinary(native_latent_input=raw)
        metadata = {
            "native_tensor_source": replay["native_tensor_source"],
            "native_tensor_workflow": True,
        }
        self.assertTrue(
            native.replay_values(metadata, {"native_latent_input": raw})["native_tensor_workflow"]
        )
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), metadata["native_tensor_source"]["inputs"][0]["sha256"]
        )
        self.assertNotIn(str(self.root), json.dumps(metadata))

    def test_source_or_mask_replacement_after_capture_fails_before_reading_saved_tensors(self):
        source = Image.new("RGB", (24, 16), "red")
        mask = Image.new("RGBA", (24, 16), (0, 0, 0, 127))
        options = dict(inpaint_toggle=True, main_view=source, inpaint_view=dict(layers=[mask]))
        data = self.ordinary(native_export_latent=True, **options)
        output = native.export_tensors(
            self.owner, data, {"samples": torch.zeros(1, 4, 2, 3)}, None, None
        )[0]
        replay = self.ordinary(native_latent_input=Path(output.path).read_bytes(), **options)
        mask.putpixel((23, 15), (0, 0, 0, 0))
        with (
            patch.object(native, "_scoped", side_effect=AssertionError("native read reached")),
            self.assertRaisesRegex(ValueError, "mask changed"),
        ):
            native.latent_input(self.owner, replay)

    def test_cascade_prior_retains_exact_paired_native_stage_b_grid_and_frame(self):
        self.owner.xl_base_patched.unet.model.model_config = type("Stable_Cascade_C", (), {})()
        vae = self.owner.xl_base_patched.vae
        vae.latent_channels = 16
        vae.spacial_compression_decode = lambda: 32
        source = Image.new("RGB", (64, 48), "red")
        options = dict(
            input_image=source,
            controlnet={"type": "img2img"},
            cascade_stage_b="owned-model.safetensors",
        )
        with patch(
            "ruinedfooocus.features.models.model_architecture.inspect_model_file",
            return_value={"native_class": "Stable_Cascade_C"},
        ):
            data = self.capture(width=64, height=48, native_export_latent=True, **options)
            stage_c = {"samples": torch.zeros(1, 16, 1, 2)}
            stage_b = {"samples": torch.arange(4 * 12 * 16).reshape(1, 4, 12, 16).float()}
            frame = dict(source_width=64, source_height=48, width=64, height=48)
            output = native.export_tensors(
                self.owner, data, stage_c, None, None, stage_b=stage_b, frame=frame
            )[0]
            replay = self.capture(native_latent_input=Path(output.path).read_bytes(), **options)
            loaded = native.latent_input(self.owner, replay)
        np.testing.assert_array_equal(
            loaded["_rf_stage_b"]["samples"].numpy(), stage_b["samples"].numpy()
        )
        native.apply_conditioning_inputs(self.owner, replay, [], [], frame=frame)
        with self.assertRaisesRegex(ValueError, "frame"):
            native.apply_conditioning_inputs(self.owner, replay, [], [], frame={})

    def test_actual_sdxl_noisy_export_resumes_only_refiner_with_original_schedule_and_seed(self):
        from test_sdxl_handoff import SDXLHandoffTests

        harness = SDXLHandoffTests(methodName="runTest")
        harness.setUp()
        self.addCleanup(harness.doCleanups)
        harness.base.model.model_config = type("SDXL", (), {"unet_config": {"in_channels": 4}})()
        harness.owner.xl_base_patched.vae = self.owner.xl_base_patched.vae
        self.modules["nodes"].CLIPSetLastLayer = harness.modules["nodes"].CLIPSetLastLayer
        self.enterContext(patch.dict("sys.modules", {"nodes": self.modules["nodes"]}))
        with patch(
            "ruinedfooocus.features.models.model_architecture.inspect_model_file",
            return_value={"native_class": "SDXL"},
        ):
            data = harness.captured() | dict(
                native_tensor_family="auto",
                native_export_handoff=True,
                base_model_hash="a" * 64,
                seed=123,
            )
            data = native.capture_request(data, harness.manager, harness.models)
            harness.sample(data, denoise=0.5)
            output = data[native.EXPORT_KEY][0]
            self.assertEqual(output.kind, "sdxl-handoff")
            raw = Path(output.path).read_bytes()
            replay = harness.captured() | dict(
                native_tensor_family="auto",
                native_latent_input=raw,
                base_model_hash="a" * 64,
                seed=123,
            )
            replay = native.capture_request(replay, harness.manager, harness.models)
            loaded = native.latent_input(harness.owner, replay)
        self.assertEqual(replay["native_latent_denoise"], 0.5)
        self.assertEqual(len(harness.records), 2)
        harness.records.clear()
        harness.sample(replay, denoise=0.5, latent_image=loaded["samples"])
        self.assertEqual(len(harness.records), 1)
        self.assertEqual(type(harness.records[0]["model"].model).__name__, "SDXLRefiner")
        self.assertFalse(torch.count_nonzero(harness.records[0]["noise"]))
        self.assertEqual(harness.records[0]["sigmas"].tolist(), [0.25, 0.125, 0.0])
        replay["positive_prompt"] = "different actual prompt"
        with self.assertRaisesRegex(ValueError, "prompts"):
            harness.sample(replay, denoise=0.5, latent_image=loaded["samples"])


if __name__ == "__main__":
    unittest.main()
