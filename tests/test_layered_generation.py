"""Pinned Qwen layered latent/conditioning/decode methods on weightless CPU inputs."""

import math
import asyncio
import json
import itertools
from contextlib import nullcontext
import pathlib
import random
import traceback
from pathlib import Path
import tempfile
from types import MethodType
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
import gradio as gr
from gradio.state_holder import SessionState
from pydantic import ValidationError

from ruinedfooocus.features.generation import layered
from ruinedfooocus.features.image_editing import lora_hooks
from ruinedfooocus.features.generation.image_output_batches import PLAN_KEY
from ruinedfooocus.features.models.native_component_layout import current_layout
from ruinedfooocus.features.models.resource_defaults import load_family_recipe
from test_qwen_image21_conditioning import native_class, native_function
from test_model_cache import method
from test_native_model_coverage import PIPELINE, route_info


def runtime():
    scope = dict(
        torch=torch,
        math=math,
        io=NS(NodeOutput=lambda *values: values),
        comfy=NS(model_management=NS(intermediate_device=lambda: "cpu")),
    )
    empty = native_class(
        "comfy_extras/nodes_qwen.py",
        "EmptyQwenImageLayeredLatentImage",
        scope,
        methods=("execute",),
    )
    cut = native_class("comfy_extras/nodes_latent.py", "LatentCut", scope, methods=("execute",))
    batch = native_class(
        "comfy_extras/nodes_latent.py", "LatentCutToBatch", scope, methods=("execute",)
    )
    setter = native_function("node_helpers.py", "conditioning_set_values", {})
    scope["node_helpers"] = NS(conditioning_set_values=setter)
    reference = native_class(
        "comfy_extras/nodes_edit_model.py", "ReferenceLatent", scope, methods=("execute",)
    )
    encode = native_class("nodes.py", "VAEEncode", scope, methods=("encode",))
    decode = native_class("nodes.py", "VAEDecode", scope, methods=("decode",))
    return {
        "comfy_extras.nodes_qwen": NS(EmptyQwenImageLayeredLatentImage=empty),
        "comfy_extras.nodes_latent": NS(LatentCut=cut, LatentCutToBatch=batch),
        "comfy_extras.nodes_edit_model": NS(ReferenceLatent=reference),
        "nodes": NS(VAEEncode=encode, VAEDecode=decode),
    }


def codec():
    return NS(
        latent_channels=16,
        output_channels=4,
        spacial_compression_encode=lambda: 8,
        spacial_compression_decode=lambda: 8,
        encode=Mock(
            side_effect=lambda pixels: torch.zeros(
                (1, 16, 1, pixels.shape[1] // 8, pixels.shape[2] // 8)
            )
        ),
    )


class LayeredTests(unittest.TestCase):
    def capture(self, folder, **values):
        path = Path(folder) / "unrelated.safetensors"
        path.write_bytes(b"owned current header fixture")
        models = NS(get_file=lambda *args: path)
        data = (
            dict(
                base_model_name=path.name,
                width=64,
                height=48,
                image_number=3,
                task_type="tool_call",
            )
            | values
        )
        with (
            patch(
                "ruinedfooocus.features.models.native_recipes.current_native_recipe",
                return_value=load_family_recipe("native-qwenimage-layered"),
            ),
            patch(
                "ruinedfooocus.features.models.native_component_layout.current_native_recipe",
                return_value=load_family_recipe("native-qwenimage-layered"),
            ),
        ):
            return layered.capture(data, NS(), models)

    def test_capture_keeps_repetitions_distinct_and_snapshots_rgba_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Image.new("RGBA", (32, 24), (20, 40, 80, 128))
            data = self.capture(folder, qwen_layer_count=2, input_image=image)
            image.putpixel((0, 0), (1, 2, 3, 255))
            self.assertEqual(data["image_number"], 3)
            self.assertEqual(
                (data[PLAN_KEY].min_per_generation, data[PLAN_KEY].max_per_generation), (2, 2)
            )
            self.assertEqual(data[layered.KEY].source().getpixel((0, 0)), (20, 40, 80, 128))
            self.assertIsNone(data["input_image"])
            self.assertEqual(data["controlnet"], {})
            self.assertNotIn("source_pixels", data["native_layered_conditioning"])
            with patch(
                "ruinedfooocus.features.models.native_recipes.current_native_recipe",
                return_value=load_family_recipe("native-qwenimage-layered"),
            ):
                repeated = layered.capture(
                    data, NS(), NS(get_file=lambda *args: Path(folder) / "unrelated.safetensors")
                )
            self.assertEqual(repeated[layered.KEY].source().getpixel((0, 0)), (20, 40, 80, 128))

    def test_actual_native_latent_cuts_reconstruction_and_preserves_layer_order(self):
        plan = layered.LayeredPlan(2, "native-qwenimage-layered", "fixture")
        with patch.dict("sys.modules", runtime()):
            latent = layered.empty_latent(plan, codec(), 64, 48)
            self.assertEqual(latent["samples"].shape, (1, 16, 3, 6, 8))
            latent["samples"][:, :, 0] = 77
            latent["samples"][:, :, 1] = 11
            latent["samples"][:, :, 2] = 22
            layers = layered.layer_latent(plan, latent)
        self.assertEqual(layers["samples"].shape, (2, 16, 1, 6, 8))
        self.assertTrue(torch.all(layers["samples"][0] == 11))
        self.assertTrue(torch.all(layers["samples"][1] == 22))
        self.assertTrue(torch.all(latent["samples"][:, :, 0] == 77))

    def test_native_reference_latent_conditions_both_branches_with_one_rgba_encode(self):
        image = Image.new("RGBA", (32, 32), (20, 40, 80, 128))
        plan = layered.LayeredPlan(
            2, "native-qwenimage-layered", "fixture", image.size, image.tobytes()
        )
        positive = [["first", {"start_percent": 0}], ["second", {"start_percent": 0.5}]]
        negative = [["negative", {}]]
        vae = codec()
        with patch.dict("sys.modules", runtime()):
            p, n, frame = layered.reference_conditioning(
                plan, vae, positive, negative, 64, 48, "preserve"
            )
        vae.encode.assert_called_once()
        pixels = vae.encode.call_args.args[0]
        self.assertEqual(pixels.shape[-1], 4)
        self.assertAlmostEqual(pixels[0, 0, 0, 3].item(), 128 / 255)
        self.assertEqual([row[0] for row in p], ["first", "second"])
        self.assertIs(p[0][1]["reference_latents"][0], n[0][1]["reference_latents"][0])
        self.assertIs(p[1][1]["reference_latents"][0], n[0][1]["reference_latents"][0])
        self.assertNotIn("reference_latents", positive[0][1])
        self.assertEqual((frame["source_width"], frame["source_height"]), (32, 32))

    def test_native_decoder_flattens_single_frame_layer_batch_with_distinct_alpha(self):
        modules = runtime()
        vae = codec()
        decoded = torch.zeros((2, 1, 48, 64, 4))
        decoded[0, ..., 3] = 0.25
        decoded[1, ..., 3] = 0.75
        vae.decode = lambda samples: decoded
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", modules):
            data = self.capture(folder, image_number=1)
            latent = layered.empty_latent(data[layered.KEY], vae, 64, 48)
            sliced = layered.layer_latent(data[layered.KEY], latent)
            pixels = modules["nodes"].VAEDecode().decode(vae, sliced)[0]
            batch = layered.outputs(data, [image.numpy() for image in pixels])
        self.assertEqual(len(batch), 2)
        self.assertTrue(np.all(batch[0][..., 3] == 0.25))
        self.assertTrue(np.all(batch[1][..., 3] == 0.75))
        self.assertEqual([m["latent_slot"] for m in batch.metadata], [1, 2])

    def test_source_bound_component_layout_selects_distinct_rgba_role_before_weight_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "not-a-model-name.safetensors"
            path.write_bytes(b"owned current source")
            with patch(
                "ruinedfooocus.features.models.native_component_layout.current_native_recipe",
                return_value=load_family_recipe("native-qwenimage-layered"),
            ):
                layout = current_layout(path, NS(), family="QwenImage")
            self.assertEqual(layout.clip_roles, ("clip_qwen25",))
            self.assertEqual(layout.vae_role, "vae_qwen_image_layered")
            from ruinedfooocus.features.models.component_profiles import component_contract

            displayed = component_contract("QwenImage", native_layout=layout.metadata())
            self.assertEqual(displayed["roles"][-1]["key"], "vae_qwen_image_layered")
            self.assertEqual(displayed["roles"][-1]["layout"], "wan21_rgba")
            layout.verify(path, "QwenImage")
            path.write_bytes(b"replacement source")
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                layout.verify(path, "QwenImage")

    def test_production_loader_selects_rgba_slot_and_invalidates_recipe_and_component_caches(self):
        import test_model_cache as caches
        from ruinedfooocus.features.models import native_component_layout as layouts
        from ruinedfooocus.features.models.resource_defaults import RESOURCE_DEFAULTS

        harness = caches.ModelCacheTests(methodName="runTest")
        harness.setUp()
        self.addCleanup(harness.doCleanups)
        checkpoint = harness.file("checkpoint.safetensors")
        for role in ("clip_qwen25", "vae_qwen_image_layered", "vae_qwen_image"):
            harness.file(RESOURCE_DEFAULTS[role])
        load, owner, native, backend, _ = harness.base_loader(checkpoint, bare=True)
        owner.known_models = {"QwenImage"}
        owner.known_model_info = route_info()
        owner.get_clip_and_vae = MethodType(
            method(PIPELINE, "get_clip_and_vae", dict(settings=NS(default_settings={}))), owner
        )
        native.return_value = NS(
            model=type("QwenImage", (), {"model_config": NS(unet_config={})})()
        )
        vae = backend.sd.VAE.return_value
        vae.latent_channels, vae.output_channels = 16, 4
        vae.spacial_compression_encode = vae.spacial_compression_decode = lambda: 8
        with patch.object(
            layouts,
            "current_native_recipe",
            return_value=load_family_recipe("native-qwenimage-layered"),
        ) as recipe:
            load()
            self.assertEqual(owner.model_info["latent"], "QWEN_LAYERED")
            self.assertEqual(owner.model_info["vae_key"], "vae_qwen_image_layered")
            self.assertEqual(
                Path(backend.utils.load_torch_file.call_args.args[0]).name,
                RESOURCE_DEFAULTS["vae_qwen_image_layered"],
            )
            load()
            self.assertEqual(native.call_count, 1)
            harness.replace(harness.root / RESOURCE_DEFAULTS["vae_qwen_image_layered"])
            load()
            self.assertEqual(native.call_count, 2)
            recipe.return_value = load_family_recipe("QwenImage")
            load()
            self.assertEqual(native.call_count, 3)
            self.assertEqual(owner.model_info["vae_key"], "vae_qwen_image")
            self.assertEqual(owner.model_info["latent"], "SD3")
            recipe.return_value = load_family_recipe("native-qwenimage-layered")
            vae.output_channels = 3
            with self.assertRaisesRegex(ValueError, "RGBA VAE"):
                load()
            self.assertIsNone(owner.xl_base_hash)

    def test_invalid_codec_counts_and_aggregate_pixel_budget_fail_before_sampling(self):
        for channels, outputs, scale in ((16, 3, 8), (64, 4, 16), (16, 4, 16)):
            bad = NS(
                latent_channels=channels,
                output_channels=outputs,
                spacial_compression_encode=lambda: scale,
                spacial_compression_decode=lambda: scale,
            )
            with (
                self.subTest(codec=(channels, outputs, scale)),
                self.assertRaisesRegex(ValueError, "RGBA VAE"),
            ):
                layered.require_vae(bad)
        with tempfile.TemporaryDirectory() as folder:
            for values in (
                dict(qwen_layer_count=0),
                dict(qwen_layer_count=17),
                dict(qwen_layer_count=True),
                dict(width=4096, height=4096, qwen_layer_count=2),
                dict(output_format="avif"),
                dict(automatic_upscale="2x"),
                dict(image_number=0),
            ):
                with self.subTest(values=values), self.assertRaises(ValueError):
                    self.capture(folder, **values)

    def test_api_captures_rgba_and_count_without_native_edit_or_extra_repetitions(self):
        from ruinedfooocus.integrations.openai_api import ImageRequest
        from test_api_image_edits import queued_data

        request = ImageRequest(prompt="overall scene", n=3, qwen_layer_count=2)
        image = Image.new("RGBA", (64, 48), (20, 40, 80, 128))
        calls = []
        with (
            patch(
                "ruinedfooocus.features.models.native_recipes.current_native_recipe",
                return_value=load_family_recipe("native-qwenimage-layered"),
            ),
            patch(
                "ruinedfooocus.features.models.native_component_layout.current_native_recipe",
                return_value=load_family_recipe("native-qwenimage-layered"),
            ),
        ):
            data = queued_data(
                "QwenImage",
                payload_options=request.model_dump(exclude_unset=True),
                source_image=image,
                decode_calls=calls,
            )
        self.assertEqual(data["image_number"], 3)
        self.assertEqual(data["qwen_layer_count"], 2)
        self.assertEqual(data["controlnet"], {})
        self.assertEqual(data[layered.KEY].source().getpixel((0, 0))[3], 128)
        self.assertTrue(calls[0]["preserve_alpha"])
        self.assertEqual(data["native_component_layout"]["vae_role"], "vae_qwen_image_layered")
        self.assertEqual(
            set(data["image_component_overrides"]), {"clip_qwen25", "vae_qwen_image_layered"}
        )
        for count in (0, 17, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValidationError):
                ImageRequest(prompt="x", qwen_layer_count=count)
        with self.assertRaisesRegex(ValueError, "identified Qwen"):
            queued_data("SDXL", payload_options={"qwen_layer_count": 2})
        # A saved default for Layered never becomes an explicit request on SDXL.
        data = queued_data("SDXL", defaults_options={"qwen_layer_count": 2})
        self.assertNotIn("qwen_layer_count", data)

    def test_shift_default_uses_native_patch_and_keeps_explicit_guidance(self):
        model = NS(model=type("QwenImage", (), {})())
        native_patch = Mock(return_value=("cloned native model",))
        with patch.dict(
            "sys.modules",
            {
                "comfy_extras.nodes_model_advanced": NS(
                    ModelSamplingAuraFlow=lambda: NS(patch_aura=native_patch)
                )
            },
        ):
            self.assertEqual(layered.patch_model(model, {}), "cloned native model")
            native_patch.assert_called_once_with(model, shift=1.0)
            self.assertIs(layered.patch_model(model, {"sampling_shift": 2.3}), model)
            native_patch.assert_called_once()

    def test_standalone_tools_drop_hidden_layer_sources_and_component_claims(self):
        from ruinedfooocus.features.generation.image_preparation import omit_generation_conditioning

        source = Image.new("RGBA", (16, 8), (20, 40, 80, 128))
        values = dict(
            input_image=source,
            controlnet={"type": "segment"},
            qwen_layer_count=2,
            qwen_layer_source=source,
            native_layered_conditioning={"source": {"width": 16, "height": 8}},
            native_component_layout={"vae_role": "vae_qwen_image_layered"},
        )
        values[layered.KEY] = layered.LayeredPlan(2, "native-qwenimage-layered", "fixture")
        values["_native_component_layout"] = object()
        captured = omit_generation_conditioning(values)
        self.assertIs(captured["input_image"], source)
        self.assertEqual(captured["controlnet"], {"type": "segment"})
        for key in values.keys() - {"input_image", "controlnet"}:
            self.assertNotIn(key, captured)
            self.assertIn(key, values)

    def test_replay_restores_layer_count_and_requires_exact_original_source(self):
        from ruinedfooocus.features.generation.image_preparation import image_control_data
        from ruinedfooocus.features.gallery.image_metadata import render_values

        with tempfile.TemporaryDirectory() as folder:
            source = Image.new("RGBA", (32, 24), (10, 40, 80, 128))
            original = self.capture(folder, qwen_layer_count=3, input_image=source)
            metadata = dict(
                prompt="literal overall scene",
                qwen_layer_count=3,
                native_layered_conditioning=original["native_layered_conditioning"],
            )
            self.assertEqual(render_values(metadata)["qwen_layer_count"], 3)
            replay = image_control_data(dict(prompt=json.dumps(metadata)))
            self.assertEqual(replay["qwen_layer_count"], 3)
            with self.assertRaisesRegex(ValueError, "Reattach the original"):
                self.capture(folder, **{k: v for k, v in replay.items() if k != "prompt"})
            restored = self.capture(
                folder, input_image=source, **{k: v for k, v in replay.items() if k != "prompt"}
            )
            self.assertEqual(
                restored[layered.KEY].source_pixels, original[layered.KEY].source_pixels
            )

    def test_real_contextual_gradio_controls_retain_rgba_and_preserve_source_draft(self):
        from ruinedfooocus.features.generation import ui_layered as ui
        import ruinedfooocus.app

        registered = {}
        shared = NS(
            settings=NS(default_settings={"qwen_layer_count": 3}),
            models=NS(get_file=lambda category, name: name),
            path_manager=NS(),
            add_ctrl=lambda key, value: registered.update({key: value}),
        )
        with (
            patch.dict("sys.modules", {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(ui, "is_layered_recipe", side_effect=lambda path, _: path == "layered"),
        ):
            with gr.Blocks(analytics_enabled=False) as app:
                selected = gr.Textbox(value="layered")
                panel = ui.create_ui(selected)
            self.assertTrue(panel.visible)
            self.assertEqual(registered["qwen_layer_source"].image_mode, "RGBA")
            self.assertEqual(registered["qwen_layer_count"].value, 3)
            callback = next(key for key, fn in app.fns.items() if fn.fn.__name__ == "context")
            output = asyncio.run(
                app.process_api(callback, ["ordinary", "layered"], state=SessionState(app))
            )["data"]
        self.assertFalse(output[0]["visible"])
        self.assertNotIn("value", output[1])

    def test_actual_worker_runs_n_times_and_saves_every_layer_with_replay_metadata(self):
        from PIL.PngImagePlugin import PngInfo
        from ruinedfooocus.core.paths import resource_path
        from ruinedfooocus.core.task_errors import describe_error
        from ruinedfooocus.features.generation.image_output_batches import (
            ImageBatchArtifacts,
            validate_artifacts,
        )
        from test_image_preparation import function

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = self.capture(folder, image_number=3, qwen_layer_count=2)
            data.update(
                task_id=1,
                task_type="tool_call",
                prompt="literal overall scene",
                negative="literal negative",
                _sampling_captured=True,
                base_model_hash="owned fixture",
                custom_steps=2,
                cfg=2.5,
                sampler_name="euler",
                scheduler="simple",
                clip_skip=1,
                loras=[],
                aspect_ratios_selection="Custom...",
                custom_width=64,
                custom_height=48,
                style_selection=[],
                seed=42,
                silent=True,
                index=(0, 1),
            )

            def process(gen_data, callback):
                return layered.outputs(
                    gen_data,
                    [
                        np.full((48, 64, 4), 63, dtype=np.uint8),
                        np.full((48, 64, 4), 191, dtype=np.uint8),
                    ],
                )

            pipe = NS(
                pipeline_type=["sdxl"],
                load_base_model=Mock(),
                load_loras=Mock(),
                process=Mock(side_effect=process),
            )
            forbidden = Mock(side_effect=AssertionError("uncaptured generation dependency"))
            shared = NS(
                models=NS(get_models_by_path=lambda *a: {}, get_model_base=lambda _: "QwenImage"),
                state={},
                performance_settings=NS(apply=forbidden),
                resolution_settings=NS(CUSTOM_RESOLUTION="Custom..."),
                path_manager=NS(model_paths={"temp_outputs_path": root}),
                shared_cache={},
            )
            sequence = itertools.count()
            render = function(
                "_process",
                dict(
                    image_stage=lambda fn: fn,
                    shared=shared,
                    settings=NS(default_settings={}),
                    control_tools=NS(get_settings=lambda data: data.get("controlnet") or {}),
                    check_interrupt=lambda _: None,
                    process_metadata=forbidden,
                    pipelines=NS(update=lambda _: pipe),
                    get_checkpoint_hashes=forbidden,
                    process_prompt=lambda styles, p, n, data: (p, n),
                    shift_attention=lambda text, distance: text,
                    parse_loras=lambda p, n: ([], p, n),
                    merge_loras=lambda *a: [],
                    TimeIt=lambda _: nullcontext(),
                    resource_path=resource_path,
                    random=random,
                    Path=Path,
                    pathlib=pathlib,
                    Image=Image,
                    PngInfo=PngInfo,
                    json=json,
                    traceback=traceback,
                    model_management=NS(
                        InterruptProcessingException=type("Cancelled", (Exception,), {})
                    ),
                    describe_error=describe_error,
                    generate_temp_filename=lambda **kwargs: str(root / f"out-{next(sequence)}.png"),
                ),
            )
            result = render(data)
            self.assertIsInstance(result, ImageBatchArtifacts)
            self.assertEqual(pipe.process.call_count, 3)
            self.assertEqual(len(result), 6)
            self.assertIs(validate_artifacts(data, result), result)
            for index, path in enumerate(result):
                with Image.open(path) as image:
                    metadata = json.loads(image.info["parameters"])
                    self.assertEqual(image.mode, "RGBA")
                    self.assertEqual(image.getpixel((0, 0))[3], (63, 191)[index % 2])
                self.assertEqual(metadata["qwen_layer_count"], 2)
                self.assertEqual(
                    metadata["native_layered_conditioning"]["codec"], "vae_qwen_image_layered"
                )
                self.assertEqual(metadata["native_image_output"]["latent_slot"], index % 2 + 1)
                self.assertEqual(metadata["native_output_group"]["generation_index"], index // 2)
            forbidden.assert_not_called()

    def test_complete_production_process_routes_latents_conditioning_and_ordered_rgba(self):
        from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY, SizePlan
        from ruinedfooocus.features.image_editing.image_guidance import resolve_guidance
        from ruinedfooocus.features.image_editing.image_geometry import restore_reference_frame

        for source in (None, Image.new("RGBA", (32, 24), (20, 40, 80, 128))):
            with self.subTest(source=source is not None), tempfile.TemporaryDirectory() as folder:
                data = self.capture(folder, image_number=1, input_image=source)
                data.update(
                    positive_prompt="literal overall scene",
                    negative_prompt="literal negative",
                    steps=2,
                    cfg=2.5,
                    sampler_name="euler",
                    scheduler="simple",
                    clip_skip=1,
                    seed=42,
                    show_preview=False,
                    silent=True,
                )
                data[SIZE_KEY] = SizePlan.create((64, 48), alignment=16)
                vae = codec()

                def decode(samples):
                    batch, _, _, height, width = samples.shape
                    result = torch.zeros(batch, 1, height * 8, width * 8, 4)
                    result[..., :3] = 0.5
                    for index in range(batch):
                        result[index, ..., 3] = samples[index, 0, 0, 0, 0]
                    return result

                vae.decode = decode
                inner = type("QwenImage", (), {"model_config": NS(unet_config={})})()
                unet = NS(model=inner)
                owner = NS(
                    xl_base_patched=NS(unet=unet, vae=vae),
                    xl_base=NS(unet=unet),
                    model_info=dict(latent="QWEN_LAYERED"),
                    known_models={"QwenImage"},
                    conditions={"+": {}, "-": {}},
                )

                def encode(sign, text, *args, **kwargs):
                    owner.conditions[sign]["cache"] = [[text, {}]]

                owner.textencode = encode
                modules = runtime()
                owner.decode_latent = lambda latent, **kwargs: (
                    modules["nodes"].VAEDecode().decode(vae, latent)[0]
                )

                def sample(**values):
                    samples = values["latent_image"].clone()
                    samples[:, :, 0] = 0.99
                    samples[:, :, 1] = 0.25
                    samples[:, :, 2] = 0.75
                    return samples

                sampler = Mock(side_effect=sample)
                native = NS(
                    model_management=NS(get_torch_device=lambda: torch.device("cpu")),
                    sample=NS(
                        sample=sampler, prepare_noise=lambda latent, *a: torch.zeros_like(latent)
                    ),
                    utils=NS(
                        ProgressBar=lambda steps: NS(update_absolute=Mock()),
                        PROGRESS_BAR_ENABLED=False,
                    ),
                )
                process = method(
                    PIPELINE,
                    "process",
                    dict(
                        torch=torch,
                        nullcontext=nullcontext,
                        np=np,
                        random=NS(),
                        settings=NS(default_settings={}),
                        control_tools=NS(
                            get_settings=lambda data: data.get("controlnet") or {},
                            controlnet_models={},
                        ),
                        resolve_guidance=resolve_guidance,
                        normalize_reference_images=lambda *args: [],
                        request_model=lambda model, *a, **k: model,
                        effective_guidance=lambda *a: {},
                        pp=NS(prompt_switch_per_step=lambda *a: []),
                        comfy=native,
                        SDXL=type("SDXL", (), {}),
                        Flux=type("Flux", (), {}),
                        fix_empty_latent_channels=lambda model, latent, *a: latent,
                        restore_reference_frame=restore_reference_frame,
                        reference_tensor=Mock(
                            side_effect=AssertionError("generic source conversion")
                        ),
                        path_manager=NS(),
                        shared=NS(),
                        lora_hooks=lora_hooks,
                    ),
                )
                with (
                    patch.dict("sys.modules", modules),
                    patch.object(layered, "patch_model", side_effect=lambda model, _: model),
                ):
                    outputs = process(owner, data)
                self.assertEqual(len(outputs), 2)
                self.assertEqual([row["latent_slot"] for row in outputs.metadata], [1, 2])
                self.assertEqual(outputs[0].shape, (48, 64, 4))
                self.assertTrue(np.all(outputs[0][..., 3] == 63))
                self.assertTrue(np.all(outputs[1][..., 3] == 191))
                call = sampler.call_args.kwargs
                self.assertEqual(call["latent_image"].shape[:3], (1, 16, 3))
                self.assertEqual(call["positive"][0][0], "literal overall scene")
                self.assertEqual(call["negative"][0][0], "literal negative")
                self.assertEqual(vae.encode.call_count, int(source is not None))
                if source is not None:
                    self.assertIs(
                        call["positive"][0][1]["reference_latents"][0],
                        call["negative"][0][1]["reference_latents"][0],
                    )


if __name__ == "__main__":
    unittest.main()
