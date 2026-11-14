"""Cascade keeps the native stage-C prior, stage-B grid and final StageA decode."""

from contextlib import contextmanager
from concurrent.futures import CancelledError
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from ruinedfooocus.features.generation.cascade import (
    sampling_settings,
    require_bundle,
    decode_stage_b,
    stage_b_defaults,
    STAGE_B_RECIPE,
    source_latents,
    source_settings,
)
from test_native_model_coverage import native_method
from test_native_variant_defaults import write_header
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.features.models.model_identity import file_signature
from test_image_preparation import Manager
from ruinedfooocus.features.generation.image_preparation import (
    capture_image_assets,
    prepared_asset,
    KEY,
)
from test_ordered_native_references import process_nodes


class NativeCascadeTests(unittest.TestCase):
    def source_runtime(self):
        upscale = Mock(
            side_effect=lambda pixels, width, height, *_: torch.nn.functional.interpolate(
                pixels, size=(height, width), mode="bicubic", align_corners=False
            )
        )
        encoder = native_method(
            "comfy_extras/nodes_stable_cascade.py",
            "StableCascade_StageC_VAEEncode",
            "execute",
            dict(
                torch=torch,
                io=NS(NodeOutput=lambda *values: values),
                comfy=NS(utils=NS(common_upscale=upscale)),
            ),
        )
        vae = NS(downscale_ratio=32)
        vae.encode = Mock(
            side_effect=lambda pixels: torch.full(
                (1, 16, pixels.shape[1] // 32, pixels.shape[2] // 32), 0.25
            )
        )
        return encoder, upscale, vae

    def test_native_source_encoder_retains_paired_grids_and_original_pixels(self):
        encoder, upscale, vae = self.source_runtime()
        pixels = torch.linspace(0, 1, 93 * 137 * 4).reshape(1, 93, 137, 4)
        data = self.data() | dict(cascade_source_denoise=0, cascade_compression=42)
        with patch.dict(
            "sys.modules",
            {"comfy_extras.nodes_stable_cascade": NS(StableCascade_StageC_VAEEncode=encoder)},
        ):
            pair = source_latents(vae, pixels, data)
        self.assertEqual(upscale.call_args.args[1:], (96, 64, "bicubic", "center"))
        self.assertTrue(torch.equal(upscale.call_args.args[0], pixels.movedim(-1, 1)))
        self.assertEqual(vae.encode.call_args.args[0].shape[-1], 3)
        self.assertEqual(tuple(pair.stage_c["samples"].shape), (1, 16, 2, 3))
        self.assertEqual(tuple(pair.stage_b["samples"].shape), (1, 4, 22, 34))
        self.assertEqual((data["width"], data["height"]), (136, 88))
        self.assertEqual(data["native_cascade"]["source_denoise"], 0)
        for bad in (
            dict(cascade_compression=True),
            dict(cascade_compression=3),
            dict(cascade_source_denoise=float("nan")),
        ):
            with self.assertRaises(ValueError):
                source_settings(data | bad)
        with self.assertRaisesRegex(ValueError, "too small"):
            source_latents(vae, torch.zeros(1, 32, 32, 3), data)

    def test_actual_preprocess_and_sampler_latent_callers_skip_application_resize(self):
        from PIL import Image
        import numpy as np

        encoder, _, vae = self.source_runtime()
        preprocess = next(
            node
            for node in process_nodes()
            if isinstance(node, ast.If)
            and "source_pixels" in ast.unparse(node)
            and any(
                isinstance(child, ast.Name) and child.id == "cascade_latents"
                for child in ast.walk(node)
            )
        )
        select = next(
            node
            for node in process_nodes()
            if isinstance(node, ast.If)
            and ast.unparse(node.test)
            in (
                "tensor_latent is not None",
                "named_source_latent is not None",
                "cascade_latents is not None",
            )
        )
        resize = Mock(side_effect=AssertionError("Cascade reached application framing"))
        scope = dict(
            qwen_reference_edit=False,
            named_source=False,
            named_source_latent=None,
            tensor_latent=None,
            input_image=Image.new("RGB", (137, 93)),
            cascade_model=True,
            cascade_latents=None,
            input_images=0,
            self=NS(xl_base_patched=NS(vae=vae)),
            gen_data=self.data() | dict(cascade_source_denoise=0),
            guidance={"image_resize_mode": "preserve"},
            reference_tensor=lambda image: (
                torch.from_numpy(np.asarray(image).copy()).float().unsqueeze(0) / 255
            ),
            prepare_reference=resize,
            ImageScaleToTotalPixels=resize,
        )
        with patch.dict(
            "sys.modules",
            {"comfy_extras.nodes_stable_cascade": NS(StableCascade_StageC_VAEEncode=encoder)},
        ):
            exec(
                compile(
                    ast.Module(body=[preprocess, select], type_ignores=[]),
                    "Cascade production source caller",
                    "exec",
                ),
                scope,
            )
        self.assertEqual(scope["input_images"], 0)
        self.assertIs(scope["latent"], scope["cascade_latents"].stage_c)
        self.assertEqual(scope["denoise"], 0)
        resize.assert_not_called()

    def test_public_api_exact_source_targets_reach_the_real_restore_and_finish_callers(self):
        from PIL import Image
        import numpy as np
        from test_api_image_edits import queued_data
        from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY
        from ruinedfooocus.features.image_editing.image_geometry import restore_reference_frame

        source = Image.new("RGB", (91, 127), (25, 80, 140))
        before = np.asarray(source).copy()
        encoder, _, vae = self.source_runtime()
        body = process_nodes()
        preprocess = next(
            node
            for node in body
            if isinstance(node, ast.If) and "source_pixels" in ast.unparse(node)
        )
        decode = next(
            node
            for node in body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "architecture == 'StableCascade_C'"
            and "decode_stage_b" in ast.unparse(node)
        )
        restore_index = next(
            index
            for index, node in enumerate(body)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "native_control_recipes.has_source(gen_data)"
        )
        upscale_index = next(
            index
            for index, node in enumerate(body[restore_index:], restore_index)
            if isinstance(node, ast.Assign)
            and ast.unparse(node.value) == "upscale_decoded(self, decoded_latent, gen_data)"
        )
        finish = body[restore_index : upscale_index + 1]
        options = {key: value for key, value in self.data().items() if key.startswith("cascade_")}

        def capture(data, *_args, **_kwargs):
            sampling_settings(data)
            return dict(data)

        for target, selection in (
            ("512x512", "Off"),
            ("513x517", "Off"),
            ("613x977", "2× General (RealESRGAN)"),
        ):
            with (
                self.subTest(target=target, upscale=selection),
                patch(
                    "ruinedfooocus.features.generation.image_preparation.capture_image_assets",
                    side_effect=capture,
                ),
            ):
                data = queued_data(
                    "Stable_Cascade_C",
                    source_image=source,
                    payload_options=dict(options, size=target, automatic_upscale=selection),
                )
            plan = data[SIZE_KEY]
            self.assertEqual(plan.target, tuple(map(int, target.split("x"))))
            self.assertEqual(data["image_resize_mode"], "pad")
            data["task_id"] = "cpu-cascade-target"
            scope = dict(
                qwen_reference_edit=False,
                named_source=False,
                named_source_latent=None,
                layered_model=False,
                native_control_recipes=NS(has_source=lambda data: False),
                input_image=source,
                cascade_model=True,
                guided_edit=False,
                cascade_latents=None,
                input_images=0,
                self=NS(xl_base_patched=NS(vae=vae)),
                gen_data=data,
                guidance=dict(image_resize_mode="pad"),
                reference_tensor=Mock(
                    side_effect=AssertionError("exact target bypassed source framing")
                ),
                architecture="StableCascade_C",
                sampled_latent={"samples": "C prior"},
                path_manager=NS(),
                seed=123,
                worker=NS(check_interrupt=Mock()),
                restore_reference_frame=Mock(wraps=restore_reference_frame),
                outpaint_composite=None,
            )
            decoded = Mock(side_effect=lambda *args, **kwargs: scope["input_image"].clone())
            upscaler = NS(scale=plan.scale, input_channels=3, output_channels=3)
            modules = {
                "comfy_extras.nodes_stable_cascade": NS(StableCascade_StageC_VAEEncode=encoder),
                "comfy_extras.nodes_upscale_model": NS(
                    ImageUpscaleWithModel=NS(
                        execute=lambda model, pixels: (
                            pixels.repeat_interleave(plan.scale, 1).repeat_interleave(
                                plan.scale, 2
                            ),
                        )
                    )
                ),
                "ruinedfooocus.features.image_processing.upscale_pipeline": NS(
                    pipeline=lambda: NS(load_upscaler_model=lambda *args: upscaler),
                    InterruptibleUpscaler=lambda model, data: model,
                ),
                "ruinedfooocus.app.async_worker": NS(check_interrupt=Mock(), add_result=Mock()),
            }
            with (
                patch.dict("sys.modules", modules),
                patch("ruinedfooocus.features.generation.cascade.decode_stage_b", decoded),
            ):
                exec(
                    compile(
                        ast.Module(body=[preprocess, decode, *finish], type_ignores=[]),
                        "public API Cascade source through exact target",
                        "exec",
                    ),
                    scope,
                )
            self.assertIs(
                decoded.call_args.kwargs["stage_b_latent"], scope["cascade_latents"].stage_b
            )
            scope["restore_reference_frame"].assert_called_once()
            self.assertEqual(
                tuple(scope["decoded_latent"].shape[1:3]), (plan.target_height, plan.target_width)
            )
            self.assertEqual(data[SIZE_KEY], plan)
            self.assertEqual(data["size_plan"], plan.metadata())
            frame = data["native_cascade"]["source_frame"]
            self.assertEqual((frame["source_width"], frame["source_height"]), source.size)
            self.assertEqual(
                (data["native_cascade"]["source_width"], data["native_cascade"]["source_height"]),
                source.size,
            )
            self.assertLessEqual(
                abs(frame["content_width"] / frame["content_height"] - 91 / 127),
                1 / frame["content_height"],
            )
        self.assertTrue(np.array_equal(np.asarray(source), before))

    def test_public_api_crop_is_refused_during_submission(self):
        from test_api_image_edits import queued_data

        options = {key: value for key, value in self.data().items() if key.startswith("cascade_")}

        def capture(data, *_args, **_kwargs):
            sampling_settings(data)
            self.fail("unsupported crop reached asset preparation")

        with patch(
            "ruinedfooocus.features.generation.image_preparation.capture_image_assets",
            side_effect=capture,
        ):
            with self.assertRaisesRegex(ValueError, "cropping is unavailable"):
                queued_data(
                    "Stable_Cascade_C", payload_options=dict(options, image_resize_mode="crop")
                )

    def test_api_metadata_and_replay_keep_compression_and_explicit_zero(self):
        from ruinedfooocus.integrations.openai_api import ImageRequest
        from ruinedfooocus.features.gallery.image_metadata import read_metadata
        from ruinedfooocus.features.generation.image_preparation import image_control_data
        import json

        request = ImageRequest(prompt="cat", cascade_compression=48, cascade_source_denoise=0)
        self.assertEqual(request.cascade_compression, 48)
        values = dict(cascade_compression=48, cascade_source_denoise=0)
        metadata = read_metadata(json.dumps(dict(prompt="cat", **values)))
        self.assertEqual({key: metadata[key] for key in values}, values)
        replay = image_control_data(dict(prompt=json.dumps(dict(prompt="cat", **values))))
        self.assertEqual({key: replay[key] for key in values}, values)

    def test_automatic_second_stage_uses_current_verified_official_source_only(self):
        row = load_registry().artifact("existing:checkpoints/stable_cascade_stage_b.safetensors")
        with tempfile.TemporaryDirectory() as folder:
            path = write_header(Path(folder) / "renamed.safetensors", {"unknown.weight": [1]})
            signature = file_signature(path)
            signature[1] = row["artifact"]["bytes"]
            metadata = dict(
                _rf_file_signature=signature,
                _rf_local_sha256=row["artifact"]["sha256"],
                _rf_verified_native_source=dict(
                    artifact=row["id"],
                    family=row["native_family"],
                    sha256=row["artifact"]["sha256"],
                ),
            )
            models = NS(
                get_file=lambda *args: path, get_models_by_path=lambda *args, **kwargs: metadata
            )
            with patch(
                "ruinedfooocus.features.models.model_identity.file_signature",
                return_value=signature,
            ):
                self.assertEqual(stage_b_defaults(path.name, models), STAGE_B_RECIPE)
            with patch(
                "ruinedfooocus.features.models.model_identity.file_signature",
                return_value=signature[:2] + [0, 0],
            ):
                self.assertEqual(stage_b_defaults(path.name, models), {})

    def test_actual_submission_captures_b_and_rejects_late_source_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            models_root = root / "checkpoints"
            models_root.mkdir()
            c = write_header(
                models_root / "stage_c.safetensors",
                {
                    "text_encoder.clip_g.text_projection": [1280, 1280],
                    "vae.encoder.backbone.1.0.block.0.1.num_batches_tracked": [],
                },
            )
            b = write_header(
                models_root / "stage_b.safetensors",
                {
                    "text_encoder.clip_g.text_projection": [1280, 1280],
                    "vae.vquantizer.codebook.weight": [8192, 4],
                },
            )
            files = {c.name: c, b.name: b}
            models = NS(
                get_model_base=lambda info: info["baseModel"],
                get_models_by_path=lambda *a, **k: dict(baseModel="Stable_Cascade_C"),
                get_file=lambda category, name: files.get(name),
                detect_checkpoint_base=lambda path: "Stable_Cascade_B",
            )
            modules = {"comfy.samplers": NS(KSampler=NS(SAMPLERS=["euler"], SCHEDULERS=["simple"]))}
            manager = Manager(root)
            with (
                patch.dict("sys.modules", modules),
                patch(
                    "ruinedfooocus.features.models.component_profiles.component_inventory",
                    return_value={"rows": []},
                ),
            ):
                captured = capture_image_assets(
                    self.data() | dict(base_model_name=c.name), manager, {}, models=models
                )
            item = captured[KEY].artifacts[0]
            self.assertEqual((item.role, item.name), ("cascade_stage_b", b.name))
            with prepared_asset(
                captured, "cascade_stage_b", b.name, manager, "checkpoints"
            ) as path:
                self.assertEqual(path, b.resolve())
            b.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed after submission"):
                with prepared_asset(captured, "cascade_stage_b", b.name, manager, "checkpoints"):
                    self.fail("changed source reached loader")

    def data(self):
        return dict(
            cascade_stage_b="stage_b.safetensors",
            cascade_b_steps=2,
            cascade_b_cfg=1.5,
            cascade_b_sampler="euler",
            cascade_b_scheduler="simple",
            width=64,
            height=64,
            positive_prompt="cat",
            negative_prompt="",
            clip_skip=1,
        )

    def test_explicit_second_stage_settings_are_required_and_bounded(self):
        with patch.dict(
            "sys.modules",
            {"comfy.samplers": NS(KSampler=NS(SAMPLERS=["euler"], SCHEDULERS=["simple"]))},
        ):
            self.assertEqual(sampling_settings(self.data()), (2, 1.5, "euler", "simple"))
            for kind in ("edit", "img2img"):
                self.assertEqual(
                    sampling_settings(
                        self.data() | dict(input_image=object(), controlnet={"type": kind})
                    ),
                    (2, 1.5, "euler", "simple"),
                )
            for values in (
                dict(cascade_b_steps=None),
                dict(cascade_b_cfg=float("nan")),
                dict(cascade_b_sampler="unknown"),
                dict(input_image=object()),
                dict(inpaint_toggle=True),
                dict(input_image=object(), controlnet={"type": "depth"}),
                dict(cascade_stage_b=""),
            ):
                with self.assertRaises(ValueError):
                    sampling_settings(self.data() | values)

    def test_bounded_native_bundle_markers_distinguish_final_codec_from_preview(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "model.safetensors"
            write_header(
                p,
                {
                    "text_encoder.clip_g.text_projection": [1280, 1280],
                    "vae.encoder.backbone.1.0.block.0.1.num_batches_tracked": [],
                },
            )
            require_bundle(p, "c")
            with self.assertRaisesRegex(ValueError, "Stage A"):
                require_bundle(p, "b")
            write_header(
                p,
                {
                    "text_encoder.clip_g.text_projection": [1280, 1280],
                    "vae.vquantizer.codebook.weight": [8192, 4],
                },
            )
            require_bundle(p, "b")

    def test_actual_native_stage_nodes_feed_prior_and_final_codec_under_lease(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = write_header(
                root / "stage_b.safetensors",
                {
                    "text_encoder.clip_g.text_projection": [1280, 1280],
                    "vae.vquantizer.codebook.weight": [8192, 4],
                },
            )
            held = []

            @contextmanager
            def lease(*args, **kwargs):
                held.append(path)
                try:
                    yield path
                finally:
                    held.pop()

            output = NS(NodeOutput=lambda *values: values)
            empty = native_method(
                "comfy_extras/nodes_stable_cascade.py",
                "StableCascade_EmptyLatentImage",
                "execute",
                dict(torch=torch, io=output),
            )
            condition = native_method(
                "comfy_extras/nodes_stable_cascade.py",
                "StableCascade_StageB_Conditioning",
                "execute",
                dict(io=output),
            )
            prior = {"samples": torch.ones(1, 16, 1, 1)}
            model = NS(model=type("StableCascade_B", (), {})())
            vae = NS(throw_exception_if_invalid=Mock(), spacial_compression_decode=lambda: 4)
            loader = Mock(return_value=(model, "clip", vae, None))

            def sample(*args):
                self.assertTrue(held)
                self.assertIs(args[6][0][1]["stable_cascade_prior"], prior["samples"])
                self.assertNotIn("stable_cascade_prior", args[7][0][1])
                self.assertEqual(tuple(args[8]["samples"].shape), (1, 4, 16, 16))
                return [args[8]]

            sampler = Mock(side_effect=sample)
            decoder = Mock(return_value=["StageA pixels"])
            modules = {
                "comfy.sd": NS(load_checkpoint_guess_config=loader),
                "comfy.samplers": NS(KSampler=NS(SAMPLERS=["euler"], SCHEDULERS=["simple"])),
                "nodes": NS(
                    CLIPTextEncode=lambda: NS(encode=lambda clip, text: [[[text, {}]]]),
                    CLIPSetLastLayer=Mock(),
                    common_ksampler=sampler,
                    VAEDecode=lambda: NS(decode=decoder),
                ),
                "comfy_extras.nodes_stable_cascade": NS(
                    StableCascade_EmptyLatentImage=empty,
                    StableCascade_StageB_Conditioning=condition,
                ),
                "node_helpers": NS(conditioning_set_values=lambda *a: None),
            }
            owner = NS()
            data = self.data()
            paired = {"samples": torch.zeros(1, 4, 16, 16)}
            manager = NS(model_paths={"embedding_path": [], "cache_path": root})
            with (
                patch.dict("sys.modules", modules),
                patch("ruinedfooocus.features.generation.image_preparation.prepared_asset", lease),
                patch("ruinedfooocus.backends.comfy.text_encoder_warmup.warm_patched_text_encoder"),
                patch(
                    "ruinedfooocus.backends.comfy.embedding_paths.register_embedding_paths",
                    return_value=[],
                ),
                patch(
                    "ruinedfooocus.features.models.embedding_browser.native_embedding_text",
                    side_effect=lambda text, *a: text,
                ),
                patch(
                    "ruinedfooocus.features.prompting.prompt_scheduling.prompt_switch_per_step",
                    return_value=None,
                ),
            ):
                self.assertEqual(
                    decode_stage_b(owner, prior, data, manager, 123, stage_b_latent=paired),
                    "StageA pixels",
                )
                self.assertEqual(decode_stage_b(owner, prior, data, manager, 123), "StageA pixels")
                before = sampler.call_count
                with self.assertRaises(CancelledError):
                    decode_stage_b(
                        owner,
                        prior,
                        data,
                        manager,
                        123,
                        check_cancelled=Mock(side_effect=CancelledError()),
                    )
                self.assertEqual(sampler.call_count, before)
            loader.assert_called_once()
            self.assertIs(sampler.call_args_list[0].args[8], paired)
            self.assertFalse(held)
            self.assertEqual(data["cascade_stage_b_source"]["steps"], 2)
            self.assertIs(decoder.call_args.args[0], vae)


if __name__ == "__main__":
    unittest.main()
