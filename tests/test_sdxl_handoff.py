"""Native advanced-sampling bodies and production SDXL staging without model weights."""

import ast
from concurrent.futures import CancelledError
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from pydantic import ValidationError

from ruinedfooocus.features.generation import sdxl_refiner as native
from ruinedfooocus.features.generation import image_preparation as preparation
from ruinedfooocus.features.models.model_identity import file_signature
from test_image_preparation import Manager
from test_native_variant_defaults import write_header
from test_ordered_native_references import process_nodes
from test_qwen_image21_conditioning import native_class, native_function
from test_sdxl_refiner import native_refiner_node


class Latent:
    latent_channels = 4
    scale_factor = 0.13025


class Sampling:
    def __init__(self, patcher):
        self.patcher = patcher

    @property
    def sigma(self):
        return self.patcher.sigma

    def calculate_input(self, *args):
        pass

    def calculate_denoised(self, *args):
        pass

    def noise_scaling(self, *args):
        pass

    def inverse_noise_scaling(self, *args):
        pass


def model(family, *, sigma=1.0, channels=4, prediction="EPS", latent=None):
    inner = type(family, (), {})()
    inner.model_config = NS(unet_config={"in_channels": channels})
    inner.model_type = prediction
    inner.latent_format = latent or Latent()
    patcher = NS(model=inner, load_device="cpu", model_options={}, sigma=sigma)
    sampling = Sampling(patcher)
    patcher.get_model_object = lambda _: sampling
    patcher.clone = lambda: patcher
    return patcher


def sampler_runtime():
    records = []

    def backend(model, noise, positive, negative, cfg, device, sampler, sigmas, options, **kwargs):
        records.append(
            dict(
                model=model,
                noise=noise,
                positive=positive,
                negative=negative,
                cfg=cfg,
                sigmas=sigmas.clone(),
                **kwargs,
            )
        )
        output = kwargs["latent_image"] + sigmas[-1]
        for index in range(len(sigmas) - 1):
            kwargs["callback"](index, output, output, len(sigmas) - 1)
        return output

    scope = dict(
        torch=torch,
        sample=backend,
        sampler_object=lambda name: name,
        calculate_sigmas=lambda sampling, _scheduler, steps: torch.linspace(
            sampling.sigma, 0, steps + 1
        ),
    )
    sampler = native_class(
        "comfy/samplers.py",
        "KSampler",
        scope,
        methods=("__init__", "calculate_sigmas", "set_steps", "sample"),
    )
    sampler.SAMPLERS = ["euler", "lcm"]
    sampler.SCHEDULERS = ["normal"]
    sampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS = set()
    comfy = NS(
        samplers=NS(KSampler=sampler),
        model_management=NS(
            intermediate_device=lambda: "cpu", intermediate_dtype=lambda: torch.float32
        ),
    )
    scope = dict(torch=torch, comfy=comfy)
    sample = native_function("comfy/sample.py", "sample", scope)
    noise = native_function("comfy/sample.py", "prepare_empty_noise", scope)

    def encode(tokens, add_dict=None):
        return [
            [
                torch.zeros(1, 2, 1280),
                dict(pooled_output=torch.zeros(1, 1280), text=tokens, **(add_dict or {})),
            ]
        ]

    clip = NS(tokenize=lambda text: text, encode_from_tokens_scheduled=encode)
    values = lambda conditioning, options: [
        [tensor, dict(previous, **options)] for tensor, previous in conditioning
    ]
    modules = {
        "comfy.sample": NS(sample=sample, prepare_empty_noise=noise),
        "comfy.samplers": NS(KSampler=sampler),
        "comfy_extras.nodes_clip_sdxl": native_refiner_node(),
        "nodes": NS(CLIPSetLastLayer=lambda: NS(set_last_layer=lambda clip, _: (clip,))),
        "node_helpers": NS(conditioning_set_values=values),
    }
    return modules, records, clip


class SDXLHandoffTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.root = root
        (root / "checkpoints").mkdir()
        (root / "clip").mkdir()
        self.base_path = write_header(
            root / "checkpoints/base.safetensors",
            {"conditioner.embedders.0.fixture": [1], "first_stage_model.fixture": [1]},
        )
        self.refiner_path = write_header(
            root / "checkpoints/refiner.safetensors", {"conditioner.embedders.0.fixture": [1]}
        )
        self.manager = Manager(root)
        self.manager.model_paths = {"embedding_path": [], "cache_path": root}
        self.manager.EXTENSIONS = (".safetensors", ".gguf")
        self.families = {self.base_path.name: "SDXL", self.refiner_path.name: "SDXLRefiner"}
        self.models = NS(
            get_file=lambda category, name: (
                (root / category / name) if name and (root / category / name).is_file() else None
            ),
            detect_checkpoint_base=lambda path: self.families.get(Path(path).name, "Other"),
            get_names=lambda _: list(self.families),
            get_models_by_path=lambda category, name, **kwargs: {
                "baseModel": self.families.get(name, "Other")
            },
            get_model_base=lambda info: info["baseModel"],
        )
        self.data = dict(
            base_model_name=self.base_path.name,
            sdxl_refiner_model=self.refiner_path.name,
            sdxl_refiner_switch_step=2,
            sdxl_refiner_cfg=0,
            steps=4,
            cfg=7,
            sampler_name="euler",
            scheduler="normal",
            width=64,
            height=80,
            positive_prompt='a sign reads "OPEN"',
            negative_prompt="blur",
            clip_skip=1,
        )
        self.modules, self.records, self.clip = sampler_runtime()
        self.enterContext(patch.dict("sys.modules", self.modules))
        self.enterContext(
            patch("ruinedfooocus.backends.comfy.text_encoder_warmup.warm_patched_text_encoder")
        )
        self.enterContext(
            patch(
                "ruinedfooocus.backends.comfy.embedding_paths.register_embedding_paths",
                return_value=[],
            )
        )
        self.enterContext(
            patch(
                "ruinedfooocus.features.models.embedding_browser.native_embedding_text",
                side_effect=lambda text, *args: text,
            )
        )
        self.base, self.refiner = model("SDXL"), model("SDXLRefiner")
        self.owner = NS(
            xl_base_patched=NS(unet=self.base), base_file_signature=file_signature(self.base_path)
        )

    def captured(self):
        result, artifacts = native.capture_assets(self.data, self.manager, self.models)
        return result | {preparation.KEY: preparation.ImageAssets(artifacts, True)}

    def sample(self, data=None, **kwargs):
        data = self.captured() if data is None else data
        with patch.dict(
            "sys.modules",
            {
                "comfy.sd": NS(
                    load_checkpoint_guess_config=Mock(
                        return_value=(self.refiner, self.clip, None, None)
                    ),
                    load_clip=Mock(return_value=self.clip),
                    CLIPType=NS(STABLE_DIFFUSION="STABLE_DIFFUSION"),
                )
            },
        ):
            return native.sample_handoff(
                self.owner,
                model=self.base,
                noise=torch.ones(1, 4, 10, 8),
                steps=4,
                cfg=7,
                sampler_name=kwargs.pop("sampler_name", "euler"),
                scheduler="normal",
                positive=[["base positive", {}]],
                negative=[["base negative", {}]],
                latent_image=kwargs.pop("latent_image", torch.zeros(1, 4, 10, 8)),
                data=data,
                manager=self.manager,
                check_cancelled=kwargs.pop("check_cancelled", lambda: None),
                seed=123,
                **kwargs,
            )

    def test_native_bodies_keep_nonzero_split_sigma_and_never_add_refiner_noise(self):
        data = self.captured()
        callback = Mock()
        mask = torch.ones(1, 10, 8)
        self.sample(data, denoise=0.5, noise_mask=mask, callback=callback)
        first, second = self.records
        self.assertEqual(first["sigmas"].tolist(), [0.5, 0.375, 0.25])
        self.assertEqual(second["sigmas"].tolist(), [0.25, 0.125, 0.0])
        self.assertTrue(torch.equal(second["latent_image"], torch.full((1, 4, 10, 8), 0.25)))
        self.assertFalse(torch.count_nonzero(second["noise"]))
        self.assertEqual(second["cfg"], 0)
        self.assertEqual(first["seed"], second["seed"])
        self.assertIs(first["denoise_mask"], second["denoise_mask"])
        self.assertEqual([call.args[0] for call in callback.call_args_list], [0, 1, 2, 3])
        self.assertTrue(all(call.args[-1] == 4 for call in callback.call_args_list))
        positive = second["positive"][0][1]
        negative = second["negative"][0][1]
        self.assertEqual(positive["text"], 'a sign reads "OPEN"')
        self.assertEqual(
            (positive["width"], positive["height"], positive["aesthetic_score"]), (64, 80, 6.0)
        )
        self.assertEqual(negative["aesthetic_score"], 2.5)
        self.assertEqual(data["sdxl_refiner_source"]["denoise"], 0.5)
        self.assertFalse(data["sdxl_refiner_source"]["refiner_add_noise"])

    def test_scheduled_prompts_use_full_shared_sigma_intervals(self):
        self.data["positive_prompt"] = "[red|blue]"
        self.sample()
        positive = self.records[1]["positive"]
        self.assertEqual([row[1]["text"] for row in positive], ["red", "blue", "red", "blue"])
        self.assertEqual(positive[2][1]["timestep_start"], 0.5)
        self.assertEqual(positive[-1][1]["timestep_end"], 0)

    def test_lcm_preserves_native_noise_controls_and_the_remaining_schedule(self):
        self.data["sampler_name"] = "lcm"
        data = self.captured()

        def custom(**kwargs):
            return kwargs["latent_image"] + kwargs["sigmas"][-1]

        custom_sample = Mock(side_effect=custom)
        builder = Mock(side_effect=lambda name, options: (name, options))
        sampler = native_class(
            "comfy_extras/nodes_advanced_samplers.py",
            "SamplerLCM",
            dict(comfy=NS(samplers=NS(ksampler=builder)), io=NS(NodeOutput=lambda value: (value,))),
            methods=("execute",),
        )
        options = dict(s_noise=0.0, s_noise_end=0.25, noise_clip_std=1.5)
        with patch.dict(
            "sys.modules",
            {
                "comfy.sample": NS(
                    sample_custom=custom_sample,
                    sample=Mock(side_effect=AssertionError("LCM lost native noise controls")),
                    prepare_empty_noise=torch.zeros_like,
                ),
                "comfy_extras.nodes_advanced_samplers": NS(SamplerLCM=sampler),
            },
        ):
            self.sample(data, sampler_name="lcm", noise_options=options)
        first, second = [call.kwargs for call in custom_sample.call_args_list]
        self.assertEqual(builder.call_args_list[0].args, ("lcm", options))
        self.assertEqual(first["sigmas"].tolist(), [1.0, 0.75, 0.5])
        self.assertEqual(second["sigmas"].tolist(), [0.5, 0.25, 0.0])
        self.assertFalse(torch.count_nonzero(second["noise"]))
        self.assertTrue(torch.equal(second["latent_image"], torch.full((1, 4, 10, 8), 0.5)))

    def test_zero_denoise_skips_loading_and_sampling_and_preserves_original_tensor(self):
        pixels = torch.randn(1, 4, 10, 8)
        data = self.captured()
        with patch.object(
            native, "refiner_scope", side_effect=AssertionError("zero denoise loaded refiner")
        ):
            self.assertIs(self.sample(data, denoise=0, latent_image=pixels), pixels)
        self.assertFalse(self.records)
        self.assertTrue(data["sdxl_refiner_source"]["skipped_denoise_zero"])

    def test_cancel_between_stages_releases_all_read_leases_and_stops_refiner(self):
        data = self.captured()
        held = []

        @contextmanager
        def lease(data, role, name, manager, category):
            held.append(role)
            try:
                yield self.manager.get_model_folders(category)[0] / name
            finally:
                held.remove(role)

        cancelled = False

        def callback(step, *args):
            nonlocal cancelled
            if step == 1:
                cancelled = True

        def check():
            if cancelled:
                raise CancelledError()

        with patch.object(preparation, "prepared_asset", lease):
            with self.assertRaises(CancelledError):
                self.sample(data, check_cancelled=check, callback=callback)
        self.assertEqual(len(self.records), 1)
        self.assertFalse(held)

    def test_runtime_pair_and_sigma_mismatches_fail_before_sampling(self):
        for refiner in (
            model("SDXL"),
            model("SDXLRefiner", channels=9),
            model("SDXLRefiner", prediction="V"),
            model("SDXLRefiner", latent=NS(latent_channels=4, scale_factor=0.5)),
        ):
            with self.subTest(refiner=refiner), self.assertRaises(ValueError):
                native.validate_pair(self.base, refiner)
        self.refiner.sigma = 2
        with self.assertRaisesRegex(ValueError, "sigma schedules differ"):
            self.sample()
        self.assertFalse(self.records)
        self.refiner.get_model_object("model_sampling").calculate_denoised = lambda *args: None
        with self.assertRaisesRegex(ValueError, "active SDXL base and refiner prediction"):
            native.validate_pair(self.base, self.refiner)

    def test_capture_freezes_both_checkpoints_and_refuses_replacement_before_loading(self):
        captured = preparation.capture_image_assets(
            self.data | dict(prompt="cat"), self.manager, {}, offline=True, models=self.models
        )
        self.assertIsInstance(captured[native.KEY], native.RefinerPlan)
        self.assertTrue(
            {"sdxl-refiner-base", "sdxl-refiner-model"}.issubset(
                {item.role for item in captured[preparation.KEY].artifacts}
            )
        )
        self.refiner_path.write_bytes(b"replaced")
        with self.assertRaisesRegex(ValueError, "changed after submission"):
            self.sample(captured)
        self.assertFalse(self.records)

    def test_split_refiner_captures_and_loads_the_selected_single_g_encoder(self):
        write_header(self.refiner_path, {"unet.fixture": [1]})
        clip_path = write_header(
            self.root / "clip/selected-G.safetensors", {"unknown.fixture": [1]}
        )
        self.data["sdxl_refiner_clip"] = clip_path.name
        captured = self.captured()
        self.assertEqual(captured[native.KEY].clip_name, clip_path.name)
        self.assertIn(
            "sdxl-refiner-clip", {item.role for item in captured[preparation.KEY].artifacts}
        )
        loader = Mock(return_value=self.clip)
        with patch.dict(
            "sys.modules",
            {
                "comfy.sd": NS(
                    load_checkpoint_guess_config=Mock(
                        return_value=(self.refiner, None, None, None)
                    ),
                    load_clip=loader,
                    CLIPType=NS(STABLE_DIFFUSION="STABLE_DIFFUSION"),
                )
            },
        ):
            with native.refiner_scope(
                self.owner, captured[native.KEY], captured, self.manager, lambda: None
            ):
                pass
        self.assertEqual(loader.call_args.kwargs["ckpt_paths"], [str(clip_path)])
        self.assertEqual(loader.call_args.kwargs["clip_type"], "STABLE_DIFFUSION")

    def test_refiner_cache_reuses_only_current_captured_sources_and_cancel_does_not_publish(self):
        captured = self.captured()
        loader = Mock(return_value=(self.refiner, self.clip, None, None))
        modules = {
            "comfy.sd": NS(
                load_checkpoint_guess_config=loader,
                load_clip=Mock(),
                CLIPType=NS(STABLE_DIFFUSION="STABLE_DIFFUSION"),
            )
        }
        with patch.dict("sys.modules", modules):

            def cancelled_after_load():
                if loader.called:
                    raise CancelledError()

            with self.assertRaises(CancelledError):
                with native.refiner_scope(
                    self.owner, captured[native.KEY], captured, self.manager, cancelled_after_load
                ):
                    self.fail("cancelled refiner was published")
            self.assertFalse(hasattr(self.owner, "sdxl_refiner_cache"))
            for _ in range(2):
                with native.refiner_scope(
                    self.owner, captured[native.KEY], captured, self.manager, lambda: None
                ):
                    pass
            self.assertEqual(loader.call_count, 2)
        self.owner.base_file_signature = ["different loaded base"]
        with self.assertRaisesRegex(ValueError, "loaded SDXL base differs"):
            self.sample(captured)

    def test_wrong_clip_features_and_mutated_queued_selection_fail_before_sampling(self):
        self.clip.encode_from_tokens_scheduled = lambda tokens, add_dict=None: [
            [torch.zeros(1, 2, 768), dict(pooled_output=torch.zeros(1, 768))]
        ]
        with self.assertRaisesRegex(ValueError, "native CLIP-G"):
            self.sample()
        captured = self.captured()
        captured["sdxl_refiner_switch_step"] = 3
        with self.assertRaisesRegex(ValueError, "exact SDXL base/refiner selection"):
            self.sample(captured)
        self.assertFalse(self.records)

    def test_explicit_selection_split_and_cfg_validation(self):
        for values in (
            dict(sdxl_refiner_switch_step=0),
            dict(sdxl_refiner_switch_step=4),
            dict(sdxl_refiner_switch_step=True),
            dict(sdxl_refiner_cfg=float("nan")),
            dict(sdxl_refiner_model="../refiner.safetensors"),
            dict(image_engine="sdcpp-vulkan"),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                native.capture_assets(self.data | values, self.manager, self.models)
        self.families[self.base_path.name] = "Flux"
        with self.assertRaisesRegex(ValueError, "ordinary SDXL base"):
            native.capture_assets(self.data, self.manager, self.models)

    def test_actual_capture_refuses_explicit_optional_engine_and_ignores_utility_refiner_fields(
        self,
    ):
        with self.assertRaisesRegex(ValueError, "native image engine"):
            preparation.capture_image_assets(
                self.data | dict(prompt="cat", image_engine="sdcpp-vulkan"),
                self.manager,
                {},
                models=self.models,
            )
        for kind in ("rembg", "upscale", "detect", "segment", "depth-map"):
            data, artifacts = native.capture_assets(
                self.data | dict(controlnet={"type": kind}), self.manager, None
            )
            self.assertFalse(artifacts)
            self.assertNotIn(native.KEY, data)
        from ruinedfooocus.features.generation import native_sampling

        (self.root / "upscalers").mkdir()
        (self.root / "upscalers" / "owned-upscaler.pth").write_bytes(b"owned fixture")
        for kind in ("rembg", "upscale"):
            with (
                self.subTest(kind=kind),
                patch.object(native, "capture_assets") as refiner_capture,
                patch.object(native_sampling, "capture_assets") as sampling_capture,
            ):
                preparation.capture_image_assets(
                    self.data | dict(prompt="cat", scheduler="Ideogram4Scheduler"),
                    self.manager,
                    {"type": kind, "upscaler": "owned-upscaler.pth"},
                    models=self.models,
                )
            refiner_capture.assert_not_called()
            sampling_capture.assert_not_called()

    def test_api_and_metadata_replay_preserve_zero_and_do_not_reuse_runtime_proof(self):
        from ruinedfooocus.integrations.openai_api import ImageRequest
        from ruinedfooocus.features.gallery.image_metadata import render_values

        values = {key: self.data[key] for key in native.FIELDS if key in self.data}
        request = ImageRequest(prompt="cat", **values)
        self.assertEqual(request.sdxl_refiner_cfg, 0)
        with self.assertRaises(ValidationError):
            ImageRequest(prompt="cat", sdxl_refiner_switch_step=True)
        metadata = dict(prompt="cat", **values)
        self.assertEqual({key: render_values(metadata)[key] for key in values}, values)
        replay = preparation.image_control_data(dict(prompt=json.dumps(metadata)))
        self.assertEqual({key: replay[key] for key in values}, values)
        result, _ = native.capture_assets(
            self.data | dict(sdxl_refiner_source={"old": True}), self.manager, self.models
        )
        self.assertNotIn("sdxl_refiner_source", result)

    def test_actual_production_sampler_caller_uses_refiner_output_and_native_arguments(self):
        patch_scope = next(
            node
            for node in process_nodes()
            if isinstance(node, ast.With) and "model_patches.apply_scope" in ast.unparse(node)
        )
        caller = next(
            node
            for node in patch_scope.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "gen_data.get('sdxl_refiner_model')"
        )
        call = Mock(return_value=torch.ones(1, 4, 10, 8))
        scope = dict(
            self=self.owner,
            gen_data=self.data,
            sampling_model=self.base,
            noise="noise",
            cfg=7,
            sampler_name="euler",
            scheduler="normal",
            positive_cond="base+",
            negative_conditioning="base-",
            latent_image="latent",
            path_manager=self.manager,
            worker=NS(check_interrupt=Mock()),
            denoise=0.6,
            noise_mask="mask",
            noise_options=None,
            positive_prompt="exact positive",
            negative_prompt="exact negative",
            callback_function=Mock(),
            comfy=NS(utils=NS(PROGRESS_BAR_ENABLED=False)),
            seed=42,
        )
        with patch.object(native, "sample_handoff", call):
            exec(
                compile(
                    ast.Module(body=[caller], type_ignores=[]),
                    "production SDXL handoff caller",
                    "exec",
                ),
                scope,
            )
        self.assertIs(scope["samples"], call.return_value)
        self.assertEqual(call.call_args.kwargs["prompts"], ("exact positive", "exact negative"))
        self.assertEqual(call.call_args.kwargs["denoise"], 0.6)
        self.assertIs(call.call_args.kwargs["data"], self.data)

        body = process_nodes()
        start = next(
            index
            for index, node in enumerate(body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "sampled_latent"
                for target in node.targets
            )
        )
        build_latent = body[start : start + 4]
        decode = next(
            node
            for node in body[start:]
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "architecture == 'StableCascade_C'"
        )
        self.owner.decode_latent = Mock(return_value="final pixels")
        scope.update(
            latent={"samples": "initial latent"},
            layered_model=False,
            architecture="SDXL",
            guidance=dict(vae_tiled=False, vae_tile_size=512, vae_overlap=64),
        )
        exec(
            compile(
                ast.Module(body=[*build_latent, decode], type_ignores=[]),
                "production SDXL final decode",
                "exec",
            ),
            scope,
        )
        self.owner.decode_latent.assert_called_once()
        self.assertIs(self.owner.decode_latent.call_args.args[0]["samples"], call.return_value)
        self.assertEqual(scope["decoded_latent"], "final pixels")

    def test_source_operation_ledger_keeps_distinct_cascade_source_and_control_packets(self):
        from tools.audit_native_model_coverage import OPERATION_FAMILIES

        rows = {row[0]: row for row in OPERATION_FAMILIES}
        self.assertEqual(
            rows["Cascade image-to-image and paired stage grids"][2], "source-route-present"
        )
        self.assertEqual(
            rows["Cascade super-resolution control and vision mixing"][2], "source-route-present"
        )
        self.assertEqual(rows["SDXL base/refiner handoff"][2], "source-route-present")


if __name__ == "__main__":
    unittest.main()
