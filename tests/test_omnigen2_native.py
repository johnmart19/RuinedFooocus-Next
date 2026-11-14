"""OmniGen2 uses pinned native guidance/scheduling with fake weights and CPU tensors."""

import ast
from concurrent.futures import CancelledError
import json
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import math
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.core.preparation import PreparationService
from ruinedfooocus.features.generation import image_preparation
from ruinedfooocus.features.generation.image_sampling import sample_omnigen2
from ruinedfooocus.features.models import component_profiles
from ruinedfooocus.features.models.component_candidates import compatibility
from ruinedfooocus.features.models.image_components import inventory
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.features.models.performance import OMNIGEN2_FP16_SHA256, UnavailableModelRecipe
from ruinedfooocus.features.models.resource_defaults import RESOURCE_DEFAULTS
from test_image_preparation import Manager
from test_model_cache import method
import test_model_recommendations as recommendation_fixtures
from test_model_recommendations import SAMPLERS, SCHEDULERS
from test_native_model_coverage import NATIVE, PIPELINE, native_method, route_info


def native_function(filename, name, scope):
    tree = ast.parse((NATIVE / filename).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), scope)
    return scope[name]


def native_class(filename, name, names, scope, *, keep_bases=False):
    tree = ast.parse((NATIVE / filename).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if not keep_bases:
        node.bases = []
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for method in node.body:
        method.returns = None
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), scope)
    return scope[name]


class FakeModel:
    def __init__(self):
        self.model_options = {}
        self.calls = []

    def clone(self):
        result = FakeModel()
        result.model_options = self.model_options.copy()
        return result

    def set_model_sampler_cfg_function(self, function):
        self.model_options["sampler_cfg_function"] = function

    def set_model_sampler_post_cfg_function(self, function):
        self.model_options["sampler_post_cfg_function"] = [function]

    def get_model_object(self, name):
        return name


class OmniGen2NativeTests(unittest.TestCase):
    def test_native_family_corrects_stale_sampling_label_without_rewriting_publisher_cache(self):
        lookup = method(
            "ruinedfooocus/features/models/model_handler.py",
            "get_models_by_path",
            dict(json=json, Path=Path),
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "arbitrary.safetensors"
            model.write_bytes(b"identity-only fixture")
            publisher = dict(baseModel="Flux.1 D", name="Publisher title", modelId=42)
            cache = root / "arbitrary.json"
            cache.write_text(json.dumps(publisher))
            before = cache.read_bytes()
            owner = SimpleNamespace(
                cache_paths={"checkpoints": root},
                offline=True,
                detect_checkpoint_base=lambda path: "Omnigen2",
            )
            self.assertEqual(
                lookup(owner, "checkpoints", model), publisher | dict(baseModel="Omnigen2")
            )
            self.assertEqual(cache.read_bytes(), before)

    def test_native_single_encoder_dispatch_and_tokenizer_keep_omnigen2_template(self):
        tree = ast.parse((NATIVE / "comfy/sd.py").read_text(encoding="utf-8"))
        branch = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "te_model == TEModel.QWEN25_3B"
        )
        factory = Mock(return_value="native-encoder")
        scope = dict(
            clip_target=SimpleNamespace(),
            clip_data=["3b-weights"],
            llama_detect=lambda data: dict(dtype_llama="native-dtype"),
            comfy=SimpleNamespace(
                text_encoders=SimpleNamespace(
                    omnigen2=SimpleNamespace(te=factory, Omnigen2Tokenizer="native-tokenizer")
                )
            ),
        )
        exec(
            compile(ast.Module(body=branch.body, type_ignores=[]), "native_omnigen2_clip", "exec"),
            scope,
        )
        factory.assert_called_once_with(dtype_llama="native-dtype")
        self.assertEqual(scope["clip_target"].tokenizer, "native-tokenizer")

        class BaseTokenizer:
            def __init__(self, **kwargs):
                self.arguments = kwargs

            def tokenize_with_weights(self, text, **kwargs):
                return text, kwargs

        tokenizer_scope = dict(
            sd1_clip=SimpleNamespace(SD1Tokenizer=BaseTokenizer),
            Qwen25_3BTokenizer="native-3b-tokenizer",
        )
        tokenizer_type = native_class(
            "comfy/text_encoders/omnigen2.py",
            "Omnigen2Tokenizer",
            ("__init__", "tokenize_with_weights"),
            tokenizer_scope,
            keep_bases=True,
        )
        tokenizer = tokenizer_type()
        text, options = tokenizer.tokenize_with_weights("a cat")
        self.assertEqual(text, tokenizer.llama_template.format("a cat"))
        self.assertEqual(tokenizer.arguments["name"], "qwen25_3b")
        self.assertIn("generates high-quality images", text)
        self.assertEqual(options, dict(return_word_ids=False))

    def test_inventory_loader_and_frozen_companions_use_dedicated_3b_and_existing_flux_vae(self):
        keys = ["clip_omnigen2_3b", "vae_flux"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manager = Manager(root)
            model = root / "arbitrary.safetensors"
            header = json.dumps(dict(weight=dict(shape=[1]))).encode()
            model.write_bytes(len(header).to_bytes(8, "little") + header)
            models = SimpleNamespace(
                get_file=lambda *args: model,
                get_model_base=lambda value: "Omnigen2",
                get_models_by_path=lambda *args, **kwargs: dict(baseModel="Omnigen2"),
            )
            paths = {}
            for key in keys:
                category = "clip" if key.startswith("clip_") else "vae"
                path = root / category / RESOURCE_DEFAULTS[key]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"owned installed fixture")
                paths[key] = path.resolve()
            rows = inventory(model, "Omnigen2", {}, manager)["rows"]
            self.assertEqual([row["key"] for row in rows], keys)
            self.assertTrue(all(row["status"] == "Installed" for row in rows))
            runtime = type("Omnigen2", (), {"model_config": SimpleNamespace(unet_config={})})()
            info = method(
                PIPELINE, "get_clip_and_vae", dict(settings=SimpleNamespace(default_settings={}))
            )(SimpleNamespace(known_model_info=route_info()), SimpleNamespace(model=runtime))
            self.assertEqual((info["clip_type"], info["latent"]), ("OMNIGEN2", "SD3"))
            self.assertEqual((info["clip_keys"], info["vae_key"]), (keys[:-1], keys[-1]))
            request = component_profiles.snapshot_request(
                dict(base_model_name=model.name), models, {}, manager
            )
            self.assertEqual(request[component_profiles.REQUEST_KEY], dict.fromkeys(keys))
            request = image_preparation.capture_image_assets(
                request, manager, {}, models=models, values={}, offline=True
            )
            self.assertFalse(image_preparation.needs_image_preparation(request))
            transfer = Mock(side_effect=AssertionError("Installed fixtures must stay offline"))
            service = PreparationService(transport=transfer, staging_root=root / "staging")
            try:
                ready = image_preparation.prepare_image_assets(
                    request, check_cancelled=lambda: None, service=service
                )
                manager.root = root / "changed-after-submission"
                for key in keys:
                    category = "clip" if key.startswith("clip_") else "vae"
                    with image_preparation.prepared_asset(
                        ready, category, key, manager, category, service=service
                    ) as actual:
                        self.assertEqual(actual, paths[key])
                transfer.assert_not_called()
            finally:
                service._executor.shutdown(wait=True)

    def test_encoder_registry_has_exact_official_identity_and_rejects_7b_substitution(self):
        key = "clip/qwen_2.5_vl_fp16.safetensors"
        catalogue = json.loads(
            resource_path("model_catalogue", "native_image_components.json").read_text()
        )
        download = catalogue[key]
        annotation = load_registry().annotations_for(key, download)
        self.assertEqual(RESOURCE_DEFAULTS["clip_omnigen2_3b"], key.split("/")[1])
        self.assertIn("/resolve/babe908cd4ed58efc7359e9e35b426cd73071949/", download["url"])
        self.assertEqual(annotation["artifact"]["bytes"], 7509337224)
        self.assertEqual(
            annotation["artifact"]["sha256"],
            "ba05dd266ad6a6aa90f7b2936e4e775d801fb233540585b43933647f8bc4fbc3",
        )
        self.assertFalse(annotation["artifact"]["full_file_hash_recomputed"])
        self.assertEqual(annotation["artifact"]["availability"], "publisher_metadata_only")
        self.assertFalse(annotation["chat_weight"])
        self.assertEqual(
            compatibility("clip_omnigen2_3b", dict(text_type="QWEN25_3B")), "compatible"
        )
        self.assertEqual(
            compatibility("clip_omnigen2_3b", dict(text_type="QWEN25_7B")), "incompatible"
        )
        self.assertEqual(compatibility("clip_qwen25", dict(text_type="QWEN25_3B")), "incompatible")

    def test_native_attention_mask_num_tokens_and_cross_attention_are_preserved(self):
        class BaseModel:
            def extra_conds(self, **kwargs):
                return {}

        wrapped = lambda value: SimpleNamespace(cond=value)
        scope = dict(
            torch=torch,
            BaseModel=BaseModel,
            comfy=SimpleNamespace(conds=SimpleNamespace(CONDRegular=wrapped, CONDConstant=wrapped)),
        )
        model = native_method(
            "comfy/model_base.py", "Omnigen2", "extra_conds", scope, bases=("BaseModel",)
        )()
        context = torch.ones(1, 4, 2048)
        for mask in (torch.tensor([[1, 1, 0, 0]]), torch.ones(1, 4), torch.zeros(1, 4)):
            result = model.extra_conds(attention_mask=mask, cross_attn=context)
            self.assertIs(result["c_crossattn"].cond, context)
            self.assertEqual(result["num_tokens"].cond, max(1, mask.sum().item()))
            if mask.sum() != mask.numel():
                self.assertIs(result["attention_mask"].cond, mask)
            else:
                self.assertNotIn("attention_mask", result)

    def test_actual_process_uses_native_latent_preserves_attention_cfg_and_decode(self):
        memory = SimpleNamespace(
            intermediate_device=lambda: "cpu",
            intermediate_dtype=lambda: torch.float32,
            get_torch_device=lambda: "cpu",
        )
        native_scope = dict(
            torch=torch,
            comfy=SimpleNamespace(model_management=memory),
            io=SimpleNamespace(NodeOutput=lambda *values: values),
        )
        empty = native_class(
            "comfy_extras/nodes_sd3.py", "EmptySD3LatentImage", ("execute",), native_scope
        )
        fix = native_function("comfy/sample.py", "fix_empty_latent_channels", native_scope)
        encode = native_class("nodes.py", "CLIPTextEncode", ("encode",), {})
        latent_format = SimpleNamespace(
            latent_channels=16,
            latent_dimensions=2,
            spacial_downscale_ratio=8,
            fix_empty_latent=lambda latent: latent,
        )
        native = type(
            "Omnigen2",
            (),
            {"model_config": SimpleNamespace(unet_config={}), "latent_format": latent_format},
        )()
        unet = SimpleNamespace(model=native, get_model_object=lambda name: latent_format)
        context, mask = torch.ones(1, 4, 2048), torch.tensor([[1, 1, 0, 0]])
        clip = SimpleNamespace(
            tokenize=lambda text: text,
            encode_from_tokens_scheduled=lambda tokens: [
                [context, dict(attention_mask=mask, prompt=tokens)]
            ],
        )
        owner = SimpleNamespace(
            xl_base_patched=SimpleNamespace(unet=unet, clip=clip, vae=SimpleNamespace()),
            xl_base=SimpleNamespace(unet=unet),
            known_models=route_info().keys(),
            model_info=dict(latent="SD3"),
            conditions={key: dict(text=None, cache=None) for key in ("+", "-", "switch")},
            decode_latent=Mock(return_value=torch.ones(1, 32, 48, 3)),
        )
        owner.textencode = MethodType(
            method(
                PIPELINE,
                "textencode",
                dict(
                    CLIPTextEncode=encode,
                    warm_patched_text_encoder=Mock(),
                    path_manager=SimpleNamespace(
                        model_paths=dict(embedding_path=[], cache_path=Path("unused"))
                    ),
                ),
            ),
            owner,
        )
        guidance = dict(
            moss_image_edit_format="Disabled",
            moss_image_edit_strength=1,
            vae_tiled=False,
            vae_tile_size=512,
            vae_overlap=64,
        )
        scope = dict(
            torch=torch,
            np=np,
            settings=SimpleNamespace(default_settings={}),
            resolve_guidance=lambda *args: guidance,
            request_model=lambda *args, **kwargs: unet,
            effective_guidance=lambda *args: {},
            normalize_reference_images=lambda *args: [],
            control_tools=SimpleNamespace(get_settings=lambda data: {}, controlnet_models={}),
            comfy=SimpleNamespace(
                model_management=memory,
                sample=SimpleNamespace(prepare_noise=lambda latent, *args: torch.ones_like(latent)),
                utils=SimpleNamespace(
                    PROGRESS_BAR_ENABLED=False,
                    ProgressBar=lambda steps: SimpleNamespace(update_absolute=Mock()),
                ),
            ),
            pp=SimpleNamespace(prompt_switch_per_step=lambda *args: []),
            Flux=type("Flux", (), {}),
            EmptySD3LatentImage=empty,
            fix_empty_latent_channels=fix,
            restore_reference_frame=lambda pixels, frame: pixels,
            # No reference adapter is active, so apply_reference never reads folders.
            path_manager=SimpleNamespace(),
            reference_tensor=lambda image, *size: image,
        )
        process = method(PIPELINE, "process", scope)
        request = dict(
            cfg=3.25,
            sampler_name="euler",
            scheduler="simple",
            clip_skip=1,
            seed=77,
            positive_prompt="a cat",
            negative_prompt="blur",
            steps=4,
            width=48,
            height=32,
            sampling_shift=1,
        )
        sample = Mock(side_effect=lambda **kwargs: kwargs["latent_image"] + 0.5)
        with (
            patch("ruinedfooocus.features.generation.image_sampling.sample_omnigen2", sample),
            patch(
                "ruinedfooocus.features.generation.automatic_upscale.upscale_decoded",
                side_effect=lambda owner, pixels, data: pixels,
            ),
            patch(
                "ruinedfooocus.features.models.embedding_browser.native_embedding_text",
                side_effect=lambda text, *args: text,
            ),
        ):
            output = process(owner, request)
        call = sample.call_args.kwargs
        self.assertEqual(tuple(call["latent_image"].shape), (1, 16, 4, 6))
        self.assertEqual((call["cfg"], call["seed"], call["steps"]), (3.25, 77, 4))
        self.assertIs(call["model"], unet)
        self.assertIs(call["positive"][0][1]["attention_mask"], mask)
        self.assertIs(call["negative"][0][1]["attention_mask"], mask)
        self.assertIs(call["positive"][0][0], context)
        self.assertIsNone(call["denoise"])
        self.assertTrue((call["noise"] == 1).all())
        self.assertTrue((owner.decode_latent.call_args.args[0]["samples"] == 0.5).all())
        self.assertEqual(output[0].shape, (32, 48, 3))
        self.assertTrue((output[0] == 255).all())

    def test_native_auto_baseline_independent_of_payload_identity_preserves_recipes(self):
        registry = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=SAMPLERS, SCHEDULERS=SCHEDULERS)
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            # Native readiness also requires model detection to be importable.
            patch.dict(
                "sys.modules",
                {"comfy.samplers": registry, "comfy.model_detection": SimpleNamespace()},
            ),
            patch(
                "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                return_value={
                    "architecture": "Omnigen2",
                    "native_class": "Omnigen2",
                    "native_config": dict(
                        image_model="omnigen2",
                        hidden_size=2520,
                        num_layers=32,
                        in_channels=16,
                        patch_size=2,
                    ),
                },
            ),
        ):
            root = Path(folder)
            model = root / "arbitrary-name.safetensors"
            model.write_bytes(b"simulated already-hashed local payload")
            metadata = dict(
                baseModel="Omnigen2",
                _rf_local_sha256=OMNIGEN2_FP16_SHA256,
                _rf_file_signature=file_signature(model),
            )
            cache = root / (model.stem + ".json")
            cache.write_text(json.dumps(metadata))
            performance = recommendation_fixtures.RecommendationTests()._performance(root)
            performance.model_recommendations.offline = True
            values = performance.get_perf_options("Model Defaults", str(model))
            self.assertEqual(
                tuple(
                    values[key]
                    for key in ("custom_steps", "cfg", "sampler_name", "scheduler", "clip_skip")
                ),
                (20, 5, "euler", "simple", 1),
            )
            self.assertNotIn("sampling_shift", values)
            manual = dict(
                performance_selection="Own settings",
                base_model_name=str(model),
                cfg=3.25,
                custom_steps=7,
            )
            self.assertEqual(performance.apply(manual, "Omnigen2"), manual)
            performance.model_recommendations.save_override(
                str(model),
                dict(custom_steps=9, cfg=2, sampler_name="euler", scheduler="normal", clip_skip=2),
                SAMPLERS,
                SCHEDULERS,
            )
            self.assertEqual(performance.get_perf_options("Model Defaults", str(model))["cfg"], 2)
            performance.model_recommendations.reset_override(str(model))
            cache.write_text(
                json.dumps(metadata | dict(generationConfig=dict(steps=12, cfgScale=4)))
            )
            values = performance.get_perf_options("Model Defaults", str(model))
            self.assertEqual((values["custom_steps"], values["cfg"]), (12, 4))
            for identity in (
                dict(_rf_local_sha256="c" * 64),
                dict(_rf_local_sha256=None, files=[dict(hashes=dict(SHA256=OMNIGEN2_FP16_SHA256))]),
                dict(_rf_file_signature=None),
            ):
                cache.write_text(json.dumps(metadata | identity))
                self.assertEqual(
                    performance.get_perf_options("Model Defaults", str(model))["cfg"], 5
                )
            cache.write_text(json.dumps(metadata))
            models = SimpleNamespace(get_file=lambda *args: model)
            with patch.dict(
                "sys.modules", {"ruinedfooocus.app.state": SimpleNamespace(models=models)}
            ):
                self.assertEqual(
                    performance.get_perf_options("Model Defaults", model.name)["cfg"], 5
                )
                moved = root / "other" / model.name
                moved.parent.mkdir()
                moved.write_bytes(model.read_bytes())
                models.get_file = lambda *args: moved
                self.assertEqual(
                    performance.get_perf_options("Model Defaults", model.name, "Omnigen2")["cfg"], 5
                )
            model.write_bytes(model.read_bytes() + b"replacement")
            self.assertEqual(
                performance.get_perf_options("Model Defaults", str(model), "Omnigen2")["cfg"], 5
            )

    def runtime(self):
        positive = torch.full((1, 2, 2, 2), 3.0, dtype=torch.float64)
        negative = torch.full_like(positive, 0.5)

        class BaseGuider:
            def __init__(self, model):
                self.inner_model = self.model_patcher = model
                self.model_options = model.model_options

            def inner_set_conds(self, conds):
                self.conds = conds

            def sample(self, noise, latent, sampler, sigmas, **kwargs):
                self.model_patcher.calls.append(
                    dict(
                        noise=noise,
                        latent=latent,
                        sampler=sampler,
                        sigmas=sigmas,
                        cfg1=self.cfg1,
                        cfg2=self.cfg2,
                        conds=self.conds,
                        **kwargs,
                    )
                )
                if kwargs["callback"] is not None:
                    kwargs["callback"](0, latent, latent, 4)
                return self.predict_noise(
                    noise, torch.tensor(1.0), self.model_options, kwargs["seed"]
                )

        calculate = Mock(
            side_effect=lambda model, scheduler, steps: torch.linspace(1, 0, steps + 1)
        )
        sampler_object = Mock(return_value="native-euler")
        samplers = SimpleNamespace(
            CFGGuider=BaseGuider, calculate_sigmas=calculate, sampler_object=sampler_object
        )
        scope = dict(
            torch=torch,
            math=math,
            comfy=SimpleNamespace(samplers=samplers),
            io=SimpleNamespace(NodeOutput=lambda *values: values),
        )
        setter = native_function("node_helpers.py", "conditioning_set_values", scope)
        scope["node_helpers"] = SimpleNamespace(conditioning_set_values=setter)
        samplers.cfg_function = native_function("comfy/samplers.py", "cfg_function", scope)
        samplers.calc_cond_batch = lambda model, conds, x, timestep, options: [
            positive if value[0][1]["role"] == "positive" else negative for value in conds
        ]
        native_class(
            "comfy_extras/nodes_custom_sampler.py",
            "Guider_DualCFG",
            ("set_cfg", "set_conds", "predict_noise"),
            scope,
            keep_bases=True,
        )
        nodes = {}
        for name in ("BasicScheduler", "KSamplerSelect", "DualCFGGuider"):
            nodes[name] = native_class(
                "comfy_extras/nodes_custom_sampler.py", name, ("execute",), scope
            )
        norm = native_class("comfy_extras/nodes_cfg.py", "CFGNorm", ("execute",), scope)
        memory = SimpleNamespace(
            intermediate_device=lambda: "cpu", intermediate_dtype=lambda: torch.float32
        )
        modules = {
            "comfy": SimpleNamespace(model_management=memory),
            "comfy_extras.nodes_custom_sampler": SimpleNamespace(**nodes),
        }
        return modules, calculate, sampler_object, norm, samplers, positive, negative

    def request(self, model, **changes):
        values = dict(
            model=model,
            noise=torch.ones(1, 2, 2, 2, dtype=torch.float64),
            steps=4,
            cfg=3.25,
            sampler_name="euler",
            scheduler="simple",
            positive=[["tokens", dict(role="positive")]],
            negative=[["tokens", dict(role="negative")]],
            latent_image=torch.zeros(1, 2, 2, 2),
            denoise=0.5,
            noise_mask=torch.ones(1, 2, 2),
            callback=Mock(),
            seed=77,
            disable_pbar=True,
        )
        return values | changes

    def test_native_guider_and_scheduler_preserve_cfg_noise_mask_seed_callback_and_inputs(self):
        modules, calculate, sampler_object, _, _, positive, negative = self.runtime()
        model = FakeModel()
        request = self.request(model)
        with patch.dict("sys.modules", modules):
            result = sample_omnigen2(**request)
        call = model.calls[0]
        self.assertEqual((call["cfg1"], call["cfg2"]), (3.25, 2))
        self.assertIs(call["noise"], request["noise"])
        self.assertIs(call["latent"], request["latent_image"])
        self.assertIs(call["denoise_mask"], request["noise_mask"])
        self.assertIs(call["callback"], request["callback"])
        self.assertEqual((call["seed"], call["disable_pbar"]), (77, True))
        self.assertNotIn("prompt_type", request["negative"][0][1])
        self.assertEqual(call["conds"]["middle"][0][1]["prompt_type"], "negative")
        calculate.assert_called_once_with("model_sampling", "simple", 8)
        sampler_object.assert_called_once_with("euler")
        torch.testing.assert_close(call["sigmas"], torch.linspace(1, 0, 9)[-5:])
        torch.testing.assert_close(result, (negative + (positive - negative) * 3.25).float())

    def test_actual_native_cfgnorm_hooks_remain_on_the_dual_guider_route(self):
        for pre in (False, True):
            modules, _, _, norm, samplers, positive, negative = self.runtime()
            model = norm.execute(FakeModel(), strength=1.0, pre_cfg=pre)[0]
            request = self.request(model, cfg=5)
            with patch.dict("sys.modules", modules):
                actual = sample_omnigen2(**request)
            self.assertTrue(model.model_options)
            plain = samplers.cfg_function(
                model,
                positive,
                negative,
                5,
                request["noise"],
                torch.tensor(1),
                model_options=model.model_options,
            )
            self.assertFalse(torch.allclose(actual.double(), plain))
            self.assertTrue(torch.isfinite(actual).all())
            self.assertEqual((model.calls[0]["cfg1"], model.calls[0]["cfg2"]), (5, 2))

    def test_native_callback_cancellation_and_errors_propagate_without_retry(self):
        modules, *_ = self.runtime()
        for error in (CancelledError("cancelled"), RuntimeError("native failure")):
            model = FakeModel()
            callback = Mock(side_effect=error)
            with patch.dict("sys.modules", modules), self.assertRaises(type(error)):
                sample_omnigen2(**self.request(model, callback=callback))
            self.assertEqual(len(model.calls), 1)
            callback.assert_called_once()

    def test_explicit_lcm_uses_native_noise_options_with_the_same_guider(self):
        modules, _, sampler_object, *_ = self.runtime()
        lcm = Mock(return_value=("native-lcm",))
        modules["comfy_extras.nodes_advanced_samplers"] = SimpleNamespace(
            SamplerLCM=SimpleNamespace(execute=lcm)
        )
        options = dict(s_noise=0.4, s_noise_end=0.2, noise_clip_std=0.0)
        model = FakeModel()
        with patch.dict("sys.modules", modules):
            sample_omnigen2(**self.request(model, sampler_name="lcm", noise_options=options))
        lcm.assert_called_once_with(**options)
        sampler_object.assert_not_called()
        self.assertEqual(model.calls[0]["sampler"], "native-lcm")


if __name__ == "__main__":
    unittest.main()
