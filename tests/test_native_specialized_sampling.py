"""Prepared native defaults reach real sigma/guider callers without diffusion weights."""

import ast
import asyncio
from contextlib import contextmanager
from concurrent.futures import CancelledError
import hashlib
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import scipy.stats
import torch
import torch.distributed.tensor
import gradio as gr
from gradio.state_holder import SessionState
from pydantic import ValidationError

from ruinedfooocus.core.cache_contract import file_identity
from ruinedfooocus.core.preparation import PreparationRequest
from ruinedfooocus.features.generation import native_sampling as native
from ruinedfooocus.features.models.model_identity import record_prepared_native_source
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.features.models.resource_defaults import load_family_recipe
import test_model_recommendations as recommendations
from test_qwen_image21_conditioning import native_class, native_function
from test_ordered_native_references import process_nodes, runtime as reference_runtime

ROOT = Path(__file__).resolve().parents[1]


def model(family, depth=None):
    config = NS(unet_config={"depth_single_blocks": depth})
    inner = type(family, (), {"model_config": config})()
    return NS(model=inner, get_model_object=lambda _: NS(sigmas=torch.linspace(0.001, 1, 1000)))


def runtime():
    modules = reference_runtime()
    scope = dict(
        torch=torch,
        math=math,
        io=NS(NodeOutput=lambda *values: values),
        _LOGSNR_MIN=-15.0,
        _LOGSNR_MAX=18.0,
    )
    for name in ("_logit_normal_schedule", "ideogram4_sigmas"):
        native_function("comfy_extras/nodes_ideogram4.py", name, scope)
    ideogram = native_class(
        "comfy_extras/nodes_ideogram4.py", "Ideogram4Scheduler", scope, methods=("execute",)
    )
    beta = native_function(
        "comfy/samplers.py", "beta_scheduler", dict(torch=torch, numpy=np, scipy=scipy)
    )
    scope["comfy"] = NS(samplers=NS(beta_scheduler=beta))
    beta_node = native_class(
        "comfy_extras/nodes_custom_sampler.py", "BetaSamplingScheduler", scope, methods=("execute",)
    )
    basic = NS(execute=Mock(return_value=(torch.tensor([1.0, 0.5, 0.0]),)))
    modules["comfy_extras.nodes_ideogram4"] = NS(Ideogram4Scheduler=ideogram)
    modules["comfy_extras.nodes_custom_sampler"] = NS(
        BetaSamplingScheduler=beta_node, BasicScheduler=basic
    )
    return modules


@contextmanager
def prepared_source(root, artifact_id):
    original = load_registry().artifact(artifact_id)
    path = root / "unrelated-name.safetensors"
    path.write_bytes(b"exact native source fixture")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    row = dict(
        original, artifact=dict(original["artifact"], sha256=checksum, bytes=path.stat().st_size)
    )
    registry = NS(
        artifact=lambda identity: row if identity == row["id"] else None, entries=lambda **_: (row,)
    )
    request = PreparationRequest(
        row["artifact"]["pinned_url"], path, sha256=checksum, expected_size=path.stat().st_size
    )
    with patch("ruinedfooocus.features.models.model_registry.load_registry", return_value=registry):
        proof = record_prepared_native_source(path, request, file_identity(path), root / "cache")
        yield path, row, proof


class NativeSamplingTests(unittest.TestCase):
    def test_prepared_source_performance_to_production_sampler_keeps_native_contract(self):
        cases = (
            (
                "existing:diffusion_models/flux2_dev_fp8mixed.safetensors",
                "Flux2",
                None,
                20,
                1,
                "Flux2Scheduler",
            ),
            (
                "existing:diffusion_models/chroma_radiance_x0.safetensors",
                "ChromaRadiance",
                None,
                30,
                3.5,
                "BetaSamplingScheduler",
            ),
            (
                "existing:diffusion_models/ideogram4_fp8_scaled.safetensors",
                "Ideogram4",
                None,
                20,
                7,
                "Ideogram4Scheduler",
            ),
        )
        registry = load_registry()
        chroma = next(
            r["id"] for r in registry.entries() if r.get("native_family") == "ChromaRadiance"
        )
        for artifact_id, family, depth, steps, cfg, scheduler in cases:
            if family == "ChromaRadiance":
                artifact_id = chroma
            with self.subTest(family=family), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                with prepared_source(root, artifact_id) as (path, row, _):
                    performance = recommendations.RecommendationTests()._performance(
                        root / "cache/checkpoints"
                    )
                    performance.model_recommendations.offline = True
                    modules = runtime()
                    modules["comfy.samplers"] = NS(
                        KSampler=NS(SAMPLERS=["euler"], SCHEDULERS=["simple", "normal"])
                    )
                    with patch.dict("sys.modules", modules):
                        data = performance.apply(
                            dict(base_model_name=str(path), performance_selection="Model Defaults"),
                            family,
                        )
                        self.assertEqual(
                            (data["custom_steps"], data["cfg"], data["scheduler"]),
                            (steps, cfg, scheduler),
                        )
                        plan = data[native.KEY]
                        loaded = model(family, depth)
                        sigmas = native.sigmas(plan, loaded, steps, 1024, 1024, 1, scheduler)
                        self.assertGreater(len(sigmas), 2)
                        self.assertEqual(sigmas[-1].item(), 0)
                        statement = next(
                            n
                            for n in ast.walk(ast.Module(body=process_nodes(), type_ignores=[]))
                            if isinstance(n, ast.If)
                            and any(
                                isinstance(c, ast.Call)
                                and isinstance(c.func, ast.Name)
                                and c.func.id == "sample_native"
                                for c in ast.walk(n)
                            )
                        )
                        native_sample = Mock(return_value=torch.zeros(1, 16, 8, 8))

                        @contextmanager
                        def secondary(*args):
                            yield "unconditional" if family == "Ideogram4" else None

                        scope = dict(
                            native_plan=plan,
                            self=NS(),
                            gen_data=data,
                            path_manager=NS(),
                            worker=NS(check_interrupt=Mock()),
                            sampling_model=loaded,
                            noise="noise",
                            cfg=cfg,
                            sampler_name="euler",
                            fixed_sigmas=sigmas,
                            positive_cond="positive",
                            negative_conditioning="negative",
                            latent_image="latent",
                            noise_options=None,
                            noise_mask="mask",
                            callback_function="callback",
                            comfy=NS(utils=NS(PROGRESS_BAR_ENABLED=False)),
                            seed=42,
                        )
                        with (
                            patch.object(native, "sample", native_sample),
                            patch.object(native, "negative_model_scope", secondary),
                        ):
                            exec(
                                compile(
                                    ast.Module(body=[statement], type_ignores=[]),
                                    "production native sampler",
                                    "exec",
                                ),
                                scope,
                            )
                        self.assertIs(native_sample.call_args.kwargs["sigmas"], sigmas)
                        self.assertEqual(native_sample.call_args.kwargs["cfg"], cfg)
                        self.assertEqual(
                            native_sample.call_args.kwargs["negative_model"],
                            "unconditional" if family == "Ideogram4" else None,
                        )
                        self.assertEqual(plan.identity_basis, "verified_publisher_artifact")

    def test_actual_native_schedules_match_canvas_parameters_and_partial_denoise(self):
        modules = runtime()
        recipes = (
            "native-flux2-dev",
            "native-chromaradiance-radiance-x0",
            "native-ideogram4-conditional",
        )
        with patch.dict("sys.modules", modules):
            for identity in recipes:
                plan = native.plan_from_recipe(load_family_recipe(identity))
                owner = model(plan.family)
                actual = native.sigmas(plan, owner, 20, 1024, 768, 0.5, plan.scheduler)
                full = native.sigmas(plan, owner, 40, 1024, 768, 1, plan.scheduler)
                np.testing.assert_array_equal(actual.numpy(), full[-21:].numpy())
                self.assertEqual(actual[-1].item(), 0)
            beta = native.plan_from_recipe(load_family_recipe(recipes[1]))
            self.assertEqual(dict(beta.parameters), dict(alpha=0.4, beta=0.4))
            old = native.apply_options(
                beta, dict(native_schedule_alpha=0.6, native_schedule_beta=0.6)
            )
            self.assertFalse(
                torch.equal(
                    native.sigmas(beta, model(beta.family), 30, 1024, 1024, 1, beta.scheduler),
                    native.sigmas(old, model(old.family), 30, 1024, 1024, 1, old.scheduler),
                )
            )
            ideogram = native.plan_from_recipe(load_family_recipe(recipes[2]))
            self.assertEqual(dict(ideogram.parameters), dict(mu=0.5, std=1.75))
            self.assertFalse(
                torch.equal(
                    native.sigmas(
                        ideogram, model(ideogram.family), 20, 1024, 1024, 1, ideogram.scheduler
                    ),
                    native.sigmas(
                        ideogram, model(ideogram.family), 20, 768, 1024, 1, ideogram.scheduler
                    ),
                )
            )

    def test_manual_ordinary_scheduler_preserves_native_guider_and_parameters(self):
        modules = runtime()
        plan = native.plan_from_recipe(load_family_recipe("native-ideogram4-conditional"))
        with patch.dict("sys.modules", modules):
            output = native.sigmas(plan, model("Ideogram4"), 20, 1024, 1024, 0.5, "normal")
        basic = modules["comfy_extras.nodes_custom_sampler"].BasicScheduler.execute
        self.assertEqual(basic.call_args.args[1:], ("normal", 20, 0.5))
        self.assertIs(output, basic.return_value[0])
        self.assertEqual(plan.guider, "DualModelGuider")
        self.assertNotIn("Ideogram4Scheduler", ["simple", "normal"])

    def test_native_model_patches_match_chroma_flow_and_flux_kv_reference_graphs(self):
        class SamplingBase:
            def __init__(self, config):
                self.config = config

            def set_parameters(self, **values):
                self.parameters = values

        scope = dict(
            comfy=NS(
                model_sampling=NS(
                    ModelSamplingDiscreteFlow=SamplingBase, CONST=type("Constant", (), {})
                )
            )
        )
        native_class(
            "comfy_extras/nodes_model_advanced.py", "ModelSamplingSD3", scope, methods=("patch",)
        )
        aura = native_class(
            "comfy_extras/nodes_model_advanced.py",
            "ModelSamplingAuraFlow",
            scope,
            methods=("patch_aura",),
            bases=("ModelSamplingSD3",),
        )
        source = model("ChromaRadiance")
        clone = model("ChromaRadiance")
        clone.add_object_patch = Mock()
        source.clone = Mock(return_value=clone)
        plan = native.plan_from_recipe(load_family_recipe("native-chromaradiance-radiance-x0"))
        with patch.dict(
            sys.modules, {"comfy_extras.nodes_model_advanced": NS(ModelSamplingAuraFlow=aura)}
        ):
            patched = native.patch_model(plan, source, guidance={"sampling_shift": None})
            self.assertIs(patched, clone)
            key, sampling = clone.add_object_patch.call_args.args
            self.assertEqual(key, "model_sampling")
            self.assertEqual(sampling.parameters, dict(shift=1.0, multiplier=1.0))
            self.assertIs(native.patch_model(plan, source, guidance={"sampling_shift": 3}), source)
            self.assertEqual(source.clone.call_count, 1)
        modules = reference_runtime()
        scope = dict(torch=torch, io=NS(NodeOutput=lambda *values: values))
        native_class(
            "comfy_extras/nodes_flux.py",
            "KV_Attn_Input",
            scope,
            methods=("__init__", "__call__", "cleanup"),
        )
        kv = native_class("comfy_extras/nodes_flux.py", "FluxKVCache", scope, methods=("execute",))
        source = model("Flux2", 24)
        source.model.diffusion_model = NS(params=NS())
        clone = model("Flux2", 24)
        clone.set_model_attn1_patch = Mock()
        clone.set_model_post_input_patch = Mock()
        clone.add_object_patch = Mock()
        source.clone = Mock(return_value=clone)
        plan = native.plan_from_recipe(
            load_family_recipe("native-flux2klein9b-klein-distilled-9b-kv")
        )
        with patch.dict(sys.modules, {"comfy_extras.nodes_flux": NS(FluxKVCache=kv)}):
            self.assertIs(native.patch_model(plan, source), clone)
        clone.add_object_patch.assert_called_once_with(
            "diffusion_model.params.default_ref_method", "index_timestep_zero"
        )
        attn = clone.set_model_attn1_patch.call_args.args[0]
        q = k = v = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
        attn(q, k, v, dict(reference_image_num_tokens=[1], block_type="single", block_index=0))
        self.assertEqual(len(attn.cache), 1)
        attn.cleanup()
        self.assertEqual(attn.cache, {})

    def test_flux2_distilled_guidance_uses_native_conditioning_separate_from_sampler_cfg(self):
        modules = reference_runtime()
        setter = modules["node_helpers"].conditioning_set_values
        scope = dict(
            node_helpers=NS(conditioning_set_values=setter),
            io=NS(NodeOutput=lambda *values: values),
        )
        node = native_class(
            "comfy_extras/nodes_flux.py", "FluxGuidance", scope, methods=("execute",)
        )
        plan = native.plan_from_recipe(load_family_recipe("native-flux2-dev"))
        positive = [["tokens", {"native": True}]]
        with patch.dict(sys.modules, {"comfy_extras.nodes_flux": NS(FluxGuidance=node)}):
            conditioned = native.condition(plan, positive)
        self.assertEqual(conditioned[0][1]["guidance"], 4)
        self.assertNotIn("guidance", positive[0][1])
        self.assertEqual(plan.guider, "BasicGuider")

    def test_native_guider_uses_text_free_second_model_and_forwards_callback_and_mask(self):
        returned = torch.ones((1, 4, 8, 8))
        guider = NS(set_conds=Mock(), set_cfg=Mock(), sample=Mock(return_value=returned))
        scope = dict(
            io=NS(NodeOutput=lambda *values: values), Guider_DualModel=Mock(return_value=guider)
        )
        node = native_class(
            "comfy_extras/nodes_custom_sampler.py", "DualModelGuider", scope, methods=("execute",)
        )
        modules = {
            "comfy": NS(
                model_management=NS(
                    intermediate_device=lambda: "cpu", intermediate_dtype=lambda: torch.float32
                )
            ),
            "comfy_extras.nodes_custom_sampler": NS(
                BasicGuider=NS(),
                CFGGuider=NS(),
                DualModelGuider=node,
                KSamplerSelect=NS(execute=lambda name: (name,)),
            ),
        }
        plan = native.plan_from_recipe(load_family_recipe("native-ideogram4-conditional"))
        with patch.dict("sys.modules", modules):
            result = native.sample(
                plan,
                model=model("Ideogram4"),
                negative_model="unconditional",
                noise="noise",
                cfg=7,
                sampler_name="euler",
                sigmas="sigmas",
                positive="positive",
                negative="unused text",
                latent_image="latent",
                noise_mask="mask",
                callback="callback",
                seed=17,
            )
        self.assertIs(result, returned)
        scope["Guider_DualModel"].assert_called_once()
        self.assertEqual(scope["Guider_DualModel"].call_args.args[1], "unconditional")
        guider.set_conds.assert_called_once_with("positive", [[None, {}]])
        guider.set_cfg.assert_called_once_with(7)
        self.assertEqual(
            guider.sample.call_args.kwargs,
            dict(denoise_mask="mask", callback="callback", disable_pbar=False, seed=17),
        )

    def test_invalid_parameters_family_and_unbounded_partial_schedule_fail_before_sampling(self):
        plan = native.plan_from_recipe(load_family_recipe("native-ideogram4-conditional"))
        for values in (
            dict(native_schedule_std=0),
            dict(native_schedule_mu=float("nan")),
            dict(native_schedule_alpha=0.4),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                native.apply_options(plan, values)
        with self.assertRaisesRegex(ValueError, "loaded checkpoint"):
            native.require_model(plan, model("Flux2"))
        with patch.dict("sys.modules", runtime()), self.assertRaisesRegex(ValueError, "10000"):
            native.sigmas(plan, model("Ideogram4"), 20, 1024, 1024, 1e-300, plan.scheduler)
        with patch.dict("sys.modules", runtime()):
            self.assertEqual(
                len(native.sigmas(plan, model("Ideogram4"), 20, 1024, 1024, 0, plan.scheduler)), 0
            )

    def test_secondary_capture_binds_named_asset_and_refuses_bad_role_or_path_before_compute(self):
        from test_image_preparation import Manager

        recipe = load_family_recipe("native-ideogram4-conditional")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "checkpoints").mkdir()
            primary = root / "checkpoints/primary.safetensors"
            secondary = root / "checkpoints/secondary.safetensors"
            primary.write_bytes(b"primary fixture")
            secondary.write_bytes(b"secondary fixture")
            models = NS(
                get_file=lambda *args: primary, get_models_by_path=lambda *args, **kwargs: {}
            )
            manager = Manager(root)
            data = dict(
                base_model_name="primary.safetensors",
                scheduler="Ideogram4Scheduler",
                native_negative_model="secondary.safetensors",
            )
            with (
                patch(
                    "ruinedfooocus.features.models.native_recipes.current_native_recipe",
                    return_value=recipe,
                ),
                patch(
                    "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                    return_value=dict(native_class="Ideogram4"),
                ),
            ):
                captured, artifacts = native.capture_assets(data, manager, models)
                self.assertEqual(captured[native.KEY].guider, "DualModelGuider")
                self.assertEqual(
                    (artifacts[0].role, artifacts[0].name),
                    ("native-negative-model", "secondary.safetensors"),
                )
                self.assertEqual(artifacts[0].request.destination, secondary.resolve())
                for value in (
                    "../secondary.safetensors",
                    str(secondary.resolve()),
                    "primary.safetensors",
                ):
                    with self.subTest(value=value), self.assertRaises(ValueError):
                        native.capture_assets(
                            dict(data, native_negative_model=value), manager, models
                        )
            with (
                patch(
                    "ruinedfooocus.features.models.native_recipes.current_native_recipe",
                    return_value=recipe,
                ),
                patch(
                    "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                    return_value=dict(native_class="Flux2"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "unconditional diffusion checkpoint"):
                    native.capture_assets(data, manager, models)

    def test_secondary_reader_lease_survives_sampling_and_cache_tracks_current_signature(self):
        plan = native.plan_from_recipe(load_family_recipe("native-ideogram4-conditional"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "negative.safetensors"
            path.write_bytes(b"first fixture")
            loaded = model("Ideogram4")
            owner = NS(xl_base_patched=NS(unet=model("Ideogram4")))
            events = []

            @contextmanager
            def lease(data, role, name, manager, category):
                self.assertEqual(
                    (role, name, category),
                    ("native-negative-model", "negative.safetensors", "checkpoints"),
                )
                events.append("open")
                try:
                    yield path
                finally:
                    events.append("close")

            loader = Mock(return_value=loaded)
            data = dict(native_negative_model="negative.safetensors")
            with (
                patch("ruinedfooocus.features.generation.image_preparation.prepared_asset", lease),
                patch.dict(
                    sys.modules,
                    {
                        "ruinedfooocus.features.generation.pipeline_utils": NS(
                            load_diffusion_model=loader
                        )
                    },
                ),
            ):
                for _ in range(2):
                    with native.negative_model_scope(
                        owner, plan, data, NS(), lambda: events.append("check")
                    ) as result:
                        self.assertIs(result, loaded)
                        self.assertEqual(events[-1], "check")
                        self.assertGreater(events.count("open"), events.count("close"))
                self.assertEqual(loader.call_count, 1)
                path.write_bytes(b"replacement fixture")
                with native.negative_model_scope(owner, plan, data, NS(), lambda: None):
                    pass
                self.assertEqual(loader.call_count, 2)
                with self.assertRaises(CancelledError):
                    with native.negative_model_scope(
                        owner, plan, data, NS(), Mock(side_effect=CancelledError)
                    ):
                        self.fail("A cancelled request must not enter sampling.")
            self.assertEqual(events.count("open"), events.count("close"))

    def test_default_secondary_is_the_registered_companion_with_expected_identity(self):
        from test_image_preparation import Manager

        registry = load_registry()
        source = registry.artifact("existing:diffusion_models/ideogram4_fp8_scaled.safetensors")
        companion = registry.artifact(source["component_contract"]["secondary_model"])
        recipe = load_family_recipe("native-ideogram4-conditional")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            primary = root / "renamed.safetensors"
            primary.write_bytes(b"owned primary")
            manager = Manager(root)
            key = companion["catalogue_key"]
            manager.DOWNLOADABLE_FILES[key] = dict(
                path="path_diffusion_models",
                filename=Path(key).name,
                url=companion["artifact"]["pinned_url"],
                size=companion["artifact"]["bytes"],
                sha256=companion["artifact"]["sha256"],
            )
            models = NS(
                get_file=lambda *args: primary, get_models_by_path=lambda *args, **kwargs: {}
            )
            with (
                patch(
                    "ruinedfooocus.features.models.native_recipes.current_native_recipe",
                    return_value=recipe,
                ),
                patch(
                    "ruinedfooocus.features.image_editing.ordered_native_references.current_source",
                    return_value=source,
                ),
            ):
                data, assets = native.capture_assets(
                    dict(base_model_name=primary.name, scheduler="Ideogram4Scheduler"),
                    manager,
                    models,
                )
            self.assertEqual(data["native_negative_model"], Path(key).name)
            self.assertEqual(data["_native_secondary_basis"], "captured_catalogue_companion")
            self.assertEqual(assets[0].request.sha256, companion["artifact"]["sha256"])
            self.assertEqual(assets[0].request.expected_size, companion["artifact"]["bytes"])
            self.assertIsNone(assets[0].signature)

    def test_secondary_config_and_native_inference_dtypes_are_validated_independently(self):
        positive, negative = model("Ideogram4"), model("Ideogram4")
        for value in (positive, negative):
            value.model.model_config.unet_config.update(in_channels=128, num_layers=34)
            value.model.model_config.supported_inference_dtypes = (torch.bfloat16, torch.float32)
        positive.model.get_dtype_inference = lambda: torch.bfloat16
        negative.model.get_dtype_inference = lambda: torch.float32
        self.assertEqual(native.validate_secondary(positive, negative), ["bfloat16", "float32"])
        negative.model.get_dtype_inference = lambda: torch.int8
        with self.assertRaisesRegex(ValueError, "inference dtype"):
            native.validate_secondary(positive, negative)
        negative.model.get_dtype_inference = lambda: torch.float32
        negative.model.model_config.unet_config["num_layers"] = 33
        with self.assertRaisesRegex(ValueError, "configurations differ"):
            native.validate_secondary(positive, negative)

    def test_typed_api_and_actual_gradio_context_transport_native_parameters(self):
        from ruinedfooocus.integrations.openai_api import ImageRequest
        from ruinedfooocus.features.generation import ui_native_sampling as ui
        from test_api_image_edits import queued_data

        request = ImageRequest(
            prompt="a scene",
            native_negative_model="unconditional.safetensors",
            native_schedule_mu=0.75,
            native_schedule_std=1.5,
        )
        recipe = load_family_recipe("native-ideogram4-conditional")
        plan = native.plan_from_recipe(recipe)
        perf = NS(
            CUSTOM_PERFORMANCE="Own settings",
            choices_for_model=lambda *args: ["Model Defaults", "Own settings"],
            selection_for_model=lambda *args: "Model Defaults",
            get_perf_options=lambda *args: dict(recipe["sampling"], **{native.KEY: plan}),
        )
        from ruinedfooocus.features.generation import image_preparation as preparation
        from ruinedfooocus.features.models.model_identity import file_signature

        original_artifact = preparation._artifact
        with tempfile.TemporaryDirectory() as folder:
            secondary = Path(folder) / "unconditional.safetensors"
            secondary.write_bytes(b"owned unconditional fixture")
            artifact = preparation.ImageArtifact(
                "native-negative-model",
                secondary.name,
                PreparationRequest("", secondary),
                tuple(file_signature(secondary)),
            )

            def selected_asset(manager, role, category, name, key=None):
                return (
                    artifact
                    if role == "native-negative-model"
                    else original_artifact(manager, role, category, name, key)
                )

            with (
                patch(
                    "ruinedfooocus.features.models.native_recipes.current_native_recipe",
                    return_value=recipe,
                ),
                patch.object(preparation, "_artifact", selected_asset),
                patch(
                    "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                    return_value=dict(native_class="Ideogram4"),
                ),
            ):
                data = queued_data(
                    "Ideogram4",
                    recipe_resolver=perf,
                    payload_options=request.model_dump(exclude_unset=True)
                    | dict(scheduler="Ideogram4Scheduler", input_image=None),
                )
                captured = next(
                    item
                    for item in data[preparation.KEY].artifacts
                    if item.role == "native-negative-model"
                )
                self.assertEqual(captured, artifact)
        self.assertEqual(
            (
                data["native_negative_model"],
                data["native_schedule_mu"],
                data["native_schedule_std"],
            ),
            ("unconditional.safetensors", 0.75, 1.5),
        )
        self.assertEqual(
            data[native.KEY], native.apply_options(plan, request.model_dump(exclude_unset=True))
        )
        self.assertEqual(dict(plan.parameters), dict(mu=0.5, std=1.75))
        with self.assertRaises(ValidationError):
            ImageRequest(prompt="x", native_schedule_std=0)
        registered = {}
        shared = NS(
            settings=NS(default_settings={}),
            models=NS(
                get_names=lambda *args: ["unconditional.safetensors"], get_file=lambda *args: None
            ),
            performance_settings=NS(
                model_recommendations=NS(
                    family_details=lambda name: recipe if name == "ideogram" else {}
                )
            ),
            add_ctrl=lambda key, value: registered.update({key: value}),
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(sys.modules["ruinedfooocus.app"], "state", shared, create=True),
        ):
            with gr.Blocks(analytics_enabled=False) as app:
                checkpoint = gr.Textbox(value="ideogram")
                ui.create_ui(checkpoint)
            callback = next(key for key, fn in app.fns.items() if fn.fn.__name__ == "context")
            output = asyncio.run(
                app.process_api(
                    callback,
                    ["ordinary", "ideogram", None, None, None, 0.75, 1.5],
                    state=SessionState(app),
                )
            )["data"]
            self.assertFalse(output[0]["visible"])
            self.assertIsNone(output[1]["value"])
            self.assertTrue(all(row["value"] is None for row in output[2:6]))

    def test_actual_companion_dropdown_filters_native_headers_and_rechecks_replaced_files(self):
        from ruinedfooocus.features.generation import ui_native_sampling as ui

        recipe = load_family_recipe("native-ideogram4-conditional")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            classes = {
                "primary.safetensors": "Ideogram4",
                "unconditional.safetensors": "Ideogram4",
                "compatible.safetensors": "Ideogram4",
                "conditional.safetensors": "Ideogram4",
                "janku.gguf": "LLM",
                "sam.safetensors": "SAM3",
            }
            for name in classes:
                (root / name).write_bytes(b"bounded header fixture")
            models = NS(
                get_names=lambda *args: list(classes),
                get_file=lambda category, name: root / name,
                get_models_by_path=lambda *args, **kwargs: {},
            )
            registered = {}
            shared = NS(
                settings=NS(default_settings={}),
                models=models,
                performance_settings=NS(
                    model_recommendations=NS(family_details=lambda name: recipe)
                ),
                add_ctrl=lambda key, value: registered.update({key: value}),
            )
            with (
                patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
                patch.object(sys.modules["ruinedfooocus.app"], "state", shared, create=True),
                patch(
                    "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                    side_effect=lambda path: {"native_class": classes[Path(path).name]},
                ),
                patch(
                    "ruinedfooocus.features.models.model_identity.verified_native_source",
                    side_effect=lambda path, metadata: (
                        {"behavioral_variant": "conditional"}
                        if Path(path).name == "conditional.safetensors"
                        else None
                    ),
                ),
            ):
                with gr.Blocks(analytics_enabled=False) as app:
                    checkpoint = gr.Textbox(value="primary.safetensors")
                    ui.create_ui(checkpoint)
                self.assertEqual(
                    {value for label, value in registered["native_negative_model"].choices},
                    {None, "unconditional.safetensors", "compatible.safetensors"},
                )
                classes["unconditional.safetensors"] = "SAM3"
                callback = next(key for key, fn in app.fns.items() if fn.fn.__name__ == "context")
                result = asyncio.run(
                    app.process_api(
                        callback,
                        [
                            "primary.safetensors",
                            "primary.safetensors",
                            "unconditional.safetensors",
                            None,
                            None,
                            0.75,
                            1.5,
                        ],
                        state=SessionState(app),
                    )
                )["data"]
                self.assertIsNone(result[1]["value"])
                self.assertEqual(
                    {value for label, value in result[1]["choices"]},
                    {None, "compatible.safetensors"},
                )
                self.assertEqual((result[4]["value"], result[5]["value"]), (0.75, 1.5))

    def test_overridden_parameters_survive_repeat_capture_preparation_and_worker_sampling(self):
        from ruinedfooocus.core.preparation import PreparationService
        from ruinedfooocus.features.generation.image_preparation import (
            ImageAssets,
            KEY as ASSETS,
            prepare_image_assets,
        )
        from test_image_preparation import Manager

        recipe = load_family_recipe("native-ideogram4-conditional")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "checkpoints").mkdir()
            primary = root / "checkpoints/primary.safetensors"
            primary.write_bytes(b"owned primary")
            (root / "checkpoints/negative.safetensors").write_bytes(b"owned negative")
            manager = Manager(root)
            models = NS(
                get_file=lambda *args: primary, get_models_by_path=lambda *args, **kwargs: {}
            )
            source = dict(
                base_model_name=primary.name,
                scheduler="Ideogram4Scheduler",
                native_negative_model="negative.safetensors",
                native_schedule_mu=0.75,
                native_schedule_std=1.5,
            )
            service = PreparationService(staging_root=root / "staging")
            try:
                with (
                    patch(
                        "ruinedfooocus.features.models.native_recipes.current_native_recipe",
                        return_value=recipe,
                    ),
                    patch(
                        "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                        return_value={"native_class": "Ideogram4"},
                    ),
                ):
                    first, artifacts = native.capture_assets(source, manager, models)
                    second, _ = native.capture_assets(first, manager, models)
                    prepared = prepare_image_assets(
                        dict(second, **{ASSETS: ImageAssets(artifacts, True)}),
                        service=service,
                        check_cancelled=lambda: None,
                    )
                    final, _ = native.capture_assets(prepared, manager, models)
                self.assertEqual(first[native.KEY], final[native.KEY])
                self.assertEqual(dict(final[native.KEY].parameters), dict(mu=0.75, std=1.5))
                with patch.dict(sys.modules, runtime()):
                    actual = native.sigmas(
                        final[native.KEY], model("Ideogram4"), 20, 1024, 1024, 1, final["scheduler"]
                    )
                    baseline = native.sigmas(
                        native.plan_from_recipe(recipe),
                        model("Ideogram4"),
                        20,
                        1024,
                        1024,
                        1,
                        final["scheduler"],
                    )
                self.assertFalse(torch.equal(actual, baseline))
                statement = next(
                    node
                    for node in process_nodes()
                    if isinstance(node, ast.If)
                    and any(
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "sigmas"
                        for call in ast.walk(node)
                    )
                )
                scope = dict(
                    native_plan=final[native.KEY],
                    positive_cond=[["positive", {}]],
                    sampling_model=model("Ideogram4"),
                    gen_data=dict(final, steps=20, width=1024, height=1024),
                    effective={},
                    denoise=1,
                    scheduler=final["scheduler"],
                )
                with patch.dict(sys.modules, runtime()):
                    exec(
                        compile(
                            ast.Module(body=[statement], type_ignores=[]),
                            "production captured sigma contract",
                            "exec",
                        ),
                        scope,
                    )
                self.assertTrue(torch.equal(scope["fixed_sigmas"], actual))
                self.assertEqual(
                    scope["gen_data"]["native_sampling"]["parameters"], dict(mu=0.75, std=1.5)
                )
            finally:
                service.close()


if __name__ == "__main__":
    unittest.main()
