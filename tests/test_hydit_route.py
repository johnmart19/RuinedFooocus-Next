"""HyDiT application callers with pinned native contracts and owned tiny fixtures."""

import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from ruinedfooocus.core.preparation import PreparationService
from ruinedfooocus.backends.comfy.component_selection import validate_component_slots
from ruinedfooocus.features.generation import image_preparation
from ruinedfooocus.features.models import component_profiles
from ruinedfooocus.features.models.component_candidates import classify_shapes, compatibility
from ruinedfooocus.features.models.image_components import inventory
from ruinedfooocus.features.models.model_architecture import inspect_tensor_model
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.features.models.performance import UnavailableModelRecipe
from ruinedfooocus.features.models.resource_defaults import RESOURCE_DEFAULTS
from test_image_preparation import Manager
from test_model_cache import method
import test_model_recommendations as recipes
from test_native_hydit import native_class, native_function
from test_native_model_coverage import NATIVE, PIPELINE, native_method, route_info

ROOT = Path(__file__).resolve().parents[1]
SHA = "4fb84f84079cda457d171b3c6b15d1be95b5a3e5d9825703951a99ddf92d1787"


def header(path, entries):
    raw = json.dumps(
        {k: dict(shape=v, dtype="F16", data_offsets=[0, 0]) for k, v in entries.items()}
    ).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw)
    return path


def native_detector():
    scope = {"count_blocks": native_function("comfy/model_detection.py", "count_blocks", {})}
    detect = native_function("comfy/model_detection.py", "detect_unet_config", scope)
    classes = dict(
        torch=torch, supported_models_base=NS(BASE=object), latent_formats=NS(SDXL="native")
    )
    hydit = native_class("comfy/supported_models.py", "HunyuanDiT", classes)
    hydit1 = native_class("comfy/supported_models.py", "HunyuanDiT1", classes)

    def config(tensors, prefix, **kwargs):
        result = detect(tensors, prefix)
        if result is None or result.get("image_model") not in ("hydit", "hydit1"):
            return None
        out = (hydit1 if result["image_model"] == "hydit1" else hydit)()
        out.unet_config = result
        return out

    return NS(unet_prefix_from_state_dict=lambda tensors: "model.", model_config_from_unet=config)


class HyDiTRouteTests(unittest.TestCase):
    def test_actual_official_header_shapes_use_native_declared_embedded_prefixes_and_no_acquisition(
        self,
    ):
        # Shapes/prefixes from hash-bound official1.2 header, no neural payload.
        entries = {
            "model.mlp_t5.0.weight": [8192, 2048],
            "model.x_embedder.proj.weight": [1408, 4, 2, 2],
            "model.extra_embedder.0.weight": [5632, 1024],
            "vae.encoder.conv_in.weight": [128, 3, 3, 3],
            "text_encoders.mt5xl.spiece_model": [4309802],
            **{f"model.blocks.{i}.fixture": [1] for i in range(40)},
        }
        detection = native_detector()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(
                "sys.modules",
                {"comfy.model_detection": detection, "comfy": NS(model_detection=detection)},
            ),
        ):
            root = Path(folder)
            model = header(root / "renamed.safetensors", entries)
            manager = Manager(root)
            details = inspect_tensor_model({k: dict(shape=v) for k, v in entries.items()})
            self.assertEqual(
                details,
                dict(
                    architecture="HunyuanDiT",
                    vae_prefixes=("vae.",),
                    text_encoder_prefixes=("text_encoders.",),
                ),
            )
            models = NS(
                get_file=lambda *a: model,
                get_model_base=lambda info: "HunyuanDiT",
                get_models_by_path=lambda *a, **kw: dict(baseModel="HunyuanDiT"),
            )
            state = inventory(model, "Unknown", {}, manager)
            self.assertEqual(state["architecture"], "HunyuanDiT")
            self.assertEqual([r["status"] for r in state["rows"]], ["Embedded in checkpoint"] * 3)
            self.assertFalse(any(r["download_available"] for r in state["rows"]))
            with self.assertRaises(ValueError):
                inventory(
                    model,
                    "HunyuanDiT",
                    {},
                    manager,
                    component_overrides={"clip_hydit_mt5": "other.safetensors"},
                )
            request = component_profiles.snapshot_request(
                dict(base_model_name=model.name), models, {}, manager
            )
            request = image_preparation.capture_image_assets(
                request, manager, {}, models=models, values={}, offline=True
            )
            self.assertFalse(image_preparation.needs_image_preparation(request))
            transfer = Mock(side_effect=AssertionError("Bundled weights require no transfer"))
            service = PreparationService(transport=transfer, staging_root=root / "staging")
            try:
                ready = image_preparation.prepare_image_assets(
                    request, check_cancelled=lambda: None, service=service
                )
                self.assertEqual(ready[image_preparation.KEY].artifacts, ())
                transfer.assert_not_called()
            finally:
                service._executor.shutdown(wait=True)

    def test_split_roles_capture_exact_files_and_reject_missing_tokenizer_or_wrong_branch(self):
        bert = {"bert.encoder.layer.0.attention.self.query.weight": (1024, 1024)}
        mt5 = {
            "encoder.block.23.layer.1.DenseReluDense.wi_1.weight": (5120, 2048),
            "mt5xl.spiece_model": (500,),
        }
        self.assertEqual(
            compatibility("clip_hydit_bert", classify_shapes(bert, lambda x: None)), "compatible"
        )
        self.assertEqual(
            compatibility("clip_hydit_mt5", classify_shapes(mt5, lambda x: None)), "compatible"
        )
        self.assertEqual(
            compatibility("clip_hydit_mt5", classify_shapes(bert, lambda x: None)), "incompatible"
        )
        del mt5["mt5xl.spiece_model"]
        self.assertEqual(
            compatibility("clip_hydit_mt5", classify_shapes(mt5, lambda x: None)), "incompatible"
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manager = Manager(root)
            model = header(root / "split.safetensors", {"fixture": [1]})
            keys = ("clip_hydit_bert", "clip_hydit_mt5", "vae_sdxl")
            paths = {}
            for key in keys:
                category = "clip" if key.startswith("clip_") else "vae"
                p = root / category / RESOURCE_DEFAULTS[key]
                p.parent.mkdir(exist_ok=True)
                p.write_bytes(b"owned local fixture")
                paths[key] = p.resolve()
            models = NS(
                get_file=lambda *a: model,
                get_model_base=lambda info: "HunyuanDiT1",
                get_models_by_path=lambda *a, **kw: dict(baseModel="HunyuanDiT1"),
            )
            request = component_profiles.snapshot_request(
                dict(base_model_name=model.name), models, {}, manager
            )
            request = image_preparation.capture_image_assets(
                request, manager, {}, models=models, values={}, offline=True
            )
            service = PreparationService(
                transport=Mock(side_effect=AssertionError("offline")), staging_root=root / "staging"
            )
            try:
                ready = image_preparation.prepare_image_assets(
                    request, check_cancelled=lambda: None, service=service
                )
                manager.root = root / "changed"
                for key, p in paths.items():
                    category = "clip" if key.startswith("clip_") else "vae"
                    with image_preparation.prepared_asset(
                        ready, category, key, manager, category, service=service
                    ) as actual:
                        self.assertEqual(actual, p)
                paths["clip_hydit_mt5"].write_bytes(b"replacement")
                with self.assertRaises((ValueError, OSError)):
                    with image_preparation.prepared_asset(
                        ready, "clip", "clip_hydit_mt5", manager, "clip", service=service
                    ):
                        pass
            finally:
                service._executor.shutdown(wait=True)

    def test_actual_split_loader_branch_uses_native_target_captured_signatures_and_bundled_bypass(
        self,
    ):
        tree = ast.parse((ROOT / PIPELINE).read_text(encoding="utf-8"))
        branch = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "'clip' in embedded"
        )

        class HunyuanDiT:
            model_config = NS(clip_target=Mock(return_value="native-target"))

        load = Mock(return_value="adapted-clip")
        native_load = Mock(
            side_effect=AssertionError("Pinned split helper must preserve tokenizer")
        )
        device = NS(type="cpu")
        scope = dict(
            embedded={},
            bundle=None,
            clip_paths=["bert.pt", "mt5.safetensors"],
            model_info=dict(clip_type="HUNYUAN_DIT"),
            embedding_paths=["embeddings"],
            component_files=[["bert", 1, 2, 3], ["mt5", 4, 5, 6]],
            is_safetensors=lambda p: p.endswith(".safetensors"),
            torch=torch,
            unet=NS(model=HunyuanDiT()),
            comfy=NS(
                model_base=NS(HunyuanDiT=HunyuanDiT),
                model_management=NS(get_torch_device=lambda: device),
                sd=NS(load_clip=native_load),
            ),
        )
        with patch("ruinedfooocus.backends.comfy.hunyuan_dit.load_hunyuan_dit_clip", load):
            exec(compile(ast.Module(body=[branch], type_ignores=[]), PIPELINE, "exec"), scope)
        self.assertEqual(scope["clip"], "adapted-clip")
        self.assertEqual(load.call_args.args, (["bert.pt", "mt5.safetensors"], "native-target"))
        self.assertEqual(load.call_args.kwargs["signatures"], scope["component_files"])
        self.assertEqual(load.call_args.kwargs["model_options"]["load_device"], torch.device("cpu"))
        scope["embedded"] = {"clip": "already-native-bundled"}
        exec(compile(ast.Module(body=[branch], type_ignores=[]), PIPELINE, "exec"), scope)
        self.assertEqual(scope["clip"], "already-native-bundled")
        load.assert_called_once()

        class CLIP:
            pass

        class VAE:
            first_stage_model = object()
            throw_exception_if_invalid = Mock()

        checkpoint = Mock(return_value=("native-unet", CLIP(), VAE(), None))
        full = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and ast.unparse(n.test)
            == "bundle is not None or Path(filename).suffix.lower() == '.gguf' or unet_only"
        )
        scope.update(
            filename="official1.2.safetensors",
            load_checkpoint_guess_config=checkpoint,
            CLIP=CLIP,
            VAE=VAE,
            self=NS(
                get_clip_and_vae=Mock(
                    return_value=dict(
                        clip_keys=["clip_hydit_bert", "clip_hydit_mt5"], vae_key="vae_sdxl"
                    )
                )
            ),
            choices={},
            validate_component_slots=validate_component_slots,
        )
        wrapper = ast.parse("def run_native_checkpoint():\n    pass\n    return locals()")
        wrapper.body[0].body = full.orelse + wrapper.body[0].body[1:]
        exec(compile(ast.fix_missing_locations(wrapper), PIPELINE, "exec"), scope)
        scope.update(scope["run_native_checkpoint"]())
        checkpoint.assert_called_once_with(
            "official1.2.safetensors", embedding_directory=["embeddings"], output_clipvision=True
        )
        self.assertIsInstance(scope["clip"], CLIP)
        load.assert_called_once()
        info = method(PIPELINE, "get_clip_and_vae", dict(settings=NS(default_settings={})))(
            NS(known_model_info=route_info()),
            NS(
                model=type(
                    "HunyuanDiT", (), {"model_config": NS(unet_config={"image_model": "hydit1"})}
                )()
            ),
        )
        self.assertEqual(info["clip_type"], "HUNYUAN_DIT")
        self.assertEqual(info["clip_keys"], ["clip_hydit_bert", "clip_hydit_mt5"])
        self.assertEqual(info["vae_key"], "vae_sdxl")

    def test_production_text_encode_and_sampler_preserve_native_dual_metadata_and_geometry(self):
        metadata = dict(
            attention_mask=torch.ones(1, 2),
            conditioning_mt5xl=torch.ones(1, 3, 4),
            attention_mask_mt5xl=torch.ones(1, 3),
        )
        cond = [[torch.ones(1, 2, 4), metadata]]
        encode = native_method("nodes.py", "CLIPTextEncode", "encode", {})()
        clip = NS(
            tokenize=Mock(return_value={"hydit_clip": "bert", "mt5xl": "mt5"}),
            encode_from_tokens_scheduled=Mock(return_value=cond),
        )
        run = method(
            PIPELINE,
            "textencode",
            dict(
                CLIPTextEncode=lambda: encode,
                warm_patched_text_encoder=Mock(),
                path_manager=NS(model_paths={"embedding_path": [], "cache_path": ROOT / "tmp"}),
            ),
        )
        owner = NS(conditions={"+": dict(text=None)}, xl_base_patched=NS(clip=clip))
        with patch(
            "ruinedfooocus.features.models.embedding_browser.native_embedding_text",
            side_effect=lambda t, *a: t,
        ):
            run(owner, "+", "a cat", 1)
        self.assertIs(owner.conditions["+"]["cache"], cond)
        clip.encode_from_tokens_scheduled.assert_called_once_with(
            {"hydit_clip": "bert", "mt5xl": "mt5"}
        )
        sample = next(
            n
            for n in ast.walk(ast.parse((ROOT / PIPELINE).read_text(encoding="utf-8")))
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == "comfy.sample.sample"
        )
        sampler = Mock(return_value="sampled")
        scope = dict(
            comfy=NS(sample=NS(sample=sampler), utils=NS(PROGRESS_BAR_ENABLED=True)),
            gen_data=dict(steps=7),
            sampling_model="native-hydit",
            noise="noise",
            cfg=3.25,
            sampler_name="euler",
            scheduler="sgm_uniform",
            positive_cond=owner.conditions["+"]["cache"],
            negative_conditioning=cond,
            latent_image=torch.zeros(1, 4, 8, 12),
            denoise=None,
            noise_mask=None,
            callback_function=None,
            seed=37,
        )
        exec(compile(ast.Module(body=[sample], type_ignores=[]), PIPELINE, "exec"), scope)
        self.assertEqual(sampler.call_args.kwargs["cfg"], 3.25)
        self.assertEqual(sampler.call_args.kwargs["seed"], 37)
        self.assertIs(sampler.call_args.kwargs["positive"][0][1], metadata)

        class Base:
            def extra_conds(self, **kwargs):
                return {}

        native_scope = dict(
            BaseModel=Base, comfy=NS(conds=NS(CONDRegular=lambda value: value)), torch=torch
        )
        extra = native_method(
            "comfy/model_base.py", "HunyuanDiT", "extra_conds", native_scope, bases=("BaseModel",)
        )().extra_conds
        native_encode = native_function("comfy/samplers.py", "encode_model_conds", {})
        for width, height in ((96, 64), (64, 96)):
            result = native_encode(
                extra,
                [metadata | dict(model_conds={})],
                torch.zeros(1, 4, height // 8, width // 8),
                "cpu",
                "positive",
            )[0]["model_conds"]
            self.assertIs(result["text_embedding_mask"], metadata["attention_mask"])
            self.assertIs(result["encoder_hidden_states_t5"], metadata["conditioning_mt5xl"])
            self.assertIs(result["text_embedding_mask_t5"], metadata["attention_mask_mt5xl"])
            self.assertEqual(
                result["image_meta_size"].tolist(), [[height, width, height, width, 0, 0]]
            )

    def test_native_auto_and_manual_cfg_or_saved_recipe_keep_precedence(self):
        registry = NS(
            KSampler=NS(SAMPLERS=recipes.SAMPLERS, SCHEDULERS=recipes.SCHEDULERS + ["sgm_uniform"])
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict("sys.modules", {"comfy.samplers": registry}),
            patch(
                "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                return_value={
                    "architecture": "HunyuanDiT",
                    "native_class": "HunyuanDiT",
                    "native_config": dict(image_model="hydit"),
                },
            ),
        ):
            root = Path(folder)
            model = root / "renamed.safetensors"
            model.write_bytes(b"already independently hashed fixture")
            data = dict(
                baseModel="HunyuanDiT",
                _rf_local_sha256=SHA,
                _rf_file_signature=file_signature(model),
            )
            cache = root / "renamed.json"
            cache.write_text(json.dumps(data))
            perf = recipes.RecommendationTests()._performance(root)
            perf.model_recommendations.offline = True
            values = perf.get_perf_options("Model Defaults", str(model))
            self.assertEqual(
                tuple(values[k] for k in ("custom_steps", "cfg", "sampler_name", "scheduler")),
                (30, 6, "euler", "sgm_uniform"),
            )
            manual = dict(
                performance_selection="Own settings", cfg=3.25, base_model_name=str(model)
            )
            self.assertEqual(perf.apply(manual, "HunyuanDiT"), manual)
            for override in (
                dict(_rf_local_sha256="0" * 64),
                dict(_rf_file_signature=None),
                dict(baseModel="HunyuanDiT1"),
            ):
                cache.write_text(json.dumps(data | override))
                if override.get("baseModel") == "HunyuanDiT1":
                    with self.assertRaises(UnavailableModelRecipe):
                        perf.get_perf_options("Model Defaults", str(model))
                else:
                    self.assertEqual(perf.get_perf_options("Model Defaults", str(model))["cfg"], 6)
            cache.write_text(
                json.dumps(
                    data
                    | dict(
                        generationConfig=dict(
                            steps=9, cfgScale=2.5, sampler="Euler", scheduleType="normal"
                        )
                    )
                )
            )
            self.assertEqual(perf.get_perf_options("Model Defaults", str(model))["cfg"], 2.5)
            cache.write_text(json.dumps(data))
            model.write_bytes(model.read_bytes() + b"changed")
            self.assertEqual(perf.get_perf_options("Model Defaults", str(model))["cfg"], 6)

    def test_registry_only_adds_exact_native_bundled_acquisition(self):
        entry = next(
            r
            for r in load_registry().entries()
            if r["id"] == "existing:checkpoints/hunyuan_dit_1.2.safetensors"
        )
        self.assertEqual(entry["artifact"]["sha256"], SHA)
        self.assertEqual(entry["artifact"]["bytes"], 8240228270)
        self.assertEqual(entry["bundled_components"], ("clip", "vae"))


if __name__ == "__main__":
    unittest.main()
