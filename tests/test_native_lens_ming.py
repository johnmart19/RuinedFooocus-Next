"""Lens and Ming use native text dispatch, latent geometry and captured roles."""

import ast
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from ruinedfooocus.features.models import component_profiles
from ruinedfooocus.features.models.component_candidates import classify_shapes, compatibility
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.features.models.resource_defaults import RESOURCE_DEFAULTS
from ruinedfooocus.features.models.performance import UnavailableModelRecipe
from ruinedfooocus.features.image_editing.image_guidance import resolve_guidance, request_model
from test_component_candidates import qwen_wan21_shapes
from test_image_guidance import FakePatcher
from test_image_preparation import Manager
from test_model_cache import method
import test_model_recommendations as recommendations
from test_native_model_coverage import NATIVE, PIPELINE, route_info, native_method
from test_native_ordinary_pair import native_function
from test_native_variant_defaults import native_backend, write_header


CONTRACTS = {
    "Lens": ("LENS", "FLUX2", ["clip_lens_gpt_oss", "vae_flux2"]),
    "MingImage": ("STABLE_DIFFUSION", "SD3", ["clip_ming_image", "vae_ming_image"]),
}


class LensMingTests(unittest.TestCase):
    def test_inventory_runtime_and_frozen_submission_use_dedicated_roles(self):
        get_info = method(PIPELINE, "get_clip_and_vae", dict(settings=NS(default_settings={})))
        for family, (clip_type, latent, keys) in CONTRACTS.items():
            with self.subTest(family=family), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                manager = Manager(root)
                model = write_header(root / "renamed.safetensors", {"unknown.weight": [1]})
                for key in keys:
                    path = (
                        root
                        / ("clip" if key.startswith("clip_") else "vae")
                        / RESOURCE_DEFAULTS[key]
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"installed inert fixture")
                models = NS(
                    get_file=lambda *args: model,
                    get_models_by_path=lambda *a, **kw: {"baseModel": family},
                    get_model_base=lambda value: value["baseModel"],
                )
                state = component_profiles.component_inventory(model, family, {}, manager)
                self.assertEqual([row["key"] for row in state["rows"]], keys)
                self.assertTrue(all(row["status"] == "Installed" for row in state["rows"]))
                runtime = type(family, (), {"model_config": NS(unet_config={})})()
                info = get_info(NS(known_model_info=route_info()), NS(model=runtime))
                self.assertEqual((info["clip_type"], info["latent"]), (clip_type, latent))
                request = component_profiles.snapshot_request(
                    {"base_model_name": model.name}, models, {}, manager
                )
                self.assertEqual(request[component_profiles.REQUEST_KEY], dict.fromkeys(keys))

    def test_pinned_native_single_encoder_dispatch_forwards_tokenizer_and_quantization(self):
        tree = ast.parse((NATIVE / "comfy/sd.py").read_text(encoding="utf-8"))
        for native_type, owner, tokenizer in (
            ("GPT_OSS_20B", "gpt_oss", "LensTokenizer"),
            ("MING_IMAGE", "ming_image", "MingImageTokenizer"),
        ):
            with self.subTest(native_type=native_type):
                branch = next(
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.If)
                    and ast.unparse(node.test) == f"te_model == TEModel.{native_type}"
                )
                factory = Mock(return_value="native encoder")
                encoder = NS(
                    **{
                        tokenizer: "native tokenizer",
                        "lens_te" if owner == "gpt_oss" else "te": factory,
                    }
                )
                quant = {"layers": "native quantization"}
                scope = dict(
                    clip_target=NS(),
                    clip_data=[
                        {
                            "tokenizer_json": "embedded tokenizer",
                            "thinker.norm.weight": NS(dtype="native dtype"),
                        }
                    ],
                    tokenizer_data={},
                    llama_detect=lambda data: dict(
                        dtype_llama="native dtype", llama_quantization_metadata=quant
                    ),
                    comfy=NS(
                        text_encoders=NS(**{owner: encoder}),
                        utils=NS(detect_layer_quantization=lambda *a: quant),
                    ),
                )
                exec(
                    compile(
                        ast.Module(body=branch.body, type_ignores=[]),
                        "native_clip_dispatch",
                        "exec",
                    ),
                    scope,
                )
                self.assertEqual(scope["tokenizer_data"]["tokenizer_json"], "embedded tokenizer")
                self.assertEqual(scope["clip_target"].tokenizer, "native tokenizer")
                self.assertEqual(
                    factory.call_args.kwargs,
                    dict(dtype_llama="native dtype", llama_quantization_metadata=quant),
                )

    def test_native_lens_layer_features_keep_attention_alignment_and_empty_negative(self):
        encoder = native_method(
            "comfy/text_encoders/gpt_oss.py",
            "LensGptOssClipModel",
            "encode_token_weights",
            dict(torch=torch),
        )()
        encoder.selected_layers, encoder.txt_offset = (1, 2), 1
        encoder.dtype, encoder.execution_device = torch.float32, "cpu"
        mask = torch.tensor([[1, 1, 0]])
        encoder._gather_tokens = Mock(return_value=(torch.ones(1, 3, dtype=torch.long), mask))
        encoder.transformer = Mock(
            return_value={"hidden_states": [torch.ones(1, 3, 2), torch.full((1, 3, 2), 2.0)]}
        )
        features, pooled, extra = encoder.encode_token_weights([[(1, 1)]])
        self.assertEqual(tuple(features.shape), (1, 2, 4))
        torch.testing.assert_close(extra["attention_mask"], mask[:, 1:])
        self.assertEqual(extra["num_layers_stacked"], 2)
        self.assertIsNone(pooled)
        encoder.config = NS(hidden_size=2)
        empty, _, empty_extra = encoder.encode_token_weights([[]])
        self.assertEqual(tuple(empty.shape), (1, 0, 4))
        self.assertEqual(tuple(empty_extra["attention_mask"].shape), (1, 0))

    def test_ming_vae_requires_rgba_and_native_single_frame_latent(self):
        shapes = qwen_wan21_shapes()
        shapes["encoder.conv1.weight"] = (96, 4, 3, 3, 3)
        shapes["decoder.head.2.weight"] = (4, 96, 3, 3, 3)
        classified = classify_shapes(shapes, detector=lambda _: None)
        self.assertEqual(compatibility("vae_ming_image", classified), "compatible")
        self.assertEqual(compatibility("vae_qwen_image", classified), "incompatible")
        self.assertEqual(
            compatibility(
                "vae_ming_image", classify_shapes(qwen_wan21_shapes(), detector=lambda _: None)
            ),
            "incompatible",
        )
        fix = native_function("comfy/sample.py", "fix_empty_latent_channels", dict(torch=torch))
        latent_format = NS(
            latent_channels=16,
            latent_dimensions=3,
            spacial_downscale_ratio=8,
            fix_empty_latent=lambda value: value,
        )
        latent = fix(NS(get_model_object=lambda key: latent_format), torch.zeros(1, 16, 6, 8), 8)
        self.assertEqual(tuple(latent.shape), (1, 16, 1, 6, 8))
        decoder = native_method("nodes.py", "VAEDecode", "decode", {})()
        rgba = torch.ones(1, 1, 48, 64, 4)
        output = decoder.decode(NS(decode=lambda value: rgba), {"samples": latent})[0]
        self.assertEqual(tuple(output.shape), (1, 48, 64, 4))

    def test_exact_catalogue_companions_and_declared_variant_recipes(self):
        registry = load_registry()
        for filename, family, variant in (
            ("lens_bf16.safetensors", "Lens", "base"),
            ("ming_image_0.1_design_int8_convrot.safetensors", "MingImage", "design"),
        ):
            row = registry.artifact("existing:checkpoints/" + filename)
            self.assertEqual(row["native_family"], family)
            self.assertEqual(row["behavioral_variant"], variant)
            self.assertEqual(len(row["companions"]["text"]), 2)
            self.assertFalse(row["artifact"]["full_file_hash_recomputed"])
        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            perf = recommendations.RecommendationTests()._performance(root)
            perf.model_recommendations.offline = True
            for family, variant, steps, cfg in (
                ("Lens", "base", 20, 5),
                ("Lens", "turbo", 4, 1),
                ("MingImage", "design", 12, 1),
            ):
                model = write_header(
                    root / "misleading_turbo.safetensors",
                    {"unknown.weight": [1]},
                    metadata={"variant": variant},
                )
                values = perf.get_perf_options("Model Defaults", str(model), family)
                self.assertEqual((values["custom_steps"], values["cfg"]), (steps, cfg))
                write_header(model, {"unknown.weight": [1]})
                with self.assertRaises(UnavailableModelRecipe):
                    perf.get_perf_options("Model Defaults", str(model), family)

    def test_recipe_guidance_is_request_local_and_explicit_values_win(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        perf = recommendations.RecommendationTests()._performance(root)
        recipe = dict(perf.default_settings, cfg_norm_strength=1)
        with patch.object(perf, "get_perf_options", return_value=recipe):
            request = perf.apply(
                dict(
                    performance_selection="Model Defaults",
                    base_model_name="selected",
                    cfg_norm_strength="",
                    cfg_norm_auto=True,
                ),
                "Lens",
            )
        lens = resolve_guidance(request, {}, "Lens")
        self.assertEqual(lens["cfg_norm_strength"], 1)
        self.assertEqual(
            resolve_guidance(dict(request, cfg_norm_strength=0), {}, "Lens")["cfg_norm_strength"], 0
        )
        self.assertIsNone(
            resolve_guidance(
                dict(request, cfg_norm_strength=None, cfg_norm_auto=False), {}, "Lens"
            )["cfg_norm_strength"]
        )
        self.assertIsNone(
            resolve_guidance({"cfg_norm_strength": None}, {}, "Lens")["cfg_norm_strength"]
        )
        cleared = perf.apply(dict(request, performance_selection="Own settings"), "MingImage")
        self.assertNotIn("_model_default_guidance", cleared)
        self.assertIsNone(resolve_guidance(cleared, {}, "MingImage")["cfg_norm_strength"])
        request = dict(
            width=832,
            height=1216,
            _model_default_guidance={"model_sampling_flux": {"max_shift": 1.15, "base_shift": 0.5}},
        )
        guidance = resolve_guidance(request, {}, "MingImage")
        patcher = Mock(side_effect=lambda **kwargs: [kwargs["model"]])
        request_model(FakePatcher(), guidance, "MingImage", {"Flux": patcher})
        self.assertEqual(patcher.call_args.kwargs["width"], 832)
        self.assertEqual(patcher.call_args.kwargs["height"], 1216)
        explicit = resolve_guidance(dict(request, sampling_shift=2), {}, "MingImage")
        self.assertNotIn("_model_sampling_flux", explicit)

    def test_encoder_candidates_require_native_type_and_embedded_tokenizer(self):
        from ruinedfooocus.features.models.component_candidates import classify_shapes

        detector = native_function(
            "comfy/sd.py",
            "detect_te_model",
            dict(TEModel=NS(GPT_OSS_20B=NS(name="GPT_OSS_20B"), MING_IMAGE=NS(name="MING_IMAGE"))),
        )
        for role, shapes in (
            (
                "clip_lens_gpt_oss",
                {
                    "layers.0.self_attn.sinks": (64,),
                    "layers.0.mlp.experts.gate_up_proj.weight": (32, 5760, 2880),
                },
            ),
            ("clip_ming_image", {"thinker.layers.1.mlp.image_gate.proj.weight": (1536, 2048)}),
        ):
            self.assertEqual(compatibility(role, classify_shapes(shapes, detector)), "incompatible")
            shapes["tokenizer_json"] = (2048,)
            self.assertEqual(compatibility(role, classify_shapes(shapes, detector)), "compatible")

    def test_verified_exact_source_variant_qualifies_without_hashing_in_auto(self):
        from ruinedfooocus.features.models.model_identity import file_signature

        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            model = write_header(root / "unhelpful.safetensors", {"unknown.weight": [1]})
            perf = recommendations.RecommendationTests()._performance(root)
            (root / "unhelpful.json").write_text(
                json.dumps({"baseModel": "Lens", "_rf_file_signature": file_signature(model)}),
                encoding="utf-8",
            )
            with patch(
                "ruinedfooocus.features.models.model_identity.verified_native_source",
                return_value={"native_family": "Lens", "behavioral_variant": "base"},
            ) as verified:
                self.assertEqual(perf.family_preset("Lens", str(model)), "Lens")
                self.assertEqual(
                    verified.call_args.kwargs["current_signature"], file_signature(model)
                )
            with patch(
                "ruinedfooocus.features.models.model_identity.verified_native_source",
                return_value={"native_family": "MingImage", "behavioral_variant": "design"},
            ):
                self.assertIsNone(perf.family_preset("Lens", str(model)))

    def test_native_ming_direct_context_survives_encoder_and_model_conditioning(self):
        encoder = native_method(
            "comfy/text_encoders/ming_image.py", "MingImageClipModel", "forward", {}
        )()
        encoder.execution_device = "cpu"
        encoder.process_tokens = Mock(return_value=("embeds", "mask", 2, "metadata"))
        encoder.transformer = Mock(
            return_value=(
                torch.ones(1, 2, 3, dtype=torch.bfloat16),
                torch.ones(1, 4, 3, dtype=torch.bfloat16),
            )
        )
        context, pooled, extra = encoder.forward("native tokens")
        self.assertEqual(context.dtype, torch.float32)
        self.assertEqual(extra["direct_context"].dtype, torch.float32)
        self.assertIsNone(pooled)

        class Parent:
            def extra_conds(self, **kwargs):
                return {}

        scope = dict(Parent=Parent, comfy=NS(conds=NS(CONDRegular=lambda value: value)))
        model = native_method(
            "comfy/model_base.py", "MingImage", "extra_conds", scope, bases=("Parent",)
        )()
        self.assertIs(model.extra_conds(**extra)["direct_context"], extra["direct_context"])

    def test_ming_dynamic_size_patch_executes_pinned_native_sampling_node(self):
        source = ast.parse((NATIVE / "comfy/model_sampling.py").read_text(encoding="utf-8"))
        node = next(
            n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "ModelSamplingFlux"
        )
        scope = dict(
            torch=torch,
            flux_time_shift=native_function(
                "comfy/model_sampling.py", "flux_time_shift", dict(math=math)
            ),
        )
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "native_flux_sampling", "exec"), scope
        )
        sampling = NS(ModelSamplingFlux=scope["ModelSamplingFlux"], CONST=type("CONST", (), {}))
        native = native_method(
            "comfy_extras/nodes_model_advanced.py",
            "ModelSamplingFlux",
            "patch",
            dict(comfy=NS(model_sampling=sampling)),
        )()

        class Patcher:
            def __init__(self):
                self.model = NS(model_config=NS(sampling_settings={}))
                self.objects = {}

            def clone(self):
                return Patcher()

            def add_object_patch(self, key, value):
                self.objects[key] = value

        base = Patcher()
        shifts = []
        for width, height in ((1024, 1024), (832, 1216)):
            guidance = resolve_guidance(
                dict(
                    width=width,
                    height=height,
                    _model_default_guidance={
                        "model_sampling_flux": {"max_shift": 1.15, "base_shift": 0.5}
                    },
                ),
                {},
                "MingImage",
            )
            patched = request_model(base, guidance, "MingImage", {"Flux": native.patch})
            shifts.append(patched.objects["model_sampling"].shift)
            self.assertTrue(torch.isfinite(patched.objects["model_sampling"].sigmas).all())
        self.assertAlmostEqual(shifts[0], 1.15)
        self.assertNotEqual(shifts[0], shifts[1])
        self.assertEqual(base.objects, {})

    def test_actual_gui_model_transition_capture_uses_auto_and_keeps_off(self):
        import gradio as gr
        from ruinedfooocus.features.image_editing import ui_guidance as ui
        from ruinedfooocus.features.image_editing.image_guidance import capabilities, FIELD_SCHEMA
        from ruinedfooocus.features.gallery.image_metadata import control_value
        from ruinedfooocus.core.settings_schema import validate_known

        source = Path(ui.__file__)
        callback = next(
            n
            for n in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name == "model_changed"
        )
        selected = {"family": "Lens"}
        scope = dict(
            gr=gr,
            defaults={},
            family_for_checkpoint=lambda name: selected["family"],
            control_visibility=ui.control_visibility,
            capabilities=capabilities,
            resolve_guidance=resolve_guidance,
            references_visible=ui.references_visible,
            normalization_auto_default=ui.normalization_auto_default,
            control_value=control_value,
            LCM_FIELDS=ui.LCM_FIELDS,
        )
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(source), "exec"), scope)
        with patch(
            "ruinedfooocus.features.generation.image_sampling.hidream_o1_dev", return_value=False
        ):
            updates = scope["model_changed"]("lens", "Disabled")
            captured = dict(
                performance_selection="Model Defaults",
                base_model_name="lens",
                cfg_norm_strength=updates[3]["value"],
                cfg_norm_auto=updates[-1]["value"],
            )
            self.assertEqual(captured["cfg_norm_strength"], "")
            self.assertTrue(captured["cfg_norm_auto"])
            root = Path(self.enterContext(tempfile.TemporaryDirectory()))
            perf = recommendations.RecommendationTests()._performance(root)
            with patch.object(
                perf,
                "get_perf_options",
                return_value=dict(perf.default_settings, cfg_norm_strength=1),
            ):
                queued = perf.apply(captured, "Lens")
                self.assertEqual(resolve_guidance(queued, {}, "Lens")["cfg_norm_strength"], 1)
                off = perf.apply(dict(captured, cfg_norm_auto=False), "Lens")
                self.assertIsNone(resolve_guidance(off, {}, "Lens")["cfg_norm_strength"])
                custom = perf.apply(dict(captured, cfg_norm_strength=0), "Lens")
                self.assertEqual(resolve_guidance(custom, {}, "Lens")["cfg_norm_strength"], 0)
                selected["family"] = "Flux"
                other_updates = scope["model_changed"]("flux", "Disabled")
                other = perf.apply(
                    dict(
                        queued,
                        base_model_name="flux",
                        cfg_norm_strength=other_updates[3]["value"],
                        cfg_norm_auto=other_updates[-1]["value"],
                    ),
                    "Flux",
                )
            self.assertFalse(other["cfg_norm_auto"])
            self.assertNotIn("_model_default_guidance", other)
            self.assertIsNone(resolve_guidance(other, {}, "Flux")["cfg_norm_strength"])
        # The preset-aware callback used by Create shares the same intent reset.
        initial = {key: spec["default"] for key, spec in FIELD_SCHEMA.items()}
        transition = ui.preset_control_state(
            initial, {}, {}, 7, "Lens", "lens", "lens", {"checkpoint": "flux", "backup": None}
        )
        self.assertTrue(transition["values"]["cfg_norm_auto"])
        self.assertFalse(ui.normalization_auto_default("Lens", {"cfg_norm_strength": None}))
        self.assertEqual(validate_known({"cfg_norm_auto": True}), {"cfg_norm_auto": True})
        with self.assertRaises(ValueError):
            validate_known({"cfg_norm_auto": "on"})


if __name__ == "__main__":
    unittest.main()
