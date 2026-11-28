import json
import tempfile
import time
import sys
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from ruinedfooocus.features.models.model_recommendations import (
    ModelRecommendations,
    example_settings,
    extract,
)
from ruinedfooocus.features.models.resource_defaults import performance_defaults


SAMPLERS = ["euler", "euler_ancestral"]
SCHEDULERS = ["normal", "simple"]


def examples(count=3, **values):
    return [
        {
            "id": i + 1,
            "meta": dict(steps=8, cfgScale=1, sampler="Euler", scheduler="normal", **values),
        }
        for i in range(count)
    ]


class RecommendationTests(unittest.TestCase):
    def test_cached_exact_variant_examples_work_offline_with_family_fallback_and_overrides(self):
        import hashlib
        from ruinedfooocus.features.models.model_identity import file_signature

        registry = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=SAMPLERS, SCHEDULERS=SCHEDULERS)
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(sys.modules, {"comfy.samplers": registry}),
        ):
            root = Path(folder)
            model = root / "renamed-fp16.safetensors"
            model.write_bytes(b"selected local variant")
            checksum = hashlib.sha256(model.read_bytes()).hexdigest()
            images = examples(3, clipSkip=2)
            for image in images:
                image["meta"].update(steps=30, cfgScale=3, sampler="Euler a")
                image["meta"].pop("scheduler")
            metadata = dict(
                id=123,
                modelId=456,
                baseModel="Illustrious",
                images=images,
                _rf_local_sha256=checksum,
                _rf_file_signature=file_signature(model),
                files=[
                    dict(name="other-fp8.safetensors", hashes=dict(SHA256="b" * 64)),
                    dict(
                        name="publisher-original.safetensors", hashes=dict(SHA256=checksum.upper())
                    ),
                ],
            )
            metadata_path = root / (model.stem + ".json")
            metadata_path.write_text(json.dumps(metadata))
            performance = self._performance(root)
            store = performance.model_recommendations
            store.offline = True
            with (
                patch(
                    "ruinedfooocus.features.models.model_sources.civitai_metadata",
                    side_effect=AssertionError("network"),
                ),
                patch(
                    "ruinedfooocus.features.models.model_identity.cached_file_sha256",
                    side_effect=AssertionError("render hash"),
                ),
            ):
                values = performance.get_perf_options("Model Defaults", str(model))
                self.assertEqual(
                    tuple(
                        values[key]
                        for key in ("custom_steps", "cfg", "sampler_name", "scheduler", "clip_skip")
                    ),
                    (30, 3, "euler_ancestral", "normal", 2),
                )
                self.assertIn(
                    "cached version example consensus",
                    store.details(str(model), SAMPLERS, SCHEDULERS)["source"],
                )
                metadata_path.write_text(
                    json.dumps(metadata | dict(generationConfig=dict(cfgScale=6)))
                )
                self.assertEqual(
                    store.details(str(model), SAMPLERS, SCHEDULERS)["recommended"]["cfg"], 6
                )
                store.save_override(str(model), values | dict(cfg=9), SAMPLERS, SCHEDULERS)
                self.assertEqual(store.recipe(str(model), SAMPLERS, SCHEDULERS)[0]["cfg"], 9)
                metadata_path.write_text(json.dumps(metadata))
                store.reset_override(str(model))
                self.assertEqual(store.recipe(str(model), SAMPLERS, SCHEDULERS)[0]["cfg"], 3)

    def test_cached_examples_reject_unknown_variant_and_changed_or_ported_signature(self):
        import hashlib
        from ruinedfooocus.features.models.model_identity import file_signature

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "same-name.safetensors"
            model.write_bytes(b"selected variant")
            checksum = hashlib.sha256(model.read_bytes()).hexdigest()
            metadata = dict(
                id=123,
                images=examples(),
                _rf_local_sha256=checksum,
                _rf_file_signature=file_signature(model),
                files=[dict(hashes=dict(SHA256=checksum))],
            )
            path = root / (model.stem + ".json")
            store = ModelRecommendations(root, root / "settings.json", offline=True)
            for changes in (
                {"_rf_local_sha256": "b" * 64},
                {"files": []},
                {"_rf_file_signature": None},
                {
                    "_rf_file_signature": [
                        "/other/os/same-name.safetensors",
                        *file_signature(model)[1:],
                    ]
                },
                {"id": 0},
            ):
                with self.subTest(changes=changes):
                    path.write_text(json.dumps(metadata | changes))
                    self.assertEqual(
                        store.details(str(model), SAMPLERS, SCHEDULERS)["recommended"], {}
                    )
            path.write_text(json.dumps(metadata))
            model.write_bytes(b"a different variant now occupies the same path")
            self.assertEqual(store.details(str(model), SAMPLERS, SCHEDULERS)["recommended"], {})

    def test_kandinsky_native_baseline_preserves_publisher_and_user_overrides(self):
        from ruinedfooocus.features.models.performance import KANDINSKY5_LITE_SHA256
        from ruinedfooocus.features.models.model_identity import file_signature

        registry = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=SAMPLERS, SCHEDULERS=SCHEDULERS)
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(
                sys.modules,
                {"comfy.samplers": registry, "comfy.model_detection": SimpleNamespace()},
            ),
            patch(
                "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                return_value={
                    "architecture": "Kandinsky5Image",
                    "native_class": "Kandinsky5Image",
                    "native_config": dict(
                        image_model="kandinsky5", model_dim=2560, visual_embed_dim=64
                    ),
                },
            ),
        ):
            root = Path(folder)
            model = root / "arbitrary-name.safetensors"
            model.write_bytes(b"simulated already-hashed local payload")
            metadata = dict(
                baseModel="Kandinsky5Image",
                _rf_local_sha256=KANDINSKY5_LITE_SHA256,
                _rf_file_signature=file_signature(model),
            )
            cache = root / (model.stem + ".json")
            cache.write_text(json.dumps(metadata))
            performance = self._performance(root)
            performance.model_recommendations.offline = True
            values = performance.get_perf_options("Model Defaults", str(model))
            self.assertEqual(
                tuple(
                    values[key]
                    for key in ("custom_steps", "cfg", "sampler_name", "scheduler", "clip_skip")
                ),
                (50, 3.5, "euler", "simple", 1),
            )
            self.assertNotIn("sampling_shift", values)  # Native model supplies its own shift3.
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
            cache.write_text(json.dumps(metadata))
            models = SimpleNamespace(get_file=lambda *args: model)
            with patch.dict(
                sys.modules, {"ruinedfooocus.app.state": SimpleNamespace(models=models)}
            ):
                self.assertEqual(
                    performance.get_perf_options("Model Defaults", model.name)["cfg"], 3.5
                )
                moved = root / "other" / model.name
                moved.parent.mkdir()
                moved.write_bytes(model.read_bytes())
                models.get_file = lambda *args: moved
                self.assertEqual(
                    performance.get_perf_options("Model Defaults", model.name, "Kandinsky5Image")[
                        "cfg"
                    ],
                    3.5,
                )
            model.write_bytes(model.read_bytes() + b"replacement")
            self.assertEqual(
                performance.get_perf_options("Model Defaults", str(model), "Kandinsky5Image")[
                    "cfg"
                ],
                3.5,
            )

    def test_kandinsky_release_file_hash_and_unknown_variants_do_not_qualify_auto(self):
        from ruinedfooocus.features.models.performance import KANDINSKY5_LITE_SHA256
        from ruinedfooocus.features.models.model_identity import file_signature

        registry = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=SAMPLERS, SCHEDULERS=SCHEDULERS)
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(sys.modules, {"comfy.samplers": registry}),
        ):
            root = Path(folder)
            model = root / "kandinsky5lite_t2i.safetensors"
            model.write_bytes(b"local fixture")
            cache = root / (model.stem + ".json")
            performance = self._performance(root)
            performance.model_recommendations.offline = True
            for identity in (
                dict(_rf_local_sha256="c" * 64),
                dict(files=[dict(hashes=dict(SHA256=KANDINSKY5_LITE_SHA256))]),
            ):
                cache.write_text(
                    json.dumps(
                        dict(baseModel="Kandinsky5Image", _rf_file_signature=file_signature(model))
                        | identity
                    )
                )
                with (
                    self.subTest(identity=identity),
                    self.assertRaisesRegex(ValueError, "Choose Own settings"),
                ):
                    performance.get_perf_options("Model Defaults", str(model))
            request = dict(
                performance_selection="Own settings",
                base_model_name=str(model),
                custom_steps=7,
                cfg=1,
            )
            self.assertEqual(performance.apply(request, "Kandinsky5Image"), request)

    def test_new_native_families_require_complete_sampling_recipe_for_auto(self):
        from ruinedfooocus.features.models.performance import EXPLICIT_RECIPE_FAMILIES

        registry = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=SAMPLERS, SCHEDULERS=SCHEDULERS)
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            performance = self._performance(root)
            for family in EXPLICIT_RECIPE_FAMILIES:
                checkpoint = family + ".safetensors"
                metadata = root / (family + ".json")
                metadata.write_text(
                    json.dumps(dict(baseModel=family, generationConfig=dict(steps=12)))
                )
                with (
                    self.subTest(family=family),
                    patch.dict(sys.modules, {"comfy.samplers": registry}),
                ):
                    with self.assertRaisesRegex(ValueError, "Choose Own settings"):
                        performance.get_perf_options("Model Defaults", checkpoint)
                    request = dict(
                        performance_selection="Own settings",
                        base_model_name=checkpoint,
                        custom_steps=17,
                        cfg=2,
                        sampler_name="euler",
                        scheduler="simple",
                        clip_skip=1,
                    )
                    self.assertEqual(performance.apply(request, family), request)
                    # A complete version-specific published recipe is usable; no family guess.
                    metadata.write_text(
                        json.dumps(
                            dict(
                                baseModel=family,
                                generationConfig=dict(
                                    steps=12, cfgScale=3, sampler_name="euler", scheduler="normal"
                                ),
                            )
                        )
                    )
                    if family == "QwenImage21":
                        with self.assertRaisesRegex(ValueError, "Choose Own settings"):
                            performance.get_perf_options("Model Defaults", checkpoint)
                    else:
                        values = performance.get_perf_options("Model Defaults", checkpoint)
                        self.assertEqual(
                            tuple(
                                values[key]
                                for key in ("custom_steps", "cfg", "sampler_name", "scheduler")
                            ),
                            (12, 3, "euler", "normal"),
                        )
                    explicit = dict(
                        custom_steps=9,
                        cfg=1,
                        sampler_name="euler_ancestral",
                        scheduler="simple",
                        clip_skip=2,
                    )
                    performance.model_recommendations.save_override(
                        checkpoint, explicit, SAMPLERS, SCHEDULERS
                    )
                    values = performance.get_perf_options("Model Defaults", checkpoint)
                    self.assertEqual({key: values[key] for key in explicit}, explicit)

    def test_qwen_image21_defaults_follow_the_verified_behavioral_variant(self):
        import hashlib
        from ruinedfooocus.features.models.checkpoint_tools import verify_checkpoint_source
        from ruinedfooocus.features.models.model_recommendations import (
            QWEN_IMAGE21_TURBO_SIGMAS,
            qwen_image21_sigmas,
        )

        sampler_module = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=SAMPLERS, SCHEDULERS=SCHEDULERS)
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(sys.modules, {"comfy.samplers": sampler_module}),
        ):
            root = Path(folder)
            model = root / "renamed.safetensors"
            model.write_bytes(b"published artifact fixture")
            row = {
                "id": "existing:checkpoints/qwen_image_2.1_int8_convrot.safetensors",
                "catalogue_key": "checkpoints/qwen_image_2.1_int8_convrot.safetensors",
                "role": "image_diffusion_checkpoint",
                "native_family": "QwenImage21",
                "behavioral_variant": "base",
                "quantization": {"label": "INT8 ConvRot"},
                "artifact": {
                    "bytes": model.stat().st_size,
                    "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                },
            }
            performance = self._performance(root)
            store = performance.model_recommendations
            with (
                patch(
                    "ruinedfooocus.features.models.model_registry.load_registry",
                    return_value=SimpleNamespace(entries=lambda: [row]),
                ),
                patch.object(store, "refresh") as refresh,
            ):
                verify_checkpoint_source(model, root, {"generationConfig": {"steps": 8}})
                values = performance.get_perf_options("Model Defaults", str(model), "QwenImage21")
                self.assertEqual(
                    tuple(
                        values[key]
                        for key in ("custom_steps", "cfg", "sampler_name", "scheduler", "clip_skip")
                    ),
                    (25, 1, "euler", "simple", 1),
                )
                self.assertNotIn("sampling_shift", values)
                self.assertNotIn("width", values)
                details = store.details(str(model), SAMPLERS, SCHEDULERS)
                self.assertIn("ComfyUI Qwen Image 2.1 base workflow", details["source"])
                self.assertIn("comfy.org/hub/workflows", details["source_url"])
                with patch.dict(row, {"quantization": {"label": "Q4_K_M"}}):
                    base = performance.get_perf_options("Model Defaults", str(model), "QwenImage21")
                    self.assertEqual(base["custom_steps"], 25)
                    self.assertNotIn("_model_default_sigmas", base)
                with patch.dict(row, {"behavioral_variant": "turbo"}):
                    request = dict(
                        performance_selection="Model Defaults", base_model_name=str(model)
                    )
                    turbo = performance.apply(request, "QwenImage21")
                    self.assertEqual(turbo["custom_steps"], 8)
                    self.assertEqual(turbo["_model_default_sigmas"], QWEN_IMAGE21_TURBO_SIGMAS)
                    self.assertIn(
                        "fixed 8-step schedule",
                        store.details(str(model), SAMPLERS, SCHEDULERS)["source"],
                    )
                    own = performance.apply(
                        turbo | {"performance_selection": "Own settings"}, "QwenImage21"
                    )
                    self.assertNotIn("_model_default_sigmas", own)
                    runtime = dict(turbo, steps=8)
                    self.assertEqual(qwen_image21_sigmas(runtime), QWEN_IMAGE21_TURBO_SIGMAS)
                    self.assertIsNone(qwen_image21_sigmas(runtime, "standard"))
                    self.assertEqual(
                        qwen_image21_sigmas(runtime, "turbo_8"), QWEN_IMAGE21_TURBO_SIGMAS
                    )
                    with self.assertRaisesRegex(ValueError, "requires 8 steps"):
                        qwen_image21_sigmas(dict(runtime, steps=12), "turbo_8")
                    store.save_override(str(model), turbo, SAMPLERS, SCHEDULERS)
                    self.assertEqual(
                        performance.apply(request, "QwenImage21")["_model_default_sigmas"],
                        QWEN_IMAGE21_TURBO_SIGMAS,
                    )
                    store.save_override(
                        str(model), turbo | {"custom_steps": 12}, SAMPLERS, SCHEDULERS
                    )
                    self.assertNotIn(
                        "_model_default_sigmas", performance.apply(request, "QwenImage21")
                    )
                    store.reset_override(str(model))
                with patch.dict(row, {"behavioral_variant": "unqualified"}):
                    with self.assertRaisesRegex(ValueError, "Choose Own settings"):
                        performance.get_perf_options("Model Defaults", str(model), "QwenImage21")
                model.write_bytes(b"replacement checkpoint fixture")
                with self.assertRaisesRegex(ValueError, "Choose Own settings"):
                    performance.get_perf_options("Model Defaults", str(model), "QwenImage21")
                refresh.assert_not_called()

    def test_qwen_image21_named_artifact_and_generic_cache_cannot_authorize_defaults(self):
        from ruinedfooocus.features.models.model_identity import file_signature

        sampler_module = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=SAMPLERS, SCHEDULERS=SCHEDULERS)
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(sys.modules, {"comfy.samplers": sampler_module}),
        ):
            root = Path(folder)
            model = root / "qwen_image_2.1_int8_convrot.safetensors"
            model.write_bytes(b"unknown local payload")
            performance = self._performance(root)
            store = performance.model_recommendations
            (root / (model.stem + ".json")).write_text(
                json.dumps(
                    {
                        "baseModel": "QwenImage21",
                        "variant": "base",
                        "_rf_file_signature": file_signature(model),
                        "generationConfig": {
                            "steps": 25,
                            "cfgScale": 1,
                            "sampler": "euler",
                            "scheduler": "simple",
                        },
                    }
                )
            )
            store.path.write_text(
                json.dumps(
                    {
                        f"{model}:local": {
                            "recommended": {
                                "custom_steps": 25,
                                "cfg": 1,
                                "sampler_name": "euler",
                                "scheduler": "simple",
                            },
                            "source": "unqualified cached source",
                        }
                    }
                )
            )
            self.assertEqual(store.details(str(model), SAMPLERS, SCHEDULERS)["recommended"], {})
            with self.assertRaisesRegex(ValueError, "Choose Own settings"):
                performance.get_perf_options("Model Defaults", str(model), "QwenImage21")

    def _flux3_checkpoint(self, root, name, distilled):
        from ruinedfooocus.features.models.model_identity import file_signature

        entries = {
            "content_mode_blocks.image.0.attn_out.weight": {"shape": [3072, 3072]},
            "single_stream_modulations.txt.lin.weight": {"shape": [9216, 3072]},
        }
        if distilled:
            entries["ref_time_in.in_layer.weight"] = {"shape": [3072, 256]}
        model = root / name
        raw = json.dumps(entries).encode()
        model.write_bytes(len(raw).to_bytes(8, "little") + raw)
        metadata = {"baseModel": "Flux.3 Klein", "_rf_file_signature": file_signature(model)}
        (root / (model.stem + ".json")).write_text(json.dumps(metadata))
        return model, metadata

    def test_flux3_native_templates_follow_header_variant_not_filename(self):
        from test_native_variant_defaults import native_backend

        registry = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=SAMPLERS + ["er_sde"], SCHEDULERS=SCHEDULERS)
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            native_backend(),
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.state": SimpleNamespace(
                        models=SimpleNamespace(
                            get_file=lambda category, name: Path(folder) / name,
                        )
                    )
                },
            ),
        ):
            root = Path(folder)
            performance = self._performance(root)
            for name, distilled, expected in (
                ("misleading_base.safetensors", True, (15, 1)),
                ("misleading_base.SFT", True, (15, 1)),
            ):
                model, _ = self._flux3_checkpoint(root, name, distilled)
                with (
                    self.subTest(distilled=distilled),
                    patch.dict(sys.modules, {"comfy.samplers": registry}),
                ):
                    values = performance.get_perf_options("Model Defaults", model.name, "Flux3")
                self.assertEqual(
                    (
                        values["custom_steps"],
                        values["cfg"],
                        values["sampler_name"],
                        values["scheduler"],
                    ),
                    (*expected, "er_sde", "simple"),
                )
                with patch.dict(sys.modules, {"comfy.samplers": registry}):
                    request = performance.apply(
                        dict(
                            performance_selection="Model Defaults",
                            base_model_name=model.name,
                            custom_steps=99,
                            seed=0,
                            inpaint_toggle=False,
                        ),
                        "Flux3",
                    )
                self.assertEqual((request["custom_steps"], request["cfg"]), expected)
                self.assertEqual(request["seed"], 0)
                self.assertFalse(request["inpaint_toggle"])
            model, _ = self._flux3_checkpoint(root, "misleading_distilled.safetensors", False)
            with patch.dict(sys.modules, {"comfy.samplers": registry}):
                with self.assertRaisesRegex(ValueError, "signature-matched local header"):
                    performance.get_perf_options("Model Defaults", model.name, "Flux3")

    def test_flux3_published_and_explicit_saved_settings_still_take_precedence(self):
        from test_native_variant_defaults import native_backend

        registry = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=SAMPLERS + ["er_sde"], SCHEDULERS=SCHEDULERS)
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            native_backend(),
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.app.state": SimpleNamespace(
                        models=SimpleNamespace(
                            get_file=lambda category, name: Path(folder) / name,
                        )
                    )
                },
            ),
        ):
            root = Path(folder)
            model, metadata = self._flux3_checkpoint(root, "local.safetensors", True)
            metadata["generationConfig"] = {"steps": 12, "cfgScale": 1.5}
            (root / "local.json").write_text(json.dumps(metadata))
            performance = self._performance(root)
            with patch.dict(sys.modules, {"comfy.samplers": registry}):
                values = performance.get_perf_options("Model Defaults", model.name)
                self.assertEqual((values["custom_steps"], values["cfg"]), (12, 1.5))
                explicit = dict(
                    custom_steps=10, cfg=2, sampler_name="euler", scheduler="normal", clip_skip=2
                )
                performance.model_recommendations.save_override(
                    model.name, explicit, SAMPLERS, SCHEDULERS
                )
                values = performance.get_perf_options("Model Defaults", model.name)
                self.assertEqual({key: values[key] for key in explicit}, explicit)

    def test_flux3_ambiguous_or_replaced_file_refuses_auto_but_keeps_explicit_choices(self):
        from test_native_variant_defaults import native_backend

        registry = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=SAMPLERS + ["er_sde"], SCHEDULERS=SCHEDULERS)
        )
        with tempfile.TemporaryDirectory() as folder, native_backend():
            root = Path(folder)
            performance = self._performance(root)
            model, _ = self._flux3_checkpoint(root, "local.safetensors", True)
            model.write_bytes(model.read_bytes() + b"changed")
            with patch.dict(sys.modules, {"comfy.samplers": registry}):
                with self.assertRaisesRegex(ValueError, "qualified Model Defaults recipe"):
                    performance.get_perf_options("Model Defaults", model.name)
                # Own settings remains a deliberate caller-provided recipe.
                self.assertIsInstance(
                    performance.get_perf_options("Own settings", model.name), dict
                )
                explicit = dict(
                    custom_steps=9, cfg=1, sampler_name="euler", scheduler="simple", clip_skip=1
                )
                performance.model_recommendations.save_override(
                    model.name, explicit, SAMPLERS, SCHEDULERS
                )
                values = performance.get_perf_options("Model Defaults", model.name)
                self.assertEqual({key: values[key] for key in explicit}, explicit)
                performance.model_recommendations.reset_override(model.name)
                (root / "local.json").write_text(json.dumps({"baseModel": "Flux.3 Klein"}))
                with self.assertRaisesRegex(ValueError, "signature-matched local header"):
                    performance.get_perf_options("Model Defaults", model.name)

    def test_hidream_sdk_guidance_is_not_native_cfg_zero(self):
        for value, expected in ((0, 1), (0.5, 1), (1, 1), (5, 5)):
            metadata = dict(baseModel="HiDreamO1", description=f"guidance_scale = {value},")
            self.assertEqual(extract(metadata, SAMPLERS, SCHEDULERS)["cfg"], expected)
            metadata["generationConfig"] = {"guidance_scale": value}
            self.assertEqual(extract(metadata, SAMPLERS, SCHEDULERS)["cfg"], expected)
        # Explicit native CFG and other families keep their documented values.
        self.assertEqual(
            extract(
                dict(baseModel="HiDreamO1", description="CFG: 0\nguidance_scale = 5,"),
                SAMPLERS,
                SCHEDULERS,
            )["cfg"],
            0,
        )
        self.assertEqual(
            extract(
                dict(baseModel="HiDreamO1", generationConfig={"cfgScale": 0}), SAMPLERS, SCHEDULERS
            )["cfg"],
            0,
        )
        self.assertEqual(
            extract(
                dict(baseModel="Krea2", description="guidance_scale = 0,"), SAMPLERS, SCHEDULERS
            )["cfg"],
            0,
        )

    def test_hidream_hf_refresh_translates_sdk_guidance_and_preserves_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            metadata = dict(
                baseModel="HiDreamO1", hf_repo_id="HiDream-ai/example", _rf_local_sha256="a" * 64
            )
            (root / "o1_dev.json").write_text(json.dumps(metadata))
            store = ModelRecommendations(root, root / "settings.json")
            store.path.write_text(
                json.dumps(
                    {
                        store.identity("o1_dev.safetensors"): dict(
                            schema=4,
                            checked=time.time(),
                            status="Public version metadata refreshed.",
                            recommended=dict(cfg=0),
                            override=dict(cfg=2),
                        )
                    }
                )
            )
            remote = dict(
                description="guidance_scale = 0,\nnum_inference_steps = 28,",
                source="Hugging Face example",
                source_url="https://huggingface.co/HiDream-ai/example",
            )
            with patch(
                "ruinedfooocus.features.models.model_sources.huggingface_recommendations",
                return_value=remote,
            ) as fetch:
                store.refresh("o1_dev.safetensors", SAMPLERS, SCHEDULERS)
                fetch.assert_called_once()
            details = store.details("o1_dev.safetensors", SAMPLERS, SCHEDULERS)
            self.assertEqual(details["recommended"], dict(cfg=1, custom_steps=28))
            self.assertEqual(store.recipe("o1_dev.safetensors", SAMPLERS, SCHEDULERS)[0]["cfg"], 2)

    def test_moss_family_fills_missing_scheduler_without_replacing_checkpoint_recipe(self):
        from ruinedfooocus.features.models.performance import PerformanceSettings

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.json").write_text(
                json.dumps(
                    {
                        "baseModel": "MOSSImage",
                        "generationConfig": {"steps": 50, "cfgScale": 3, "sampler_name": "er_sde"},
                    }
                ),
                encoding="utf-8",
            )
            performance = PerformanceSettings.__new__(PerformanceSettings)
            performance.performance_options = performance_defaults()
            performance.model_recommendations = ModelRecommendations(root, root / "settings.json")
            # Registry-only dependency: this resolver must not need a GPU context.
            registry = SimpleNamespace(
                KSampler=SimpleNamespace(SAMPLERS=["er_sde"], SCHEDULERS=["simple", "karras"])
            )
            with patch.dict(sys.modules, {"comfy.samplers": registry}):
                values = performance.get_perf_options("Model Defaults", "model.safetensors")
            self.assertEqual(
                (
                    values["custom_steps"],
                    values["cfg"],
                    values["sampler_name"],
                    values["scheduler"],
                ),
                (50, 3, "er_sde", "simple"),
            )

    def _performance(self, root):
        from ruinedfooocus.features.models.performance import PerformanceSettings

        performance = PerformanceSettings.__new__(PerformanceSettings)
        performance.legacy_options = {}
        performance.performance_options = performance_defaults()
        performance.model_recommendations = ModelRecommendations(root, root / "settings.json")
        return performance

    def test_distilled_families_use_native_template_fallbacks(self):
        registry = SimpleNamespace(
            KSampler=SimpleNamespace(
                SAMPLERS=["euler", "dpmpp_2m_sde_gpu"], SCHEDULERS=["simple", "karras"]
            )
        )
        cases = [
            ("Krea 2", "desire_krea2.safetensors", (8, 1)),
            ("Krea2", "desire_krea2_backend.safetensors", (8, 1)),
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            performance = self._performance(root)
            for family, checkpoint, expected in cases:
                with self.subTest(checkpoint=checkpoint):
                    (root / (Path(checkpoint).stem + ".json")).write_text(
                        json.dumps({"baseModel": family}), encoding="utf-8"
                    )
                    with patch.dict(sys.modules, {"comfy.samplers": registry}):
                        values = performance.get_perf_options("Model Defaults", checkpoint)
                    self.assertEqual(
                        (
                            values["custom_steps"],
                            values["cfg"],
                            values["sampler_name"],
                            values["scheduler"],
                        ),
                        (*expected, "euler", "simple"),
                    )

    def test_krea_backend_family_keeps_provider_and_saved_overrides(self):
        registry = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=SAMPLERS, SCHEDULERS=SCHEDULERS)
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "krea.json").write_text(
                json.dumps(
                    {"baseModel": "Krea 2", "generationConfig": {"steps": 12, "cfgScale": 1.5}}
                ),
                encoding="utf-8",
            )
            performance = self._performance(root)
            with patch.dict(sys.modules, {"comfy.samplers": registry}):
                values = performance.get_perf_options("Model Defaults", "krea.safetensors", "Krea2")
                self.assertEqual((values["custom_steps"], values["cfg"]), (12, 1.5))
                performance.model_recommendations.save_override(
                    "krea.safetensors",
                    dict(
                        custom_steps=10,
                        cfg=2,
                        sampler_name="euler",
                        scheduler="normal",
                        clip_skip=1,
                    ),
                    SAMPLERS,
                    SCHEDULERS,
                )
                values = performance.get_perf_options("Model Defaults", "krea.safetensors", "Krea2")
                self.assertEqual(
                    (values["custom_steps"], values["cfg"], values["scheduler"]), (10, 2, "normal")
                )

    def test_qwen_edit_published_and_user_settings_override_native_fallback(self):
        registry = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=SAMPLERS, SCHEDULERS=SCHEDULERS)
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "qwen_image_edit.json").write_text(
                json.dumps(
                    {
                        "baseModel": "QwenImage",
                        "generationConfig": {
                            "steps": 32,
                            "cfgScale": 3.2,
                            "sampler": "euler",
                            "scheduler": "simple",
                        },
                    }
                )
            )
            performance = self._performance(root)
            with patch.dict(sys.modules, {"comfy.samplers": registry}):
                values = performance.get_perf_options(
                    "Model Defaults", "qwen_image_edit.safetensors"
                )
                self.assertEqual((values["custom_steps"], values["cfg"]), (32, 3.2))
                performance.model_recommendations.save_override(
                    "qwen_image_edit.safetensors",
                    dict(
                        custom_steps=50,
                        cfg=4,
                        sampler_name="euler",
                        scheduler="normal",
                        clip_skip=1,
                    ),
                    SAMPLERS,
                    SCHEDULERS,
                )
                values = performance.get_perf_options(
                    "Model Defaults", "qwen_image_edit.safetensors"
                )
                self.assertEqual(
                    (values["custom_steps"], values["cfg"], values["scheduler"]), (50, 4, "normal")
                )

    def test_generation_uses_cache_while_explicit_refresh_discovers_published_recipe(self):
        registry = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=SAMPLERS, SCHEDULERS=SCHEDULERS)
        )
        version = {"id": 7, "modelId": 3, "baseModel": "Krea 2", "images": examples(4)}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.json").write_text(
                json.dumps({"id": 7, "modelId": 3, "baseModel": "Krea 2"}), encoding="utf-8"
            )
            performance = self._performance(root)
            with (
                patch.dict(sys.modules, {"comfy.samplers": registry}),
                patch(
                    "ruinedfooocus.features.models.model_sources.civitai_metadata",
                    side_effect=lambda path: version if path == "model-versions/7" else None,
                ) as fetch,
            ):
                first = performance.get_perf_options("Model Defaults", "model.safetensors")
                fetch.assert_not_called()
                performance.model_recommendations.refresh("model.safetensors", SAMPLERS, SCHEDULERS)
                calls = fetch.call_count
                refreshed = performance.get_perf_options("Model Defaults", "model.safetensors")
                performance.get_perf_options("Model Defaults", "model.safetensors")
            self.assertEqual(
                (first["custom_steps"], first["cfg"], first["sampler_name"]), (8, 1, "euler")
            )
            self.assertGreater(calls, 0)
            self.assertEqual(refreshed["custom_steps"], 8)
            self.assertEqual(fetch.call_count, calls)  # Submission never refreshes.

    def test_majority_and_conflicting_fields(self):
        images = examples(10)
        for image in images[-3:]:
            image["meta"]["scheduler"] = "simple"
        self.assertEqual(
            example_settings({"images": images}, SAMPLERS, SCHEDULERS),
            dict(custom_steps=8, cfg=1, sampler_name="euler", scheduler="normal"),
        )
        for image in images[5:]:
            image["meta"]["scheduler"] = "simple"
        self.assertNotIn("scheduler", example_settings({"images": images}, SAMPLERS, SCHEDULERS))

    def test_sparse_duplicate_and_invalid_examples(self):
        for images in (
            examples(2),
            [examples(1)[0]] * 10,
            [
                {
                    "meta": {
                        "steps": True,
                        "cfgScale": float("nan"),
                        "sampler": "unknown",
                        "scheduler": [],
                    }
                }
            ]
            * 5,
        ):
            self.assertEqual(example_settings({"images": images}, SAMPLERS, SCHEDULERS), {})
        self.assertEqual(example_settings({"images": "invalid"}, SAMPLERS, SCHEDULERS), {})

    def test_lora_gallery_recipes_do_not_set_bare_checkpoint_defaults(self):
        accelerated = examples(6)
        for index, image in enumerate(accelerated):
            image["id"] += 100
            image["meta"]["steps"] = 4
            image["meta"]["additionalResources" if index % 2 else "resources"] = [
                {"type": "LoRA", "name": "Lightning.safetensors", "strength": 1}
            ]
        clean = examples(3)
        for image in clean:
            image["meta"].update(steps=30, cfgScale=4)
        self.assertEqual(
            example_settings({"images": accelerated + clean}, SAMPLERS, SCHEDULERS),
            dict(custom_steps=30, cfg=4, sampler_name="euler", scheduler="normal"),
        )
        self.assertEqual(example_settings({"images": accelerated}, SAMPLERS, SCHEDULERS), {})
        clean[0]["meta"]["resources"] = [None, 1, {"type": "checkpoint"}]
        self.assertEqual(
            example_settings({"images": clean}, SAMPLERS, SCHEDULERS)["custom_steps"], 30
        )

    def test_old_accelerated_consensus_is_refreshed_without_losing_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.json").write_text(json.dumps({"id": 123}), encoding="utf-8")
            store = ModelRecommendations(root, root / "settings.json")
            store.path.write_text(
                json.dumps(
                    {
                        "model.safetensors:123": dict(
                            schema=3,
                            checked=time.time(),
                            status="Public version metadata refreshed.",
                            recommended=dict(custom_steps=4, cfg=1),
                            override=dict(clip_skip=2),
                        )
                    }
                )
            )
            images = examples(4)
            for image in images:
                image["meta"]["additionalResources"] = [{"type": "lora", "name": "Lightning"}]
            remote = dict(id=123, images=images)
            with patch(
                "ruinedfooocus.features.models.model_sources.civitai_metadata", return_value=remote
            ) as fetch:
                store.refresh("model.safetensors", SAMPLERS, SCHEDULERS)
                fetch.assert_called_once()
            details = store.details("model.safetensors", SAMPLERS, SCHEDULERS)
            self.assertEqual(details["recommended"], {})
            self.assertEqual(details["override"], {"clip_skip": 2})

    def test_refresh_migrates_cache_preserves_override_and_card_priority(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.json").write_text(json.dumps({"id": 123}), encoding="utf-8")
            store = ModelRecommendations(root, root / "settings.json")
            store.save_override(
                "model.safetensors",
                dict(custom_steps=12, cfg=2, sampler_name="euler", scheduler="simple", clip_skip=1),
                SAMPLERS,
                SCHEDULERS,
            )
            data = store.load()
            data["model.safetensors:123"].update(
                schema=2,
                checked=time.time(),
                status="Public version metadata refreshed.",
                recommended={},
            )
            store.path.write_text(json.dumps(data), encoding="utf-8")
            remote = dict(
                id=123,
                modelId=456,
                baseModel="Example",
                generationConfig={"steps": 10},
                images=examples(),
            )
            card = dict(
                id=456, modelVersions=[remote], generationConfig={"cfgScale": 7, "steps": 20}
            )
            with patch(
                "ruinedfooocus.features.models.model_sources.civitai_metadata",
                side_effect=[remote, card],
            ) as fetch:
                store.refresh("model.safetensors", SAMPLERS, SCHEDULERS)
                self.assertEqual(fetch.call_count, 2)
            details = store.details("model.safetensors", SAMPLERS, SCHEDULERS)
            self.assertEqual(details["recommended"]["custom_steps"], 10)
            self.assertEqual(details["recommended"]["cfg"], 7)
            self.assertEqual(details["override"]["cfg"], 2)
            self.assertIn("example consensus", details["source"])
            store.reset_override("model.safetensors")
            self.assertEqual(store.recipe("model.safetensors", SAMPLERS, SCHEDULERS)[0]["cfg"], 7)

    def test_display_sampler_alias(self):
        images = examples()
        for image in images:
            image["meta"]["sampler"] = "Euler a"
        self.assertEqual(
            example_settings({"images": images}, SAMPLERS, SCHEDULERS)["sampler_name"],
            "euler_ancestral",
        )

    def test_wrong_version_and_offline_do_not_infer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.json").write_text('{"id":123}', encoding="utf-8")
            store = ModelRecommendations(root, root / "settings.json")
            with patch(
                "ruinedfooocus.features.models.model_sources.civitai_metadata",
                return_value=dict(id=999, images=examples()),
            ):
                store.refresh("model.safetensors", SAMPLERS, SCHEDULERS)
            self.assertEqual(
                store.details("model.safetensors", SAMPLERS, SCHEDULERS)["recommended"], {}
            )
            store.offline = True
            with patch("ruinedfooocus.features.models.model_sources.civitai_metadata") as fetch:
                store.refresh("model.safetensors", SAMPLERS, SCHEDULERS, force=True)
                fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
