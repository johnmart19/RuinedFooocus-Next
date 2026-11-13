from ruinedfooocus.core.paths import asset_path, resource_path

"""Native LCM options, automatic O1 recipes and API geometry; CPU only."""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from ruinedfooocus.features.image_editing.image_guidance import (
    resolve_guidance,
    restore_guidance_metadata,
    validate_guidance,
)
from ruinedfooocus.features.generation.image_sampling import hidream_o1_dev, lcm_options, sample_lcm
from ruinedfooocus.features.models.resource_defaults import performance_defaults

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))


def native_method(path, owner, method, namespace):
    tree = ast.parse((NATIVE / path).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), path, "exec"), namespace)
    return lambda **kwargs: namespace[method](None, **kwargs)


class NativeSamplingTests(unittest.TestCase):
    def test_full_and_dev_automatic_recipes_keep_published_and_user_overrides(self):
        from ruinedfooocus.features.models.model_recommendations import ModelRecommendations
        from ruinedfooocus.features.models.performance import PerformanceSettings
        from ruinedfooocus.features.models.model_identity import file_signature
        from test_native_variant_defaults import native_backend, write_header

        registry = SimpleNamespace(
            KSampler=SimpleNamespace(
                SAMPLERS=["lcm", "dpmpp_2m_sde_gpu", "euler"], SCHEDULERS=["normal", "simple"]
            )
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder, native_backend():
            store = ModelRecommendations(folder, Path(folder) / "overrides.json", offline=True)
            perf = PerformanceSettings.__new__(PerformanceSettings)
            perf.model_recommendations = store
            perf.performance_options = performance_defaults()
            with patch.dict(
                sys.modules,
                {
                    "comfy.samplers": registry,
                    "ruinedfooocus.app.state": SimpleNamespace(performance_settings=perf),
                },
            ):
                for name, expected in (
                    ("hidream_o1_image_bf16.safetensors", (40, 5, "dpmpp_2m_sde_gpu")),
                    ("folder/HiDream_O1_Image_DEV_fp8.safetensors", (28, 1, "lcm")),
                    ("folder\\hidream-o1-device.safetensors", (40, 5, "dpmpp_2m_sde_gpu")),
                ):
                    name = name.replace("\\", "/").rsplit("/", 1)[-1]
                    model = write_header(
                        Path(folder) / name,
                        {"unknown.weight": [1]},
                        metadata={"variant": "dev" if expected[0] == 28 else "full"},
                    )
                    (Path(folder) / (model.stem + ".json")).write_text(
                        json.dumps(
                            dict(baseModel="HiDreamO1", _rf_file_signature=file_signature(model))
                        )
                    )
                    values = perf.get_perf_options("Model Defaults", name, "HiDreamO1")
                    self.assertEqual(
                        (values["custom_steps"], values["cfg"], values["sampler_name"]), expected
                    )
                    self.assertEqual(values["scheduler"], "normal")
                    self.assertEqual(hidream_o1_dev(name), expected[0] == 28)
                name = "hidream_o1_image_dev.safetensors"
                store.save_override(
                    name,
                    dict(
                        custom_steps=12,
                        cfg=2,
                        sampler_name="euler",
                        scheduler="simple",
                        clip_skip=1,
                    ),
                    registry.KSampler.SAMPLERS,
                    registry.KSampler.SCHEDULERS,
                )
                self.assertEqual(
                    perf.get_perf_options("Model Defaults", name, "HiDreamO1")["custom_steps"], 12
                )
        self.assertFalse(hidream_o1_dev("C:\\models\\DEV.safetensors"))
        self.assertFalse(hidream_o1_dev("device.safetensors"))

    @patch(
        "ruinedfooocus.features.models.performance.PerformanceSettings._declared_variant",
        return_value="dev",
    )
    def test_native_lcm_clipping_and_denoise_are_used_without_reimplementing_sampling(
        self, _variant
    ):
        tree = ast.parse((NATIVE / "comfy/k_diffusion/sampling.py").read_text(encoding="utf-8"))
        function = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sample_lcm"
        )
        function.decorator_list = []
        namespace = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            torch=torch,
            trange=lambda count, **kwargs: range(count),
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), "native_lcm", "exec"), namespace)
        native_lcm = namespace["sample_lcm"]
        calls = []
        calculate = Mock(
            side_effect=lambda model, schedule, count: torch.linspace(1, 0, count + 1, device="cpu")
        )
        native = SimpleNamespace(
            samplers=SimpleNamespace(
                calculate_sigmas=calculate,
                ksampler=lambda name, options: SimpleNamespace(
                    function=native_lcm, options=options
                ),
            )
        )
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            comfy=native,
            torch=torch,
            io=SimpleNamespace(NodeOutput=lambda *args: args),
        )
        scheduler = native_method(
            "comfy_extras/nodes_custom_sampler.py", "BasicScheduler", "execute", dict(scope)
        )
        sampler = native_method(
            "comfy_extras/nodes_advanced_samplers.py", "SamplerLCM", "execute", dict(scope)
        )
        sampling = SimpleNamespace(noise_scaling=lambda sigma, noise, x: x + sigma * noise)
        patcher = SimpleNamespace(get_model_object=lambda name: sampling)

        class Model:
            inner_model = SimpleNamespace(model_patcher=patcher)

            def get_model_object(self, name):
                return sampling

            def __call__(self, x, sigma, **kwargs):
                return x * 0.5

        model = Model()
        noise = torch.zeros(1, 1, 4, 4, device="cpu")
        noise[0, 0, 0, 0] = 100

        def custom(**kwargs):
            calls.append(kwargs)
            return kwargs["sampler"].function(
                model,
                kwargs["noise"],
                kwargs["sigmas"],
                noise_sampler=lambda *args: noise,
                disable=True,
                **kwargs["sampler"].options,
            )

        modules = {
            "comfy.sample": SimpleNamespace(sample_custom=custom),
            "comfy_extras.nodes_custom_sampler": SimpleNamespace(
                BasicScheduler=SimpleNamespace(execute=scheduler)
            ),
            "comfy_extras.nodes_advanced_samplers": SimpleNamespace(
                SamplerLCM=SimpleNamespace(execute=sampler)
            ),
        }
        inputs = dict(
            model=model,
            noise=noise,
            steps=3,
            cfg=1,
            scheduler="normal",
            positive=["positive"],
            negative=["negative"],
            latent_image=torch.zeros_like(noise),
            denoise=0.25,
            seed=42,
            noise_mask="mask",
            callback="callback",
        )
        with patch.dict(sys.modules, modules):
            clipped = sample_lcm(
                **inputs, noise_options=lcm_options({}, "HiDreamO1", "o1_dev.safetensors")
            )
            unclipped = sample_lcm(**inputs, noise_options=lcm_options({}, "SDXL"))
        self.assertFalse(torch.equal(clipped, unclipped))
        self.assertTrue(torch.isfinite(clipped).all())
        calculate.assert_called_with(sampling, "normal", 12)
        self.assertEqual(len(calls[0]["sigmas"]), 4)
        for key in (
            "positive",
            "negative",
            "latent_image",
            "noise_mask",
            "callback",
            "seed",
            "cfg",
        ):
            if isinstance(inputs[key], torch.Tensor):
                self.assertIs(calls[0][key], inputs[key])
            else:
                self.assertEqual(calls[0][key], inputs[key])
        self.assertEqual(
            calls[0]["sampler"].options, dict(s_noise=1.0, s_noise_end=1.0, noise_clip_std=2.5)
        )

    def test_options_validate_and_effective_recipe_replays(self):
        from ruinedfooocus.integrations.openai_api import ImageRequest
        from pydantic import ValidationError

        for key, invalid in (("lcm_noise", 65), ("lcm_noise_end", -1), ("lcm_noise_clip", 11)):
            with self.assertRaises(ValueError):
                validate_guidance({key: invalid})
            with self.assertRaises(ValidationError):
                ImageRequest(**{key: invalid})
        values = resolve_guidance(
            {"lcm_noise": "", "lcm_noise_end": 0, "lcm_noise_clip": 0}, {}, "HiDreamO1", "o1_dev"
        )
        self.assertEqual(
            lcm_options(values, "HiDreamO1", "o1_dev"),
            dict(s_noise=1.0, s_noise_end=0.0, noise_clip_std=0.0),
        )
        self.assertEqual(
            restore_guidance_metadata(
                {"image_guidance": {"lcm_noise_clip": None, "effective_lcm_noise_clip": 2.5}}
            ),
            {"lcm_noise_clip": 2.5},
        )
        self.assertEqual(restore_guidance_metadata({"effective_lcm_noise_clip": float("nan")}), {})

    @patch(
        "ruinedfooocus.features.models.performance.PerformanceSettings._declared_variant",
        return_value="dev",
    )
    def test_pipeline_routes_lcm_options_and_keeps_other_native_samplers(self, _variant):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        process = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process"
        )
        branch = next(
            n
            for n in process.body
            if isinstance(n, ast.If) and ast.unparse(n.test) == "sampler_name == 'lcm'"
        )
        dispatch = next(
            n
            for n in ast.walk(process)
            if isinstance(n, ast.If)
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "sample_native"
                for call in ast.walk(n)
            )
        )
        normal = Mock(return_value="normal-result")
        lcm = Mock(return_value="lcm-result")
        for sampler, expected in (("lcm", "lcm-result"), ("euler", "normal-result")):
            data = dict(steps=28, image_guidance={})
            scope = dict(
                asset_path=asset_path,
                resource_path=resource_path,
                sampler_name=sampler,
                guidance={},
                architecture="HiDreamO1",
                native_plan=None,
                checkpoint="o1_dev",
                sampling_model=object(),
                noise="noise",
                gen_data=data,
                cfg=1,
                scheduler="normal",
                positive_cond=["positive"],
                negative_conditioning=["negative"],
                latent_image="latent",
                denoise=0.25,
                noise_mask="mask",
                callback_function="callback",
                fixed_sigmas=None,
                seed=42,
                comfy=SimpleNamespace(
                    sample=SimpleNamespace(sample=normal),
                    utils=SimpleNamespace(PROGRESS_BAR_ENABLED=True),
                ),
            )
            with patch("ruinedfooocus.features.generation.image_sampling.sample_lcm", lcm):
                exec(
                    compile(
                        ast.Module(body=[branch, dispatch], type_ignores=[]),
                        "sampling_route",
                        "exec",
                    ),
                    scope,
                )
            self.assertEqual(scope["samples"], expected)
            kwargs = (lcm if sampler == "lcm" else normal).call_args.kwargs
            for key, expected_value in (
                ("positive", ["positive"]),
                ("negative", ["negative"]),
                ("seed", 42),
                ("denoise", 0.25),
                ("noise_mask", "mask"),
                ("callback", "callback"),
            ):
                self.assertEqual(kwargs[key], expected_value)
            self.assertIs(kwargs["model"], scope["sampling_model"])
            if sampler == "lcm":
                self.assertEqual(data["image_guidance"]["effective_lcm_noise_clip"], 2.5)
            else:
                self.assertEqual(data["image_guidance"], {})
                self.assertEqual(kwargs["sampler_name"], "euler")
        self.assertEqual((lcm.call_count, normal.call_count), (1, 1))

    @patch(
        "ruinedfooocus.features.models.performance.PerformanceSettings._declared_variant",
        return_value="dev",
    )
    def test_native_recommended_sizes_fit_api_budget_and_keep_model_specific_side_limit(
        self, _variant
    ):
        import ruinedfooocus.app.api_runtime as api_runtime
        from ruinedfooocus.features.models.resolutions import ResolutionSettings

        sizes = ResolutionSettings.__new__(ResolutionSettings).recommended_sizes("HiDreamO1")
        self.assertEqual(len(sizes), 11)
        self.assertIn((1312, 3104), sizes)
        source = (NATIVE / "comfy_extras/nodes_hidream_o1.py").read_text(encoding="utf-8")
        for width, height in sizes:
            self.assertIn(f"{width}x{height}", source)
            self.assertEqual((width % 32, height % 32), (0, 0))
        resolution = ResolutionSettings.__new__(ResolutionSettings)
        resolution.base_ratios = {"Square": (1024, 1024), "Saved custom": (600, 400)}
        resolution.builtin_names = {"Square"}
        resolution.video_models = dict.fromkeys(resolution.base_ratios, [])
        resolution.image_models = dict.fromkeys(resolution.base_ratios, [])
        resolution.aspect_ratios = {
            "1024x1024 (Square)": (1024, 1024),
            "600x400 (Saved custom)": (600, 400),
        }
        self.assertEqual(
            resolution.selection_for_model("HiDreamO1", "1024x1024 (Square)"),
            "2048x2048 (HiDreamO1 recommended)",
        )
        self.assertEqual(
            resolution.selection_for_model("HiDreamO1", "600x400 (Saved custom)"),
            "600x400 (Saved custom)",
        )
        selected = resolution.selection_for_model("SDXL", "1024x1024 (Square)")
        self.assertEqual(resolution.get_aspect_ratios(selected), (1024, 1024))
        perf = SimpleNamespace(
            CUSTOM_PERFORMANCE="Own settings",
            choices_for_model=lambda *a: ["Model Defaults"],
            selection_for_model=lambda *a: "Model Defaults",
            get_perf_options=lambda *a: dict(
                custom_steps=28, cfg=1, sampler_name="lcm", scheduler="normal", clip_skip=1
            ),
        )
        shared = SimpleNamespace(
            settings=SimpleNamespace(default_settings={"performance": "Model Defaults"}),
            performance_settings=perf,
            resolution_settings=resolution,
            models=SimpleNamespace(offline=True),
            path_manager=object(),
        )
        registry = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=["lcm"], SCHEDULERS=["normal"])
        )
        with (
            patch.dict(
                sys.modules, {"ruinedfooocus.app.state": shared, "comfy.samplers": registry}
            ),
            patch.object(sys.modules["ruinedfooocus.app"], "state", shared, create=True),
            patch.object(api_runtime, "image_defaults", {}),
            patch.object(api_runtime, "submit", side_effect=lambda run: run),
            patch(
                "ruinedfooocus.features.models.component_profiles.snapshot_request",
                side_effect=lambda data, *a, **kw: data,
            ),
            patch(
                "ruinedfooocus.features.prompting.local_model.capture_prompt_preparation",
                side_effect=lambda data, *a, **kw: data,
            ),
            patch(
                "ruinedfooocus.features.generation.image_preparation.capture_image_assets",
                side_effect=lambda data, *a, **kw: data,
            ),
            patch(
                "ruinedfooocus.features.generation.sdcpp_pipeline.capture_image_request",
                side_effect=lambda data, *a, **kw: data,
            ),
        ):
            for width, height in sizes:
                operation = api_runtime.images(
                    dict(prompt="photo", size=f"{width}x{height}"),
                    dict(name="o1_dev.safetensors", family="HiDreamO1"),
                )
                data = dict(
                    zip(
                        operation.__code__.co_freevars,
                        (c.cell_contents for c in operation.__closure__),
                    )
                )["data"]
                self.assertEqual((data["custom_width"], data["custom_height"]), (width, height))
                self.assertFalse(data["hidream_o1_seam_smoothing"])
            for size, family in (
                ("2560x1440", "SDXL"),
                ("4096x2048", "HiDreamO1"),
                ("4104x64", "HiDreamO1"),
                ("2048x2056", "HiDreamO1"),
            ):
                with self.assertRaises(ValueError):
                    api_runtime.images(
                        dict(prompt="photo", size=size), dict(name="checkpoint", family=family)
                    )


if __name__ == "__main__":
    unittest.main()
