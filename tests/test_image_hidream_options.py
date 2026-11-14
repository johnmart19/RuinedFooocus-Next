from ruinedfooocus.core.paths import asset_path, resource_path

"""HiDream native patch callbacks and geometry; no model weights or GPU work."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch

from ruinedfooocus.features.image_editing.image_geometry import (
    prepare_reference,
    restore_reference_frame,
)
import ruinedfooocus.app
from ruinedfooocus.features.image_editing.image_guidance import (
    FIELD_SCHEMA,
    resolve_guidance,
    request_model,
    effective_guidance,
    restore_guidance_metadata,
)

ROOT = Path(__file__).resolve().parents[1]


class Sampling:
    def __init__(self, config):
        self.shift, self.multiplier, self.noise_scale = 3.0, 1.0, 8.0

    def set_parameters(self, *, shift, multiplier):
        self.shift, self.multiplier = shift, multiplier

    def set_noise_scale(self, value):
        self.noise_scale = value

    def percent_to_sigma(self, fraction):
        return 1 - fraction


class Patcher:
    def __init__(self):
        self.model = SimpleNamespace(model_config=object())
        self.sampling = Sampling(self.model.model_config)
        self.wrappers = {}

    def clone(self):
        other = Patcher()
        other.model = self.model
        other.sampling = self.sampling
        other.wrappers = dict(self.wrappers)
        return other

    def get_model_object(self, name):
        return self.sampling

    def add_object_patch(self, name, value):
        self.sampling = value

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers[key] = wrapper


def native_nodes():
    path = ROOT / "repositories/ComfyUI/comfy_extras/nodes_hidream_o1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "HiDreamO1PatchSeamSmoothing"
    )
    cls.bases = []
    cls.body = [
        n for n in cls.body if not isinstance(n, ast.FunctionDef) or n.name != "define_schema"
    ]
    namespace = dict(
        asset_path=asset_path,
        resource_path=resource_path,
        torch=torch,
        io=SimpleNamespace(NodeOutput=lambda *args: args),
        comfy=SimpleNamespace(
            patcher_extension=SimpleNamespace(
                WrappersMP=SimpleNamespace(DIFFUSION_MODEL="diffusion")
            )
        ),
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    smoothing = namespace[cls.name].execute
    advanced = ast.parse((path.parent / "nodes_model_advanced.py").read_text(encoding="utf-8"))
    noise_cls = next(
        n for n in advanced.body if isinstance(n, ast.ClassDef) and n.name == "ModelNoiseScale"
    )
    method = next(n for n in noise_cls.body if isinstance(n, ast.FunctionDef) and n.name == "patch")
    scope = {
        "asset_path": asset_path,
        "resource_path": resource_path,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), "native_noise_scale", "exec"), scope)
    return dict(seam=smoothing, noise_scale=lambda **kwargs: scope["patch"](None, **kwargs))


class HiDreamOptionsTests(unittest.TestCase):
    @patch(
        "ruinedfooocus.features.models.performance.PerformanceSettings._declared_variant",
        return_value="dev",
    )
    def test_dev_automatic_noise_and_seams_respect_explicit_overrides(self, _variant):
        from ruinedfooocus.features.image_editing.ui_guidance import preset_control_state

        native = native_nodes()
        base = Patcher()
        name = "hidream_o1_image_dev_fp8.safetensors"
        values = resolve_guidance({}, {}, "HiDreamO1", name)
        self.assertFalse(values["hidream_o1_seam_smoothing"])
        automatic = request_model(base, values, "HiDreamO1", native, checkpoint=name)
        self.assertEqual(automatic.sampling.noise_scale, 7.6)
        self.assertEqual(automatic.wrappers, {})
        self.assertEqual(base.sampling.noise_scale, 8)
        explicit = resolve_guidance(
            {"hidream_o1_noise_scale": 7.5, "hidream_o1_seam_smoothing": True},
            {},
            "HiDreamO1",
            name,
        )
        overridden = request_model(base, explicit, "HiDreamO1", native, checkpoint=name)
        self.assertEqual(overridden.sampling.noise_scale, 7.5)
        self.assertEqual(len(overridden.wrappers), 1)
        saved = resolve_guidance({}, {"hidream_o1_seam_smoothing": True}, "HiDreamO1", name)
        self.assertTrue(saved["hidream_o1_seam_smoothing"])
        current = {key: spec["default"] for key, spec in FIELD_SCHEMA.items()}
        current.update(lcm_noise_clip=4, hidream_o1_seam_smoothing=True)
        changed = preset_control_state(
            current, {}, {}, 7, "HiDreamO1", name, name, {"checkpoint": "base", "backup": None}
        )
        self.assertIsNone(changed["values"]["hidream_o1_seam_smoothing"])
        self.assertFalse(
            resolve_guidance(changed["values"], {}, "HiDreamO1", name)["hidream_o1_seam_smoothing"]
        )
        self.assertIsNone(changed["values"]["lcm_noise_clip"])
        self.assertIsNone(
            resolve_guidance({"hidream_o1_seam_smoothing": "auto"}, {}, "SDXL")[
                "hidream_o1_noise_scale"
            ]
        )
        from ruinedfooocus.core.settings_validation import validate_settings

        self.assertIsNone(
            validate_settings({"hidream_o1_seam_smoothing": "auto"})["hidream_o1_seam_smoothing"]
        )
        self.assertFalse(
            resolve_guidance({"hidream_o1_seam_smoothing": "off"}, {}, "HiDreamO1")[
                "hidream_o1_seam_smoothing"
            ]
        )

    def test_noise_auto_preserves_native_sampling_and_override_never_changes_cache(self):
        native = native_nodes()
        base = Patcher()
        options = resolve_guidance({"hidream_o1_seam_smoothing": False}, {}, "HiDream O1")
        auto = request_model(base, options, "HiDreamO1", native)
        self.assertIs(auto.sampling, base.sampling)
        explicit = request_model(
            base, dict(options, hidream_o1_noise_scale=7.5), "HiDreamO1", native
        )
        self.assertEqual(explicit.sampling.noise_scale, 7.5)
        self.assertEqual((explicit.sampling.shift, explicit.sampling.multiplier), (3.0, 1.0))
        self.assertEqual(base.sampling.noise_scale, 8)
        metadata = effective_guidance(
            dict(options, hidream_o1_noise_scale=None), explicit, "HiDreamO1"
        )
        self.assertEqual(metadata["effective_hidream_o1_noise_scale"], 7.5)
        replay = restore_guidance_metadata({"image_guidance": metadata}, family="HiDreamO1")
        self.assertEqual(replay["hidream_o1_noise_scale"], 7.5)
        self.assertIsNone(
            restore_guidance_metadata({"image_guidance": metadata}, family="MOSSImage")[
                "hidream_o1_noise_scale"
            ]
        )

    def test_all_native_seam_patterns_pass_counts_and_blends_keep_the_source(self):
        native = native_nodes()
        base = Patcher()
        image = torch.linspace(0, 1, 3 * 96 * 128, device="cpu").reshape(1, 3, 96, 128)
        for pattern in ("single_shift", "symmetric"):
            for passes, count in (("2", 2), ("4", 4), ("ramp_2_4", 4), ("ramp_2_4_8", 8)):
                for blend in ("average", "window", "median"):
                    with self.subTest(pattern=pattern, passes=passes, blend=blend):
                        values = resolve_guidance(
                            dict(
                                hidream_o1_seam_pattern=pattern,
                                hidream_o1_seam_passes=passes,
                                hidream_o1_seam_blend=blend,
                            ),
                            {},
                            "HiDreamO1",
                        )
                        model = request_model(base, values, "HiDreamO1", native)
                        self.assertEqual(len(model.wrappers), 1)
                        self.assertEqual(base.wrappers, {})
                        wrapper = next(iter(model.wrappers.values()))
                        executor = Mock(side_effect=lambda image, *args, **kwargs: image)
                        result = wrapper(executor, image, torch.tensor([0.01], device="cpu"))
                        self.assertTrue(torch.allclose(result, image, atol=1e-6))
                        self.assertEqual(executor.call_count, count + (pattern == "symmetric"))
                        executor.reset_mock()
                        self.assertIs(
                            wrapper(executor, image, torch.tensor([0.9], device="cpu")), image
                        )
                        self.assertEqual(executor.call_count, 1)
        disabled = request_model(
            base,
            resolve_guidance({"hidream_o1_seam_smoothing": False}, {}, "HiDreamO1"),
            "HiDreamO1",
            native,
        )
        self.assertEqual(disabled.wrappers, {})

    def test_hidream_canvas_handles_unaligned_reference_and_text_to_image_sizes(self):
        image = Image.new("RGB", (117, 239), (17, 67, 193))
        pixels, plan = prepare_reference(image, (1024, 1024), multiple=32)
        self.assertEqual(tuple(pixels.shape[1:3]), (256, 128))
        model = request_model(
            Patcher(),
            resolve_guidance({"hidream_o1_seam_blend": "window"}, {}, "HiDreamO1"),
            "HiDreamO1",
            native_nodes(),
        )
        result = next(iter(model.wrappers.values()))(
            lambda image, *args: image, pixels.movedim(-1, 1), torch.tensor([0.1], device="cpu")
        ).movedim(1, -1)
        restored = restore_reference_frame(result, plan)
        self.assertEqual(tuple(restored.shape), (1, 239, 117, 3))
        self.assertTrue(
            torch.allclose(
                restored,
                pixels[:, plan["top"] : plan["top"] + 239, plan["left"] : plan["left"] + 117],
            )
        )
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        process = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process"
        )
        branches = [
            n
            for n in ast.walk(process)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "input_image"
        ]
        self.assertEqual(len(branches), 1)
        branch = branches[0]
        data = {"width": 520, "height": 776}
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            input_image=None,
            architecture="HiDreamO1",
            gen_data=data,
        )
        exec(compile(ast.Module(body=[branch], type_ignores=[]), "native_canvas", "exec"), scope)
        self.assertEqual((data["width"], data["height"]), (544, 800))
        self.assertEqual(
            tuple(
                restore_reference_frame(
                    torch.zeros(1, 800, 544, 3, device="cpu"), scope["reference_frame"]
                ).shape
            ),
            (1, 776, 520, 3),
        )
        for architecture, size in (("HiDreamO1", (512, 768)), ("SDXL", (520, 776))):
            with self.subTest(architecture=architecture, size=size):
                unchanged = dict(width=size[0], height=size[1])
                scope.update(architecture=architecture, gen_data=unchanged, reference_frame=None)
                exec(
                    compile(ast.Module(body=[branch], type_ignores=[]), "native_canvas", "exec"),
                    scope,
                )
                self.assertEqual((unchanged["width"], unchanged["height"]), size)
                self.assertIsNone(scope["reference_frame"])

    def test_api_validation_family_guards_and_metadata_ranges(self):
        from ruinedfooocus.integrations.openai_api import ImageRequest
        from ruinedfooocus.features.image_editing.image_guidance import validate_guidance
        from pydantic import ValidationError

        for values in (
            {"hidream_o1_noise_scale": -1},
            {"hidream_o1_noise_scale": 65},
            {"hidream_o1_seam_smoothing": "true"},
            {"hidream_o1_seam_pattern": "random"},
            {"hidream_o1_seam_passes": "8"},
            {"hidream_o1_seam_blend": "custom"},
            {"hidream_o1_seam_start": 0.9, "hidream_o1_seam_end": 0.8},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_guidance(values)
        for value in (float("nan"), -1, 65):
            with self.assertRaises(ValidationError):
                ImageRequest(hidream_o1_noise_scale=value)
        other = resolve_guidance({"hidream_o1_noise_scale": 7.5}, {}, "MOSSImage")
        self.assertFalse(other["hidream_o1_seam_smoothing"])
        self.assertIsNone(other["hidream_o1_noise_scale"])
        restored = restore_guidance_metadata(
            {"hidream_o1_seam_start": 0.9, "hidream_o1_seam_end": 0.8}
        )
        self.assertFalse({"hidream_o1_seam_start", "hidream_o1_seam_end"} & restored.keys())

    def test_api_freezes_hidream_options_without_carrying_other_model_overrides(self):
        import ruinedfooocus.app.api_runtime as api_runtime

        perf = SimpleNamespace(
            CUSTOM_PERFORMANCE="Own settings",
            choices_for_model=lambda *args: ["Own settings"],
            selection_for_model=lambda *args: "Own settings",
            get_perf_options=lambda *args: dict(
                custom_steps=28, cfg=1, sampler_name="lcm", scheduler="normal", clip_skip=1
            ),
        )
        shared = SimpleNamespace(
            settings=SimpleNamespace(default_settings={"performance": "Own settings"}),
            performance_settings=perf,
            models=SimpleNamespace(offline=True, get_file=lambda *args: None),
            path_manager=SimpleNamespace(get_model_folders=lambda category: []),
            resolution_settings=SimpleNamespace(CUSTOM_RESOLUTION="Custom...", aspect_ratios={}),
        )
        samplers = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=["lcm"], SCHEDULERS=["normal"])
        )
        defaults = dict(base_model_name="other.safetensors", hidream_o1_noise_scale=8)
        payload = dict(
            prompt="photo",
            size="512x512",
            hidream_o1_noise_scale=7.6,
            hidream_o1_seam_smoothing=True,
            hidream_o1_seam_pattern="symmetric",
            hidream_o1_seam_start=0.7,
            hidream_o1_seam_end=0.9,
            hidream_o1_seam_passes="4",
            hidream_o1_seam_blend="window",
            hidream_o1_seam_strength=0.5,
        )
        with (
            patch.dict(
                sys.modules, {"ruinedfooocus.app.state": shared, "comfy.samplers": samplers}
            ),
            # Patch the package attribute too: 'import ruinedfooocus.app.state as x' reads it.
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(api_runtime, "image_defaults", defaults),
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
            # The checkpoint plan reads an installed file; this test covers options only.
            patch(
                "ruinedfooocus.features.models.effective_image_plan.capture_effective_image_plan",
                side_effect=lambda data, *a, **kw: data,
            ),
            patch(
                "ruinedfooocus.features.generation.sdcpp_pipeline.capture_image_request",
                side_effect=lambda data, *a, **kw: data,
            ),
        ):
            operation = api_runtime.images(
                payload, dict(family="HiDreamO1", name="dev.safetensors")
            )
            auto = api_runtime.images(
                dict(prompt="photo", size="512x512"),
                dict(family="HiDreamO1", name="dev.safetensors"),
            )
        data = dict(
            zip(operation.__code__.co_freevars, (c.cell_contents for c in operation.__closure__))
        )["data"]
        auto_data = dict(
            zip(auto.__code__.co_freevars, (c.cell_contents for c in auto.__closure__))
        )["data"]
        self.assertIsNone(auto_data["hidream_o1_noise_scale"])
        from ruinedfooocus.features.generation.size_plan import KEY as SIZE_PLAN_KEY

        self.assertEqual(data[SIZE_PLAN_KEY].target, (512, 512))
        self.assertEqual(data[SIZE_PLAN_KEY].alignment, 32)
        for key in FIELD_SCHEMA:
            if key.startswith("hidream_o1_"):
                self.assertEqual(data[key], payload[key])
        payload["hidream_o1_noise_scale"] = 0
        defaults["hidream_o1_noise_scale"] = 0
        self.assertEqual(data["hidream_o1_noise_scale"], 7.6)

    def test_real_gradio_controls_have_auto_and_keep_existing_defaults(self):
        import gradio as gr
        from ruinedfooocus.features.image_editing.ui_guidance import (
            create_hidream_controls,
            create_lcm_controls,
        )

        with gr.Blocks(analytics_enabled=False):
            fields, options = create_hidream_controls({})
            lcm = create_lcm_controls({})
        self.assertEqual(
            set(fields), {key for key in FIELD_SCHEMA if key.startswith("hidream_o1_")}
        )
        self.assertEqual(fields["hidream_o1_noise_scale"].value, "")
        self.assertEqual(fields["hidream_o1_seam_smoothing"].value, "auto")
        self.assertEqual(fields["hidream_o1_seam_passes"].value, "ramp_2_4")
        self.assertEqual(fields["hidream_o1_seam_blend"].value, "median")
        self.assertEqual(
            {key: control.value for key, control in lcm.items()},
            dict(lcm_noise="", lcm_noise_end="", lcm_noise_clip=""),
        )

    def test_gallery_replay_can_be_submitted_through_hidden_seam_control(self):
        import gradio as gr
        from test_image_metadata_boundary import gallery
        from ruinedfooocus.features.image_editing.ui_guidance import create_hidream_controls

        updates = gallery({"prompt": "Keep the source."})
        with gr.Blocks(analytics_enabled=False):
            fields, _ = create_hidream_controls({})
        field = fields["hidream_o1_seam_smoothing"]
        value = updates["hidream_o1_seam_smoothing"]["value"]
        self.assertEqual(field.preprocess(value), "off")


if __name__ == "__main__":
    unittest.main()
