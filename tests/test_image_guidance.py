"""Request guidance validation/patch isolation without loading model weights."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from unittest.mock import patch
import sys

from ruinedfooocus.features.image_editing.image_guidance import (
    FIELD_SCHEMA,
    capabilities,
    resolve_guidance,
    request_model,
    effective_guidance,
    validate_guidance,
    normalize_reference_images,
)


ROOT = Path(__file__).resolve().parents[1]


class FakePatcher:
    def __init__(self, effects=(), shift=3.1):
        self.effects, self.shift = list(effects), shift

    def clone(self):
        return FakePatcher(self.effects, self.shift)

    def get_model_object(self, key):
        return SimpleNamespace(shift=self.shift)


class ImageGuidanceTests(unittest.TestCase):
    def mock_independent_resource_capture(self):
        # These tests own guidance transport. Resource/prompt/immutable-plan
        # capture has independent fixtures and must not acquire model files here.
        for target in (
            "ruinedfooocus.features.models.component_profiles.snapshot_request",
            "ruinedfooocus.features.prompting.local_model.capture_prompt_preparation",
            "ruinedfooocus.features.generation.image_preparation.capture_image_assets",
            "ruinedfooocus.features.generation.sdcpp_pipeline.capture_image_request",
            "ruinedfooocus.features.models.effective_image_plan.capture_effective_image_plan",
        ):
            self.enterContext(patch(target, side_effect=lambda data, *args, **kwargs: data))

    def test_reference_gallery_normalization_and_family_guards(self):
        from PIL import Image

        primary = Image.new("RGB", (32, 24))
        reference = Image.new("RGBA", (16, 8))
        normalized = normalize_reference_images(
            [(reference, "caption")], primary, "MOSSImage", "ai-toolkit"
        )
        self.assertEqual(normalized[0].mode, "RGB")
        self.assertTrue(capabilities("MageFlow")["multiple_refs"])
        self.assertTrue(capabilities("QwenImage")["multiple_refs"])
        self.assertTrue(capabilities("Flux2Klein4B")["multiple_refs"])
        for refs, source, family, mode in (
            ([reference], None, "MageFlow", "Disabled"),
            ([reference], primary, "MOSSImage", "Disabled"),
            ([reference] * 3, primary, "QwenImage", "Disabled"),
            ([reference], primary, "SDXL", "Disabled"),
            ([reference] * 16, primary, "MageFlow", "Disabled"),
            (["C:/image.png"], primary, "MageFlow", "Disabled"),
        ):
            with self.assertRaises(ValueError):
                normalize_reference_images(refs, source, family, mode)

    def test_defaults_and_explicit_off_preserve_default_contract(self):
        defaults = dict(
            moss_image_slg=True,
            moss_image_edit_format="Moss-Image-Edit",
            moss_image_edit_strength=2,
            cfg_norm_strength=1,
        )
        values = resolve_guidance({}, defaults, "MOSSImage")
        self.assertTrue(values["moss_image_slg"])
        self.assertEqual(values["moss_image_edit_format"], "Moss-Image-Edit")
        self.assertEqual(values["moss_image_edit_strength"], 2)
        self.assertIsNone(values["sampling_shift"])
        off = resolve_guidance(
            {"cfg_norm_strength": None, "moss_image_slg": False}, defaults, "MOSSImage"
        )
        self.assertIsNone(off["cfg_norm_strength"])
        self.assertFalse(off["moss_image_slg"])

    def test_qwen_schedule_is_explicit_and_confined_to_its_model_family(self):
        self.assertEqual(resolve_guidance({}, {}, "QwenImage21")["qwen_image21_schedule"], "auto")
        for mode in FIELD_SCHEMA["qwen_image21_schedule"]["choices"]:
            values = {"qwen_image21_schedule": mode}
            self.assertEqual(validate_guidance(values), values)
            self.assertEqual(
                resolve_guidance(values, {}, "QwenImage21")["qwen_image21_schedule"], mode
            )
        self.assertEqual(
            resolve_guidance({}, {"qwen_image21_schedule": "turbo_8"}, "SDXL")[
                "qwen_image21_schedule"
            ],
            "auto",
        )
        self.assertTrue(capabilities("QwenImage21")["qwen_image21"])
        self.assertFalse(capabilities("QwenImage")["qwen_image21"])
        with self.assertRaises(ValueError):
            validate_guidance({"qwen_image21_schedule": "guess"})

    def test_native_patches_never_accumulate_on_cached_lora_model(self):
        calls = []

        def patch(kind):
            def apply(model, **kwargs):
                calls.append((kind, kwargs))
                changed = model.clone()
                changed.effects.append(kind)
                if "shift" in kwargs:
                    changed.shift = kwargs["shift"]
                return (changed,)

            return apply

        nodes = {key: patch(key) for key in ("AuraFlow", "SD3", "slg", "cfg_norm")}
        base = FakePatcher(["lora"])
        enabled = resolve_guidance(
            dict(moss_image_slg=True, sampling_shift=5, cfg_norm_strength=1), {}, "MOSSImage"
        )
        first = request_model(base, enabled, "MOSSImage", nodes)
        second = request_model(base, resolve_guidance({}, {}, "MOSSImage"), "MOSSImage", nodes)
        self.assertEqual(base.effects, ["lora"])
        self.assertEqual(second.effects, ["lora"])
        self.assertIsNot(second, base)
        self.assertEqual(first.effects, ["lora", "AuraFlow", "slg", "cfg_norm"])
        self.assertEqual(
            calls[1],
            ("slg", dict(double_layers="", single_layers="22", start_percent=0.0, end_percent=0.5)),
        )
        self.assertEqual(calls[2], ("cfg_norm", dict(strength=1.0, pre_cfg=False)))
        self.assertEqual(
            effective_guidance(enabled, first, "MOSSImage")["effective_sampling_shift"], 5
        )
        automatic = effective_guidance(resolve_guidance({}, {}, "QwenImage"), base, "QwenImage")
        self.assertIsNone(automatic["sampling_shift"])
        self.assertEqual(automatic["effective_sampling_shift"], 3.1)

    def test_invalid_numbers_formats_and_family_shift_fail(self):
        for values in (
            {"sampling_shift": 0},
            {"sampling_shift": 101},
            {"cfg_norm_strength": float("nan")},
            {"cfg_norm_strength": float("inf")},
            {"moss_image_edit_strength": True},
            {"moss_image_slg": "false"},
            {"moss_image_edit_format": "not a format"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_guidance(values)
        self.assertEqual(
            validate_guidance({"cfg_norm_strength": "", "sampling_shift": ""}),
            {"cfg_norm_strength": None, "sampling_shift": None},
        )
        with self.assertRaises(ValueError):
            resolve_guidance({"sampling_shift": 3}, {}, "SDXL")
        self.assertTrue(capabilities("QwenImage")["sampling_shift"])
        self.assertTrue(capabilities("Fibo2")["multiple_refs"])
        from PIL import Image

        primary, reference = Image.new("RGB", (16, 16)), Image.new("RGBA", (8, 8))
        self.assertEqual(normalize_reference_images([reference], primary, "Fibo2")[0].mode, "RGB")

    def test_api_schema_and_null_patch_switches(self):
        from ruinedfooocus.integrations.openai_api import ImageRequest
        from pydantic import ValidationError

        request = ImageRequest(
            prompt="edit",
            moss_image_edit_format="ai-toolkit",
            moss_image_edit_strength=0,
            moss_image_slg=True,
            sampling_shift=3,
            cfg_norm_strength=None,
            cfg_norm_auto=False,
        )
        self.assertTrue(set(FIELD_SCHEMA).issubset(ImageRequest.model_fields))
        self.assertIn("cfg_norm_strength", request.model_fields_set)
        self.assertFalse(request.model_dump()["cfg_norm_auto"])
        self.assertTrue(ImageRequest(cfg_norm_auto=True).cfg_norm_auto)
        self.assertEqual(request.model_dump()["sampling_shift"], 3)
        for schedule in FIELD_SCHEMA["qwen_image21_schedule"]["choices"]:
            self.assertEqual(
                ImageRequest(qwen_image21_schedule=schedule).qwen_image21_schedule, schedule
            )
        for framing in FIELD_SCHEMA["image_resize_mode"]["choices"]:
            with self.subTest(framing=framing):
                self.assertEqual(ImageRequest(image_resize_mode=framing).image_resize_mode, framing)
        for values in (
            {"cfg_norm_strength": 101},
            {"sampling_shift": 0},
            {"moss_image_edit_strength": float("nan")},
            {"moss_image_edit_format": "bogus"},
            {"image_resize_mode": "stretch"},
            {"qwen_image21_schedule": "guess"},
        ):
            with self.assertRaises(ValidationError):
                ImageRequest(**values)
        self.assertEqual(ImageRequest(reference_images=["a", "b"]).reference_images, ["a", "b"])
        with self.assertRaises(ValidationError):
            ImageRequest(reference_images=["a"] * 16)

    def test_api_guidance_snapshot_reaches_queued_worker_data_without_generation(self):
        self.mock_independent_resource_capture()
        import ruinedfooocus.app.api_runtime as api_runtime

        perf = SimpleNamespace(
            CUSTOM_PERFORMANCE="Own settings",
            choices_for_model=lambda *args: ["Own settings"],
            selection_for_model=lambda *args: "Own settings",
            get_perf_options=lambda *args: dict(
                custom_steps=8, cfg=4, sampler_name="euler", scheduler="simple", clip_skip=1
            ),
        )
        shared = SimpleNamespace(
            settings=SimpleNamespace(default_settings={"performance": "Own settings"}),
            performance_settings=perf,
            models=SimpleNamespace(offline=True, get_file=lambda *args: None),
            path_manager=SimpleNamespace(
                get_model_folders=lambda category: [], DOWNLOADABLE_FILES={}
            ),
            resolution_settings=SimpleNamespace(),
        )
        fake_samplers = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["simple"])
        )
        defaults = dict(
            base_model_name="local.safetensors",
            performance_selection="Own settings",
            moss_image_edit_format="ai-toolkit",
            moss_image_edit_strength=2,
            moss_image_slg=True,
            cfg_norm_strength=1,
            reference_images=["must never inherit these"],
        )
        with (
            patch.dict(
                sys.modules, {"ruinedfooocus.app.state": shared, "comfy.samplers": fake_samplers}
            ),
            patch.object(sys.modules["ruinedfooocus.app"], "state", shared, create=True),
            patch.object(api_runtime, "image_defaults", defaults),
            patch.object(api_runtime, "submit", side_effect=lambda run: run),
        ):
            operation = api_runtime.images(
                dict(
                    prompt="edit",
                    size="512x512",
                    cfg_norm_strength=None,
                    cfg_norm_auto=True,
                    moss_image_edit_strength=0,
                    vae_tiled=True,
                    vae_tile_size=256,
                    vae_overlap=0,
                    moss_image_slg_double_layers="1, 3",
                    moss_image_slg_single_layers="8, 9",
                    moss_image_slg_start=0.2,
                    moss_image_slg_end=0.8,
                    cfg_norm_pre_cfg=True,
                ),
                dict(family="MOSSImage", name="local.safetensors"),
            )
        cells = dict(
            zip(
                operation.__code__.co_freevars,
                (cell.cell_contents for cell in operation.__closure__),
            )
        )
        data = cells["data"]
        defaults["moss_image_slg"] = False
        self.assertTrue(data["moss_image_slg"])
        self.assertEqual(data["moss_image_edit_format"], "ai-toolkit")
        self.assertEqual(data["moss_image_edit_strength"], 0)
        self.assertIsNone(data["cfg_norm_strength"])
        self.assertTrue(data["cfg_norm_auto"])
        self.assertEqual(data["reference_images"], [])
        self.assertEqual(data["image_resize_mode"], "pad")
        self.assertEqual(
            (data["vae_tiled"], data["vae_tile_size"], data["vae_overlap"]), (True, 256, 0)
        )
        self.assertEqual(
            (
                data["moss_image_slg_double_layers"],
                data["moss_image_slg_single_layers"],
                data["moss_image_slg_start"],
                data["moss_image_slg_end"],
                data["cfg_norm_pre_cfg"],
            ),
            ("1, 3", "8, 9", 0.2, 0.8, True),
        )
        from PIL import Image

        image = Image.new("RGB", (32, 24))
        with (
            patch.dict(
                sys.modules, {"ruinedfooocus.app.state": shared, "comfy.samplers": fake_samplers}
            ),
            patch.object(sys.modules["ruinedfooocus.app"], "state", shared, create=True),
            patch.object(api_runtime, "image_defaults", defaults),
            patch.object(api_runtime, "decode_image", side_effect=lambda value, **kwargs: image),
            patch.object(api_runtime, "submit", side_effect=lambda run: run),
        ):
            operation = api_runtime.images(
                dict(
                    prompt="edit",
                    size="512x512",
                    input_image="primary",
                    reference_images=["additional"],
                ),
                dict(family="MOSSImage", name="local.safetensors"),
            )
        cells = dict(
            zip(
                operation.__code__.co_freevars,
                (cell.cell_contents for cell in operation.__closure__),
            )
        )
        self.assertEqual(len(cells["data"]["reference_images"]), 1)
        self.assertEqual(cells["data"]["reference_images"][0].mode, "RGB")
        self.assertEqual(cells["data"]["input_image"].size, image.size)
        with (
            patch.dict(
                sys.modules, {"ruinedfooocus.app.state": shared, "comfy.samplers": fake_samplers}
            ),
            patch.object(sys.modules["ruinedfooocus.app"], "state", shared, create=True),
            patch.object(api_runtime, "image_defaults", defaults),
            patch.object(api_runtime, "decode_image", return_value=image),
            patch.object(api_runtime, "submit", side_effect=lambda run: run),
        ):
            for size, framing in (("auto", "preserve"), ("512x512", "preserve")):
                payload = dict(
                    prompt="edit", size=size, input_image="primary", image_resize_mode=framing
                )
                shared.settings.default_settings["resolution"] = "Custom..."
                shared.resolution_settings = SimpleNamespace(
                    selection_for_model=lambda *args: "Custom...", aspect_ratios={}
                )
                operation = api_runtime.images(
                    payload, dict(family="MOSSImage", name="local.safetensors")
                )
                queued = dict(
                    zip(
                        operation.__code__.co_freevars,
                        (cell.cell_contents for cell in operation.__closure__),
                    )
                )["data"]
                self.assertEqual(queued["image_resize_mode"], framing)
                self.assertEqual(queued["input_image"].size, image.size)

    def test_api_cross_model_ignores_ui_guidance_but_same_model_inherits(self):
        self.mock_independent_resource_capture()
        import ruinedfooocus.app.api_runtime as api_runtime

        perf = SimpleNamespace(
            CUSTOM_PERFORMANCE="Own settings",
            choices_for_model=lambda *args: ["Model Defaults"],
            selection_for_model=lambda *args: "Model Defaults",
            get_perf_options=lambda *args: dict(
                custom_steps=8, cfg=1, sampler_name="euler", scheduler="normal", clip_skip=1
            ),
        )
        settings = {"performance": "Model Defaults", "cfg_norm_strength": 0.5}
        shared = SimpleNamespace(
            settings=SimpleNamespace(default_settings=settings),
            performance_settings=perf,
            models=SimpleNamespace(offline=True, get_file=lambda *args: None),
            path_manager=SimpleNamespace(
                get_model_folders=lambda category: [], DOWNLOADABLE_FILES={}
            ),
            resolution_settings=SimpleNamespace(),
        )
        samplers = SimpleNamespace(
            KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["normal"])
        )
        defaults = dict(
            base_model_name="moss.safetensors",
            sampling_shift=3,
            cfg_norm_strength=2,
            moss_image_edit_format="ai-toolkit",
            moss_image_slg=True,
        )
        with (
            patch.dict(
                sys.modules, {"ruinedfooocus.app.state": shared, "comfy.samplers": samplers}
            ),
            patch.object(sys.modules["ruinedfooocus.app"], "state", shared, create=True),
            patch.object(api_runtime, "image_defaults", defaults),
            patch.object(api_runtime, "submit", side_effect=lambda run: run),
        ):

            def queued(name, family, **extra):
                operation = api_runtime.images(
                    dict(prompt="photo", size="512x512", **extra), dict(name=name, family=family)
                )
                return dict(
                    zip(
                        operation.__code__.co_freevars,
                        (cell.cell_contents for cell in operation.__closure__),
                    )
                )["data"]

            other = queued("krea.safetensors", "Krea2")
            self.assertIsNone(other["sampling_shift"])
            self.assertEqual(other["cfg_norm_strength"], 0.5)
            self.assertEqual(other["moss_image_edit_format"], "Disabled")
            same = queued("moss.safetensors", "MOSSImage")
            self.assertEqual(same["sampling_shift"], 3)
            self.assertEqual(same["cfg_norm_strength"], 2)
            self.assertTrue(same["moss_image_slg"])
            explicit = queued(
                "qwen.safetensors", "QwenImage", sampling_shift=4, cfg_norm_strength=None
            )
            self.assertEqual(explicit["sampling_shift"], 4)
            self.assertIsNone(explicit["cfg_norm_strength"])
            # A MOSS preset controls guidance without changing Main's Krea model.
            defaults.update(
                base_model_name="krea.safetensors", _image_guidance_checkpoint="moss.safetensors"
            )
            main = queued("krea.safetensors", "Krea2")
            self.assertIsNone(main["sampling_shift"])
            self.assertEqual(main["cfg_norm_strength"], 0.5)
            self.assertEqual(main["moss_image_edit_format"], "Disabled")

    def test_production_sampler_uses_request_clone_and_metadata_uses_effective_recipe(self):
        pipeline = (ROOT / "ruinedfooocus/features/generation/image_pipeline.py").read_text(
            encoding="utf-8"
        )
        worker = (ROOT / "ruinedfooocus/app/async_worker.py").read_text(encoding="utf-8")
        tree = ast.parse(pipeline)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "pipeline")
        process = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process"
        )
        calls = [
            n
            for n in ast.walk(process)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "comfy.sample.sample"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            ast.unparse(next(k.value for k in calls[0].keywords if k.arg == "model")),
            "sampling_model",
        )
        self.assertNotIn('"SkipLayerGuidanceDiTSimple":', pipeline)
        updates = [
            ast.unparse(node)
            for node in ast.walk(ast.parse(worker))
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "prompt.update"
        ]
        self.assertIn("prompt.update(gen_data['image_guidance'])", updates)


if __name__ == "__main__":
    unittest.main()
