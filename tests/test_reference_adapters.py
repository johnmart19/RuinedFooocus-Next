"""Native image-prompt nodes use captured local files and request-local images."""

import ast
from concurrent.futures import CancelledError
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from ruinedfooocus.features.image_editing import reference_adapters as adapters
from ruinedfooocus.features.generation.image_preparation import ImageAssets, KEY
from ruinedfooocus.features.models.model_identity import file_signature
from test_native_model_coverage import NATIVE, native_method, PIPELINE
from test_native_variant_defaults import write_header


def native_values():
    tree = ast.parse((NATIVE / "node_helpers.py").read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "conditioning_set_values"
    )
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "native_conditions", "exec"), scope)
    return scope[node.name]


class ReferenceAdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.manager = NS(
            get_model_folders=lambda category: [self.root / category],
            DOWNLOADABLE_FILES={},
            model_paths={"embedding_path": [], "cache_path": self.root},
        )
        self.checkpoint = write_header(self.root / "checkpoint.safetensors", {"opaque.weight": [1]})
        self.family = "Flux"
        self.models = NS(
            get_file=lambda *a: self.checkpoint,
            get_models_by_path=lambda *a, **k: {},
            get_model_base=lambda info: self.family,
        )
        self.image = Image.new("RGB", (8, 8), "red")
        for category in ("style_models", "clip_vision", "photomaker"):
            (self.root / category).mkdir()

    def files(self, mode):
        if mode == "photomaker":
            path = self.root / "photomaker/native.bin"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"opaque native safe-load fixture")
            return path.name, ""
        style = (
            {"redux_up.weight": [12288, 1152], "redux_down.weight": [4096, 12288]}
            if mode == "redux"
            else {"style_embedding": [1, 8, 1024], "proj": [1024, 768]}
        )
        model = write_header(self.root / f"style_models/{mode}.safetensors", style)
        width, positions, last = (1152, 729, 26) if mode == "redux" else (1024, 257, 23)
        vision = write_header(
            self.root / f"clip_vision/{mode}.safetensors",
            {
                "vision_model.embeddings.patch_embedding.weight": [width, 3, 14, 14],
                "vision_model.embeddings.position_embedding.weight": [positions, width],
                f"vision_model.encoder.layers.{last}.layer_norm1.weight": [width],
            },
        )
        return model.name, vision.name

    def data(self, mode, **options):
        self.family = {"redux": "Flux", "style": "SD 1.5", "photomaker": "SDXL"}[mode]
        model, vision = self.files(mode)
        result, artifacts = adapters.capture_assets(
            dict(
                reference_adapter=mode,
                reference_adapter_model=model,
                reference_adapter_vision=vision,
                reference_adapter_image=self.image,
                reference_adapter_strength=1,
                base_model_name="checkpoint",
                clip_skip=1,
                positive_prompt="portrait of person photomaker",
                steps=2,
                **options,
            ),
            self.manager,
            self.models,
            {},
        )
        result[KEY] = ImageAssets(artifacts, True)
        return result

    def owner(self, mode):
        name = {"redux": "Flux", "style": "SD15", "photomaker": "SDXL"}[mode]
        config = type(
            name, (), {"unet_config": {"in_channels": 64}, "inpaint_model": lambda self: False}
        )()
        return NS(
            base_file_signature=file_signature(self.checkpoint),
            xl_base_patched=NS(unet=NS(model=NS(model_config=config)), clip=None),
        )

    @staticmethod
    def pixels(image):
        return torch.from_numpy(np.asarray(image.convert("RGB")).copy()).float()[None] / 255

    def test_local_choices_are_shape_filtered_and_do_not_unpickle_photomaker(self):
        self.files("redux")
        self.files("style")
        self.files("photomaker")
        self.assertEqual(
            adapters.local_choices(self.manager, "redux", "model"), ["redux.safetensors"]
        )
        self.assertEqual(
            adapters.local_choices(self.manager, "style", "model"), ["style.safetensors"]
        )
        self.assertEqual(
            adapters.local_choices(self.manager, "redux", "vision"), ["redux.safetensors"]
        )
        with patch.object(
            Path, "open", side_effect=AssertionError("PhotoMaker picker read payload")
        ):
            self.assertEqual(
                adapters.local_choices(self.manager, "photomaker", "model"), ["native.bin"]
            )

    def test_capture_owns_image_zero_strength_and_refuses_wrong_family_or_missing_source(self):
        data = self.data("redux")
        self.image.paste("blue", (0, 0, 8, 8))
        self.assertEqual(data["reference_adapter_image"].getpixel((0, 0)), (255, 0, 0))
        self.assertIsNot(data["reference_adapter_image"], self.image)
        payload = dict(data, reference_adapter_strength=0)
        captured, _ = adapters.capture_assets(payload, self.manager, self.models, {})
        self.assertEqual(captured[adapters.KEY].strength, 0)
        for changes in (
            {"reference_adapter_image": None},
            {"image_engine": "sdcpp-cpu"},
            {"reference_adapter_strength": False},
            {"reference_adapter_strength": float("nan")},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                adapters.capture_assets(dict(data, **changes), self.manager, self.models, {})
        self.family = "SDXL"
        with self.assertRaisesRegex(ValueError, "family"):
            adapters.capture_assets(data, self.manager, self.models, {})

    def test_native_torch_reference_choices_do_not_read_weights_and_capture_exact_sources(self):
        from ruinedfooocus.core.model_formats import NATIVE_TORCH_SUFFIXES

        for category in ("style_models", "clip_vision"):
            for suffix in NATIVE_TORCH_SUFFIXES:
                (self.root / category / ("local" + suffix.upper())).write_bytes(b"unknown weights")
        with patch.object(Path, "open", side_effect=AssertionError("picker read weights")):
            for mode in ("style", "redux"):
                for role in ("model", "vision"):
                    self.assertEqual(
                        set(adapters.local_choices(self.manager, mode, role)),
                        {"local" + suffix.upper() for suffix in NATIVE_TORCH_SUFFIXES},
                    )
            self.assertFalse(adapters.compatible_file("none", "model", "local.PT"))
            self.assertFalse(adapters.compatible_file("photomaker", "vision", "local.PT"))
        self.family = "Flux"
        data, artifacts = adapters.capture_assets(
            dict(
                reference_adapter="redux",
                reference_adapter_model="local.PT",
                reference_adapter_vision="local.BIN",
                reference_adapter_image=self.image,
                reference_adapter_strength=1,
                base_model_name="checkpoint",
            ),
            self.manager,
            self.models,
            {},
        )
        self.assertEqual([item.name for item in artifacts], ["local.PT", "local.BIN"])
        self.assertEqual(data[adapters.KEY].artifacts, tuple(artifacts))

    def test_pinned_style_apply_preserves_schedules_negative_and_a_b_a_image_isolation(self):
        native_encode = native_method("nodes.py", "CLIPVisionEncode", "encode", {})
        native_apply = native_method(
            "nodes.py", "StyleModelApply", "apply_stylemodel", {"torch": torch}
        )
        for mode, width in (("redux", 4096), ("style", 768)):
            data = self.data(mode)
            owner = self.owner(mode)
            vision_width = 1152 if mode == "redux" else 1024
            vision = NS(
                encode_image=lambda pixels, **kw: NS(
                    last_hidden_state=pixels.mean().expand(1, 2, vision_width)
                )
            )
            style = NS(
                get_cond=lambda encoded: (
                    encoded.last_hidden_state.mean(-1, keepdim=True).expand(1, 2, width).clone()
                )
            )
            load = Mock(return_value=style)
            native = NS(
                sd=NS(load_style_model=load), clip_vision=NS(load=Mock(return_value=vision))
            )
            positive = [[torch.ones(1, 3, width), {"start_percent": 0.25, "end_percent": 0.75}]]
            negative = [[torch.zeros(1, 3, width), {"negative": "unchanged"}]]
            modules = {
                "comfy": native,
                "comfy.sd": native.sd,
                "comfy.clip_vision": native.clip_vision,
                "nodes": NS(CLIPVisionEncode=native_encode, StyleModelApply=native_apply),
            }
            values = []
            with patch.dict(sys.modules, modules):
                for color in ("red", "white", "red"):
                    current = dict(data, reference_adapter_image=Image.new("RGB", (8, 8), color))
                    output = adapters.apply_reference(
                        owner, positive, current, self.manager, self.pixels
                    )
                    values.append(output[0][0].clone())
                    self.assertEqual(output[0][1], positive[0][1])
                    self.assertTrue(torch.equal(negative[0][0], torch.zeros_like(negative[0][0])))
                    self.assertEqual(current["reference_adapter_info"]["source_size"], [8, 8])
                self.assertTrue(torch.equal(values[0], values[2]))
                self.assertFalse(torch.equal(values[0], values[1]))
                self.assertTrue(torch.equal(positive[0][0], torch.ones_like(positive[0][0])))
                load.assert_called_once()

    def test_mutated_file_wrong_primary_and_unknown_capture_are_rejected_before_native_load(self):
        data = self.data("redux")
        owner = self.owner("redux")
        owner.base_file_signature = ()
        with self.assertRaisesRegex(ValueError, "another checkpoint"):
            adapters.apply_reference(owner, [], data, self.manager, self.pixels)
        owner.base_file_signature = file_signature(self.checkpoint)
        model = data[adapters.KEY].artifacts[0].request.destination
        model.write_bytes(b"replacement")
        with self.assertRaisesRegex(ValueError, "changed"):
            adapters.apply_reference(owner, [], data, self.manager, self.pixels)
        with self.assertRaisesRegex(ValueError, "Capture"):
            adapters.apply_reference(
                owner, [], {"reference_adapter": "redux"}, self.manager, self.pixels
            )

    def test_native_state_dictionary_refusal_does_not_publish_cached_components(self):
        data = self.data("redux")
        owner = self.owner("redux")
        native = NS(
            sd=NS(load_style_model=Mock(side_effect=RuntimeError("weights-only refused"))),
            clip_vision=NS(load=Mock()),
        )
        with patch.dict(
            sys.modules,
            {"comfy": native, "comfy.sd": native.sd, "comfy.clip_vision": native.clip_vision},
        ):
            with self.assertRaisesRegex(ValueError, "invalid or incompatible.*weights-only"):
                adapters.apply_reference(owner, [], data, self.manager, self.pixels)
        self.assertFalse(hasattr(owner, "reference_adapter_models"))
        native.clip_vision.load.assert_not_called()
        native.sd.load_style_model.side_effect = CancelledError()
        with patch.dict(
            sys.modules,
            {"comfy": native, "comfy.sd": native.sd, "comfy.clip_vision": native.clip_vision},
        ):
            with self.assertRaises(CancelledError):
                adapters.apply_reference(owner, [], data, self.manager, self.pixels)

    def test_actual_photomaker_loader_and_encoder_use_exact_file_trigger_and_scheduled_clip(self):
        data = self.data("photomaker")
        data["positive_prompt"] = "[portrait of person photomaker|photo of person photomaker]"
        owner = self.owner("photomaker")

        class Clip:
            def tokenize(self, text, **kwargs):
                return {name: [[(i, 1, i) for i in range(77)]] for name in ("l", "g")}

            def encode_from_tokens(self, tokens, **kwargs):
                return torch.zeros(1, 77, 2048), "native-pooled"

        owner.xl_base_patched.clip = Clip()

        class Identity:
            load_device = "cpu"
            load_state_dict = Mock()

            def __call__(self, id_pixel_values, prompt_embeds, class_tokens_mask):
                return prompt_embeds + id_pixel_values.mean() * class_tokens_mask.unsqueeze(-1)

        identity = Identity()
        files = NS(
            add_model_folder_path=Mock(),
            get_full_path_or_raise=lambda *a: str(self.root / "photomaker/native.bin"),
        )
        utils = NS(load_torch_file=Mock(return_value={"id_encoder": {"native": "weights"}}))
        native = NS(utils=utils, clip_vision=NS(clip_preprocess=lambda image: image))
        loader = native_method(
            "comfy_extras/nodes_photomaker.py",
            "PhotoMakerLoader",
            "execute",
            dict(
                folder_paths=files,
                PhotoMakerIDEncoder=lambda: identity,
                comfy=native,
                io=NS(NodeOutput=lambda *v: v),
            ),
        )
        encoder = native_method(
            "comfy_extras/nodes_photomaker.py",
            "PhotoMakerEncode",
            "execute",
            dict(torch=torch, comfy=native, io=NS(NodeOutput=lambda *v: v)),
        )
        modules = {
            "folder_paths": files,
            "comfy_extras.nodes_photomaker": NS(PhotoMakerLoader=loader, PhotoMakerEncode=encoder),
            "node_helpers": NS(conditioning_set_values=native_values()),
        }
        positive = [[torch.zeros(1, 77, 2048), {}]]
        with (
            patch.dict(sys.modules, modules),
            patch("ruinedfooocus.backends.comfy.text_encoder_warmup.warm_patched_text_encoder"),
        ):
            output = adapters.apply_reference(owner, positive, data, self.manager, self.pixels)
        self.assertEqual(len(output), 2)
        self.assertEqual(
            [(c[1]["start_percent"], c[1]["end_percent"]) for c in output], [(0, 0.5), (0.5, 1)]
        )
        self.assertEqual(output[0][1]["pooled_output"], "native-pooled")
        self.assertTrue(torch.equal(positive[0][0], torch.zeros_like(positive[0][0])))
        utils.load_torch_file.assert_called_once_with(
            str(self.root / "photomaker/native.bin"), safe_load=True
        )
        identity.load_state_dict.assert_called_once_with({"native": "weights"})

    def test_actual_pipeline_calls_adapter_after_cached_text_without_touching_negative(self):
        tree = ast.parse(
            (Path(__file__).resolve().parents[1] / PIPELINE).read_text(encoding="utf-8")
        )
        process = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process"
        )
        start = next(
            i
            for i, n in enumerate(process.body)
            if isinstance(n, ast.ImportFrom)
            and n.module == "ruinedfooocus.features.image_editing.reference_adapters"
        )
        code = process.body[start : start + 3]
        positive, negative, schedule = object(), object(), object()
        scope = dict(
            self="owner",
            gen_data="captured",
            path_manager="paths",
            reference_tensor="convert",
            positive_conditioning=positive,
            negative_conditioning=negative,
            switched_prompt=schedule,
        )
        adapter = Mock(return_value="native-request-positive")
        with patch.object(adapters, "apply_reference", adapter):
            exec(compile(ast.Module(body=code, type_ignores=[]), "adapter_caller", "exec"), scope)
        adapter.assert_called_once_with("owner", schedule, "captured", "paths", "convert")
        self.assertEqual(scope["switched_prompt"], "native-request-positive")
        self.assertIs(scope["negative_conditioning"], negative)

    def test_actual_generation_utility_generation_capture_preserves_ui_intent(self):
        from test_image_preparation import function
        from ruinedfooocus.features.generation import image_preparation as assets
        from ruinedfooocus.features.models.yaml_recipes import KEY as RECIPE_KEY

        data = self.data("style")
        recipe = self.root / "checkpoints/selected.yaml"
        recipe.parent.mkdir()
        recipe.write_text("model: {params: {parameterization: eps}}")
        data.update(
            model_recipe_path=str(recipe),
            clip_layer_source="request",
            prompt="literal",
            controlnet={},
        )
        upscaler = self.root / "upscalers/scale.pth"
        upscaler.parent.mkdir()
        upscaler.write_bytes(b"local")
        shared = NS(
            models=self.models,
            path_manager=self.manager,
            resolution_settings=None,
            performance_settings=NS(apply=lambda data, family: dict(data)),
        )
        capture = function(
            "capture_image_data",
            dict(
                shared=shared,
                settings=NS(default_settings={}),
                control_tools=NS(get_settings=lambda data: data.get("controlnet") or {}),
            ),
        )
        with (
            patch(
                "ruinedfooocus.features.models.component_profiles.snapshot_request",
                side_effect=lambda data, *a: data,
            ),
            patch(
                "ruinedfooocus.features.models.component_profiles.component_inventory",
                return_value={"rows": []},
            ),
            patch(
                "ruinedfooocus.features.prompting.local_model.capture_prompt_preparation",
                side_effect=lambda data, *a: data,
            ),
            patch(
                "ruinedfooocus.features.generation.sdcpp_pipeline.capture_image_request",
                side_effect=lambda data, *a, **kw: data,
            ),
        ):
            first = capture(data)
            for control in ({"type": "rembg"}, {"type": "upscale", "upscaler": "scale.pth"}):
                utility = capture(dict(data, controlnet=control))
                self.assertIs(utility["_sampling_captured"], False)
                self.assertNotIn(RECIPE_KEY, utility)
                self.assertNotIn(adapters.KEY, utility)
                self.assertNotIn("reference_adapter_image", utility)
            with patch(
                "ruinedfooocus.features.generation.image_preparation.capture_image_assets",
                side_effect=lambda data, *a, **kw: data,
            ):
                control_request = capture(dict(data, controlnet={"type": "control"}))
                self.assertNotIn("reference_adapter", control_request)
                self.assertNotIn("reference_adapter_image", control_request)
                self.assertNotIn(adapters.KEY, control_request)
            resumed = capture(data)
        for current in (first, resumed):
            self.assertIn(RECIPE_KEY, current)
            self.assertIn(adapters.KEY, current)
            self.assertIs(current["_sampling_captured"], True)
        self.assertEqual(data["reference_adapter"], "style")
        self.assertEqual(data["model_recipe_path"], str(recipe))


if __name__ == "__main__":
    unittest.main()
