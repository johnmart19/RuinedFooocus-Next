from ruinedfooocus.core.paths import asset_path, resource_path

"""Exercise native image-edit routing without loading diffusion weights."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image
import torch
from ruinedfooocus.core.settings_validation import validate_settings, MOSS_EDIT_FORMATS
from ruinedfooocus.features.image_editing.image_guidance import reference_image_map
from ruinedfooocus.features.image_editing.image_geometry import prepare_reference
from ruinedfooocus.features.generation.size_plan import SizePlan

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))


class ImageEditTests(unittest.TestCase):
    def test_moss_edit_settings_match_native_formats_and_strength_bounds(self):
        tree = ast.parse((NATIVE / "comfy_extras/nodes_moss_image.py").read_text(encoding="utf-8"))
        formats = next(
            n.value
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "EDIT_FORMATS" for t in n.targets)
        )
        self.assertEqual(set(MOSS_EDIT_FORMATS) - {"Disabled"}, set(ast.literal_eval(formats)))
        for value in (0, 1.0, 100):
            self.assertEqual(
                validate_settings({"moss_image_edit_strength": value})["moss_image_edit_strength"],
                value,
            )
        for value in (-1, 101, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate_settings({"moss_image_edit_strength": value})
        with self.assertRaises(ValueError):
            validate_settings({"moss_image_edit_format": "unknown"})

    def encode(
        self, architecture, moss_format="Disabled", references=None, mode="crop", control=None
    ):
        tree = ast.parse(
            (
                Path(__file__).resolve().parents[1]
                / "ruinedfooocus/features/generation/image_pipeline.py"
            ).read_text()
        )
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "pipeline")
        process = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process"
        )
        branch = next(
            n
            for n in ast.walk(process)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "image_edit and input_images > 0"
        )
        function = ast.parse("def encode(self): pass").body[0]
        # input_image is a local of process(); the edit branch may rebind it.
        function.body = (
            ast.parse(
                "input_images = 1; edit_latent = None; input_image = INPUT_IMAGE; controlnet = INPUT_CONTROL"
            ).body
            + branch.body
            + ast.parse("return edit_latent, img2img_mode, input_images, controlnet").body
        )
        native_latent = {"samples": torch.zeros(1, 128, 24, 32)}
        mage = Mock()
        mage.execute.return_value = SimpleNamespace(result=("positive", "negative", native_latent))
        qwen = Mock()
        qwen.execute.side_effect = [("positive",), ("negative",)]
        moss = Mock()
        moss.execute.side_effect = [("positive",), ("negative",)]
        model = type(architecture, (), {})()
        owner = SimpleNamespace(
            xl_base_patched=SimpleNamespace(
                unet=SimpleNamespace(model=model), clip="clip", vae="vae"
            ),
            xl_base=SimpleNamespace(vae="vae"),
        )
        image = Image.new("RGBA", (64, 112))
        pixels, frame = prepare_reference(image, (512, 384), mode)
        namespace = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            callback=None,
            joy_reference_edit=False,
            clean_prompt_cond_caches=lambda: {"+": {}, "-": {}},
            warm_patched_text_encoder=lambda clip: False,
            TextEncodeMageFlowEdit=lambda: mage,
            TextEncodeFibo2=lambda: mage,
            TextEncodeQwenImageEdit=lambda: qwen,
            TextEncodeMOSSImageEdit=lambda: moss,
            np=np,
            torch=torch,
            positive_prompt="edit",
            negative_prompt="avoid",
            moss_edit_format=moss_format,
            moss_edit_strength=1.5,
            INPUT_IMAGE=pixels,
            edit_latent=None,
            img2img_mode=bool(control and control.get("type") == "img2img"),
            INPUT_CONTROL=control or {},
            input_images=1,
            gen_data={
                "input_image": image,
                "width": frame["canvas_width"],
                "height": frame["canvas_height"],
            },
        )
        namespace.update(
            additional_references=references or [],
            reference_image_map=reference_image_map,
            reference_tensor=lambda image, *size: torch.zeros(1, image.height, image.width, 3),
        )
        self.gen_data = namespace["gen_data"]
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                "edit_route",
                "exec",
            ),
            namespace,
        )
        result = namespace["encode"](owner)
        self.control = result[3]
        return result[:3], mage, qwen, moss, native_latent

    def test_mage_keeps_native_target_and_encodes_rgb_at_requested_size(self):
        result, mage, _, _, latent = self.encode("MageFlow")
        self.assertIs(result[0], latent)
        self.assertEqual(result[1:], (False, 0))
        args = mage.execute.call_args.kwargs
        self.assertEqual((args["width"], args["height"]), (512, 384))
        self.assertEqual(args["images"]["image_1"].shape[-1], 3)

    def test_fibo_keeps_native_target_and_forwards_all_edit_references(self):
        reference = Image.new("RGB", (32, 24))
        result, fibo, _, _, latent = self.encode("Fibo2", references=[reference])
        self.assertIs(result[0], latent)
        self.assertEqual(result[1:], (False, 0))
        args = fibo.execute.call_args.kwargs
        self.assertEqual((args["width"], args["height"]), (512, 384))
        self.assertEqual(args["prompt"], "edit")
        self.assertEqual(args["negative_prompt"], "avoid")
        self.assertEqual(args["vae"], "vae")
        self.assertEqual(list(args["images"]), ["image_1", "image_2"])
        self.assertEqual(args["images"]["image_1"].shape, (1, 384, 512, 3))
        self.assertEqual(args["images"]["image_2"].shape, (1, 24, 32, 3))
        native = ast.parse((NATIVE / "comfy_extras/nodes_fibo2.py").read_text(encoding="utf-8"))
        execute = next(
            node
            for node in ast.walk(native)
            if isinstance(node, ast.FunctionDef) and node.name == "execute"
        )
        self.assertTrue(set(args).issubset({arg.arg for arg in execute.args.args}))

    def test_edit_capabilities_use_upstream_options_for_all_native_families(self):
        tree = ast.parse(
            (
                Path(__file__).resolve().parents[1]
                / "ruinedfooocus/features/generation/image_pipeline.py"
            ).read_text(encoding="utf-8")
        )
        info = next(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "known_model_info"
                for target in node.targets
            )
        )
        for key, value in zip(info.keys, info.values):
            if key.value in ("Fibo2", "MageFlow", "QwenImage"):
                fields = dict(zip((field.value for field in value.keys), value.values))
                self.assertEqual(ast.literal_eval(fields["options"]), {"has_image_edit": True})
                self.assertNotIn("flags", fields)

    def test_explicit_img2img_validates_denoise_for_native_and_ordinary_models(self):
        tree = ast.parse(
            (
                Path(__file__).resolve().parents[1]
                / "ruinedfooocus/features/generation/image_pipeline.py"
            ).read_text(encoding="utf-8")
        )
        process = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "process"
        )
        validation = next(
            node
            for node in process.body
            if isinstance(node, ast.If)
            and "control_type in control_tools.controlnet_models" in ast.unparse(node.test)
        )
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            control_type="img2img",
            input_image=object(),
            controlnet={"strength": 2},
            gen_data={},
            control_tools=SimpleNamespace(controlnet_models={"img2img": None}),
        )
        code = compile(ast.Module(body=[validation], type_ignores=[]), "edit_preflight", "exec")
        for native in (True, False):
            with self.assertRaisesRegex(ValueError, "denoise"):
                exec(code, scope | {"image_edit": native})

    def test_native_img2img_keeps_the_denoise_control_and_source_aspect(self):
        for family in ("MageFlow", "Fibo2", "QwenImage", "MOSSImage"):
            result, mage, qwen, moss, _ = self.encode(
                family, "ai-toolkit", mode="preserve", control={"type": "img2img", "denoise": 0.25}
            )
            self.assertTrue(result[1])
            self.assertEqual(self.control, {"type": "img2img", "denoise": 0.25})
            self.assertEqual((self.gen_data["width"], self.gen_data["height"]), (64, 112))
            node = (
                mage if family in ("MageFlow", "Fibo2") else qwen if family == "QwenImage" else moss
            )
            call = node.execute.call_args.kwargs
            image = call["image"] if family == "QwenImage" else call["images"]["image_1"]
            self.assertEqual(tuple(image.shape), (1, 112, 64, 3))

    def test_native_img2img_samples_the_source_latent_instead_of_the_empty_edit_target(self):
        tree = ast.parse(
            (
                Path(__file__).resolve().parents[1]
                / "ruinedfooocus/features/generation/image_pipeline.py"
            ).read_text()
        )
        process = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process"
        )
        branch = next(
            n
            for n in process.body
            if isinstance(n, ast.If) and ast.unparse(n.test).startswith("edit_latent is not None")
        )
        source_latent = {"samples": torch.zeros(1, 4, 7, 4)}
        edit_target = {"samples": torch.ones(1, 4, 7, 4)}
        encoder = Mock(return_value=[source_latent])
        owner = SimpleNamespace(
            xl_base_patched=SimpleNamespace(
                vae=SimpleNamespace(spacial_compression_decode=lambda: 16)
            )
        )
        pixels = torch.zeros(1, 112, 64, 3)
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            self=owner,
            edit_latent=edit_target,
            img2img_mode=True,
            input_image=pixels,
            controlnet={"type": "img2img", "denoise": 0.25},
            gen_data={},
            VAEEncode=lambda: SimpleNamespace(encode=encoder),
        )
        code = compile(ast.Module(body=[branch], type_ignores=[]), "edit_sampling", "exec")
        exec(code, scope)
        self.assertIs(scope["latent"], source_latent)
        self.assertEqual(scope["denoise"], 0.25)
        self.assertIs(encoder.call_args.kwargs["pixels"], pixels)
        encoder.reset_mock()
        scope["img2img_mode"] = False
        exec(code, scope)
        self.assertIs(scope["latent"], edit_target)
        self.assertIsNone(scope["denoise"])
        encoder.assert_not_called()

    def test_repeated_preserved_frames_do_not_shrink_the_batch_pixel_budget(self):
        tree = ast.parse(
            (
                Path(__file__).resolve().parents[1]
                / "ruinedfooocus/features/generation/image_pipeline.py"
            ).read_text()
        )
        process = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process"
        )
        preprocess = next(
            n
            for n in ast.walk(process)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "input_image"
        )
        original = Image.new("RGB", (928, 1664))
        data = {"width": 1152, "height": 896}
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            gen_data=data,
            image_edit=True,
            ordered_recipe=None,
            pid_model=False,
            sd4x_model=False,
            refiner_model=False,
            instruct_model=False,
            size_plan=None,
            SizePlan=SizePlan,
            hidream_edit=False,
            control_type="img2img",
            architecture="QwenImage",
            prepare_reference=prepare_reference,
            guidance={"image_resize_mode": "preserve"},
        )
        code = compile(ast.Module(body=[preprocess], type_ignores=[]), "reference_batch", "exec")
        frames = []
        for _ in range(3):
            scope["input_image"] = original
            exec(code, scope)
            frame = scope["reference_frame"]
            frames.append(dict(frame))
            # The output stage records the unpadded result for PNG metadata.
            data.update(width=frame["width"], height=frame["height"])
        self.assertEqual(frames, [frames[0]] * 3)
        self.assertEqual(data["_image_requested_size"], (1152, 896))

    def test_qwen_reference_does_not_enable_img2img_or_loopback(self):
        result, _, qwen, _, _ = self.encode("QwenImage")
        self.assertEqual(result, (None, False, 0))
        self.assertEqual(qwen.execute.call_count, 2)
        self.assertEqual((self.gen_data["width"], self.gen_data["height"]), (512, 384))
        self.assertEqual(qwen.execute.call_args.kwargs["image"].shape, (1, 384, 512, 3))

    def test_native_mage_and_moss_receive_additional_images_without_sampling(self):
        reference = Image.new("RGB", (32, 24))
        _, mage, _, moss, _ = self.encode("MageFlow", references=[reference, reference])
        self.assertEqual(
            list(mage.execute.call_args.kwargs["images"]), ["image_1", "image_2", "image_3"]
        )
        _, _, _, moss, _ = self.encode("MOSSImage", "ai-toolkit", references=[reference])
        for call in moss.execute.call_args_list:
            self.assertEqual(list(call.kwargs["images"]), ["image_1", "image_2"])
            self.assertEqual(call.kwargs["images"]["image_2"].shape, (1, 24, 32, 3))

    def test_moss_passes_explicit_native_format_to_both_prompts(self):
        for edit_format in ("ai-toolkit", "ai-toolkit (kv_cache off)", "Moss-Image-Edit"):
            result, _, _, moss, _ = self.encode("MOSSImage", edit_format)
            self.assertEqual(result, (None, False, 0))
            self.assertEqual(moss.execute.call_count, 2)
            for call in moss.execute.call_args_list:
                self.assertEqual(call.kwargs["edit_format"], edit_format)
                self.assertEqual(call.kwargs["image_strength"], 1.5)
                self.assertEqual(call.kwargs["vae"], "vae")

    def test_mage_target_survives_latent_initialization(self):
        tree = ast.parse(
            (
                Path(__file__).resolve().parents[1]
                / "ruinedfooocus/features/generation/image_pipeline.py"
            ).read_text()
        )
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "pipeline")
        process = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process"
        )
        branch = next(
            n
            for n in process.body
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "edit_latent is not None and (not img2img_mode)"
        )
        function = ast.parse("def initialize(edit_latent, img2img_mode=False): pass").body[0]
        function.body = [branch] + ast.parse("return latent, denoise").body
        namespace = {
            "asset_path": asset_path,
            "resource_path": resource_path,
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                "target_latent",
                "exec",
            ),
            namespace,
        )
        latent = {"samples": torch.zeros(1, 128, 24, 32)}
        result, denoise = namespace["initialize"](latent)
        self.assertIs(result, latent)
        self.assertIsNone(denoise)


if __name__ == "__main__":
    unittest.main()
