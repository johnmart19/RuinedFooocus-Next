"""Native instruct conditioning and required input contracts without model weights."""

import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch

from ruinedfooocus.features.image_editing.image_geometry import (
    prepare_reference,
    restore_reference_frame,
)
from ruinedfooocus.features.image_editing.native_conditioning import (
    INSTRUCT_FAMILIES,
    instruct_conditioning,
    is_instruct_pix2pix,
    require_instruct_edit,
)
from ruinedfooocus.features.models.image_components import inventory
from ruinedfooocus.features.image_editing.sd4x_conditioning import require_sd4x_source
from ruinedfooocus.features.image_editing.image_guidance import normalize_reference_images
from test_model_cache import method
from test_native_model_coverage import route_info
from test_powerup_family_controls import functions
from test_api_image_edits import queued_data

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))
PIPELINE = "ruinedfooocus/features/generation/image_pipeline.py"


def native_instruct_node():
    source = NATIVE / "comfy_extras/nodes_ip2p.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "InstructPixToPixConditioning"
    )
    node.bases = []
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "execute"]
    node.body[0].returns = None
    scope = dict(torch=torch, io=SimpleNamespace(NodeOutput=lambda *values: values))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return SimpleNamespace(InstructPixToPixConditioning=scope[node.name])


def model(name):
    return SimpleNamespace(model_config=type(name, (), {})())


class InstructPix2PixTests(unittest.TestCase):
    def test_native_instruct_label_corrects_stale_sampling_family_without_rewriting_publisher_cache(
        self,
    ):
        lookup = method(
            "ruinedfooocus/features/models/model_handler.py",
            "get_models_by_path",
            dict(json=json, Path=Path),
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "arbitrary.safetensors"
            path.write_bytes(b"identity-only source")
            cached = root / "arbitrary.json"
            publisher = dict(baseModel="SDXL 1.0", name="Publisher title", modelId=42)
            cached.write_text(json.dumps(publisher))
            before = cached.read_bytes()
            for family in INSTRUCT_FAMILIES:
                owner = SimpleNamespace(
                    cache_paths={"checkpoints": root},
                    offline=True,
                    detect_checkpoint_base=lambda path: family,
                )
                self.assertEqual(
                    lookup(owner, "checkpoints", path), publisher | dict(baseModel=family)
                )
                self.assertEqual(cached.read_bytes(), before)

    def process_body(self):
        tree = ast.parse((ROOT / PIPELINE).read_text(encoding="utf-8"))
        return next(
            n.body for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process"
        )

    def test_runtime_and_inventory_use_the_matching_native_sd15_or_sdxl_roles(self):
        get_info = method(
            PIPELINE, "get_clip_and_vae", dict(settings=SimpleNamespace(default_settings={}))
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "arbitrary.safetensors"
            header = json.dumps({"weight": {"shape": [1]}}).encode()
            checkpoint.write_bytes(len(header).to_bytes(8, "little") + header)
            paths = SimpleNamespace(
                paths={"path_clip": [str(root)], "path_vae": str(root)},
                get_abspath=Path,
                DOWNLOADABLE_FILES={},
            )
            for family, keys in (
                ("SD15_instructpix2pix", ["clip_l", "vae_sd"]),
                ("SDXL_instructpix2pix", ["clip_l", "clip_g", "vae_sdxl"]),
            ):
                unet = SimpleNamespace(
                    model=type(family, (), {"model_config": SimpleNamespace(unet_config={})})()
                )
                info = get_info(SimpleNamespace(known_model_info=route_info()), unet)
                self.assertEqual(info["clip_type"], "STABLE_DIFFUSION")
                self.assertEqual(info["clip_keys"], keys[:-1])
                self.assertEqual(info["vae_key"], keys[-1])
                state = inventory(checkpoint, family, {}, paths)
                self.assertEqual([row["key"] for row in state["rows"]], keys)

    def test_ui_and_api_select_required_native_edit_without_img2img_or_mask_substitution(self):
        scope, models = functions()
        scope["controlnet"].controlnet_models["edit"] = None
        scope["controlnet"].cn_options["Native image edit"] = {"type": "edit"}
        for family in INSTRUCT_FAMILIES:
            models.detect_checkpoint_base.return_value = family
            features = scope["checkpoint_features"]("arbitrary")
            self.assertTrue(features["native_edit"])
            self.assertTrue(features["instruct"])
            self.assertFalse(features["sdxl"])
            self.assertFalse(features["inpaint"])
            self.assertIn("Edit", scope["control_types"](features))
            self.assertNotIn("Img2img", scope["control_types"](features))
            self.assertNotIn("Img2Img", dict(scope["powerup_choices"](**features)).values())
            self.assertEqual(
                scope["fallback_selection"](features, "Img2Img", False),
                ("Native image edit", False),
            )
            update = scope["input_context"](True, "Native image edit", "Edit", "arbitrary")
            self.assertEqual(update[2], {"visible": False, "value": False})
            self.assertEqual(queued_data(family)["controlnet"]["type"], "edit")

    def test_real_process_preflight_requires_source_and_keeps_native_edit_selected(self):
        body = self.process_body()

        def assignment(name):
            return next(
                i
                for i, n in enumerate(body)
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
            )

        code = compile(
            ast.Module(body=body[assignment("controlnet") : assignment("device")], type_ignores=[]),
            "native_instruct_preflight",
            "exec",
        )
        for family in INSTRUCT_FAMILIES:
            source = object()

            def run(kind, image):
                data = dict(input_image=image, inpaint_toggle=False)
                scope = dict(
                    gen_data=data,
                    self=SimpleNamespace(
                        xl_base_patched=SimpleNamespace(unet=SimpleNamespace(model=model(family)))
                    ),
                    instruct_model=True,
                    sd4x_model=False,
                    image_edit=False,
                    hidream_edit=False,
                    architecture=family,
                    qwen_local_edit=False,
                    pid_model=False,
                    refiner_model=False,
                    guidance={"moss_image_edit_format": "Disabled"},
                    normalize_reference_images=normalize_reference_images,
                    require_instruct_edit=require_instruct_edit,
                    require_sd4x_source=require_sd4x_source,
                    control_tools=SimpleNamespace(
                        get_settings=lambda data: {"type": kind},
                        controlnet_models={"edit": None, "img2img": None},
                    ),
                    SDXL=type("SDXL", (), {}),
                )
                exec(code, scope)
                self.assertIs(data["input_image"], image)
                return scope

            with self.assertRaisesRegex(ValueError, "Add a source image"):
                run("edit", None)
            with self.assertRaisesRegex(ValueError, "requires Native image edit"):
                run("img2img", source)
            result = run("edit", source)
            self.assertEqual(result["control_type"], "edit")
            self.assertIs(result["input_image"], source)

    def test_actual_process_routes_native_source_conditioning_and_zero_latent_to_sampler(self):
        body = self.process_body()
        preprocess = next(
            n
            for n in ast.walk(ast.Module(body=body, type_ignores=[]))
            if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "input_image"
        )
        instruct = next(
            n
            for n in body
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Name)
            and n.test.id == "instruct_model"
        )
        target = next(
            n
            for n in body
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "edit_latent is not None and (not img2img_mode)"
        )
        sample = next(
            n
            for n in ast.walk(ast.Module(body=body, type_ignores=[]))
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == "comfy.sample.sample"
        )
        source = Image.new("RGB", (117, 101), (90, 120, 210))
        for scheduled in (False, True):
            encoded = []

            def encode(pixels):
                encoded.append(pixels.clone())
                return torch.ones(1, 4, pixels.shape[1] // 8, pixels.shape[2] // 8)

            request = dict(width=192, height=128, inpaint_toggle=False, steps=12)
            scope = dict(
                self=SimpleNamespace(
                    xl_base_patched=SimpleNamespace(vae=SimpleNamespace(encode=encode))
                ),
                gen_data=request,
                input_image=source,
                prepare_reference=prepare_reference,
                image_edit=False,
                hidream_edit=False,
                instruct_model=True,
                sd4x_model=False,
                pid_model=False,
                refiner_model=False,
                architecture="SD15_instructpix2pix",
                guidance={"image_resize_mode": "preserve"},
                control_type="edit",
                input_images=0,
                positive_conditioning=[["plain", {}]],
                negative_conditioning=[["negative", {}]],
                switched_prompt=[
                    ["early", {"start_percent": 0.0}],
                    ["late", {"start_percent": 0.5}],
                ]
                if scheduled
                else [],
                edit_latent=None,
                img2img_mode=False,
            )
            with patch.dict("sys.modules", {"comfy_extras.nodes_ip2p": native_instruct_node()}):
                exec(
                    compile(
                        ast.Module(body=[preprocess, instruct, target], type_ignores=[]),
                        PIPELINE,
                        "exec",
                    ),
                    scope,
                )
            sampler = Mock(return_value="sampled")
            scope.update(
                comfy=SimpleNamespace(
                    sample=SimpleNamespace(sample=sampler),
                    utils=SimpleNamespace(PROGRESS_BAR_ENABLED=True),
                ),
                positive_cond=scope["switched_prompt"]
                if scheduled
                else scope["positive_conditioning"],
                latent_image=scope["latent"]["samples"],
                cfg=7.0,
                sampling_model="native-IP2P",
                noise="native-noise",
                sampler_name="euler",
                scheduler="simple",
                noise_mask=None,
                fixed_sigmas=None,
                callback_function=None,
                seed=7,
            )
            exec(compile(ast.Module(body=[sample], type_ignores=[]), PIPELINE, "exec"), scope)
            call = sampler.call_args.kwargs
            self.assertTrue((call["latent_image"] == 0).all())
            self.assertEqual(call["cfg"], 7.0)
            self.assertIsNone(call["denoise"])
            self.assertIn("concat_latent_image", call["negative"][0][1])
            self.assertTrue(all("concat_latent_image" in entry[1] for entry in call["positive"]))
            self.assertEqual(len(call["positive"]), 2 if scheduled else 1)
            self.assertEqual(request["_image_requested_size"], (192, 128))
            self.assertEqual(
                tuple(restore_reference_frame(encoded[0], scope["reference_frame"]).shape),
                (1, 101, 117, 3),
            )

    def test_native_config_identity_requires_edit_and_source_without_accepting_masked_mode(self):
        for family in INSTRUCT_FAMILIES:
            native = model(family)
            self.assertTrue(is_instruct_pix2pix(native))
            for kind in (None, "img2img", "canny"):
                with self.assertRaisesRegex(ValueError, "requires Native image edit"):
                    require_instruct_edit(native, kind, object())
            with self.assertRaisesRegex(ValueError, "Add a source image"):
                require_instruct_edit(native, "edit", None)
            with self.assertRaisesRegex(ValueError, "Masked inpainting is not available"):
                require_instruct_edit(native, "edit", object(), inpainting=True)
            require_instruct_edit(native, "edit", object())
        for family in ("SD15", "SDXL", "SDXLRefiner", "FluxInpaint"):
            self.assertFalse(is_instruct_pix2pix(model(family)))
            require_instruct_edit(model(family), None, None)

    def test_pinned_node_adds_same_source_to_positive_and_negative_and_zero_sampler_latent(self):
        pixels = torch.full((1, 32, 48, 3), 0.75)
        source_latent = torch.full((1, 4, 4, 6), 0.25)
        positive = [["early", {"start_percent": 0.0}], ["late", {"start_percent": 0.5}]]
        negative = [["negative", {"pooled_output": "native-pooled"}]]
        vae = SimpleNamespace(encode=lambda value: source_latent)
        with patch.dict("sys.modules", {"comfy_extras.nodes_ip2p": native_instruct_node()}):
            pos, neg, latent = instruct_conditioning(positive, negative, vae, pixels)
        self.assertEqual(tuple(latent["samples"].shape), tuple(source_latent.shape))
        self.assertTrue((latent["samples"] == 0).all())
        for original, conditioned in zip(positive + negative, pos + neg):
            self.assertIs(conditioned[1]["concat_latent_image"], source_latent)
            self.assertNotIn("concat_latent_image", original[1])
            for key, value in original[1].items():
                self.assertEqual(conditioned[1][key], value)
        self.assertNotIn("noise_mask", latent)

    def test_existing_framing_prevents_native_grid_crop_from_losing_source_edges(self):
        image = Image.new("RGB", (117, 101), (10, 80, 220))
        pixels, frame = prepare_reference(image, (192, 128), "preserve")
        encoded = []

        def encode(value):
            encoded.append(value.clone())
            return torch.zeros((1, 4, value.shape[1] // 8, value.shape[2] // 8))

        with patch.dict("sys.modules", {"comfy_extras.nodes_ip2p": native_instruct_node()}):
            _, _, latent = instruct_conditioning([], [], SimpleNamespace(encode=encode), pixels)
        torch.testing.assert_close(encoded[0], pixels)
        self.assertEqual(tuple(restore_reference_frame(encoded[0], frame).shape), (1, 101, 117, 3))
        self.assertEqual(
            tuple(latent["samples"].shape[-2:]),
            (frame["canvas_height"] // 8, frame["canvas_width"] // 8),
        )

    def test_second_source_cannot_mutate_the_first_requests_or_cached_text_conditions(self):
        positive = [["same words", {"weight": 1}]]
        negative = [["negative", {}]]
        results = []
        with patch.dict("sys.modules", {"comfy_extras.nodes_ip2p": native_instruct_node()}):
            for value in (0.25, 0.75):
                results.append(
                    instruct_conditioning(
                        positive,
                        negative,
                        SimpleNamespace(encode=lambda pixels: pixels.movedim(-1, 1)),
                        torch.full((1, 16, 16, 3), value),
                    )
                )
        self.assertTrue((results[0][0][0][1]["concat_latent_image"] == 0.25).all())
        self.assertTrue((results[1][0][0][1]["concat_latent_image"] == 0.75).all())
        self.assertEqual(positive, [["same words", {"weight": 1}]])
        self.assertEqual(negative, [["negative", {}]])


if __name__ == "__main__":
    unittest.main()
