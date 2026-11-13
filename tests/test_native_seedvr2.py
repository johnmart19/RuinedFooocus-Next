"""Pinned still-node calls and their real capture, API and contextual UI callers."""

import ast
import base64
from concurrent.futures import CancelledError
from contextlib import contextmanager, nullcontext
from io import BytesIO
import json
import pathlib
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import ruinedfooocus.app
from ruinedfooocus.features.generation.image_preparation import (
    KEY,
    capture_image_assets,
    prepare_image_assets,
    prepared_asset,
)
from ruinedfooocus.features.image_processing import native_seedvr2 as native
from test_image_preparation import Manager
from test_native_variant_defaults import NATIVE, SHAPES, native_backend, write_header
from test_utility_sampling_capture import function


def native_nodes():
    """Execute pinned CPU node bodies; replace only sampler/model/device services."""
    scope = dict(torch=torch, SEEDVR2_LATENT_CHANNELS=16, io=NS(NodeOutput=lambda *v: v))
    tree = ast.parse((NATIVE / "comfy_extras/nodes_seedvr.py").read_text(encoding="utf-8"))
    names = (
        "div_pad",
        "cut_videos",
        "_seedvr2_input_shorter_edge",
        "_seedvr2_pad",
        "_resolve_seedvr2_diffusion_model",
        "SeedVR2Preprocess",
        "SeedVR2PostProcessing",
        "SeedVR2Conditioning",
    )
    for node in tree.body:
        if getattr(node, "name", None) not in names:
            continue
        if isinstance(node, ast.ClassDef):
            node.bases = []
            node.body = [
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name != "define_schema"
            ]
            for item in node.body:
                item.returns = None
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                "pinned_seedvr2",
                "exec",
            ),
            scope,
        )
    scope.update(_ATTR_MISSING=object(), _SEEDVR2_INVALID_MODEL_MSG_PREFIX="bad model")
    return NS(**{name: scope[name] for name in names if name.startswith("SeedVR2")})


class NativeSeedVR2Tests(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch.dict(
                SHAPES,
                {
                    "SeedVR2": {
                        "blocks.31.mlp.all.proj_in_gate.weight": [2, 2],
                        "positive_conditioning": [2, 3],
                        "negative_conditioning": [2, 3],
                    },
                },
            )
        )
        self.enterContext(native_backend())
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for kind in ("diffusion_models", "checkpoints", "vae"):
            (self.root / kind).mkdir()
        self.model = write_header(
            self.root / "diffusion_models/arbitrary.safetensors", SHAPES["SeedVR2"]
        )
        self.codec = write_header(
            self.root / "vae/arbitrary.safetensors", {native.CODEC_KEY: [4, 1, 1, 1, 1]}
        )
        self.manager = Manager(self.root)
        self.image = Image.new("RGBA", (11, 7), (80, 100, 180, 128))

    def request(self, **values):
        return capture_image_assets(
            dict(
                input_image=self.image,
                seedvr2_model=str(self.model),
                seedvr2_vae=str(self.codec),
                seedvr2_scale=1,
                seed=17,
            )
            | values,
            self.manager,
            {"type": native.KIND},
            offline=True,
        )

    def test_capture_freezes_source_and_both_installed_roles_without_generation_resources(self):
        data = self.request(prompt="#!not a command", base_model_name="missing", loras=["unused"])
        self.image.putpixel((0, 0), (0, 0, 0, 0))
        self.assertEqual(data["input_image"].getpixel((0, 0)), (80, 100, 180, 128))
        self.assertEqual(
            [row.role for row in data[KEY].artifacts], ["seedvr2-model", "seedvr2-vae"]
        )
        self.assertEqual((data["custom_width"], data["custom_height"]), (11, 7))
        self.assertEqual(
            (data["prompt"], data["loras"], data["_sampling_captured"]), ("", [], False)
        )
        self.assertEqual(data["seed"], 17)
        self.assertEqual(native.model_choices(self.manager), [(self.model.name, str(self.model))])
        self.assertEqual(
            native.model_choices(self.manager, True), [(self.codec.name, str(self.codec))]
        )
        service = Mock()
        self.model.unlink()
        with self.assertRaisesRegex(ValueError, "Offline mode"):
            prepare_image_assets(data, check_cancelled=Mock(), service=service)
        service.submit.assert_not_called()

    def test_capture_rejects_wrong_roles_escape_nonfinite_and_oversized_output(self):
        outside = write_header(self.root / "outside.safetensors", SHAPES["SeedVR2"])
        for values in (
            dict(seedvr2_model=str(outside)),
            dict(seedvr2_model="../outside.safetensors"),
            dict(seedvr2_model=str(self.codec)),
            dict(seedvr2_vae=str(self.model)),
            dict(seedvr2_scale=float("nan")),
            dict(seedvr2_scale=True),
            dict(seedvr2_scale=4.01),
            dict(seedvr2_color="unknown"),
            dict(seed=-1),
            dict(seed=True),
            dict(seed=2**64),
            dict(inpaint_toggle=True),
            dict(input_image=None),
            dict(input_image=Image.new("L", (4096, 4096)), seedvr2_scale=4),
        ):
            with self.subTest(values=str(values)[:80]), self.assertRaises(ValueError):
                self.request(**values)

    def install_nodes(self):
        native_module = native_nodes()
        self.calls, self.held = [], set()
        self.loaded_model = NS(
            model=NS(
                model_config=type("SeedVR2", (), {})(),
                diffusion_model=NS(
                    positive_conditioning=torch.zeros(2, 3), negative_conditioning=torch.ones(2, 3)
                ),
            )
        )

        def encode(vae, pixels, *values):
            self.assertEqual(self.held, {"seedvr2-model", "seedvr2-vae"})
            self.calls.append(("encode", tuple(pixels.shape), values))
            self.padded = pixels
            return ({"samples": torch.zeros(1, 16, 1, pixels.shape[2] // 8, pixels.shape[3] // 8)},)

        def sample(model, seed, steps, cfg, sampler, scheduler, pos, neg, latent, **options):
            self.assertEqual(self.held, {"seedvr2-model", "seedvr2-vae"})
            self.calls.append(("sample", seed, steps, cfg, sampler, scheduler, options))
            self.assertEqual(tuple(pos[0][1]["condition"].shape[:3]), (1, 17, 1))
            return (latent,)

        def decode(vae, latent, *values):
            self.calls.append(("decode", values))
            return (self.padded.flatten(0, 1),)

        @contextmanager
        def lease(data, role, name, manager, category):
            with prepared_asset(data, role, name, manager, category) as path:
                self.held.add(role)
                try:
                    yield path
                finally:
                    self.held.remove(role)

        def resize(pixels, method, settings):
            self.assertEqual((method, settings["resize_type"]), ("lanczos", "scale by multiplier"))
            size = tuple(round(side * settings["multiplier"]) for side in pixels.shape[1:3])
            return (F.interpolate(pixels.movedim(-1, 1), size=size, mode="nearest").movedim(1, -1),)

        self.vae = NS(
            latent_channels=16,
            latent_dim=3,
            patcher=object(),
            first_stage_model=type("VideoAutoencoderKLWrapper", (), {})(),
            throw_exception_if_invalid=Mock(),
        )
        comfy = sys.modules["comfy"]
        self.load = Mock(return_value=self.loaded_model)
        self.unload = Mock()
        comfy.sd = NS(load_diffusion_model=self.load, VAE=Mock(return_value=self.vae))
        comfy.utils = NS(load_torch_file=Mock(return_value=({}, {})))
        comfy.model_management = NS(unload_model_and_clones=self.unload)
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "comfy.sd": comfy.sd,
                    "comfy.utils": comfy.utils,
                    "comfy.model_management": comfy.model_management,
                    "comfy_extras.nodes_seedvr": native_module,
                    "comfy_extras.nodes_post_processing": NS(
                        ResizeImageMaskNode=NS(execute=resize)
                    ),
                    "nodes": NS(
                        VAEEncodeTiled=lambda: NS(encode=encode),
                        VAEDecodeTiled=lambda: NS(decode=decode),
                        KSampler=lambda: NS(sample=sample),
                    ),
                },
            )
        )
        self.enterContext(
            patch("ruinedfooocus.features.generation.image_preparation.prepared_asset", lease)
        )

    def test_native_one_frame_recipe_preserves_complete_odd_source_alpha_precision_and_cache(self):
        self.install_nodes()
        data, owner = self.request(), NS()
        result = native.process(owner, data, self.manager, Mock())
        self.assertEqual((result.mode, result.size), ("RGBA", (11, 7)))
        self.assertEqual(result.getpixel((0, 0)), (80, 100, 180, 128))
        self.assertEqual(self.calls[0], ("encode", (1, 1, 16, 16, 3), (512, 128, 4096, 8)))
        self.assertEqual(self.calls[1], ("sample", 17, 1, 1.0, "euler", "simple", {"denoise": 1.0}))
        self.assertEqual(self.calls[2], ("decode", (512, 128, 4096, 8)))
        self.assertEqual(data["native_seedvr2_source"]["model_role"], "image_utility_model")
        self.assertNotIn(str(self.root), json.dumps(data["native_seedvr2_source"]))
        precise = dict(
            data, output_format="exr", output_bit_depth="32-bit float", output_color_space="linear"
        )
        pixels = native.process(owner, precise, self.manager, Mock())
        self.assertEqual((pixels.shape, pixels.dtype), ((7, 11, 4), np.dtype("float32")))
        self.load.assert_called_once()
        self.assertFalse(self.held)
        native.clear_model(owner)
        self.assertIsNone(owner._native_seedvr2)
        self.assertEqual(self.unload.call_count, 2)

    def test_cancellation_stops_before_sampling_and_releases_both_file_leases(self):
        self.install_nodes()
        owner = NS()
        cancelled = Mock(side_effect=[None, None, None, None, CancelledError("stop")])
        with self.assertRaises(CancelledError):
            native.process(owner, self.request(), self.manager, cancelled)
        self.assertEqual([row[0] for row in self.calls], ["encode"])
        self.assertFalse(self.held)
        native.clear_model(owner)

    def test_patterned_odd_edges_and_alpha_survive_real_native_postprocessing(self):
        self.install_nodes()
        for width, height in ((11, 7), (10, 7), (11, 8)):
            with self.subTest(size=(width, height)):
                pixels = np.zeros((height, width, 4), dtype=np.uint8)
                pixels[..., 3] = 255
                pixels[-1, :, :] = [255, 40, 80, 113]
                pixels[:, -1, :] = [20, 210, 70, 197]
                pixels[-1, -1, :] = [19, 23, 251, 61]
                data = self.request(input_image=Image.fromarray(pixels))
                result = native.process(NS(), data, self.manager, Mock())
                np.testing.assert_array_equal(np.asarray(result), pixels)
                source = data["native_seedvr2_source"]
                self.assertEqual(source["source_size"], [width, height])
                self.assertEqual(source["output_size"], [width, height])
                self.assertEqual(
                    source["native_reference_size"], [(width + 1) // 2 * 2, (height + 1) // 2 * 2]
                )
                self.assertEqual(source["padded_size"], [16, 16])

    def test_exact_target_capture_cannot_reframe_or_crop_the_still_utility(self):
        from ruinedfooocus.features.generation.size_plan import capture_size_plan, KEY as SIZE_KEY

        data = dict(
            input_image=self.image,
            seedvr2_scale=1,
            size_mode="target",
            custom_width=640,
            custom_height=512,
            resolution_auto=False,
            automatic_upscale="Off",
        )
        captured = capture_size_plan(data, None, controls={"type": native.KIND})
        self.assertNotIn(SIZE_KEY, captured)
        final = self.request(**captured)
        self.assertEqual((final["custom_width"], final["custom_height"]), self.image.size)
        self.assertIs(captured["input_image"], self.image)

    def test_replaced_capture_is_refused_before_native_loading(self):
        self.install_nodes()
        data = self.request()
        self.codec.write_bytes(self.codec.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "changed after submission"):
            native.process(NS(), data, self.manager, Mock())
        self.load.assert_not_called()
        self.assertFalse(self.held)

    def test_real_api_capture_owns_the_source_components_geometry_and_output_format(self):
        from ruinedfooocus.app import api_runtime as api

        raw = BytesIO()
        self.image.save(raw, format="PNG")
        payload = dict(
            controlnet={"type": native.KIND},
            input_image=base64.b64encode(raw.getvalue()).decode(),
            seedvr2_model=str(self.model),
            seedvr2_vae=str(self.codec),
            seedvr2_scale=1,
            seed=18,
            size="11x7",
            output_format="png",
            output_bit_depth="16-bit",
        )
        shared = NS(
            path_manager=self.manager,
            models=NS(offline=True),
            settings=NS(default_settings={}),
            state={},
        )
        worker = NS(
            prepare_image_data=Mock(side_effect=lambda data, **_: data),
            _process=Mock(return_value=[]),
        )
        with (
            patch.dict(
                sys.modules,
                {"ruinedfooocus.app.state": shared, "ruinedfooocus.app.async_worker": worker},
            ),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(ruinedfooocus.app, "async_worker", worker, create=True),
            patch.object(api, "submit", side_effect=lambda run: run),
        ):
            run = api.images(payload, {})
            self.assertEqual(run.resource_group, "image")
            run.prepare(object())
            data = worker.prepare_image_data.call_args.args[0]
            self.assertEqual(
                (data["seed"], data["output_bit_depth"], data["custom_width"]), (18, "16-bit", 11)
            )
            self.assertEqual(data["input_image"].mode, "RGBA")
            for invalid in (
                dict(size="10x6"),
                dict(n=2),
                dict(mask="not used"),
                dict(reference_images=["unused"]),
            ):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    api.images(payload | invalid, {})

    def test_public_json_and_multipart_route_without_generation_model_resolution(self):
        from fastapi.testclient import TestClient
        from ruinedfooocus.integrations.openai_api import create_apps
        from test_api_live_contracts import job

        output = self.root / "output.png"
        self.image.save(output)
        raw = output.read_bytes()
        backend = NS(
            images=Mock(side_effect=lambda *_: job(("result", [str(output)]))),
            resolve_model=Mock(side_effect=AssertionError("generation resolution")),
        )
        app, _ = create_apps(backend)
        fields = dict(seedvr2_model=str(self.model), seedvr2_vae=str(self.codec), seedvr2_scale=1)
        with TestClient(app) as client:
            json_response = client.post(
                "/images/generations",
                json=dict(
                    fields,
                    controlnet={"type": native.KIND},
                    input_image=base64.b64encode(raw).decode(),
                ),
            )
            multipart = client.post(
                "/images/edits",
                files={"image": ("source.png", raw, "image/png")},
                data=dict(fields, controlnet=json.dumps({"type": native.KIND})),
            )
        self.assertEqual(json_response.status_code, 200, json_response.text)
        self.assertEqual(multipart.status_code, 200, multipart.text)
        self.assertTrue(all(call.args[1] == {} for call in backend.images.call_args_list))
        backend.resolve_model.assert_not_called()

    def test_contextual_ui_registers_submitted_controls_and_only_shows_for_seedvr2(self):
        import gradio as gr
        from ruinedfooocus.features.image_processing.ui_native_seedvr2 import create_ui

        registered = {}
        shared = NS(
            path_manager=self.manager,
            settings=NS(default_settings={}),
            add_ctrl=lambda name, control: registered.setdefault(name, control),
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            gr.Blocks() as app,
        ):
            selection = gr.Dropdown(choices=[native.KIND], value=native.KIND)
            kind = gr.Textbox("")
            controls = create_ui(
                selection,
                kind,
                gr.Number(value=0, precision=0, visible=False),
                lambda selected, _: selected,
            )
        self.assertEqual(set(registered), set(native.FIELDS))
        self.assertEqual(controls["seedvr2_scale"].value, 4.0)
        self.assertEqual(len(app.fns), 3)

    def test_native_utility_mounts_accept_the_production_metadata_revision_number(self):
        import gradio as gr
        from ruinedfooocus.features.image_processing.ui_native_seedvr2 import create_ui
        from ruinedfooocus.features.image_processing.ui_native_vision import create_ui as vision_ui
        from ruinedfooocus.features.image_processing.ui_native_background import (
            create_ui as background_ui,
        )

        tree = ast.parse(
            (Path(__file__).resolve().parents[1] / "ruinedfooocus/ui/app.py").read_text(
                encoding="utf-8"
            )
        )
        assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "model_metadata_refresh"
                for target in node.targets
            )
        )
        self.assertEqual(ast.unparse(assignment.value.func), "gr.Number")
        shared = NS(path_manager=self.manager, settings=NS(default_settings={}), add_ctrl=Mock())
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            gr.Blocks(),
        ):
            selection = gr.Dropdown(choices=[native.KIND], value=native.KIND)
            kind, refresh, image = (
                gr.Textbox(""),
                gr.Number(value=0, precision=0, visible=False),
                gr.Image(type="pil"),
            )
            kind_for = lambda selected, _: selected
            create_ui(selection, kind, refresh, kind_for)
            vision_ui(selection, kind, image, refresh, kind_for)
            background_ui(selection, kind, refresh, kind_for)

    def test_actual_worker_png_caller_uses_restored_geometry_and_utility_metadata(self):
        from ruinedfooocus.features.image_processing.native_vision_pipeline import pipeline
        from ruinedfooocus.core.paths import resource_path
        from PIL.PngImagePlugin import PngInfo

        forbidden = Mock(side_effect=AssertionError("generation dependency accessed"))
        pipe = pipeline()

        def process(gen_data, callback):
            gen_data["native_seedvr2_source"] = dict(native="SeedVR2", output_size=[11, 7])
            return [Image.new("RGBA", (11, 7), (80, 100, 180, 128))]

        pipe.process = process
        pipe.load_base_model = forbidden
        controls = NS(get_settings=lambda data: data.get("controlnet", {}))
        shared = NS(
            models=NS(get_models_by_path=forbidden, get_model_base=forbidden),
            state={},
            performance_settings=NS(apply=forbidden),
            resolution_settings=object(),
            path_manager=NS(model_paths={"temp_outputs_path": self.root}),
            shared_cache={},
        )
        route = dict(
            state={"pipeline": None},
            shared=shared,
            controlnet=controls,
            native_vision_pipeline=NS(pipeline=lambda: pipe),
        )
        function("ruinedfooocus/features/generation/pipelines.py", "selected_pipeline", route)
        update = function("ruinedfooocus/features/generation/pipelines.py", "update", route)
        render = function(
            "ruinedfooocus/app/async_worker.py",
            "_process",
            dict(
                shared=shared,
                settings=NS(default_settings={}),
                control_tools=controls,
                check_interrupt=lambda _: None,
                process_metadata=forbidden,
                pipelines=NS(update=update),
                get_checkpoint_hashes=forbidden,
                process_prompt=forbidden,
                shift_attention=forbidden,
                parse_loras=forbidden,
                merge_loras=forbidden,
                TimeIt=lambda _: nullcontext(),
                resource_path=resource_path,
                random=random,
                Path=Path,
                pathlib=pathlib,
                Image=Image,
                PngInfo=PngInfo,
                json=json,
                generate_temp_filename=lambda **_: str(self.root / "render.png"),
            ),
        )
        data = self.request(task_id=1, task_type="tool_call", silent=True, index=(0, 1))
        result = render(data)
        self.assertEqual(result, [str(self.root / "render.png")])
        with Image.open(result[0]) as image:
            self.assertEqual((image.mode, image.size), ("RGBA", (11, 7)))
            metadata = json.loads(image.info["parameters"])
        self.assertEqual((metadata["width"], metadata["height"]), (11, 7))
        self.assertEqual(metadata["controlnet"], {"type": native.KIND})
        self.assertEqual(metadata["seedvr2_model"], self.model.name)
        self.assertNotIn(str(self.root), json.dumps(metadata))
        self.assertNotIn("base_model_name", metadata)
        forbidden.assert_not_called()

    def test_ui_submission_bypasses_presets_prompt_expansion_and_input_upscale(self):
        from ruinedfooocus.features.generation.ui_request import build_ui_request
        import ruinedfooocus.features.image_editing as editing

        control = NS(get_settings=lambda _: {"type": native.KIND})
        values = dict(
            input_image=self.image,
            image_number=1,
            preset_selection="does-not-exist.png",
            automatic_upscale="2× General (RealESRGAN)",
            automatic_upscale_target="Input image only",
            cn_selection="Restore still image (SeedVR2)",
            prompt="unused",
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.features.image_editing.controlnet": control}),
            patch.object(editing, "controlnet", control, create=True),
        ):
            result = build_ui_request(values, models=object(), settings={})
        self.assertIs(result["input_image"], self.image)
        worker = NS(add_task=Mock(return_value=17))
        append = function(
            "ruinedfooocus/ui/app.py",
            "append_work",
            dict(
                worker=worker,
                ui_controlnet=NS(tool_kind=lambda *_: native.KIND),
                use_random_prompt=Mock(side_effect=AssertionError("prompt expansion")),
            ),
        )
        self.assertEqual(append(result), 17)
        submitted = worker.add_task.call_args.args[0]
        self.assertEqual((submitted["prompt"], submitted["image_total"]), ("", 1))


if __name__ == "__main__":
    unittest.main()
