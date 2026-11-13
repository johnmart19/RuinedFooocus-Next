"""Native curve/LUT and GLSL call-body contracts; never initialize EGL/GLES."""

import asyncio
import base64
import builtins
from concurrent.futures import CancelledError
from contextlib import nullcontext
import hashlib
import importlib.util
from io import BytesIO
import json
import pathlib
from pathlib import Path
import random
import re
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import gradio as gr
from gradio.state_holder import SessionState
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import torch

import ruinedfooocus.app
from ruinedfooocus.app import api_runtime
from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.image_processing import image_shaders as shaders
from ruinedfooocus.features.image_processing import image_effect_cpu
from ruinedfooocus.features.image_processing import still_image_utilities as dispatch
from ruinedfooocus.features.image_processing import ui_image_shaders as ui
from ruinedfooocus.integrations.openai_api import create_apps
from test_api_live_contracts import job
from test_depth_utilities import native_functions
from test_native_model_coverage import NATIVE, native_method
from test_utility_sampling_capture import function


class NodeOutput(tuple):
    def __new__(cls, *values, ui=None):
        result = super().__new__(cls, values)
        result.ui = ui or {}
        return result

    @property
    def result(self):
        return self


class ImageShaders(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        # This source module contains only native NumPy curve interpolation.
        # nodes_glsl.py is read as an AST; its top-level GL imports never run.
        source = NATIVE / "comfy_api/latest/_input/curve_types.py"
        spec = importlib.util.spec_from_file_location("owned_native_curve_types", source)
        self.curve_types = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.curve_types)
        self.calls = []
        scope = dict(
            torch=torch,
            np=np,
            io=NS(NodeOutput=NodeOutput, Autogrow=NS(Type=dict)),
            CurveInput=self.curve_types.CurveInput,
            SizeModeInput=dict,
            MAX_OUTPUTS=4,
            _render_shader_batch=self.render_boundary,
        )
        self.editor = native_method("comfy_extras/nodes_curve.py", "CurveEditor", "execute", scope)
        self.shader = native_method("comfy_extras/nodes_glsl.py", "GLSLShader", "execute", scope)
        self.native_graph_preview = Mock(
            side_effect=AssertionError("native graph preview must not save duplicate source files")
        )
        self.shader._build_ui_output = self.native_graph_preview
        cpu_scope = dict(torch=torch, io=NS(NodeOutput=NodeOutput))
        native_functions("node_helpers.py", {"image_alpha_fix"}, cpu_scope)
        cpu_scope["node_helpers"] = NS(image_alpha_fix=cpu_scope["image_alpha_fix"])
        blend = native_method(
            "comfy_extras/nodes_post_processing.py", "Blend", "execute", cpu_scope
        )
        mode = native_method(
            "comfy_extras/nodes_post_processing.py", "Blend", "blend_mode", cpu_scope
        )
        blend.blend_mode = mode.__dict__["blend_mode"]
        invert = native_method("nodes.py", "ImageInvert", "invert", cpu_scope)
        self.cpu_policy = shaders.cpu_mode
        self.gl_presence = shaders.gl_dependencies_present
        # Existing native-call-body tests explicitly exercise an accelerated
        # request using a renderer tracer, without touching actual GPU/GL code.
        self.enterContext(patch.object(shaders, "cpu_mode", return_value=False))
        self.enterContext(patch.object(shaders, "gl_dependencies_present", return_value=True))
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "comfy_extras.nodes_curve": NS(CurveEditor=self.editor),
                    "comfy_extras.nodes_glsl": NS(GLSLShader=self.shader),
                    "comfy.model_management": NS(unload_model_and_clones=Mock()),
                    "comfy_extras.nodes_post_processing": NS(Blend=blend),
                    "nodes": NS(ImageInvert=invert),
                },
            )
        )
        self.source = Image.fromarray(
            np.arange(4 * 5 * 4, dtype=np.uint8).reshape(4, 5, 4) * 3, "RGBA"
        )
        self.manager = NS(get_model_folders=Mock(side_effect=AssertionError("no model folder")))

    def render_boundary(self, program, width, height, batches, floats, ints, bools, curves):
        # This is a traced render boundary, not a shader evaluator. Pixel values
        # intentionally do not represent any effect's rendered appearance.
        self.calls.append(
            dict(
                program=program,
                size=(width, height),
                batches=batches,
                floats=floats,
                ints=ints,
                bools=bools,
                curves=curves,
            )
        )
        return [
            [np.full((height, width, 4), value, dtype=np.float32) for value in (0.25, 0.5, 0.75, 1)]
            for batch in batches
        ]

    def capture(self, kind, **options):
        data = dict(
            input_image=self.source,
            controlnet={"type": kind},
            prompt="#!untrusted command",
            image_engine="sdcpp-vulkan",
            **options,
        )
        return assets.capture_image_assets(data, self.manager, data["controlnet"], offline=True)

    def test_curveeditor_native_luts_reach_native_glsl_consumer_in_channel_order(self):
        data = self.capture(
            "image-curves",
            curve_master=[[0, 1], [1, 0]],
            curve_red=[[0, 0], [0.5, 0.8], [1, 1]],
            curve_interpolation="linear",
        )
        result = dispatch.process(NS(), data, self.manager, Mock())[0]
        call = self.calls[-1]
        self.assertEqual(
            (call["size"], len(call["batches"]), len(call["batches"][0])), ((5, 4), 1, 1)
        )
        self.assertEqual(call["batches"][0][0].shape, (4, 5, 3))
        self.assertEqual(len(call["curves"]), 4)
        self.assertTrue(
            all(lut.shape == (256,) and lut.dtype == np.float32 for lut in call["curves"])
        )
        self.assertEqual((call["curves"][0][0], call["curves"][0][-1]), (1, 0))
        self.assertGreater(call["curves"][1][128], 0.8)
        np.testing.assert_allclose(call["curves"][2], np.linspace(0, 1, 256), atol=1e-7)
        self.assertIn("curveValue(u_curve0", call["program"])
        self.assertIn("curveValue(u_curve3", call["program"])
        self.assertIn("255.0 + 0.5) / 256.0", call["program"])
        self.assertEqual(result.getpixel((0, 0))[:3], (64, 64, 64))
        self.assertEqual(result.getchannel("A").tobytes(), self.source.getchannel("A").tobytes())
        self.native_graph_preview.assert_not_called()

    def test_monotone_native_curve_and_named_effect_uniforms_have_bounded_contracts(self):
        data = self.capture("image-curves", curve_master=[[0, 0], [0.5, 0.75], [1, 1]])
        shaders.process(NS(), data, self.manager, Mock())
        master = self.calls[-1]["curves"][0]
        self.assertTrue(np.isfinite(master).all())
        self.assertGreater(master[128], 0.75)
        for name in shaders.EFFECTS:
            data = self.capture("glsl-effect", shader_effect=name, shader_amount=0, shader_levels=7)
            shaders.process(NS(), data, self.manager, Mock())
            call = self.calls[-1]
            self.assertEqual(call["floats"], [0])
            self.assertEqual(call["ints"], [7] if name == "posterize" else [])
            self.assertEqual(call["curves"], [])
            self.assertEqual(call["bools"], [])
            self.assertEqual(data["image_utility_source"]["shader_recipe"], name)
            self.assertEqual(data["image_utility_source"]["output_slot"], 0)

    def test_native_shader_source_parsers_confirm_one_output_and_one_pass_per_recipe(self):
        scope = {"re": re, "MAX_OUTPUTS": 4}
        native_functions(
            "comfy_extras/nodes_glsl.py", {"_detect_output_count", "_detect_pass_count"}, scope
        )
        for recipe in ("curves", *shaders.EFFECTS):
            program = resource_path("native_image_effects", recipe + ".frag").read_text(
                encoding="utf-8"
            )
            self.assertEqual(scope["_detect_output_count"](program), 1)
            self.assertEqual(scope["_detect_pass_count"](program), 1)
            self.assertLess(len(program.encode("utf-8")), 4096)

    def test_capture_freezes_source_points_scalars_and_bundled_program(self):
        table = [[0, 0], [1, 1]]
        program = (
            resource_path("native_image_effects", "curves.frag")
            .read_bytes()
            .replace(b"\r\n", b"\n")
        )
        recipe = self.root / "owned-curves.frag"
        recipe.write_bytes(program.replace(b"\n", b"\r\n"))
        with patch.object(shaders, "resource_path", return_value=recipe):
            data = self.capture("image-curves", curve_master=table)
        table[0][1] = 1
        recipe.write_text("changed after capture", encoding="utf-8")
        data["input_image"].paste((0, 0, 0, 0), (0, 0, 5, 4))
        shaders.process(NS(), data, self.manager, Mock())
        self.assertEqual(self.calls[-1]["program"], program.decode("utf-8"))
        self.assertEqual(self.calls[-1]["curves"][0][0], 0)
        self.assertEqual(
            data["image_utility_source"]["shader_sha256"], hashlib.sha256(program).hexdigest()
        )
        self.assertEqual(data[shaders.KEY].source.rgba, self.source.tobytes())
        self.assertEqual(data[assets.KEY].artifacts, ())
        self.assertFalse(data["_sampling_captured"])

    def test_capture_and_real_ui_mount_do_not_import_or_probe_gl(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.startswith(("OpenGL", "comfy_angle", "comfy_extras.nodes_glsl")):
                raise AssertionError("GL import outside explicit processing")
            return original(name, *args, **kwargs)

        controls = {}
        shared = NS(
            settings=NS(default_settings={"shader_amount": 0}),
            add_ctrl=lambda key, control: controls.update({key: control}),
        )
        with (
            patch("builtins.__import__", side_effect=guarded),
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
        ):
            for kind in shaders.KINDS:
                self.capture(kind)
            with gr.Blocks(analytics_enabled=False) as demo:
                choice = gr.Textbox(value="None")
                tool = gr.Textbox(value="None")
                ui.create_ui(choice, tool, lambda choice, tool: tool)
        self.assertEqual(set(controls), set(shaders.FIELDS))
        self.assertEqual(controls["shader_amount"].value, 0)
        self.assertFalse(self.calls)
        context = next(
            index for index, event in demo.fns.items() if event.fn.__name__ == "shader_context"
        )
        for kind in shaders.KINDS:
            values = asyncio.run(demo.process_api(context, [kind, kind], state=SessionState(demo)))[
                "data"
            ]
            self.assertEqual(
                [item["visible"] for item in values],
                [True, kind == "image-curves", kind == "glsl-effect"],
            )
        table = controls["curve_red"]
        points = table.preprocess(table.postprocess([[0, 0], [0.5, 0.6], [1, 1]]))
        self.assertEqual(points, [[0, 0], [0.5, 0.6], [1, 1]])
        self.capture("image-curves", curve_red=points)

    def test_curve_bounds_and_named_recipe_whitelist_reject_arbitrary_code(self):
        for value in (
            [],
            [[0, 0]],
            [[0, 0]] * 33,
            [[0, 0], [0.5, 0.5], [0.5, 0.6], [1, 1]],
            [[0, 0], [0.9, 0.5], [0.1, 0.6], [1, 1]],
            [[0.1, 0], [1, 1]],
            [[0, 0], [1, float("nan")]],
            [[0, True], [1, 1]],
            [[0, 0, 0], [1, 1]],
            "shader code",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.capture("image-curves", curve_master=value)
        for options in (
            {"shader_effect": "../curves"},
            {"shader_effect": []},
            {"shader_amount": float("inf")},
            {"shader_effect": "posterize", "shader_levels": 1},
            {"shader_effect": "posterize", "shader_levels": True},
        ):
            with self.assertRaises(ValueError):
                self.capture("glsl-effect", **options)
        with self.assertRaises(ValueError):
            self.capture("image-curves", curve_interpolation="custom")
        with (
            patch.object(shaders, "MAX_PIXELS", 19),
            self.assertRaisesRegex(ValueError, "4 megapixels"),
        ):
            self.capture("glsl-effect")
        with patch.object(shaders, "MAX_DIMENSION", 4), self.assertRaises(ValueError):
            self.capture("image-curves")

    def test_cancel_drops_output_and_invalid_native_results_fall_back_to_valid_cpu_pixels(self):
        data = self.capture("glsl-effect")
        with self.assertRaises(CancelledError):
            shaders.process(NS(), data, self.manager, Mock(side_effect=CancelledError()))
        self.assertFalse(self.calls)
        for output in (
            (torch.zeros(1, 4, 5, 4),) * 3,
            (torch.zeros(1, 3, 5, 4),) * 4,
            (torch.full((1, 4, 5, 4), float("nan")),) * 4,
        ):
            with (
                patch.object(self.shader, "execute", return_value=output),
                self.assertLogs(level="WARNING"),
            ):
                result = shaders.process(NS(), data, self.manager, Mock())[0]
            self.assertEqual(result.size, self.source.size)
            np.testing.assert_array_equal(
                np.asarray(result)[..., :3], 255 - np.asarray(self.source)[..., :3]
            )
            self.assertTrue(data["image_utility_source"]["acceleration_fallback"])
            self.assertEqual(data["image_utility_source"]["execution_backend"], "cpu")
        with (
            patch.object(self.shader, "execute", side_effect=RuntimeError("owned render failure")),
            self.assertLogs(level="WARNING"),
        ):
            result = shaders.process(NS(), data, self.manager, Mock())[0]
        self.assertEqual(result.size, self.source.size)
        checks = Mock(side_effect=[None] * 5 + [CancelledError()])
        with (
            self.assertRaises(CancelledError),
            patch.object(
                image_effect_cpu,
                "render",
                side_effect=AssertionError("cancellation cannot fall back"),
            ),
        ):
            shaders.process(NS(), data, self.manager, checks)
        self.assertNotIn("image_utility_source", data)

    def test_missing_native_gl_runtime_warns_and_falls_back_without_install(self):
        original = builtins.__import__

        def unavailable(name, *args, **kwargs):
            if name == "comfy_extras.nodes_glsl":
                raise ImportError("owned unavailable runtime fixture")
            return original(name, *args, **kwargs)

        data = self.capture("glsl-effect")
        with (
            patch("builtins.__import__", side_effect=unavailable),
            self.assertLogs(level="WARNING") as terminal,
        ):
            result = shaders.process(NS(), data, self.manager, Mock())[0]
        self.assertFalse(self.calls)
        self.assertIn("Using CPU fallback", " ".join(terminal.output))
        np.testing.assert_array_equal(
            np.asarray(result)[..., :3], 255 - np.asarray(self.source)[..., :3]
        )
        self.assertEqual(data["image_utility_source"]["execution_backend"], "cpu")

    def test_cpu_policy_is_authoritative_without_device_discovery(self):
        false_args = NS(args=NS(cpu=False))
        modules = {"ruinedfooocus.app.arguments": false_args, "comfy.cli_args": false_args}
        with (
            patch.dict(sys.modules, modules),
            patch.dict("os.environ", {"TORCH_PLATFORM": ""}),
            patch.object(sys, "argv", ["owned", "--cpu"]),
        ):
            self.assertTrue(self.cpu_policy())
        with (
            patch.dict(sys.modules, modules),
            patch.dict("os.environ", {"TORCH_PLATFORM": "cpu"}),
            patch.object(sys, "argv", ["owned"]),
        ):
            self.assertTrue(self.cpu_policy())
        with (
            patch.dict(sys.modules, modules),
            patch.dict("os.environ", {"TORCH_PLATFORM": ""}),
            patch.object(sys, "argv", ["owned"]),
        ):
            self.assertFalse(self.cpu_policy())
        with (
            patch.dict(sys.modules, {**modules, "comfy.cli_args": NS(args=NS(cpu=True))}),
            patch.object(sys, "argv", ["owned"]),
        ):
            self.assertTrue(self.cpu_policy())
        with patch("importlib.util.find_spec", return_value=None):
            self.assertFalse(self.gl_presence())

    def test_cpu_named_effect_pixels_and_alpha_need_no_gl_import_or_probe(self):
        original = builtins.__import__

        def no_gl(name, *args, **kwargs):
            if name.startswith(("comfy_extras.nodes_glsl", "comfy_angle", "OpenGL")):
                raise AssertionError("CPU execution must not import GL")
            return original(name, *args, **kwargs)

        with (
            patch.object(shaders, "cpu_mode", self.cpu_policy),
            patch.object(sys, "argv", ["owned", "--cpu"]),
            patch.object(
                shaders,
                "gl_dependencies_present",
                side_effect=AssertionError("CPU cannot inspect GL"),
            ),
            patch("builtins.__import__", side_effect=no_gl),
        ):
            for recipe in shaders.EFFECTS:
                for amount in (0, 0.25, 1):
                    data = self.capture(
                        "glsl-effect", shader_effect=recipe, shader_amount=amount, shader_levels=4
                    )
                    result = shaders.process(NS(), data, self.manager, Mock())[0]
                    self.assertEqual(
                        result.getchannel("A").tobytes(), self.source.getchannel("A").tobytes()
                    )
                    self.assertEqual(data["image_utility_source"]["execution_backend"], "cpu")
                    if amount == 0:
                        self.assertEqual(result.tobytes(), self.source.tobytes())
                    elif recipe == "invert":
                        np.testing.assert_array_equal(
                            np.asarray(result)[..., :3],
                            (
                                np.asarray(self.source)[..., :3] * (1 - amount)
                                + (255 - np.asarray(self.source)[..., :3]) * amount
                            )
                            .round()
                            .astype(np.uint8),
                        )
            self.source = Image.new("RGBA", (4, 1))
            self.source.putdata(
                [(255, 0, 0, 0), (0, 255, 0, 64), (0, 0, 255, 128), (255, 255, 255, 255)]
            )
            result = shaders.process(
                NS(),
                self.capture("glsl-effect", shader_effect="bt709-grayscale"),
                self.manager,
                Mock(),
            )[0]
            self.assertEqual([result.getpixel((x, 0))[0] for x in range(4)], [54, 182, 18, 255])
            self.assertTrue(all(r == g == b for r, g, b, a in result.getdata()))
            self.source = Image.new("RGBA", (8, 1))
            self.source.putdata([(x, x, x, x) for x in (0, 42, 43, 127, 128, 212, 213, 255)])
            result = shaders.process(
                NS(),
                self.capture("glsl-effect", shader_effect="posterize", shader_levels=4),
                self.manager,
                Mock(),
            )[0]
            self.assertEqual(
                [result.getpixel((x, 0))[0] for x in range(8)], [0, 0, 85, 85, 170, 170, 255, 255]
            )
        self.assertFalse(self.calls)

    def test_cpu_curve_pixels_use_native_luts_and_exact_endpoint_center_sampling(self):
        self.source = Image.new("RGBA", (256, 1))
        self.source.putdata([(x, x, x, x) for x in range(256)])
        data = self.capture(
            "image-curves",
            curve_red=[[0, 1], [1, 0]],
            curve_blue=[[0, 0.2], [1, 0.8]],
            curve_interpolation="linear",
            output_format="exr",
            output_bit_depth="32-bit float",
            output_color_space="sRGB",
        )
        with (
            patch.object(shaders, "cpu_mode", self.cpu_policy),
            patch.object(sys, "argv", ["owned", "--cpu"]),
            patch.object(shaders, "gl_pixels", side_effect=AssertionError("CPU cannot call GL")),
        ):
            result = shaders.process(NS(), data, self.manager, Mock())[0]
        values = np.arange(256, dtype=np.float32) / 255
        np.testing.assert_allclose(result[0, :, 0], 1 - values, atol=2e-7)
        np.testing.assert_allclose(result[0, :, 1], values, atol=2e-7)
        np.testing.assert_allclose(result[0, :, 2], 0.2 + 0.6 * values, atol=2e-7)
        np.testing.assert_allclose(result[0, :, 3], values, atol=1e-7)
        curve = self.curve_types.CurveInput.from_raw(
            {"points": [[0, 0], [0.4, 0.8], [1, 0.5]], "interpolation": "monotone_cubic"}
        )
        lut = curve.to_lut().astype(np.float32)
        coordinates = torch.tensor([[[0, 0.5 / 255, 0.4, 254.5 / 255, 1]]], dtype=torch.float32)
        sampled = image_effect_cpu.sample_lut(coordinates, lut)
        np.testing.assert_allclose(
            sampled.numpy()[0, 0],
            np.interp(coordinates.numpy()[0, 0] * 255, np.arange(256), lut),
            atol=2e-7,
        )
        self.assertFalse(self.calls)

    def test_absent_gl_dependencies_warn_without_attempting_gl_and_cancel_stays_terminal(self):
        with (
            patch.object(shaders, "gl_dependencies_present", return_value=False),
            patch.object(
                shaders,
                "gl_pixels",
                side_effect=AssertionError("known absent GL must not initialize"),
            ),
            self.assertLogs(level="WARNING") as terminal,
        ):
            data = self.capture("glsl-effect")
            result = shaders.process(NS(), data, self.manager, Mock())[0]
        self.assertIn("Using CPU fallback", " ".join(terminal.output))
        self.assertEqual(result.size, self.source.size)
        self.assertFalse(self.calls)
        with (
            patch.object(shaders, "cpu_mode", self.cpu_policy),
            patch.object(sys, "argv", ["owned", "--cpu"]),
            self.assertRaises(CancelledError),
        ):
            shaders.process(
                NS(),
                self.capture("image-curves"),
                self.manager,
                Mock(side_effect=[None, None, CancelledError()]),
            )

    def test_modified_bundled_program_cannot_change_cpu_recipe_identity(self):
        owned = self.root / "changed.frag"
        owned.write_text("arbitrary changed program", encoding="utf-8")
        with (
            patch.object(shaders, "resource_path", return_value=owned),
            self.assertRaisesRegex(ValueError, "reviewed recipe"),
        ):
            self.capture("glsl-effect")

    def test_rgba_utility_avif_is_rejected_before_any_processing(self):
        for kind in dispatch.KINDS:
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "RGBA"):
                self.capture(kind, output_format="avif")
        self.assertFalse(self.calls)

    def test_precision_output_and_library_replay_keep_exact_recipe_intent(self):
        data = self.capture(
            "image-curves",
            curve_master=[[0, 0], [0.5, 0.7], [1, 1]],
            output_format="exr",
            output_bit_depth="32-bit float",
            output_color_space="sRGB",
        )
        result = dispatch.process(NS(), data, self.manager, Mock())[0]
        self.assertEqual((result.shape, result.dtype), ((4, 5, 4), np.dtype("float32")))
        np.testing.assert_allclose(result[..., 3], np.asarray(self.source)[..., 3] / 255, atol=1e-7)
        metadata = dispatch.output_metadata(data)
        self.assertEqual(metadata["image_utility_source"]["curve_lut_size"], 256)
        self.assertNotIn("base_model_name", metadata)
        self.assertNotIn("steps", metadata)
        from ruinedfooocus.features.gallery.image_metadata import read_metadata

        scope = dict(
            gr=gr,
            json=json,
            get_ctrl=lambda key: key,
            read_metadata=read_metadata,
            resolve_setup=Mock(side_effect=AssertionError("generation setup")),
        )
        apply = function("ruinedfooocus/features/gallery/ui.py", "apply_metadata", scope)
        scope["apply_metadata"] = apply
        reuse = function("ruinedfooocus/features/gallery/ui.py", "reuse_setup", scope)
        replay = reuse(json.dumps(metadata))
        self.assertIsNone(replay["input_image"]["value"])
        self.assertIsNone(replay["image_utility_model"]["value"])
        self.assertEqual(replay["curve_master"]["value"], [[0, 0], [0.5, 0.7], [1, 1]])

    def test_public_json_and_multipart_are_typed_and_generation_independent(self):
        source = BytesIO()
        self.source.save(source, format="PNG")
        output = self.root / "result.png"
        self.source.save(output)
        backend = NS(
            images=Mock(side_effect=lambda *args: job(("result", [str(output)]))),
            resolve_model=Mock(side_effect=AssertionError("generation model")),
        )
        api, _ = create_apps(backend)
        with TestClient(api) as client:
            for kind in shaders.KINDS:
                body = dict(
                    input_image=base64.b64encode(source.getvalue()).decode(),
                    controlnet={"type": kind},
                    curve_master=[[0, 0], [1, 1]],
                    shader_amount=0,
                    shader_levels=3,
                    response_format="b64_json",
                )
                response = client.post("/images/generations", json=body)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(backend.images.call_args.args[1], {})
                form = {
                    key: (
                        json.dumps(value) if key in ("controlnet", "curve_master") else str(value)
                    )
                    for key, value in body.items()
                    if key != "input_image"
                }
                response = client.post(
                    "/images/edits",
                    data=form,
                    files={"image": ("source.png", source.getvalue(), "image/png")},
                )
                self.assertEqual(response.status_code, 200, response.text)
                for extra in (
                    {"fragment_shader": "arbitrary"},
                    {"shader_effect": "custom"},
                    {"shader_amount": True},
                    {"shader_levels": 33},
                    {"curve_red": [[0, 0], [1, True]]},
                    {"n": 2},
                ):
                    self.assertEqual(
                        client.post("/images/generations", json=body | extra).status_code, 400
                    )
        backend.resolve_model.assert_not_called()

    def test_actual_api_runtime_captures_curve_recipe_before_worker_admission(self):
        raw = BytesIO()
        self.source.save(raw, format="PNG")
        shared = NS(
            path_manager=self.manager,
            models=NS(offline=True),
            settings=NS(default_settings={}),
            state={},
        )
        worker = NS(
            prepare_image_data=Mock(side_effect=lambda data, **kw: dict(data)),
            _process=Mock(return_value=[str(self.root / "result.png")]),
        )
        with (
            patch.dict(
                sys.modules,
                {"ruinedfooocus.app.state": shared, "ruinedfooocus.app.async_worker": worker},
            ),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(ruinedfooocus.app, "async_worker", worker, create=True),
            patch.object(api_runtime, "submit", side_effect=lambda run: run),
        ):
            for kind in shaders.KINDS:
                payload = dict(
                    input_image=base64.b64encode(raw.getvalue()).decode(),
                    controlnet={"type": kind},
                    shader_amount=0,
                )
                operation = api_runtime.images(payload, {})
                operation.prepare(object())
                operation(None, 4)
                data = worker._process.call_args.args[0]
                self.assertIsInstance(data[shaders.KEY], shaders.ShaderRequest)
                self.assertFalse(data["_sampling_captured"])
                for extra in (
                    {"size": "30x40"},
                    {"reference_images": [payload["input_image"]]},
                    {"mask": payload["input_image"]},
                ):
                    with self.assertRaises(ValueError):
                        api_runtime.images(payload | extra, {})

    def test_actual_serial_worker_writes_one_source_sized_image_and_recipe_provenance(self):
        from ruinedfooocus.features.image_processing.depth_utility_pipeline import pipeline

        forbidden = Mock(side_effect=AssertionError("generation dependency"))
        shared = NS(
            models=NS(get_models_by_path=forbidden, get_model_base=forbidden),
            state={},
            performance_settings=NS(apply=forbidden),
            resolution_settings=object(),
            path_manager=NS(model_paths={"temp_outputs_path": self.root}),
            shared_cache={},
        )
        control = NS(get_settings=lambda data: data.get("controlnet", {}))
        worker = NS(add_result=Mock(), check_interrupt=lambda data: None)
        with (
            patch.dict(
                sys.modules,
                {"ruinedfooocus.app.state": shared, "ruinedfooocus.app.async_worker": worker},
            ),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(ruinedfooocus.app, "async_worker", worker, create=True),
            patch.object(shaders, "cpu_mode", self.cpu_policy),
            patch.object(sys, "argv", ["owned", "--cpu"]),
        ):
            for kind in shaders.KINDS:
                data = self.capture(
                    kind, task_id=1, task_type="tool_call", silent=True, index=(0, 1)
                )
                pipe = pipeline()
                route = dict(state={"pipeline": None}, shared=shared, controlnet=control)
                function(
                    "ruinedfooocus/features/generation/pipelines.py", "selected_pipeline", route
                )
                update = function("ruinedfooocus/features/generation/pipelines.py", "update", route)
                with patch(
                    "ruinedfooocus.features.image_processing.depth_utility_pipeline.pipeline",
                    return_value=pipe,
                ):
                    render = function(
                        "ruinedfooocus/app/async_worker.py",
                        "_process",
                        dict(
                            shared=shared,
                            settings=NS(default_settings={}),
                            control_tools=control,
                            check_interrupt=lambda data: None,
                            process_metadata=forbidden,
                            pipelines=NS(update=update),
                            get_checkpoint_hashes=forbidden,
                            process_prompt=forbidden,
                            shift_attention=forbidden,
                            parse_loras=forbidden,
                            merge_loras=forbidden,
                            TimeIt=lambda name: nullcontext(),
                            resource_path=resource_path,
                            random=random,
                            Path=Path,
                            pathlib=pathlib,
                            Image=Image,
                            PngInfo=PngInfo,
                            json=json,
                            generate_temp_filename=lambda **kw: str(
                                self.root / (kind + "." + kw["extension"])
                            ),
                        ),
                    )
                    outputs = render(data)
                self.assertEqual(len(outputs), 1)
                with Image.open(outputs[0]) as image:
                    self.assertEqual((image.size, image.mode), (self.source.size, "RGBA"))
                    self.assertEqual(
                        image.getchannel("A").tobytes(), self.source.getchannel("A").tobytes()
                    )
                    metadata = json.loads(image.info["parameters"])
                    if kind == "image-curves":
                        self.assertEqual(image.tobytes(), self.source.tobytes())
                    else:
                        np.testing.assert_array_equal(
                            np.asarray(image)[..., :3], 255 - np.asarray(self.source)[..., :3]
                        )
                details = metadata["image_utility_source"]
                self.assertEqual(details["execution_backend"], "cpu")
                self.assertEqual(details["shader_sha256"], data[shaders.KEY].program_sha256)
                self.assertEqual(
                    details["native_revision"], "fca60fe70e5b84c5c6b058263653d15cb3401e44"
                )
                self.assertNotIn("steps", metadata)
                self.assertNotIn("base_model_name", metadata)
                self.assertNotIn(str(self.root), json.dumps(metadata))
        forbidden.assert_not_called()
        self.assertFalse(self.calls)


if __name__ == "__main__":
    unittest.main()
