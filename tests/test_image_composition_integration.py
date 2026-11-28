"""Real pure-image processing through worker, API and source-local Library callers."""

import ast
import base64
from contextlib import nullcontext
from io import BytesIO
import json
from pathlib import Path
import pathlib
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.image_processing import image_composition as composition
import ruinedfooocus.features.image_editing as editing

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, scope):
    node = next(
        n
        for n in ast.parse((ROOT / path).read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), scope)
    return scope[name]


def inline(image):
    stream = BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


class CompositionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.base = Image.new("RGBA", (8, 6), (0, 0, 255, 255))
        self.overlay = Image.new("RGBA", (4, 6), (255, 0, 0, 128))
        self.forbidden = Mock(side_effect=AssertionError("generation state was used"))
        self.manager = NS(model_paths={"temp_outputs_path": self.root})

    def test_capture_and_preparation_own_only_operation_inputs_with_no_asset_work(self):
        source = dict(
            composition_images=[self.base, self.overlay],
            prompt="#! unsafe",
            base_model_name="missing",
            model_recipe_path="missing.yaml",
            reference_adapter="redux",
            automatic_upscale_model="missing",
            size_plan={"invalid": True},
            native_sampling={"stale": True},
            _native_yaml_recipe=object(),
            _prompt_preparation=object(),
        )
        data = assets.capture_image_assets(
            source, self.manager, {"type": "image-composite"}, offline=True, models=self.forbidden
        )
        self.assertEqual(data[assets.KEY].artifacts, ())
        self.assertTrue(data[assets.KEY].offline)
        for key in (
            "size_plan",
            "native_sampling",
            "_native_yaml_recipe",
            "_prompt_preparation",
            "model_recipe_path",
            "reference_adapter",
            "automatic_upscale_model",
        ):
            self.assertNotIn(key, data)
        service = Mock()
        prepared = assets.prepare_image_assets(data, check_cancelled=lambda: None, service=service)
        service.submit.assert_not_called()
        service.artifact_lease.assert_not_called()
        self.assertIs(prepared[composition.KEY], data[composition.KEY])
        self.assertEqual(source["prompt"], "#! unsafe")
        captured = function(
            "ruinedfooocus/app/async_worker.py",
            "capture_image_data",
            dict(
                control_tools=NS(get_settings=lambda data: {"type": "image-composite"}),
                shared=NS(path_manager=self.manager, models=NS(offline=True)),
            ),
        )

        class UnrelatedState:
            def __deepcopy__(self, memo):
                raise AssertionError("Copied unrelated generation state")

        early = captured(source | {"loras": UnrelatedState()})
        self.assertEqual(early[assets.KEY].artifacts, ())
        self.assertEqual(early["loras"], [])

    def test_cold_capture_never_imports_torch_comfy_or_the_dataset_module(self):
        code = """
import importlib.abc, sys
class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'comfy', 'comfy_extras', 'comfy_api'):
            raise AssertionError('runtime import during pure capture: ' + fullname)
sys.meta_path.insert(0, RejectRuntime())
from PIL import Image
from ruinedfooocus.features.image_processing.image_composition import capture, KEY
data = capture({'composition_images': [Image.new('RGBA', (8, 6))]}, 'image-grid')
assert data[KEY].output_size == (1032, 512)
assert not any(name.split('.')[0] in ('torch', 'comfy', 'comfy_extras', 'comfy_api') for name in sys.modules)
print('cold-capture-no-runtime: passed')
"""
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), "cold-capture-no-runtime: passed")

    def test_actual_create_submission_bypasses_original_source_and_prompt_preparation(self):
        from ruinedfooocus.features.generation.ui_request import build_ui_request
        from ruinedfooocus.features.image_processing import native_image_outputs

        controls = NS(get_settings=lambda values: {"type": "image-grid"})
        submit = Mock(return_value=17)
        append = function(
            "ruinedfooocus/ui/app.py",
            "append_work",
            dict(
                ui_controlnet=NS(tool_kind=lambda selection, kind: "image-grid"),
                worker=NS(add_task=submit),
                use_random_prompt=self.forbidden,
                get_promptlist=self.forbidden,
            ),
        )
        values = dict(
            composition_images=[self.base],
            preset_selection="missing.png",
            prompt="#! wrong\n---\nsecond",
            image_number=0,
            cn_selection="Make image grid",
            _native_main_source={"bad": True},
            automatic_upscale="old",
            automatic_upscale_target="Input image only",
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.features.image_editing.controlnet": controls}),
            patch.object(
                editing,
                "controlnet",
                controls,
                create=True,
            ),
            patch.object(native_image_outputs, "capture_original_source", self.forbidden),
        ):
            request = build_ui_request(values, models=self.forbidden, settings={})
            self.assertEqual(append(request), 17)
        data = submit.call_args.args[0]
        self.assertEqual((data["prompt"], data["image_number"], data["image_total"]), ("", 1, 1))
        self.assertFalse(data["generate_forever"])
        self.assertIs(data["composition_images"][0], self.base)
        self.assertIsNone(data["preset_selection"])
        self.forbidden.assert_not_called()

    def test_create_generator_validates_composition_before_copying_unrelated_controls(self):
        class ReachedComposition(BaseException):
            pass

        class UnrelatedState:
            def __deepcopy__(self, memo):
                raise AssertionError("Copied unrelated Create state")

        controls = NS(get_settings=lambda data: {"type": "image-grid"})
        callback = function(
            "ruinedfooocus/ui/app.py",
            "generate_request",
            dict(
                ui_controlnet=NS(
                    tool_kind=lambda *args: "image-grid", family_for_checkpoint=self.forbidden
                ),
                shared=NS(models=self.forbidden, settings=NS(default_settings={})),
                settings={},
                generation_task="owned",
                update_clicked=lambda: {},
                append_work=Mock(side_effect=ReachedComposition()),
            ),
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.features.image_editing.controlnet": controls}),
            patch.object(editing, "controlnet", controls, create=True),
        ):
            stream = callback(
                dict(
                    composition_images=[],
                    image_number=0,
                    loras=UnrelatedState(),
                    automatic_upscale="stale",
                    automatic_upscale_target="Input image only",
                )
            )
            self.assertEqual(next(stream), {"owned": None})
            with self.assertRaises(ReachedComposition):
                next(stream)

    def test_actual_worker_writes_one_alpha_png_and_precise_recipe_without_generation_metadata(
        self,
    ):
        import ruinedfooocus.app as app

        worker = NS(check_interrupt=lambda data: None, add_result=Mock())
        shared = NS(
            models=self.forbidden,
            path_manager=self.manager,
            state={"preview_total": 1},
            shared_cache={},
        )
        controls = NS(get_settings=lambda data: data["controlnet"])
        route_scope = dict(
            state={"pipeline": None},
            shared=shared,
            controlnet=controls,
            selected_pipeline=lambda: route_scope["state"]["pipeline"],
        )
        update = function("ruinedfooocus/features/generation/pipelines.py", "update", route_scope)
        process = function(
            "ruinedfooocus/app/async_worker.py",
            "_process",
            dict(
                shared=shared,
                settings=NS(default_settings={}),
                control_tools=controls,
                pipelines=NS(update=update),
                check_interrupt=worker.check_interrupt,
                process_prompt=self.forbidden,
                process_metadata=self.forbidden,
                get_checkpoint_hashes=self.forbidden,
                parse_loras=self.forbidden,
                resource_path=resource_path,
                random=random,
                TimeIt=nullcontext,
                model_management=NS(InterruptProcessingException=InterruptedError),
                generate_temp_filename=lambda **kwargs: str(self.root / "result.png"),
                Path=Path,
                pathlib=pathlib,
                Image=Image,
                PngInfo=PngInfo,
                json=json,
            ),
        )
        data = assets.capture_image_assets(
            dict(
                task_type="process",
                silent=True,
                task_id=7,
                index=(0, 1),
                composition_images=[self.base, self.overlay],
                output_format="png",
                prompt="search: wrong",
            ),
            self.manager,
            {"type": "image-composite"},
            offline=True,
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.async_worker": worker}),
            patch.object(app, "async_worker", worker, create=True),
        ):
            outputs = process(data)
        self.assertEqual(outputs, [str(self.root / "result.png")])
        with Image.open(outputs[0]) as result:
            self.assertEqual((result.mode, result.size), ("RGBA", (8, 6)))
            self.assertEqual(result.getpixel((2, 0)), (128, 0, 127, 255))
            saved = json.loads(result.info["parameters"])
        self.assertEqual(saved["controlnet"], {"type": "image-composite"})
        self.assertEqual(saved["native_composition_source"]["output_size"], [8, 6])
        self.assertEqual(saved["composition_width"], 8)
        for key in ("Prompt", "prompt", "seed", "sampler_name", "loras", "cfg", "base_model_name"):
            self.assertNotIn(key, saved)
        self.assertNotIn(str(self.root), json.dumps(saved))
        self.assertEqual(route_scope["state"]["pipeline"].pipeline_type, ["image_composition"])
        self.forbidden.assert_not_called()

    def test_api_closure_captures_order_and_format_before_queue_without_defaults_or_checkpoint(
        self,
    ):
        import ruinedfooocus.app as app
        from ruinedfooocus.app import api_runtime as api

        shared = NS(path_manager=self.manager, models=NS(offline=True), state={})
        worker = NS(
            prepare_image_data=Mock(side_effect=lambda data, **kw: data),
            _process=Mock(return_value=["result.png"]),
        )
        payload = dict(
            controlnet={"type": "image-composite"},
            composition_images=[inline(self.base), inline(self.overlay)],
            output_format="png",
            output_bit_depth="16-bit",
            output_color_space="sRGB",
            model="missing",
            prompt="ignored",
            size="8x6",
        )
        with (
            patch.dict(
                sys.modules,
                {"ruinedfooocus.app.state": shared, "ruinedfooocus.app.async_worker": worker},
            ),
            patch.object(app, "state", shared, create=True),
            patch.object(app, "async_worker", worker, create=True),
        ):
            run = api._image_operation(payload, self.forbidden, self.forbidden)
            self.assertEqual(run.resource_group, "image")
            job = object()
            run.prepare(job)
            self.assertEqual(run(job, 31), ["result.png"])
            data = worker._process.call_args.args[0]
            self.assertEqual(data["output_bit_depth"], "16-bit")
            self.assertEqual(composition.captured(data).images[1].size, (4, 6))
            self.assertIs(data["_api_job"], job)
            self.assertEqual(data["task_id"], 31)
            with self.assertRaisesRegex(ValueError, "canvas"):
                api._image_operation(payload | {"size": "9x6"}, {}, {})
            animated = BytesIO()
            self.base.convert("RGB").save(
                animated, format="GIF", save_all=True, append_images=[self.overlay.convert("RGB")]
            )
            with self.assertRaises(ValueError):
                api._image_operation(
                    payload
                    | {
                        "composition_images": [
                            inline(self.base),
                            base64.b64encode(animated.getvalue()).decode(),
                        ]
                    },
                    {},
                    {},
                )

    def test_composition_float_alpha_survives_actual_pinned_png16_codec(self):
        from ruinedfooocus.features.image_processing import native_image_formats as formats
        from test_native_image_formats import native_codecs

        base = Image.new("RGBA", (2, 2), (0, 0, 255, 128))
        overlay = Image.new("RGBA", (2, 2), (255, 0, 0, 128))
        values = dict(output_format="png", output_bit_depth="16-bit", output_color_space="sRGB")
        data = composition.capture(
            dict(composition_images=[base, overlay], **values), "image-composite"
        )
        pixels = composition.process(data, lambda: None)
        with patch.object(formats, "_native", return_value=native_codecs()):
            path = formats.write_image(
                self.root / "precise.png",
                pixels,
                values,
                {"native_composition_source": data["native_composition_source"]},
            )
        imported = formats.read_image(path)
        self.assertEqual(imported.pixels.dtype, np.uint16)
        np.testing.assert_array_equal(imported.pixels, (pixels * 65535).astype(np.uint16))
        self.assertTrue(np.any(imported.pixels % 257))
        saved = json.loads(imported.metadata["parameters"])
        self.assertEqual(
            saved["native_composition_source"]["native"], "PorterDuffImageComposite.execute"
        )
        self.assertTrue(saved["native_export"]["alpha"])

    def test_typed_public_api_bypasses_model_selection_and_rejects_mixed_or_oversized_inputs(self):
        from fastapi.testclient import TestClient
        from pydantic import ValidationError
        from ruinedfooocus.integrations.openai_api import ImageRequest, create_apps
        from test_api_live_contracts import job

        output = self.root / "output.png"
        self.base.save(output)
        backend = NS(
            resolve_model=self.forbidden, images=Mock(return_value=job(("result", [str(output)])))
        )
        api, _ = create_apps(backend)
        payload = dict(
            model="missing",
            controlnet={"type": "image-grid"},
            composition_images=[inline(self.base)],
            grid_cell_width=32,
            grid_cell_height=32,
            response_format="b64_json",
        )
        with TestClient(api) as client:
            response = client.post("/images/generations", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(backend.images.call_args.args[1], {})
        for changed in (
            dict(n=2),
            dict(input_image="bad"),
            dict(grid_columns=True),
            dict(composition_mode="bad"),
            dict(controlnet={"type": "image-grid", "strength": 1}),
            dict(composition_mask="bad"),
            dict(controlnet={"type": "image-composite"}),
            dict(controlnet=None),
        ):
            with self.subTest(changed=changed), self.assertRaises(ValidationError):
                ImageRequest(**(payload | changed))
        with (
            patch.object(composition, "MAX_INLINE_CHARS", 4),
            self.assertRaisesRegex(ValidationError, "combined"),
        ):
            ImageRequest(**payload)
        self.forbidden.assert_not_called()

    def test_library_original_seeds_ordered_inputs_and_recipe_replay_clears_uploads(self):
        from ruinedfooocus.features.gallery.actions import source_record
        from ruinedfooocus.features.gallery.image_metadata import render_values

        path = self.root / "original.png"
        self.base.save(path)
        before = path.read_bytes()
        scope = dict(
            get_ctrl=lambda key: key,
            gr=NS(update=lambda **kw: kw, Error=ValueError),
            REFERENCE_RESETS={"composition_images": [], "composition_mask": None},
        )
        tools = NS(
            checkpoint_features=self.forbidden,
            edit_tool=self.forbidden,
            tool_kind=lambda *args: None,
            controlnet=NS(cn_options={"Compose images": {"type": "image-composite"}}),
        )
        with patch.dict(sys.modules, {"ruinedfooocus.features.image_editing.ui": tools}):
            prepare = function("ruinedfooocus/features/gallery/ui.py", "prepare_image", scope)
            values = prepare(source_record(path), "image-composite", "missing", "Disabled", "Off")
        self.assertEqual(values["cn_selection"]["value"], "Compose images")
        self.assertEqual(
            values["composition_images"]["value"][0].getpixel((0, 0)), (0, 0, 255, 255)
        )
        self.assertEqual(before, path.read_bytes())
        packet = composition.capture(
            dict(composition_images=[self.base, self.overlay]), "image-composite"
        )
        composition.process(packet, lambda: None)
        metadata = render_values(packet)
        self.assertIn("native_composition_source", metadata)
        self.assertNotIn(composition.KEY, metadata)
        replayed = composition.replay_values(metadata)
        self.assertEqual(replayed["composition_images"], [])
        self.assertIsNone(replayed["composition_mask"])
        self.assertEqual(replayed["composition_width"], 8)
        with (
            patch.dict(
                sys.modules,
                {
                    "ruinedfooocus.features.image_editing.controlnet": NS(
                        cn_options=tools.controlnet.cn_options, NEWCN="Custom..."
                    )
                },
            ),
            patch.object(
                editing,
                "controlnet",
                NS(cn_options=tools.controlnet.cn_options, NEWCN="Custom..."),
                create=True,
            ),
        ):
            restore = function("ruinedfooocus/features/gallery/ui.py", "composition_setup", scope)
            restored = restore(metadata)
        self.assertEqual(restored["cn_selection"]["value"], "Compose images")
        self.assertEqual(restored["composition_images"]["value"], [])
        self.assertIsNone(restored["composition_mask"]["value"])
        # Check the special replay branch against the production event's
        # declared component names, including its fresh-upload resets.
        gallery = ast.parse(
            (ROOT / "ruinedfooocus/features/gallery/ui.py").read_text(encoding="utf-8")
        )
        constants = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in gallery.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("METADATA_CONTROLS", "REFERENCE_RESETS")
        }
        from ruinedfooocus.features.image_editing.image_guidance import FIELD_SCHEMA

        binding = next(
            node
            for node in ast.walk(gallery)
            if isinstance(node, ast.FunctionDef) and node.name == "bind_actions"
        )
        end = next(
            index
            for index, node in enumerate(binding.body)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "setup_names"
        )
        declarations = [
            node
            for node in binding.body[: end + 1]
            if isinstance(node, ast.ImportFrom)
            or (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("metadata_names", "setup_names")
            )
        ]
        output_scope = dict(constants, FIELD_SCHEMA=FIELD_SCHEMA)
        exec(
            compile(ast.Module(body=declarations, type_ignores=[]), "setup outputs", "exec"),
            output_scope,
        )
        self.assertLessEqual(set(restored), set(output_scope["setup_names"]))
        self.assertLessEqual(set(constants["REFERENCE_RESETS"]), set(output_scope["setup_names"]))
        self.forbidden.assert_not_called()

    def test_composition_context_hides_single_source_and_sampling_controls_for_any_checkpoint(self):
        presets = json.loads(resource_path("defaults", "powerup.default").read_text())
        scope = dict(
            gr=NS(update=lambda **kw: kw),
            t=lambda value: value,
            tool_kind=lambda selected, kind: selected,
            checkpoint_features=self.forbidden,
            SDXL_CONTROLS=(),
            input_label=lambda features: "Single source",
            controlnet=NS(cn_options=presets),
            NEWCN="Custom...",
        )
        context = function("ruinedfooocus/features/image_editing/ui.py", "input_context", scope)
        choices = function("ruinedfooocus/features/image_editing/ui.py", "powerup_choices", scope)
        for kind in composition.KINDS:
            values = context(
                True,
                kind,
                "",
                "missing",
                upscale_scale="old-upscaler",
                upscale_target="Input image only",
                features=dict(inpaint=True, sdxl=True),
            )
            self.assertFalse(values[0]["visible"])
            self.assertFalse(values[2]["visible"])
            self.assertFalse(values[2]["value"])
        self.assertIn(("Make image grid", "Make image grid"), choices())
        self.assertIn(("Compose images", "Compose images"), choices(sdxl=False, sd4x=True))
        self.forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
