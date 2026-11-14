"""Adjustment controls, strict requests, native preparation and owned mask handoff."""

import asyncio
import base64
from concurrent.futures import CancelledError
from io import BytesIO
import json
from pathlib import Path
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

import ruinedfooocus.app
from ruinedfooocus.app import api_runtime
from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.core.settings_schema import validate_known
from ruinedfooocus.features.gallery import editor_session
from ruinedfooocus.features.gallery.editing_workflow import build_library_request
from ruinedfooocus.features.gallery.actions import source_record
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.image_processing import still_image_utilities as utilities
from ruinedfooocus.features.image_processing import ui_image_adjustments as ui
from ruinedfooocus.features.image_processing.native_image_formats import ImportedImage
from ruinedfooocus.integrations.openai_api import ImageRequest, create_apps
from test_api_live_contracts import job
from test_utility_sampling_capture import function


class AdjustmentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.manager = NS(get_model_folders=Mock(side_effect=AssertionError("generation model")))
        self.source = Image.fromarray(np.arange(5 * 3 * 4, dtype=np.uint8).reshape(3, 5, 4))
        self.path = self.root / "source.png"
        self.source.save(self.path)
        self.original_bytes = self.path.read_bytes()
        self.owner = ("fixture", "adjustment-owner")
        self.store = editor_session.EditorSessionStore(self.root / "history")

    def encoded(self, image=None):
        raw = BytesIO()
        (image or self.source).save(raw, format="PNG")
        return base64.b64encode(raw.getvalue()).decode("ascii")

    def test_one_alias_resolves_three_kinds_and_replay_clears_assets(self):
        presets = json.loads(resource_path("defaults", "powerup.default").read_text())
        self.assertEqual(presets[utilities.ADJUSTMENT_LABEL], {"type": "image-transform"})
        for kind in utilities.ADJUSTMENT_KINDS:
            self.assertEqual(utilities.adjustment_kind("Adjust image", kind.capitalize()), kind)
            values = utilities.replay_values(dict(controlnet={"type": kind}))
            self.assertEqual(
                (values["cn_selection"], values["cn_type"]), ("Adjust image", kind.capitalize())
            )
            self.assertIsNone(values["input_image"])
            self.assertIsNone(values["_native_main_source"])
            self.assertNotIn("prompt", values)
            self.assertNotIn("negative", values)
            self.assertTrue(all(values[field] is None for field in utilities.ADJUSTMENT_IMAGES))
        self.assertEqual(utilities.adjustment_kind("Adjust image", "Canny"), "image-transform")
        self.assertIsNone(utilities.adjustment_kind("Img2Img", "Image-mask"))

    def test_ui_has_every_field_and_real_context_events_only_change_visibility(self):
        controls = {}
        shared = NS(
            settings=NS(default_settings={}),
            add_ctrl=lambda name, component: controls.update({name: component}),
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
        ):
            with gr.Blocks(analytics_enabled=False) as demo:
                selection = gr.Textbox("Adjust image")
                tool = gr.Dropdown(
                    choices=[kind.capitalize() for kind in utilities.ADJUSTMENT_KINDS],
                    value="Image-transform",
                )
                ui.create_ui(selection, tool, utilities.adjustment_kind)
        self.assertEqual(
            set(controls),
            set(utilities.ADJUSTMENT_DEFAULTS)
            | set(utilities.ADJUSTMENT_IMAGES)
            | {"_adjustment_hdr_precision"},
        )
        event = next(
            index for index, event in demo.fns.items() if event.fn.__name__ == "adjustment_context"
        )
        for kind in utilities.ADJUSTMENT_KINDS:
            result = asyncio.run(
                demo.process_api(
                    event,
                    ["Adjust image", kind.capitalize(), "crop", "composite", "quantize"],
                    state=SessionState(demo),
                )
            )["data"]
            self.assertTrue(result[0]["visible"])
            self.assertEqual(result[1]["value"], kind)
            self.assertEqual(
                [row["visible"] for row in result[2:5]],
                [kind == candidate for candidate in utilities.ADJUSTMENT_KINDS],
            )
        callback = demo.fns[event].fn
        updates = callback("Adjust image", "Image-mask", "resize", "composite", "blend")
        for component, update in zip(demo.fns[event].outputs, updates):
            if component in [controls[name] for name in utilities.ADJUSTMENT_IMAGES]:
                self.assertNotIn("value", update)
        choose = next(
            index for index, event in demo.fns.items() if event.fn.__name__ == "choose_adjustment"
        )
        value = asyncio.run(demo.process_api(choose, ["image-mask"], state=SessionState(demo)))[
            "data"
        ][0]
        self.assertEqual(value["value"], "Image-mask")

    def test_editor_adjust_action_opens_more_tools_and_its_group_without_changing_drafts(self):
        from ruinedfooocus.features.gallery.ui_workspace import EditorWorkspace

        controls = {}
        forbidden = Mock(side_effect=AssertionError("model execution"))
        shared = NS(
            settings=NS(default_settings=dict(transform_width=731, color_blend_factor=0.27)),
            add_ctrl=lambda name, component: controls.update({name: component}),
            models=forbidden,
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
        ):
            with gr.Blocks(analytics_enabled=False) as demo:
                workspace = EditorWorkspace(store=self.store)
                controls.update(
                    prompt=gr.Textbox("retained draft"),
                    negative=gr.Textbox("retained negative"),
                    input_image=gr.Image(type="pil"),
                    image_source=gr.State(None),
                )
                with gr.Tabs(selected="generation") as tabs:
                    with gr.Tab("Generation", id="generation"):
                        gr.Markdown("Existing generation settings")
                    with gr.Tab("More tools", id="image-tools"):
                        selection = gr.Dropdown(choices=["None", "Adjust image"], value="None")
                        kind = gr.Dropdown(
                            choices=[value.capitalize() for value in utilities.ADJUSTMENT_KINDS],
                            value="Image-mask",
                        )
                        controls.update(cn_selection=selection, cn_type=kind)
                        ui.create_ui(selection, kind, utilities.adjustment_kind)
                run_controls = dict(
                    task=gr.State(None),
                    preview=gr.Image(type="filepath"),
                    native_source=gr.State(None),
                    inpaint_toggle=gr.Checkbox(False),
                    inpaint=gr.ImageEditor(type="pil"),
                    gallery=gr.Gallery(),
                    artifacts=gr.State([]),
                    run=gr.Button("Generate"),
                    stop=gr.Button("Stop"),
                    progress=gr.Markdown(),
                    hint=gr.Markdown(),
                    metadata=gr.JSON(),
                    originals=gr.File(),
                )
                workspace.bind(
                    controls,
                    run_request=forbidden,
                    stop_request=forbidden,
                    run_controls=run_controls,
                    create_tabs=tabs,
                )
        state = SessionState(demo)
        action = next(
            index
            for index, event in demo.fns.items()
            if getattr(event.fn, "__name__", "") == "adjustment_action"
        )
        result = asyncio.run(demo.process_api(action, ["adjust-image", "Image-mask"], state=state))[
            "data"
        ]
        self.assertEqual(result[0], "Adjust image")
        self.assertEqual(result[1]["value"], "Image-mask")
        self.assertEqual(result[2]["selected"], "image-tools")
        self.assertEqual(state[tabs._id].selected, "image-tools")
        context = next(
            index
            for index, event in demo.fns.items()
            if getattr(event.fn, "__name__", "") == "adjustment_context"
        )
        visibility = asyncio.run(
            demo.process_api(
                context, [result[0], result[1]["value"], "resize", "grow", "blend"], state=state
            )
        )["data"]
        self.assertTrue(visibility[0]["visible"])
        self.assertEqual([item["visible"] for item in visibility[2:5]], [False, True, False])
        other = asyncio.run(demo.process_api(action, ["prompt-edit", "Image-mask"], state=state))[
            "data"
        ]
        self.assertTrue(all("selected" not in item and "value" not in item for item in other))
        self.assertEqual(
            (controls["prompt"].value, controls["negative"].value),
            ("retained draft", "retained negative"),
        )
        self.assertEqual(
            (controls["transform_width"].value, controls["color_blend_factor"].value), (731, 0.27)
        )
        self.assertNotIn(controls["prompt"], demo.fns[action].outputs)
        self.assertNotIn(controls["input_image"], demo.fns[action].outputs)
        forbidden.assert_not_called()

    def test_strict_json_api_bypasses_checkpoint_and_rejects_unowned_group_fields(self):
        output = self.root / "output.png"
        self.source.save(output)
        backend = NS(
            images=Mock(side_effect=lambda *args: job(("result", [str(output)]))),
            resolve_model=Mock(side_effect=AssertionError("checkpoint")),
        )
        api, _ = create_apps(backend)
        with TestClient(api) as client:
            for kind, fields in (
                ("image-transform", {"transform_operation": "crop", "transform_width": 3}),
                ("image-mask", {"mask_operation": "channel"}),
                ("image-color", {"color_operation": "quantize"}),
            ):
                body = dict(
                    input_image=self.encoded(),
                    controlnet={"type": kind},
                    response_format="b64_json",
                    **fields,
                )
                response = client.post("/images/generations", json=body)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(backend.images.call_args.args[1], {})
                for invalid in (
                    {"n": 2},
                    {"transform_width": True},
                    {"color_blend_factor": float("inf")},
                ):
                    if "color_blend_factor" not in invalid:
                        self.assertEqual(
                            client.post("/images/generations", json=body | invalid).status_code, 400
                        )
        backend.resolve_model.assert_not_called()
        with self.assertRaises(ValueError):
            ImageRequest(input_image="image", controlnet={"type": "image-transform"}, mask_expand=1)
        with self.assertRaises(ValueError):
            ImageRequest(input_image="image", color_operation="quantize")
        with self.assertRaises(ValueError):
            ImageRequest(input_image="image", controlnet={"type": "image-mask"}, mask_invert=1)
        self.assertEqual(
            validate_known({"transform_width": 0, "mask_expand": -1, "color_blend_factor": 0})[
                "mask_expand"
            ],
            -1,
        )

    def test_actual_capture_first_use_preparation_and_native_processors_need_no_model(self):
        for kind, fields, shape in (
            (
                "image-transform",
                {"transform_operation": "rotate", "transform_rotation": "90 degrees"},
                (3, 5),
            ),
            ("image-mask", {"mask_operation": "channel"}, (5, 3)),
            ("image-color", {"color_operation": "quantize", "color_quantize_colors": 4}, (5, 3)),
        ):
            data = dict(
                input_image=self.source,
                controlnet={"type": kind},
                prompt="Keep this UI draft",
                _source_generation={"prompt": "original recipe", "seed": 19},
                **fields,
            )
            captured = assets.capture_image_assets(
                data, self.manager, data["controlnet"], offline=True
            )
            self.assertEqual(captured[assets.KEY].artifacts, ())
            prepared = assets.prepare_image_assets(captured, check_cancelled=lambda: None)
            self.assertEqual((prepared["custom_width"], prepared["custom_height"]), shape)
            self.assertFalse(prepared["_sampling_captured"])
            self.assertEqual(
                prepared["_source_generation"], {"prompt": "original recipe", "seed": 19}
            )
            result = utilities.process(NS(), prepared, self.manager, lambda: None)[0]
            self.assertEqual(result.size, shape)
            metadata = utilities.output_metadata(prepared)
            self.assertEqual(metadata["controlnet"], {"type": kind})
            self.assertNotIn("base_model_name", metadata)
        self.manager.get_model_folders.assert_not_called()

    def test_actual_api_runtime_captures_every_group_and_checks_the_native_output_size(self):
        shared = NS(
            path_manager=self.manager,
            models=NS(offline=True),
            settings=NS(default_settings={}),
            state={},
        )
        worker = NS(
            prepare_image_data=Mock(
                side_effect=lambda data, **kw: assets.prepare_image_assets(
                    data, check_cancelled=lambda: None
                )
            ),
            check_interrupt=Mock(),
        )
        with (
            patch.dict(
                sys.modules,
                {"ruinedfooocus.app.state": shared, "ruinedfooocus.app.async_worker": worker},
            ),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(ruinedfooocus.app, "async_worker", worker, create=True),
            patch.object(api_runtime, "submit", side_effect=lambda operation: operation),
        ):
            for kind, fields, size in (
                (
                    "image-transform",
                    dict(transform_operation="rotate", transform_rotation="90 degrees"),
                    "3x5",
                ),
                ("image-mask", dict(mask_operation="channel"), "5x3"),
                ("image-color", dict(color_operation="quantize"), "5x3"),
            ):
                payload = dict(
                    input_image=self.encoded(), controlnet={"type": kind}, size=size, **fields
                )
                operation = api_runtime.images(payload, {})
                captured = operation.editor_request
                self.assertIn(utilities._handler(kind).KEY, captured)
                self.assertEqual(f"{captured['custom_width']}x{captured['custom_height']}", size)
                plan = captured["_effective_image_plan"]
                self.assertEqual(f"{plan.size.width}x{plan.size.height}", size)
                operation.prepare(NS())
                operation.validate()
                with self.assertRaisesRegex(ValueError, "captured size"):
                    api_runtime.images(payload | {"size": "9x9"}, {})
        self.manager.get_model_folders.assert_not_called()

    def test_ui_capture_ignores_inactive_uploads_without_clearing_the_ui_values(self):
        values = dict(
            input_image=self.source,
            controlnet={"type": "image-mask"},
            cn_selection="Adjust image",
            cn_type="Image-mask",
            mask_operation="channel",
            mask_image="stale upload",
            mask_reference="stale second mask",
        )
        captured, _ = utilities.capture_assets(values, self.manager)
        self.assertEqual(values["mask_image"], "stale upload")
        self.assertNotIn("mask_image", captured)
        with self.assertRaisesRegex(ValueError, "group changed"):
            utilities.capture_assets(values | {"cn_type": "Image-transform"}, self.manager)

    def test_precision_inline_decode_and_editor_original_never_use_preview_samples(self):
        values = np.array([[1, 257, 65534], [33, 4097, 60000]], dtype=np.uint16)
        original = self.root / "precision.png"
        Image.fromarray(values).save(original)
        decoded = api_runtime._decode_adjustment_image(
            base64.b64encode(original.read_bytes()).decode()
        )
        self.assertIsInstance(decoded, ImportedImage)
        np.testing.assert_array_equal(decoded.pixels.squeeze(), values)
        snapshot = self.store.start(original, owner=self.owner)
        context = dict(
            session_id=snapshot.session_id,
            revision=snapshot.revision,
            source_id=snapshot.current.artifact_id,
        )
        with patch.object(editor_session, "default_store", return_value=self.store):
            payload = api_runtime._editor_payload(
                dict(editor_context=context, controlnet={"type": "image-transform"}),
                owner=self.owner,
            )
        np.testing.assert_array_equal(payload["_editor_adjustment_source"].pixels.squeeze(), values)
        self.assertNotIn("input_image", payload)

    def test_file_capture_uses_oriented_still_samples(self):
        image = Image.fromarray(np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3))
        exif = image.getexif()
        exif[274] = 6
        path = self.root / "oriented.png"
        image.save(path, exif=exif)
        source = utilities.read_adjustment_source(path)
        self.assertEqual(source.size, (3, 5))
        np.testing.assert_array_equal(np.asarray(source)[..., :3], np.rot90(np.asarray(image), -1))

    def test_editor_adjustment_uses_original_freezes_controls_and_retains_prompt_drafts(self):
        snapshot = self.store.start(self.path, owner=self.owner)
        context = dict(
            session_id=snapshot.session_id,
            revision=snapshot.revision,
            source_id=snapshot.current.artifact_id,
        )
        presets = json.loads(resource_path("defaults", "powerup.default").read_text())
        controls = dict(
            prompt="literal draft",
            negative="negative draft",
            cn_selection="Adjust image",
            cn_type="Image-transform",
            transform_operation="crop",
            transform_width=3,
            transform_height=2,
            output_format="png",
            output_bit_depth="8-bit",
            output_color_space="sRGB",
        )
        data = build_library_request(
            controls,
            source_record(self.path),
            prompt=controls["prompt"],
            negative=controls["negative"],
            checkpoint=None,
            action="image-transform",
            resets={},
            features={},
            presets=presets,
            custom="Custom...",
            editor_context=context,
            editor_store=self.store,
            owner=self.owner,
        )
        self.assertEqual(data["input_image"].tobytes(), self.source.tobytes())
        self.assertEqual(data[editor_session.KEY].operation, "image-transform")
        captured = assets.capture_image_assets(data, self.manager, data["controlnet"], offline=True)
        controls["transform_width"] = 1
        result = utilities.process(NS(), captured, self.manager, lambda: None)[0]
        self.assertEqual(result.size, (3, 2))
        self.assertEqual(
            (controls["prompt"], controls["negative"]), ("literal draft", "negative draft")
        )
        self.assertEqual(self.path.read_bytes(), self.original_bytes)

    def test_editor_precision_adjustment_decodes_original_once_without_working_copy(self):
        from ruinedfooocus.features.gallery import editing_workflow
        from ruinedfooocus.features.image_processing import native_image_formats as formats

        values = np.array([[1, 257, 65534], [33, 4097, 60000]], dtype=np.uint16)
        path = self.root / "precision-adjustment.png"
        Image.fromarray(values).save(path)
        controls = dict(output_format="png", output_bit_depth="16-bit", output_color_space="sRGB")
        presets = json.loads(resource_path("defaults", "powerup.default").read_text())
        with (
            patch.object(
                editing_workflow,
                "working_source",
                side_effect=AssertionError("working-copy decode"),
            ),
            patch.object(
                formats, "convert_color", side_effect=AssertionError("working-copy conversion")
            ),
            patch.object(formats, "read_image", wraps=formats.read_image) as native_decode,
        ):
            data = build_library_request(
                controls,
                source_record(path),
                prompt="retained draft",
                negative="retained negative",
                checkpoint=None,
                action="image-transform",
                resets={},
                features={},
                presets=presets,
                custom="Custom...",
            )
        self.assertEqual(native_decode.call_count, 1)
        self.assertIsInstance(data["input_image"], ImportedImage)
        np.testing.assert_array_equal(data["input_image"].pixels.squeeze(), values)
        details = data["image_source"]
        self.assertEqual(
            (details["original_dtype"], details["working_dtype"], details["conversion"]),
            ("uint16", "uint16", "none"),
        )
        self.assertEqual((details["width"], details["height"]), (3, 2))
        self.assertEqual(
            (data["prompt"], data["negative"]), ("retained draft", "retained negative")
        )

    def owned_adjustment(self, path, action="image-transform", **controls):
        snapshot = self.store.start(path, owner=self.owner)
        return build_library_request(
            controls,
            source_record(path),
            prompt="retained draft",
            negative="retained negative",
            checkpoint=None,
            action=action,
            resets={},
            features={},
            presets=json.loads(resource_path("defaults", "powerup.default").read_text()),
            custom="Custom...",
            editor_context=dict(
                session_id=snapshot.session_id,
                revision=snapshot.revision,
                source_id=snapshot.current.artifact_id,
            ),
            editor_store=self.store,
            owner=self.owner,
        )

    def test_editor_untouched_default_preserves_png16_for_all_adjustment_captures(self):
        from ruinedfooocus.features.image_processing import native_image_formats as formats
        from test_native_image_formats import native_codecs

        values = np.array([[[1, 257, 65534], [33, 4097, 60000]]], dtype=np.uint16)
        path = self.root / "native-precision.png"
        with patch.object(formats, "_native", return_value=native_codecs()):
            formats.write_image(path, values, dict(output_format="png", output_bit_depth="16-bit"))
        original = path.read_bytes()
        for action, parameters in (
            ("image-transform", dict(transform_operation="flip")),
            ("image-mask", dict(mask_operation="channel", mask_channel="red")),
            ("image-color", dict(color_operation="morphology")),
        ):
            with self.subTest(action=action):
                controls = dict(
                    output_format="png",
                    output_bit_depth="8-bit",
                    output_color_space="sRGB",
                    _editor_output_explicit=False,
                    **parameters,
                )
                data = self.owned_adjustment(path, action, **controls)
                captured = assets.capture_image_assets(
                    data, self.manager, data["controlnet"], offline=True
                )
                request = captured[utilities._handler(action).KEY]
                self.assertEqual(request.source.dtype, "uint16")
                self.assertEqual(dict(request.output_settings)["output_bit_depth"], "16-bit")
                self.assertEqual(data["image_source"]["conversion"], "none")
                self.assertEqual(controls["output_bit_depth"], "8-bit")
        data = self.owned_adjustment(
            path, transform_operation="flip", _editor_output_explicit=False
        )
        captured = assets.capture_image_assets(data, self.manager, data["controlnet"], offline=True)
        result = utilities.process(NS(), captured, self.manager, lambda: None)[0]
        with patch.object(formats, "_native", return_value=native_codecs()):
            output = formats.write_image(self.root / "flipped16.png", result, data)
        decoded = formats.read_image(output).pixels
        self.assertEqual(decoded.dtype, np.uint16)
        np.testing.assert_allclose(
            decoded.astype(np.int32), values[:, ::-1].astype(np.int32), atol=1, rtol=0
        )
        self.assertTrue(np.any(decoded % 257 != 0))
        self.assertEqual(path.read_bytes(), original)

    def test_editor_float_hdr_samples_and_hdr_operation_choose_exr(self):
        from ruinedfooocus.features.image_processing import native_image_formats as formats
        from test_native_image_formats import native_codecs

        values = np.array([[[0.1234567, 1.25, 3.0], [-0.25, 0.3456789, 0.5]]], dtype=np.float32)
        path = self.root / "linear-hdr.exr"
        with patch.object(formats, "_native", return_value=native_codecs()):
            formats.write_image(
                path,
                values,
                dict(
                    output_format="exr",
                    output_bit_depth="32-bit float",
                    output_color_space="linear",
                ),
            )
        data = self.owned_adjustment(
            path, transform_operation="flip", _editor_output_explicit=False
        )
        captured = assets.capture_image_assets(data, self.manager, data["controlnet"], offline=True)
        request = captured[utilities._handler("image-transform").KEY]
        self.assertEqual(request.source.dtype, "float32")
        self.assertEqual(dict(request.output_settings)["output_format"], "exr")
        self.assertEqual(dict(request.output_settings)["output_color_space"], "linear")
        np.testing.assert_array_equal(
            np.frombuffer(request.source.pixels, np.float32).reshape(values.shape), values
        )
        data = self.owned_adjustment(
            self.path, "image-color", color_operation="sdr-to-hdr", _editor_output_explicit=False
        )
        captured = assets.capture_image_assets(data, self.manager, data["controlnet"], offline=True)
        choice = dict(captured[utilities._handler("image-color").KEY].output_settings)
        self.assertEqual((choice["output_format"], choice["output_color_space"]), ("exr", "HDR"))

    def test_editor_explicit_or_missing_output_intent_retains_strict_native_policy(self):
        values = np.array([[1, 257, 65534]], dtype=np.uint16)
        path = self.root / "explicit-precision.png"
        Image.fromarray(values).save(path)
        for marker in ({}, {"_editor_output_explicit": True}):
            with self.subTest(marker=marker):
                data = self.owned_adjustment(path, transform_operation="flip", **marker)
                self.assertNotIn("output_selection", data["image_source"])
                with self.assertRaisesRegex(ValueError, "PNG16 or EXR"):
                    assets.capture_image_assets(
                        data, self.manager, data["controlnet"], offline=True
                    )
        for explicit in (False, True):
            data = self.owned_adjustment(
                path,
                transform_operation="flip",
                output_format="exr",
                output_bit_depth="32-bit float",
                output_color_space="sRGB",
                _editor_output_explicit=explicit,
            )
            self.assertNotIn("output_selection", data["image_source"])
            captured = assets.capture_image_assets(
                data, self.manager, data["controlnet"], offline=True
            )
            self.assertEqual(
                dict(captured[utilities._handler("image-transform").KEY].output_settings)[
                    "output_format"
                ],
                "exr",
            )

    def test_editor_uint8_default_output_is_unchanged(self):
        data = self.owned_adjustment(
            self.path, transform_operation="flip", _editor_output_explicit=False
        )
        self.assertNotIn("output_selection", data["image_source"])
        captured = assets.capture_image_assets(data, self.manager, data["controlnet"], offline=True)
        choice = dict(captured[utilities._handler("image-transform").KEY].output_settings)
        self.assertEqual(
            (choice["output_format"], choice["output_bit_depth"], choice["output_color_space"]),
            ("png", "8-bit", "sRGB"),
        )

    def mask_result(self, *, size=None, metadata=True):
        snapshot = self.store.start(self.path, owner=self.owner)
        ticket = self.store.capture(
            snapshot.session_id,
            snapshot.revision,
            "image-mask",
            source_id=snapshot.current.artifact_id,
            owner=self.owner,
        )
        path = self.root / ("mask-" + snapshot.session_id + ".png")
        Image.new("RGB", size or self.source.size, (127, 127, 127)).save(path)
        execution = dict(
            alpha_policy="explicit grayscale selection mask extraction",
            geometry=dict(
                policy="native image-mask",
                source=list(self.source.size),
                result=list(size or self.source.size),
            ),
        )
        if metadata:
            execution["image_utility_source"] = dict(mask_role="inpaint-selection")
        return self.store.complete(ticket, path, execution=execution, owner=self.owner)

    def test_mask_handoff_atomically_restores_rgb_alpha_and_keeps_result_history(self):
        completed = self.mask_result()
        restored, mask = self.store.use_result_as_mask(
            completed.session_id,
            completed.revision,
            completed.current.artifact_id,
            owner=self.owner,
        )
        self.assertEqual(restored.current.artifact_id, completed.original.artifact_id)
        self.assertEqual(mask.source_image_id, restored.current.artifact_id)
        self.assertEqual(mask.source_frame_id, restored.current.frame_id)
        self.assertEqual(Path(restored.current.path).read_bytes(), self.original_bytes)
        with Image.open(mask.path) as image:
            self.assertEqual(
                (image.mode, image.size, image.getpixel((0, 0))), ("L", self.source.size, 127)
            )
        redone = self.store.redo(restored.session_id, restored.revision, owner=self.owner)
        self.assertEqual(redone.current.artifact_id, completed.current.artifact_id)

    def test_mask_handoff_rejects_foreign_stale_wrong_frame_and_unproven_results(self):
        completed = self.mask_result()
        with self.assertRaises(PermissionError):
            self.store.use_result_as_mask(
                completed.session_id,
                completed.revision,
                completed.current.artifact_id,
                owner=("foreign",),
            )
        with self.assertRaises(CancelledError):
            self.store.use_result_as_mask(
                completed.session_id,
                completed.revision - 1,
                completed.current.artifact_id,
                owner=self.owner,
            )
        with self.assertRaises(ValueError):
            wrong = self.mask_result(size=(2, 2))
            self.store.use_result_as_mask(
                wrong.session_id, wrong.revision, wrong.current.artifact_id, owner=self.owner
            )
        with self.assertRaises(ValueError):
            unproven = self.mask_result(metadata=False)
            self.store.use_result_as_mask(
                unproven.session_id,
                unproven.revision,
                unproven.current.artifact_id,
                owner=self.owner,
            )

    def test_worker_admits_explicit_mask_alpha_and_transform_geometry_before_history(self):
        for operation, size in (("image-mask", (5, 3)), ("image-transform", (3, 5))):
            snapshot = self.store.start(self.path, owner=self.owner)
            ticket = self.store.capture(
                snapshot.session_id,
                snapshot.revision,
                operation,
                source_id=snapshot.current.artifact_id,
                owner=self.owner,
            )
            path = self.root / (operation + ".png")
            metadata = dict(
                image_utility_source=dict(operation=operation, mask_role="inpaint-selection")
            )
            info = PngInfo()
            info.add_text("parameters", json.dumps(metadata))
            Image.new(
                "RGB" if operation == "image-mask" else "RGBA",
                size,
                (127, 127, 127) if operation == "image-mask" else (12, 24, 48, 127),
            ).save(path, pnginfo=info)
            item = NS(state={})
            stage = function(
                "ruinedfooocus/app/async_worker.py",
                "stage_editor_result",
                dict(
                    json=json,
                    current_job=lambda: item,
                    check_interrupt=Mock(),
                    validate_editor_data=Mock(),
                    _source_has_opaque_alpha=Mock(
                        side_effect=AssertionError("must preserve source alpha")
                    ),
                ),
            )
            stage(
                dict(
                    _editor_operation=ticket, _sampling_captured=False, transform_operation="rotate"
                ),
                [str(path)],
            )
            original, execution = item.state["_editor_publication"]
            completed = self.store.complete(ticket, original, execution=execution, owner=self.owner)
            self.assertEqual(completed.current.frame.size, size)
            if operation == "image-mask":
                self.assertEqual(
                    execution["alpha_policy"], "explicit grayscale selection mask extraction"
                )
            else:
                self.assertEqual(execution["geometry"]["policy"], "native rotate")

    def test_real_editor_selection_event_restores_original_and_sets_inpaint_context(self):
        import test_editor_workspace_events as fixture

        case = fixture.EditorWorkspaceEventTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        session = case.session("adjustment-selection-event")
        snapshot = case.store.start(self.path, owner=session.owner)
        ticket = case.store.capture(
            snapshot.session_id,
            snapshot.revision,
            "image-mask",
            source_id=snapshot.current.artifact_id,
            owner=session.owner,
        )
        path = self.root / "event-mask.png"
        Image.new("RGB", self.source.size, (127, 127, 127)).save(path)
        completed = case.store.complete(
            ticket,
            path,
            execution=dict(
                alpha_policy="selection", image_utility_source=dict(mask_role="inpaint-selection")
            ),
            owner=session.owner,
        )
        session.state[case.workspace.context._id] = case.workspace.public_context(completed)
        case.call(session, case.event(case.workspace.use_selection))
        context = case.value(session, case.workspace.context)
        self.assertEqual(context["source_id"], snapshot.original.artifact_id)
        self.assertIn("mask_id", context)
        self.assertEqual(case.value(session, case.workspace.action), "inpaint")


if __name__ == "__main__":
    unittest.main()
