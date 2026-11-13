"""The finite residual image operations share real controls and admission contracts."""

import asyncio
import base64
from io import BytesIO
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState
import numpy as np
from PIL import Image

import ruinedfooocus.app
from ruinedfooocus.app import api_runtime
from ruinedfooocus.core.settings_schema import validate_known
from ruinedfooocus.features.gallery.editor_session import EditorSessionStore
from ruinedfooocus.features.image_processing import still_image_utilities as tools
from ruinedfooocus.features.image_processing import depth_utilities as utilities
from ruinedfooocus.features.image_processing import ui_image_adjustments, ui_depth_utilities
from ruinedfooocus.features.image_processing import ui_native_formats, ui_color_operations
from ruinedfooocus.integrations.openai_api import ImageRequest


class ResidualImageToolsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.source = Image.new("RGBA", (64, 48), (30, 60, 90, 127))

    def mount(self, selected_kind="Image-transform", saved=None, choice="None"):
        controls = {}
        manager = NS(get_model_folders=lambda category: [], DOWNLOADABLE_FILES={})
        shared = NS(
            path_manager=manager,
            settings=NS(default_settings=saved or {}),
            add_ctrl=lambda key, component: controls.update({key: component}),
            get_ctrl=controls.get,
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
        ):
            with gr.Blocks(analytics_enabled=False) as demo:
                selection = gr.Dropdown(
                    choices=["None", "Adjust image", "MoGe depth", "Face landmarks"], value=choice
                )
                kind = gr.Dropdown(
                    choices=[value.capitalize() for value in tools.ADJUSTMENT_KINDS],
                    value=selected_kind,
                )
                image = gr.Image(type="pil")
                refresh = gr.Number(0)
                mapper = lambda choice, kind: (
                    tools.adjustment_kind(choice, kind)
                    or {"MoGe depth": "moge-depth", "Face landmarks": "face-landmarks"}.get(choice)
                )
                ui_image_adjustments.create_ui(selection, kind, mapper)
                ui_depth_utilities.create_ui(selection, kind, image, refresh, mapper)
                ui_native_formats.create_ui()
                ui_color_operations.bind_output_controls()
        self.controls, self.demo, self.state = controls, demo, SessionState(demo)

    def event(self, name):
        return next(
            index
            for index, event in self.demo.fns.items()
            if getattr(event.fn, "__name__", "") == name
        )

    def test_saved_active_mask_and_crop_factories_show_required_upload_without_change(self):
        for kind, saved, expected in (
            ("Image-mask", {"mask_operation": "grow"}, True),
            ("Image-mask", {"mask_operation": "channel"}, False),
            ("Image-transform", {"transform_operation": "crop-to-mask"}, True),
            ("Image-transform", {"transform_operation": "resize"}, False),
            ("Image-color", {"color_operation": "morphology"}, False),
        ):
            with self.subTest(kind=kind, saved=saved):
                self.mount(selected_kind=kind, saved=saved, choice="Adjust image")
                self.assertIsInstance(self.controls["mask_image"], gr.File)
                self.assertEqual(self.controls["mask_image"].visible, expected)
                event = self.demo.fns[self.event("adjustment_context")]
                self.assertTrue(event.outputs[0].visible)
                changed = self.call(
                    "adjustment_context",
                    [
                        "Adjust image",
                        kind,
                        saved.get("transform_operation", "resize"),
                        saved.get("mask_operation", "grow"),
                        saved.get("color_operation", "blend"),
                    ],
                )
                self.assertEqual(
                    changed[event.outputs.index(self.controls["mask_image"])]["visible"], expected
                )

    def call(self, name, values):
        return asyncio.run(self.demo.process_api(self.event(name), values, state=self.state))[
            "data"
        ]

    def encoded(self, image):
        stream = BytesIO()
        image.save(stream, format="PNG")
        return base64.b64encode(stream.getvalue()).decode()

    def test_all_residual_fields_are_reachable_and_only_relevant_uploads_are_captured(self):
        self.mount()
        expected = (
            set(tools.ADJUSTMENT_DEFAULTS) | set(tools.ADJUSTMENT_IMAGES) | set(utilities.FIELDS)
        )
        self.assertTrue(expected <= set(self.controls))
        operation = self.controls["color_operation"]
        self.assertEqual(
            {value for _, value in operation.choices},
            {"blend", "transfer", "quantize", "text-overlay", "morphology", "sdr-to-hdr"},
        )
        self.assertIn(
            "crop-to-mask", {value for _, value in self.controls["transform_operation"].choices}
        )
        context = self.demo.fns[self.event("adjustment_context")]
        mask_index = context.outputs.index(self.controls["mask_image"])
        crop = self.call(
            "adjustment_context",
            ["Adjust image", "Image-transform", "crop-to-mask", "grow", "blend"],
        )
        self.assertTrue(crop[mask_index]["visible"])
        ordinary = self.call(
            "adjustment_context", ["Adjust image", "Image-transform", "resize", "grow", "blend"]
        )
        self.assertFalse(ordinary[mask_index]["visible"])
        for operation in ("text-overlay", "morphology", "sdr-to-hdr"):
            self.assertEqual(
                tools.active_adjustment_images("image-color", {"color_operation": operation}), ()
            )
        self.assertEqual(
            tools.active_adjustment_images(
                "image-transform", {"transform_operation": "crop-to-mask"}
            ),
            ("mask_image",),
        )

    def test_hdr_mirror_changes_real_output_settings_only_after_explicit_input(self):
        self.mount()
        event = self.demo.fns[self.event("adjustment_context")]
        for field in ("output_format", "output_bit_depth", "output_color_space"):
            self.assertNotIn(self.controls[field], event.outputs)
        self.call(
            "adjustment_context", ["Adjust image", "Image-color", "resize", "grow", "sdr-to-hdr"]
        )
        self.assertEqual(self.controls["output_format"].value, "png")
        result = self.call("choose_hdr_output", ["32-bit float"])
        self.assertEqual([item["value"] for item in result[:3]], ["exr", "32-bit float", "HDR"])
        self.assertFalse(result[3]["visible"])
        for field in ("mask_image", "color_text"):
            self.assertNotIn(
                self.controls[field], self.demo.fns[self.event("choose_hdr_output")].outputs
            )

    def test_moge_and_face_contexts_expose_supported_choices_without_model_work(self):
        self.mount()
        normal = self.call("moge_output_context", ["normal", "perspective"])
        self.assertTrue(normal[1]["visible"])
        self.assertFalse(normal[2]["visible"])
        panorama = self.call("moge_output_context", ["normal", "panorama"])
        self.assertEqual(panorama[0]["value"], "depth")
        self.assertEqual([value for _, value in panorama[0]["choices"]], ["depth"])
        self.assertFalse(panorama[1]["visible"])
        self.assertTrue(panorama[2]["visible"])
        self.assertEqual(
            [row["visible"] for row in self.call("face_output_context", ["region-mask", "custom"])],
            [True, True],
        )
        self.assertFalse(self.call("face_output_context", ["overlay", "custom"])[0]["visible"])

    def test_typed_api_and_settings_share_native_scalar_validation(self):
        for operation, fields in (
            ("text-overlay", dict(color_text="bounded text", color_text_font_size=4.5)),
            ("morphology", dict(color_morphology_operation="close", color_morphology_kernel=4)),
            (
                "sdr-to-hdr",
                dict(
                    output_format="exr",
                    output_bit_depth="32-bit float",
                    output_color_space="HDR",
                    color_hdr_luminance=0,
                ),
            ),
        ):
            request = ImageRequest(
                input_image="source",
                controlnet={"type": "image-color"},
                color_operation=operation,
                **fields,
            )
            self.assertEqual(request.color_operation, operation)
        for invalid in (
            dict(color_operation="sdr-to-hdr"),
            dict(color_operation="morphology", color_morphology_kernel=True),
            dict(color_operation="text-overlay", color_text_outline=1),
            dict(color_operation="text-overlay", color_text="x" * 2049),
        ):
            with self.assertRaises(ValueError):
                ImageRequest(input_image="source", controlnet={"type": "image-color"}, **invalid)
        for kind, values in (
            (
                "moge-depth",
                dict(
                    moge_output="normal",
                    moge_projection="perspective",
                    moge_normal_convention="directx",
                ),
            ),
            (
                "moge-depth",
                dict(moge_output="depth", moge_projection="panorama", moge_panorama_batch_size=1),
            ),
            (
                "face-landmarks",
                dict(
                    face_output="region-mask",
                    face_mask_regions="custom",
                    face_mask_face_oval=False,
                    face_mask_lips=True,
                ),
            ),
        ):
            request = ImageRequest(
                input_image="source",
                controlnet={"type": kind},
                image_utility_model="registered.safetensors",
                **values,
            )
            validated = validate_known(values)
            for field, value in values.items():
                self.assertEqual(getattr(request, field), validated[field])
        with self.assertRaises(ValueError):
            ImageRequest(
                input_image="source",
                controlnet={"type": "moge-depth"},
                image_utility_model="registered",
                moge_output="normal",
                moge_projection="panorama",
            )
        with self.assertRaises(ValueError):
            validate_known(
                dict(
                    face_output="region-mask",
                    face_mask_regions="custom",
                    **{
                        key: False
                        for key in utilities.FACE_OUTPUT_DEFAULTS
                        if key.startswith("face_mask_") and key != "face_mask_regions"
                    },
                )
            )

    def test_actual_api_capture_and_cpu_processing_freeze_text_morphology_hdr_and_crop(self):
        manager = NS(get_model_folders=Mock(side_effect=AssertionError("model lookup")))
        shared = NS(
            path_manager=manager,
            models=NS(offline=True),
            settings=NS(default_settings=dict(image_device="CPU", image_backend="cpu")),
            state={},
        )
        mask = Image.new("L", self.source.size, 0)
        mask.paste(255, (3, 4, 13, 16))
        cases = [
            (
                "image-transform",
                dict(transform_operation="crop-to-mask", mask_image=self.encoded(mask)),
                (10, 12),
            ),
            (
                "image-color",
                dict(color_operation="text-overlay", color_text="CPU", color_text_font_size=15),
                self.source.size,
            ),
            (
                "image-color",
                dict(color_operation="morphology", color_morphology_kernel=4),
                self.source.size,
            ),
            (
                "image-color",
                dict(
                    color_operation="sdr-to-hdr",
                    output_format="exr",
                    output_bit_depth="32-bit float",
                    output_color_space="HDR",
                ),
                self.source.size,
            ),
        ]
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
            patch.object(api_runtime, "submit", side_effect=lambda operation: operation),
        ):
            for kind, fields, size in cases:
                operation = api_runtime.images(
                    dict(
                        input_image=self.encoded(self.source), controlnet={"type": kind}, **fields
                    ),
                    {},
                )
                data = operation.editor_request
                request = data[tools._handler(kind).KEY]
                fields.clear()
                result = tools.process(NS(), data, manager, lambda: None)[0]
                shape = (
                    result.size
                    if isinstance(result, Image.Image)
                    else (result.shape[1], result.shape[0])
                )
                self.assertEqual(shape, size)
                self.assertEqual(
                    (
                        data["_effective_image_plan"].size.width,
                        data["_effective_image_plan"].size.height,
                    ),
                    size,
                )
                self.assertEqual(tools.output_metadata(data)["controlnet"], {"type": kind})
                self.assertTrue(request.parameters)
        manager.get_model_folders.assert_not_called()

    def test_proven_face_region_result_attaches_soft_mask_and_restores_exact_rgba_source(self):
        source = self.root / "source.png"
        self.source.save(source)
        original = source.read_bytes()
        store = EditorSessionStore(self.root / "sessions")
        owner = ("fixture", "face")
        snapshot = store.start(source, owner=owner)
        ticket = store.capture(
            snapshot.session_id,
            snapshot.revision,
            "face-landmarks",
            source_id=snapshot.current.artifact_id,
            owner=owner,
        )
        result = self.root / "face-mask.png"
        Image.new("RGBA", self.source.size, (111, 111, 111, 127)).save(result)
        done = store.complete(
            ticket,
            result,
            execution=dict(image_utility_source=dict(mask_role="inpaint-selection")),
            owner=owner,
        )
        restored, mask = store.use_result_as_mask(
            done.session_id, done.revision, done.current.artifact_id, owner=owner
        )
        self.assertEqual(Path(restored.current.path).read_bytes(), original)
        self.assertEqual(mask.source_frame_id, snapshot.original.frame_id)
        with Image.open(mask.path) as image:
            self.assertEqual((image.mode, image.getpixel((0, 0))), ("L", 111))

    def test_editor_crop_uses_its_owned_selection_without_replacing_source_or_uploads(self):
        from ruinedfooocus.features.gallery import editor_session

        source = self.root / "crop-source.png"
        self.source.save(source)
        selection = self.root / "crop-selection.png"
        mask = Image.new("L", self.source.size, 0)
        mask.paste(77, (2, 3, 12, 15))
        mask.save(selection)
        store = EditorSessionStore(self.root / "crop-sessions")
        owner = ("fixture", "crop")
        snapshot = store.start(source, owner=owner)
        bound = store.bind_mask(
            snapshot.session_id, selection, snapshot.current.artifact_id, owner=owner
        )
        context = dict(
            session_id=snapshot.session_id,
            revision=snapshot.revision,
            source_id=snapshot.current.artifact_id,
            mask_id=bound.artifact_id,
        )
        with patch.object(editor_session, "default_store", return_value=store):
            payload = api_runtime._editor_payload(
                dict(
                    editor_context=context,
                    controlnet={"type": "image-transform"},
                    transform_operation="crop-to-mask",
                ),
                owner=owner,
            )
        self.assertIn("_editor_adjustment_mask", payload)
        self.assertEqual(np.asarray(payload["_editor_adjustment_mask"])[3, 2, 0], 77)
        self.assertEqual(
            store.snapshot(snapshot.session_id, owner=owner).current.artifact_id,
            snapshot.current.artifact_id,
        )

    def test_missing_registered_face_picker_is_passive_and_uses_existing_first_use_path(self):
        destination = self.root / "detection/mediapipe_face_fp32.safetensors"
        destination.parent.mkdir()
        key = "detection/mediapipe_face_fp32.safetensors"
        manager = NS(
            get_model_folders=lambda category: [destination.parent],
            DOWNLOADABLE_FILES={key: {}},
            preparation_request=Mock(return_value=NS(destination=destination)),
        )
        choices = utilities.component_choices(manager, "face-landmarks", "image_utility_model")
        self.assertEqual([value for _, value in choices], [str(destination)])
        self.assertFalse(destination.exists())
        manager.preparation_request.assert_called_once_with(key)


if __name__ == "__main__":
    unittest.main()
