"""Real native precision/ICC inputs become private model copies, never history edits."""

from concurrent.futures import CancelledError
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageCms
from PIL.PngImagePlugin import PngInfo

from test_native_image_formats import native_codecs

from ruinedfooocus.features.gallery.editor_session import EditorSessionStore
from ruinedfooocus.features.gallery.editing_workflow import working_image
from ruinedfooocus.features.image_editing import image_workflows as workflows
from ruinedfooocus.features.image_editing.mask_handoffs import capture_mask_handoff, inpaint_inputs
from ruinedfooocus.features.image_processing import native_image_formats as formats
from ruinedfooocus.features.image_processing import image_composition
from ruinedfooocus.features.image_processing.still_image_utilities import read_adjustment_source


class EditorWorkingCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = native_codecs()

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.owner = ("working-copy", "owner")
        self.store = EditorSessionStore(self.root / "sessions")
        self.enterContext(patch.object(formats, "_native", return_value=self.native))

    def native_source(self, pixels, name="source.png", color="sRGB"):
        path = self.root / name
        formats.write_image(
            path,
            pixels,
            dict(
                output_format=path.suffix[1:],
                output_bit_depth="32-bit float" if path.suffix == ".exr" else "16-bit",
                output_color_space=color,
            ),
            {"prompt": "original metadata — деталі", "seed": 7},
        )
        return path

    def step(self, snapshot, action="prompt-edit", parameters=None, **kwargs):
        return workflows.capture_step(
            self.store,
            snapshot.session_id,
            snapshot.revision,
            action,
            parameters or {},
            owner=self.owner,
            **kwargs,
        )

    def test_png16_prompt_edit_is_automatic_exact_frame_and_undo_keeps_original(self):
        path = self.root / "source.png"
        pixels = np.arange(73 * 121, dtype=np.uint16).reshape(121, 73)
        metadata = PngInfo()
        metadata.add_text("parameters", json.dumps({"prompt": "початкові дані", "seed": 7}))
        Image.fromarray(pixels).save(path, pnginfo=metadata)
        original = path.read_bytes()
        snapshot = self.store.start(path, provenance={"prompt": "original"}, owner=self.owner)
        step = self.step(
            snapshot,
            parameters={"prompt": "Make her more detailed", "base_model_name": "Janku.safetensors"},
            family="SDXL",
        )
        request = workflows.worker_request(step, self.store, owner=self.owner)
        expected = (pixels.astype(np.float32) / 65535 * 255).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(request["input_image"])[..., 0], expected)
        self.assertEqual(request["input_image"].size, (73, 121))
        self.assertEqual(request["prompt"], "Make her more detailed")
        self.assertEqual(request["controlnet"], {"type": "img2img", "denoise": 0.3})
        self.assertEqual(request["image_source"]["original_dtype"], "uint16")
        self.assertEqual(request["image_source"]["working_color_space"], "sRGB")
        self.assertEqual(step.ticket.source, snapshot.original)
        request["input_image"].putpixel((0, 0), (255, 0, 0, 255))
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(
            self.store.snapshot(snapshot.session_id, owner=self.owner).revision, snapshot.revision
        )
        result = self.root / "result.png"
        Image.new("RGBA", (73, 121)).save(result)
        execution = workflows.execution_metadata(step, (73, 121))
        self.assertIn("request-local", execution["precision_conversion"])
        completed = self.store.complete(step.ticket, result, execution=execution, owner=self.owner)
        restored = self.store.undo(snapshot.session_id, completed.revision, owner=self.owner)
        self.assertEqual(restored.current, snapshot.original)
        self.assertEqual(path.read_bytes(), original)

    def test_native_rgba16_alpha_is_retained_for_worker_and_rgb_result(self):
        pixels = np.array([[[0.13, 0.42, 0.77, 0.33333], [0.9, 0.2, 0.5, 0]]], dtype=np.float32)
        path = self.native_source(pixels)
        snapshot = self.store.start(path, owner=self.owner)
        step = self.step(snapshot)
        request = workflows.worker_request(step, self.store, owner=self.owner)
        imported = formats.read_image(path)
        expected = (imported.pixels[..., 3].astype(np.float32) / 65535 * 255).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(request["input_image"])[..., 3], expected)
        result = workflows.finish_pixels(
            step, self.store, Image.new("RGB", (2, 1)), owner=self.owner
        )
        np.testing.assert_array_equal(np.asarray(result)[..., 3], expected)
        np.testing.assert_array_equal(formats.read_image(path).pixels, imported.pixels)

    def test_linear_exr_uses_native_color_conversion_and_adjustments_keep_precision(self):
        pixels = np.array(
            [[[0.125, 0.25, 0.5, 0.33333], [0.01, 0.2, 1.25, 0.75]]], dtype=np.float32
        )
        path = self.native_source(pixels, "source.exr", "linear")
        original = path.read_bytes()
        native = formats.read_image(path)
        snapshot = self.store.start(path, owner=self.owner)
        request = workflows.worker_request(self.step(snapshot), self.store, owner=self.owner)
        expected = (
            formats.convert_color(native.pixels, "linear", "sRGB").clamp(0, 1).numpy() * 255
        ).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(request["input_image"]), expected)
        adjusted = read_adjustment_source(path)
        self.assertEqual(adjusted.pixels.dtype, np.float32)
        self.assertEqual(adjusted.color_space, "linear")
        np.testing.assert_array_equal(adjusted.pixels, native.pixels)
        self.assertEqual(path.read_bytes(), original)

    def test_embedded_icc_is_converted_and_alpha_is_independent(self):
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        path = self.root / "icc.png"
        original = Image.new("RGBA", (13, 17), (37, 119, 211, 83))
        original.save(path, icc_profile=profile)
        before = path.read_bytes()
        snapshot = self.store.start(path, owner=self.owner)
        self.assertEqual(snapshot.current.frame.color_space, "embedded ICC")
        request = workflows.worker_request(self.step(snapshot), self.store, owner=self.owner)
        np.testing.assert_array_equal(np.asarray(request["input_image"]), np.asarray(original))
        self.assertEqual(request["image_source"]["original_color_space"], "embedded ICC")
        self.assertEqual(path.read_bytes(), before)

        lab_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB"))
        lab = Image.new("LAB", (13, 17), (135, 140, 110))
        lab_path = self.root / "lab.tiff"
        lab.save(lab_path, icc_profile=lab_profile.tobytes())
        image, information = working_image(lab_path)
        expected = ImageCms.profileToProfile(
            lab, lab_profile, ImageCms.createProfile("sRGB"), outputMode="RGB"
        )
        np.testing.assert_array_equal(np.asarray(image)[..., :3], np.asarray(expected))
        self.assertEqual(information["working_color_space"], "sRGB")

    def test_precision_source_mask_uses_exact_owned_frame_and_soft_coverage(self):
        path = self.native_source(np.full((17, 13, 4), 0.375, dtype=np.float32))
        snapshot = self.store.start(path, owner=self.owner)
        mask_path = self.root / "mask.png"
        Image.new("L", (13, 17), 93).save(mask_path)
        mask = self.store.bind_mask(
            snapshot.session_id, mask_path, snapshot.current.artifact_id, owner=self.owner
        )
        handoff = capture_mask_handoff(
            self.store,
            snapshot.session_id,
            snapshot.revision,
            snapshot.current.artifact_id,
            mask.artifact_id,
            owner=self.owner,
            choice="union",
        )
        inputs = inpaint_inputs(self.store, handoff, owner=self.owner)
        self.assertEqual(inputs["main_view"].size, (13, 17))
        self.assertEqual(inputs["inpaint_view"]["layers"][0].getchannel("A").getextrema(), (93, 93))
        step = self.step(snapshot, "inpaint", mask=handoff)
        request = workflows.worker_request(step, self.store, owner=self.owner)
        self.assertEqual(request["main_view"].size, (13, 17))
        self.assertEqual(step.ticket.source, snapshot.original)
        with self.assertRaises(PermissionError):
            workflows.worker_request(step, self.store, owner=("working-copy", "other"))

    def test_additional_precision_images_are_private_and_changed_sources_reject(self):
        source = self.root / "base.png"
        Image.new("RGBA", (13, 17)).save(source)
        snapshot = self.store.start(source, owner=self.owner)
        foreground = self.native_source(np.full((9, 7, 4), 0.25, dtype=np.float32))
        before = foreground.read_bytes()
        ref = self.store.bind_image(snapshot.session_id, foreground, owner=self.owner)
        step = self.step(snapshot, "compose", {"foreground_id": ref.artifact_id})
        request = workflows.worker_request(step, self.store, owner=self.owner)
        self.assertEqual(
            [image.size for image in request["composition_images"]], [(13, 17), (7, 9)]
        )
        self.assertEqual(foreground.read_bytes(), before)
        foreground.write_bytes(before + b"changed")
        with self.assertRaises((ValueError, CancelledError)):
            workflows.worker_request(step, self.store, owner=self.owner)

    def test_direct_background_composition_rebinds_owned_roles_and_fits_current_canvas(self):
        path = self.root / "foreground.png"
        foreground = Image.new("RGBA", (20, 10), (200, 0, 0, 128))
        foreground.save(path)
        snapshot = self.store.start(path, owner=self.owner)
        background = self.root / "background.png"
        Image.new("RGBA", (5, 10), (0, 0, 200, 255)).save(background)
        ref = self.store.bind_image(snapshot.session_id, background, owner=self.owner)
        step = self.step(snapshot, "compose", {"background_id": ref.artifact_id})
        request = workflows.worker_request(step, self.store, owner=self.owner)
        # Rebinding must discard arbitrary request arrays and rebuild from both IDs.
        request["composition_images"] = [Image.new("RGBA", (1, 1)), Image.new("RGBA", (1, 1))]
        with (
            patch(
                "ruinedfooocus.features.gallery.editor_session.default_store",
                return_value=self.store,
            ),
            patch("ruinedfooocus.app.resource_lifecycle.current_owner", return_value=self.owner),
        ):
            captured = image_composition.capture(request, "image-composite")
        packet = image_composition.captured(captured)
        self.assertEqual(packet.output_size, (20, 10))
        self.assertEqual(packet.images[0].image().getpixel((0, 0)), (0, 0, 0, 0))
        self.assertEqual(packet.images[0].image().getpixel((8, 0)), (0, 0, 200, 255))
        np.testing.assert_array_equal(np.asarray(packet.images[1].image()), np.asarray(foreground))
        result = image_composition.process(captured, lambda: None)
        self.assertEqual(result.getpixel((0, 0)), (200, 0, 0, 128))
        self.assertEqual(result.getpixel((8, 0)), (100, 0, 100, 255))
        self.assertEqual(self.store.snapshot(snapshot.session_id, owner=self.owner).steps, ())
        self.assertEqual(step.ticket.source_id, snapshot.current.artifact_id)
        self.assertEqual(step.ticket.additional_images[0].artifact_id, ref.artifact_id)
        self.assertIn(
            "no crop or stretch",
            workflows.execution_metadata(step, result.size)["background_fit"]["policy"],
        )

    def test_existing_foreground_composition_route_and_conflicting_role_rejection(self):
        path = self.root / "background.png"
        Image.new("RGBA", (20, 10), (0, 0, 200, 255)).save(path)
        snapshot = self.store.start(path, owner=self.owner)
        foreground = self.root / "foreground.png"
        Image.new("RGBA", (5, 10), (200, 0, 0, 128)).save(foreground)
        ref = self.store.bind_image(snapshot.session_id, foreground, owner=self.owner)
        with self.assertRaisesRegex(ValueError, "not both"):
            self.step(
                snapshot,
                "compose",
                {"background_id": ref.artifact_id, "foreground_id": ref.artifact_id},
            )
        step = self.step(snapshot, "compose", {"foreground_id": ref.artifact_id})
        request = workflows.worker_request(step, self.store, owner=self.owner)
        with (
            patch(
                "ruinedfooocus.features.gallery.editor_session.default_store",
                return_value=self.store,
            ),
            patch("ruinedfooocus.app.resource_lifecycle.current_owner", return_value=self.owner),
        ):
            captured = image_composition.capture(request, "image-composite")
        packet = image_composition.captured(captured)
        self.assertEqual(packet.images[0].image().getpixel((0, 0)), (0, 0, 200, 255))
        self.assertEqual(packet.images[1].size, (5, 10))
        self.assertEqual(image_composition.process(captured, lambda: None).size, (20, 10))

    def test_mask_adjustment_precision_handoff_preserves_photo_and_mask_result(self):
        path = self.native_source(np.full((17, 13, 4), 0.375, dtype=np.float32))
        snapshot = self.store.start(path, owner=self.owner)
        ticket = self.store.capture(
            snapshot.session_id,
            snapshot.revision,
            "image-mask",
            {},
            source_id=snapshot.current.artifact_id,
            owner=self.owner,
        )
        result = self.native_source(np.full((17, 13, 3), 0.5, dtype=np.float32), "mask16.png")
        mask_bytes = result.read_bytes()
        completed = self.store.complete(
            ticket,
            result,
            owner=self.owner,
            execution={
                "image_utility_source": {"mask_role": "inpaint-selection"},
                "alpha_policy": "scalar selection",
            },
        )
        selected, mask = self.store.use_result_as_mask(
            completed.session_id,
            completed.revision,
            completed.current.artifact_id,
            owner=self.owner,
        )
        self.assertEqual(selected.current, snapshot.original)
        self.assertEqual(result.read_bytes(), mask_bytes)
        with Image.open(mask.path) as image:
            self.assertEqual(image.mode, "L")
            self.assertEqual(image.getextrema(), (127, 127))

    def test_native_segmentation_selection_can_keep_a_precision_photo_as_current(self):
        path = self.native_source(np.full((17, 13, 4), 0.375, dtype=np.float32))
        original = path.read_bytes()
        snapshot = self.store.start(path, owner=self.owner)
        ticket = self.store.capture(
            snapshot.session_id,
            snapshot.revision,
            "segment",
            {},
            source_id=snapshot.current.artifact_id,
            owner=self.owner,
        )
        mask = self.root / "object-mask.png"
        Image.new("L", (13, 17), 93).save(mask)
        results = self.store.capture_derived_results(ticket, [mask], owner=self.owner)
        completed = self.store.complete_derived_results(
            ticket, results, output="mask", owner=self.owner
        )
        self.assertTrue(completed.derived_results["selection_eligible"])
        prepared = self.store.prepare_derived_selection(
            completed.session_id, completed.revision, [0], owner=self.owner
        )
        selected, selection = self.store.commit_derived_selection(prepared, owner=self.owner)
        self.assertEqual(selected.current, snapshot.original)
        self.assertEqual(selection.source_image_id, snapshot.original.artifact_id)
        handoff = capture_mask_handoff(
            self.store,
            selected.session_id,
            selected.revision,
            selected.current.artifact_id,
            selection.artifact_id,
            owner=self.owner,
            choice="union",
        )
        self.assertEqual(
            inpaint_inputs(self.store, handoff, owner=self.owner)["main_view"].size, (13, 17)
        )
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
