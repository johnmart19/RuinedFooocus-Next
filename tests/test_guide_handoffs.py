"""Actual owned Editor maps, separate source/guide capture and existing geometry."""

from concurrent.futures import CancelledError
from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from ruinedfooocus.features.gallery import editor_session as editor
from ruinedfooocus.features.gallery.actions import source_record
from ruinedfooocus.features.gallery.editing_workflow import build_library_request
from ruinedfooocus.features.image_editing import guide_handoffs as guides
from ruinedfooocus.features.image_editing.mask_handoffs import capture_mask_handoff, inpaint_inputs


@dataclass(frozen=True)
class CompatibleControl:
    recipe: str = "ordinary"
    union_type: str = "depth"
    invert_channels: tuple = ()

    @property
    def controls(self):
        return dict(
            type="depth",
            strength=0.8,
            start=0.1,
            stop=0.9,
            depth_preprocess="none",
            recipe="ordinary",
            union_type="depth",
        )

    def verify(self, manager, family, controls, spec):
        if controls != self.controls:
            raise ValueError("The captured control changed")

    def metadata(self):
        return dict(kind="depth", family="SDXL", verified=True)


class GuideHandoffTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.owner, self.other = ("guide-owner",), ("foreign-owner",)
        self.store = editor.EditorSessionStore(self.root / "sessions")
        self.source = np.arange(5 * 7 * 4, dtype=np.uint8).reshape(5, 7, 4)
        self.source[..., 3] = np.array([0, 17, 64, 128, 191, 239, 255], dtype=np.uint8)
        self.source_path = self.root / "original.png"
        Image.fromarray(self.source).save(self.source_path)
        self.original_sha = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        self.initial = self.store.start(
            self.source_path, provenance=dict(prompt="original prompt", seed=123), owner=self.owner
        )

    def result(self, *, normal=False, info=None, operation="moge-depth", size=None):
        snapshot = self.store.snapshot(self.initial.session_id, owner=self.owner)
        ticket = self.store.capture(
            snapshot.session_id,
            snapshot.revision,
            operation,
            source_id=snapshot.current.artifact_id,
            owner=self.owner,
        )
        size = size or (7, 5)
        if normal:
            ramp = np.arange(size[0] * size[1], dtype=np.uint8).reshape(size[1], size[0])
            image = Image.fromarray(np.stack((ramp * 5, ramp + 70, 255 - ramp), axis=-1), "RGB")
        else:
            pixels = np.arange(size[0] * size[1], dtype=np.uint8).reshape(size[1], size[0]) * 7
            image = Image.fromarray(pixels, "L").convert("RGB")
        image = image.convert("RGBA")
        image.putalpha(Image.fromarray(self.source[..., 3]).crop((0, 0, *size)))
        path = self.root / ("normal.png" if normal else "depth.png")
        image.save(path)
        details = dict(
            operation=operation,
            width=size[0],
            height=size[1],
            moge_output="normal" if normal else "depth",
            projection="perspective",
            convention="native-opengl" if normal else "white-near; invalid-black",
        )
        if normal:
            details.update(normalization="native-normal*0.5+0.5", normal_frame="native-camera")
        details.update(info or {})
        execution = dict(image_utility_source=details)
        if size != (7, 5):
            execution["geometry"] = dict(
                policy="explicit fixture crop", source=[7, 5], result=list(size)
            )
        return self.store.complete(ticket, path, execution=execution, owner=self.owner)

    def adopt(self, **kwargs):
        result = self.result(**kwargs)
        restored, handoff = guides.restore_result_guide(
            self.store,
            result.session_id,
            result.revision,
            result.current.artifact_id,
            owner=self.owner,
        )
        return result, restored, handoff

    def capture(self, handoff, *, compatibility=None, **values):
        data = dict(
            control_guide_denoise=0.37,
            prompt="change the lighting",
            negative="literal negative",
            custom_width=37,
            custom_height=29,
            image_resize_mode="pad",
            controlnet=dict(
                type="depth", strength=0.8, start=0.1, stop=0.9, depth_preprocess="none"
            ),
        )
        data.update(values)
        # Compatibility has its own real-header/source-identity suite. This suite
        # isolates ownership/capture; it does not certify an invented model file.
        with patch(
            "ruinedfooocus.features.image_editing.guide_control_compat.validate_guide_control",
            return_value=compatibility or CompatibleControl(),
        ) as validate:
            result = guides.capture_guide_inputs(
                self.store, handoff, owner=self.owner, data=data, manager=object(), family="SDXL"
            )
        validate.assert_called_once()
        return result, data

    def test_atomic_restore_preserves_owned_original_alpha_history_and_provenance(self):
        result = self.result()
        spec = guides.describe_result_guide(
            self.store,
            result.session_id,
            result.revision,
            result.current.artifact_id,
            owner=self.owner,
        )
        self.assertEqual((spec.kind, spec.convention), ("depth", "white-near; invalid-black"))
        restored, handoff = guides.restore_result_guide(
            self.store,
            result.session_id,
            result.revision,
            result.current.artifact_id,
            owner=self.owner,
        )
        self.assertEqual(restored.revision, 3)
        self.assertEqual(restored.current.artifact_id, self.initial.current.artifact_id)
        self.assertEqual(restored.source_generation["prompt"], "original prompt")
        self.assertEqual(len(restored.steps), 1)
        images = guides.handoff_images(self.store, handoff, owner=self.owner)
        np.testing.assert_array_equal(np.asarray(images["main_view"]), self.source)
        self.assertFalse(np.array_equal(np.asarray(images["input_image"]), self.source[..., :3]))
        self.assertEqual(
            set(handoff.public_context()),
            {"session_id", "revision", "source_id", "additional_image_ids"},
        )
        self.assertNotIn(str(self.root), str(handoff.metadata()))
        self.assertEqual(
            hashlib.sha256(self.source_path.read_bytes()).hexdigest(), self.original_sha
        )

    def test_foreign_stale_frame_and_convention_changes_are_rejected(self):
        result, restored, handoff = self.adopt(normal=True)
        with self.assertRaises(PermissionError):
            guides.handoff_images(self.store, handoff, owner=self.other)
        with self.assertRaises(ValueError):
            guides.handoff_images(
                self.store, replace(handoff, source_frame_id="changed"), owner=self.owner
            )
        with self.assertRaises(ValueError):
            guides.handoff_images(
                self.store,
                replace(handoff, spec=replace(handoff.spec, convention="native-directx")),
                owner=self.owner,
            )
        self.store.redo(restored.session_id, restored.revision, owner=self.owner)
        with self.assertRaises(CancelledError):
            guides.handoff_images(self.store, handoff, owner=self.owner)
        with self.assertRaises(CancelledError):
            guides.restore_result_guide(
                self.store,
                result.session_id,
                result.revision,
                result.current.artifact_id,
                owner=self.owner,
            )

    def test_pending_and_wrong_frame_never_restore_original(self):
        result = self.result(size=(6, 5))
        with self.assertRaisesRegex(ValueError, "source frame"):
            guides.restore_result_guide(
                self.store,
                result.session_id,
                result.revision,
                result.current.artifact_id,
                owner=self.owner,
            )
        self.assertEqual(
            self.store.snapshot(result.session_id, owner=self.owner).revision, result.revision
        )

    def test_arbitrary_image_or_unproven_normal_convention_is_not_eligible(self):
        result = self.result(normal=True, info=dict(convention="unspecified-normal"))
        with self.assertRaisesRegex(ValueError, "camera-normal"):
            guides.describe_result_guide(
                self.store,
                result.session_id,
                result.revision,
                result.current.artifact_id,
                owner=self.owner,
            )

    def test_pending_operation_rejects_handoff_without_changing_revision(self):
        result = self.result()
        ticket = self.store.capture(
            result.session_id,
            result.revision,
            "edit",
            source_id=result.current.artifact_id,
            owner=self.owner,
        )
        with self.assertRaisesRegex(ValueError, "pending"):
            guides.restore_result_guide(
                self.store,
                result.session_id,
                result.revision,
                result.current.artifact_id,
                owner=self.owner,
            )
        self.assertEqual(
            self.store.snapshot(result.session_id, owner=self.owner).revision, result.revision
        )
        self.assertTrue(self.store.cancel(ticket, owner=self.owner))

    def test_capture_ignores_display_widgets_and_freezes_intent_without_rewriting_draft(self):
        _, _, handoff = self.adopt()
        captured, original_data = self.capture(handoff, input_image=Image.new("RGB", (1, 1), "red"))
        np.testing.assert_array_equal(np.asarray(captured["input_image"]), self.source)
        for field in (
            "prompt",
            "negative",
            "custom_width",
            "custom_height",
            "image_resize_mode",
            guides.DENOISE_FIELD,
        ):
            self.assertEqual(captured[field], original_data[field])
        request = captured[guides.KEY]
        original_data["controlnet"]["strength"] = 0.01
        captured["input_image"].paste("white", (0, 0, 7, 5))
        np.testing.assert_array_equal(np.asarray(guides.request_image(request.source)), self.source)
        self.assertEqual(request.controls()["strength"], 0.8)
        self.assertEqual(request.denoise, 0.37)
        self.assertIs(
            guides.validate_request(
                captured, store=self.store, owner=self.owner, manager=object(), family="SDXL"
            ),
            request,
        )

    def test_existing_geometry_keeps_source_guide_alignment_and_restores_original_alpha(self):
        _, _, handoff = self.adopt(normal=True)
        captured, _ = self.capture(handoff)
        request = captured[guides.KEY]
        for mode in ("preserve", "pad", "crop"):
            with self.subTest(mode=mode):
                source, guide, frame, alpha = guides.prepare_reference_pair(request, (7, 5), mode)
                self.assertEqual(source.shape, guide.shape)
                self.assertFalse(torch.equal(source, guide))
                from ruinedfooocus.features.image_editing.image_geometry import (
                    restore_reference_frame,
                )

                decoded = restore_reference_frame(source, frame)
                final = guides.restore_source_alpha(decoded, alpha, frame)
                expected = restore_reference_frame(alpha, frame)
                np.testing.assert_array_equal(final[..., 3:].numpy(), expected.numpy())

    def test_explicit_size_override_pads_one_uniform_source_and_guide_frame(self):
        _, _, handoff = self.adopt(normal=True)
        captured, _ = self.capture(handoff)
        request = captured[guides.KEY]
        source, guide, frame, _ = guides.prepare_reference_pair(request, (13, 17), "pad")
        self.assertEqual((frame["width"], frame["height"]), (13, 17))
        self.assertEqual((frame["content_width"], frame["content_height"]), (13, 9))
        self.assertEqual(source.shape, guide.shape)
        self.assertEqual(frame["content_top"], 4)
        _, _, default, _ = guides.prepare_reference_pair(request, (40, 40))
        self.assertEqual((default["width"], default["height"]), (7, 5))

    def test_explicit_zero_fractional_and_full_denoise_are_preserved(self):
        _, _, handoff = self.adopt()
        for value in (0, 0.35, 1):
            with self.subTest(denoise=value):
                captured, original = self.capture(handoff, control_guide_denoise=value)
                self.assertEqual(captured[guides.KEY].denoise, value)
                self.assertEqual(captured[guides.DENOISE_FIELD], original[guides.DENOISE_FIELD])

    def test_native_normal_channel_conversion_is_once_only_without_spatial_flip(self):
        for convention, channels in (("native-opengl", (0,)), ("native-directx", (0, 1))):
            with self.subTest(convention=convention):
                _, _, handoff = self.adopt(normal=True, info=dict(convention=convention))
                captured, _ = self.capture(
                    handoff, compatibility=CompatibleControl(invert_channels=channels)
                )
                request = captured[guides.KEY]
                original = np.asarray(guides.request_image(request.guide)).copy()
                expected = original.copy()
                for channel in channels:
                    expected[..., channel] = 255 - expected[..., channel]
                np.testing.assert_array_equal(
                    np.asarray(guides.consumer_guide_image(request)), expected
                )
                np.testing.assert_array_equal(
                    np.asarray(guides.consumer_guide_image(request)), expected
                )
                np.testing.assert_array_equal(
                    np.asarray(guides.request_image(request.guide)), original
                )
                self.assertEqual(
                    request.metadata()["derived_guide"]["invert_channels"], list(channels)
                )
                self.assertFalse(request.metadata()["derived_guide"]["spatial_flip"])

    def test_file_pixels_changed_with_restored_stat_are_rejected(self):
        _, _, handoff = self.adopt()
        path = Path(
            self.store.artifact(
                handoff.session_id, handoff.guide_id, revision=handoff.revision, owner=self.owner
            ).path
        )
        stat = path.stat()
        raw = bytearray(path.read_bytes())
        # A PNG ancillary/CRC change leaves geometry decodable but invalidates its exact hash.
        raw[-1] ^= 1
        path.write_bytes(raw)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaises(ValueError):
            guides.handoff_images(self.store, handoff, owner=self.owner)

    def test_face_selection_is_consumed_by_actual_inpaint_request_builder(self):
        result = self.result(operation="face-landmarks", info=dict(mask_role="inpaint-selection"))
        restored, mask = self.store.use_result_as_mask(
            result.session_id, result.revision, result.current.artifact_id, owner=self.owner
        )
        handoff = capture_mask_handoff(
            self.store,
            restored.session_id,
            restored.revision,
            restored.current.artifact_id,
            mask.artifact_id,
            owner=self.owner,
            choice="union",
        )
        inputs = inpaint_inputs(self.store, handoff, owner=self.owner)
        request = build_library_request(
            dict(inpaint_denoise=0.35, inpaint_mask_grow=2),
            source_record(self.source_path),
            prompt="change only the face lighting",
            negative="literal negative",
            checkpoint="selected",
            action="inpaint",
            resets={},
            features=dict(inpaint=True),
            presets={},
            custom="Custom...",
            editor_context=dict(
                handoff.public_context()
                if hasattr(handoff, "public_context")
                else dict(
                    session_id=restored.session_id,
                    revision=restored.revision,
                    source_id=restored.current.artifact_id,
                ),
                mask_id=mask.artifact_id,
            ),
            editor_store=self.store,
            owner=self.owner,
        )
        np.testing.assert_array_equal(np.asarray(request["main_view"]), self.source)
        np.testing.assert_array_equal(
            np.asarray(request["inpaint_view"]["layers"][0].getchannel("A")),
            np.asarray(inputs["inpaint_view"]["layers"][0].getchannel("A")),
        )
        self.assertEqual(request["inpaint_denoise"], 0.35)
        self.assertEqual(request["inpaint_mask_grow"], 2)
        self.assertEqual(request[editor.KEY].mask_id, mask.artifact_id)


if __name__ == "__main__":
    unittest.main()
