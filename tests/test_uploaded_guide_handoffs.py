"""Request-owned upload bindings use real semantic/header capture, no Editor fiction."""

from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ruinedfooocus.features.image_editing import guide_handoffs as guides
from ruinedfooocus.features.image_processing.image_geometry_tools import capture_source
import test_guide_control_compat as controls


class UploadedGuideHandoffTests(unittest.TestCase):
    def setUp(self):
        self.control = controls.GuideControlCompatibilityTests()
        self.control.setUp()
        self.addCleanup(self.control.doCleanups)
        _, self.options = self.control.fixture(union=6)
        self.owner = ("api", "owned-fixture")
        self.source = np.arange(5 * 7 * 4, dtype=np.uint8).reshape(5, 7, 4)
        self.source[..., 3] = np.array([0, 17, 64, 128, 191, 239, 255], dtype=np.uint8)
        self.depth = np.arange(35, dtype=np.uint8).reshape(5, 7) * 7

    def capture(
        self, *, kind="depth", convention="white-near", source=None, guide=None, denoise=0, **extra
    ):
        return guides.capture_uploaded_guide_inputs(
            source=Image.fromarray(self.source) if source is None else source,
            guide=Image.fromarray(self.depth) if guide is None else guide,
            kind=kind,
            convention=convention,
            denoise=denoise,
            data=dict(
                controlnet=dict(self.options),
                prompt="literal original edit prompt",
                negative="literal negative",
                custom_width=13,
                custom_height=17,
                **extra,
            ),
            manager=self.control.manager,
            family="SDXL",
            owner=self.owner,
        )

    def test_uploads_freeze_distinct_pixels_and_verify_without_editor_or_weight_hashing(self):
        image, guide = Image.fromarray(self.source), Image.fromarray(self.depth)
        data = self.capture(source=image, guide=guide)
        request = guides.captured(data)
        self.assertIsInstance(request.handoff, guides.UploadedGuideHandoff)
        self.assertEqual(request.denoise, 0)
        self.assertEqual(data["prompt"], "literal original edit prompt")
        self.assertEqual((data["custom_width"], data["custom_height"]), (13, 17))
        self.assertNotIn("_editor_operation", data)
        image.paste("white", (0, 0, 7, 5))
        guide.paste(255, (0, 0, 7, 5))
        np.testing.assert_array_equal(np.asarray(guides.request_image(request.source)), self.source)
        with patch.object(
            controls.guide, "cached_file_sha256", side_effect=AssertionError("worker hash")
        ):
            self.assertIs(
                guides.validate_request(
                    data, store=None, owner=self.owner, manager=self.control.manager, family="SDXL"
                ),
                request,
            )
        self.assertEqual(request.compatibility.union_type, "depth")
        self.assertEqual(request.controls()["depth_preprocess"], "none")

    def test_declared_normal_variants_use_only_the_receipted_channel_conversion(self):
        normal = np.stack((self.depth, 255 - self.depth, np.full_like(self.depth, 211)), axis=-1)
        for convention, channels in (
            ("moge-opengl", (0,)),
            ("moge-directx", (0, 1)),
            ("normal-bae", ()),
        ):
            with self.subTest(convention=convention):
                data = self.capture(
                    kind="normal", convention=convention, guide=normal, denoise=0.35
                )
                request = guides.captured(data)
                self.assertEqual(request.compatibility.invert_channels, channels)
                self.assertEqual(request.compatibility.union_type, "normal")
                expected = normal.copy()
                for channel in channels:
                    expected[..., channel] = 255 - expected[..., channel]
                np.testing.assert_array_equal(
                    np.asarray(guides.consumer_guide_image(request)), expected
                )

    def test_wrong_owner_or_changed_content_cannot_reuse_captured_upload(self):
        data = self.capture()
        with self.assertRaises(PermissionError):
            guides.validate_request(
                data,
                store=None,
                owner=("api", "foreign"),
                manager=self.control.manager,
                family="SDXL",
            )
        data[guides.KEY] = replace(
            data[guides.KEY], source=capture_source(np.zeros_like(self.source))
        )
        with self.assertRaisesRegex(ValueError, "content changed"):
            guides.validate_request(
                data, store=None, owner=self.owner, manager=self.control.manager, family="SDXL"
            )

    def test_frame_precision_color_map_and_forged_editor_context_are_rejected(self):
        for values in (
            dict(guide=Image.new("L", (6, 5))),
            dict(source=self.source.astype(np.float32) / 255),
            dict(guide=Image.new("RGB", (7, 5), "red")),
            dict(convention="unknown"),
            dict(editor_context={}),
            dict(denoise=True),
            dict(denoise=float("nan")),
        ):
            with self.subTest(values=tuple(values)), self.assertRaises(ValueError):
                self.capture(**values)

    def test_exported_technical_facts_have_no_editor_or_owner_handles(self):
        request = guides.captured(self.capture())
        metadata = request.metadata()
        self.assertEqual(metadata["provenance"], "caller-declared-upload")
        hidden = {
            "owner_key",
            "session_id",
            "revision",
            "source_id",
            "guide_id",
            "source_frame_id",
            "guide_frame_id",
        }

        def keys(value):
            if isinstance(value, dict):
                return set(value).union(*(keys(child) for child in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(child) for child in value))
            return set()

        self.assertFalse(keys(metadata) & hidden)
        self.assertNotIn(str(self.control.root), json.dumps(metadata))
        with self.assertRaises(ValueError):
            guides.captured({guides.KEY: dict(source="forged")})


if __name__ == "__main__":
    unittest.main()
