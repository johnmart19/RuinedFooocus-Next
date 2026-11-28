"""Typed source/guide facts enter plan identity and path-free technical exports."""

from dataclasses import dataclass, replace
import json
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ruinedfooocus.features.gallery.editor_session import _mapping_json
from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY, SizePlan
from ruinedfooocus.features.image_editing import guide_handoffs as guides
from ruinedfooocus.features.models import effective_image_plan as plans
import test_effective_image_plan as plan_fixtures
import test_guide_handoffs as guide_fixtures


@dataclass(frozen=True)
class PlanControl(guide_fixtures.CompatibleControl):
    artifact_sha256: str = "a" * 64

    def metadata(self):
        return dict(
            super().metadata(),
            sha256=self.artifact_sha256,
            recipe=self.recipe,
            union_type=self.union_type,
        )


class GuideEffectivePlanTests(unittest.TestCase):
    def setUp(self):
        self.plan_fixture = plan_fixtures.EffectiveImagePlanTests()
        self.plan_fixture.setUp()
        self.addCleanup(self.plan_fixture.doCleanups)
        self.guide_fixture = guide_fixtures.GuideHandoffTests()
        self.guide_fixture.setUp()
        self.addCleanup(self.guide_fixture.doCleanups)
        _, _, handoff = self.guide_fixture.adopt()
        self.guide_data, _ = self.guide_fixture.capture(handoff, compatibility=PlanControl())
        self.request = self.guide_data[guides.KEY]

    def capture(self, request=None):
        data = dict(self.plan_fixture.request, **self.guide_data)
        if request is not None:
            data[guides.KEY] = request
        return self.plan_fixture.capture(data)

    def facts(self, captured):
        return captured[plans.KEY].metadata()["native_facts"]["control_guide"]

    def test_owned_guide_facts_bind_actual_source_and_guide_without_security_handles(self):
        captured = self.capture()
        facts = self.facts(captured)
        self.assertEqual(facts["guide"]["kind"], "depth")
        self.assertEqual(facts["guide"]["convention"], "white-near; invalid-black")
        self.assertEqual(facts["source_pixel_sha256"], self.request.source.source_sha256)
        self.assertEqual(facts["guide_pixel_sha256"], self.request.guide.source_sha256)
        self.assertNotEqual(facts["source_pixel_sha256"], facts["guide_pixel_sha256"])
        self.assertEqual(facts["source_pixels"]["channels"], 4)
        self.assertEqual(facts["guide_pixels"]["channels"], 3)
        self.assertEqual(facts["denoise"], 0.37)
        self.assertEqual(facts["controls"]["strength"], 0.8)
        self.assertEqual(facts["control"]["sha256"], "a" * 64)
        exported = json.dumps(facts)
        for key in (
            "owner_key",
            "session_id",
            "revision",
            "source_id",
            "guide_id",
            "frame_id",
            "source_frame_id",
            "guide_frame_id",
        ):
            self.assertNotIn('"' + key + '"', exported)
        self.assertNotIn(str(self.guide_fixture.root), exported)
        self.assertNotIn(self.request.handoff.session_id, exported)

    def test_source_guide_convention_control_and_denoise_drift_require_a_new_plan(self):
        captured = self.capture()
        h = self.request.handoff
        changed = (
            replace(self.request, denoise=0.7),
            replace(self.request, source=replace(self.request.source, source_sha256="b" * 64)),
            replace(self.request, guide=replace(self.request.guide, source_sha256="c" * 64)),
            replace(
                self.request,
                handoff=replace(
                    h,
                    source_signature=(
                        h.source_signature[0],
                        h.source_signature[1] + 1,
                        *h.source_signature[2:],
                    ),
                ),
            ),
            replace(
                self.request, handoff=replace(h, spec=replace(h.spec, convention="white-near"))
            ),
            replace(
                self.request,
                compatibility=replace(self.request.compatibility, artifact_sha256="d" * 64),
            ),
            replace(
                self.request,
                controls_json=json.dumps(dict(self.request.controls(), union_type="normal")),
            ),
            replace(self.request, conversion_sources=(("comfy_extras/nodes_mask.py", "e" * 64),)),
        )
        for request in changed:
            with (
                self.subTest(request=request),
                self.assertRaisesRegex(ValueError, "request changed"),
            ):
                captured[plans.KEY].verify_request(dict(captured, **{guides.KEY: request}))

    def test_uploaded_packet_uses_the_same_technical_identity_without_forging_editor_handles(self):
        uploaded = replace(
            self.request,
            handoff=guides.UploadedGuideHandoff(
                "private-api-owner",
                self.request.handoff.source_sha256,
                self.request.handoff.guide_sha256,
                replace(
                    self.request.handoff.spec, producer="explicit-upload", source_id="", guide_id=""
                ),
            ),
        )
        facts = self.facts(self.capture(uploaded))
        self.assertEqual(facts["provenance"], "caller-declared-upload")
        self.assertEqual(facts["guide"]["producer"], "explicit-upload")
        self.assertEqual(facts["source_pixel_sha256"], self.request.source.source_sha256)
        self.assertEqual(facts["guide_pixel_sha256"], self.request.guide.source_sha256)
        self.assertEqual(facts["source_signature"], [])
        self.assertNotIn("private-api-owner", json.dumps(facts))

    def test_projection_refuses_new_private_metadata_fields_and_redacts_model_paths(self):
        metadata = self.request.metadata()
        metadata.update(owner_key="secret-owner", session_id="secret-session")
        metadata["guide"]["source_id"] = "secret-source-handle"
        metadata["control"]["owner_key"] = "secret-control-owner"
        private_model = str(self.plan_fixture.root / "private-model.safetensors")
        for key in ("control", "controls"):
            metadata[key]["model"] = private_model
            metadata[key]["model_signature"] = [private_model, 123, 456]
        with patch.object(guides.GuideRequest, "metadata", return_value=metadata):
            facts = self.facts(self.capture())
        exported = json.dumps(facts)
        self.assertNotIn("secret-", exported)
        self.assertNotIn(str(self.plan_fixture.root), exported)
        for key in ("control", "controls"):
            self.assertEqual(facts[key]["model"], "private-model.safetensors")
            self.assertEqual(facts[key]["model_signature"], [123, 456])

    def test_plan_json_editor_fingerprint_and_png_round_trip_keep_only_technical_guide_facts(self):
        captured = self.capture()
        metadata = captured[plans.KEY].metadata()
        encoded = _mapping_json(metadata, "effective plan")
        self.assertEqual(json.loads(encoded), metadata)
        png = PngInfo()
        png.add_text("parameters", encoded)
        path = self.plan_fixture.root / "guided-plan.png"
        Image.new("RGBA", (2, 3), (1, 2, 3, 4)).save(path, pnginfo=png)
        with Image.open(path) as image:
            self.assertEqual(json.loads(image.info["parameters"]), metadata)

    def test_ordinary_requests_keep_their_plan_identity_and_receive_no_previous_guide_facts(self):
        before = self.plan_fixture.capture()
        self.capture()
        after = self.plan_fixture.capture()
        self.assertEqual(before[plans.KEY].metadata(), after[plans.KEY].metadata())
        self.assertNotIn("control_guide", after[plans.KEY].metadata()["native_facts"])
        controls = before.get("controlnet") or {}
        self.assertEqual(
            plans._intent(before, controls),
            plans._intent(dict(before, **{guides.KEY: None}), controls),
        )

    def test_untyped_guide_mapping_cannot_claim_plan_execution_authority(self):
        with self.assertRaisesRegex(ValueError, "raw mapping"):
            self.capture({"denoise": 0.5})

    def test_default_auto_portrait_plan_keeps_source_size_and_explicit_sizes_win(self):
        portrait = Image.new("RGBA", (832, 1216), (30, 50, 70, 128))
        fixture = self.guide_fixture
        fixture.source = np.asarray(portrait).copy()
        portrait.save(fixture.source_path)
        fixture.initial = fixture.store.start(fixture.source_path, owner=fixture.owner)
        _, _, handoff = fixture.adopt(size=portrait.size)
        guide_data, _ = fixture.capture(handoff, compatibility=PlanControl())
        data = dict(self.plan_fixture.request, **guide_data)
        data.pop(SIZE_KEY)
        data.update(
            resolution_auto=True,
            image_resize_mode="preserve",
            width=1024,
            height=1024,
            custom_width=1024,
            custom_height=1024,
            aspect_ratios_selection="Square default",
        )
        explicit = SizePlan.create((640, 960), 2)
        for changes, expected in (
            ({}, portrait.size),
            ({"resolution_auto": False}, (1024, 1024)),
            ({SIZE_KEY: explicit}, explicit.target),
        ):
            with self.subTest(changes=changes):
                request = dict(data, **changes)
                captured = self.plan_fixture.capture(request)
                size = captured[plans.KEY].size
                self.assertEqual((size.width, size.height), expected)
                for key in (
                    "custom_width",
                    "custom_height",
                    "aspect_ratios_selection",
                    "prompt",
                    "control_guide_denoise",
                ):
                    self.assertEqual(captured[key], request[key])


if __name__ == "__main__":
    unittest.main()
