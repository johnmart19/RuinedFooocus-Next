"""Explicit geometry consumers reuse native header admission and source identity."""

from dataclasses import replace, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.features.image_editing import guide_control_compat as guide
from ruinedfooocus.features.models.model_identity import file_signature


@dataclass(frozen=True)
class Spec:
    kind: str = "depth"
    convention: str = "white-near; invalid-black"
    projection: str = "perspective"
    producer: str = "moge-depth"
    source_id: str = "source-owned"
    guide_id: str = "guide-owned"
    width: int = 13
    height: int = 17


class GuideControlCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.manager = SimpleNamespace(
            get_model_folders=lambda _: [self.root], model_paths={"cache_path": self.root / "cache"}
        )
        self.spec = Spec()

    def fixture(self, *, union=0, context=2048):
        header = {
            "controlnet_cond_embedding.conv_in.weight": {"shape": [16, 3, 3, 3]},
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight": {
                "shape": [640, context]
            },
        }
        if union:
            header["task_embedding"] = {"shape": [union, 320]}
            header["control_add_embedding.linear_1.bias"] = {"shape": [1280]}
        raw = json.dumps(header).encode()
        path = self.root / "explicit-renamed.safetensors"
        path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"fixture-payload")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sources = {
            digest: (
                path.stat().st_size,
                "Reviewed CPU fixture",
                "union" if union else "ordinary",
                union,
            )
        }
        self.enterContext(patch.object(guide, "SOURCES", sources))
        return path, dict(
            model=path.name,
            model_signature=file_signature(path),
            strength=0.63,
            start=0.1,
            stop=0.9,
        )

    def test_depth_source_is_exact_and_keeps_original_semantics(self):
        path, controls = self.fixture()
        receipt = guide.validate_guide_control(self.manager, "SDXL 1.0", controls, self.spec)
        self.assertEqual(receipt.controls["depth_preprocess"], "none")
        self.assertEqual(receipt.controls["strength"], 0.63)
        self.assertEqual(receipt.recipe, "ordinary")
        self.assertEqual(receipt.metadata()["guide"]["convention"], self.spec.convention)
        self.assertEqual(receipt.metadata()["semantic_convention"], "white-near-disparity")
        self.assertEqual(receipt.model_signature, tuple(file_signature(path)))
        with patch.object(
            guide, "cached_file_sha256", side_effect=AssertionError("worker must not hash")
        ):
            self.assertIs(
                receipt.verify(self.manager, "SDXL", receipt.controls, self.spec), receipt
            )

    def test_union_uses_explicit_native_depth_mode(self):
        _, controls = self.fixture(union=6)
        receipt = guide.validate_guide_control(self.manager, "SDXL", controls, self.spec)
        self.assertEqual(
            (receipt.recipe, receipt.union_type, receipt.controls["type"]),
            ("union", "depth", "control"),
        )
        self.assertEqual(guide.union_types().index(receipt.union_type) - 1, 1)
        self.assertNotIn("strength", receipt.ui_values())
        self.assertNotIn("denoise", receipt.ui_values())
        self.assertEqual(receipt.ui_values()["control_union_type"], "depth")

    def test_normal_requires_exact_native_producer_convention(self):
        _, controls = self.fixture(union=6)
        with patch.object(
            guide, "cached_file_sha256", side_effect=AssertionError("no needless hash")
        ):
            for convention in ("unknown", "OpenGL", "white-near"):
                with (
                    self.subTest(convention=convention),
                    self.assertRaisesRegex(ValueError, "exact native camera convention"),
                ):
                    guide.validate_guide_control(
                        self.manager,
                        "SDXL",
                        controls,
                        replace(self.spec, kind="normal", convention=convention),
                    )

    def test_normal_base6_mode_and_canonical_vectors(self):
        _, controls = self.fixture(union=6)
        # Native MoGe outward camera normals. The shared FrameNet/NormalBae
        # convention has signed view-ray-positive normals: a front wall is +Z.
        canonical = (
            ((0, 0, -1), (0.5, 0.5, 1.0)),
            ((1, 0, 0), (0.0, 0.5, 0.5)),
            ((0, -1, 0), (0.5, 1.0, 0.5)),
        )
        for convention, y_sign, expected_channels in (
            ("native-opengl", -1, (0,)),
            ("native-directx", 1, (0, 1)),
        ):
            spec = replace(self.spec, kind="normal", convention=convention)
            receipt = guide.validate_guide_control(self.manager, "SDXL", controls, spec)
            self.assertEqual(receipt.invert_channels, expected_channels)
            self.assertEqual(receipt.union_type, "normal")
            self.assertEqual(guide.union_types().index(receipt.union_type) - 1, 4)
            self.assertEqual(receipt.controls["type"], "control")
            self.assertEqual(receipt.metadata()["semantic_convention"], "normalbae-scannet-camera")
            self.assertIn("inferred", receipt.metadata()["conversion_evidence"])
            for raw, expected_rgb in canonical:
                encoded = tuple(
                    (value * sign + 1) / 2 for value, sign in zip(raw, (1, y_sign, -1), strict=True)
                )
                converted = tuple(
                    1 - value if index in receipt.invert_channels else value
                    for index, value in enumerate(encoded)
                )
                self.assertEqual(converted, expected_rgb)
            with patch.object(
                guide, "cached_file_sha256", side_effect=AssertionError("worker must not hash")
            ):
                self.assertIs(receipt.verify(self.manager, "SDXL", receipt.controls, spec), receipt)

    def test_depth_only_payload_cannot_admit_normal_and_axis_changes_are_stale(self):
        _, controls = self.fixture()
        with self.assertRaisesRegex(ValueError, "depth-only"):
            guide.validate_guide_control(
                self.manager,
                "SDXL",
                controls,
                replace(self.spec, kind="normal", convention="native-opengl"),
            )
        _, controls = self.fixture(union=6)
        spec = replace(self.spec, kind="normal", convention="native-opengl")
        receipt = guide.validate_guide_control(self.manager, "SDXL", controls, spec)
        with self.assertRaisesRegex(ValueError, "changed after capture"):
            receipt.verify(
                self.manager, "SDXL", receipt.controls, replace(spec, convention="native-directx")
            )

    def test_panorama_requires_its_own_conditioning_contract(self):
        _, controls = self.fixture()
        with self.assertRaisesRegex(ValueError, "perspective"):
            guide.validate_guide_control(
                self.manager,
                "SDXL",
                controls,
                replace(
                    self.spec,
                    projection="equirectangular",
                    convention="white-near; uncovered-native-far-fill",
                ),
            )

    def test_header_and_checkpoint_family_must_both_be_known(self):
        _, controls = self.fixture(context=768)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            guide.validate_guide_control(self.manager, "SDXL", controls, self.spec)
        with self.assertRaisesRegex(ValueError, "SDXL checkpoint"):
            guide.validate_guide_control(self.manager, "SD15", controls, self.spec)

    def test_a_header_is_not_source_identity(self):
        path, controls = self.fixture()
        source = next(iter(guide.SOURCES.values()))
        with (
            patch.object(guide, "SOURCES", {"f" * 64: source}),
            self.assertRaisesRegex(ValueError, "verified SDXL"),
        ):
            guide.validate_guide_control(self.manager, "SDXL", controls, self.spec)
        self.assertTrue(path.is_file())

    def test_mode_count_must_match_the_reviewed_payload(self):
        _, controls = self.fixture(union=5)
        digest, (size, label, recipe, _) = next(iter(guide.SOURCES.items()))
        with (
            patch.object(guide, "SOURCES", {digest: (size, label, recipe, 6)}),
            self.assertRaisesRegex(ValueError, "mode table"),
        ):
            guide.validate_guide_control(self.manager, "SDXL", controls, self.spec)

    def test_explicit_selection_signature_is_required_and_rechecked(self):
        path, controls = self.fixture()
        with self.assertRaisesRegex(ValueError, "explicit installed"):
            guide.validate_guide_control(self.manager, "SDXL", dict(model=path.name), self.spec)
        receipt = guide.validate_guide_control(self.manager, "SDXL", controls, self.spec)
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            receipt.verify(self.manager, "SDXL", receipt.controls, self.spec)

    def test_signature_is_rechecked_after_source_verification(self):
        path, controls = self.fixture()
        digest = next(iter(guide.SOURCES))

        def change(*_):
            path.write_bytes(path.read_bytes() + b"changed")
            return digest

        with (
            patch.object(guide, "cached_file_sha256", side_effect=change),
            self.assertRaisesRegex(ValueError, "changed during source verification"),
        ):
            guide.validate_guide_control(self.manager, "SDXL", controls, self.spec)

    def test_request_setting_and_guide_changes_are_refused(self):
        _, controls = self.fixture()
        receipt = guide.validate_guide_control(self.manager, "SDXL", controls, self.spec)
        for change in (
            dict(strength=0.4),
            dict(depth_preprocess="auto"),
            dict(recipe="union"),
            dict(type="control"),
            dict(model="../escape"),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                receipt.verify(self.manager, "SDXL", dict(receipt.controls, **change), self.spec)
        for change in (dict(width=14), dict(source_id="new-source"), dict(convention="white-near")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                receipt.verify(self.manager, "SDXL", receipt.controls, replace(self.spec, **change))

    def test_planner_changes_only_safe_existing_fields_and_returns_detached_data(self):
        path, _ = self.fixture()
        receipt = guide.plan_guide_control(
            self.spec, "SDXL", self.manager, path.name, file_signature(path)
        )
        self.assertEqual(
            set(receipt.ui_values()),
            {
                "cn_type",
                "cn_model",
                "cn_model_signature",
                "cn_depth_preprocess",
                "control_recipe",
                "control_union_type",
            },
        )
        detached = receipt.controls
        detached["model_signature"][1] = -1
        self.assertNotEqual(detached, receipt.controls)
        with self.assertRaisesRegex(ValueError, "local list"):
            guide.plan_guide_control(self.spec, "SDXL", self.manager, "", [])

    def test_invalid_control_ranges_fail_before_queueing(self):
        _, controls = self.fixture()
        for change in (
            dict(strength=0),
            dict(strength=True),
            dict(strength=float("nan")),
            dict(start=1),
            dict(stop=0),
        ):
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "Guide control requires"),
            ):
                guide.validate_guide_control(
                    self.manager, "SDXL", dict(controls, **change), self.spec
                )

    def test_explicit_upload_conventions_are_declared_and_normalbae_is_passthrough(self):
        _, controls = self.fixture(union=6)
        uploaded = replace(
            self.spec, producer="explicit-upload", source_id="a" * 64, guide_id="b" * 64
        )
        for kind, convention, channels in (
            ("depth", "white-near", ()),
            ("normal", "native-opengl", (0,)),
            ("normal", "native-directx", (0, 1)),
            ("normal", "normalbae-scannet-camera", ()),
        ):
            with self.subTest(kind=kind, convention=convention):
                spec = replace(uploaded, kind=kind, convention=convention)
                receipt = guide.validate_guide_control(self.manager, "SDXL", controls, spec)
                self.assertEqual(receipt.invert_channels, channels)
                self.assertEqual(receipt.union_type, kind)
                metadata = receipt.metadata()
                self.assertEqual(
                    metadata["semantic_convention"],
                    "normalbae-scannet-camera" if kind == "normal" else "white-near-disparity",
                )
                self.assertEqual(metadata["provenance"], "caller-declared-upload")
                self.assertIn("caller-declared", metadata["convention_qualification"])
                self.assertIn("caller_declared", metadata["conversion_evidence"])
                self.assertEqual(metadata["guide"]["producer"], "explicit-upload")
                self.assertFalse(metadata["runtime_qualified"])
                with patch.object(
                    guide, "cached_file_sha256", side_effect=AssertionError("worker must not hash")
                ):
                    self.assertIs(
                        receipt.verify(self.manager, "SDXL", receipt.controls, spec), receipt
                    )

    def test_explicit_upload_does_not_expand_owned_producers_or_accept_unbound_axes(self):
        _, controls = self.fixture(union=6)
        uploaded = replace(
            self.spec,
            producer="explicit-upload",
            source_id="a" * 64,
            guide_id="b" * 64,
            convention="white-near",
        )
        rejected = (
            replace(uploaded, source_id="source-owned"),
            replace(uploaded, guide_id="B" * 64),
            replace(uploaded, convention="white-near; invalid-black"),
            replace(uploaded, kind="normal", convention="unknown"),
            replace(uploaded, kind="normal", convention="normal-bae"),
            replace(uploaded, projection="equirectangular"),
            replace(self.spec, kind="normal", convention="normalbae-scannet-camera"),
            replace(self.spec, producer="lotus-depth", kind="normal", convention="native-opengl"),
            replace(self.spec, producer="arbitrary-upload"),
        )
        with patch.object(
            guide,
            "cached_file_sha256",
            side_effect=AssertionError("invalid declarations must fail before hashing"),
        ):
            for spec in rejected:
                with self.subTest(spec=spec), self.assertRaises(ValueError):
                    guide.validate_guide_control(self.manager, "SDXL", controls, spec)

    def test_uploaded_normal_still_requires_exact_signed_union_consumer(self):
        _, controls = self.fixture()
        spec = replace(
            self.spec,
            producer="explicit-upload",
            source_id="a" * 64,
            guide_id="b" * 64,
            kind="normal",
            convention="normalbae-scannet-camera",
        )
        with self.assertRaisesRegex(ValueError, "depth-only"):
            guide.validate_guide_control(self.manager, "SDXL", controls, spec)
        _, controls = self.fixture(union=6)
        _, source = next(iter(guide.SOURCES.items()))
        with (
            patch.object(guide, "SOURCES", {"f" * 64: source}),
            self.assertRaisesRegex(ValueError, "verified SDXL"),
        ):
            guide.validate_guide_control(self.manager, "SDXL", controls, spec)


if __name__ == "__main__":
    unittest.main()
