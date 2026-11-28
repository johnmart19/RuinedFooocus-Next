"""Explicit local encoders/codecs remain local through real API image capture."""

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from ruinedfooocus.app import api_runtime as api
from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.core.preparation import PreparationService
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.models import effective_image_plan as plans
from ruinedfooocus.features.models.resource_defaults import resource_name
import test_bernini_image as bernini
from test_component_candidates import qwen_wan21_shapes
import test_plan_acquisition_transport as transport_fixtures


def capture(fixture, family, *, values=None, options=None):
    fixture.models.get_models_by_path = lambda *args, **kwargs: {"baseModel": family}
    fixture.models.get_model_base = lambda row: row["baseModel"]
    performance = NS(
        CUSTOM_PERFORMANCE="Own settings",
        choices_for_model=lambda *args: ["Own settings"],
        selection_for_model=lambda *args: "Own settings",
        get_perf_options=lambda *args: dict(
            custom_steps=12, cfg=3.25, sampler_name="euler", scheduler="simple", clip_skip=1
        ),
    )
    runtime = (
        dict(
            performance="Own settings",
            resolution="default",
            image_engine="native",
            **(values or {}),
        ),
        performance,
        NS(
            CUSTOM_RESOLUTION="Custom...",
            selection_for_model=lambda *args: "default",
            aspect_ratios={"default": (64, 96)},
        ),
        fixture.manager,
        fixture.models,
    )
    shared = NS()
    with (
        patch.dict(
            sys.modules,
            {
                "ruinedfooocus.app.state": shared,
                "comfy.samplers": NS(KSampler=NS(SAMPLERS=["euler"], SCHEDULERS=["simple"])),
            },
        ),
        patch.object(sys.modules["ruinedfooocus.app"], "state", shared, create=True),
        patch.object(api, "_image_runtime_snapshot", return_value=runtime),
        patch.object(
            api,
            "image_defaults",
            dict(base_model_name=str(fixture.checkpoint), performance_selection="Own settings"),
        ),
        patch.object(api, "decode_image", return_value=Image.new("RGBA", (64, 96), (2, 4, 6, 90))),
        patch.object(api, "submit", side_effect=lambda operation: operation),
    ):
        operation = api.images(
            dict(
                dict(
                    prompt="Keep the frame",
                    negative_prompt="No text",
                    size="64x96",
                    input_image=None,
                ),
                **(options or {}),
            ),
            dict(family=family, name=str(fixture.checkpoint)),
        )
    cells = dict(
        zip(operation.__code__.co_freevars, (cell.cell_contents for cell in operation.__closure__))
    )
    return cells["data"]


class ApiLocalEncoderTests(unittest.TestCase):
    def setUp(self):
        self.fixture = transport_fixtures.PlanAcquisitionTransportTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.registry_patch.return_value = self.fixture.real_registry
        self.fixture.manager.DOWNLOADABLE_FILES = {}
        for name in ("clip.json", "vae.json"):
            self.fixture.manager.DOWNLOADABLE_FILES.update(
                json.loads(resource_path("model_catalogue", name).read_text(encoding="utf-8"))
            )
        self.overrides = {}
        self.files = {}
        for role, category in (("clip_qwen3vl_4b_scaled", "clip"), ("vae_qwen_image", "vae")):
            name = resource_name(role, {})
            self.overrides[role] = name
            path = self.fixture.root / category / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.fixture.content)
            self.files[role] = path

    def capture(self):
        return capture(
            self.fixture, "Krea2", options=dict(image_component_overrides=self.overrides)
        )

    def service(self):
        service = PreparationService(
            transport=Mock(side_effect=AssertionError("local file must not download")),
            staging_root=self.fixture.root / "staging",
        )
        self.addCleanup(service.close)
        return service

    def test_actual_krea_api_local_encoder_keeps_provenance_and_never_requests_a_download(self):
        canonical = copy.deepcopy(self.fixture.manager.DOWNLOADABLE_FILES)
        with patch.object(
            self.fixture.manager,
            "preparation_request",
            side_effect=AssertionError("explicit local file must not acquire"),
        ):
            data = self.capture()
        plan = data[plans.KEY]
        for row in plan.required_components:
            self.assertIsNone(row.download_key)
            self.assertTrue(row.source)
            self.assertIsNone(row.catalogue_identity)
            self.assertEqual(
                (row.acquisition.url, row.acquisition.expected_size, row.acquisition.sha256),
                ("", None, None),
            )
            self.assertIsNotNone(row.signature)
        ready = assets.prepare_image_assets(
            data, check_cancelled=lambda: None, service=self.service()
        )
        self.assertEqual(ready["image_component_overrides"], self.overrides)
        self.assertEqual(ready["prompt"], "Keep the frame")
        self.assertEqual(ready["negative"], "No text")
        self.assertEqual(self.fixture.manager.DOWNLOADABLE_FILES, canonical)

    def test_missing_local_encoder_rejects_even_if_it_appears_between_plan_and_asset_capture(self):
        encoder = self.files["clip_qwen3vl_4b_scaled"]
        original = assets.capture_image_assets
        for late in (False, True):
            encoder.unlink(missing_ok=True)

            def before_assets(data, *args, **kwargs):
                row = next(
                    row for row in data[plans.KEY].required_components if row.category == "clip"
                )
                self.assertIsNone(row.signature)
                self.assertEqual(row.acquisition.url, "")
                if late:
                    encoder.write_bytes(self.fixture.content)
                return original(data, *args, **kwargs)

            with (
                self.subTest(appeared_late=late),
                patch.object(assets, "capture_image_assets", side_effect=before_assets),
            ):
                with self.assertRaisesRegex(FileNotFoundError, "component .*missing"):
                    self.capture()

    def test_replaced_or_deleted_local_encoder_cannot_be_prepared(self):
        data = self.capture()
        encoder = self.files["clip_qwen3vl_4b_scaled"]
        encoder.write_bytes(self.fixture.content + b"replaced")
        for deleted in (False, True):
            if deleted:
                encoder.unlink()
            with self.subTest(deleted=deleted):
                with self.assertRaisesRegex(ValueError, "changed after capture"):
                    assets._plan_component_assets(data, self.fixture.manager, ())
                with self.assertRaisesRegex(ValueError, "changed after capture"):
                    assets.prepare_image_assets(
                        data, check_cancelled=lambda: None, service=self.service()
                    )


class ApiLocalBerniniCodecTests(unittest.TestCase):
    def _assert_codec_contract(self, *, same_name):
        fixture = bernini.BerniniImageTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.checkpoint = fixture.files["high"]
        fixture.manager.paths = {
            "path_" + category: str(fixture.root / category)
            for category in ("clip", "vae", "diffusion_models")
        }
        fixture.manager.get_abspath = Path
        fixture.models.get_names = lambda category: []
        fixture.models.model_dirs = {"loras": [str(fixture.root / "loras")]}
        fixture.models.offline = True
        clip_header = json.dumps(
            {
                "shared.weight": dict(shape=[256384, 4096], dtype="F16", data_offsets=[0, 0]),
                "encoder.block.23.layer.1.DenseReluDense.wi_1.weight": dict(
                    shape=[10240, 4096], dtype="F16", data_offsets=[0, 0]
                ),
            }
        ).encode()
        clip_content = len(clip_header).to_bytes(8, "little") + clip_header
        fixture.files["clip"].write_bytes(clip_content)
        fixture.rows["clip"].update(
            bytes=len(clip_content),
            sha256=hashlib.sha256(clip_content).hexdigest(),
            header_sha256=hashlib.sha256(clip_header).hexdigest(),
        )
        fixture.catalogue[fixture.rows["clip"]["catalogue_key"]].update(
            size=len(clip_content), sha256=hashlib.sha256(clip_content).hexdigest()
        )
        codec = (
            fixture.root
            / "vae"
            / (fixture.rows["vae"]["filename"] if same_name else "reused-native-codec.safetensors")
        )
        header = json.dumps(
            {
                key: dict(shape=list(shape), dtype="BF16", data_offsets=[0, 0])
                for key, shape in qwen_wan21_shapes().items()
            }
        ).encode()
        content = len(header).to_bytes(8, "little") + header
        codec.write_bytes(content)
        key = "vae/" + codec.name
        fixture.catalogue[key] = dict(
            path="path_vae",
            filename=codec.name,
            url="https://example.invalid/main/" + codec.name,
            size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        producer = fixture.manager.preparation_request

        def request(selected_key, destination=None):
            if selected_key == key:
                raise AssertionError("The explicit local codec must not request publisher metadata")
            return producer(selected_key, destination=destination)

        with (
            patch.object(fixture.manager, "preparation_request", side_effect=request),
            patch.object(
                assets.preparation_service,
                "submit",
                side_effect=AssertionError("API capture must not acquire"),
            ),
        ):
            data = capture(
                fixture,
                bernini.image.FAMILY,
                values=fixture.resources(),
                options=dict(
                    input_image="source",
                    bernini_low_model=fixture.rows["low"]["filename"],
                    bernini_split_step=4,
                    image_component_overrides={bernini.image.VAE_ROLE: codec.name},
                ),
            )
        row = next(row for row in data[plans.KEY].required_components if row.category == "vae")
        self.assertIsNone(row.download_key)
        self.assertEqual(row.source, fixture.catalogue[key]["url"])
        self.assertEqual(
            (row.acquisition.url, row.acquisition.expected_size, row.acquisition.sha256),
            ("", None, None),
        )
        self.assertIsNotNone(row.signature)
        self.assertEqual(data["image_component_overrides"][bernini.image.VAE_ROLE], codec.name)
        self.assertEqual(data[bernini.image.KEY].source.size, (64, 96))
        codec_asset = next(item for item in data[assets.KEY].artifacts if item.role == "vae")
        self.assertEqual(codec_asset.request, row.acquisition)
        service = PreparationService(
            transport=Mock(
                side_effect=AssertionError("installed Bernini components must not download")
            ),
            staging_root=fixture.root / "staging",
        )
        self.addCleanup(service.close)
        ready = assets.prepare_image_assets(data, check_cancelled=lambda: None, service=service)
        prepared_codec = next(item for item in ready[assets.KEY].artifacts if item.role == "vae")
        self.assertEqual(prepared_codec.request, row.acquisition)
        self.assertEqual(prepared_codec.signature, row.signature)
        codec.unlink()
        with patch.object(fixture.manager, "preparation_request", side_effect=request):
            with self.assertRaisesRegex(FileNotFoundError, "not installed"):
                assets._artifact(fixture.manager, "vae", "vae", codec.name, local_only=True)

    def test_actual_bernini_api_keeps_explicit_native_codec_local_in_the_plan(self):
        self._assert_codec_contract(same_name=False)

    def test_actual_bernini_api_same_named_explicit_codec_keeps_local_integrity(self):
        self._assert_codec_contract(same_name=True)


if __name__ == "__main__":
    unittest.main()
