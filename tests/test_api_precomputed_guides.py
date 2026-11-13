"""Public uploads preserve separate source/guide inputs through API and worker capture."""

import base64
from dataclasses import replace
import io
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from ruinedfooocus.app import api_runtime as api, resource_lifecycle as life
from ruinedfooocus.core.path import PathManager
from ruinedfooocus.features.image_editing import guide_handoffs as guides
from ruinedfooocus.features.image_processing.image_geometry_tools import capture_source
from ruinedfooocus.features.models.effective_image_plan import KEY as PLAN_KEY
from ruinedfooocus.integrations.openai_api import ImageRequest
import test_api_live_contracts as http_fixtures
import test_guide_control_compat as control_fixtures
from ui_test_helpers import function_from


def encoded(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


class UploadedGuideApiTests(unittest.TestCase):
    def setUp(self):
        self.fixture = control_fixtures.GuideControlCompatibilityTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.manager = PathManager.__new__(PathManager)
        self.manager.paths = {"path_controlnet": str(self.root)}
        self.manager.model_paths = {}
        self.manager.DOWNLOADABLE_FILES = {}
        self.checkpoint = self.root / "bundled-sdxl.safetensors"
        raw = b'{"conditioner.embedders.0.x":{},"first_stage_model.x":{}}'
        self.checkpoint.write_bytes(len(raw).to_bytes(8, "little") + raw)
        self.models = NS(
            offline=True,
            get_file=lambda category, name: self.checkpoint if category == "checkpoints" else None,
            get_models_by_path=lambda *args, **kwargs: {"baseModel": "SDXL"},
            get_model_base=lambda value: value["baseModel"],
        )
        self.source = Image.new("RGBA", (64, 96), (12, 34, 56, 90))
        self.guide = Image.new("RGB", (64, 96), (77, 77, 77))
        self.shared = NS(state={})

    def capture(
        self, *, normal=False, convention=None, denoise=0, options=None, recommended_size=(64, 96)
    ):
        _, controls = self.fixture.fixture(union=6 if normal else 0)
        performance = NS(
            CUSTOM_PERFORMANCE="Own settings",
            choices_for_model=lambda *args: ["Own settings"],
            selection_for_model=lambda *args: "Own settings",
            get_perf_options=lambda *args: dict(
                custom_steps=8, cfg=4, sampler_name="euler", scheduler="simple", clip_skip=1
            ),
        )
        runtime = (
            dict(
                performance="Own settings",
                resolution="default",
                image_engine="native",
                resolution_auto=True,
            ),
            performance,
            NS(
                CUSTOM_RESOLUTION="Custom...",
                selection_for_model=lambda *args: "default",
                aspect_ratios={"default": recommended_size},
            ),
            self.manager,
            self.models,
        )
        payload = dict(
            prompt="Keep the border; change lighting",
            negative_prompt="No text",
            input_image=encoded(self.source),
            size="64x96",
            controlnet=dict(
                type="control" if normal else "depth",
                model=controls["model"],
                strength=0.63,
                start=0.1,
                stop=0.9,
            ),
            control_guide=dict(
                image=encoded(self.guide),
                kind="normal" if normal else "depth",
                convention=convention or ("moge-opengl" if normal else "white-near"),
                denoise=denoise,
            ),
        )
        payload.update(options or {})
        payload = ImageRequest.model_validate(payload).model_dump(exclude_none=True)
        samplers = NS(KSampler=NS(SAMPLERS=["euler"], SCHEDULERS=["simple"]))
        with (
            patch.dict(
                sys.modules, {"ruinedfooocus.app.state": self.shared, "comfy.samplers": samplers}
            ),
            patch.object(sys.modules["ruinedfooocus.app"], "state", self.shared, create=True),
            patch.object(api, "_image_runtime_snapshot", return_value=runtime),
            patch.object(
                api,
                "image_defaults",
                dict(base_model_name=self.checkpoint.name, performance_selection="Own settings"),
            ),
            patch.object(api, "submit", side_effect=lambda operation: operation),
        ):
            operation = api.images(payload, dict(family="SDXL", name=self.checkpoint.name))
        cells = dict(
            zip(
                operation.__code__.co_freevars,
                (cell.cell_contents for cell in operation.__closure__),
            )
        )
        return operation, cells["data"]

    def test_actual_api_plan_and_worker_prepare_keep_original_and_guide_separate(self):
        operation, data = self.capture()
        request = guides.captured(data)
        self.assertIsInstance(request.handoff, guides.UploadedGuideHandoff)
        self.assertEqual(guides.request_image(request.source).getpixel((0, 0)), (12, 34, 56, 90))
        self.assertEqual(guides.request_image(request.guide).getpixel((0, 0)), (77, 77, 77))
        self.assertEqual(request.denoise, 0)
        self.assertEqual(data["controlnet"]["depth_preprocess"], "none")
        self.assertEqual(data["controlnet"]["strength"], 0.63)
        self.assertEqual(data["prompt"], "Keep the border; change lighting")
        self.assertEqual((data[PLAN_KEY].size.width, data[PLAN_KEY].size.height), (64, 96))
        data[PLAN_KEY].verify_request(data)
        self.assertNotIn("owner_key", str(data[PLAN_KEY].metadata()))
        registry = life.JobRegistry()
        item = registry.create(owner=operation.owner)
        scope = dict(validate_editor_data=lambda *args, **kwargs: None)
        prepare = function_from("ruinedfooocus/app/async_worker.py", "prepare_image_data", scope)
        output = self.root / "inert-worker-output.png"
        self.source.save(output)
        worker = NS(
            prepare_image_data=prepare,
            check_interrupt=lambda values: item.check(),
            _process=Mock(return_value=[str(output)]),
        )
        job = NS(owner=operation.owner, cancelled=threading.Event())
        with (
            life.job_scope(item),
            patch.dict(sys.modules, {"ruinedfooocus.app.async_worker": worker}),
            patch.object(sys.modules["ruinedfooocus.app"], "async_worker", worker, create=True),
        ):
            operation.prepare(job)
            self.assertEqual(operation(job, 123), [str(output)])
        ready = worker._process.call_args.args[0]
        worker._process.assert_called_once()
        self.assertIs(guides.captured(ready), request)
        self.assertEqual(ready["controlnet"]["depth_preprocess"], "none")
        self.assertNotIn("_editor_operation", ready)

    def test_uploaded_normal_conventions_and_original_alpha_are_explicit(self):
        for convention, channels in (
            ("moge-opengl", (0,)),
            ("moge-directx", (0, 1)),
            ("normal-bae", ()),
        ):
            with self.subTest(convention=convention):
                operation, data = self.capture(normal=True, convention=convention, denoise=0.4)
                request = guides.captured(data)
                self.assertEqual(request.compatibility.invert_channels, channels)
                self.assertEqual(request.denoise, 0.4)
                self.assertEqual(request.source.channels, 4)
                self.assertEqual(data["control_union_type"], "normal")

    def test_portrait_auto_keeps_source_and_explicit_pad_target_wins(self):
        self.source = Image.new("RGBA", (832, 1216), (12, 34, 56, 90))
        self.guide = Image.new("RGB", (832, 1216), (77, 77, 77))
        for size in (None, "auto"):
            with self.subTest(size=size):
                _, automatic = self.capture(options=dict(size=size), recommended_size=(1024, 1024))
                self.assertIs(automatic["resolution_auto"], True)
                self.assertEqual(
                    (automatic[PLAN_KEY].size.width, automatic[PLAN_KEY].size.height), (832, 1216)
                )
                self.assertEqual(guides.automatic_source_size(automatic), (832, 1216))
                self.assertEqual(automatic["prompt"], "Keep the border; change lighting")
                self.assertEqual(guides.captured(automatic).denoise, 0)
                automatic[PLAN_KEY].verify_request(automatic)
        _, explicit = self.capture(
            options=dict(size="960x640", image_resize_mode="pad"), recommended_size=(1024, 1024)
        )
        self.assertIs(explicit["resolution_auto"], False)
        self.assertEqual(
            (explicit[PLAN_KEY].size.width, explicit[PLAN_KEY].size.height), (960, 640)
        )
        self.assertEqual(explicit["image_resize_mode"], "pad")
        self.assertEqual(guides.captured(explicit).source.size, (832, 1216))
        self.assertEqual(guides.captured(explicit).guide.size, (832, 1216))
        self.assertIsNone(guides.automatic_source_size(explicit))
        self.assertEqual(explicit["prompt"], "Keep the border; change lighting")
        self.assertEqual(guides.captured(explicit).denoise, 0)
        explicit[PLAN_KEY].verify_request(explicit)

    def test_foreign_owner_changed_source_and_control_identity_reject(self):
        operation, data = self.capture()
        with self.assertRaises(PermissionError):
            guides.validate_request(
                data, store=None, owner=("api", "foreign"), manager=self.manager, family="SDXL"
            )
        request = guides.captured(data)
        changed = dict(
            data,
            **{
                guides.KEY: replace(
                    request, source=capture_source(Image.new("RGBA", (64, 96), "red"))
                )
            },
        )
        with self.assertRaisesRegex(ValueError, "content changed"):
            guides.validate_request(
                changed, store=None, owner=operation.owner, manager=self.manager, family="SDXL"
            )
        control = Path(request.compatibility.model_signature[0])
        control.write_bytes(control.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            guides.validate_request(
                data, store=None, owner=operation.owner, manager=self.manager, family="SDXL"
            )

    def test_exact_frame_missing_media_and_private_tickets_reject(self):
        self.guide = Image.new("RGB", (65, 96))
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.capture()
        for options in (
            dict(input_image=None),
            dict(control_guide=dict(image="x", kind="normal", convention="white-near")),
            dict(controlnet=dict(type="depth", model="x", depth_preprocess="auto")),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.capture(options=options)
        with self.assertRaisesRegex(ValueError, "private Editor"):
            api.images({"_editor_guide_request": {}}, {"family": "SDXL"})
        with self.assertRaisesRegex(ValueError, "compatible control"):
            api.images(
                {"control_guide": {}, "input_image": "x", "controlnet": {"type": "depth-map"}},
                {"family": "SDXL"},
            )

    def test_ordinary_api_has_no_guide_request_and_default_denoise_is_only_when_unset(self):
        _, ordinary = self.capture(
            options=dict(control_guide=None, controlnet=dict(type="img2img", denoise=0.25))
        )
        self.assertNotIn(guides.KEY, ordinary)
        schema = ImageRequest(
            input_image="source",
            controlnet=dict(type="depth", model="control.safetensors"),
            control_guide=dict(image="map", kind="depth", convention="white-near"),
        )
        self.assertEqual(schema.control_guide.denoise, 0.35)
        with self.assertRaises(ValueError):
            ImageRequest(
                input_image="source",
                controlnet=dict(type="depth", model="control.safetensors"),
                control_guide=dict(
                    image="map", kind="depth", convention="white-near", denoise=True
                ),
            )


class PrecomputedGuideMediaBoundaryTests(unittest.TestCase):
    def test_existing_owned_api_media_transport_and_arbitrary_urls(self):
        fixture = http_fixtures.ApiLiveContracts()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        output = Path(self.enterContext(tempfile.TemporaryDirectory())) / "guide.png"
        Image.new("RGB", (64, 96), "gray").save(output)
        fixture.backend.images.side_effect = lambda *args, **kwargs: http_fixtures.job(
            ("result", [str(output)])
        )
        first = fixture.client.post(
            "/v1/images/generations", headers=fixture.headers, json={"prompt": "fixture"}
        )
        owned_url = first.json()["data"][0]["url"]
        payload = dict(
            input_image=encoded(Image.new("RGB", (64, 96))),
            controlnet=dict(type="depth", model="fixture.safetensors"),
            control_guide=dict(image=owned_url, kind="depth", convention="white-near", denoise=0),
        )
        second = fixture.client.post(
            "/v1/images/generations", headers=fixture.headers, json=payload
        )
        self.assertEqual(second.status_code, 200)
        received = fixture.backend.images.call_args.args[0]
        self.assertEqual(
            received["control_guide"]["image"], encoded(Image.new("RGB", (64, 96), "gray"))
        )
        self.assertEqual(received["control_guide"]["denoise"], 0)
        fixture.backend.images.reset_mock()
        payload["control_guide"]["image"] = "https://example.invalid/arbitrary.png"
        with patch("urllib.request.urlopen", side_effect=AssertionError("no remote fetch")):
            refused = fixture.client.post(
                "/v1/images/generations", headers=fixture.headers, json=payload
            )
        self.assertEqual(refused.status_code, 400)
        fixture.backend.images.assert_not_called()


if __name__ == "__main__":
    unittest.main()
