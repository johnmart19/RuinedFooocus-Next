"""API image plans keep catalogue identity distinct from verified transport."""

from dataclasses import replace
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from PIL import Image

from ruinedfooocus.app import api_runtime
from ruinedfooocus.core.path import PathManager
from ruinedfooocus.core.paths import resource_path
from ruinedfooocus.core.preparation import PreparationService
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.models import effective_image_plan as plans
from ruinedfooocus.features.models.model_registry import (
    ModelRegistry,
    download_identity,
    load_registry,
)


def payload():
    header = json.dumps({"fixture.weight": {"shape": [1], "dtype": "F16", "data_offsets": [0, 0]}})
    raw = header.encode()
    return len(raw).to_bytes(8, "little") + raw


class PlanAcquisitionTransportTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.real_registry = load_registry()
        self.enterContext(patch.dict(os.environ, RF_OFFLINE="0", HF_HUB_OFFLINE="0"))
        self.manager = PathManager.__new__(PathManager)
        self.manager.paths = {
            "path_" + category: str(self.root / category)
            for category in ("checkpoints", "clip", "vae", "loras")
        }
        self.manager.model_paths = {}
        self.manager.DOWNLOADABLE_FILES = {}
        self.content = payload()
        self.checkpoint = self.root / "fixture.safetensors"
        self.checkpoint.write_bytes(self.content)
        self.rows, self.payloads = [], {}
        for category, filename in (
            ("clip", "qwen3vl_8b_fp8_scaled.safetensors"),
            ("vae", "ae.safetensors"),
        ):
            key = category + "/" + filename
            info = dict(
                url=f"https://huggingface.co/fixture/components/resolve/main/{filename}",
                path="path_" + category,
                filename=filename,
            )
            self.manager.DOWNLOADABLE_FILES[key] = info
            pin = info["url"].replace("/main/", "/" + "a" * 40 + "/")
            self.payloads[pin] = self.content
            self.rows.append(
                dict(
                    id="existing:" + key,
                    catalogue_key=key,
                    state="existing",
                    profile_id="fixture",
                    role="image_utility_model",
                    chat_weight=False,
                    download_identity=download_identity(info),
                    artifact=dict(
                        bytes=len(self.content),
                        sha256=hashlib.sha256(self.content).hexdigest(),
                        pinned_url=pin,
                    ),
                    companions=dict(text=[], image=[]),
                )
            )
        self.registry = ModelRegistry(
            dict(
                schema_version=1,
                profiles={"fixture": {"consumers": ["Image tests"]}},
                artifacts=self.rows,
                benchmarks=[],
            )
        )
        self.registry_patch = self.enterContext(
            patch(
                "ruinedfooocus.features.models.model_registry.load_registry",
                return_value=self.registry,
            )
        )
        self.models = NS(
            offline=False,
            get_file=lambda category, name: self.checkpoint if category == "checkpoints" else None,
            get_models_by_path=lambda *args, **kwargs: {"baseModel": "Boogu"},
            get_model_base=lambda value: value["baseModel"],
        )
        performance = NS(
            CUSTOM_PERFORMANCE="Own settings",
            choices_for_model=lambda *args: ["Own settings"],
            selection_for_model=lambda *args: "Own settings",
            get_perf_options=lambda *args: dict(
                custom_steps=8, cfg=4, sampler_name="euler", scheduler="simple", clip_skip=1
            ),
        )
        self.runtime = (
            {"performance": "Own settings", "resolution": "default"},
            performance,
            NS(
                CUSTOM_RESOLUTION="Custom...",
                selection_for_model=lambda *args: "default",
                aspect_ratios={"default": (512, 512)},
            ),
            self.manager,
            self.models,
        )

    def capture(self):
        shared = NS()
        samplers = NS(KSampler=NS(SAMPLERS=["euler"], SCHEDULERS=["simple"]))
        with (
            patch.dict(
                sys.modules, {"ruinedfooocus.app.state": shared, "comfy.samplers": samplers}
            ),
            patch.object(sys.modules["ruinedfooocus.app"], "state", shared, create=True),
            patch.object(api_runtime, "_image_runtime_snapshot", return_value=self.runtime),
            patch.object(api_runtime, "image_defaults", dict(base_model_name=self.checkpoint.name)),
            patch.object(api_runtime, "decode_image", return_value=Image.new("RGBA", (64, 48))),
            patch.object(api_runtime, "submit", side_effect=lambda operation: operation),
        ):
            operation = api_runtime.images(
                dict(
                    prompt="Preserve the border",
                    negative_prompt="No lettering",
                    size="512x512",
                    input_image="fixture",
                    native_edit_recipe="boogu-edit",
                ),
                dict(family="Boogu", name=self.checkpoint.name),
            )
        cells = dict(
            zip(
                operation.__code__.co_freevars,
                (cell.cell_contents for cell in operation.__closure__),
            )
        )
        return cells["data"]

    def service(self, *, corrupt=False):
        self.requests = []

        def transfer(request, staging, progress, cancelled):
            self.requests.append(request)
            content = self.payloads[request.url]
            if corrupt:
                content = content[:-1] + bytes([content[-1] ^ 1])
            staged = staging / "payload.part"
            staged.write_bytes(content)
            return staged

        service = PreparationService(transport=transfer, staging_root=self.root / "staging")
        self.addCleanup(service.close)
        return service

    def test_actual_api_plan_and_preparation_share_one_pinned_request(self):
        canonical = copy.deepcopy(self.manager.DOWNLOADABLE_FILES)
        captured = self.capture()
        plan = plans.get_effective_image_plan(captured)
        self.assertIs(plan.components, captured[plans.COMPONENTS_KEY])
        by_key = {item.name: item for item in captured[assets.KEY].artifacts}
        for row in plan.required_components:
            self.assertEqual(row.source, canonical[row.download_key]["url"])
            self.assertIn("/" + "a" * 40 + "/", row.acquisition.url)
            self.assertIs(by_key[row.key].request, row.acquisition)
            self.assertEqual(row.catalogue_identity, download_identity(canonical[row.download_key]))
        json.dumps(plan.metadata(), allow_nan=False)
        ready = assets.prepare_image_assets(
            captured, check_cancelled=lambda: None, service=self.service()
        )
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(all(item.signature is not None for item in ready[assets.KEY].artifacts))
        self.assertEqual(self.manager.DOWNLOADABLE_FILES, canonical)
        self.assertEqual(ready["prompt"], "Preserve the border")
        self.assertEqual(ready["negative"], "No lettering")
        self.assertEqual(ready["controlnet"], {"type": "edit"})
        self.assertEqual(ready["native_edit_recipe"], "boogu-edit")
        self.assertIs(plans.get_effective_image_plan(ready), plan)

    def test_actual_boogu_catalogue_captures_immutable_transport_without_acquisition(self):
        self.registry_patch.return_value = self.real_registry
        self.manager.DOWNLOADABLE_FILES = {}
        for name in ("clip.json", "vae.json"):
            self.manager.DOWNLOADABLE_FILES.update(
                json.loads(resource_path("model_catalogue", name).read_text(encoding="utf-8"))
            )
        canonical = copy.deepcopy(self.manager.DOWNLOADABLE_FILES)
        with patch.object(
            assets.preparation_service, "submit", side_effect=AssertionError("passive capture")
        ):
            captured = self.capture()
        row = next(
            item for item in captured[plans.KEY].required_components if item.category == "clip"
        )
        self.assertEqual(row.source, canonical[row.download_key]["url"])
        self.assertIn("/resolve/main/", row.source)
        self.assertIn("/resolve/8606e7df2f5389282617b8b534db290e1f5d7811/", row.acquisition.url)
        self.assertEqual(row.acquisition.expected_size, 10588637512)
        self.assertEqual(
            row.acquisition.sha256,
            "4ba424cf62e51392e4d1a39933e803706f4e823c1065f36aaf149c6453f66bcd",
        )
        self.assertEqual(self.manager.DOWNLOADABLE_FILES, canonical)

    def test_transport_url_destination_size_and_sha_drift_all_reject(self):
        captured = self.capture()
        row = plans.get_effective_image_plan(captured).required_components[0]
        request = row.acquisition
        original = self.manager.preparation_request
        changes = (
            dict(url=request.url + "?changed"),
            dict(destination=request.destination.with_name("other.safetensors")),
            dict(expected_size=request.expected_size + 1),
            dict(sha256="f" * 64),
        )
        for change in changes:

            def changed(key, *, destination=None):
                return (
                    replace(request, **change)
                    if key == row.download_key
                    else original(key, destination=destination)
                )

            with (
                self.subTest(change=change),
                patch.object(self.manager, "preparation_request", side_effect=changed),
            ):
                with self.assertRaisesRegex(ValueError, "acquisition metadata changed"):
                    assets._plan_component_assets(captured, self.manager, ())

    def test_canonical_source_tuple_drift_rejects_even_when_transport_stays_identical(self):
        captured = self.capture()
        row = plans.get_effective_image_plan(captured).required_components[0]
        for field, value in (
            ("url", row.source + "?changed"),
            ("path", "path_vae"),
            ("filename", "other.safetensors"),
        ):
            manager = copy.copy(self.manager)
            manager.DOWNLOADABLE_FILES = copy.deepcopy(self.manager.DOWNLOADABLE_FILES)
            manager.DOWNLOADABLE_FILES[row.download_key][field] = value
            with (
                self.subTest(field=field),
                patch.object(
                    manager, "preparation_request", return_value=row.acquisition
                ) as producer,
            ):
                with self.assertRaisesRegex(ValueError, "acquisition metadata changed"):
                    assets._plan_component_assets(captured, manager, ())
                producer.assert_not_called()

    def test_corrupt_pinned_transport_still_fails_sha_before_publication(self):
        captured = self.capture()
        with self.assertRaisesRegex(IOError, "SHA256"):
            assets.prepare_image_assets(
                captured, check_cancelled=lambda: None, service=self.service(corrupt=True)
            )
        self.assertFalse(
            any(item.request.destination.exists() for item in captured[assets.KEY].artifacts)
        )


if __name__ == "__main__":
    unittest.main()
