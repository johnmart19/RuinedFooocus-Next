"""Explicit-use acquisition uses captured roles and the existing preparation service."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import replace
import hashlib
import builtins
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.preparation import PreparationRequest, PreparationService
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.image_processing import native_vision
from ruinedfooocus.features.models import (
    effective_image_plan as plans,
    gguf_projectors,
    lora_sources,
)
from ruinedfooocus.features.models.model_registry import ModelRegistry, download_identity
import test_native_vision as fixtures
import test_native_variant_defaults as variants


class FirstUseImageAssetsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.NativeVisionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.manager = self.fixture.root, self.fixture.manager
        self.enterContext(patch.dict(os.environ, RF_OFFLINE="0", HF_HUB_OFFLINE="0"))
        self.rows, self.payloads = [], {}
        self.registry = None
        self.enterContext(
            patch(
                "ruinedfooocus.features.models.model_registry.load_registry",
                side_effect=lambda: self.registry,
            )
        )
        self.manager.model_paths = {"cache_path": self.root / "cache"}

    def register(
        self,
        path,
        family,
        *,
        category="diffusion_models",
        role="image_utility_model",
        architecture=None,
        variant=None,
    ):
        payload = path.read_bytes()
        key = category + "/" + path.name
        info = dict(
            path="path_" + category,
            filename=path.name,
            url="https://example.invalid/immutable/" + category + "/" + path.name,
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        self.manager.DOWNLOADABLE_FILES[key] = info
        self.rows.append(
            dict(
                id="existing:" + key,
                catalogue_key=key,
                state="existing",
                profile_id="utility",
                role=role,
                chat_weight=False,
                native_family=family,
                native_class=architecture or family,
                architecture=architecture or family,
                behavioral_variant=variant,
                download_identity=download_identity(info),
                artifact=dict(bytes=len(payload), sha256=info["sha256"], pinned_url=info["url"]),
                companions=dict(text=[], image=[]),
            )
        )
        self.registry = ModelRegistry(
            dict(
                schema_version=1,
                profiles={"utility": {"consumers": ["Image tools"]}},
                artifacts=self.rows,
                benchmarks=[],
            )
        )
        self.payloads[info["url"]] = payload
        path.unlink()
        return key

    def capture(self, key, **values):
        return assets.capture_image_assets(
            dict(
                input_image=self.fixture.image,
                native_vision_model=key,
                segment_positive_points='[{"x":10,"y":5}]',
                **values,
            ),
            self.manager,
            {"type": "segment"},
        )

    def service(self, transport=None):
        def transfer(request, staging, progress, cancelled):
            payload = self.payloads[request.url]
            progress(len(payload), len(payload), 0)
            path = staging / "payload.part"
            path.write_bytes(payload)
            return path

        service = PreparationService(
            transport=transport or transfer, staging_root=self.root / "staging"
        )
        self.addCleanup(service.close)
        return service

    def test_capture_and_readiness_are_passive_and_only_selected_tool_is_prepared(self):
        key = self.register(self.fixture.files["SAM31"], "SAM31")
        other = self.register(self.fixture.files["RT_DETR_v4"], "RT_DETR_v4")
        with patch.object(
            assets.preparation_service, "submit", side_effect=AssertionError("passive transfer")
        ):
            data = self.capture(
                key, loras=[("inactive.safetensors", 0)], enable_prompt_expansion=True
            )
            readiness = assets.image_readiness(data)
            self.assertEqual(readiness.state, "preparable")
            self.assertEqual(len(readiness.requirements), 1)
            self.assertEqual(readiness.requirements[0].integrity, "SHA256 supplied")
            self.assertEqual(
                readiness.missing_bytes,
                len(self.payloads[self.manager.DOWNLOADABLE_FILES[key]["url"]]),
            )
            native_vision.native_model_choices(self.manager, "segment")
        data = assets.prepare_image_assets(
            data, check_cancelled=lambda: None, service=self.service()
        )
        self.assertEqual(assets.image_readiness(data).state, "ready")
        self.assertFalse(self.manager.preparation_request(other).destination.exists())
        with assets.prepared_asset(
            data, "native-vision", data["native_vision_model"], self.manager, "diffusion_models"
        ) as path:
            self.assertEqual(native_vision.validate_model(path, "segment"), "SAM31")

    def test_registered_source_requires_exact_tuple_compatible_role_and_unambiguous_selection(self):
        key = self.register(self.fixture.files["SAM31"], "SAM31")
        self.manager.DOWNLOADABLE_FILES[key]["url"] += "-changed"
        with self.assertRaisesRegex(ValueError, "no longer matches download"):
            self.capture(key)
        self.manager.DOWNLOADABLE_FILES[key]["url"] = self.rows[0]["artifact"]["pinned_url"]
        wrong = self.register(self.fixture.files["RT_DETR_v4"], "RT_DETR_v4")
        for value in (
            wrong,
            "https://example.invalid/unknown.safetensors",
            "missing-custom.safetensors",
            "../escape.safetensors",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.capture(value)
        duplicate = self.root / "checkpoints" / "SAM31.safetensors"
        duplicate.write_bytes(self.payloads[self.manager.DOWNLOADABLE_FILES[key]["url"]])
        self.register(duplicate, "SAM31", category="checkpoints")
        with self.assertRaisesRegex(ValueError, "one compatible registered"):
            self.capture("SAM31.safetensors")

    def test_capture_freezes_destination_source_and_offline_policy_across_settings_changes(self):
        key = self.register(self.fixture.files["SAM31"], "SAM31")
        data = self.capture(key)
        item = data[assets.KEY].artifacts[0]
        self.manager.root = self.root / "later"
        self.manager.DOWNLOADABLE_FILES[key]["url"] = "https://example.invalid/later"
        with patch.dict(os.environ, RF_OFFLINE="1", HF_HUB_OFFLINE="1"):
            ready = assets.prepare_image_assets(
                data, check_cancelled=lambda: None, service=self.service()
            )
        self.assertEqual(ready[assets.KEY].artifacts[0].request, item.request)
        self.assertTrue(item.request.destination.is_file())
        self.assertFalse((self.manager.root / "diffusion_models").exists())

    def test_offline_utility_and_complete_preflight_never_submit_any_transfer(self):
        key = self.register(self.fixture.files["SAM31"], "SAM31")
        with patch.dict(os.environ, HF_HUB_OFFLINE="1"):
            data = self.capture(key)
        service = self.service(Mock(side_effect=AssertionError("offline transfer")))
        with self.assertRaisesRegex(ValueError, "Offline"):
            assets.prepare_image_assets(data, check_cancelled=lambda: None, service=service)
        online = replace(data[assets.KEY], offline=False)
        bad = assets.ImageArtifact(
            "custom", "missing", PreparationRequest("", self.root / "missing.safetensors"), None
        )
        data[assets.KEY] = replace(online, artifacts=(*online.artifacts, bad))
        with self.assertRaisesRegex(ValueError, "Install it locally"):
            assets.prepare_image_assets(data, check_cancelled=lambda: None, service=service)

    def test_disk_and_unknown_size_policies_are_checked_before_subscribing(self):
        key = self.register(self.fixture.files["SAM31"], "SAM31")
        data = self.capture(key)
        submit = Mock(side_effect=AssertionError("blocked transfer"))
        service = self.service(submit)
        with patch("ruinedfooocus.features.models.acquisition._space", return_value=1):
            with self.assertRaisesRegex(ValueError, "free-space reserve"):
                assets.prepare_image_assets(data, check_cancelled=lambda: None, service=service)
        item = data[assets.KEY].artifacts[0]
        unknown = replace(item, request=replace(item.request, expected_size=None, sha256=None))
        data[assets.KEY] = replace(data[assets.KEY], artifacts=(unknown,), allow_unknown_size=False)
        status = assets.image_readiness(data)
        self.assertEqual(status.requirements[0].integrity, "Unknown")
        self.assertIsNone(status.missing_bytes)
        with self.assertRaisesRegex(ValueError, "unknown download size"):
            assets.prepare_image_assets(data, check_cancelled=lambda: None, service=service)
        submit.assert_not_called()

    def test_failed_transfer_and_explicit_retry_keep_request_identity_and_publish_atomically(self):
        key = self.register(self.fixture.files["SAM31"], "SAM31")
        data, calls = self.capture(key), []

        def transfer(request, staging, progress, cancelled):
            calls.append(request)
            self.assertFalse(request.destination.exists())
            if len(calls) == 1:
                raise IOError("interrupted fixture transfer")
            path = staging / "complete.part"
            path.write_bytes(self.payloads[request.url])
            return path

        service = self.service(transfer)
        with self.assertRaisesRegex(IOError, "interrupted"):
            assets.prepare_image_assets(data, check_cancelled=lambda: None, service=service)
        self.assertFalse(calls[0].destination.exists())
        ready = assets.prepare_image_assets(data, check_cancelled=lambda: None, service=service)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(assets.image_readiness(ready).state, "ready")

    def test_two_subscribers_share_transfer_and_cancel_is_local(self):
        key = self.register(self.fixture.files["SAM31"], "SAM31")
        entered, both, release, cancel = (threading.Event() for _ in range(4))
        self.addCleanup(release.set)
        calls = []

        def transfer(request, staging, progress, cancelled):
            calls.append(request)
            entered.set()
            self.assertTrue(release.wait(4))
            path = staging / "complete.part"
            path.write_bytes(self.payloads[request.url])
            return path

        service = self.service(transfer)
        original, count = service.submit, []

        def subscribe(*args):
            handle = original(*args)
            count.append(handle)
            if len(count) == 2:
                both.set()
            return handle

        service.submit = subscribe

        def check():
            if cancel.is_set():
                raise CancelledError("subscriber cancelled")

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                assets.prepare_image_assets,
                self.capture(key),
                check_cancelled=check,
                service=service,
            )
            self.assertTrue(entered.wait(3))
            second = pool.submit(
                assets.prepare_image_assets,
                self.capture(key),
                check_cancelled=lambda: None,
                service=service,
            )
            self.assertTrue(both.wait(3))
            cancel.set()
            with self.assertRaises(CancelledError):
                first.result(3)
            release.set()
            ready = second.result(3)
        self.assertEqual(len(calls), 1)
        self.assertEqual(assets.image_readiness(ready).state, "ready")

    def test_last_subscriber_cancel_cannot_publish_or_return_a_ready_request(self):
        key = self.register(self.fixture.files["SAM31"], "SAM31")
        entered, release, cancel, finished = (threading.Event() for _ in range(4))
        self.addCleanup(release.set)

        def transfer(request, staging, progress, cancelled):
            entered.set()
            self.assertTrue(release.wait(4))
            path = staging / "complete.part"
            path.write_bytes(self.payloads[request.url])
            finished.set()
            return path

        service = self.service(transfer)
        data = self.capture(key)

        def check():
            if cancel.is_set():
                raise CancelledError("subscriber cancelled")

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                assets.prepare_image_assets, data, check_cancelled=check, service=service
            )
            self.assertTrue(entered.wait(3))
            cancel.set()
            with self.assertRaises(CancelledError):
                future.result(3)
            release.set()
        self.assertTrue(finished.wait(3))
        service.close()
        self.assertFalse(data[assets.KEY].artifacts[0].request.destination.exists())
        self.assertTrue(assets.needs_image_preparation(data))

    def test_verified_receipt_avoids_second_hash_and_metadata_failure_retains_file(self):
        key = self.register(self.fixture.files["SAM31"], "SAM31")
        with (
            patch.object(assets, "verify_file", side_effect=AssertionError("duplicate hash")),
            patch(
                "ruinedfooocus.features.models.model_identity.record_prepared_native_source",
                side_effect=OSError("metadata fixture failure"),
            ) as record,
        ):
            ready = assets.prepare_image_assets(
                self.capture(key), check_cancelled=lambda: None, service=self.service()
            )
        self.assertTrue(record.call_args.args[2])
        self.assertTrue(Path(ready["native_vision_model"]).is_file())
        self.assertEqual(len(assets.image_readiness(ready).source_warnings), 1)

    def test_invalid_digest_never_publishes_and_invalid_reader_never_reaches_compute(self):
        key = self.register(self.fixture.files["SAM31"], "SAM31")
        data = self.capture(key)
        item = data[assets.KEY].artifacts[0]
        corrupted = bytearray(self.payloads[item.request.url])
        corrupted[-1] ^= 1
        self.payloads[item.request.url] = bytes(corrupted)
        with self.assertRaisesRegex(IOError, "SHA256"):
            assets.prepare_image_assets(data, check_cancelled=lambda: None, service=self.service())
        self.assertFalse(item.request.destination.exists())
        self.payloads[item.request.url] = self.fixture.files["wrong"].read_bytes()
        request = replace(
            item.request,
            expected_size=len(self.payloads[item.request.url]),
            sha256=hashlib.sha256(self.payloads[item.request.url]).hexdigest(),
        )
        data[assets.KEY] = replace(data[assets.KEY], artifacts=(replace(item, request=request),))
        with self.assertRaisesRegex(ValueError, "does not support"):
            assets.prepare_image_assets(data, check_cancelled=lambda: None, service=self.service())

    def primary(self, *, shapes=None, metadata=None, variant="base"):
        path = variants.write_header(
            self.root / "checkpoints" / "selected-primary.safetensors",
            shapes or variants.SHAPES["Omnigen2"],
            metadata=metadata,
        )
        return self.register(
            path,
            "Omnigen2",
            category="checkpoints",
            role="image_diffusion_checkpoint",
            variant=variant,
        )

    def test_missing_primary_prepares_only_explicit_trusted_source_and_preserves_manual_intent(
        self,
    ):
        key = self.primary()
        intent = dict(
            base_model_name=key,
            prompt="owned prompt",
            custom_steps=7,
            cfg=2.25,
            loras=[["inactive.safetensors", 0]],
            custom_width=704,
            custom_height=448,
        )
        data = assets.capture_pending_image_selection(intent, self.manager)
        pending = assets.pending_image_selection(data)
        self.assertEqual((pending.catalogue_key, pending.native_family), (key, "Omnigen2"))
        self.assertEqual(assets.image_readiness(data).state, "preparable")
        intent["loras"][0][1] = 1
        intent["custom_steps"] = 100
        service = self.service()
        ready = assets.prepare_pending_image_selection(
            data, check_cancelled=lambda: None, service=service
        )
        self.assertEqual(
            (ready["custom_steps"], ready["cfg"], ready["custom_width"], ready["custom_height"]),
            (7, 2.25, 704, 448),
        )
        self.assertEqual(ready["loras"], [["inactive.safetensors", 0]])
        self.assertTrue(Path(ready["base_model_name"]).is_file())
        self.assertNotIn(assets.PENDING_KEY, ready)
        self.assertNotIn(assets.KEY, ready)
        self.assertEqual(ready["_prepared_primary_asset"].request, pending.primary.request)
        self.assertIsNone(assets.capture_pending_image_selection(ready, self.manager))

    def test_missing_primary_fails_native_class_and_variant_disagreement_before_final_plan(self):
        key = self.primary(shapes=fixtures.HEADERS["SAM31"])
        pending = assets.capture_pending_image_selection(dict(base_model_name=key), self.manager)
        with self.assertRaisesRegex(ValueError, "native header differs"):
            assets.prepare_pending_image_selection(
                pending, check_cancelled=lambda: None, service=self.service()
            )
        self.assertNotIn("_effective_image_plan", pending)
        self.rows.clear()
        key = self.primary(metadata={"modelspec.variant": "turbo"})
        pending = assets.capture_pending_image_selection(dict(base_model_name=key), self.manager)
        with self.assertRaisesRegex(ValueError, "different behavioral variant"):
            assets.prepare_pending_image_selection(
                pending, check_cancelled=lambda: None, service=self.service()
            )

    def test_missing_primary_source_drift_offline_and_custom_paths_do_not_transfer(self):
        key = self.primary()
        data = assets.capture_pending_image_selection(
            dict(base_model_name=key), self.manager, offline=True
        )
        service = self.service(Mock(side_effect=AssertionError("blocked primary transfer")))
        self.assertEqual(assets.image_readiness(data).state, "blocked")
        with self.assertRaisesRegex(ValueError, "Offline"):
            assets.prepare_pending_image_selection(
                data, check_cancelled=lambda: None, service=service
            )
        self.assertIsNone(
            assets.capture_pending_image_selection(
                dict(base_model_name="missing-custom.safetensors"), self.manager
            )
        )
        self.rows[0]["behavioral_variant"] = "changed"
        self.registry = ModelRegistry(
            dict(
                schema_version=1,
                profiles={"utility": {"consumers": []}},
                artifacts=self.rows,
                benchmarks=[],
            )
        )
        with self.assertRaisesRegex(ValueError, "source contract changed"):
            assets.prepare_pending_image_selection(
                data, check_cancelled=lambda: None, service=service
            )

    def test_utilities_and_passive_filter_capture_do_not_prepare_hidden_primary(self):
        key = self.primary()
        data = dict(
            base_model_name=key,
            input_image=self.fixture.image,
            controlnet={"type": "adjust-contrast"},
        )
        with patch.object(
            assets.preparation_service, "submit", side_effect=AssertionError("preview acquisition")
        ):
            self.assertIsNone(assets.capture_pending_image_selection(data, self.manager))
            captured = assets.capture_image_assets(data, self.manager, data["controlnet"])
            self.assertEqual(assets.image_readiness(captured).requirements, ())
            self.assertFalse(assets.needs_image_preparation(captured))
        self.assertFalse(self.manager.preparation_request(key).destination.exists())

    def test_primary_class_can_be_validated_with_native_source_predicates_without_runtime_import(
        self,
    ):
        key = self.primary()
        data = assets.capture_pending_image_selection(dict(base_model_name=key), self.manager)
        with (
            patch.dict("sys.modules", {"comfy.model_detection": None}),
            patch(
                "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                side_effect=AssertionError("runtime detector import"),
            ),
        ):
            ready = assets.prepare_pending_image_selection(
                data, check_cancelled=lambda: None, service=self.service()
            )
        self.assertTrue(Path(ready["base_model_name"]).is_file())

    def test_first_use_primary_stays_under_existing_native_component_read_scope(self):
        from ruinedfooocus.core.preparation import PreparationBusy

        key = self.primary()
        service = self.service()
        data = assets.prepare_pending_image_selection(
            assets.capture_pending_image_selection(dict(base_model_name=key), self.manager),
            check_cancelled=lambda: None,
            service=service,
        )
        primary = data["_prepared_primary_asset"]
        data[assets.KEY] = assets.ImageAssets((primary,), False)

        @assets.component_read_leases
        def load(prepared_data):
            with self.assertRaises(PreparationBusy):
                with service.artifact_lease(
                    primary.request.destination, write=True, blocking=False
                ):
                    self.fail("primary was not protected during native loading")

        with patch.object(assets, "preparation_service", service):
            load(prepared_data=data)
        with service.artifact_lease(primary.request.destination, write=True, blocking=False):
            pass

    def test_p1_required_components_are_consumed_without_recapturing_optional_alternatives(self):
        from ruinedfooocus.features.models.component_profiles import snapshot_request
        from ruinedfooocus.features.models.resource_defaults import RESOURCE_DEFAULTS

        checkpoint = variants.write_header(
            self.root / "checkpoints" / "ordinary.safetensors", variants.SHAPES["Omnigen2"]
        )
        models = SimpleNamespace(
            get_file=lambda *args: checkpoint,
            get_models_by_path=lambda *args, **kwargs: {"baseModel": "Omnigen2"},
            get_model_base=lambda info: info["baseModel"],
        )
        for role in ("clip_omnigen2_3b", "vae_flux"):
            self.manager.add(
                role, "clip" if role.startswith("clip_") else "vae", RESOURCE_DEFAULTS[role]
            )
            self.payloads[self.manager.DOWNLOADABLE_FILES[role]["url"]] = b"good"
        self.registry = ModelRegistry(
            dict(schema_version=1, profiles={}, artifacts=[], benchmarks=[])
        )
        captured = snapshot_request(
            dict(base_model_name=checkpoint.name, custom_steps=7, cfg=2, controlnet={}),
            models,
            {},
            self.manager,
        )
        captured = plans.capture_effective_image_plan(captured, models, {}, self.manager)
        plan = plans.get_effective_image_plan(captured)
        optional = replace(plan.components[0], key="inactive-alternative", active=False)
        unrequired = replace(plan.components[0], key="optional-alternative", required=False)
        components = (*plan.components, optional, unrequired)
        captured[plans.COMPONENTS_KEY] = components
        captured[plans.KEY] = replace(plan, components=components)
        with patch(
            "ruinedfooocus.features.models.component_profiles.component_inventory",
            side_effect=AssertionError("plan inventory recaptured"),
        ):
            request = assets.capture_image_assets(captured, self.manager, {}, models=models)
        request.pop(plans.KEY)
        request = plans.capture_effective_image_plan(request, models, {}, self.manager)
        self.assertEqual(
            {item.name for item in request[assets.KEY].artifacts}, {"clip_omnigen2_3b", "vae_flux"}
        )
        ready = assets.prepare_image_assets(
            request, check_cancelled=lambda: None, service=self.service()
        )
        self.assertEqual(assets.image_readiness(ready).state, "ready")
        with self.assertRaisesRegex(ValueError, "request changed"):
            assets.prepare_image_assets(
                request | {"cfg": 99}, check_cancelled=lambda: None, service=self.service()
            )

    def test_seedvr2_first_use_prepares_only_selected_model_and_matching_codec(self):
        import test_native_seedvr2 as seed_fixtures

        fixture = seed_fixtures.NativeSeedVR2Tests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.manager = fixture.root, fixture.manager
        self.manager.model_paths = {"cache_path": self.root / "cache"}
        model = self.register(fixture.model, "SeedVR2")
        codec = self.register(
            fixture.codec, None, category="vae", role="image_codec", architecture="SeedVR2 codec"
        )
        request = assets.capture_image_assets(
            dict(
                input_image=fixture.image, seedvr2_model=model, seedvr2_vae=codec, seedvr2_scale=1
            ),
            self.manager,
            {"type": "restore-seedvr2"},
        )
        self.assertEqual(
            {item.role for item in request[assets.KEY].artifacts}, {"seedvr2-model", "seedvr2-vae"}
        )
        ready = assets.prepare_image_assets(
            request, check_cancelled=lambda: None, service=self.service()
        )
        self.assertEqual(assets.image_readiness(ready).state, "ready")
        self.assertEqual(ready["input_image"].tobytes(), fixture.image.tobytes())

    def test_depth_utility_first_use_finalizes_captured_file_signatures_before_compute(self):
        import test_depth_utilities as depth_fixtures
        from ruinedfooocus.features.image_processing import depth_utilities

        fixture = depth_fixtures.DepthUtilities()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.manager = fixture.root, fixture.manager
        self.manager.model_paths = {"cache_path": self.root / "cache"}
        model = self.register(fixture.pose, "SDPose", category="checkpoints")
        request = assets.capture_image_assets(
            dict(input_image=fixture.source, image_utility_model=model),
            self.manager,
            {"type": "pose-map"},
        )
        self.assertIsNone(request[depth_utilities.KEY].files[0].signature)
        from ruinedfooocus.features.image_processing import still_image_utilities

        owner = still_image_utilities._handler("pose-map")
        self.assertIsInstance(request[owner.KEY], owner.UtilityRequest)
        ready = assets.prepare_image_assets(
            request, check_cancelled=lambda: None, service=self.service()
        )
        self.assertIsNotNone(ready[depth_utilities.KEY].files[0].signature)
        self.assertEqual(
            ready[depth_utilities.KEY].files[0].signature, ready[assets.KEY].artifacts[0].signature
        )

    def test_sam_text_requires_selected_checkpoint_language_encoder_without_separate_download(self):
        key = self.register(self.fixture.files["SAM31"], "SAM31")
        request = self.capture(key, segment_text="cat:1")
        self.assertEqual(len(request[assets.KEY].artifacts), 1)
        with self.assertRaisesRegex(ValueError, "paired language encoder"):
            assets.prepare_image_assets(
                request, check_cancelled=lambda: None, service=self.service()
            )

    def test_pending_primary_cannot_skip_validation_through_companion_entrypoint(self):
        key = self.primary()
        request = assets.capture_pending_image_selection(dict(base_model_name=key), self.manager)
        with self.assertRaisesRegex(ValueError, "recapture its image plan"):
            assets.prepare_image_assets(
                request, check_cancelled=lambda: None, service=self.service()
            )
        self.assertFalse(self.manager.preparation_request(key).destination.exists())

    def test_pending_capture_bypasses_local_commands_and_preserves_literal_image_prompts(self):
        key = self.primary()
        for prompt in ("RuinedFooocusLogo", "#!example", "SEARCH: example"):
            data = dict(base_model_name=key, prompt=prompt)
            with (
                self.subTest(prompt=prompt),
                patch(
                    "ruinedfooocus.features.models.model_registry.load_registry",
                    side_effect=AssertionError("local command inspected image source"),
                ),
            ):
                self.assertIsNone(assets.capture_pending_image_selection(data, self.manager))
            for literal in (
                {"task_type": "tool_call"},
                {"_api_job": object()},
                {"_metadata_prompt": True},
            ):
                with self.subTest(prompt=prompt, literal=tuple(literal)):
                    captured = assets.capture_pending_image_selection(
                        dict(data, **literal), self.manager
                    )
                    self.assertIsNotNone(assets.pending_image_selection(captured))
        self.assertFalse(self.manager.preparation_request(key).destination.exists())

    def test_primary_source_contract_change_during_transfer_retains_file_and_blocks_final_capture(
        self,
    ):
        key = self.primary()
        data = assets.capture_pending_image_selection(dict(base_model_name=key), self.manager)

        def transfer(request, staging, progress, cancelled):
            path = staging / "complete.part"
            path.write_bytes(self.payloads[request.url])
            self.rows[0]["behavioral_variant"] = "changed"
            self.registry = ModelRegistry(
                dict(schema_version=1, profiles={"utility": {}}, artifacts=self.rows, benchmarks=[])
            )
            return path

        with self.assertRaisesRegex(ValueError, "source contract changed during"):
            assets.prepare_pending_image_selection(
                data, check_cancelled=lambda: None, service=self.service(transfer)
            )
        self.assertTrue(self.manager.preparation_request(key).destination.is_file())
        self.assertIsNotNone(assets.pending_image_selection(data))

    def test_registered_quantized_primary_uses_existing_gguf_header_reader(self):
        import gguf
        import numpy as np

        path = self.root / "checkpoints" / "selected-primary.gguf"
        writer = gguf.GGUFWriter(str(path), "omnigen2")
        writer.add_tensor(
            next(iter(variants.SHAPES["Omnigen2"])),
            gguf.quantize(np.zeros(32, dtype=np.float32), gguf.GGMLQuantizationType.Q8_0),
            raw_dtype=gguf.GGMLQuantizationType.Q8_0,
        )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        key = self.register(
            path,
            "Omnigen2",
            category="checkpoints",
            role="image_diffusion_checkpoint",
            variant="base",
        )
        request = assets.capture_pending_image_selection(dict(base_model_name=key), self.manager)
        original_import = builtins.__import__

        def no_native_import(name, *args, **kwargs):
            if (
                name == "torch"
                or name.startswith("torch.")
                or name == "comfy"
                or name.startswith("comfy.")
                or name == "nodes"
                or name.startswith("comfy_extras")
            ):
                raise AssertionError("Cold preparation imported native compute: " + name)
            return original_import(name, *args, **kwargs)

        with (
            patch.dict("sys.modules"),
            patch.object(builtins, "__import__", side_effect=no_native_import),
            patch(
                "ruinedfooocus.features.models.model_architecture.inspect_model_file",
                side_effect=AssertionError("warm detector used by cold preparation"),
            ),
        ):
            sys.modules.pop("comfy.model_detection", None)
            self.assertEqual(assets.image_readiness(request).state, "preparable")
            ready = assets.prepare_pending_image_selection(
                request, check_cancelled=lambda: None, service=self.service()
            )
        self.assertEqual(Path(ready["base_model_name"]).suffix, ".gguf")

    def test_present_but_uncaptured_file_needs_verification_and_passive_stat_races_stay_typed(self):
        key = self.register(self.fixture.files["SAM31"], "SAM31")
        data = self.capture(key)
        request = data[assets.KEY].artifacts[0].request
        request.destination.write_bytes(self.payloads[request.url])
        self.assertEqual(assets.image_readiness(data).state, "preparable")
        self.assertTrue(assets.needs_image_preparation(data))
        ready = assets.prepare_image_assets(
            data, check_cancelled=lambda: None, service=self.service()
        )
        with patch.object(
            assets, "file_signature", side_effect=FileNotFoundError("racing removal")
        ):
            status = assets.image_readiness(ready)
        self.assertEqual(status.state, "blocked")
        self.assertIn("removed after submission", status.reasons[0])

    def test_missing_tool_semantic_role_and_explicit_access_gates_are_required_before_transfer(
        self,
    ):
        key = self.register(self.fixture.files["SAM31"], "SAM31", role="image_diffusion_checkpoint")
        with self.assertRaisesRegex(ValueError, "compatible registered source"):
            self.capture(key)
        self.rows[0].update(role="image_utility_model", access="gated")
        self.registry = ModelRegistry(
            dict(schema_version=1, profiles={"utility": {}}, artifacts=self.rows, benchmarks=[])
        )
        with self.assertRaisesRegex(ValueError, "requires access approval"):
            self.capture(key)
        self.assertFalse(self.manager.preparation_request(key).destination.exists())

    def test_acquired_gguf_encoder_finalizes_only_its_new_projector_packet_in_the_plan(self):
        from ruinedfooocus.features.models.component_profiles import snapshot_request
        from ruinedfooocus.features.models.resource_defaults import RESOURCE_DEFAULTS
        import test_gguf_projector_dependencies as projector_fixtures

        checkpoint = variants.write_header(
            self.root / "checkpoints" / "ordinary.safetensors", variants.SHAPES["Omnigen2"]
        )
        models = SimpleNamespace(
            get_file=lambda *args: checkpoint,
            get_models_by_path=lambda *args, **kwargs: {"baseModel": "Omnigen2"},
            get_model_base=lambda info: info["baseModel"],
        )
        temporary = projector_fixtures.tiny_gguf(self.root / "Qwen-Q4_K_M.gguf")
        payload = temporary.read_bytes()
        temporary.unlink()
        self.manager.add("clip_omnigen2_3b", "clip", "Qwen-Q4_K_M.gguf")
        self.manager.add("vae_flux", "vae", RESOURCE_DEFAULTS["vae_flux"])
        info = self.manager.DOWNLOADABLE_FILES["clip_omnigen2_3b"]
        info.update(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        self.payloads[info["url"]] = payload
        self.payloads[self.manager.DOWNLOADABLE_FILES["vae_flux"]["url"]] = b"good"
        self.registry = ModelRegistry(
            dict(schema_version=1, profiles={}, artifacts=[], benchmarks=[])
        )
        self.enterContext(patch.dict(RESOURCE_DEFAULTS, clip_omnigen2_3b="Qwen-Q4_K_M.gguf"))
        values = {}
        request = snapshot_request(
            dict(base_model_name=checkpoint.name, custom_steps=7, cfg=2, controlnet={}),
            models,
            values,
            self.manager,
        )
        request = plans.capture_effective_image_plan(request, models, values, self.manager)
        request = assets.capture_image_assets(
            request, self.manager, {}, models=models, values=values
        )
        request.pop(plans.KEY)
        request = plans.capture_effective_image_plan(request, models, values, self.manager)
        before = plans.get_effective_image_plan(request)
        self.assertEqual(gguf_projectors.captured(request), ())
        with patch.object(
            gguf_projectors, "_source", return_value=projector_fixtures.NATIVE_GGUF / "loader.py"
        ):
            ready = assets.prepare_image_assets(
                request, check_cancelled=lambda: None, service=self.service()
            )
        after = plans.get_effective_image_plan(ready)
        self.assertEqual(len(gguf_projectors.captured(ready)), 1)
        self.assertIsNone(gguf_projectors.captured(ready)[0].projector)
        self.assertIs(after.components, before.components)
        self.assertIs(after.recipe, before.recipe)
        self.assertIs(after.size, before.size)
        after.verify_request(ready)

    def pending_lora_plan(self, name, payload, *, prompt="", loras=None):
        from ruinedfooocus.features.models.component_profiles import snapshot_request
        from ruinedfooocus.features.models.resource_defaults import RESOURCE_DEFAULTS

        checkpoint = variants.write_header(
            self.root / "checkpoints" / "lora-primary.safetensors", variants.SHAPES["Omnigen2"]
        )
        models = SimpleNamespace(
            model_dirs={"loras": [self.root / "loras"]},
            get_file=lambda category, filename: (
                path if (path := self.root / category / filename).is_file() else None
            ),
            get_models_by_path=lambda *args, **kwargs: {"baseModel": "Omnigen2"},
            get_model_base=lambda info: info["baseModel"],
            get_names=lambda category: [name, "inactive.safetensors"],
            get_model_path=Mock(side_effect=AssertionError("late LoRA resolver")),
        )
        for role in ("clip_omnigen2_3b", "vae_flux"):
            self.manager.add(
                role, "clip" if role.startswith("clip_") else "vae", RESOURCE_DEFAULTS[role]
            )
            destination = self.manager.preparation_request(role).destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"good")
        key = "trusted:" + name
        self.manager.add(key, "loras", name)
        entry = self.manager.DOWNLOADABLE_FILES[key]
        entry.update(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        self.payloads[entry["url"]] = payload
        self.manager.add("inactive-lora", "loras", "inactive.safetensors")
        self.registry = ModelRegistry(
            dict(schema_version=1, profiles={}, artifacts=[], benchmarks=[])
        )
        request = snapshot_request(
            dict(
                base_model_name=checkpoint.name,
                custom_steps=7,
                cfg=2,
                controlnet={},
                prompt=prompt,
                loras=(
                    [["thumbnail", f"-0.5 - {name}"], ["thumbnail", "0 - inactive.safetensors"]]
                    if loras is None
                    else loras
                ),
            ),
            models,
            {},
            self.manager,
        )
        request = plans.capture_effective_image_plan(request, models, {}, self.manager)
        request = assets.capture_image_assets(request, self.manager, {}, models=models)
        request.pop(plans.KEY)
        return plans.capture_effective_image_plan(request, models, {}, self.manager), models, key

    def test_active_trusted_native_and_gguf_loras_use_preparation_and_bind_frozen_files(self):
        import test_lora_source_capture as lora_fixtures

        folder = self.root / "loras"
        folder.mkdir()
        for name in ("active.safetensors", "active.gguf"):
            with self.subTest(name=name):
                path = (
                    lora_fixtures.LoraSourceCaptureTests.gguf(self, name, folder)
                    if name.endswith(".gguf")
                    else variants.write_header(folder / name, {"linear.lora_up.weight": [2, 1]})
                )
                payload = path.read_bytes()
                path.unlink()
                request, models, key = self.pending_lora_plan(name, payload)
                before = plans.get_effective_image_plan(request)
                self.assertIsNone(lora_sources.captured(request)[0].signature)
                self.assertEqual(
                    [
                        (item.role, item.name)
                        for item in request[assets.KEY].artifacts
                        if item.role == "loras"
                    ],
                    [("loras", "lora:" + name)],
                )
                models.model_dirs["loras"] = [self.root / "later-loras"]
                models.get_file = Mock(side_effect=AssertionError("late LoRA root lookup"))
                ready = assets.prepare_image_assets(
                    request, check_cancelled=lambda: None, service=self.service()
                )
                bound = lora_sources.captured(ready)[0]
                self.assertEqual(bound.path, self.manager.preparation_request(key).destination)
                self.assertEqual(bound.path.read_bytes(), payload)
                self.assertIsNotNone(bound.signature)
                self.assertIs(plans.get_effective_image_plan(ready).components, before.components)
                plans.get_effective_image_plan(ready).verify_request(ready)
                self.assertFalse(
                    self.manager.preparation_request("inactive-lora").destination.exists()
                )
                models.get_model_path.assert_not_called()
                models.get_file.assert_not_called()

    def test_prompt_batch_shares_one_lora_transfer_and_preserves_each_child_selection(self):
        name, payload = "shared.safetensors", b"tiny selected LoRA fixture"
        prompts = [f"first <lora:{name}:0.5>", f"second <lora:{name}:-0.25>", "third without LoRA"]
        request, _, key = self.pending_lora_plan(
            name,
            payload,
            prompt=prompts,
            loras=[["thumbnail", "0 - inactive.safetensors"]],
        )
        prompt_table = lora_sources.captured_prompts(request)
        selected = [item for item in request[assets.KEY].artifacts if item.role == "loras"]
        self.assertEqual(len(selected), 1)
        transfers = []

        def transfer(acquisition, staging, progress, cancelled):
            transfers.append(acquisition)
            path = staging / "payload.part"
            path.write_bytes(self.payloads[acquisition.url])
            return path

        ready = assets.prepare_image_assets(
            request, check_cancelled=lambda: None, service=self.service(transfer)
        )
        self.assertEqual(transfers, [self.manager.preparation_request(key)])
        self.assertIs(lora_sources.captured_prompts(ready), prompt_table)
        union = lora_sources.captured(ready)
        for prompt, weights in zip(prompts, ([0.5], [-0.25], []), strict=True):
            child = lora_sources.sources_for_prompt(prompt, union, prompt_table)
            self.assertEqual([item.weight for item in child], weights)
            self.assertTrue(all(item.signature is not None for item in child))
        plans.get_effective_image_plan(ready).verify_request(ready)
        self.assertFalse(self.manager.preparation_request("inactive-lora").destination.exists())

    def test_pending_lora_digest_drift_is_rejected_before_staging(self):
        request, models, key = self.pending_lora_plan("active.safetensors", b"tiny fixture")
        self.manager.DOWNLOADABLE_FILES[key]["sha256"] = "b" * 64
        with (
            patch.object(
                assets.preparation_service, "submit", side_effect=AssertionError("drift transfer")
            ),
            self.assertRaisesRegex(ValueError, "LoRA acquisition metadata changed"),
        ):
            assets.capture_image_assets(request, self.manager, {}, models=models)
        self.assertFalse(self.manager.preparation_request(key).destination.exists())


if __name__ == "__main__":
    unittest.main()
