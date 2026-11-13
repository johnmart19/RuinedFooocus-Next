"""Pending-primary orchestration through real API/worker callers; inert transfers."""

import base64
from concurrent.futures import CancelledError
from dataclasses import replace
import hashlib
import io
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from ruinedfooocus.app import api_runtime
from ruinedfooocus.app import resource_lifecycle as life
from ruinedfooocus.core.preparation import PreparationRequest
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.models.effective_image_plan import get_effective_image_plan
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.model_registry import ModelRegistry, download_identity
import test_chat_image_service as service_fixtures
import test_image_preparation as preparation_fixtures


def encoded(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


class PendingPaths(preparation_fixtures.Manager):
    def get_model_folders(self, category):
        return [Path(value) for value in self.paths.get("path_" + category, ())]

    def preparation_request(self, key, *, destination=None):
        entry = self.DOWNLOADABLE_FILES[key]
        category = entry["path"].removeprefix("path_")
        return PreparationRequest(
            entry["url"],
            destination or self.get_model_folders(category)[0] / entry["filename"],
            entry["size"],
            entry["sha256"],
        )


class PendingModels:
    def __init__(self, root):
        self.offline = False
        self.model_dirs = {category: [root / category] for category in ("checkpoints", "loras")}
        self.cache_paths = {category: root / "cache" / category for category in self.model_dirs}
        self.names = {"checkpoints": {}, "loras": {"manual.safetensors": {}}}

    def get_models_by_path(self, *args, **kwargs):
        return {"baseModel": "SDXL"}

    def get_model_base(self, row):
        return row["baseModel"]

    def get_file(self, category, name):
        for folder in self.model_dirs.get(category, ()):
            candidate = folder / name
            if candidate.is_file():
                return candidate
        return None

    def get_names(self, category):
        return list(self.names.get(category, {}))


class PendingImageCallerTests(unittest.TestCase):
    def setUp(self):
        service_fixtures.ChatImageServiceTests.setUp(self)
        self.folder = self.folder.resolve()
        self.manager = PendingPaths(self.folder)
        self.manager.paths.update(
            {
                "path_" + category: [str(self.folder / category)]
                for category in ("checkpoints", "diffusion_models", "loras")
            }
        )
        self.manager.model_paths = {
            "cache_path": self.folder / "cache",
            "temp_outputs_path": self.folder,
        }
        for category in ("checkpoints", "loras", "clip", "vae"):
            (self.folder / category).mkdir()
        self.shared.path_manager = self.manager
        self.shared.settings.default_settings.update(
            performance="Own settings", resolution="square"
        )
        self.shared.models = PendingModels(self.folder)
        self.destination = self.folder / "checkpoints" / "pending.safetensors"
        self.primary_bytes = self.checkpoint.read_bytes()
        self.key = "checkpoints/pending.safetensors"
        self.info = dict(
            path="path_checkpoints",
            filename=self.destination.name,
            url="https://example.invalid/immutable/pending.safetensors",
            size=len(self.primary_bytes),
            sha256=hashlib.sha256(self.primary_bytes).hexdigest(),
        )
        self.manager.DOWNLOADABLE_FILES[self.key] = self.info
        self.row = dict(
            id="existing:" + self.key,
            catalogue_key=self.key,
            state="existing",
            profile_id="fixture",
            role="image_diffusion_checkpoint",
            chat_weight=False,
            native_family="SDXL",
            native_class="SDXL",
            architecture="SDXL",
            behavioral_variant="base",
            download_identity=download_identity(self.info),
            artifact=dict(
                bytes=len(self.primary_bytes),
                sha256=self.info["sha256"],
                pinned_url=self.info["url"],
            ),
            companions=dict(text=[], image=[]),
        )
        self.catalogue = self.registry_for(self.row)
        self.enterContext(
            patch(
                "ruinedfooocus.features.models.model_registry.load_registry",
                side_effect=lambda: self.catalogue,
            )
        )
        (self.folder / "loras" / "manual.safetensors").write_bytes(b"inactive adapter fixture")
        self.shared.performance_settings = SimpleNamespace(
            CUSTOM_PERFORMANCE="Own settings",
            choices_for_model=lambda *a: ["Own settings"],
            selection_for_model=lambda *a: "Own settings",
            get_perf_options=lambda *a: dict(
                custom_steps=9, cfg=4.5, sampler_name="euler", scheduler="normal", clip_skip=1
            ),
            apply=lambda data, family: dict(data),
        )
        self.shared.resolution_settings.selection_for_model = lambda *a: "square"
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "comfy.samplers": SimpleNamespace(
                        KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["normal"])
                    )
                },
            )
        )
        self.selected = dict(name=self.key, family="SDXL", kind="image")
        self.defaults = dict(self.settings, base_model_name=self.key)
        self.payload = dict(
            prompt="Exactly two blue cups",
            negative_prompt="No words or crop",
            size="128x64",
            steps=13,
            cfg=3.5,
            sampler="euler",
            scheduler="normal",
            seed=17,
            loras=[{"name": "manual.safetensors", "strength": 0.0}],
        )
        self.job = SimpleNamespace(cancelled=self.item.cancelled)
        self.primary = Mock(side_effect=self.install_primary)
        self.enterContext(patch.object(assets, "prepare_pending_image_selection", self.primary))

    def registry_for(self, row):
        return ModelRegistry(
            dict(
                schema_version=1,
                profiles={"fixture": {"consumers": ["Create"]}},
                artifacts=[row],
                benchmarks=[],
            )
        )

    def install_primary(self, data, *, check_cancelled, progress=None):
        check_cancelled()
        pending = assets.pending_image_selection(data)
        self.assertIsNotNone(pending)
        path = pending.primary.request.destination
        path.write_bytes(self.primary_bytes)
        result = dict(
            data,
            base_model_name=str(path),
            _prepared_primary_asset=replace(pending.primary, signature=tuple(file_signature(path))),
            _image_source_warnings=(),
            _image_preparation_policy=(0, True),
        )
        result.pop(assets.PENDING_KEY, None)
        result.pop(assets.KEY, None)
        return result

    def operation(self, payload=None):
        return api_runtime._image_operation(
            payload or self.payload, self.selected, self.defaults, owner=self.item.owner
        )

    def assert_manual_plan(self, data):
        plan = get_effective_image_plan(data)
        self.assertIsNotNone(plan)
        self.assertEqual((plan.size.width, plan.size.height), (128, 64))
        self.assertEqual(data["negative"], "No words or crop")
        self.assertEqual(data["seed"], 17)
        self.assertEqual(plan.recipe.sampling["custom_steps"], 13)
        self.assertEqual(plan.recipe.sampling["cfg"], 3.5)
        self.assertIn("manual.safetensors", str(data["loras"]))
        self.assertIn("0.0", str(data["loras"]))
        plan.verify_request(data)

    def change_live_roots_and_inventory(self):
        later = self.folder / "later-navigation"
        self.manager.root = later
        self.manager.paths["path_checkpoints"][0] = str(later / "checkpoints")
        self.manager.paths["path_loras"][0] = str(later / "loras")
        self.manager.model_paths["cache_path"] = later / "cache"
        self.shared.models.model_dirs["loras"][0] = later / "loras"
        self.shared.models.cache_paths["checkpoints"] = later / "cache" / "checkpoints"
        self.shared.models.names["loras"].clear()
        self.shared.models.names["loras"]["later.safetensors"] = {}
        self.manager.DOWNLOADABLE_FILES[self.key]["url"] = "https://example.invalid/later-source"

    def assert_frozen_roots_and_inventory(self, data):
        paths, models = data["_image_capture_paths"], data["_image_capture_models"]
        self.assertIsNot(paths, self.manager)
        self.assertIsNot(models, self.shared.models)
        self.assertEqual(paths.get_model_folders("checkpoints"), [self.folder / "checkpoints"])
        self.assertEqual(paths.get_model_folders("loras"), [self.folder / "loras"])
        self.assertEqual(paths.model_paths["cache_path"], self.folder / "cache")
        self.assertEqual(models.model_dirs["loras"], [self.folder / "loras"])
        self.assertEqual(models.cache_paths["checkpoints"], self.folder / "cache" / "checkpoints")
        self.assertEqual(models.get_names("loras"), ["manual.safetensors"])
        self.assertEqual(
            models.get_file("loras", "manual.safetensors"),
            self.folder / "loras" / "manual.safetensors",
        )
        self.assertEqual(
            paths.DOWNLOADABLE_FILES[self.key]["url"],
            "https://example.invalid/immutable/pending.safetensors",
        )

    def test_api_invalid_source_mask_size_and_options_never_prepare_the_primary(self):
        source = encoded(Image.new("RGB", (64, 64)))
        mismatched_mask = encoded(Image.new("L", (32, 32)))
        for options in (
            {"input_image": "malformed image bytes"},
            {"input_image": source, "mask": mismatched_mask},
            {"size": "1x1"},
            {"size": "999999x64"},
            {"steps": 0},
            {"automatic_upscale": "not-a-listed-upscaler"},
        ):
            with (
                self.subTest(
                    options={
                        key: "fixture image" if key in ("input_image", "mask") else value
                        for key, value in options.items()
                    }
                ),
                patch.object(self.worker, "prepare_image_data") as companions,
            ):
                self.primary.reset_mock()
                error = None
                try:
                    self.operation(dict(self.payload, **options)).prepare(self.job)
                except ValueError as observed:
                    error = observed
                finally:
                    if self.destination.exists():
                        self.destination.unlink()
                self.assertIsNotNone(error)
                self.assertEqual(
                    self.primary.call_count,
                    0,
                    "Invalid request prepared its primary before rejection",
                )
                companions.assert_not_called()
                self.compute.assert_not_called()

    def test_api_primary_failure_or_cancel_prevents_companion_preparation_and_compute(self):
        for failure in (IOError("fixture transfer failed"), CancelledError("fixture cancelled")):
            with (
                self.subTest(failure=failure),
                patch.object(self.worker, "prepare_image_data") as companions,
            ):
                self.primary.side_effect = failure
                operation = self.operation()
                with self.assertRaises(type(failure)):
                    operation.prepare(self.job)
                companions.assert_not_called()
                self.compute.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, "preparation has not completed"):
                    operation(self.job, 7)

    def test_api_final_recapture_preserves_manual_plan_after_pending_transfer(self):
        operation = self.operation()
        self.primary.assert_not_called()
        with patch.object(
            self.worker, "prepare_image_data", wraps=self.worker.prepare_image_data
        ) as prepare:
            operation.prepare(self.job)
        self.primary.assert_called_once()
        self.assert_manual_plan(prepare.call_args.args[0])
        output = self.folder / "output.png"
        Image.new("RGB", (128, 64), "blue").save(output)
        self.compute.return_value = [str(output)]
        self.assertEqual(operation(self.job, 7), [str(output)])
        self.assert_manual_plan(self.compute.call_args.args[0])

    def test_api_owned_source_is_revalidated_before_primary_transfer(self):
        source = self.folder / "owned-source.png"
        Image.new("RGB", (64, 64), "white").save(source)
        snapshot = self.store.start(source, owner=self.item.owner)
        context = dict(
            session_id=snapshot.session_id,
            revision=snapshot.revision,
            source_id=snapshot.current.artifact_id,
        )
        operation = self.operation(dict(self.payload, editor_context=context))
        with patch.object(
            self.worker, "prepare_image_data", wraps=self.worker.prepare_image_data
        ) as prepare:
            operation.prepare(self.job)
        data = prepare.call_args.args[0]
        self.assert_manual_plan(data)
        self.assertEqual(data["input_image"].size, (64, 64))
        self.assertEqual(data["input_image"].getpixel((0, 0)), (255, 255, 255))
        self.assertIsNotNone(data.get("_editor_operation"))
        self.primary.assert_called_once()

    def test_stale_or_foreign_owned_source_never_prepares_the_primary(self):
        source = self.folder / "owned-source.png"
        Image.new("RGB", (64, 64), "white").save(source)
        snapshot = self.store.start(source, owner=self.item.owner)
        context = dict(
            session_id=snapshot.session_id,
            revision=snapshot.revision,
            source_id=snapshot.current.artifact_id,
        )
        with self.assertRaises(PermissionError):
            api_runtime._image_operation(
                dict(self.payload, editor_context=context),
                self.selected,
                self.defaults,
                owner=("ui", "foreign"),
            )
        operation = self.operation(dict(self.payload, editor_context=context))
        ticket = self.store.capture(
            snapshot.session_id,
            snapshot.revision,
            "processing",
            source_id=snapshot.current.artifact_id,
            owner=self.item.owner,
        )
        output = self.folder / "new-current.png"
        Image.new("RGB", (64, 64), "blue").save(output)
        self.store.complete(ticket, output, execution={"kind": "processing"}, owner=self.item.owner)
        with patch.object(self.worker, "prepare_image_data") as companions:
            with self.assertRaises((CancelledError, ValueError)):
                operation.prepare(self.job)
            companions.assert_not_called()
        self.primary.assert_not_called()
        self.compute.assert_not_called()

    def test_api_pending_barrier_keeps_selected_roots_settings_and_manual_request(self):
        source = self.folder / "owned-source.png"
        Image.new("RGB", (64, 64), "white").save(source)
        snapshot = self.store.start(source, owner=self.item.owner)
        context = dict(
            session_id=snapshot.session_id,
            revision=snapshot.revision,
            source_id=snapshot.current.artifact_id,
        )
        runtime_snapshots = []
        capture_runtime = api_runtime._image_runtime_snapshot

        def capture(shared):
            runtime = capture_runtime(shared)
            runtime_snapshots.append(runtime)
            return runtime

        with patch.object(api_runtime, "_image_runtime_snapshot", side_effect=capture):
            operation = self.operation(dict(self.payload, editor_context=context))
        self.assertEqual(len(runtime_snapshots), 1)
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        errors = []
        frozen_url = self.info["url"]

        def transfer(data, **options):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("Primary barrier was not released")
            return self.install_primary(data, **options)

        def prepare():
            try:
                with life.job_scope(self.item):
                    operation.prepare(self.job)
            except Exception as error:
                errors.append(error)

        self.primary.side_effect = transfer
        with patch.object(
            self.worker, "prepare_image_data", wraps=self.worker.prepare_image_data
        ) as companions:
            thread = threading.Thread(target=prepare)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                self.payload.update(negative_prompt="later request", size="512x512")
                self.payload["loras"][0]["strength"] = 1
                self.defaults.update(negative="later saved default", custom_steps=99)
                self.shared.settings.default_settings.update(image_engine="sdcpp-vulkan")
                self.shared.resolution_settings.aspect_ratios["square"] = (1024, 1024)
                self.change_live_roots_and_inventory()
            finally:
                release.set()
                thread.join(timeout=4)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assert_manual_plan(companions.call_args.args[0])
            self.assert_frozen_roots_and_inventory(companions.call_args.args[0])
            self.assertIs(
                companions.call_args.args[0]["_image_capture_paths"], runtime_snapshots[0][3]
            )
            self.assertIs(
                companions.call_args.args[0]["_image_capture_models"], runtime_snapshots[0][4]
            )
            self.assertEqual(
                companions.call_args.args[0]["input_image"].getpixel((0, 0)), (255, 255, 255)
            )
        prepared_primary = self.primary.call_args.args[0][assets.PENDING_KEY].primary
        self.assertEqual(prepared_primary.request.url, frozen_url)
        self.assertEqual(prepared_primary.request.destination, self.destination)
        self.compute.assert_not_called()

    def test_cancellation_after_primary_blocks_final_capture_and_companions(self):
        operation = self.operation()

        def cancelled_after_transfer(data, **options):
            installed = self.install_primary(data, **options)
            self.item.cancelled.set()
            return installed

        self.primary.side_effect = cancelled_after_transfer
        with (
            patch.object(
                api_runtime,
                "_image_operation",
                side_effect=AssertionError("Cancelled request recaptured"),
            ),
            patch.object(self.worker, "prepare_image_data") as companions,
        ):
            with self.assertRaises(CancelledError):
                operation.prepare(self.job)
            companions.assert_not_called()
        self.primary.assert_called_once()
        self.compute.assert_not_called()

    def test_worker_two_stage_capture_returns_one_final_plan_and_preserves_manual_intent(self):
        raw = dict(
            self.settings,
            task_type="tool_call",
            base_model_name=self.key,
            prompt=self.payload["prompt"],
            negative="No words or crop",
            seed=17,
            custom_steps=13,
            cfg=3.5,
            custom_width=128,
            custom_height=64,
            loras=[["manual.safetensors", "0.0 - manual.safetensors"]],
        )
        pending = self.worker.capture_image_data(raw)
        self.assertIsNotNone(assets.pending_image_selection(pending))
        self.assertIsNone(get_effective_image_plan(pending))
        raw.update(negative="later navigation", custom_width=512)
        self.shared.settings.default_settings.update(image_engine="sdcpp-vulkan")
        self.change_live_roots_and_inventory()
        ready = self.worker.prepare_image_data(pending, check_cancelled=self.item.check)
        self.primary.assert_called_once()
        self.assertIsNone(assets.pending_image_selection(ready))
        self.assert_manual_plan(ready)
        self.assert_frozen_roots_and_inventory(ready)
        self.assertEqual(ready["image_engine"], "native")
        self.assertEqual(
            ready["_image_capture_paths"].get_model_folders("checkpoints"),
            [self.folder / "checkpoints"],
        )
        self.compute.assert_not_called()

    def test_worker_primary_failure_or_cancel_stops_before_final_assets(self):
        pending = self.worker.capture_image_data(
            dict(self.settings, base_model_name=self.key, task_type="tool_call")
        )
        for failure in (IOError("fixture failed"), CancelledError("fixture cancelled")):
            with (
                self.subTest(failure=failure),
                patch.object(assets, "prepare_image_assets") as companions,
            ):
                self.primary.side_effect = failure
                with self.assertRaises(type(failure)):
                    self.worker.prepare_image_data(pending, check_cancelled=self.item.check)
                companions.assert_not_called()
                self.compute.assert_not_called()

    def test_worker_invalid_size_steps_or_inpaint_source_never_prepares_the_primary(self):
        for name, options, source in (
            ("small canvas", {"custom_width": 1, "custom_height": 64}, {}),
            ("large canvas", {"custom_width": 999999, "custom_height": 64}, {}),
            ("invalid steps", {"custom_steps": 0}, {}),
            (
                "inpaint source missing",
                {"inpaint_toggle": False},
                {"behavioral_variant": "inpaint"},
            ),
        ):
            with (
                self.subTest(case=name),
                patch.object(assets, "prepare_image_assets") as companions,
            ):
                self.catalogue = self.registry_for(dict(self.row, **source))
                self.primary.reset_mock()
                raw = dict(
                    self.settings,
                    base_model_name=self.key,
                    task_type="tool_call",
                    **options,
                )
                with self.assertRaises(ValueError):
                    pending = self.worker.capture_image_data(raw)
                    self.worker.prepare_image_data(pending, check_cancelled=self.item.check)
                self.primary.assert_not_called()
                companions.assert_not_called()
                self.compute.assert_not_called()

    def test_only_eligible_catalogue_selection_can_enter_pending_preparation(self):
        self.catalogue = self.registry_for(dict(self.row, gated=True))
        with self.assertRaisesRegex(ValueError, "access approval"):
            self.operation()
        self.primary.assert_not_called()
        self.catalogue = self.registry_for(self.row)
        self.assertIsNone(
            assets.capture_pending_image_selection(
                {"base_model_name": "missing-custom.safetensors"}, self.manager
            )
        )
        self.assertIsNone(
            assets.capture_pending_image_selection(
                {"base_model_name": "https://example.invalid/arbitrary.safetensors"}, self.manager
            )
        )
        self.primary.assert_not_called()


if __name__ == "__main__":
    unittest.main()
