"""Native first-use files share preparation without holding compute admission."""

import ast
from concurrent.futures import CancelledError, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.preparation import PreparationRequest, PreparationService
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.app.resource_lifecycle import JobRegistry, FairAdmission, job_scope

ROOT = Path(__file__).resolve().parents[1]


def function(name, scope):
    source = ROOT / "ruinedfooocus/app/async_worker.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope[name]


class Manager:
    def __init__(self, root):
        self.root = root
        self.DOWNLOADABLE_FILES = {}
        self.paths = {"path_" + category: [str(root / category)] for category in ("clip", "vae")}
        self.get_abspath = lambda value: Path(value).resolve()
        self.model_paths = {}

    def add(self, key, category, filename):
        self.DOWNLOADABLE_FILES[key] = {
            "path": "path_" + category,
            "filename": filename,
            "url": "https://example.invalid/" + filename,
            "size": 4,
            "sha256": hashlib.sha256(b"good").hexdigest(),
        }

    def get_model_folders(self, category):
        return [self.root / category]

    def preparation_request(self, key, *, destination=None):
        entry = self.DOWNLOADABLE_FILES[key]
        return PreparationRequest(
            entry["url"],
            destination or self.root / entry["path"].removeprefix("path_") / entry["filename"],
            entry["size"],
            entry["sha256"],
        )


class NativeImagePreparationTests(unittest.TestCase):
    def setUp(self):
        online = patch.dict(os.environ, {"RF_OFFLINE": "0", "HF_HUB_OFFLINE": "0"})
        online.start()
        self.addCleanup(online.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.manager = Manager(self.root)
        self.manager.add("cn_canny", "controlnet", assets.CONTROL_MODELS["canny"])
        self.manager.add("cn_depth", "controlnet", assets.CONTROL_MODELS["depth"])
        self.manager.add("RealESRGAN_x2plus.pth", "upscalers", "RealESRGAN_x2plus.pth")
        self.manager.add("da3", "geometry_estimation", "depth_anything_3_small.safetensors")

    def service(self, transport):
        service = PreparationService(transport=transport, staging_root=self.root / "staging")
        self.addCleanup(service._executor.shutdown, wait=True)
        return service

    def local(self, category, filename, content=b"good"):
        path = self.root / category / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def capture(self, control, **data):
        return assets.capture_image_assets(
            dict(input_image="owned input", **data), self.manager, control
        )

    def test_installed_fast_path_is_local_and_captured_roots_controls_do_not_drift(self):
        path = self.local("controlnet", assets.CONTROL_MODELS["canny"])
        controls = {"type": "canny", "strength": 0.4}
        data = self.capture(controls)
        controls["strength"] = 9
        self.manager.root = self.root / "changed"
        self.assertFalse(assets.needs_image_preparation(data))
        self.assertEqual(data["controlnet"]["strength"], 0.4)
        service = self.service(Mock(side_effect=AssertionError("no transfer")))
        with assets.prepared_asset(
            data, "controlnet", "canny", self.manager, "controlnet", service=service
        ) as actual:
            self.assertEqual(actual, path.resolve())

    def test_actual_worker_admits_ready_chat_while_first_use_controlnet_waits(self):
        self.ready_chat_during(self.capture({"type": "canny"}))

    def ready_chat_during(self, capture):
        entered, release = threading.Event(), threading.Event()
        admission = FairAdmission()

        def transfer(request, staging, progress, cancelled):
            self.assertEqual(admission.held, {})
            progress(2, 4, 0)
            entered.set()
            self.assertTrue(release.wait(3))
            staged = staging / "payload.part"
            staged.write_bytes(b"good")
            return staged

        service = self.service(transfer)
        registry = JobRegistry()
        ready = threading.Event()
        scope = dict(
            registry=registry,
            shared=SimpleNamespace(),
            settings=SimpleNamespace(),
            api_jobs={},
            job_scope=job_scope,
            start_worker=Mock(side_effect=ready.set),
            capture_image_data=lambda data: dict(capture, **data),
        )
        scope["prepare_image_data"] = lambda data, **kwargs: assets.prepare_image_assets(
            data, service=service, **kwargs
        )
        scope["add_result"] = lambda task, flag, value: registry.publish(task, flag, value)
        scope["describe_error"] = str
        function("validate_editor_data", scope)
        function("discard_editor_result", scope)
        submit = function("add_task", scope)
        from ruinedfooocus.app.preparation_admission import PreparationAdmission

        queue = PreparationAdmission()
        self.addCleanup(queue.executor.shutdown, wait=True)
        with patch("ruinedfooocus.app.preparation_admission.preparation_admission", queue):
            first = submit({"task_type": "process"})
            self.assertTrue(entered.wait(2))
            flag, value = registry.result(first)
            self.assertEqual(flag, "preview")
            self.assertIn("2 / 4 bytes", value[1])
            second = submit({"task_type": "llama", "unload": True})
            item, data = registry.take(0)
            self.assertEqual(item.task_id, second)
            with admission.lease(item):
                registry.publish(second, "results", "ready Chat B replied")
            self.assertEqual(registry.result(second), ("results", "ready Chat B replied"))
            registry.finish(item)
            ready.clear()
            release.set()
            self.assertTrue(ready.wait(2))
            item, data = registry.take(0)
            self.assertEqual(item.task_id, first)
            self.assertTrue(all(item.signature is not None for item in data[assets.KEY].artifacts))
            registry.finish(item)

    def test_upscaler_depth_and_component_preparation_also_leave_ready_compute_free(self):
        self.ready_chat_during(
            self.capture({"type": "upscale", "upscaler": "RealESRGAN_x2plus.pth"})
        )
        self.local("controlnet", assets.CONTROL_MODELS["depth"])
        self.ready_chat_during(
            self.capture({"type": "depth", "depth_preprocess": "auto", "depth_model": "small"})
        )
        from ruinedfooocus.features.models.resource_defaults import RESOURCE_DEFAULTS

        for role in ("clip_l", "clip_g", "vae_sdxl"):
            category = "clip" if role.startswith("clip_") else "vae"
            self.manager.add(role, category, RESOURCE_DEFAULTS[role])
        for missing in ("clip_l", "vae_sdxl"):
            with self.subTest(missing=missing):
                for role in ("clip_l", "clip_g", "vae_sdxl"):
                    category = "clip" if role.startswith("clip_") else "vae"
                    if role != missing:
                        self.local(category, RESOURCE_DEFAULTS[role])
                    else:
                        (self.root / category / RESOURCE_DEFAULTS[role]).unlink(missing_ok=True)
                request = assets.capture_image_assets(
                    {"base_model_name": "fixture"}, self.manager, {}, models=self.models({}, "SDXL")
                )
                self.ready_chat_during(request)

    def test_upscaler_and_automatic_upscaler_capture_same_exact_asset(self):
        explicit = self.capture({"type": "upscale", "upscaler": "RealESRGAN_x2plus.pth"})
        automatic = self.capture({}, automatic_upscale="2× General (RealESRGAN)")
        self.assertEqual(
            explicit[assets.KEY].artifacts[0].request, automatic[assets.KEY].artifacts[0].request
        )
        self.assertEqual(automatic[assets.KEY].automatic_upscale_scale, 2)
        self.assertTrue(assets.needs_image_preparation(explicit))

    def test_text_only_chat_does_not_capture_or_prepare_irrelevant_image_companions(self):
        registry = JobRegistry()
        scope = dict(
            registry=registry,
            shared=SimpleNamespace(path_manager=self.manager),
            settings=SimpleNamespace(default_settings={}),
            api_jobs={},
            start_worker=Mock(),
            capture_image_data=Mock(side_effect=AssertionError("text-only image capture")),
        )
        submit = function("add_task", scope)
        with (
            patch(
                "ruinedfooocus.features.chat.model_preparation.capture_chat_request",
                side_effect=lambda data, *args: dict(data),
            ),
            patch("ruinedfooocus.features.chat.llama_models.local_models", return_value={}),
        ):
            task = submit(
                {
                    "task_type": "llama",
                    "enable_image_tools": False,
                    "image_settings": {"base_model_name": "missing-companions"},
                }
            )
        item, data = registry.take(0)
        self.assertEqual(item.task_id, task)
        self.assertNotIn("image_settings", data)
        registry.finish(item)

    def test_offline_unknown_custom_and_disappeared_assets_do_not_load_or_download(self):
        data = assets.capture_image_assets(
            {"input_image": "source"}, self.manager, {"type": "canny"}, offline=True
        )
        service = self.service(Mock(side_effect=AssertionError("offline transfer")))
        with self.assertRaisesRegex(ValueError, "Offline"):
            assets.prepare_image_assets(data, check_cancelled=lambda: None, service=service)
        with self.assertRaisesRegex(FileNotFoundError, "not installed"):
            self.capture({"type": "upscale", "upscaler": "custom.pth"})
        path = self.local("controlnet", assets.CONTROL_MODELS["canny"])
        data = self.capture({"type": "canny"})
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            with assets.prepared_asset(
                data, "controlnet", "canny", self.manager, "controlnet", service=service
            ):
                self.fail("changed file reached loader")

    def test_two_subscribers_share_publication_and_cancel_only_their_own_request(self):
        entered, release, stop = threading.Event(), threading.Event(), threading.Event()
        calls = []

        def transfer(request, staging, progress, cancelled):
            calls.append(request)
            entered.set()
            self.assertTrue(release.wait(3))
            staged = staging / "asset.part"
            staged.write_bytes(b"good")
            return staged

        service = self.service(transfer)
        request = self.capture({"type": "upscale", "upscaler": "RealESRGAN_x2plus.pth"})

        def cancelled():
            if stop.is_set():
                raise CancelledError()

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                assets.prepare_image_assets, request, check_cancelled=cancelled, service=service
            )
            self.assertTrue(entered.wait(2))
            subscribed = threading.Event()
            original = service.submit

            def submit(*args):
                handle = original(*args)
                subscribed.set()
                return handle

            with patch.object(service, "submit", side_effect=submit):
                second = executor.submit(
                    assets.prepare_image_assets,
                    request,
                    check_cancelled=lambda: None,
                    service=service,
                )
                self.assertTrue(subscribed.wait(2))
                stop.set()
                with self.assertRaises(CancelledError):
                    first.result(2)
                release.set()
                prepared = second.result(2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            prepared[assets.KEY].artifacts[0].request.destination.read_bytes(), b"good"
        )

    def test_failed_transfer_keeps_final_absent_and_never_supplies_a_loader_path(self):
        data = self.capture({"type": "canny"})
        service = self.service(Mock(side_effect=IOError("network failed")))
        with self.assertRaisesRegex(IOError, "network failed"):
            assets.prepare_image_assets(data, check_cancelled=lambda: None, service=service)
        self.assertFalse(data[assets.KEY].artifacts[0].request.destination.exists())
        with self.assertRaisesRegex(FileNotFoundError, "Prepare"):
            with assets.prepared_asset(
                data, "controlnet", "canny", self.manager, "controlnet", service=service
            ):
                self.fail("unprepared file reached native loader")

    def test_transport_timeout_is_terminal_instead_of_an_endless_poll(self):
        data = self.capture({"type": "canny"})
        service = self.service(Mock(side_effect=TimeoutError("bounded transport timeout")))
        with self.assertRaisesRegex(TimeoutError, "bounded transport"):
            assets.prepare_image_assets(data, check_cancelled=lambda: None, service=service)

    def models(self, tensors, family):
        checkpoint = self.root / "fixture.safetensors"
        header = json.dumps(tensors).encode()
        checkpoint.write_bytes(len(header).to_bytes(8, "little") + header)
        return SimpleNamespace(
            get_file=lambda *args: checkpoint,
            get_models_by_path=lambda *args, **kwargs: {"baseModel": family},
            get_model_base=lambda value: value["baseModel"],
        )

    def test_bundled_and_pixel_space_models_do_not_plan_external_components(self):
        for family, tensors in (
            ("SDXL", {"conditioner.embedders.0.x": {}, "first_stage_model.x": {}}),
            ("HiDream O1", {}),
        ):
            data = assets.capture_image_assets(
                {"base_model_name": "fixture"},
                self.manager,
                {},
                models=self.models(tensors, family),
            )
            self.assertEqual(data[assets.KEY].artifacts, ())

    def test_sft_external_companions_are_prepared_before_native_loading(self):
        from ruinedfooocus.features.models.resource_defaults import RESOURCE_DEFAULTS

        models = self.models({}, "Flux")
        checkpoint = models.get_file()
        alias = checkpoint.with_suffix(".SFT")
        checkpoint.rename(alias)
        models.get_file = lambda *args: alias
        for role in ("clip_l", "clip_t5", "vae_flux"):
            self.manager.add(
                role, "clip" if role.startswith("clip_") else "vae", RESOURCE_DEFAULTS[role]
            )
        data = assets.capture_image_assets(
            {"base_model_name": alias.name}, self.manager, {}, models=models
        )
        self.assertEqual(
            {item.name for item in data[assets.KEY].artifacts}, {"clip_l", "clip_t5", "vae_flux"}
        )
        self.assertTrue(assets.needs_image_preparation(data))

    def test_flux_multiencoder_roles_and_missing_custom_override_are_explicit(self):
        from ruinedfooocus.features.models.resource_defaults import RESOURCE_DEFAULTS

        for role in ("clip_l", "clip_t5", "vae_flux"):
            self.manager.add(
                role, "clip" if role.startswith("clip_") else "vae", RESOURCE_DEFAULTS[role]
            )
        models = self.models({}, "Flux")
        from ruinedfooocus.features.models.component_profiles import component_inventory

        state = component_inventory(models.get_file(), "Flux", {}, self.manager)
        self.assertTrue(all(row["download_available"] for row in state["rows"]), repr(state))
        data = assets.capture_image_assets(
            {"base_model_name": "fixture"}, self.manager, {}, models=models
        )
        self.assertEqual(
            {item.name for item in data[assets.KEY].artifacts}, {"clip_l", "clip_t5", "vae_flux"}
        )
        with self.assertRaisesRegex(FileNotFoundError, "reset this role"):
            assets.capture_image_assets(
                {
                    "base_model_name": "fixture",
                    "image_component_overrides": {"clip_l": "missing-custom.safetensors"},
                },
                self.manager,
                {},
                models=models,
            )

    def test_depth_supplied_maps_skip_da3_and_explicit_small_never_queries_devices(self):
        supplied = self.capture({"type": "depth", "depth_preprocess": "none"})
        self.assertEqual([item.role for item in supplied[assets.KEY].artifacts], ["controlnet"])
        explicit = self.capture(
            {"type": "depth", "depth_preprocess": "auto", "depth_model": "small"}
        )
        self.assertEqual(
            [item.role for item in explicit[assets.KEY].artifacts],
            ["controlnet", "depth-preprocessor"],
        )
        self.assertEqual(explicit[assets.KEY].depth_name, "depth_anything_3_small.safetensors")

    def test_imported_metadata_captures_effective_control_and_disables_unrecorded_upscale(self):
        prompt = json.dumps(
            {
                "Prompt": "saved",
                "controlnet": {"type": "depth", "depth_preprocess": "auto", "depth_model": "small"},
            }
        )
        data = self.capture(
            {"type": "canny"}, prompt=prompt, automatic_upscale="2× General (RealESRGAN)"
        )
        self.assertEqual(
            [item.role for item in data[assets.KEY].artifacts], ["controlnet", "depth-preprocessor"]
        )
        self.assertEqual(data["controlnet"]["type"], "depth")
        self.assertIsNone(data[assets.KEY].automatic_upscale_name)


if __name__ == "__main__":
    unittest.main()
