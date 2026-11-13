"""Native BiRefNet caller keeps foreground alpha, frozen sources and native owners."""

from contextlib import contextmanager
from contextlib import nullcontext
from functools import wraps
from concurrent.futures import CancelledError
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
import ruinedfooocus.app
from PIL import Image

from ruinedfooocus.features.image_processing import native_background as native
from test_native_model_coverage import native_method
from test_native_variant_defaults import write_header
from test_image_preparation import Manager
from ruinedfooocus.features.generation.image_preparation import capture_image_assets, KEY
from test_model_cache import method
from test_image_preparation import function


class NativeBackgroundTests(unittest.TestCase):
    def test_contextual_ui_and_scalar_replay_preserve_explicit_local_choice(self):
        from ruinedfooocus.features.image_processing import ui_native_background as ui
        from ruinedfooocus.features.gallery.image_metadata import render_values
        from ruinedfooocus.integrations.openai_api import ImageRequest

        created = []
        registered = {}

        class Widget:
            def __init__(self, *args, **kwargs):
                self.value = kwargs.get("value", args[0] if args else None)
                self.options = kwargs
                self.handlers = []
                created.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def change(self, fn, **kwargs):
                self.handlers.append((fn, kwargs))

        fake = NS(
            Group=Widget,
            Dropdown=Widget,
            update=lambda **v: v,
            on=lambda triggers, fn, **kwargs: [trigger(fn, **kwargs) for trigger in triggers],
        )
        manager = Manager(self.root)
        manager.paths["path_background_removal"] = [str(self.path.parent)]
        shared = NS(
            settings=NS(default_settings=dict(rembg_native_model=self.path.name)),
            path_manager=manager,
            add_ctrl=lambda key, value: registered.update({key: value}),
        )
        selection = Widget(value="Remove background")
        kind = Widget(value="Rembg")
        refresh = Widget()
        with (
            patch.object(ui, "gr", fake),
            patch.dict("sys.modules", {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
        ):
            model = ui.create_ui(selection, kind, refresh, lambda *a: "rembg")
        self.assertIs(registered["rembg_native_model"], model)
        self.assertEqual(model.value, self.path.name)
        self.assertTrue(created[-2].options["visible"])
        update = refresh.handlers[0][0]("missing.safetensors")
        self.assertEqual(update["value"], "missing.safetensors")
        self.assertEqual(
            render_values(dict(rembg_native_model=self.path.name))["rembg_native_model"],
            self.path.name,
        )
        self.assertEqual(
            ImageRequest(rembg_native_model=self.path.name).rembg_native_model, self.path.name
        )
        shared.settings.default_settings = {}
        with (
            patch.object(ui, "gr", fake),
            patch.dict("sys.modules", {"ruinedfooocus.app.state": shared}),
            patch.object(ruinedfooocus.app, "state", shared, create=True),
        ):
            self.assertEqual(
                ui.create_ui(selection, kind, None, lambda *a: "rembg").value, ui.DEFAULT_MODEL
            )
            shared.settings.default_settings = {"rembg_native_model": ""}
            self.assertEqual(ui.create_ui(selection, kind, None, lambda *a: "rembg").value, "")

    def test_catalogue_model_is_prepared_before_compute_and_missing_offline_is_rejected(self):
        from ruinedfooocus.features.generation.image_preparation import (
            needs_image_preparation,
            prepare_image_assets,
        )

        manager = Manager(self.root)
        manager.add("background", "background_removal", "birefnet.safetensors")
        data = capture_image_assets(
            dict(rembg_native_model="birefnet.safetensors", input_image=Image.new("RGB", (2, 2))),
            manager,
            {"type": "rembg"},
            offline=True,
        )
        self.assertTrue(needs_image_preparation(data))
        item = data[KEY].artifacts[0]
        self.assertEqual(item.role, "background-removal")
        self.assertEqual(item.request.sha256, manager.DOWNLOADABLE_FILES["background"]["sha256"])
        with self.assertRaisesRegex(ValueError, "Offline mode.*background-removal"):
            prepare_image_assets(data, check_cancelled=Mock())

    def test_actual_api_transports_local_choice_and_uses_shared_native_image_service(self):
        import ruinedfooocus.app.api_runtime as api
        from test_api_image_edits import queued_data

        original = api._image_operation
        groups = []

        def capture(*args, **kwargs):
            operation = original(*args, **kwargs)
            groups.append(operation.resource_group)
            return operation

        with patch.object(api, "_image_operation", side_effect=capture):
            data = queued_data(
                "SDXL 1.0",
                controlnet={"type": "rembg"},
                payload_options=dict(rembg_native_model=str(self.path)),
            )
        self.assertEqual(data["rembg_native_model"], str(self.path))
        self.assertEqual(groups, ["image"])
        self.assertEqual(data[KEY].artifacts[0].role, "background-removal")

    def test_actual_worker_stage_retains_native_interrupt_and_compute_owner(self):
        groups = []
        job = NS(state={})

        @contextmanager
        def service(group, *args, **kwargs):
            groups.append(group)
            yield

        scope = dict(
            wraps=wraps,
            current_job=lambda: job,
            control_tools=NS(get_settings=lambda data: {"type": "rembg"}),
            service_scope=service,
            shared=NS(state={}),
            cleanup_interrupt=Mock(),
        )
        decorate = function("image_stage", scope)
        with patch(
            "ruinedfooocus.features.prompting.local_model.prompt_preparation_scope",
            return_value=nullcontext(),
        ):
            result = decorate(lambda data: job.state["native_active"])(
                dict(_job=job, rembg_native_model=self.path.name)
            )
        self.assertTrue(result)
        self.assertEqual(groups, ["image"])
        self.assertNotIn("native_active", job.state)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        folder = self.root / "background_removal"
        folder.mkdir()
        self.path = write_header(
            folder / "model.safetensors",
            {"bb.layers.1.blocks.0.attn.relative_position_index": [49, 49]},
        )
        self.held = []

        @contextmanager
        def lease(*args, **kwargs):
            self.held.append(self.path)
            try:
                yield self.path
            finally:
                self.held.pop()

        self.model = NS(
            patcher=object(),
            load_device=torch.device("cpu"),
            encode_image=Mock(
                side_effect=lambda pixels: (
                    torch.full(pixels.shape[:3], 0.5)
                    if self.held
                    else self.fail("native read escaped owned source")
                )
            ),
        )
        self.load = Mock(return_value=self.model)
        node = native_method(
            "comfy_extras/nodes_bg_removal.py",
            "RemoveBackground",
            "execute",
            dict(IO=NS(NodeOutput=lambda *values: values)),
        )
        self.modules = {
            "comfy.bg_removal_model": NS(load=self.load),
            "comfy_extras.nodes_bg_removal": NS(RemoveBackground=node),
            "comfy.model_management": NS(unload_model_and_clones=Mock()),
        }
        self.enterContext(patch.dict("sys.modules", self.modules))
        self.enterContext(
            patch("ruinedfooocus.features.generation.image_preparation.prepared_asset", lease)
        )

    def test_actual_native_node_preserves_rgb_combines_existing_alpha_and_caches_signature(self):
        owner = NS()
        image = Image.new("RGBA", (12, 8), (200, 100, 20, 128))
        data = dict(rembg_native_model=self.path.name, rembg_force=True)
        result = native.remove_background(owner, image, data, object(), Mock())
        self.assertEqual(result.size, image.size)
        self.assertEqual(result.getpixel((0, 0)), (200, 100, 20, 64))
        native.remove_background(owner, image, data, object(), Mock())
        self.load.assert_called_once()
        self.assertFalse(self.held)
        self.assertEqual(data["native_background_source"]["native"], "BiRefNet")
        self.assertEqual(data["native_background_source"]["device"], "cpu")
        self.assertNotIn(str(self.root), str(data["native_background_source"]))
        self.path.write_bytes(self.path.read_bytes() + b"new")
        native.remove_background(owner, image, data, object(), Mock())
        self.assertEqual(self.load.call_count, 2)
        self.modules["comfy.model_management"].unload_model_and_clones.assert_called_once_with(
            self.model.patcher
        )

    def test_transparent_input_skips_native_loading_until_explicit_force(self):
        image = Image.new("RGBA", (2, 2), (5, 9, 12, 40))
        result = native.remove_background(
            NS(), image, dict(rembg_native_model=self.path.name), object(), Mock()
        )
        self.assertEqual(result.getpixel((0, 0)), image.getpixel((0, 0)))
        self.load.assert_not_called()

    def test_cancellation_releases_lease_and_does_not_publish_alpha(self):
        check = Mock(side_effect=[None, None, CancelledError("cancel")])
        data = dict(rembg_native_model=self.path.name, rembg_force=True)
        with self.assertRaises(CancelledError):
            native.remove_background(NS(), Image.new("RGB", (2, 2)), data, object(), check)
        self.assertFalse(self.held)
        self.assertNotIn("native_background_source", data)

    def test_actual_capture_freezes_installed_local_model_without_download(self):
        manager = Manager(self.root)
        data = capture_image_assets(
            dict(rembg_native_model=self.path.name, input_image=Image.new("RGB", (2, 2))),
            manager,
            {"type": "rembg"},
        )
        self.assertEqual(
            [(x.role, x.name) for x in data[KEY].artifacts],
            [("background-removal", self.path.name)],
        )
        independent = capture_image_assets(
            dict(
                rembg_native_model=self.path.name,
                input_image=Image.new("RGB", (2, 2)),
                image_engine="sdcpp-vulkan",
            ),
            manager,
            {"type": "rembg"},
        )
        self.assertEqual(independent[KEY].artifacts[0].role, "background-removal")
        with self.assertRaisesRegex(ValueError, "native BiRefNet"):
            other = write_header(
                self.root / "background_removal" / "other.safetensors", {"wrong.weight": [1]}
            )
            capture_image_assets(
                dict(rembg_native_model=other.name, input_image=Image.new("RGB", (2, 2))),
                manager,
                {"type": "rembg"},
            )

    def test_actual_rembg_pipeline_routes_explicit_native_choice_and_closes_onnx(self):
        worker = NS(check_interrupt=Mock(), add_result=Mock())
        run = method(
            "ruinedfooocus/features/image_processing/rembg_pipeline.py",
            "process",
            dict(worker=worker, get_settings=lambda data: {"force": True}),
        )
        owner = NS(
            remove_background=Mock(side_effect=AssertionError("ONNX must not run")),
            _rembg_session=NS(close=Mock()),
        )
        close = owner._rembg_session.close
        data = dict(
            rembg_native_model=self.path.name, input_image=Image.new("RGB", (2, 2)), task_id="owned"
        )
        with patch.dict("sys.modules", {"ruinedfooocus.app.state": NS(path_manager=object())}):
            images = run(owner, data)
        self.assertEqual(images[0].mode, "RGBA")
        close.assert_called_once()
        self.assertIsNone(owner._rembg_session)
        self.assertTrue(worker.check_interrupt.called)

    def test_discovery_and_saved_missing_selection_are_honest(self):
        from ruinedfooocus.features.image_processing.ui_native_background import choices

        manager = Manager(self.root)
        manager.paths["path_background_removal"] = [str(self.path.parent)]
        self.assertEqual(list(native.local_models(manager)), [self.path.name])
        options = choices(manager, "missing.safetensors")
        self.assertIn("missing.safetensors", [value for _, value in options])
        self.assertEqual(options[0][1], "")

    def test_native_memory_failure_recovers_on_cpu_without_hiding_invalid_models(self):
        worker = NS(check_interrupt=Mock(), add_result=Mock())
        cpu = Mock(return_value=NS(runtime_status=""))
        run = method(
            "ruinedfooocus/features/image_processing/rembg_pipeline.py",
            "process",
            dict(
                worker=worker,
                get_settings=lambda data: {"force": True},
                new_session=cpu,
                session_request_identity=lambda cpu=False: ("CPU" if cpu else "Auto",),
            ),
        )
        image = Image.new("RGBA", (2, 2), (8, 16, 24, 128))
        owner = NS(
            _rembg_session=None,
            _rembg_identity=("Auto",),
            remove_background=Mock(return_value=image),
        )
        data = dict(
            rembg_native_model=self.path.name,
            input_image=image,
            task_id="owned",
            native_background_source={"device": "cuda:0"},
        )
        with (
            patch.dict("sys.modules", {"ruinedfooocus.app.state": NS(path_manager=object())}),
            patch.object(native, "remove_background", side_effect=torch.OutOfMemoryError()),
            patch.object(native, "clear_model") as clear,
        ):
            self.assertEqual(run(owner, data), [image])
        cpu.assert_called_once_with(cpu=True)
        self.assertEqual(owner._rembg_identity, ("CPU",))
        clear.assert_called_once_with(owner)
        self.assertIs(owner.remove_background.call_args.args[0], image)
        self.assertTrue(owner.remove_background.call_args.kwargs["force"])
        self.assertIn("CPU recovery", data["rembg_runtime_status"])
        self.assertNotIn("native_background_source", data)
        automatic = Mock(return_value=NS())
        onnx_remove = method(
            "ruinedfooocus/features/image_processing/rembg_pipeline.py",
            "remove_background",
            dict(
                PIL=NS(Image=NS(Image=object)),
                Any=object,
                args=object(),
                force_cpu_for_auxiliary=lambda args: False,
                session_request_identity=lambda cpu=False: ("CPU" if cpu else "Auto",),
                new_session=automatic,
                rembg=NS(remove=lambda image, **kwargs: image),
            ),
        )
        self.assertIs(onnx_remove(owner, image, force=True), image)
        automatic.assert_called_once_with(cpu=False)
        self.assertEqual(owner._rembg_identity, ("Auto",))
        cpu.reset_mock()
        with (
            patch.dict("sys.modules", {"ruinedfooocus.app.state": NS(path_manager=object())}),
            patch.object(native, "remove_background", side_effect=ValueError("invalid weights")),
            self.assertRaisesRegex(ValueError, "invalid weights"),
        ):
            run(owner, data)
        cpu.assert_not_called()


if __name__ == "__main__":
    unittest.main()
