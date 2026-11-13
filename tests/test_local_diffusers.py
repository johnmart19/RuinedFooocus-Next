"""Local bundle selection captures exact files without downloads or native imports."""

import ast
import builtins
from concurrent.futures import CancelledError
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import runpy
from unittest.mock import Mock, patch

from ruinedfooocus.features.models import local_diffusers as bundles
from ruinedfooocus.features.models import component_profiles as profiles
from ruinedfooocus.features.generation import image_preparation as assets
from ruinedfooocus.features.models.selection import checkpoint_records
import ruinedfooocus.app

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("RF_TEST_COMFY_SOURCE", ROOT / "repositories/ComfyUI"))


def bundle_fixture(root, second=False):
    index = root / "model_index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(
        json.dumps(
            {
                "_class_name": "external.code.is.not.imported",
                "text_encoder_2": ["transformers", "CLIP"] if second else None,
            }
        )
    )
    roles = [
        ("unet", bundles.DIFFUSION_FILES[-1]),
        ("vae", bundles.DIFFUSION_FILES[-1]),
        ("text_encoder", bundles.TEXT_FILES[-1]),
    ]
    if second:
        roles.append(("text_encoder_2", bundles.TEXT_FILES[-1]))
    for folder, name in roles:
        target = root / folder / name
        target.parent.mkdir()
        target.write_bytes(b"owned container-selection fixture")
    return index


def fixture_models(index):
    """The initialized metadata/file caller required by asset capture at startup."""
    return SimpleNamespace(
        get_models_by_path=lambda *args, **kwargs: {"baseModel": "Unknown"},
        get_model_base=lambda metadata: metadata["baseModel"],
        get_file=lambda *args: index,
    )


class LocalDiffusersTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_file_preference_matches_actual_pinned_directory_loader(self):
        tree = ast.parse((COMFY / "comfy/diffusers_load.py").read_text(encoding="utf-8"))
        loader = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_diffusers"
        )
        names = {
            target.id: ast.literal_eval(n.value)
            for n in loader.body
            if isinstance(n, ast.Assign)
            for target in n.targets
            if isinstance(target, ast.Name)
            and target.id in ("diffusion_model_names", "text_encoder_model_names")
        }
        self.assertEqual(tuple(names["diffusion_model_names"]), bundles.DIFFUSION_FILES)
        self.assertEqual(tuple(names["text_encoder_model_names"]), bundles.TEXT_FILES)

    def test_data_only_discovery_and_exact_native_priority_without_imports(self):
        index = bundle_fixture(self.root / "bundle", second=True)
        preferred = index.parent / "unet" / bundles.DIFFUSION_FILES[0]
        preferred.write_bytes(b"preferred local fixture")
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "comfy", "diffusers", "transformers", "requests"):
                raise AssertionError("Passive local bundle operation imported runtime/network code")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            self.assertEqual(list(bundles.discover_bundles([self.root])), [index])
            captured = bundles.capture_bundle(index)
            metadata = bundles.bundle_metadata(index)
        self.assertEqual(captured.diffusion.path, preferred.resolve())
        self.assertEqual(len(captured.encoders), 2)
        self.assertEqual(metadata["baseModel"], "Unknown")
        self.assertEqual(metadata["rf_display_name"], "bundle")

    def test_companion_changes_and_new_preference_invalidate_identity_without_retargeting_old_job(
        self,
    ):
        index = bundle_fixture(self.root)
        first = bundles.capture_bundle(index)
        preferred = self.root / "text_encoder" / bundles.TEXT_FILES[0]
        preferred.write_bytes(b"new preferred file")
        second = bundles.capture_bundle(index)
        self.assertNotEqual(first.identity, second.identity)
        first.verify()  # queued request still owns its original selected file
        self.assertNotEqual(first.encoders[0].path, second.encoders[0].path)
        first.encoders[0].path.write_bytes(b"changed original source bytes")
        with self.assertRaisesRegex(ValueError, "changed"):
            first.verify()

    def test_resolved_path_identity_and_cancellation_are_checked(self):
        index = bundle_fixture(self.root)
        captured = bundles.capture_bundle(index)
        original = Path.resolve
        item = captured.encoders[0]

        def retarget(path, *args, **kwargs):
            return (
                self.root / "another-target"
                if path == item.source
                else original(path, *args, **kwargs)
            )

        with patch.object(Path, "resolve", retarget), self.assertRaisesRegex(ValueError, "changed"):
            captured.verify()
        cancelled = Mock(side_effect=CancelledError())
        with self.assertRaises(CancelledError):
            bundles.capture_bundle(index, check_cancelled=cancelled)

    def test_missing_declared_companion_third_encoder_and_malformed_index_refuse(self):
        index = bundle_fixture(self.root, second=True)
        (self.root / "text_encoder_2" / bundles.TEXT_FILES[-1]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "text_encoder_2"):
            bundles.capture_bundle(index)
        index.write_text('{"text_encoder_3":["external","code"]}')
        with self.assertRaisesRegex(ValueError, "third"):
            bundles.capture_bundle(index)
        index.write_text("[]")
        with self.assertRaisesRegex(ValueError, "JSON object"):
            bundles.capture_bundle(index)

    def test_manifest_growth_after_initial_stat_still_has_a_bounded_read(self):
        index = bundle_fixture(self.root)
        verify = bundles.BundleFile.verify
        grown = []

        def grow_after_stat(item):
            verify(item)
            if item.role == "index" and not grown:
                grown.append(True)
                item.path.write_bytes(b"x" * (1024**2 + 2))

        with patch.object(bundles.BundleFile, "verify", grow_after_stat):
            with self.assertRaisesRegex(ValueError, "too large"):
                bundles.capture_bundle(index)

    def test_actual_submission_and_asset_capture_freeze_all_local_companions(self):
        index = bundle_fixture(self.root, second=True)
        models = SimpleNamespace(
            get_models_by_path=lambda *args, **kwargs: {"baseModel": "Unknown"},
            get_model_base=lambda value: value["baseModel"],
            get_file=lambda *args: index,
        )
        manager = SimpleNamespace(DOWNLOADABLE_FILES={})
        captured = profiles.snapshot_request({"base_model_name": str(index)}, models, {}, manager)
        self.assertIn(bundles.BUNDLE_KEY, captured)
        self.assertEqual(captured[profiles.REQUEST_KEY], {})
        request = assets.capture_image_assets(captured, manager, {})
        self.assertEqual(
            {item.name for item in request[assets.KEY].artifacts},
            {"index", "diffusion", "text_encoder", "text_encoder_2", "vae"},
        )
        self.assertFalse(assets.needs_image_preparation(request))
        for item in captured[bundles.BUNDLE_KEY].files:
            with assets.prepared_asset(request, "bundle", item.role, manager, "diffusers") as path:
                self.assertEqual(path, item.path)
        with self.assertRaisesRegex(ValueError, "fixed"):
            profiles.snapshot_request(
                {
                    "base_model_name": str(index),
                    profiles.REQUEST_KEY: {
                        "clip_l": "another.safetensors",
                    },
                },
                models,
                {},
                manager,
            )

    def test_registry_refresh_and_existing_picker_cards_expose_local_bundle_identity(self):
        state = SimpleNamespace(
            path_manager=SimpleNamespace(
                get_model_folders=lambda _: [self.root],
                model_paths={
                    key: self.root / key
                    for key in (
                        "modelfile_path",
                        "lorafile_path",
                        "embedding_path",
                        "inbox_path",
                        "cache_path",
                    )
                },
            ),
            settings=SimpleNamespace(),
            translate=lambda text, **_: text,
        )
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": state}),
            patch.object(
                ruinedfooocus.app,
                "state",
                state,
                create=True,
            ),
        ):
            models = runpy.run_path(str(ROOT / "ruinedfooocus/features/models/model_handler.py"))[
                "Models"
            ]
            owner = models(offline=True)
            owner.detect_checkpoint_base = Mock(side_effect=AssertionError("Passive import"))
            index = bundle_fixture(self.root / "local bundle")
            owner.update_all_models()
        self.assertTrue(owner.ready["checkpoints"])
        self.assertEqual(owner.get_names("checkpoints"), [str(index)])
        with patch.dict(sys.modules):
            sys.modules.pop("comfy.model_detection", None)
            records = checkpoint_records(
                owner.get_names("checkpoints"),
                lambda name: owner.get_models_by_path("checkpoints", name, fetch=True),
            )
        self.assertEqual(records[0]["name"], str(index))
        self.assertEqual(records[0]["label"], "local bundle")
        owner.detect_checkpoint_base.assert_not_called()

    def test_existing_component_editor_locks_bundle_files_and_has_no_download_action(self):
        from ruinedfooocus.features.models import ui_model_components as ui

        index = bundle_fixture(self.root, second=True)
        settings = SimpleNamespace(default_settings={}, save_profile=Mock())
        manager = SimpleNamespace(paths={}, DOWNLOADABLE_FILES={})
        state = SimpleNamespace(settings=settings, path_manager=manager)
        with (
            patch.dict(sys.modules, {"ruinedfooocus.app.state": state}),
            patch.object(
                ruinedfooocus.app,
                "state",
                state,
                create=True,
            ),
            patch.object(ui, "selected", return_value=(index, "Unknown")),
            patch.object(
                ui,
                "offline",
                return_value=False,
            ),
        ):
            editor = runpy.run_path(
                str(ROOT / "ruinedfooocus/features/models/ui_component_overrides.py")
            )
            output = ui.refresh(str(index))
            self.assertEqual(len(output[0]), 3)
            self.assertFalse(output[3]["interactive"])
            self.assertEqual(output[-1]["choices"][0], ("Text encoder", "text_encoder"))
            choice = editor["role_choices"](str(index), "text_encoder", None, None, True)
            self.assertFalse(choice["interactive"])
            with self.assertRaisesRegex(ui.gr.Error, "Built-in components cannot be replaced"):
                editor["save_choice"](
                    str(index),
                    "text_encoder",
                    "other.bin",
                    True,
                    {},
                    profiles.checkpoint_identity(index),
                )
            settings.save_profile.assert_not_called()

    def test_startup_native_preload_forwards_captured_request_but_optional_engine_does_not(self):
        source = ROOT / "ruinedfooocus/app/async_worker.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "preload_startup_model"
        )
        index = bundle_fixture(self.root)
        request = {
            "base_model_name": str(index),
            profiles.REQUEST_KEY: {},
            bundles.BUNDLE_KEY: bundles.capture_bundle(index),
        }
        for types in (["sdxl"], ["sdxl", "sdcpp"]):
            pipeline = SimpleNamespace(pipeline_type=types, load_base_model=Mock())
            scope = dict(
                settings=SimpleNamespace(
                    default_settings={"preload_image_model": True, "base_model": str(index)}
                ),
                shared=SimpleNamespace(
                    models=fixture_models(index),
                    path_manager=SimpleNamespace(DOWNLOADABLE_FILES={}),
                ),
                pipelines=SimpleNamespace(update=lambda data: pipeline),
            )
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
            with patch.object(profiles, "snapshot_request", return_value=request):
                scope["preload_startup_model"]()
            call = pipeline.load_base_model.call_args
            self.assertIsNotNone(call)
            if "sdcpp" in types:
                self.assertNotIn("prepared_data", call.kwargs)
            else:
                prepared = call.kwargs["prepared_data"]
                self.assertIs(prepared[bundles.BUNDLE_KEY], request[bundles.BUNDLE_KEY])
                self.assertEqual(
                    {item.name: item.signature for item in prepared[assets.KEY].artifacts},
                    {item.role: item.signature for item in request[bundles.BUNDLE_KEY].files},
                )

    def test_actual_native_loader_uses_captured_files_and_bundle_cache_identity(self):
        from test_model_cache import ModelCacheTests

        fixture = ModelCacheTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.addCleanup(fixture.folder.cleanup)
        index = bundle_fixture(fixture.root)
        _, owner, native_diffusion, backend, _ = fixture.base_loader(index, bare=True)

        def capture():
            bundle = bundles.capture_bundle(index)
            return assets.capture_image_assets(
                {"base_model_name": str(index), bundles.BUNDLE_KEY: bundle},
                SimpleNamespace(DOWNLOADABLE_FILES={}),
                {},
            )

        request = capture()
        bundle = request[bundles.BUNDLE_KEY]
        owner.load_base_model(str(index), prepared_data=request)
        self.assertIsNone(owner.load_error)
        self.assertEqual(native_diffusion.call_args.args[0], str(bundle.diffusion.path))
        self.assertEqual(
            backend.sd.load_clip.call_args.kwargs["ckpt_paths"], [str(bundle.encoders[0].path)]
        )
        self.assertEqual(backend.utils.load_torch_file.call_args.args[0], str(bundle.vae.path))
        self.assertIs(backend.utils.load_torch_file.call_args.kwargs["return_metadata"], True)
        owner.load_base_model(str(index), prepared_data=request)
        self.assertEqual(native_diffusion.call_count, 1)
        preferred = index.parent / "unet" / bundles.DIFFUSION_FILES[0]
        preferred.write_bytes(b"new preferred source; native loading mocked")
        changed = capture()
        owner.load_base_model(str(index), prepared_data=changed)
        self.assertEqual(native_diffusion.call_count, 2)
        self.assertEqual(native_diffusion.call_args.args[0], str(preferred.resolve()))
        self.assertNotEqual(
            request[bundles.BUNDLE_KEY].identity, changed[bundles.BUNDLE_KEY].identity
        )
        bundle.encoders[0].path.write_bytes(b"replaced after capture")
        with self.assertRaisesRegex(ValueError, "changed"):
            owner.load_base_model(str(index), prepared_data=request)

    def test_startup_refuses_encoder_changed_during_diffusion_load_before_reading_it(self):
        from test_model_cache import ModelCacheTests

        fixture = ModelCacheTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.addCleanup(fixture.folder.cleanup)
        index = bundle_fixture(fixture.root)
        _, owner, native, backend, _ = fixture.base_loader(index, bare=True)
        unet = native.return_value
        captured = bundles.capture_bundle(index)
        request = {
            "base_model_name": str(index),
            bundles.BUNDLE_KEY: captured,
            profiles.REQUEST_KEY: {},
        }

        def change_encoder(*args):
            captured.encoders[0].path.write_bytes(b"different encoder after diffusion loading")
            return unet

        native.side_effect = change_encoder
        owner.pipeline_type = ["sdxl"]
        source = ROOT / "ruinedfooocus/app/async_worker.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "preload_startup_model"
        )
        scope = dict(
            settings=SimpleNamespace(
                default_settings={"preload_image_model": True, "base_model": str(index)}
            ),
            shared=SimpleNamespace(
                models=fixture_models(index), path_manager=SimpleNamespace(DOWNLOADABLE_FILES={})
            ),
            pipelines=SimpleNamespace(update=lambda _: owner),
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        with patch.object(profiles, "snapshot_request", return_value=request):
            scope["preload_startup_model"]()
        backend.sd.load_clip.assert_not_called()
        self.assertIsNone(owner.xl_base)
        self.assertIn("changed after submission", owner.load_error)
