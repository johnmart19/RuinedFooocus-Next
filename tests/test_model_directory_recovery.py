"""Owned model folders recover without creating missing custom mounts or weights."""

import ast
import copy
from dataclasses import replace
import errno
import hashlib
import os
from pathlib import Path
import runpy
import shutil
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.maintenance_reset import recheck_folders
from ruinedfooocus.core.model_folders import MODEL_DIRECTORIES, ensure_model_root
from ruinedfooocus.core.path import PathManager
from ruinedfooocus.core.preparation import PreparationRequest, PreparationService
from ruinedfooocus.features.models.selection import checkpoint_choices, primary_checkpoint_selection

ROOT = Path(__file__).resolve().parents[1]


class ModelDirectoryRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(patch.dict(os.environ, RF_DATA_DIR=str(self.root)))
        self.manager = PathManager.__new__(PathManager)
        self.manager.paths = copy.deepcopy(PathManager.DEFAULT_PATHS)
        self.manager.DOWNLOADABLE_FILES = {
            "fixture": {
                "path": "path_checkpoints",
                "filename": "fixture.safetensors",
                "url": "https://fixture.invalid/model",
                "size": 2,
                "sha256": hashlib.sha256(b"ok").hexdigest(),
            }
        }

    def remove(self, path):
        path = Path(path).absolute()
        self.assertTrue(path.is_relative_to(self.root) and path != self.root)
        shutil.rmtree(path)

    def service(self, transport=None):
        def transfer(request, staging, progress, cancelled):
            result = staging / "complete"
            result.write_bytes(b"ok")
            return result

        service = PreparationService(
            max_workers=1,
            max_jobs=1,
            transport=transport or transfer,
            staging_root=self.root / "cache/staging",
        )
        self.addCleanup(service.close)
        return service

    def test_startup_recovers_exact_image_chat_categories_without_legacy_aliases(self):
        values = [
            v
            for value in self.manager.DEFAULT_PATHS.values()
            for v in (value if isinstance(value, list) else [value])
        ]
        expected = {Path(value).name for value in values if value.startswith("../models/")}
        expected.add("rag_embeddings")
        self.assertEqual(set(MODEL_DIRECTORIES), expected)
        self.assertEqual(len(MODEL_DIRECTORIES), 26)
        self.manager.get_model_paths()
        self.assertEqual({p.name for p in (self.root / "models").iterdir()}, expected)
        self.assertTrue(all(p.is_dir() for p in (self.root / "models").iterdir()))
        for alias in ("clip", "unet", "embedding", "t2i_adapter"):
            self.assertFalse((self.root / "models" / alias).exists())

    def test_recheck_after_live_root_deletion_recovers_empty_folders_and_keeps_originals(self):
        self.manager.model_paths = self.manager.get_model_paths()
        original = self.root / "outputs/original.png"
        original.write_bytes(b"user original")
        self.remove(self.root / "models")
        recheck_folders(self.manager)
        self.assertTrue(all((self.root / "models" / name).is_dir() for name in MODEL_DIRECTORIES))
        self.assertEqual(original.read_bytes(), b"user original")
        self.assertFalse(any((self.root / "models").rglob("*.*")))

    def test_actual_filename_scan_recovers_deleted_category_and_keeps_other_weights(self):
        self.manager.get_model_paths()
        kept = self.root / "models/vae/kept.safetensors"
        kept.write_bytes(b"existing weight")
        folder = self.root / "models/upscale_models"
        self.remove(folder)
        self.assertEqual(self.manager.get_model_filenames(folder), [])
        self.assertTrue(folder.is_dir())
        self.assertEqual(kept.read_bytes(), b"existing weight")

    def test_warm_path_property_does_not_recover_or_scan_all_model_categories(self):
        self.manager.model_paths = self.manager.get_model_paths()
        folder = self.root / "models/upscale_models"
        self.remove(folder)
        with (
            patch.object(self.manager, "ensure_model_directories") as recover,
            patch.object(Path, "rglob") as scan,
        ):
            for _ in range(3):
                self.manager.model_paths
        recover.assert_not_called()
        scan.assert_not_called()
        self.assertFalse(folder.exists())

    def test_real_models_and_refresh_callback_clear_deleted_selection(self):
        self.manager.model_paths = self.manager.get_model_paths()
        selected = self.root / "models/checkpoints/selected.safetensors"
        selected.write_bytes(b"local weight fixture")
        shared = SimpleNamespace(
            path_manager=self.manager, settings=SimpleNamespace(), translate=lambda text, **_: text
        )
        with patch.dict("sys.modules", {"ruinedfooocus.app.state": shared}):
            models_type = runpy.run_path(
                str(ROOT / "ruinedfooocus/features/models/model_handler.py")
            )["Models"]
            shared.models = models_type(offline=True)
            self.assertEqual(shared.models.get_names("checkpoints"), [selected.name])
            self.remove(self.root / "models")
            scope = dict(
                shared=shared,
                gr=SimpleNamespace(update=lambda **values: values),
                update_model_filter=lambda _: ([], []),
                adapters=SimpleNamespace(discover=lambda: ([], "")),
                primary_checkpoint_selection=primary_checkpoint_selection,
                checkpoint_choices=checkpoint_choices,
                load_styles=lambda: {},
                path_manager=self.manager,
                model_gallery="gallery",
                model_gallery_records="records",
                style_selection="styles",
                preset_gallery="presets",
                base_model="base",
                model_metadata_refresh="revision",
            )
            scope["adapters"].catalog, scope["adapters"].status = "adapters", "status"
            tree = ast.parse((ROOT / "ruinedfooocus/ui/app.py").read_text(encoding="utf-8"))
            function = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "model_refresh_clicked"
            )
            function.decorator_list = []
            exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), scope)
            result = scope["model_refresh_clicked"]([], selected.name, "", "", 0)
        self.assertIsNone(result["base"]["value"])
        self.assertEqual(shared.models.get_names("checkpoints"), [])
        self.assertTrue(all((self.root / "models" / name).is_dir() for name in MODEL_DIRECTORIES))

    def test_missing_custom_roots_are_neither_created_nor_replaced_by_defaults(self):
        custom = self.root / "unmounted/models"
        self.manager.paths["path_clip"] = [str(custom)]
        self.assertEqual(self.manager.get_model_paths()["clip_path"], custom)
        self.assertFalse(custom.parent.exists())
        self.assertFalse((self.root / "models/text_encoders").exists())

    def test_missing_custom_download_root_rejected_before_submit(self):
        custom = self.root / "unmounted/checkpoints"
        self.manager.paths["path_checkpoints"] = [str(custom)]
        with patch("ruinedfooocus.core.preparation.preparation_service.submit") as submit:
            with self.assertRaises(FileNotFoundError):
                self.manager.prepare_file("fixture")
        submit.assert_not_called()
        self.assertFalse(custom.parent.exists())

    def test_queued_custom_root_removed_before_transfer_is_not_recreated(self):
        custom = self.root / "custom"
        custom.mkdir()
        self.manager.paths["path_checkpoints"] = [str(custom)]
        request = self.manager.preparation_request("fixture")
        self.remove(custom)
        transport = Mock(side_effect=AssertionError("no transfer"))
        with self.assertRaises(FileNotFoundError):
            self.service(transport).submit(request).result(timeout=5)
        transport.assert_not_called()
        self.assertFalse(custom.exists())

    def test_captured_default_transfer_recovers_deleted_root_and_installs_once(self):
        request = self.manager.preparation_request("fixture")
        self.remove(self.root / "models")
        self.assertEqual(self.service().submit(request).result(timeout=5).read_bytes(), b"ok")

    def test_request_cannot_write_outside_its_captured_model_root(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            PreparationRequest(
                "https://fixture.invalid",
                self.root / "other/file",
                model_root=self.root / "models/checkpoints",
            )

    def test_missing_data_mount_is_not_recreated(self):
        missing = self.root / "missing/application"
        with patch.dict(os.environ, RF_DATA_DIR=str(missing)), self.assertRaises(FileNotFoundError):
            self.manager.ensure_model_directories()
        self.assertFalse(missing.parent.exists())

    def test_permission_denial_is_reported_and_no_other_category_created(self):
        mkdir = Path.mkdir

        def denied(path, *args, **kwargs):
            if path == self.root / "models":
                raise PermissionError(errno.EACCES, "denied fixture")
            return mkdir(path, *args, **kwargs)

        with patch.object(Path, "mkdir", denied), self.assertRaises(PermissionError):
            self.manager.ensure_model_directories()
        self.assertFalse((self.root / "models").exists())

    def test_owned_file_or_reparse_point_is_not_followed_or_replaced(self):
        model_root = self.root / "models"
        model_root.write_bytes(b"keep original")
        with self.assertRaises(NotADirectoryError):
            self.manager.ensure_model_directories()
        self.assertEqual(model_root.read_bytes(), b"keep original")
        model_root.unlink()
        model_root.mkdir()
        lstat = Path.lstat

        def reparse(path, *args, **kwargs):
            if path == model_root:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return lstat(path, *args, **kwargs)

        with patch.object(Path, "lstat", reparse), self.assertRaisesRegex(OSError, "reparse"):
            ensure_model_root(model_root / "checkpoints", required=True)
        self.assertFalse((model_root / "checkpoints").exists())

    def test_queued_default_reparse_replacement_is_rejected_before_transfer(self):
        request = self.manager.preparation_request("fixture")
        lstat = Path.lstat
        category = self.root / "models/checkpoints"

        def reparse(path, *args, **kwargs):
            result = lstat(path, *args, **kwargs)
            return (
                SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
                if path == category
                else result
            )

        transport = Mock(side_effect=AssertionError("no transfer"))
        with patch.object(Path, "lstat", reparse), self.assertRaisesRegex(OSError, "reparse"):
            self.service(transport).submit(request).result(timeout=5)
        transport.assert_not_called()

    def test_default_rag_capture_retains_owned_root_for_first_use_recovery(self):
        from huggingface_hub.errors import LocalEntryNotFoundError
        from ruinedfooocus.features.chat import rag_preparation as rag

        with patch.object(
            rag, "snapshot_download", side_effect=LocalEntryNotFoundError("offline fixture")
        ):
            prepared = rag.capture_rag([["text", "private document"]])
        request = replace(
            prepared.artifacts[0], expected_size=2, sha256=hashlib.sha256(b"ok").hexdigest()
        )
        self.assertEqual(request.model_root, self.root / "models/rag_embeddings")
        self.assertEqual(self.service().submit(request).result(timeout=5).read_bytes(), b"ok")


if __name__ == "__main__":
    unittest.main()
