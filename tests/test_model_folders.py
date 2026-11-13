"""Canonical model folders resolve existing weights without moving or downloading."""

import ast
import json
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.backends.comfy.model_paths import (
    NATIVE_IMAGE_FOLDERS,
    register_image_model_paths,
)
from ruinedfooocus.core.path import PathManager
from ruinedfooocus.core.model_folders import model_folder_lines
from ruinedfooocus.core.settings_validation import validate_settings
from ruinedfooocus.features.models.image_components import inventory

ROOT = Path(__file__).resolve().parents[1]


class ModelFolderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        environment = patch.dict(os.environ, RF_DATA_DIR=str(self.root))
        environment.start()
        self.addCleanup(environment.stop)
        self.paths = PathManager.__new__(PathManager)
        self.paths.paths = PathManager.DEFAULT_PATHS.copy()
        self.paths.DOWNLOADABLE_FILES = {}
        self.paths.download_file = Mock(side_effect=AssertionError("unexpected download"))

    def weight(self, name, value=b"unchanged fixture"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return path

    def test_default_roots_include_canonical_image_models_and_legacy_unet(self):
        paths = self.paths.get_model_paths()
        self.assertEqual(
            paths["modelfile_path"],
            [self.root / "models" / name for name in ("checkpoints", "diffusion_models", "unet")],
        )
        self.assertTrue(all(path.is_dir() for path in paths["modelfile_path"][:2]))
        self.assertFalse((self.root / "models/unet").exists())
        self.assertEqual(paths["clip_path"], self.root / "models/text_encoders")
        self.assertFalse((self.root / "models/embeddings").exists())

    def test_old_clip_setting_keeps_precedence_and_discovers_canonical_encoder(self):
        self.paths.paths["path_clip"] = "../models/clip/"
        self.assertEqual(
            self.paths.get_model_folders("text_encoders"),
            [self.root / "models/clip", self.root / "models/text_encoders"],
        )
        self.assertEqual(self.paths.paths["path_clip"], "../models/clip/")
        self.assertFalse((self.root / "models").exists())

    def test_loading_existing_profile_retains_legacy_and_custom_path_values(self):
        self.paths.settings_path = self.root / "settings/paths.json"
        self.paths.settings_path.parent.mkdir()
        existing = {
            "path_clip": "../models/clip/",
            "path_checkpoints": [str(self.root / "shared")],
            "custom_path_option": "keep this value",
        }
        self.paths.settings_path.write_text(json.dumps(existing), encoding="utf-8")
        loaded = self.paths.load_paths()
        for key, value in existing.items():
            self.assertEqual(loaded[key], value)
            self.assertEqual(json.loads(self.paths.settings_path.read_text())[key], value)
        self.assertEqual(
            loaded["path_diffusion_models"], PathManager.DEFAULT_PATHS["path_diffusion_models"]
        )

    def test_custom_roots_are_authoritative_and_keep_order(self):
        first, second = self.root / "shared", self.root / "other"
        self.paths.paths["path_clip"] = [str(first), str(second), str(first)]
        self.assertEqual(self.paths.get_model_folders("clip"), [first, second])
        self.paths.paths["path_diffusion_models"] = []
        self.assertEqual(self.paths.get_model_folders("diffusion_models"), [])
        self.assertFalse(first.exists())

    def test_encoder_settings_roundtrip_preserves_single_and_multiple_roots(self):
        from ruinedfooocus.core.settings import prospective_profile

        for original in ("../models/clip/", [str(self.root / "first"), str(self.root / "second")]):
            with self.subTest(original=original):
                self.paths.paths["path_clip"] = original
                values = validate_settings({"path_clip": model_folder_lines(original)})
                settings, paths = prospective_profile(
                    {"prompt": "keep"}, self.paths.paths, values, set(PathManager.DEFAULT_PATHS)
                )
                expected = original if isinstance(original, list) else [original]
                self.assertEqual(paths["path_clip"], expected)
                self.assertEqual(self.paths.paths["path_clip"], original)
                self.assertEqual(settings, {"prompt": "keep"})
                self.assertEqual(
                    self.paths.get_model_paths()["clip_path"],
                    self.paths.get_model_download_folder("clip"),
                )

    def test_encoder_secondary_lookup_and_primary_download_remain_distinct(self):
        first, second = self.root / "first", self.root / "second"
        self.paths.paths["path_clip"] = [str(first), str(second)]
        installed = self.weight("second/encoder.safetensors")
        self.assertEqual(self.paths.get_folder_file_path("clip", installed.name), installed)
        primary = self.paths.get_model_paths()["clip_path"]
        self.assertIsInstance(primary, Path)
        self.assertEqual(primary, first)
        self.paths.DOWNLOADABLE_FILES = {
            "clip/missing.safetensors": {"path": "path_clip", "filename": "missing.safetensors"}
        }
        self.paths.download_file = Mock()
        self.assertEqual(
            self.paths.get_folder_file_path("clip", "missing.safetensors"),
            first / "missing.safetensors",
        )
        self.paths.download_file.assert_called_once_with(
            "clip/missing.safetensors", progress=None, destination=primary / "missing.safetensors"
        )

    def test_native_alias_names_resolve_existing_legacy_folders(self):
        for category, expected in (
            ("embedding", ("embeddings", "embedding")),
            ("unet", ("diffusion_models", "unet")),
            ("t2i_adapter", ("controlnet", "t2i_adapter")),
        ):
            with self.subTest(category=category):
                self.assertEqual(
                    self.paths.get_model_folders(category),
                    [self.root / "models" / name for name in expected],
                )

    def test_catalog_alias_and_canonical_name_reuse_legacy_encoder_in_place(self):
        legacy = self.weight("models/clip/encoder.safetensors")
        before = legacy.stat()
        self.paths.DOWNLOADABLE_FILES = {
            "clip_encoder": {"path": "path_clip", "filename": legacy.name}
        }
        for key in (
            "clip_encoder",
            "clip/encoder.safetensors",
            "text_encoders/encoder.safetensors",
        ):
            self.assertEqual(self.paths.get_file_path(key), legacy)
        self.paths.download_file.assert_not_called()
        self.assertEqual(legacy.read_bytes(), b"unchanged fixture")
        self.assertEqual(legacy.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertFalse((self.root / "models/text_encoders").exists())

    def test_current_and_legacy_duplicate_encoder_prefers_configured_primary(self):
        canonical = self.weight("models/text_encoders/encoder.safetensors", b"canonical")
        legacy = self.weight("models/clip/encoder.safetensors", b"legacy")
        self.assertEqual(self.paths.get_folder_file_path("clip", legacy.name), canonical)
        self.paths.paths["path_clip"] = "../models/clip/"
        self.assertEqual(self.paths.get_folder_file_path("clip", legacy.name), legacy)
        self.assertEqual(canonical.read_bytes(), b"canonical")
        self.assertEqual(legacy.read_bytes(), b"legacy")

    def test_folder_list_combines_aliases_without_duplicate_filenames(self):
        self.weight("models/text_encoders/shared.safetensors")
        self.weight("models/clip/shared.safetensors")
        self.weight("models/clip/old.safetensors")
        self.assertEqual(
            self.paths.get_folder_list("text_encoders"), ["old.safetensors", "shared.safetensors"]
        )

    def test_component_inventory_finds_canonical_weights_with_old_clip_setting(self):
        self.paths.paths["path_clip"] = "../models/clip/"
        encoder = self.weight("models/text_encoders/qwen3vl_4b_bf16.safetensors")
        self.weight("models/vae/mage_flow_vae_bf16.safetensors")
        header = json.dumps({"model.weight": {"shape": [1]}}).encode()
        checkpoint = self.weight(
            "models/diffusion_models/image.safetensors", len(header).to_bytes(8, "little") + header
        )
        state = inventory(checkpoint, "MageFlow", {}, self.paths)
        self.assertEqual([row["status"] for row in state["rows"]], ["Installed", "Installed"])
        self.assertEqual(state["rows"][0]["location"], str(encoder))
        self.paths.download_file.assert_not_called()

    def test_diffusion_roots_use_settings_list_contract_and_can_be_disabled(self):
        self.assertEqual(
            validate_settings({"path_diffusion_models": " one \n\n two "})["path_diffusion_models"],
            ["one", "two"],
        )
        self.assertEqual(
            validate_settings({"path_diffusion_models": ""})["path_diffusion_models"], []
        )
        self.paths.paths["path_diffusion_models"] = []
        self.assertEqual(
            self.paths.get_model_paths()["modelfile_path"], [self.root / "models/checkpoints"]
        )
        with self.assertRaisesRegex(ValueError, "path_clip"):
            validate_settings({"path_clip": " "})

    def test_image_browser_discovers_diffusion_roots_and_deduplicates_relative_names(self):
        self.weight("models/checkpoints/shared.safetensors")
        self.weight("models/diffusion_models/shared.safetensors")
        self.weight("models/diffusion_models/image.safetensors")
        legacy = self.weight("models/unet/legacy.safetensors")
        self.paths.model_paths = self.paths.get_model_paths()
        state = SimpleNamespace(
            path_manager=self.paths,
            settings=SimpleNamespace(),
            translate=lambda text, **_: text,
        )
        with patch.dict("sys.modules", {"ruinedfooocus.app.state": state}):
            models = runpy.run_path(str(ROOT / "ruinedfooocus/features/models/model_handler.py"))[
                "Models"
            ]
            owner = models(offline=True)
            owner.update_all_models()
            owner.get_models_by_path = Mock(return_value={})
            owner.copy_embedded_preview = Mock(return_value=False)
            owner._update_models("checkpoints", owner.model_dirs["checkpoints"])
        self.assertEqual(
            owner.get_names("checkpoints"),
            ["image.safetensors", "legacy.safetensors", "shared.safetensors"],
        )
        metadata_paths = [call.args[1] for call in owner.get_models_by_path.call_args_list]
        self.assertIn(self.root / "models/checkpoints/shared.safetensors", metadata_paths)
        self.assertNotIn(self.root / "models/diffusion_models/shared.safetensors", metadata_paths)
        self.assertEqual(len(metadata_paths), 3)
        self.assertEqual(owner.get_file_from_name("checkpoints", legacy.name), legacy)
        self.assertEqual(
            owner.get_file_from_name("checkpoints", "shared.safetensors"),
            self.root / "models/checkpoints/shared.safetensors",
        )

    def test_native_registry_retains_extensions_defaults_and_configured_precedence(self):
        # Use the pinned functions, avoiding folder_paths' device-related imports.
        source = ROOT / "repositories/ComfyUI/folder_paths.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        categories = {
            target.slice.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "folder_names_and_paths"
        }
        self.assertTrue(set(NATIVE_IMAGE_FOLDERS) <= categories)
        extensions = {".safetensors", ".pt"}
        registry = {
            name: ([str(self.root / "native" / name)], extensions.copy())
            for name in NATIVE_IMAGE_FOLDERS
        }
        namespace = {"folder_names_and_paths": registry}
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"map_legacy", "add_model_folder_path", "get_folder_paths"}
        ]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        with patch.dict("sys.modules", {"folder_paths": SimpleNamespace(**namespace)}):
            register_image_model_paths(self.paths)
            first = {name: list(value[0]) for name, value in registry.items()}
            register_image_model_paths(self.paths)
        self.assertEqual(first, {name: value[0] for name, value in registry.items()})
        self.assertEqual(
            registry["text_encoders"][0],
            [
                str(self.root / "models/text_encoders"),
                str(self.root / "models/clip"),
                str(self.root / "native/text_encoders"),
            ],
        )
        self.assertTrue(all(value[1] == extensions for value in registry.values()))
        self.assertFalse((self.root / "models").exists())


if __name__ == "__main__":
    unittest.main()
