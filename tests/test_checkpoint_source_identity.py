"""Explicit known-source verification never turns browsing into tensor loading."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr

from ruinedfooocus.features.models import checkpoint_tools as tools
from ruinedfooocus.features.models.image_components import inventory
from ruinedfooocus.features.models.model_identity import verified_native_source
from test_model_cache import method
from test_checkpoint_selection_lane import checkpoint_ui_scope


class CheckpointSourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "renamed-source.ckpt"
        self.source.write_bytes(b"private source identity fixture")
        self.cache = self.root / "cache/checkpoints"
        self.annotation = dict(
            id="existing:checkpoints/official.ckpt",
            catalogue_key="checkpoints/official.ckpt",
            role="image_diffusion_checkpoint",
            native_family="SD_X4Upscaler",
            bundled_components=["clip", "vae"],
            artifact=dict(
                bytes=self.source.stat().st_size,
                sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
            ),
        )
        self.enterContext(
            patch(
                "ruinedfooocus.features.models.model_registry.load_registry",
                return_value=SimpleNamespace(entries=lambda: [self.annotation]),
            )
        )
        self.enterContext(
            patch.object(
                tools, "safe_open", side_effect=AssertionError("Identity is not a tensor scan")
            )
        )

    def test_explicit_sha_size_match_survives_rename_and_publishes_signature_bound_family(self):
        result = tools.verify_checkpoint_source(
            self.source, self.cache, {"name": "Publisher title"}
        )
        self.assertEqual(result["family"], "SD_X4Upscaler")
        cache = (self.cache / self.source.name).with_suffix(".json")
        metadata = json.loads(cache.read_text(encoding="utf-8"))
        self.assertEqual(metadata["name"], "Publisher title")
        self.assertEqual(metadata["_rf_verified_native_source"]["artifact"], self.annotation["id"])
        lookup = method(
            "ruinedfooocus/features/models/model_handler.py",
            "get_models_by_path",
            dict(json=json, Path=Path),
        )
        owner = SimpleNamespace(
            cache_paths={"checkpoints": self.cache},
            offline=True,
            detect_checkpoint_base=lambda path: None,
        )
        self.assertEqual(lookup(owner, "checkpoints", self.source)["baseModel"], "SD_X4Upscaler")
        paths = SimpleNamespace(
            model_paths={"cache_path": self.cache.parent},
            paths={},
            get_abspath=Path,
            DOWNLOADABLE_FILES={},
        )
        state = inventory(self.source, "Unknown", {}, paths)
        self.assertEqual(state["architecture"], "SD_X4Upscaler")
        self.assertEqual(
            [row["status"] for row in state["rows"]],
            ["Embedded in checkpoint", "Embedded in checkpoint"],
        )
        self.assertFalse(any(row["download_available"] for row in state["rows"]))
        self.source.with_suffix(".civitai.info").write_text(
            json.dumps({"baseModel": "SD 1.5", "name": "Legacy publisher title"}), encoding="utf-8"
        )
        overridden = lookup(owner, "checkpoints", self.source)
        self.assertEqual(overridden["baseModel"], "SD_X4Upscaler")
        self.assertEqual(overridden["name"], "Legacy publisher title")
        self.assertEqual(overridden["_rf_local_sha256"], result["sha256"])
        self.source.with_suffix(".civitai.info").unlink()
        self.source.write_bytes(self.source.read_bytes() + b"replacement")
        self.assertNotIn("baseModel", lookup(owner, "checkpoints", self.source))
        with self.assertRaisesRegex(ValueError, "No reviewed native family"):
            tools.verify_checkpoint_source(self.source, self.cache)

    def test_unknown_checksum_or_wrong_published_size_cannot_assign_family(self):
        for field, value in (("sha256", "0" * 64), ("bytes", self.source.stat().st_size + 1)):
            before = dict(self.annotation["artifact"])
            self.annotation["artifact"][field] = value
            with self.assertRaisesRegex(ValueError, "No reviewed native family"):
                tools.verify_checkpoint_source(self.source, self.cache)
            self.assertFalse((self.cache / self.source.name).with_suffix(".json").exists())
            self.annotation["artifact"] = before

    def test_wrong_registry_kind_and_wrong_captured_signature_cannot_qualify(self):
        tools.verify_checkpoint_source(self.source, self.cache)
        metadata = json.loads(
            (self.cache / self.source.name).with_suffix(".json").read_text(encoding="utf-8")
        )
        self.assertIsNotNone(
            verified_native_source(
                self.source, metadata, current_signature=metadata["_rf_file_signature"]
            )
        )
        self.assertIsNone(
            verified_native_source(self.source, metadata, current_signature=["wrong", 0, 0, 0])
        )
        for kind, value in (("role", "image_codec"), ("catalogue_key", "vae/official.ckpt")):
            original = self.annotation[kind]
            self.annotation[kind] = value
            self.assertIsNone(verified_native_source(self.source, metadata))
            with self.assertRaisesRegex(ValueError, "No reviewed native family"):
                tools.verify_checkpoint_source(self.source, self.cache)
            self.annotation[kind] = original

    def test_ui_selection_is_passive_and_only_explicit_inspect_verifies_source(self):
        model = SimpleNamespace(cache_paths={"checkpoints": self.cache}, revision=0)
        scope = dict(
            gr=gr,
            Path=Path,
            translate_ui=lambda value: value,
            NATIVE_TORCH_SUFFIXES=(".ckpt",),
            is_safetensors=lambda name: str(name).endswith(".safetensors"),
            tools=tools,
            shared=SimpleNamespace(models=model),
            _selected=lambda name: (self.source, {}),
            SafetensorError=ValueError,
            verified_native_source=verified_native_source,
        )
        scope = checkpoint_ui_scope(scope)
        with gr.Blocks() as demo:
            picker = gr.Dropdown(choices=[self.source.name], value=self.source.name)
            refresh = gr.Number(value=0, visible=False)
            scope["create_ui"](picker, model_refresh=refresh)
        callbacks = {getattr(fn.fn, "__name__", ""): fn.fn for fn in demo.fns.values()}
        session = scope["CheckpointToolSelection"]()

        def select(version):
            callbacks["begin_selection"](self.source.name, session, version)
            return callbacks["selection_current"](self.source.name, session, version)["updates"]

        with patch.object(
            tools, "verify_checkpoint_source", wraps=tools.verify_checkpoint_source
        ) as verify:
            message, convert, check = select(1)
            self.assertIn("click inspect", message.casefold())
            self.assertFalse(convert["interactive"])
            self.assertFalse(check["interactive"])
            verify.assert_not_called()
            message, _, _, revision = callbacks["inspect_current"](self.source.name, 7, session, 1)[
                "updates"
            ]
            self.assertEqual(revision, 8)
            self.assertIn("refresh automatically", message)
            verify.assert_called_once()
            self.assertIn("Verified published source: SD_X4Upscaler", message)
            self.assertIn("No tensors were loaded", message)
            self.assertEqual(model.revision, 1)
            registration = next(
                fn
                for fn in demo.fns.values()
                if getattr(fn.fn, "__name__", "") == "inspect_current"
            )
            publication = next(
                item
                for item in demo.config["dependencies"]
                if item["trigger_after"] == registration._id
            )
            self.assertEqual(publication["outputs"][-1], refresh._id)
            metadata = json.loads(
                (self.cache / self.source.name).with_suffix(".json").read_text(encoding="utf-8")
            )
            scope["_selected"] = lambda name: (self.source, metadata)
            self.assertIn(
                "Verified published family",
                select(2)[0],
            )
            self.source.write_bytes(self.source.read_bytes() + b"changed")
            self.assertIn(
                "click inspect",
                select(3)[0].casefold(),
            )
            verify.assert_called_once()
            message, _, _, failed_revision = callbacks["inspect_current"](
                self.source.name, 8, session, 3
            )["updates"]
            self.assertIn("Checkpoint tool:", message)
            self.assertEqual(failed_revision, gr.skip())

    def test_safetensors_explicit_inspect_verifies_only_size_candidates_and_keeps_header_on_failure(
        self,
    ):
        renamed = self.source.with_suffix(".safetensors")
        self.source.rename(renamed)
        self.source = renamed
        model = SimpleNamespace(
            cache_paths={"checkpoints": self.cache},
            revision=0,
            get_model_base=lambda data: data.get("baseModel", "Unknown"),
        )
        scope = dict(
            gr=gr,
            Path=Path,
            translate_ui=lambda value: value,
            NATIVE_TORCH_SUFFIXES=(".ckpt",),
            is_safetensors=lambda name: str(name).endswith(".safetensors"),
            tools=tools,
            shared=SimpleNamespace(models=model),
            _selected=lambda name: (self.source, {}),
            SafetensorError=ValueError,
            verified_native_source=verified_native_source,
        )
        scope = checkpoint_ui_scope(scope)
        with gr.Blocks() as demo:
            picker = gr.Dropdown(choices=[self.source.name], value=self.source.name)
            refresh = gr.Number(value=0, visible=False)
            scope["create_ui"](picker, model_refresh=refresh)
        callbacks = {getattr(fn.fn, "__name__", ""): fn.fn for fn in demo.fns.values()}
        session = scope["CheckpointToolSelection"]()

        def select(version):
            callbacks["begin_selection"](self.source.name, session, version)
            return callbacks["selection_current"](self.source.name, session, version)["updates"]

        report = lambda path: dict(
            name=path.name,
            tensors=1,
            bytes=path.stat().st_size,
            dtypes={"F16": 4},
            saving_bytes=0,
            previous_repair=False,
        )
        with (
            patch.object(tools, "inspect_checkpoint", side_effect=report),
            patch.object(
                tools, "verify_checkpoint_source", wraps=tools.verify_checkpoint_source
            ) as verify,
        ):
            passive = select(1)[0]
            self.assertIn("Weight values have not been checked", passive)
            verify.assert_not_called()
            self.assertFalse(self.cache.exists())
            message, _, _, revision = callbacks["inspect_current"](self.source.name, 3, session, 1)[
                "updates"
            ]
            self.assertIn("Verified published source", message)
            self.assertIn("Weight values have not been checked", message)
            self.assertEqual(revision, 4)
            self.assertEqual(model.revision, 1)
            verify.assert_called_once()
            cached = json.loads(
                (self.cache / self.source.name).with_suffix(".json").read_text(encoding="utf-8")
            )
            self.assertIsNotNone(verified_native_source(self.source, cached))
            self.source.write_bytes(b"x" * self.source.stat().st_size)
            select(2)
            message, _, _, revision = callbacks["inspect_current"](self.source.name, 4, session, 2)[
                "updates"
            ]
            self.assertIn("Weight values have not been checked", message)
            self.assertIn("Source verification failed", message)
            self.assertIn("Choose Own settings", message)
            self.assertEqual(revision, gr.skip())
            self.assertEqual(model.revision, 1)
            self.assertIsNone(verified_native_source(self.source, cached))
            self.source.write_bytes(self.source.read_bytes() + b"unknownsize")
            before = verify.call_count
            select(3)
            message, _, _, revision = callbacks["inspect_current"](self.source.name, 4, session, 3)[
                "updates"
            ]
            self.assertIn("Weight values have not been checked", message)
            self.assertNotIn("verification failed", message)
            self.assertEqual(verify.call_count, before)
            self.assertEqual(revision, gr.skip())


if __name__ == "__main__":
    unittest.main()
