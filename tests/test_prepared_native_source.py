"""Completed SHA receipts seed current source defaults without reading weights again."""

import hashlib
import json
from pathlib import Path
import tempfile
import runpy
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

from ruinedfooocus.core.cache_contract import file_identity
from ruinedfooocus.core.preparation import PreparationRequest
from ruinedfooocus.features.models import model_identity as identity
from ruinedfooocus.features.models.model_recommendations import ModelRecommendations


class PreparedSourceTests(unittest.TestCase):
    def test_model_handler_uses_current_proof_before_background_hash_and_keeps_source_family(self):
        with patch.dict(
            "sys.modules", {"ruinedfooocus.app.state": SimpleNamespace(translate=lambda text: text)}
        ):
            Models = runpy.run_path(
                str(
                    Path(__file__).resolve().parents[1]
                    / "ruinedfooocus/features/models/model_handler.py"
                )
            )["Models"]

        self.write()
        models = Models.__new__(Models)
        models.cache_paths = {"checkpoints": self.cache / "checkpoints"}
        models.offline = False
        models.cache_paths["checkpoints"].mkdir()
        with (
            patch.object(models, "model_sha256", side_effect=AssertionError("no duplicate hash")),
            patch.object(models, "detect_checkpoint_base", return_value=None),
            patch.object(models, "search_civitai_with_hash", return_value={}),
            patch(
                "ruinedfooocus.features.models.model_sources.huggingface_preview", return_value={}
            ),
        ):
            for fetch in (False, True):
                metadata = models.get_models_by_path("checkpoints", self.model, fetch=fetch)
                self.assertEqual(metadata["baseModel"], "QwenImage21")
                self.assertEqual(metadata["_rf_local_sha256"], self.checksum)

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.model = self.root / "arbitrary-name.safetensors"
        self.model.write_bytes(b"verified preparation payload")
        self.checksum = hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.url = (
            "https://huggingface.co/publisher/model/resolve/" + "a" * 40 + "/weights.safetensors"
        )
        self.row = dict(
            id="existing:diffusion_models/source.safetensors",
            catalogue_key="diffusion_models/source.safetensors",
            role="image_diffusion_checkpoint",
            native_family="QwenImage21",
            behavioral_variant="base",
            artifact=dict(
                sha256=self.checksum, bytes=self.model.stat().st_size, pinned_url=self.url
            ),
        )
        self.registry = SimpleNamespace(
            entries=lambda **kw: (self.row,),
            artifact=lambda name: self.row if name == self.row["id"] else None,
        )
        self.enterContext(
            patch(
                "ruinedfooocus.features.models.model_registry.load_registry",
                return_value=self.registry,
            )
        )
        self.request = PreparationRequest(
            self.url, self.model, expected_size=self.model.stat().st_size, sha256=self.checksum
        )
        self.signature = MappingProxyType(file_identity(self.model))
        self.cache = self.root / "cache-Юнікод"

    def write(self):
        return identity.record_prepared_native_source(
            self.model, self.request, self.signature, self.cache
        )

    def test_verified_receipt_is_reused_offline_and_keeps_explicit_overrides(self):
        store = ModelRecommendations(
            self.cache / "checkpoints", self.root / "settings.json", offline=True
        )
        values = dict(custom_steps=30, cfg=3, sampler_name="euler", scheduler="simple", clip_skip=1)
        store.save_override(str(self.model), values, ["euler"], ["simple"])
        with patch.object(
            identity, "cached_file_sha256", side_effect=AssertionError("no duplicate weight hash")
        ):
            proof = self.write()
            self.assertEqual(proof["baseModel"], "QwenImage21")
            metadata = store.metadata(str(self.model))
            self.assertIs(identity.verified_native_source(self.model, metadata), self.row)
            details = store.details(str(self.model), ["euler"], ["simple"])
            self.assertEqual(details["recommended"]["custom_steps"], 25)
            self.assertEqual(details["override"], values)
        self.assertFalse((self.cache / "checkpoints").exists())
        self.assertEqual(len(list((self.cache / "model-sources").glob("*.json"))), 1)

    def test_stale_file_wrong_destination_and_size_only_receipts_never_seed_source(self):
        self.assertIsNone(
            identity.record_prepared_native_source(self.model, self.request, None, self.cache)
        )
        size_only = PreparationRequest(
            self.url, self.model, expected_size=self.model.stat().st_size
        )
        self.assertIsNone(
            identity.record_prepared_native_source(
                self.model, size_only, self.signature, self.cache
            )
        )
        other = PreparationRequest(
            self.url,
            self.root / "other.safetensors",
            expected_size=self.model.stat().st_size,
            sha256=self.checksum,
        )
        with self.assertRaisesRegex(ValueError, "different destination"):
            identity.record_prepared_native_source(self.model, other, self.signature, self.cache)
        self.model.write_bytes(b"changed after actual verified completion")
        with self.assertRaisesRegex(ValueError, "changed after verification"):
            self.write()
        self.assertEqual(identity.read_prepared_native_source(self.model, self.cache), {})

    def test_same_payload_joins_only_aliases_with_one_variant_and_recipe_contract(self):
        alias = self.row | dict(id="existing:diffusion_models/alias.safetensors")
        with patch.object(self.registry, "entries", return_value=(self.row, alias)):
            self.assertTrue(self.write())
        for changed in (
            dict(behavioral_variant="turbo"),
            dict(evidence=dict(default_recipe_id="different-behavior")),
        ):
            other = alias | changed
            with patch.object(self.registry, "entries", return_value=(self.row, other)):
                with self.assertRaisesRegex(ValueError, "conflicts with its published native"):
                    self.write()

    def test_corrupt_unknown_schema_stale_registry_and_regeneration_are_safe(self):
        self.write()
        record = next((self.cache / "model-sources").glob("*.json"))
        for data in ("{", '{"schema_version":99}'):
            record.write_text(data, encoding="utf-8")
            self.assertEqual(identity.read_prepared_native_source(self.model, self.cache), {})
            self.write()
            self.assertTrue(identity.read_prepared_native_source(self.model, self.cache))
        with patch.dict(self.row, artifact=dict(self.row["artifact"], sha256="b" * 64)):
            self.assertEqual(identity.read_prepared_native_source(self.model, self.cache), {})
        self.model.write_bytes(b"replaced again")
        self.assertEqual(identity.read_prepared_native_source(self.model, self.cache), {})

    def test_constructor_write_failure_and_concurrent_source_change_preserve_inputs(self):
        self.write()
        record = next((self.cache / "model-sources").glob("*.json"))
        previous = record.read_bytes()
        with patch("ruinedfooocus.core.cache_contract.save_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.write()
        self.assertEqual(record.read_bytes(), previous)
        with patch.object(
            identity,
            "write_record",
            side_effect=lambda *args: self.model.write_bytes(
                b"changed during metadata publication"
            ),
        ):
            with self.assertRaisesRegex(ValueError, "changed during source metadata"):
                self.write()
        self.assertEqual(identity.read_prepared_native_source(self.model, self.cache), {})
        not_directory = self.root / "cache-file"
        not_directory.write_text("preserve", encoding="utf-8")
        self.model.write_bytes(b"verified preparation payload")
        fresh = MappingProxyType(file_identity(self.model))
        with self.assertRaises(OSError):
            identity.record_prepared_native_source(self.model, self.request, fresh, not_directory)
        self.assertEqual(not_directory.read_text(), "preserve")

    def test_new_proof_does_not_relabel_unbound_basename_metadata(self):
        self.write()
        checkpoints = self.cache / "checkpoints"
        checkpoints.mkdir()
        old = checkpoints / (self.model.stem + ".json")
        old.write_text(
            json.dumps(
                dict(id=123, baseModel="Different model", generationConfig=dict(cfgScale=19))
            ),
            encoding="utf-8",
        )
        previous = old.read_bytes()
        store = ModelRecommendations(checkpoints, self.root / "settings.json", offline=True)
        self.assertEqual(store.metadata(str(self.model))["baseModel"], "QwenImage21")
        self.assertNotIn("id", store.metadata(str(self.model)))
        self.assertEqual(old.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
