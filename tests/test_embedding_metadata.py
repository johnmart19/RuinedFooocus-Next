"""Background embedding metadata uses native discovery and respects offline mode."""

import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.features.models.model_identity import file_signature


def method(name, **scope):
    source = Path(__file__).resolve().parents[1] / "ruinedfooocus/features/models/model_handler.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Models"
    )
    node = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = dict(
        Path=Path, json=json, file_signature=file_signature, save_json=save_json, **scope
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


class EmbeddingMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "embedding.pt"
        self.path.write_bytes(b"fixture: never parse pickle or load weights")
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.owner = SimpleNamespace(
            offline=True,
            names={},
            ready={},
            cache_paths={"embeddings": self.cache},
            get_models_by_path=Mock(return_value={}),
            get_image=Mock(),
            detect_checkpoint_base=Mock(side_effect=AssertionError("backend detection forbidden")),
            model_sha256=Mock(side_effect=AssertionError("hashing forbidden")),
        )

    def test_worker_discovers_native_formats_and_offline_never_fetches_previews(self):
        (self.root / "other.pt2").write_bytes(b"fixture")
        (self.root / "not-embedding.gguf").write_bytes(b"fixture")
        with patch(
            "ruinedfooocus.features.models.resource_cards.resource_preview", return_value=None
        ):
            method("_update_embeddings")(self.owner, [self.root])
        self.assertEqual(
            [call.args[1].name for call in self.owner.get_models_by_path.call_args_list],
            ["embedding.pt", "other.pt2"],
        )
        self.assertEqual(self.owner.names, {})
        self.assertTrue(self.owner.ready["embeddings"])
        self.owner.get_image.assert_not_called()

    def test_online_preview_fetch_only_uses_explicit_sfw_image_artwork(self):
        safe = {"nsfwLevel": 1, "type": "image", "url": "https://example.invalid/safe"}
        self.owner.offline = False
        self.owner.get_models_by_path.return_value = {
            "images": [
                safe,
                {"nsfwLevel": 2, "url": "https://example.invalid/nsfw"},
                {"url": "https://example.invalid/unrated"},
                {"nsfwLevel": 1, "type": "video", "url": "https://example.invalid/video"},
            ]
        }
        with patch(
            "ruinedfooocus.features.models.resource_cards.resource_preview", return_value=None
        ):
            method("_update_embeddings")(self.owner, [self.root])
        self.assertEqual(self.owner.get_image.call_args.args[0]["images"], [safe])

    def test_cached_embedding_metadata_bypasses_checkpoint_backend_and_network(self):
        data = {"id": 42, "model": {"name": "Friendly embedding"}}
        (self.cache / "embedding.json").write_text(json.dumps(data), encoding="utf-8")
        with patch(
            "ruinedfooocus.features.models.model_sources.civitai_metadata",
            side_effect=AssertionError("offline metadata must not fetch"),
        ):
            result = method("get_models_by_path")(self.owner, "embeddings", self.path)
        self.assertEqual(result, data)
        self.owner.detect_checkpoint_base.assert_not_called()
        self.owner.model_sha256.assert_not_called()

    def test_online_worker_reuses_known_version_without_weight_hashing(self):
        self.owner.offline = False
        data = {"id": 42, "_rf_local_sha256": "A" * 64}
        (self.cache / "embedding.json").write_text(json.dumps(data), encoding="utf-8")
        remote = {"id": 42, "model": {"name": "Friendly embedding"}, "images": []}
        with patch(
            "ruinedfooocus.features.models.model_sources.civitai_metadata", return_value=remote
        ) as fetch:
            result = method("get_models_by_path")(self.owner, "embeddings", self.path, fetch=True)
        fetch.assert_called_once_with("model-versions/42")
        self.assertEqual(result["model"]["name"], "Friendly embedding")
        self.owner.detect_checkpoint_base.assert_not_called()
        self.owner.model_sha256.assert_not_called()

    def test_all_model_refresh_includes_embedding_worker(self):
        thread = Mock()
        dirs = {kind: [self.root] for kind in ("checkpoints", "loras", "embeddings", "inbox")}
        owner = SimpleNamespace(model_dirs=dirs, civit_update_worker=Mock(), offline=False)
        method("start_metadata_updates", threading=SimpleNamespace(Thread=thread))(owner)
        self.assertEqual(
            [call.kwargs["args"][0] for call in thread.call_args_list],
            ["checkpoints", "loras", "embeddings"],
        )
        self.assertEqual(thread.return_value.start.call_count, 3)


if __name__ == "__main__":
    unittest.main()
