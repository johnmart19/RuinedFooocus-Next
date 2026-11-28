"""Bare embedding names in prompts and automatic prompt targets, without loading weights."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ruinedfooocus.features.models import embedding_browser as eb  # noqa: E402


class TargetTests(unittest.TestCase):
    def target(self, name, metadata=None, default="Positive"):
        return eb.embedding_target(eb.Embedding(name, (), ()), metadata, default)

    def test_negative_names_go_to_negative_prompt(self):
        for name in (
            "lazyneg",
            "lazyhand",
            "lazydn",
            "EasyNegative",
            "bad_prompt_v2",
            "badhandv4",
            "ng_deepnegative_v1_75t",
            "sub/lazyneg.safetensors",
            "lazyneg_v2",
            "neg_hands",
            "verybadimagenegative_v1.3",
        ):
            self.assertEqual(self.target(name), "Negative", name)

    def test_positive_and_dual_use_names_keep_chosen_target(self):
        for name in (
            "lazypos",
            "lazympos",
            "lazynsfw",
            "lazyreal",
            "handsome",
            "negligee_style",
            "renegade",
            "hand",
            "badlands",
            "negotiation",
        ):
            self.assertEqual(self.target(name), "Positive", name)
            self.assertEqual(self.target(name, default="Negative"), "Negative", name)

    def test_explicit_sidecar_destination_overrides_known_negative_identity(self):
        self.assertEqual(self.target("lazyneg", {"prompt_target": "Positive"}), "Positive")

    def test_description_recommending_only_negative(self):
        only = {"description": "<p>Put lazywet in negative to remove wet skin.</p>"}
        both = {"description": "Use it in positive or negative depending on your needs."}
        self.assertEqual(self.target("lazywet", only), "Negative")
        self.assertEqual(self.target("lazynsfw", both), "Positive")


class NativeTextTests(unittest.TestCase):
    def test_bare_names_expand_and_new_files_apply_without_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cache = Path(tmp) / "embeddings", Path(tmp) / "cache"
            (root / "nested").mkdir(parents=True)
            cache.mkdir()
            (root / "lazyneg.safetensors").write_bytes(b"a")
            text = eb.native_embedding_text("lazyneg, lazyhand, worst quality", [root], cache)
            self.assertEqual(text, "embedding:lazyneg.safetensors , lazyhand, worst quality")
            (root / "nested" / "lazyhand.safetensors").write_bytes(b"b")
            text = eb.native_embedding_text("lazyneg, lazyhand, worst quality", [root], cache)
            self.assertIn("embedding:nested/lazyhand", text)
            self.assertIn("embedding:lazyneg", text)

    def test_publisher_trigger_word_from_new_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cache = Path(tmp) / "embeddings", Path(tmp) / "cache"
            root.mkdir()
            cache.mkdir()
            (root / "lazy_negative_v2.safetensors").write_bytes(b"a")
            self.assertEqual(eb.native_embedding_text("lazyneg2", [root], cache), "lazyneg2")
            (cache / "lazy_negative_v2.json").write_text(
                json.dumps({"trainedWords": ["lazyneg2"]}), encoding="utf-8"
            )
            self.assertEqual(
                eb.native_embedding_text("lazyneg2", [root], cache),
                "embedding:lazy_negative_v2.safetensors",
            )


if __name__ == "__main__":
    unittest.main()
