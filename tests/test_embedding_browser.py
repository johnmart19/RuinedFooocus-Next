"""Embedding discovery and explicit prompt edits never enter the LoRA/model loader."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr

from ruinedfooocus.features.models.embedding_browser import (
    insert_embedding,
    scan_embeddings,
    selected_embedding,
)


class EmbeddingBrowserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, root=None):
        path = (root or self.root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"private fixture; discovery must not read tensor contents")
        return path

    def test_discovery_is_read_only_deduplicates_alias_and_nested_roots(self):
        self.write("nested/lazypos.safetensors")
        self.write("other.pt")
        self.write("ignored.gguf")
        self.write("ignored.png")
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        missing = self.root / "absent"
        with patch("builtins.open", side_effect=AssertionError("discovery must not open weights")):
            records, message = scan_embeddings(
                [self.root, self.root / ".", self.root / "nested", missing]
            )
        self.assertEqual(
            [entry.name for entry in records], ["nested/lazypos.safetensors", "other.pt"]
        )
        self.assertIn("2 embeddings available", message)
        self.assertFalse(missing.exists())
        self.assertEqual(
            before, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        )

    def test_unsafe_names_are_reported_without_transforming_their_stems(self):
        for name in ("two words.pt", "name(1).pt", "__wildcard__.pt", "batch---split.pt"):
            self.write(name)
        self.write("café.pth")
        records, message = scan_embeddings(self.root)
        self.assertEqual([entry.name for entry in records], ["café.pth"])
        self.assertTrue(message.startswith("1 embedding available."))
        self.assertIn("4 filenames contain whitespace or prompt syntax", message)

    def test_unreadable_root_is_reported_without_breaking_discovery(self):
        with patch.object(Path, "is_dir", side_effect=PermissionError("private fixture")):
            records, message = scan_embeddings(self.root)
        self.assertEqual(records, [])
        self.assertIn("1 file or folder could not be read", message)

    def test_duplicate_names_across_roots_are_hidden_but_subfolders_remain_distinct(self):
        first, second = self.root / "first", self.root / "second"
        self.write("same.pt", first)
        self.write("same.pt", second)
        self.write("a/style.pt", first)
        self.write("b/style.pt", second)
        records, message = scan_embeddings([first, second])
        self.assertEqual([entry.name for entry in records], ["a/style.pt", "b/style.pt"])
        self.assertIn("2 conflicting prompt names", message)
        self.assertNotIn("style.pt", records[0].aliases)
        self.assertNotIn("style", records[1].aliases)

    def test_basename_collision_in_recursive_native_lookup_is_not_silently_selected(self):
        self.write("style.pt")
        self.write("nested/style.pt")
        records, message = scan_embeddings(self.root)
        self.assertEqual([entry.name for entry in records], ["nested/style.pt"])
        self.assertIn("1 conflicting prompt name is hidden", message)

    def test_explicit_formats_and_unique_aliases_do_not_confuse_two_same_stem_files(self):
        self.write("style.pt")
        self.write("style.safetensors")
        records, _ = scan_embeddings(self.root)
        self.assertEqual(
            [entry.token for entry in records],
            ["embedding:style.pt", "embedding:style.safetensors"],
        )
        self.assertNotIn("style", records[0].aliases)
        self.assertEqual(
            insert_embedding("embedding:style.pt", records[1]),
            "embedding:style.safetensors embedding:style.pt",
        )

    def test_insertion_preserves_text_and_is_idempotent_for_native_aliases_and_weights(self):
        self.write("nested/lazypos.safetensors")
        records, _ = scan_embeddings(self.root)
        entry = records[0]
        original = '  Two cups, 42 "CAFÉ" <lora:kept:0.5>\n'
        result = insert_embedding(original, entry)
        self.assertEqual(result, "embedding:nested/lazypos.safetensors " + original)
        self.assertEqual(insert_embedding(result, entry), result)
        for prompt in (
            "embedding:lazypos",
            "text embedding:lazypos, more",
            "(embedding:lazypos:0.8)",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(insert_embedding(prompt, entry), prompt)
        for prompt in (
            "embedding:lazypos-extra",
            '"embedding:lazypos"',
            "embedding:lazypos,more",
            "EMBEDDING:lazypos",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(insert_embedding(prompt, entry), entry.token + " " + prompt)

    def test_selection_rejects_forged_removed_and_replaced_files(self):
        path = self.write("test.pt")
        records, _ = scan_embeddings(self.root)
        self.assertEqual(selected_embedding(records, "test.pt"), records[0])
        with self.assertRaisesRegex(ValueError, "Choose an installed"):
            selected_embedding(records, "../forged.pt")
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed or was removed"):
            selected_embedding(records, "test.pt")
        path.unlink()
        with self.assertRaisesRegex(ValueError, "changed or was removed"):
            selected_embedding(records, "test.pt")

    def test_generated_reference_resolves_in_the_pinned_native_loader(self):
        import ast
        import logging
        import os
        import re
        from types import SimpleNamespace

        import torch
        from safetensors.torch import save_file

        source = Path(__file__).resolve().parents[1] / "repositories/ComfyUI/comfy/sd1_clip.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("expand_directory_list", "bundled_embed", "load_embed")
        ]
        tokenizer = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SDTokenizer"
        )
        lookup = next(
            node
            for node in tokenizer.body
            if isinstance(node, ast.FunctionDef) and node.name == "_try_get_embedding"
        )
        scope = dict(os=os, torch=torch, re=re, logging=logging)
        exec(
            compile(ast.Module(body=functions + [lookup], type_ignores=[]), str(source), "exec"),
            scope,
        )
        path = self.root / "nested/lazypos.safetensors"
        path.parent.mkdir()
        save_file({"clip_l": torch.zeros(2, 768)}, path)
        before = path.read_bytes()
        records, _ = scan_embeddings(self.root)
        reference = insert_embedding("", records[0])
        owner = SimpleNamespace(
            embedding_directory=[str(self.root)], embedding_size=768, embedding_key="clip_l"
        )
        embedding, name, remaining = scope["_try_get_embedding"](
            owner, reference.removeprefix("embedding:") + ", more text"
        )
        self.assertEqual(tuple(embedding.shape), (2, 768))
        self.assertEqual(embedding.device.type, "cpu")
        self.assertEqual(name, "nested/lazypos.safetensors,")
        self.assertEqual(remaining, ", more text")
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
