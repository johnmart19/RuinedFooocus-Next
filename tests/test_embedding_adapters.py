"""Configured embedding roots and native adapter semantics with tiny CPU fixtures."""

import ast
import logging
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

from ruinedfooocus.core.path import PathManager
from ruinedfooocus.core.settings_validation import validate_settings
from ruinedfooocus.backends.comfy.embedding_paths import register_embedding_paths
from ruinedfooocus.features.models.model_identity import file_signature

ROOT = Path(__file__).resolve().parents[1]


def lora_ui_callbacks():
    """Exercise the new browser's pure callbacks with per-client control values."""
    from ruinedfooocus.features.models.ui_adapters import lora_values
    import gradio as gr

    source = ROOT / "ruinedfooocus/features/models/ui_adapters.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {
        "checked",
        "change_lora",
        "set_lora",
        "delete_lora",
        "update_weight",
        "sync_weight",
        "lora_keywords",
        "triggers_for",
    }
    functions = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    scope = dict(
        gr=gr,
        math=__import__("math"),
        lora_values=lora_values,
        thumbnail=lambda name: "thumbnail " + name,
        keywords=lambda name: name,
        weight_min=0,
        weight_max=2,
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
    return scope


class EmbeddingAdapterTests(unittest.TestCase):
    @staticmethod
    def cards(*names):
        return [{"name": name, "kind": "LoRA"} for name in names]

    def test_new_lora_uses_default_weight_instead_of_inheriting_last_selection(self):
        scope = lora_ui_callbacks()
        cards = self.cards("A", "B")
        gallery = [(None, "0.35 - A")]
        # Selecting an inactive card always resets its own slider to 1.
        self.assertEqual(scope["sync_weight"](gallery, ("LoRA", "B"))["value"], 1)
        result = scope["set_lora"](cards, ("LoRA", "B"), gallery, 1)
        self.assertEqual([item[1] for item in result["value"]], ["0.35 - A", "1 - B"])
        for old, new, expected in (
            (" ", "", ""),
            ("\n\t", " ", ""),
            (" ", "style trigger, detail", "style trigger, detail"),
            ("existing trigger", " ", "existing trigger"),
            ("existing trigger", "new trigger", "existing trigger, new trigger"),
        ):
            scope["keywords"] = lambda name: old if name == "A" else new
            self.assertEqual(scope["lora_keywords"](cards, result["value"]), expected)

    def test_lora_deletion_preserves_only_remaining_nonempty_keywords(self):
        scope = lora_ui_callbacks()
        for remaining, expected in (
            ([" "], ""),
            (["\n\t", ""], ""),
            ([" ", "real trigger, detail"], "real trigger, detail"),
            (["first trigger", " ", "last trigger"], "first trigger, last trigger"),
            ([], ""),
        ):
            keywords = {f"kept-{index}": value for index, value in enumerate(remaining)}
            scope["keywords"] = lambda name: keywords[name]
            gallery = [(None, "0.35 - removed")] + [(None, f"0.7 - {name}") for name in keywords]
            result = scope["delete_lora"](
                self.cards("removed", *keywords), ("LoRA", "removed"), gallery, 0.35
            )
            self.assertEqual(
                scope["lora_keywords"](self.cards(*keywords), result["value"]), expected
            )
            self.assertEqual(
                [item[1] for item in result["value"]], [f"0.7 - {name}" for name in keywords]
            )

    def test_lora_weight_selection_is_isolated_between_clients(self):
        scope = lora_ui_callbacks()
        first = [(None, "0.3 - A"), (None, "0.6 - B")]
        second = [(None, "0.4 - C"), (None, "0.7 - D")]
        for gallery, selected, expected in (
            (first, "A", ["0.9 - A", "0.6 - B"]),
            (second, "D", ["0.4 - C", "0.9 - D"]),
        ):
            result = scope["update_weight"](
                self.cards("A", "B", "C", "D"), ("LoRA", selected), gallery, 0.9
            )
            self.assertEqual([item[1] for item in result["value"]], expected)
        self.assertEqual(first[0][1], "0.3 - A")

    def test_lora_deletion_preserves_neighbor_weight_and_clears_active_state(self):
        scope = lora_ui_callbacks()
        gallery = [(None, "0.3 - A"), (None, "0.6 - B")]
        for selected, remaining, weight in (("A", "B", 0.6), ("B", "A", 0.3)):
            result = scope["delete_lora"](self.cards("A", "B"), ("LoRA", selected), gallery, 0.9)[
                "value"
            ]
            self.assertEqual(scope["sync_weight"](result, ("LoRA", remaining))["value"], weight)
            empty = scope["delete_lora"](self.cards("A", "B"), ("LoRA", remaining), result, weight)
            self.assertEqual(empty["value"], [])
            self.assertEqual(scope["sync_weight"]([], ("LoRA", remaining))["value"], 1)
        self.assertEqual(len(gallery), 2)

    def test_lora_stale_or_deselected_choice_cannot_edit_a_different_adapter(self):
        scope = lora_ui_callbacks()
        import gradio as gr

        gallery = [(None, "0.6 - B")]
        for selected in (None, ("LoRA", "removed A")):
            with self.assertRaises(gr.Error):
                scope["update_weight"](self.cards("B"), selected, gallery, 0.9)
        self.assertNotIn(
            "value", scope["update_weight"](self.cards("A", "B"), ("LoRA", "A"), gallery, 0.9)
        )
        reordered = [(None, "0.7 - C"), *gallery]
        changed = scope["update_weight"](self.cards("B", "C"), ("LoRA", "B"), reordered, 0.9)
        self.assertEqual(changed["value"][1][1], "0.9 - B")

    def test_duplicate_lora_add_is_idempotent_and_keeps_other_weights(self):
        scope = lora_ui_callbacks()
        cards = self.cards("A", "B")
        gallery = [(None, "0.3 - A")]
        added = scope["set_lora"](cards, ("LoRA", "B"), gallery, 1)["value"]
        duplicate = scope["set_lora"](cards, ("LoRA", "B"), added, 1)["value"]
        self.assertEqual(len(duplicate), 2)
        self.assertEqual(duplicate[0][1], "0.3 - A")
        self.assertEqual(len(gallery), 1)

    def test_external_lora_gallery_updates_sync_selection_and_displayed_weight(self):
        scope = lora_ui_callbacks()
        for gallery, selected, expected in (
            ([(None, "0.5 - B")], ("LoRA", "A"), 1),
            ([(None, "0.5 - A")], ("LoRA", "A"), 0.5),
            ([], ("LoRA", "A"), 1),
            (None, ("LoRA", "A"), 1),
        ):
            self.assertEqual(scope["sync_weight"](gallery, selected)["value"], expected)
        self.assertFalse(scope["sync_weight"]([], None)["visible"])

    def test_configured_canonical_and_singular_roots_resolve_without_moving_or_mkdir(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            root = Path(folder).resolve()
            legacy = root / "models/embedding/lazypos.safetensors"
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(b"fixture stays here")
            before = legacy.read_bytes()
            owner = PathManager.__new__(PathManager)
            owner.paths = PathManager.DEFAULT_PATHS.copy()
            owner.get_abspath_folder = Mock(return_value=root)
            paths = owner.get_model_paths()
            self.assertEqual(
                paths["embedding_path"], [root / "models/embeddings", root / "models/embedding"]
            )
            self.assertFalse((root / "models/embeddings").exists())
            self.assertEqual(legacy.read_bytes(), before)
            owner.paths["path_embeddings"] = [str(root / "custom")]
            self.assertEqual(owner.get_model_paths()["embedding_path"], [root / "custom"])
            self.assertFalse((root / "custom").exists())

    def test_embedding_folder_setting_uses_multiline_list_contract(self):
        self.assertEqual(
            validate_settings({"path_embeddings": " one \n\n two "})["path_embeddings"],
            ["one", "two"],
        )
        self.assertEqual(validate_settings({"path_embeddings": ""})["path_embeddings"], [])

    def test_registry_preserves_native_roots_for_checkpoint_and_gguf_loaders(self):
        roots = [str(Path("native embedding root").resolve())]

        def add(kind, path, is_default):
            self.assertEqual(kind, "embeddings")
            self.assertTrue(is_default)
            if path in roots:
                roots.remove(path)
            roots.insert(0, path)

        native = SimpleNamespace(
            add_model_folder_path=add, get_folder_paths=lambda kind: list(roots)
        )
        configured = [Path("configured first"), Path("configured second")]
        with patch.dict("sys.modules", {"folder_paths": native}):
            result = register_embedding_paths(configured)
            self.assertEqual(
                result,
                [str(path.resolve()) for path in configured]
                + [str(Path("native embedding root").resolve())],
            )
            self.assertEqual(register_embedding_paths(configured), result)

    def test_native_embedding_lookup_keeps_both_sdxl_clip_branches_and_comma_syntax(self):
        source = ROOT / "repositories/ComfyUI/comfy/sd1_clip.py"
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
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.safetensors"
            save_file({"clip_l": torch.zeros(2, 768), "clip_g": torch.ones(2, 1280)}, path)
            before = path.read_bytes()
            for branch, width in (("clip_l", 768), ("clip_g", 1280)):
                owner = SimpleNamespace(
                    embedding_directory=[folder], embedding_size=width, embedding_key=branch
                )
                embed, name, remaining = scope["_try_get_embedding"](owner, "fixture, more text")
                self.assertEqual(tuple(embed.shape), (2, width))
                self.assertEqual(name, "fixture,")
                self.assertEqual(remaining, ", more text")
                self.assertEqual(embed.device.type, "cpu")
            self.assertEqual(path.read_bytes(), before)

    def test_embedding_syntax_is_preserved_by_inline_lora_parser(self):
        source = ROOT / "ruinedfooocus/features/prompting/prompt_processing.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "parse_loras"
        )
        scope = dict(re=re)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
        loras, positive, negative = scope["parse_loras"](
            "(embedding:lazypos:0.8) <lora:style:0.7>", "embedding:negative"
        )
        self.assertEqual(positive, "(embedding:lazypos:0.8) ")
        self.assertEqual(negative, "embedding:negative")
        self.assertEqual(loras, [{"name": "style", "weight": 0.7, "hash": None}])

    def test_text_encoder_lora_uses_native_clip_branch_and_removal_restores_base(self):
        source = ROOT / "ruinedfooocus/features/generation/image_pipeline.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "load_loras"
        )
        unet = SimpleNamespace(patches={})
        clip = SimpleNamespace(patcher=SimpleNamespace(patches={}))
        base = SimpleNamespace(unet=unet, clip=clip, vae="vae", clip_vision=None)
        applied = Mock(
            return_value=(
                unet,
                SimpleNamespace(patcher=SimpleNamespace(patches={"clip.weight": ["patch"]})),
            )
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "style.safetensors"
            path.write_bytes(b"tiny adapter fixture")
            scope = dict(
                file_signature=file_signature,
                shared=SimpleNamespace(
                    models=SimpleNamespace(get_model_path=Mock(return_value=path))
                ),
                comfy=SimpleNamespace(
                    utils=SimpleNamespace(load_torch_file=Mock(return_value="fixture")),
                    sd=SimpleNamespace(load_lora_for_models=applied),
                ),
            )
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
            owner = SimpleNamespace(
                xl_base=base,
                xl_base_patched=base,
                xl_base_patched_hash=None,
                StableDiffusionModel=lambda **values: SimpleNamespace(**values),
                conditions="stale",
            )
            scope["load_loras"](owner, [{"name": "style.safetensors", "weight": 0.65}])
            applied.assert_called_once_with(unet, clip, "fixture", 0.65, 0.65)
            self.assertIsNot(owner.xl_base_patched.clip, clip)
            self.assertEqual(clip.patcher.patches, {})
            self.assertIsNone(owner.conditions)
            scope["load_loras"](owner, [])
            self.assertIs(owner.xl_base_patched, base)
            self.assertIs(owner.xl_base_patched.clip, clip)


if __name__ == "__main__":
    unittest.main()
