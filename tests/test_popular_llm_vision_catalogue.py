"""Curated popular LLM & Vision catalogue entries, VRAM metadata, and UI badges."""

import unittest

from ruinedfooocus.app.state import path_manager
from ruinedfooocus.features.chat.llama_models import _vram_hint, model_catalogue
from ruinedfooocus.features.models.model_registry import load_registry
from ruinedfooocus.features.models.ui_acquisition import _vram_badge


class PopularLLMVisionCatalogueTests(unittest.TestCase):
    EXPECTED_MODELS = (
        "llm/Qwen3-4B-Q4_K_M.gguf",
        "llm/Qwen3-8B-Q4_K_M.gguf",
        "llm/Qwen3-14B-Q4_K_M.gguf",
        "llm/Qwen3-30B-A3B-Q4_K_M.gguf",
        "llm/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        "llm/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "llm/Qwen3-VL-4B-Instruct-Q4_K_M.gguf",
        "llm/Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
        "llm/gemma-3-12b-it-Q4_K_M.gguf",
        "llm/gemma-3-27b-it-Q4_K_M.gguf",
        "llm/Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf",
        "llm/phi-4-Q4_K_M.gguf",
        "llm/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        "llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "llm/InternVL3-8B-Instruct-Q4_K_M.gguf",
    )

    def test_all_curated_popular_models_exist_with_vram_planning_profiles(self):
        registry = load_registry()
        for key in self.EXPECTED_MODELS:
            with self.subTest(key=key):
                self.assertIn(key, path_manager.DOWNLOADABLE_FILES)
                entry = registry.annotations_for(key, path_manager.DOWNLOADABLE_FILES[key])
                self.assertEqual(entry["role"], "language_model")
                self.assertTrue(entry["chat_weight"])
                self.assertGreater(entry["artifact"]["bytes"], 1024**3)
                self.assertEqual(len(entry["artifact"]["sha256"]), 64)
                profiles = entry["hardware"]["planning_profiles"]
                self.assertTrue(profiles)
                self.assertTrue(profiles[0]["estimated_full_offload_vram_range_gib"])

    def test_vram_badges_and_chat_quant_labels_show_approximate_vram(self):
        registry = load_registry()
        key = "llm/Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
        entry = registry.annotations_for(key, path_manager.DOWNLOADABLE_FILES[key])
        badge = _vram_badge({"task": "Chat", "category": "llm", "annotation": entry})
        self.assertIn("GiB VRAM", badge)
        self.assertIn("7–8.5", badge)

        hint = _vram_hint(
            "Qwen3-VL-8B-Instruct-Q4_K_M.gguf", "Qwen3-VL-8B-Instruct-Q4_K_M.gguf", False, registry
        )
        self.assertEqual(hint, "~7–8.5 GiB VRAM")

        cat = model_catalogue()
        self.assertIn("Qwen3-VL-8B-Instruct", cat)
        label = cat["Qwen3-VL-8B-Instruct"][0][0]
        self.assertIn("Q4_K_M", label)
        self.assertIn("GiB VRAM", label)


if __name__ == "__main__":
    unittest.main()
