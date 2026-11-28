from ruinedfooocus.core.paths import asset_path, resource_path

"""Prompt-rewrite presets and their output handling, without loading a model."""
import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def helpers():
    tree = ast.parse(
        (ROOT / "ruinedfooocus/features/chat/llama_pipeline.py").read_text(encoding="utf-8")
    )
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("clean_rewrite", "restore_tags")
    ]
    scope = {"asset_path": asset_path, "resource_path": resource_path, "re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "llama_pipeline", "exec"), scope)
    return scope["clean_rewrite"], scope["restore_tags"]


class RewriteOutputTests(unittest.TestCase):
    def test_echoed_labels_and_wrapping_quotes_are_removed(self):
        clean, _ = helpers()
        self.assertEqual(clean('Output: "a fox in the snow"'), "a fox in the snow")
        self.assertEqual(clean("  'a fox'  "), "a fox")
        self.assertEqual(clean('a "quoted" word stays'), 'a "quoted" word stays')
        self.assertEqual(clean(None), "")
        self.assertEqual(clean("close‑up, low–angle"), "close-up, low-angle")

    def test_dropped_lora_and_style_tags_are_restored_once(self):
        _, restore = helpers()
        original = "a fox <lora:detail:0.5> in snow <style:Cinematic> <lora:detail:0.5>"
        self.assertEqual(
            restore(original, "Solitude"), "Solitude <lora:detail:0.5> <style:Cinematic>"
        )
        kept = "a red fox, dusk <lora:detail:0.5> <style:Cinematic>"
        self.assertEqual(restore(original, kept), kept)
        self.assertEqual(restore("no tags here", "rewritten,"), "rewritten,")


class RewriteReasoningTests(unittest.TestCase):
    def test_reasoning_controls_follow_verified_model_template_not_filename(self):
        from ruinedfooocus.features.chat.model_profiles import profile_for, reasoning_kwargs
        from ruinedfooocus.features.models.model_registry import load_registry
        from ruinedfooocus.core.settings_validation import validate_settings

        registry = load_registry()
        lfm = profile_for(registry.artifact("existing:llm/LFM3-1.7B-Q4_K_M.gguf"))
        gpt = profile_for(registry.artifact("existing:llm/gpt-oss-20b-MXFP4.gguf"))
        qwen = profile_for(
            registry.artifact("existing:llm/Qwen3.5-9B-ultra-uncensored-heretic-v2-Q4_K_M.gguf")
        )
        self.assertEqual(reasoning_kwargs(lfm, "Low"), {"enable_thinking": True})
        for effort in ("Low", "Medium", "High"):
            self.assertEqual(reasoning_kwargs(gpt, effort), {"reasoning_effort": effort.lower()})
            self.assertEqual(reasoning_kwargs(qwen, effort), {"enable_thinking": effort != "Low"})
        self.assertEqual(reasoning_kwargs({}, "Low"), {})
        self.assertEqual(reasoning_kwargs(gpt, None), {})
        with self.assertRaises(ValueError):
            reasoning_kwargs(gpt, "Extreme")
        self.assertEqual(
            validate_settings({"llm_rewrite_reasoning": "High"})["llm_rewrite_reasoning"], "High"
        )
        with self.assertRaises(ValueError):
            validate_settings({"llm_rewrite_reasoning": "Extreme"})


class PresetFileTests(unittest.TestCase):
    def test_presets_have_a_name_rules_and_fit_the_context(self):
        presets = sorted((ROOT / "ruinedfooocus/resources/prompt_presets").glob("*.txt"))
        self.assertTrue(presets)
        names = set()
        for path in presets:
            with self.subTest(preset=path.name):
                name, _, body = path.read_text(encoding="utf-8").partition("\n")
                self.assertTrue(name.strip())
                self.assertNotIn(name.strip(), names)
                names.add(name.strip())
                self.assertIn("Reply with", body)
                # The system prompt shares the context with the user's prompt and reply.
                self.assertLess(len(body.split()), 450)


if __name__ == "__main__":
    unittest.main()
