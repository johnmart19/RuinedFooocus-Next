from ruinedfooocus.core.paths import asset_path, resource_path

"""Style templates may add text without rewriting placeholders in the user's text."""
import ast
from pathlib import Path
import random
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StylePayloadTests(unittest.TestCase):
    def apply(self, selected, prompt, keywords="metadata {prompt}"):
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/prompting/sdxl_styles.py").read_text(encoding="utf-8")
        )
        node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_style"
        )
        scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            random=random,
            re=re,
            STYLE_ALIASES={},
            styles={
                "Frame": ("Frame {prompt}, , done", ""),
                "Outer": ("Outer {prompt}", ""),
                "Keywords": ("{prompt} {lora_keywords}", ""),
                "Negative only": (None, "noise"),
                "Empty": ("", ""),
                "Negative empty positive": ("", "noise"),
                "Addition": ("paper collage", ""),
            },
            default_style=("{prompt}", ""),
            allstyles=[],
        )
        scope["load_styles"] = lambda: scope["styles"]
        exec(compile(ast.Module(body=[node], type_ignores=[]), "style_payload", "exec"), scope)
        return scope["apply_style"](selected, prompt, "literal __negative__", keywords)

    def test_prompt_markers_and_punctuation_are_literal_across_multiple_styles(self):
        prompt = " , Sign says {prompt}, , {lora_keywords}.  "
        positive, negative = self.apply(["Frame", "Outer"], prompt)
        self.assertEqual(positive, "Outer Frame " + prompt + ",  done, , ")
        self.assertTrue(negative.endswith("literal __negative__"))

    def test_keywords_are_expanded_only_in_templates_and_not_reexpanded(self):
        prompt = "Sign says {lora_keywords} and {prompt}."
        positive, _ = self.apply(["Keywords", "Outer"], prompt)
        self.assertEqual(positive, "Outer " + prompt + " metadata {prompt}, , ")
        positive, _ = self.apply(["Keywords"], prompt, None)
        self.assertEqual(positive, prompt + " {lora_keywords}, ")

    def test_negative_only_and_empty_styles_leave_positive_payload_unchanged(self):
        prompt = " , Sign says {prompt}, , {lora_keywords}.  "
        for selected in ([], ["Negative only"], ["Empty"], ["Negative empty positive"]):
            positive, _ = self.apply(selected, prompt)
            self.assertEqual(positive, prompt)
        positive, _ = self.apply(["Frame", "Negative only"], prompt)
        self.assertEqual(positive, "Frame " + prompt + ",  done, ")

    def test_styles_without_negative_text_leave_negative_prompt_exact(self):
        _, negative = self.apply(["Frame"], "Two cups.")
        self.assertEqual(negative, "literal __negative__")

    def test_legacy_headers_before_or_after_styles_never_erase_prompt(self):
        prompt = 'Keep exactly two red apples and the text "café".'
        for selection in (
            [">>>>>> PAPERCRAFT"],
            [">>>>>> PAPERCRAFT", "Frame"],
            ["Frame", ">>>>>> PHOTO"],
        ):
            before = selection.copy()
            positive, _ = self.apply(selection, prompt)
            self.assertIn(prompt, positive)
            self.assertEqual(selection, before)

    def test_style_without_placeholder_adds_to_prompt_instead_of_replacing_it(self):
        positive, _ = self.apply(["Addition"], "Two red apples.")
        self.assertIn("Two red apples.", positive)
        self.assertIn("paper collage", positive)


if __name__ == "__main__":
    unittest.main()
