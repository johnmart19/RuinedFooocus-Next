from ruinedfooocus.core.paths import asset_path, resource_path

"""Resolve requested wildcards literally and terminate cycles without model imports."""
import ast
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import random
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class WildcardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.expand = Mock(return_value=r"blue (cup)\1\g<name>")
        self.scope = dict(
            asset_path=asset_path,
            resource_path=resource_path,
            os=os,
            re=re,
            random=random,
            build_dynamic_prompt=self.expand,
            available_styles={},
            STYLE_ALIASES={},
            path_manager=SimpleNamespace(paths={"path_wildcards": [str(self.folder)]}),
        )
        self.scope["load_styles"] = lambda: self.scope["available_styles"]
        tree = ast.parse(
            (ROOT / "ruinedfooocus/features/prompting/prompt_processing.py").read_text(
                encoding="utf-8"
            )
        )
        nodes = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("process_wildcards", "use_random_prompt", "process_prompt")
        ]
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), "prompt_processing", "exec"),
            self.scope,
        )
        self.process = self.scope["process_wildcards"]

    def choice(self, name, text):
        (self.folder / (name + ".txt")).write_text(text, encoding="utf-8")

    def test_replacement_bytes_are_literal_and_directory_accepts_path_or_string(self):
        value = r"café C:\photos\1.png \1 \g<name> $5 (cup)"
        self.choice("subject", value)
        for directory in (self.folder, str(self.folder), [str(self.folder)]):
            self.assertEqual(self.process("Keep __subject__!", directory), f"Keep {value}!")

    def test_utf8_bom_does_not_become_an_invisible_prompt_character(self):
        (self.folder / "subject.txt").write_text("café cup", encoding="utf-8-sig")
        self.assertEqual(self.process("__subject__"), "café cup")

    def test_relative_roots_follow_writable_data_from_another_working_directory(self):
        user = self.folder / "wildcards"
        user.mkdir()
        (user / "data_root_subject.txt").write_text("blue cup", encoding="utf-8")
        self.scope["path_manager"].paths["path_wildcards"] = "wildcards"
        with patch.dict(os.environ, RF_DATA_DIR=str(self.folder)):
            self.assertEqual(self.process("__data_root_subject__"), "blue cup")

    def test_adjacent_duplicate_nested_and_independent_references(self):
        self.choice("subject", "__color__ cup")
        self.choice("color", "red")
        self.assertEqual(self.process("__subject____color__ __subject__"), "red cupred red cup")

    def test_missing_and_unmatched_placeholders_stay_literal(self):
        with redirect_stdout(StringIO()):
            self.assertEqual(
                self.process("Keep __missing__ and __unclosed"), "Keep __missing__ and __unclosed"
            )
        self.assertEqual(self.process("a__single"), "a__single")

    def test_direct_and_growing_indirect_cycles_fail_clearly(self):
        self.choice("self", "growing __self__")
        with self.assertRaisesRegex(ValueError, "self -> self"):
            self.process("__self__")
        self.choice("a", "one __b__")
        self.choice("b", "two __a__")
        with self.assertRaisesRegex(ValueError, "a -> b -> a"):
            self.process("__a__")

    def test_one_button_override_with_regex_punctuation_replaces_exactly(self):
        self.assertEqual(
            self.process(r"Keep __onebuttonsubject:cup (blue)\1__!"), r"Keep blue (cup)\1\g<name>!"
        )
        self.assertEqual(self.expand.call_args.kwargs["givensubject"], r"cup (blue)\1")

    def test_empty_files_error_and_intentional_blank_choices_are_kept(self):
        self.choice("empty", "# no choices\n")
        with self.assertRaisesRegex(ValueError, "no choices"):
            self.process("__empty__")
        self.choice("blank", "\n")
        self.assertEqual(self.process("A __blank__ cup"), "A  cup")

    def test_positive_preprocessing_preserves_negative_and_does_not_rescan_missing_text(self):
        self.scope["apply_style"] = lambda styles, prompt, negative, keywords, **kwargs: (
            prompt,
            negative,
        )
        self.choice("subject", r"red \1 cup")
        with redirect_stdout(StringIO()):
            result = self.scope["process_prompt"](
                [], "__subject__ __missing__", " , __negative__", {}
            )
        self.assertEqual(result, (r"red \1 cup __missing__", " , __negative__"))

    def test_inline_styles_consume_only_known_names_and_leave_literal_unknowns(self):
        apply = Mock(
            side_effect=lambda styles, prompt, negative, keywords, **kwargs: (prompt, negative)
        )
        self.scope.update(
            apply_style=apply, available_styles={"Style: Known": object(), "Custom": object()}
        )
        result = self.scope["process_prompt"](
            [], "A <style:Known> cup <style:Missing> <style:Custom>", "", {}
        )
        self.assertEqual(result, ("A  cup <style:Missing> ", ""))
        self.assertEqual(apply.call_args.args[0], ["Style: Known", "Custom"])

    def test_saved_processed_prompts_bypass_random_styles_wildcards_and_auto_negative(self):
        forbidden = Mock(side_effect=AssertionError("Saved prompt was reprocessed"))
        self.scope.update(
            apply_style=forbidden,
            build_dynamic_prompt=forbidden,
            build_dynamic_negative=forbidden,
            process_wildcards=forbidden,
        )
        prompt = " , Keep __subject__ and <style:Known> (red:0~1).  "
        negative = " , Original __negative__ text. "
        result = self.scope["process_prompt"](
            ["Known"],
            prompt,
            negative,
            {"_metadata_prompt": True, "obp_assume_direct_control": True, "auto_negative": True},
        )
        self.assertEqual(result, (prompt, negative))
        forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
