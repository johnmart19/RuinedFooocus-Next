"""Behavior corrections separate from dataset cache equivalence."""

import io
import os
import random
import tempfile
import unittest
from unittest.mock import patch


class PromptEngineControlsTests(unittest.TestCase):
    def test_unicode_prompt_is_returned_without_printing_to_legacy_console(self):
        from ruinedfooocus.features.prompting.one_button.build_dynamic_prompt import (
            build_dynamic_prompt,
        )

        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1251", errors="strict")
        self.addCleanup(stream.close)
        literal = "café fox 🦊 \udcff"
        with patch("sys.stdout", stream):
            prompt = build_dynamic_prompt(
                seed=0, artists="none", givensubject=literal, prompt_enhancer="none"
            )
        self.assertIn(literal, prompt)

    def test_explicit_subtypes_survive_broad_subject_and_combined_subject_fills_defaults(self):
        from ruinedfooocus.features.prompting.one_button.build_dynamic_prompt import (
            translate_subject_options,
        )

        self.assertEqual(
            translate_subject_options("object", "vehicles", "cat", "city", "human", "event"),
            ("object", "vehicles", "cat", "city", "human", "event"),
        )
        self.assertEqual(
            translate_subject_options("object - vehicle", "all", "", "all", "all", "all"),
            ("object", "vehicles", "vehicles", "vehicles", "vehicles", "vehicles"),
        )
        self.assertEqual(
            translate_subject_options("animal - cat", "all", "dog", "all", "all", "all")[2], "dog"
        )

    def test_explicit_zero_seed_repeats_text_and_rng_state(self):
        from ruinedfooocus.features.prompting.one_button.build_dynamic_prompt import (
            build_dynamic_prompt,
        )

        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, RF_DATA_DIR=folder):
            random.seed(123)
            first = build_dynamic_prompt(seed=0, artists="none", prompt_enhancer="none")
            state = random.getstate()
            random.seed(456)
            second = build_dynamic_prompt(seed=0, artists="none", prompt_enhancer="none")
            self.assertEqual(first, second)
            self.assertEqual(state, random.getstate())


if __name__ == "__main__":
    unittest.main()
