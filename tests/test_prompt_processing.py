"""Batch submission preserves lettering and shares its delimiter with producers."""

import ast
import json
from pathlib import Path
import unittest

from ruinedfooocus.features.prompting.prompt_batches import split_prompt_batch
from ruinedfooocus.features.prompting.prompt_preview import apply_preview, undo_preview
from ruinedfooocus.features.prompting import ui_evolve

ROOT = Path(__file__).resolve().parents[1]


class PromptBatchTests(unittest.TestCase):
    def submit(self, prompt, **options):
        source = ROOT / "ruinedfooocus/features/prompting/prompt_processing.py"
        node = next(
            node
            for node in ast.parse(source.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef) and node.name == "get_promptlist"
        )
        scope = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        return scope["get_promptlist"](dict(prompt=prompt, **options))

    def test_inline_and_escaped_triple_hyphens_keep_one_exact_prompt(self):
        for text in (
            'A sign reading "LEFT---RIGHT" with exact lettering.',
            "A sign reading 'LEFT---RIGHT'.",
            "A sign reading “LEFT---RIGHT”.",
            "A sign reading «LEFT---RIGHT».",
            r"Keep LEFT\---RIGHT exactly.",
            "  A literal line:\n\\---\nfin café  ",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.submit(text), [text])

    def test_standalone_delimiters_split_only_the_intentional_batch(self):
        self.assertEqual(self.submit(" first \n---\n second "), [" first ", " second "])
        self.assertEqual(self.submit("first\r\n \t--- \t\r\nsecond"), ["first", "second"])
        self.assertEqual(self.submit("first---second"), ["first---second"])

    def test_multiline_quoted_lettering_keeps_its_standalone_hyphens(self):
        for quote, close in ((chr(34), chr(34)), ("'", "'"), ("“", "”"), ("«", "»")):
            text = f"A sign reading {quote}LEFT\n---\nRIGHT{close}."
            with self.subTest(quote=quote):
                self.assertEqual(self.submit(text + "\n---\nnext"), [text, "next"])

    def test_saved_metadata_bypasses_batch_parsing(self):
        prompt = json.dumps({"Prompt": "first\n---\nsecond", "Negative": ""})
        self.assertEqual(self.submit(prompt), [prompt])
        text = "processed\n---\nprompt"
        self.assertEqual(self.submit(text, _metadata_prompt=True), [text])

    def test_actual_preview_append_producer_round_trips_exact_prompts_and_undo(self):
        original = '  A sign says "LEFT---RIGHT".\n'
        proposed = "next café  "
        joined, receipt = apply_preview(original, original, proposed, append=True)
        self.assertEqual(self.submit(joined), [original, proposed])
        self.assertEqual(undo_preview(joined, receipt), original)
        joined, _ = apply_preview("first\n---\nsecond", "first\n---\nsecond", proposed, append=True)
        self.assertEqual(self.submit(joined), ["first", "second", proposed])

    def test_batch_producers_reject_quotes_that_would_silently_merge_two_prompts(self):
        original, proposed = 'left "', 'right "'
        with self.assertRaisesRegex(ValueError, "quote"):
            apply_preview(original, original, proposed, append=True)
        session = ui_evolve.new_session(original)
        session["variants"] = [original, proposed]
        with self.assertRaisesRegex(ValueError, "quote"):
            ui_evolve.generation_batch(original, session)

    def test_actual_evolve_producer_keeps_literal_hyphens_and_rejects_extra_requests(self):
        original = 'A sign says "LEFT---RIGHT".'
        session = ui_evolve.new_session(original)
        self.assertEqual(session["parents"], [original])
        session["variants"] = [original, 'A sign says "LEFT\n---\nRIGHT".']
        batch, count = ui_evolve.generation_batch(original, session)
        self.assertEqual(count, 1)
        self.assertEqual(self.submit(batch), session["variants"])
        session["variants"] = ["first\n---\nsecond"]
        with self.assertRaisesRegex(ValueError, "separator"):
            ui_evolve.generation_batch(original, session)

    def test_literal_prose_apostrophes_do_not_hide_a_delimiter(self):
        self.assertEqual(
            split_prompt_batch("Alice's cat\n---\nBob's dog"), ["Alice's cat", "Bob's dog"]
        )


if __name__ == "__main__":
    unittest.main()
