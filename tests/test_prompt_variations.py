"""Deliberate variation keeps syntax and has no image-generation side effects."""

import random
import unittest
from unittest.mock import patch

import gradio as gr

from ruinedfooocus.features.prompting import ui_evolve
from ui_test_helpers import raise_refusal


class PromptVariationTests(unittest.TestCase):
    def test_word_replacement_keeps_directives_labels_numbers_and_spacing(self):
        prompt = 'cup  42 "cup" <lora:cup:0.5> __cup__ [cup|bowl] (cup:1.2) cup'
        with patch.object(
            ui_evolve, "wildcard_word_index", return_value=(("bowl",), frozenset(("cup",)))
        ):
            self.assertEqual(
                ui_evolve.randomize_words(prompt, 100, rng=random.Random(1)),
                'bowl  42 "cup" <lora:cup:0.5> __cup__ [cup|bowl] (cup:1.2) bowl',
            )
            self.assertEqual(ui_evolve.randomize_words(prompt, 0), prompt)

    def test_preview_retains_exact_original_and_selection_rejects_stale_values(self):
        original = "  cup\n"
        with patch.object(ui_evolve, "randomize_words", return_value="bowl"):
            variants = ui_evolve.prompt_variants(original, "Words", 10)
        self.assertEqual(variants, [original, "bowl", "bowl", "bowl", "bowl"])
        self.assertEqual(ui_evolve.select_variant(variants, 0), original)
        self.assertEqual(ui_evolve.select_variant(variants, 4), "bowl")
        for index in (None, -1, 5):
            with self.assertRaises(ValueError):
                ui_evolve.select_variant(variants, index)
        with self.assertRaises(ValueError):
            ui_evolve.prompt_variants(" ", "Words", 10)
        with self.assertRaises(ValueError):
            ui_evolve.prompt_variants(original, "Tokens", 10)

    def test_one_button_preview_reuses_existing_variant_engine(self):
        with patch(
            "ruinedfooocus.features.prompting.one_button.build_dynamic_prompt.createpromptvariant",
            return_value="creative variant",
        ) as create:
            self.assertEqual(
                ui_evolve.prompt_variants("original", "One Button", 50),
                ["original"] + ["creative variant"] * 4,
            )
        self.assertEqual(create.call_count, 4)
        create.assert_called_with("original", 5, advancedprompting=False)

    def test_preview_only_ui_has_no_generation_event(self):
        with gr.Blocks() as demo:
            prompt = gr.Textbox(value="original")
            image_count = gr.Number(value=3)
            run_event = gr.Number(value=0)
            ui_evolve.add_evolve_controls(prompt)
        dependencies = demo.config["dependencies"]
        prompt_writers = [event for event in dependencies if prompt._id in event["outputs"]]
        self.assertEqual(len(prompt_writers), 3)  # Apply, copy parent and guarded undo.
        self.assertTrue(
            all(
                image_count._id not in event["outputs"] and run_event._id not in event["outputs"]
                for event in dependencies
            )
        )
        self.assertFalse(
            any(
                component.get("props", {}).get("label") == "Tokens"
                for component in demo.config["components"]
            )
        )

    def test_parent_loading_and_promotion_do_not_dispatch_reset_selection(self):
        with (
            patch.object(ui_evolve, "translate_ui", side_effect=lambda text: text),
            gr.Blocks() as demo,
        ):
            prompt = gr.Textbox(value="LEFT---RIGHT")
            ui_evolve.add_evolve_controls(prompt)
        callbacks = {getattr(fn.fn, "__name__", ""): fn for fn in demo.fns.values()}
        candidate = next(
            component
            for component in demo.blocks.values()
            if getattr(component, "label", None) == "Candidate index"
        )
        parent = next(
            component
            for component in demo.blocks.values()
            if getattr(component, "label", None) == "Parent index"
        )
        loaded = callbacks["load"].fn("LEFT---RIGHT")
        self.assertEqual(loaded[0]["parents"], ["LEFT---RIGHT"])
        self.assertEqual(loaded[0]["variants"], [])
        self.assertIsNone(loaded[4]["value"])
        self.assertEqual(loaded[5], "")
        self.assertEqual(callbacks["pick"].targets, [(candidate._id, "input")])
        self.assertEqual(callbacks["pick_parent"].targets, [(parent._id, "input")])
        # Gradio change handlers run for callback-produced values; input handlers
        # do not. Both resets already return their correct preview text directly.
        for component in (candidate, parent):
            self.assertFalse(
                any((component._id, "change") in fn.targets for fn in demo.fns.values())
            )
        with patch.object(ui_evolve, "randomize_words", return_value="new candidate"):
            previewed = callbacks["preview_variants"].fn(
                "LEFT---RIGHT", loaded[0], 0, "Words", 10, 3
            )
        promoted = callbacks["parents"].fn("LEFT---RIGHT", previewed[0])
        self.assertEqual(promoted[0]["parents"], ["LEFT---RIGHT", "new candidate", "new candidate"])
        self.assertEqual(promoted[0]["variants"], [])
        self.assertIsNone(promoted[3]["value"])
        self.assertEqual(promoted[4], "")
        self.assertEqual(callbacks["pick_parent"].fn(promoted[0], 0), "LEFT---RIGHT")

    def test_user_selection_and_apply_still_reject_missing_invalid_or_stale_candidates(self):
        with (
            patch.object(ui_evolve, "translate_ui", side_effect=lambda text: text),
            gr.Blocks() as demo,
        ):
            prompt = gr.Textbox(value="original")
            ui_evolve.add_evolve_controls(prompt)
        callbacks = {getattr(fn.fn, "__name__", ""): fn.fn for fn in demo.fns.values()}
        session = callbacks["load"]("original")[0]
        with patch.object(ui_evolve, "randomize_words", return_value="variant"):
            session = callbacks["preview_variants"]("original", session, 0, "Words", 10, 3)[0]
        self.assertEqual(callbacks["pick"](session, 1), "variant")
        for callback, args in (
            ("pick", (session, None)),
            ("pick", (session, 99)),
            ("pick_parent", (None, 0)),
            ("pick_parent", (session, 99)),
            ("apply", ("original", session, None)),
            ("apply", ("edited draft", session, 1)),
        ):
            with self.subTest(callback=callback, args=args), self.assertRaises(gr.Error):
                raise_refusal(callbacks[callback](*args, *([{}] if callback == "apply" else [])))


if __name__ == "__main__":
    unittest.main()
