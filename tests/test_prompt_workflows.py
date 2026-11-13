"""Prompt tools use reviewed text and cached tokenizers without image work."""

import ast
from concurrent.futures import CancelledError, ThreadPoolExecutor
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr
import ruinedfooocus.app

from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.prompting import prompt_preview, token_variation, ui_evolve
from ruinedfooocus.features.prompting.prompt_preview import (
    apply_preview,
    undo_preview,
    prompt_result,
)
from ruinedfooocus.features.prompting.ui_rewrite import safe_rewrite
from ruinedfooocus.features.prompting.prompt_enhancement import image_prompt_result

# Construct real Gradio controls, but do not start application model discovery.
_shared = ModuleType("ruinedfooocus.app.state")
_shared.settings = SimpleNamespace(default_settings={"OBP_preset": "Standard"})
_shared.add_ctrl = Mock()
_spec = importlib.util.spec_from_file_location(
    "prompt_ui_under_test",
    Path(__file__).resolve().parents[1] / "ruinedfooocus/features/prompting/ui_random.py",
)
ui_random = importlib.util.module_from_spec(_spec)
with (
    patch.dict(sys.modules, {"ruinedfooocus.app.state": _shared}),
    patch.object(ruinedfooocus.app, "state", _shared, create=True),
):
    _spec.loader.exec_module(ui_random)


class TinyTokenizer:
    all_special_tokens = ["<special>"]
    all_special_ids = [0]

    def get_vocab(self):
        return {"<special>": 0, "piece": 1, "cup": 2, "42": 3, "<lora:bad>": 4}

    def tokenize(self, word):
        # Distinct pieces prove that this is token mutation, not word lookup.
        return ["first", "second"]

    def convert_tokens_to_string(self, pieces):
        return " ".join(pieces)


class PromptWorkflowTests(unittest.TestCase):
    def test_delayed_result_checks_draft_revision_and_requires_explicit_apply(self):
        ticket = ("original", 4, "built", False)
        self.assertEqual(prompt_result("original", 4, ticket), ("built", ("original", "built")))
        self.assertIsNone(prompt_result("edited", 5, ticket))
        # Editing then restoring the same text is still a different draft.
        self.assertIsNone(prompt_result("original", 6, ticket))
        self.assertEqual(
            prompt_result("edited", 5, ticket, explicit=True),
            ("built", ("edited", "built")),
        )
        with self.assertRaises(ValueError):
            prompt_result("original", 4, None)

    def test_rewrite_removes_invented_resources_and_protects_lettering_and_original_syntax(self):
        original = 'two cups "CAFÉ 42" <lora:real:0.5> embedding:real __subject__'
        rewritten = safe_rewrite(
            original, 'two cups "CAFÉ 42", soft light <lora:invented:1> embedding:fake'
        )
        for literal in ('"CAFÉ 42"', "<lora:real:0.5>", "embedding:real", "__subject__"):
            self.assertIn(literal, rewritten)
        self.assertNotIn("invented", rewritten)
        self.assertNotIn("embedding:fake", rewritten)
        with self.assertRaisesRegex(gr.Error, "quoted text"):
            safe_rewrite(original, 'two cups "CAFE 42"')
        weighted = safe_rewrite(
            "cup (embedding:real:1.2)", "cup, soft light (embedding:invented:0.7)"
        )
        self.assertIn("(embedding:real:1.2)", weighted)
        self.assertNotIn("invented", weighted)

    def test_preset_optional_subject_override_preserves_seed_zero_and_blank_behavior(self):
        self.addCleanup(random.setstate, random.getstate())
        values = dict(OBP_preset="Standard", seed=0, prompt_enhancer="none")
        first = ui_random.build_dynamic_prompt(**values)
        self.assertEqual(first, ui_random.build_dynamic_prompt(**values, subject_override=""))
        self.assertEqual(first, ui_random.build_dynamic_prompt(**values, subject_override=None))
        subject = "copper teapot"
        overridden = ui_random.build_dynamic_prompt(**values, subject_override=subject)
        self.assertIn(subject, overridden)
        self.assertEqual(
            overridden, ui_random.build_dynamic_prompt(**values, subject_override=subject)
        )

    def test_rewrite_distinguishes_possessives_and_contractions_from_quoted_lettering(self):
        original = "a woman's face beside a child's toy; she isn't smiling"
        proposed = "a woman beside a child's toy, looking serious"
        self.assertEqual(safe_rewrite(original, proposed), proposed)
        for quoted in ('"CAFÉ 42"', "'CAFÉ 42'", "'it isn't blue'", "“CAFÉ 42”", "«CAFÉ 42»"):
            with self.subTest(quoted=quoted):
                source = original + " with a sign saying " + quoted
                result = proposed + " and a sign saying " + quoted
                self.assertEqual(safe_rewrite(source, result), result)
                with self.assertRaisesRegex(gr.Error, "quoted text"):
                    safe_rewrite(source, proposed + " and a sign saying changed lettering")

    def test_preview_append_and_undo_preserve_exact_text_and_reject_stale_edits(self):
        source = '  Alice 42 "CAFÉ" <lora:name:0.5>\n'
        result, saved = apply_preview(source, source, "next", append=True)
        self.assertEqual(result, source + "\n---\nnext")
        self.assertEqual(undo_preview(result, saved), source)
        for current, original, proposed in (("edited", source, "next"), (source, source, " ")):
            with self.assertRaises(ValueError):
                apply_preview(current, original, proposed)
        with self.assertRaises(ValueError):
            undo_preview(result + "typed", saved)
        with self.assertRaises(ValueError):
            undo_preview(source, None)

    def test_quick_policy_does_not_read_configurable_controls_or_call_chat(self):
        with (
            patch.object(ui_random, "format_for", return_value="SDXL") as format_for,
            patch.object(ui_random, "build_dynamic_prompt", return_value="quick") as build,
            patch.object(ui_random, "finalize", side_effect=lambda text, fmt: text),
        ):
            self.assertEqual(ui_random.quick_prompt("my subject", "selected", 123), "quick")
        format_for.assert_called_once_with(ui_random.AUTO, "selected")
        self.assertEqual(
            build.call_args.kwargs,
            dict(
                insanitylevel=5,
                forcesubject="all",
                artists="all",
                imagetype="all",
                givensubject="my subject",
                smartsubject=True,
                seed=123,
                base_model="SDXL",
                OBP_preset="",
                prompt_enhancer="none",
            ),
        )
        for seed in (None, True, -2, 1.5):
            with self.assertRaises(ValueError):
                ui_random.quick_prompt(seed=seed)

    def test_nine_candidate_indexed_evolution_preserves_original_parent_and_batch_count(self):
        text = "  first\n---\nsecond  "
        initial = ui_evolve.new_session(text)
        with patch.object(ui_evolve, "randomize_words", return_value="variant"):
            session = ui_evolve.evolve_session(text, initial, 1, "Words", 10, 9)
        self.assertEqual(session["original"], text)
        self.assertEqual(session["parents"], ["  first", "second  "])
        self.assertEqual(session["variants"], ["second  "] + ["variant"] * 8)
        self.assertEqual(initial["variants"], [])
        batch, images = ui_evolve.generation_batch(text, session)
        self.assertEqual(images, 1)
        self.assertEqual(batch.split("\n---\n"), session["variants"])
        self.assertEqual(ui_evolve.prompt_variants("first", "Words", 10, 1), ["first"])
        for count in (0, 10, 1.5, True):
            with self.assertRaises(ValueError):
                ui_evolve.prompt_variants("parent", "Words", 10, count)
        with self.assertRaises(ValueError):
            ui_evolve.generation_batch("edited", session)

    def test_token_mutation_uses_real_pieces_and_protects_nested_syntax_and_literals(self):
        prompt = 'cup  42 "cup (cup)" <special> <lora:cup:1> __folder/cup__ ((cup:1.2):1.3) [cup|[cup]] {cup} AND BREAK cup'
        rng = Mock(random=Mock(return_value=0), choice=Mock(return_value="piece"))
        output = token_variation.randomize_tokens(prompt, 100, TinyTokenizer(), rng=rng)
        self.assertEqual(
            output, prompt.replace("cup  42", "piece piece  42").removesuffix("cup") + "piece piece"
        )
        self.assertEqual(rng.choice.call_count, 4)
        for call in rng.choice.call_args_list:
            self.assertEqual(call.args[0], ["piece", "cup"])
        self.assertEqual(token_variation.randomize_tokens(prompt, 0, TinyTokenizer()), prompt)

    def test_words_also_protect_nested_and_quoted_groups(self):
        prompt = 'cup "cup (cup)" ((cup:1.2):1.3) [cup|[cup]] cup'
        with patch.object(ui_evolve, "wildcard_word_index", return_value=(("bowl",), {"cup"})):
            self.assertEqual(
                ui_evolve.randomize_words(prompt, 100, rng=random.Random(1)),
                'bowl "cup (cup)" ((cup:1.2):1.3) [cup|[cup]] bowl',
            )

    def test_multiline_and_escaped_quoted_lettering_is_preserved(self):
        prompt = 'cup "cup\\"cup\ncup (cup)" cup'
        with patch.object(ui_evolve, "wildcard_word_index", return_value=(("bowl",), {"cup"})):
            self.assertEqual(
                ui_evolve.randomize_words(prompt, 100, rng=random.Random(1)),
                'bowl "cup\\"cup\ncup (cup)" bowl',
            )

    def test_quoted_group_delimiters_do_not_expose_protected_group_words(self):
        prompt = 'cup (cup "cup) cup" cup) [cup "[cup" cup] cup'
        with patch.object(ui_evolve, "wildcard_word_index", return_value=(("bowl",), {"cup"})):
            self.assertEqual(
                ui_evolve.randomize_words(prompt, 100, rng=random.Random(1)),
                'bowl (cup "cup) cup" cup) [cup "[cup" cup] bowl',
            )
        tokenizer = TinyTokenizer()
        rng = Mock(random=Mock(return_value=0), choice=Mock(return_value="piece"))
        self.assertEqual(
            token_variation.randomize_tokens(prompt, 100, tokenizer, rng=rng),
            'piece piece (cup "cup) cup" cup) [cup "[cup" cup] piece piece',
        )

    def test_token_vocabulary_is_built_only_once_for_nine_candidate_preview(self):
        tokenizer = TinyTokenizer()
        with patch.object(tokenizer, "get_vocab", wraps=tokenizer.get_vocab) as vocabulary:
            variants = ui_evolve.prompt_variants("cup", "Tokens", 100, 9, tokenizer=tokenizer)
        self.assertEqual(len(variants), 9)
        vocabulary.assert_called_once()

    def test_real_pinned_clip_tokenizer_mutates_tokens_without_loading_encoder(self):
        from transformers import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(
            Path(__file__).resolve().parents[1] / "repositories/ComfyUI/comfy/sd1_tokenizer",
            local_files_only=True,
        )
        prompt = 'cup 42 "CAFÉ" <lora:cup:1> __folder/cup__ ((cup:1.2):1.3) AND BREAK'
        first = token_variation.randomize_tokens(prompt, 100, tokenizer, rng=random.Random(42))
        second = token_variation.randomize_tokens(prompt, 100, tokenizer, rng=random.Random(42))
        self.assertEqual(first, second)
        self.assertNotEqual(first.split(" 42")[0], "cup")
        self.assertEqual(first[first.index(" 42") :], prompt[prompt.index(" 42") :])

    def test_tokenizer_is_only_taken_from_selected_unchanged_cached_single_stream(self):
        tokenizer = TinyTokenizer()
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "selected.safetensors"
            checkpoint.write_bytes(b"private fake model identity")
            pipeline = SimpleNamespace(
                base_file_signature=file_signature(checkpoint),
                component_file_signatures=[],
                component_settings={},
                xl_base=SimpleNamespace(
                    clip=SimpleNamespace(
                        tokenizer=SimpleNamespace(
                            clip="clip_l", clip_l=SimpleNamespace(tokenizer=tokenizer)
                        )
                    )
                ),
            )
            models = Mock()
            models.get_file.return_value = checkpoint
            shared = SimpleNamespace(
                state={"pipeline": pipeline},
                models=models,
                settings=SimpleNamespace(default_settings={}),
            )
            with patch.dict(sys.modules, {"ruinedfooocus.app.state": shared}):
                self.assertIs(token_variation.cached_tokenizer("selected"), tokenizer)
                models.get_file.assert_called_once_with("checkpoints", "selected")
                pipeline.xl_base.clip.tokenizer = SimpleNamespace(
                    clip_l=tokenizer, clip_g=tokenizer
                )
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    token_variation.cached_tokenizer("selected")
                shared.settings.default_settings["clip_l"] = "changed.safetensors"
                with self.assertRaisesRegex(ValueError, "overrides changed"):
                    token_variation.cached_tokenizer("selected")
                shared.settings.default_settings.clear()
                checkpoint.write_bytes(b"changed private fake identity")
                with self.assertRaisesRegex(ValueError, "does not match"):
                    token_variation.cached_tokenizer("selected")
                shared.state["pipeline"] = None
                with self.assertRaisesRegex(ValueError, "No model is loaded"):
                    token_variation.cached_tokenizer("selected")

    def test_enhancement_keeps_original_literals_numbers_names_and_syntax(self):
        original = 'Alice 42 "CAFÉ" <lora:portrait:0.5> __subject__ ((cup:1.1):1.2)'

        def result(addition):
            return image_prompt_result(
                {"finish_reason": "stop", "message": {"content": addition}}, original
            )

        self.assertEqual(
            result('{"addition":"soft diffuse light"}'), original + "\nsoft diffuse light"
        )
        for addition in (
            '{"addition":"add 3 cups"}',
            '{"addition":"<lora:new:1>"}',
            '{"addition":"__change__"}',
            '{"addition":"replace Alice"}',
            '{"addition":"sign \\"CAFE\\""}',
            '{"addition":null}',
        ):
            self.assertEqual(result(addition), original)

    def test_enhancement_rejects_new_composition_and_embedding_syntax(self):
        original = 'Alice "CAFÉ"'
        for addition in (
            "soft light BREAK another scene",
            "soft light AND texture",
            "embedding:unknown",
            "soft|hard light",
        ):
            with self.subTest(addition=addition):
                self.assertEqual(
                    image_prompt_result(
                        {"finish_reason": "stop", "message": {"content": addition}}, original
                    ),
                    original,
                )


class PromptCallbackTests(unittest.TestCase):
    @staticmethod
    def publication(fn, *values):
        envelope = (
            fn.fn(*values)
            if len(values) == len(fn.inputs)
            else fn.fn(*values, {"revision": 0, "serial": 1})
        )
        return tuple(envelope["updates"])

    def test_localized_append_checkbox_defaults_to_replace_in_real_component_config(self):
        from gradio.i18n import I18nData

        with patch.object(
            prompt_preview, "translate_ui", side_effect=lambda text: I18nData("test." + text)
        ):
            with gr.Blocks() as app:
                prompt = gr.Textbox(value="original draft")
                bind = prompt_preview.create_prompt_action(prompt)
                bind(lambda: "generated draft", [])
        build = next(
            fn for fn in app.fns.values() if getattr(fn.fn, "__name__", None) == "build_prompt"
        )
        append = build.inputs[2]
        self.assertIsInstance(append, gr.Checkbox)
        properties = next(
            item["props"] for item in app.config["components"] if item["id"] == append._id
        )
        self.assertIs(properties["value"], False)
        self.assertEqual(properties["label"].key, "test.Append as another prompt")
        ticket = build.fn(*(component.value for component in build.inputs))
        publish = next(
            fn for fn in app.fns.values() if getattr(fn.fn, "__name__", None) == "publish_prompt"
        )
        result = self.publication(publish, prompt.value, build.inputs[1].value, ticket)
        self.assertEqual(result[:2], ("generated draft", ("original draft", "generated draft")))
        self.assertNotIn("---", result[0])

    @classmethod
    def setUpClass(cls):
        # Translate markers independently of whichever catalog update runs first.
        with (
            patch.object(ui_random, "translate_ui", side_effect=lambda text: text),
            patch.object(ui_evolve, "translate_ui", side_effect=lambda text: text),
            patch.object(prompt_preview, "translate_ui", side_effect=lambda text: text),
            patch.object(ui_random, "add_ctrl"),
        ):
            with gr.Blocks() as cls.demo:
                cls.prompt = gr.Textbox(value="original")
                cls.run_control = gr.Number(value=0)
                cls.images = gr.Number(value=4)
                cls.checkpoint = gr.Textbox(value="selected")
                bind_generation = ui_random.ui_onebutton(
                    cls.prompt, cls.run_control, cls.checkpoint, image_number=cls.images
                )
                # Match app ordering: image controls are constructed after Prompt tools.
                cls.image_replacement = gr.Checkbox(value=True)
                bind_generation(cls.image_replacement)

    def event(self, function_name):
        fn = next(
            fn for fn in self.demo.fns.values() if getattr(fn.fn, "__name__", None) == function_name
        )
        event = next(event for event in self.demo.config["dependencies"] if event["id"] == fn._id)
        return fn, event

    def test_prompt_builds_are_separate_and_cannot_admit_image_work(self):
        builds = [
            fn
            for fn in self.demo.fns.values()
            if getattr(fn.fn, "__name__", None) == "build_prompt"
        ]
        self.assertEqual(len(builds), 2)
        self.assertTrue(set(builds[0].outputs).isdisjoint(builds[1].outputs))
        for build in builds:
            self.assertNotIn(self.prompt, build.outputs)
            self.assertNotIn(self.run_control, build.outputs)
            event = next(
                event for event in self.demo.config["dependencies"] if event["id"] == build._id
            )
            publish = next(
                candidate
                for candidate in self.demo.config["dependencies"]
                if candidate["trigger_after"] == event["id"]
            )
            self.assertTrue(publish["trigger_only_on_success"])
            self.assertNotIn(self.run_control._id, publish["outputs"])
        # Evolve is the sole deliberate image action in Prompt tools.
        admissions = [
            event
            for event in self.demo.config["dependencies"]
            if self.run_control._id in event["outputs"]
        ]
        self.assertEqual(len(admissions), 1)
        self.assertTrue(admissions[0]["trigger_only_on_success"])
        with patch.object(ui_random, "quick_prompt", return_value="quick") as quick:
            self.assertEqual(
                builds[0].fn("original", 0, False, "subject", 0, "selected", {}),
                ["original", 0, "quick", False, {}],
            )
        quick.assert_called_once_with("subject", "selected", 0)

    def test_late_publish_apply_append_undo_and_dismiss_preserve_drafts(self):
        publish, event = self.event("publish_prompt")
        self.assertNotIn(self.prompt._id, event["outputs"])
        client = next(
            candidate
            for candidate in self.demo.config["dependencies"]
            if candidate["trigger_after"] == event["id"]
        )
        self.assertIn(self.image_replacement._id, client["outputs"])
        self.assertIn(self.prompt._id, client["outputs"])
        self.assertFalse(client["backend_fn"])
        ticket = ("original", 0, "reviewed", True)
        result = self.publication(publish, "original", 0, ticket)
        self.assertEqual(
            result[:2], ("original\n---\nreviewed", ("original", "original\n---\nreviewed"))
        )
        self.assertEqual(result[-2:], (False, False))
        stale = self.publication(publish, "edited", 1, ticket)
        self.assertEqual(stale[0], gr.skip())
        self.assertEqual(stale[2], "reviewed")
        self.assertTrue(stale[3]["visible"])
        self.assertEqual(stale[-2:], (gr.skip(), gr.skip()))
        apply, _ = self.event("apply_result")
        applied = self.publication(apply, "edited", 1, ticket)
        self.assertEqual(
            applied[:2], ("edited\n---\nreviewed", ("edited", "edited\n---\nreviewed"))
        )
        undo, _ = self.event("undo_prompt")
        self.assertEqual(self.publication(undo, applied[0], applied[1]), ("edited", None))
        with self.assertRaises(gr.Error):
            self.publication(undo, "typed later", applied[1])
        # A dismissed/missing result cannot later be applied.
        with self.assertRaises(gr.Error):
            self.publication(apply, "edited", 1, None)

    def test_prompt_build_failure_leaves_draft_and_randomization_controls_untouched(self):
        build, _ = self.event("build_prompt")
        for error in (RuntimeError("failed"), CancelledError("cancelled")):
            with patch.object(ui_random, "quick_prompt", side_effect=error):
                with self.assertRaises(type(error)):
                    build.fn("original", 0, False, "", 0, "selected", {})
        with patch.object(ui_random, "quick_prompt", return_value=" "):
            with self.assertRaises(gr.Error):
                build.fn("original", 0, False, "", 0, "selected", {})

    def test_build_barrier_cannot_overwrite_a_draft_edited_while_work_is_running(self):
        build, _ = self.event("build_prompt")
        publish, _ = self.event("publish_prompt")
        started, finish = threading.Event(), threading.Event()

        def blocked_build(*args):
            started.set()
            if not finish.wait(5):
                raise TimeoutError("test build barrier was not released")
            return "built after edit"

        with patch.object(ui_random, "quick_prompt", side_effect=blocked_build):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(build.fn, "original", 0, False, "", 0, "selected", {})
                try:
                    self.assertTrue(started.wait(5))
                    current, revision = "user's new draft", 1
                finally:
                    finish.set()
                ticket = future.result(timeout=5)
        result = self.publication(publish, current, revision, ticket)
        self.assertEqual(result[0], gr.skip())
        self.assertEqual(result[2], "built after edit")
        self.assertTrue(result[3]["visible"])

    def test_primary_actions_and_presets_are_available_without_advanced_controls(self):
        buttons = [
            component for component in self.demo.blocks.values() if isinstance(component, gr.Button)
        ]
        self.assertEqual(sum(button.value == "Generate prompt" for button in buttons), 2)
        action_sizes = {
            button.size for button in buttons if button.value in ("Generate prompt", "Undo")
        }
        self.assertEqual(len(action_sizes), 1)
        self.assertFalse(any("instant" in str(button.value) for button in buttons))
        configured = next(
            component
            for component in self.demo.blocks.values()
            if isinstance(component, gr.Accordion)
            and component.label == "One Button Prompt — configurable"
        )
        self.assertTrue(configured.open)
        advanced = next(
            component
            for component in self.demo.blocks.values()
            if isinstance(component, gr.Accordion) and component.label == "Advanced Custom options"
        )
        self.assertFalse(advanced.open)

    def test_evolve_generate_disables_replacement_sets_one_image_per_candidate(self):
        fn, event = self.event("generate_reviewed")
        session = ui_evolve.new_session("original")
        session["variants"] = ["original", "second"]
        self.assertEqual(
            self.publication(fn, "original", session, 0),
            ("original\n---\nsecond", 1, False, ("original", "original\n---\nsecond"), False, 1),
        )
        client = next(
            candidate
            for candidate in self.demo.config["dependencies"]
            if candidate["trigger_after"] == event["id"]
        )
        self.assertIn(self.images._id, client["outputs"])
        with self.assertRaises(gr.Error):
            self.publication(fn, "typed later", session, 0)

    def test_onebutton_only_uses_explicit_enhancement_and_optional_subject(self):
        fn = [
            fn
            for fn in self.demo.fns.values()
            if getattr(fn.fn, "__name__", None) == "build_prompt"
        ][1]
        values = [
            5,
            "all",
            "all",
            "all",
            "",
            "",
            "",
            "",
            True,
            "",
            20,
            "all",
            "all",
            "all",
            "all",
            "",
            "Standard",
            "none",
            "SDXL",
            "selected",
        ]
        with (
            patch.object(ui_random, "build_dynamic_prompt", return_value="new") as build,
            patch.object(ui_random, "format_for", return_value="SDXL"),
        ):
            self.assertEqual(
                fn.fn("original", 0, False, *values, "copper teapot", {}),
                ["original", 0, "new", False, {}],
            )
        self.assertEqual(build.call_args.args[25], "none")
        self.assertEqual(build.call_args.kwargs["subject_override"], "copper teapot")
        values[17] = "llama"
        with (
            patch.object(ui_random, "build_dynamic_prompt", return_value="enhanced") as build,
            patch.object(ui_random, "format_for", return_value="SDXL"),
        ):
            fn.fn("original", 0, False, *values, "", {})
        self.assertEqual(build.call_args.args[25], "llama")

    def test_configurable_onebutton_advances_randomness_between_previews(self):
        preview = [
            fn
            for fn in self.demo.fns.values()
            if getattr(fn.fn, "__name__", None) == "build_prompt"
        ][1]
        self.addCleanup(random.setstate, random.getstate())
        values = [
            5,
            "all",
            "none",
            "all",
            "",
            "",
            "",
            "",
            True,
            "",
            20,
            "all",
            "all",
            "all",
            "all",
            "",
            "",
            "none",
            "SDXL",
            "selected",
        ]
        with patch.object(ui_random, "format_for", return_value="SDXL"):
            random.seed(73)
            source, _, first, _, _ = preview.fn("original", 0, False, *values, "", {})
            first_state = random.getstate()
            _, _, second, _, _ = preview.fn("original", 0, False, *values, "", {})
        self.assertEqual(source, "original")
        self.assertNotEqual(first, second)
        self.assertNotEqual(first_state, random.getstate())

    def test_reviewed_image_conditioned_prompt_reaches_fake_worker_with_switches_initially_true(
        self,
    ):
        root = Path(__file__).resolve().parents[1]

        def load_function(relative, name, scope):
            tree = ast.parse((root / relative).read_text(encoding="utf-8"))
            node = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            exec(compile(ast.Module(body=[node], type_ignores=[]), relative, "exec"), scope)
            return scope[name]

        processing = "ruinedfooocus/features/prompting/prompt_processing.py"
        scope = {}
        scope["use_random_prompt"] = load_function(processing, "use_random_prompt", {})
        scope["get_promptlist"] = load_function(processing, "get_promptlist", {})
        scope["worker"] = Mock(add_task=Mock(return_value="private-task"))
        scope["ui_controlnet"] = SimpleNamespace(tool_kind=lambda *_: None)
        append_work = load_function("ruinedfooocus/ui/app.py", "append_work", scope)
        data = dict(
            prompt="original",
            image_number=1,
            input_image=object(),
            cn_selection="img2img",
            obp_assume_direct_control=True,
            obp_image_prompt=True,
            negative='keep "negative"',
            loras=[("private-lora", 0.7)],
            seed=0,
            style_selection=["Custom"],
        )
        self.assertTrue(scope["use_random_prompt"](data, data["prompt"]))
        publish, _ = self.event("publish_prompt")
        result = self.publication(
            publish, data["prompt"], 0, (data["prompt"], 0, 'reviewed "CAFÉ 42"', False)
        )
        reviewed = result[0]
        data["obp_assume_direct_control"], data["obp_image_prompt"] = result[-2:]
        data["prompt"] = reviewed
        self.assertEqual(append_work(data), "private-task")
        submitted = scope["worker"].add_task.call_args.args[0]
        self.assertEqual(submitted["prompt"], 'reviewed "CAFÉ 42"')
        for name in ("negative", "loras", "seed", "style_selection", "input_image"):
            self.assertEqual(submitted[name], data[name])
        self.assertEqual(submitted["image_total"], 1)


if __name__ == "__main__":
    unittest.main()
