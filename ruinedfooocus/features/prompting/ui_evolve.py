"""Indexed prompt evolution with previews and a separate explicit batch action."""

import random

import gradio as gr
from ruinedfooocus.features.prompting.ui_draft_fence import fence_event

from ruinedfooocus.core.translation_manager import translate_ui
from ruinedfooocus.features.prompting.wildcards import wildcard_word_index
from ruinedfooocus.features.prompting.prompt_batches import join_prompt_batch, split_prompt_batch
from ruinedfooocus.features.prompting.prompt_preview import (
    apply_preview,
    bind_explicit_generation,
    undo_preview,
)
from ruinedfooocus.features.prompting.token_variation import (
    cached_tokenizer,
    map_prompt_words,
    randomize_tokens,
    safe_token_vocabulary,
)


def randomize_words(prompt, strength, *, rng=None):
    rng = rng or random
    words, vocabulary = wildcard_word_index()

    def replace(match):
        word = match.group()
        if (
            words
            and word not in ("AND", "BREAK")
            and not word.isdigit()
            and word.lower() in vocabulary
            and rng.random() < strength / 100
        ):
            return rng.choice(words)
        return word

    return map_prompt_words(prompt, replace)


def prompt_variants(prompt, mode, strength, count=5, *, tokenizer=None):
    """Return an exact parent plus count-1 alternatives; the UI supports up to nine."""
    if not prompt.strip():
        raise ValueError("Type a prompt before previewing variations.")
    if isinstance(count, bool) or int(count) != count or not 1 <= count <= 9:
        raise ValueError("Choose between 1 and 9 candidates, including the parent.")
    if not 0 <= float(strength) <= 100:
        raise ValueError("Variation strength must be between 0 and 100.")
    if mode == "Words":
        create = lambda: randomize_words(prompt, strength)
    elif mode == "Tokens":
        if tokenizer is None:
            raise ValueError(
                "Tokens needs a compatible cached tokenizer for the selected image model. No encoder is loaded by this tool."
            )
        candidates = safe_token_vocabulary(tokenizer) if strength and count > 1 else None
        create = lambda: randomize_tokens(prompt, strength, tokenizer, candidates=candidates)
    elif mode == "One Button":
        from ruinedfooocus.features.prompting.one_button.build_dynamic_prompt import (
            createpromptvariant,
        )

        create = lambda: createpromptvariant(
            prompt, max(int(strength / 10), 3), advancedprompting=False
        )
    else:
        raise ValueError("Choose Words, Tokens or One Button variation.")
    return [prompt, *(create() for _ in range(int(count) - 1))]


def select_variant(variants, index):
    if (
        index is None
        or isinstance(index, bool)
        or int(index) != index
        or not variants
        or not 0 <= int(index) < len(variants)
    ):
        raise ValueError("Preview variations before selecting one.")
    return variants[int(index)]


def new_session(text):
    parents = split_prompt_batch(text)
    if not text.strip() or len(parents) > 9 or any(not value.strip() for value in parents):
        raise ValueError("Use 1–9 nonempty parent prompts, separated by a standalone --- line.")
    return {"original": text, "source": text, "parents": parents, "variants": [], "parent": 0}


def evolve_session(text, session, parent, mode, strength, count, *, tokenizer=None):
    if not session or text != session["source"]:
        session = new_session(text)
    session = dict(session)
    chosen = select_variant(session["parents"], parent)
    session["parent"] = int(parent)
    session["variants"] = prompt_variants(chosen, mode, strength, count, tokenizer=tokenizer)
    return session


def generation_batch(current, session):
    """Only this deliberate action prepares one image per reviewed candidate."""
    if not session or current != session["source"] or not session["variants"]:
        raise ValueError(
            "The prompt changed or no variants were previewed. Preview again before generating."
        )
    if any(len(split_prompt_batch(value)) != 1 for value in session["variants"]):
        raise ValueError("A candidate contains the --- batch separator. Edit it before generating.")
    return join_prompt_batch(session["variants"]), 1


def add_evolve_controls(
    prompt,
    run_event=None,
    image_number=None,
    checkpoint=None,
    replacement_control=None,
    *,
    bind_generation=bind_explicit_generation,
):
    with gr.Accordion(translate_ui("Prompt variations"), open=False, elem_id="evolve-controls"):
        gr.Markdown(
            translate_ui(
                "Choose an indexed parent, preview up to nine candidates, then apply one or explicitly generate the reviewed batch. The first candidate is the unchanged parent. Generate sets one image per candidate and turns off Replace typed prompt and the image tools' Generate random prompt switch."
            )
        )
        state = gr.State(None)
        previous = gr.JSON(None, visible="hidden")
        with gr.Row():
            parent = gr.Dropdown(
                [(f"Parent {i}", i - 1) for i in range(1, 10)],
                value=0,
                label=translate_ui("Parent index"),
                interactive=True,
                min_width=200,
            )
            load_parents = gr.Button(
                translate_ui("Use current prompts as parents"), size="sm", min_width=200
            )
            use_candidates = gr.Button(
                translate_ui("Use preview as parents"), size="sm", min_width=200
            )
        original = gr.Textbox(
            label=translate_ui("Original prompt (preserved)"), lines=2, interactive=False
        )
        parent_text = gr.Textbox(label=translate_ui("Selected parent"), lines=2, interactive=False)
        mode = gr.Dropdown(
            [
                (translate_ui("Words"), "Words"),
                (translate_ui("Tokens"), "Tokens"),
                (translate_ui("One Button"), "One Button"),
            ],
            value="Words",
            label=translate_ui("Variation method"),
        )
        gr.Markdown(
            translate_ui(
                "Tokens is opt-in and uses only the selected image model's compatible tokenizer already cached in memory. Unloaded, changed and multi-encoder models are unsupported; this tool never loads an encoder. Quoted text, numbers and prompt syntax are protected."
            )
        )
        strength = gr.Slider(0, 100, value=10, step=1, label=translate_ui("Variation strength %"))
        count = gr.Slider(
            1, 9, value=9, step=1, label=translate_ui("Candidate count (including parent)")
        )
        preview_button = gr.Button(translate_ui("Preview variations"), size="sm")
        selected = gr.Dropdown(choices=[], label=translate_ui("Candidate index"), interactive=True)
        preview = gr.Textbox(label=translate_ui("Variant preview"), lines=4, interactive=False)
        with gr.Row():
            apply_button = gr.Button(translate_ui("Apply candidate to prompt"), size="sm")
            copy_parent = gr.Button(translate_ui("Copy selected parent to prompt"), size="sm")
            undo_button = gr.Button(translate_ui("Undo prompt change"), size="sm")
            generate = gr.Button(
                translate_ui("Generate reviewed candidates"),
                size="sm",
                visible=run_event is not None and image_number is not None,
            )

        def checked(operation, *args, **kwargs):
            try:
                return operation(*args, **kwargs)
            except ValueError as error:
                raise gr.Error(str(error)) from error

        def load(text):
            session = checked(new_session, text)
            return session, 0, text, session["parents"][0], gr.update(choices=[], value=None), ""

        def parents(current, session):
            if not session or current != session["source"] or not session["variants"]:
                raise gr.Error("Preview variations for the current prompt first.")
            session = {**session, "parents": list(session["variants"]), "parent": 0, "variants": []}
            return session, 0, session["parents"][0], gr.update(choices=[], value=None), ""

        def preview_variants(
            text, session, index, method, chance, number, selected_checkpoint=None
        ):
            tokenizer = (
                checked(cached_tokenizer, selected_checkpoint) if method == "Tokens" else None
            )
            session = checked(
                evolve_session, text, session, index, method, chance, number, tokenizer=tokenizer
            )
            choices = [
                (f"{i + 1}: " + ("Unchanged parent" if i == 0 else f"Variant {i}"), i)
                for i in range(len(session["variants"]))
            ]
            return (
                session,
                gr.update(choices=choices, value=0),
                session["variants"][0],
                session["original"],
                session["parents"][session["parent"]],
            )

        def pick(session, index):
            return checked(select_variant, session["variants"] if session else [], index)

        def pick_parent(session, index):
            return checked(select_variant, session["parents"] if session else [], index)

        def apply(current, session, index, parent_only=False):
            if not session:
                raise gr.Error("Preview or load parent prompts first.")
            values = session["parents"] if parent_only else session["variants"]
            proposed = checked(select_variant, values, index)
            return checked(apply_preview, current, session["source"], proposed)

        def undo(current, saved):
            return checked(undo_preview, current, saved), None

        load_parents.click(
            load,
            inputs=prompt,
            outputs=[state, parent, original, parent_text, selected, preview],
            api_visibility="undocumented",
            queue=False,
        )
        use_candidates.click(
            parents,
            inputs=[prompt, state],
            outputs=[state, parent, parent_text, selected, preview],
            api_visibility="undocumented",
            queue=False,
        )
        preview_button.click(
            preview_variants,
            inputs=[prompt, state, parent, mode, strength, count]
            + ([checkpoint] if checkpoint is not None else []),
            outputs=[state, selected, preview, original, parent_text],
            api_visibility="undocumented",
        )
        selected.input(
            pick,
            inputs=[state, selected],
            outputs=preview,
            api_visibility="undocumented",
            queue=False,
        )
        parent.input(
            pick_parent,
            inputs=[state, parent],
            outputs=parent_text,
            api_visibility="undocumented",
            queue=False,
        )
        fence_event(
            apply_button.click,
            apply,
            prompt=prompt,
            inputs=[prompt, state, selected],
            outputs=[prompt, previous],
            history_output=1,
            api_visibility="undocumented",
            queue=False,
        )
        fence_event(
            copy_parent.click,
            lambda current, session, index: apply(current, session, index, True),
            prompt=prompt,
            inputs=[prompt, state, parent],
            outputs=[prompt, previous],
            history_output=1,
            api_visibility="undocumented",
            queue=False,
        )
        fence_event(
            undo_button.click,
            undo,
            prompt=prompt,
            inputs=[prompt, previous],
            outputs=[prompt, previous],
            history_input=1,
            api_visibility="undocumented",
            queue=False,
        )
        if run_event is not None and image_number is not None:

            def generate_reviewed(current, session):
                text, images = checked(generation_batch, current, session)
                return (
                    (text, images, False, (current, text))
                    if replacement_control is not None
                    else (text, images, (current, text))
                )

            bind_generation(
                generate,
                generate_reviewed,
                inputs=[prompt, state],
                outputs=[prompt, image_number]
                + ([replacement_control] if replacement_control is not None else [])
                + [previous],
                run_event=run_event,
            )
