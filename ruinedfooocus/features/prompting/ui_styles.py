from ruinedfooocus.core.translation_manager import localize, localized_output, translate_ui

"""Grouped visual styles; prompt processors remain explicit, separate tools."""

from dataclasses import dataclass
from pathlib import Path

import gradio as gr
from ruinedfooocus.features.prompting.ui_draft_fence import fence_event

from ruinedfooocus.features.prompting.style_catalog import (
    load_catalog,
    merge_selection,
    normalize_selection,
    style_label,
    style_preview_url,
    resolve_inline_styles,
)
from ruinedfooocus.features.prompting.style_catalog import style_templates as load_styles

NEURAL_PROCESSORS = ("Local chat model",)


@dataclass
class StyleControls:
    selection: gr.Dropdown
    preview: gr.Button
    bake: gr.Button
    undo: gr.Button
    panel: gr.Group
    positive: gr.Textbox
    negative: gr.Textbox
    previous: gr.State


def create_ui(settings):
    initial = (
        normalize_selection(settings.get("style", []))
        if settings.get("style_application") == 1
        else []
    )
    selection = gr.Dropdown(
        choices=list(load_styles()), value=initial, multiselect=True, visible=False
    )
    search = gr.Textbox(
        label=translate_ui("Search styles"),
        placeholder=translate_ui("Find a style or category"),
        value="",
        elem_id="style-search",
    )
    summary = gr.Textbox(
        label=translate_ui("Applied in order"), interactive=False, lines=1, max_lines=3
    )
    catalog = load_catalog()
    groups = list(dict.fromkeys([style.group for style in catalog.values()] + ["Custom"]))
    assets = Path(__file__).resolve().parents[2] / "ui/assets"

    def style_cards_payload(current, query, locale=None):
        chosen = normalize_selection(current)
        text = (query or "").casefold().strip()
        items = []
        active_groups = []
        for name, style in load_catalog().items():
            group = style.group if style.group in groups else "Custom"
            label = style_label(name)
            if text and text not in (name + " " + label + " " + group).casefold():
                continue
            items.append(
                dict(
                    name=name,
                    label=label,
                    group=group,
                    preview=style_preview_url(name),
                )
            )
            if group not in active_groups:
                active_groups.append(group)
        return dict(
            heading=localize("All styles", locale),
            allLabel=localize("All", locale),
            empty=localize("No matching styles.", locale),
            hasQuery=bool(text),
            groups=active_groups,
            selected=chosen,
            items=items,
        )

    style_cards = gr.HTML(
        value=style_cards_payload(initial, ""),
        elem_id="style-cards",
        padding=False,
        html_template=(assets / "style_cards.html").read_text(encoding="utf-8"),
        js_on_load=(assets / "style_cards.js").read_text(encoding="utf-8"),
        css_template=(assets / "style_cards.css").read_text(encoding="utf-8"),
    )

    def toggle_style_card(current, evt: gr.EventData):
        data = getattr(evt, "_data", None) or {}
        name = data.get("name")
        if not isinstance(name, str) or not name or name not in load_catalog():
            return gr.skip()
        chosen = normalize_selection(current)
        next_selected = [] if name in chosen else [name]
        return merge_selection(chosen, [name], next_selected)

    style_cards.click(
        toggle_style_card,
        inputs=[selection],
        outputs=selection,
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )

    checkboxes, accordions = [], []

    def choices_for(group, query):
        text = (query or "").casefold().strip()
        return [
            (style_label(name), name)
            for name, style in load_catalog().items()
            if (style.group if style.group in groups else "Custom") == group
            and text in (name + " " + style_label(name) + " " + style.group).casefold()
        ]

    with gr.Accordion(translate_ui("Category list"), open=False, visible=False):
        for group in groups:
            with gr.Accordion(group, open=False, visible=bool(choices_for(group, ""))) as accordion:
                options = gr.CheckboxGroup(
                    label=group,
                    show_label=False,
                    choices=choices_for(group, ""),
                    elem_classes="rf-style-choices",
                )
            checkboxes.append(options)
            accordions.append(accordion)

            def choose(current, chosen, query, group=group):
                return merge_selection(
                    current, [name for _, name in choices_for(group, query)], chosen
                )

            options.input(
                choose,
                inputs=[selection, options, search],
                outputs=selection,
                queue=False,
                api_visibility="undocumented",
            )

    def refresh(current, query):
        chosen = normalize_selection(current)
        description = "\n".join(
            f"{i + 1}. {style_label(name)}" for i, name in enumerate(chosen)
        ) or translate_ui("No styles or prompt processors selected.")
        updates = [gr.update(value=description)]
        for group in groups:
            choices = choices_for(group, query)
            values = [name for _, name in choices]
            updates.append(
                gr.update(choices=choices, value=[name for name in chosen if name in values])
            )
        for group in groups:
            updates.append(
                gr.update(
                    visible=bool(choices_for(group, query)), **({"open": True} if query else {})
                )
            )
        updates.append(gr.update(value=style_cards_payload(chosen, query)))
        return updates

    gr.on(
        triggers=[selection.change, search.input],
        fn=refresh,
        inputs=[selection, search],
        outputs=[summary, *checkboxes, *accordions, style_cards],
        queue=False,
        api_visibility="undocumented",
    )
    localized_output(
        lambda locale, current, query: style_cards_payload(current, query, locale=locale),
        [selection, search],
        [style_cards],
    )
    # Populate the noninteractive summary and initial checkbox state without activating legacy defaults.
    initial_updates = refresh(initial, "")
    summary.value = initial_updates[0]["value"]
    for checkbox, update in zip(checkboxes, initial_updates[1:]):
        checkbox.value = update["value"]
    def _has_clearable_styles(values):
        return any(
            name not in NEURAL_PROCESSORS and name != "LoRA keywords"
            for name in normalize_selection(values)
        )

    random_style = gr.Checkbox(
        label=translate_ui("Pick a random visual style per image"),
        value="Style: Pick Random" in initial,
    )
    bind_subset(selection, random_style, ("Style: Pick Random",))
    with gr.Row(elem_classes=["rf-actions", "rf-actions-compact"]):
        clear = gr.Button(
            translate_ui("Clear styles"), size="sm", interactive=_has_clearable_styles(initial)
        )
        preview = gr.Button(translate_ui("Preview styles"), size="sm")
    selection.change(
        lambda values: gr.update(interactive=_has_clearable_styles(values)),
        inputs=selection,
        outputs=clear,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    clear.click(
        lambda values: [
            name
            for name in normalize_selection(values)
            if name in NEURAL_PROCESSORS or name == "LoRA keywords"
        ],
        inputs=selection,
        outputs=selection,
        queue=False,
        api_visibility="undocumented",
    )
    with gr.Row(elem_classes=["rf-actions", "rf-actions-compact"]):
        bake = gr.Button(translate_ui("Bake into prompt"), size="sm")
        undo = gr.Button(translate_ui("Undo bake"), size="sm", interactive=False)
    with gr.Group(visible=False) as panel:
        positive = gr.Textbox(
            label=translate_ui("Styled prompt"), interactive=False, lines=3, max_lines=8
        )
        negative = gr.Textbox(
            label=translate_ui("Style negative prompt"), interactive=False, lines=2, max_lines=6
        )
        gr.Markdown(
            translate_ui(
                "Preview does not run chat enhancement, expand wildcards or generate an image. Random styles are picked again for each image."
            )
        )
    return StyleControls(
        selection, preview, bake, undo, panel, positive, negative, gr.JSON(None, visible=False)
    )


def bind_subset(selection, control, names):
    def choose(current, value):
        selected = [names[0]] if isinstance(value, bool) and value else value or []
        return merge_selection(current, names, selected)

    def reflect(current):
        chosen = [name for name in normalize_selection(current) if name in names]
        return gr.update(value=bool(chosen) if isinstance(control, gr.Checkbox) else chosen)

    control.input(
        choose,
        inputs=[selection, control],
        outputs=selection,
        queue=False,
        api_visibility="undocumented",
    )
    selection.change(
        reflect, inputs=selection, outputs=control, queue=False, api_visibility="undocumented"
    )
    control.value = reflect(selection.value)["value"]


def add_processor_controls(selection, *, accordion=True):
    info_text = translate_ui(
        "Uses the selected Chat model only when enabled. Your original prompt is retained; Off makes no LLM request."
    )
    if accordion:
        with gr.Accordion(translate_ui("Prompt processors"), open=False):
            processors = gr.Checkbox(
                label=translate_ui("Enhance with local chat model"),
                value=False,
                info=info_text,
            )
    else:
        processors = gr.Checkbox(
            label=translate_ui("Enhance with local chat model"),
            value=False,
            info=info_text,
        )
    bind_subset(selection, processors, NEURAL_PROCESSORS)


def add_lora_keywords_control(selection):
    keywords = gr.Checkbox(
        label=translate_ui("Append LoRA trigger words to the prompt"), value=False
    )
    bind_subset(selection, keywords, ("LoRA keywords",))


def preview_styles(prompt, negative, selected, keywords):
    from ruinedfooocus.features.prompting.sdxl_styles import apply_style

    selected = [name for name in normalize_selection(selected) if name not in NEURAL_PROCESSORS]
    selected, prompt = resolve_inline_styles(
        prompt, selected, load_styles(), deferred=NEURAL_PROCESSORS
    )
    return apply_style(selected, prompt, negative, keywords)


def bind_actions(controls, prompt, negative, keywords):
    inputs = [prompt, negative, controls.selection, keywords]
    pending_bake = gr.State(None)

    def preview(*values):
        positive, negative = preview_styles(*values)
        return positive, negative, gr.update(visible=True)

    def prepare_bake(current, negative, selected, words):
        positive, styled_negative = preview_styles(current, negative, selected, words)
        remaining = [name for name in normalize_selection(selected) if name in NEURAL_PROCESSORS]
        record = (current, negative, selected, positive, styled_negative, remaining, words)
        return positive, styled_negative, remaining, record, gr.update(interactive=True)

    def publish_bake(current, negative, selected, words, pending):
        if pending is None:
            raise gr.Error("Preview styles before baking.", print_exception=False)
        record = pending[3]
        if (current, negative, selected, words) != (record[0], record[1], record[2], record[6]):
            raise gr.Error(
                "The prompt or styles changed. Bake again for the current draft.",
                print_exception=False,
            )
        return pending

    def undo(current, negative, selected, record):
        if record is None:
            return gr.skip(), gr.skip(), gr.skip(), None, gr.update(interactive=False)
        if (current, negative, selected) != (record[3], record[4], record[5]):
            raise gr.Error(
                "The prompt or styles changed after baking. Undo would overwrite your edits.",
                print_exception=False,
            )
        return record[0], record[1], record[2], None, gr.update(interactive=False)

    controls.preview.click(
        preview,
        inputs=inputs,
        outputs=[controls.positive, controls.negative, controls.panel],
        api_visibility="undocumented",
    )
    prepare_event = controls.bake.click(
        prepare_bake,
        inputs=inputs,
        outputs=pending_bake,
        api_visibility="undocumented",
    )
    fence_event(
        prepare_event.success,
        publish_bake,
        prompt=prompt,
        inputs=[prompt, negative, controls.selection, keywords, pending_bake],
        outputs=[prompt, negative, controls.selection, controls.previous, controls.undo],
        history_output=3,
        history_revision=7,
        queue=False,
        api_visibility="undocumented",
    )
    fence_event(
        controls.undo.click,
        undo,
        prompt=prompt,
        inputs=[prompt, negative, controls.selection, controls.previous],
        outputs=[prompt, negative, controls.selection, controls.previous, controls.undo],
        history_input=3,
        history_revision=7,
        queue=False,
        api_visibility="undocumented",
    )
