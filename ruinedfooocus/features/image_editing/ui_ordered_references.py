"""Contextual named native reference recipes, separate from denoise-based editing."""

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui as t
from ruinedfooocus.features.image_editing.ordered_native_references import FLUX2_FAMILIES
from ruinedfooocus.features.image_editing.ordered_native_references import (
    omit_inactive_recipe as omit_inactive_recipe,
)


def recipe_choices(family):
    choices = [(t("Automatic"), "auto")]
    if family == "Boogu":
        choices.append((t("Boogu Edit: up to sixteen references"), "boogu-edit"))
    elif family == "Krea2":
        choices.append((t("Krea2 style reference: up to three images"), "krea2-style"))
    elif family in FLUX2_FAMILIES:
        choices.append((t("Flux.2 / Klein ordered references"), "flux2-reference"))
    elif family == "QwenImage":
        choices += [
            (t("Qwen Image Edit: one reference"), "qwen-edit"),
            (t("Qwen Edit Plus 2509: up to three references"), "qwen-edit-2509"),
            (t("Qwen Edit Plus 2511: up to three references"), "qwen-edit-2511"),
        ]
    elif family == "LongCatImage":
        choices.append((t("LongCat Image Edit: distinct edit checkpoint"), "longcat-edit"))
    return choices


def create_ui(base_model, selection, tool_type, model_refresh, kind_for):
    from ruinedfooocus.app import state as shared
    from ruinedfooocus.features.image_editing.ui_guidance import family_for_checkpoint

    family = family_for_checkpoint(base_model.value)
    available = recipe_choices(family)
    saved = shared.settings.default_settings.get("native_edit_recipe", "auto")
    if saved not in {value for _, value in available}:
        saved = "auto"
    initial = kind_for(selection.value, tool_type.value) == "edit" and len(available) > 1
    with gr.Accordion(t("Edit recipe"), open=False, visible=initial) as panel:
        recipe = gr.Dropdown(label=t("Edit recipe"), choices=available, value=saved)
        gr.Markdown(
            t(
                "Choose the matching Edit checkpoint and recipe. Add the primary image first, followed by the other references in upload order."
            )
        )
        from ruinedfooocus.features.image_editing.krea_style import DEFAULT_LORA

        krea_active = family == "Krea2" and saved in ("auto", "krea2-style")
        with gr.Group(visible=krea_active) as krea_panel:
            style_name = shared.settings.default_settings.get("krea_style_lora") or DEFAULT_LORA
            get_names = getattr(getattr(shared, "models", None), "get_names", lambda _category: ())
            style_lora = gr.Dropdown(
                label=t("Krea2 style LoRA"),
                choices=sorted(
                    set((DEFAULT_LORA, style_name, *(get_names("loras") if krea_active else ())))
                ),
                value=style_name,
                allow_custom_value=True,
                info=t(
                    "Required model-only style adapter. Missing registered files are prepared on Run."
                ),
            )
            strength = gr.Slider(
                label=t("Krea2 style strength"),
                minimum=-10,
                maximum=10,
                step=0.05,
                value=shared.settings.default_settings.get("krea_style_strength", 1.0),
            )
    shared.add_ctrl("native_edit_recipe", recipe)
    shared.add_ctrl("krea_style_lora", style_lora)
    shared.add_ctrl("krea_style_strength", strength)

    def context(name, tool, kind, current):
        family = family_for_checkpoint(name)
        choices = recipe_choices(family)
        active = kind_for(tool, kind) == "edit" and len(choices) > 1
        value = current if current in {value for _, value in choices} else "auto"
        return gr.update(visible=active), gr.update(choices=choices, value=value)

    gr.on(
        triggers=[base_model.change, selection.change, tool_type.change]
        + ([model_refresh.change] if model_refresh is not None else []),
        fn=context,
        inputs=[base_model, selection, tool_type, recipe],
        outputs=[panel, recipe],
        trigger_mode="always_last",
        show_progress="hidden",
        show_progress_on=[],
        api_visibility="undocumented",
    )

    def style_context(name, tool, kind, current):
        return gr.update(
            visible=family_for_checkpoint(name) == "Krea2"
            and kind_for(tool, kind) == "edit"
            and current in ("auto", "krea2-style")
        )

    gr.on(
        triggers=[base_model.change, recipe.change, selection.change, tool_type.change],
        fn=style_context,
        inputs=[base_model, selection, tool_type, recipe],
        outputs=krea_panel,
        trigger_mode="always_last",
        show_progress="hidden",
        api_visibility="undocumented",
    )
    return recipe
