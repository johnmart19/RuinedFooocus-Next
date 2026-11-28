"""Contextual standalone WeVisOne object and subject controls."""

import gradio as gr

from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui as t
from ruinedfooocus.features.image_processing import wevisone


def choices(manager, selected=None, *, locale=None):
    values = wevisone.model_choices(manager)
    if selected and selected not in {value for _, value in values}:
        values.append((localize("Unavailable: {name}", locale, name=selected), selected))
    return values


def create_ui(selection, control_type, model_refresh, kind_for):
    from ruinedfooocus.app import state as shared

    saved = shared.settings.default_settings
    kind = kind_for(selection.value, control_type.value)
    selected = saved.get("wevisone_model") or None
    with gr.Group(visible=kind in wevisone.KINDS) as group:
        model = gr.Dropdown(
            label=t("WeVisOne image encoder"),
            choices=choices(shared.path_manager, selected)
            if kind in wevisone.KINDS
            else [(selected, selected)]
            if selected
            else [],
            value=selected,
        )
        query = gr.Textbox(
            label=t("Objects or subject"),
            value=saved.get("wevisone_query", "person"),
            lines=3,
            max_lines=8,
            info=t(
                "For detection, enter one object label per line. For matting, describe the subject."
            ),
        )
        output = gr.Radio(
            label=t("Matting output"),
            choices=[
                (t("Cutout"), "cutout"),
                (t("Grayscale matte"), "mask"),
                (t("Overlay"), "overlay"),
            ],
            value=saved.get("wevisone_output", "cutout"),
            visible=kind == "wevisone-matte",
        )
        length = gr.Number(
            label=t("Maximum answer length"),
            value=saved.get("wevisone_max_length", 4096),
            minimum=1,
            maximum=32768,
            precision=0,
        )
        gr.Markdown(
            t(
                "WeVisOne keeps the source size. Cutouts multiply the original alpha by the native matte; grayscale mattes contain the native mask."
            )
        )
    controls = dict(
        wevisone_model=model,
        wevisone_query=query,
        wevisone_output=output,
        wevisone_max_length=length,
    )
    for key, control in controls.items():
        shared.add_ctrl(key, control)

    def context(selected_tool, custom_type, model_value, request: gr.Request = None):
        active = kind_for(selected_tool, custom_type)
        values = (
            choices(shared.path_manager, model_value, locale=resolve_locale(request))
            if active in wevisone.KINDS
            else [(model_value, model_value)]
            if model_value
            else []
        )
        return (
            gr.update(visible=active in wevisone.KINDS),
            gr.update(choices=values, value=model_value),
            gr.update(visible=active == "wevisone-matte"),
        )

    gr.on(
        triggers=[selection.change, control_type.change, model_refresh.change],
        fn=context,
        inputs=[selection, control_type, model],
        outputs=[group, model, output],
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
        api_visibility="undocumented",
    )
    return controls
