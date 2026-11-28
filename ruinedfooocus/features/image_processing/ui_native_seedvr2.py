"""Contextual SeedVR2 controls on the existing Image tools surface."""

import gradio as gr

from ruinedfooocus.core.translation_manager import localize, translate_ui as t
from ruinedfooocus.features.image_processing.native_seedvr2 import (
    COLORS,
    DEFAULTS,
    KIND,
)
from ruinedfooocus.features.image_processing.utility_defaults import component_choices, picker_value


def model_choices(manager, codec=False):
    return component_choices(manager, KIND, "seedvr2_vae" if codec else "seedvr2_model")


def selection_choices(manager, selected=None, *, codec=False):
    values = model_choices(manager, codec=codec)
    if selected and selected not in {value for _, value in values}:
        values.append((localize("Unavailable: {name}", name=selected), selected))
    return values


def create_ui(selection, control_type, model_refresh, kind_for):
    from ruinedfooocus.app import state as shared

    saved = shared.settings.default_settings
    selected = {
        field: picker_value(
            shared.path_manager,
            KIND,
            field,
            saved.get(field),
            model_choices(shared.path_manager, codec=field == "seedvr2_vae"),
        )
        for field in ("seedvr2_model", "seedvr2_vae")
    }
    with gr.Group(visible=kind_for(selection.value, control_type.value) == KIND) as group:
        gr.Markdown(
            t(
                "Restore one still image with SeedVR2 and its matching codec. The complete resized source and alpha are preserved, including odd-sized edges."
            )
        )
        controls = {
            "seedvr2_model": gr.Dropdown(
                label=t("SeedVR2 restoration model"),
                choices=selection_choices(shared.path_manager, selected["seedvr2_model"]),
                value=selected["seedvr2_model"] or None,
            ),
            "seedvr2_vae": gr.Dropdown(
                label=t("SeedVR2 codec"),
                choices=selection_choices(shared.path_manager, selected["seedvr2_vae"], codec=True),
                value=selected["seedvr2_vae"] or None,
            ),
            "seedvr2_scale": gr.Slider(
                label=t("Restoration scale"),
                minimum=1,
                maximum=4,
                step=0.25,
                value=saved.get("seedvr2_scale", DEFAULTS["seedvr2_scale"]),
            ),
            "seedvr2_color": gr.Dropdown(
                label=t("Restoration color correction"),
                choices=list(COLORS),
                value=saved.get("seedvr2_color", DEFAULTS["seedvr2_color"]),
            ),
        }
    for name, control in controls.items():
        shared.add_ctrl(name, control)
    for trigger in (selection, control_type):
        trigger.change(
            lambda selected, kind: gr.update(visible=kind_for(selected, kind) == KIND),
            inputs=[selection, control_type],
            outputs=group,
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )

    def refresh_components(model, codec):
        updates = []
        for field, value in (("seedvr2_model", model), ("seedvr2_vae", codec)):
            codec_field = field == "seedvr2_vae"
            value = picker_value(
                shared.path_manager,
                KIND,
                field,
                value,
                model_choices(shared.path_manager, codec=codec_field),
            )
            updates.append(
                gr.update(
                    choices=selection_choices(shared.path_manager, value, codec=codec_field),
                    value=value,
                )
            )
        return updates

    model_refresh.change(
        refresh_components,
        inputs=[controls["seedvr2_model"], controls["seedvr2_vae"]],
        outputs=[controls["seedvr2_model"], controls["seedvr2_vae"]],
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    return controls
