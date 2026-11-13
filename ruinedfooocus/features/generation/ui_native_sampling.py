"""Contextual parameters for native still-image schedules and dual-model guidance."""

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui as t
from ruinedfooocus.features.generation.native_sampling import (
    plan_from_recipe,
    negative_model_choices,
)


def create_ui(base_model, model_refresh=None):
    from ruinedfooocus.app import state as shared

    def plan(name):
        return plan_from_recipe(
            shared.performance_settings.model_recommendations.family_details(name)
        )

    def choices(name, selected):
        return (
            [(t("Automatic"), None)]
            + [(value, value) for value in negative_model_choices(shared.models, name)]
            if selected is not None and selected.guider == "DualModelGuider"
            else [(t("Automatic"), None)]
        )

    initial = plan(base_model.value)
    parameters = dict(initial.parameters) if initial is not None else {}
    defaults = shared.settings.default_settings
    available = choices(base_model.value, initial)
    saved_negative = defaults.get("native_negative_model")
    if saved_negative not in {value for _, value in available}:
        saved_negative = None
    with gr.Accordion(t("Native sampling"), open=False, visible=initial is not None) as panel:
        negative = gr.Dropdown(
            label=t("Unconditional checkpoint"),
            choices=available,
            value=saved_negative,
            visible=initial is not None and initial.guider == "DualModelGuider",
        )
        gr.Markdown(t("Leave schedule parameters empty to use the model's defaults."))
        controls = []
        for key, label, minimum, maximum in (
            ("alpha", t("Schedule alpha"), 0.01, 50),
            ("beta", t("Schedule beta"), 0.01, 50),
            ("mu", t("Schedule mean"), -10, 10),
            ("std", t("Schedule standard deviation"), 0.1, 5),
        ):
            control = gr.Number(
                label=label,
                value=defaults.get("native_schedule_" + key) if key in parameters else None,
                minimum=minimum,
                maximum=maximum,
                visible=key in parameters,
            )
            shared.add_ctrl("native_schedule_" + key, control)
            controls.append(control)
    shared.add_ctrl("native_negative_model", negative)
    previous = gr.State(base_model.value)

    def context(name, previous_name, negative_name, *values):
        selected = plan(name)
        parameters = dict(selected.parameters) if selected is not None else {}
        available = choices(name, selected)
        if negative_name not in {value for _, value in available}:
            negative_name = None
        return [
            gr.update(visible=selected is not None),
            gr.update(
                visible=selected is not None and selected.guider == "DualModelGuider",
                choices=available,
                value=negative_name,
            ),
            *[
                gr.update(visible=key in parameters, value=value if key in parameters else None)
                for key, value in zip(("alpha", "beta", "mu", "std"), values)
            ],
            name,
        ]

    gr.on(
        triggers=[base_model.change]
        + ([model_refresh.change] if model_refresh is not None else []),
        fn=context,
        inputs=[base_model, previous, negative, *controls],
        outputs=[panel, negative, *controls, previous],
        trigger_mode="always_last",
        show_progress="hidden",
        show_progress_on=[],
        api_visibility="undocumented",
    )
    return panel
