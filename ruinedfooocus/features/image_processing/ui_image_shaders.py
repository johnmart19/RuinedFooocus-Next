"""Point-table curves and a bounded named GLSL recipe selector."""

from copy import deepcopy

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui
from ruinedfooocus.features.image_processing.image_shaders import (
    CURVE_FIELDS,
    DEFAULTS,
    EFFECTS,
    KINDS,
)


def create_ui(selection, control_type, kind_for):
    import ruinedfooocus.app.state as shared

    saved = shared.settings.default_settings
    kind = kind_for(selection.value, control_type.value)
    with gr.Group(visible=kind in KINDS) as panel:
        gr.Markdown(
            translate_ui(
                "Apply curves or a named image effect on CPU or with native GLSL acceleration. One source up to 4 megapixels and 4096 pixels per side; size and alpha are preserved. No model is needed. CPU mode and automatic CPU fallback are supported."
            )
        )
        with gr.Group(visible=kind == "image-curves") as curve_group:
            gr.Markdown(
                translate_ui(
                    "Use 2 to 32 points per curve, with input and output values from 0 to 1. Input values must increase, starting at 0 and ending at 1. Master RGB runs before the separate channels."
                )
            )
            interpolation = gr.Radio(
                label=translate_ui("Curve interpolation"),
                choices=[
                    (translate_ui("Linear"), "linear"),
                    (translate_ui("Monotone cubic"), "monotone_cubic"),
                ],
                value=saved.get("curve_interpolation", "monotone_cubic"),
            )
            curve_controls = {}
            with gr.Tabs():
                for field, label in zip(CURVE_FIELDS, ("Master RGB", "Red", "Green", "Blue")):
                    with gr.Tab(translate_ui(label)):
                        curve_controls[field] = gr.Dataframe(
                            headers=[translate_ui("Input"), translate_ui("Output")],
                            datatype=["number", "number"],
                            row_count=(2, "dynamic"),
                            column_count=(2, "fixed"),
                            type="array",
                            value=deepcopy(saved.get(field, DEFAULTS[field])),
                            interactive=True,
                        )
        with gr.Group(visible=kind == "glsl-effect") as effect_group:
            effect = gr.Radio(
                label=translate_ui("Image effect"),
                choices=[(translate_ui(label), value) for value, label in EFFECTS.items()],
                value=saved.get("shader_effect", "invert"),
            )
            amount = gr.Slider(
                label=translate_ui("Effect amount"),
                minimum=0,
                maximum=1,
                step=0.01,
                value=saved.get("shader_amount", 1.0),
            )
            levels = gr.Slider(
                label=translate_ui("Posterize levels"),
                minimum=2,
                maximum=32,
                step=1,
                value=saved.get("shader_levels", 8),
                visible=effect.value == "posterize",
            )
    controls = {
        **curve_controls,
        "curve_interpolation": interpolation,
        "shader_effect": effect,
        "shader_amount": amount,
        "shader_levels": levels,
    }
    for field, component in controls.items():
        shared.add_ctrl(field, component)

    def shader_context(choice, tool):
        active = kind_for(choice, tool)
        return (
            gr.update(visible=active in KINDS),
            gr.update(visible=active == "image-curves"),
            gr.update(visible=active == "glsl-effect"),
        )

    gr.on(
        triggers=[selection.change, control_type.change],
        fn=shader_context,
        inputs=[selection, control_type],
        outputs=[panel, curve_group, effect_group],
        queue=False,
        api_visibility="undocumented",
    )
    effect.change(
        lambda value: gr.update(visible=value == "posterize"),
        inputs=effect,
        outputs=levels,
        queue=False,
        api_visibility="undocumented",
    )
    return controls
