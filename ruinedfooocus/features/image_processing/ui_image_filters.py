"""Contextual controls for explicit native still-image filters; Off by default."""

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui
from ruinedfooocus.features.image_processing.image_filters import DEFAULTS, KINDS


def create_ui(selection, control_type, kind_for):
    import ruinedfooocus.app.state as shared

    saved = shared.settings.default_settings
    kind = kind_for(selection.value, control_type.value)
    with gr.Group(visible=kind in KINDS) as panel:
        gr.Markdown(
            translate_ui(
                "Apply this filter to one source image. Its size and alpha are preserved. No model or prompt is needed."
            )
        )
        with gr.Row(visible=kind in ("image-blur", "image-sharpen")) as kernel:
            radius = gr.Slider(
                label=translate_ui("Filter radius"),
                minimum=1,
                maximum=31,
                step=1,
                value=saved.get("filter_radius", 1),
            )
            sigma = gr.Slider(
                label=translate_ui("Filter sigma"),
                minimum=0.1,
                maximum=10,
                step=0.1,
                value=saved.get("filter_sigma", 1.0),
            )
        strength = gr.Slider(
            label=translate_ui("Sharpen strength"),
            minimum=0,
            maximum=5,
            step=0.01,
            value=saved.get("filter_strength", 1.0),
            visible=kind == "image-sharpen",
        )
        factor = gr.Slider(
            label=translate_ui("Adjustment factor"),
            minimum=0,
            maximum=2,
            step=0.01,
            value=saved.get("filter_factor", 1.0),
            visible=kind in ("adjust-brightness", "adjust-contrast"),
            info=translate_ui("1 keeps the original brightness or contrast."),
        )
    controls = dict(zip(DEFAULTS, (radius, sigma, strength, factor)))
    for field, component in controls.items():
        shared.add_ctrl(field, component)

    def filter_context(choice, tool):
        active = kind_for(choice, tool)
        return (
            gr.update(visible=active in KINDS),
            gr.update(visible=active in ("image-blur", "image-sharpen")),
            gr.update(visible=active == "image-sharpen"),
            gr.update(visible=active in ("adjust-brightness", "adjust-contrast")),
        )

    gr.on(
        triggers=[selection.change, control_type.change],
        fn=filter_context,
        inputs=[selection, control_type],
        outputs=[panel, kernel, strength, factor],
        queue=False,
        api_visibility="undocumented",
    )
    return controls
