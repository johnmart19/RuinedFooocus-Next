"""Qwen Layered controls appear only for a currently identified native source."""

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui as t
from ruinedfooocus.features.generation.layered import MAX_LAYERS, is_layered_recipe


def create_ui(base_model, model_refresh=None):
    from ruinedfooocus.app import state as shared

    def available(name):
        path = shared.models.get_file("checkpoints", name) if name else None
        return is_layered_recipe(path, shared.path_manager)

    active = available(base_model.value)
    with gr.Accordion(t("Qwen Layered"), open=False, visible=active) as panel:
        gr.Markdown(
            t(
                "Each generation returns ordered transparent layers. Image count controls repetitions."
            )
        )
        layers = gr.Slider(
            label=t("Output layers"),
            minimum=1,
            maximum=MAX_LAYERS,
            step=1,
            value=shared.settings.default_settings.get("qwen_layer_count", 2),
            precision=0,
        )
        source = gr.Image(label=t("Layer source (optional)"), type="pil", image_mode="RGBA")
        gr.Markdown(
            t("Describe the overall scene. Keep automatic upscaling off and use PNG or EXR.")
        )
    shared.add_ctrl("qwen_layer_count", layers)
    shared.add_ctrl("qwen_layer_source", source)
    previous = gr.State(base_model.value)

    def context(name, old_name):
        return (
            gr.update(visible=available(name)),
            # Keep the uploaded draft when a preset/model hides this panel.
            # Layered capture ignores it for other families and validates it
            # against the selected operation when the user submits.
            gr.update(),
            name,
        )

    gr.on(
        triggers=[base_model.change]
        + ([model_refresh.change] if model_refresh is not None else []),
        fn=context,
        inputs=[base_model, previous],
        outputs=[panel, source, previous],
        trigger_mode="always_last",
        show_progress="hidden",
        show_progress_on=[],
        api_visibility="undocumented",
    )
    return panel
