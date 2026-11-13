"""Native GPU background removal with an explicit alternative ONNX choice."""

import gradio as gr
from ruinedfooocus.core.translation_manager import translate_ui
from ruinedfooocus.features.image_processing.native_background import local_models

DEFAULT_MODEL = "birefnet.safetensors"


def choices(manager, selected=""):
    values = [(translate_ui("Hardware background removal (ONNX)"), "")]
    values.extend((name, name) for name in local_models(manager))
    if DEFAULT_MODEL not in {value for _, value in values}:
        values.append((translate_ui("BiRefNet (download on first use)"), DEFAULT_MODEL))
    if selected and selected not in {value for _, value in values}:
        values.append((translate_ui("Unavailable: ") + selected, selected))
    return values


def create_ui(selection, control_type, model_refresh, kind_for):
    import ruinedfooocus.app.state as shared

    saved = shared.settings.default_settings.get("rembg_native_model", DEFAULT_MODEL) or ""
    with gr.Group(visible=kind_for(selection.value, control_type.value) == "rembg") as group:
        model = gr.Dropdown(
            label=translate_ui("Background model"),
            choices=choices(shared.path_manager, saved),
            value=saved,
            info=translate_ui(
                "BiRefNet uses the image device and downloads on first use. Choose Hardware background removal for ONNX settings."
            ),
        )
    shared.add_ctrl("rembg_native_model", model)
    gr.on(
        triggers=[selection.change, control_type.change],
        fn=lambda choice, kind: gr.update(visible=kind_for(choice, kind) == "rembg"),
        inputs=[selection, control_type],
        outputs=group,
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    if model_refresh is not None:
        model_refresh.change(
            lambda selected: gr.update(
                choices=choices(shared.path_manager, selected), value=selected
            ),
            inputs=model,
            outputs=model,
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )
    return model
