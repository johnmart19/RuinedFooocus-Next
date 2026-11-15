"""Browser-only viewing controls, separate from generated image dimensions."""

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui as t


def create_ui():
    with gr.Row(elem_id="image-view-controls"):
        mode = gr.Radio(
            choices=[(t("Fit"), "fit"), (t("Actual size"), "actual")],
            value="fit",
            label=t("Image view"),
            show_label=False,
            container=False,
            min_width=180,
            scale=1,
        )
        height = gr.Slider(
            minimum=25,
            maximum=150,
            step=5,
            value=65,
            label=t("Viewer height (%)"),
            min_width=180,
            scale=2,
        )
    for component, method in ((mode, "mode"), (height, "height")):
        component.input(
            fn=None,
            inputs=[component],
            outputs=[],
            js=f"(value) => {{ window.rfImageViewer.{method}(value); return []; }}",
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
    return mode, height
