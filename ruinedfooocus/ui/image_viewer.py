"""Browser-only viewing controls, separate from generated image dimensions."""

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui as t


def create_ui():
    controls = {}
    with gr.Row(elem_id="image-view-controls"):
        for key, label in (
            ("fit", "Fit"),
            ("fill", "Fill"),
            ("actual", "Actual size"),
            ("zoom-out", "Zoom out"),
        ):
            controls[key] = gr.Button(
                t(label), elem_id=f"viewer-{key}", size="sm", scale=0, min_width=44
            )
        gr.HTML(
            '<output aria-live="polite" aria-atomic="true">100%</output>',
            elem_id="viewer-zoom-value",
            container=False,
            scale=0,
            min_width=52,
        )
        for key, label in (("zoom-in", "Zoom in"), ("resize", "Resize viewer")):
            controls[key] = gr.Button(
                t(label), elem_id=f"viewer-{key}", size="sm", scale=0, min_width=44
            )
    for key in ("fit", "fill", "actual", "zoom-out", "zoom-in"):
        action = (
            f"mode('{key}')"
            if key in ("fit", "fill", "actual")
            else f"zoom({1.2 if key == 'zoom-in' else 1 / 1.2})"
        )
        controls[key].click(
            fn=None,
            inputs=[],
            outputs=[],
            js=f"() => {{ window.rfImageViewer.{action}; return []; }}",
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
    return controls
