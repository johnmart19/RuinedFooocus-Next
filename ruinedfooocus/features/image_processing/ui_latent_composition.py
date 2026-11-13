"""Contextual bytes-only controls for a saved native latent operation."""

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui as t
from ruinedfooocus.features.image_processing import latent_composition as composition


def create_ui(selection, control_type, kind_for):
    from ruinedfooocus.app import state as shared

    saved = shared.settings.default_settings
    values = composition.settings(saved)
    with gr.Group(
        visible=composition.handles(kind_for(selection.value, control_type.value))
    ) as panel:
        gr.Markdown(
            t(
                "Compose two saved latents from this installation with the same original context, dtype and complete shape. No generation checkpoint or sampling is used. The result is one latent file; the image gallery is retained."
            )
        )
        with gr.Row():
            primary = gr.File(
                label=t("Destination saved latent"), type="binary", file_types=[".latent"]
            )
            secondary = gr.File(
                label=t("Overlay saved latent"), type="binary", file_types=[".latent"]
            )
        mode = gr.Dropdown(
            label=t("Latent composition operation"),
            choices=[
                (t("Native replacement with edge feathering"), "replace"),
                (t("Native masked replacement"), "masked"),
            ],
            value=values["latent_composition_mode"],
        )
        with gr.Row():
            x = gr.Number(
                label=t("Latent overlay left (pixels)"),
                value=values["latent_composition_x"],
                minimum=0,
                maximum=16384,
                precision=0,
            )
            y = gr.Number(
                label=t("Latent overlay top (pixels)"),
                value=values["latent_composition_y"],
                minimum=0,
                maximum=16384,
                precision=0,
            )
            feather = gr.Number(
                label=t("Latent edge feathering (pixels)"),
                value=values["latent_composition_feather"],
                minimum=0,
                maximum=16384,
                precision=0,
                visible=mode.value == "replace",
            )
        mask = gr.Image(
            label=t("Latent selection mask: white copies the overlay"),
            type="pil",
            image_mode="L",
            format="png",
            visible=mode.value == "masked",
        )
        gr.Markdown(
            t(
                "Offsets and feathering must align to the saved codec grid (8 or 16 pixels). The destination canvas stays fixed; explicit offsets clip the overlay at its edges. A mask must match the complete saved image canvas."
            )
        )
        gr.Markdown(
            t(
                "Original uploads and context receipts are required. Re-upload the same two originals and mask to repeat this operation; saved output records their hashes and exact parameters."
            )
        )
        clear = gr.Button(t("Clear latent composition inputs"), size="sm")
    controls = dict(zip(composition.INPUTS, (primary, secondary)))
    controls.update(
        latent_composition_mode=mode,
        latent_composition_x=x,
        latent_composition_y=y,
        latent_composition_feather=feather,
        latent_composition_mask=mask,
    )
    for key, control in controls.items():
        shared.add_ctrl(key, control)
    quiet = dict(queue=False, show_progress="hidden", api_visibility="undocumented")

    def operation_context(value):
        return gr.update(visible=value == "replace"), gr.update(visible=value == "masked")

    mode.change(operation_context, inputs=mode, outputs=[feather, mask], **quiet)
    gr.on(
        triggers=[selection.change, control_type.change],
        fn=lambda selected, kind: gr.update(visible=composition.handles(kind_for(selected, kind))),
        inputs=[selection, control_type],
        outputs=panel,
        **quiet,
    )
    clear.click(lambda: (None, None, None), outputs=[primary, secondary, mask], **quiet)
    return controls
