"""Explicit generation inputs and exports for ordinary native tensor contracts."""

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui as t
from ruinedfooocus.features.generation.native_tensor_context import SAVED_FAMILIES


def create_ui():
    from ruinedfooocus.app import state as shared

    with gr.Accordion(t("Advanced latent and conditioning files"), open=False):
        export_latent = gr.Checkbox(label=t("Export sampled native latent"), value=False)
        export_conditioning = gr.Checkbox(
            label=t("Export native positive and negative conditioning"), value=False
        )
        export_handoff = gr.Checkbox(
            label=t("Export noisy SDXL refiner handoff"),
            value=False,
            info=t(
                "Requires a staged SDXL base/refiner request. Resume this file with its original prompts, sources, seed and full schedule."
            ),
        )
        with gr.Accordion(t("Import latent and conditioning files"), open=False):
            tensor_family = gr.Dropdown(
                label=t("Original tensor model family"),
                choices=["auto", *sorted(SAVED_FAMILIES)],
                value="auto",
                info=t(
                    "Choose the original family for external files without saved RuinedFooocus context."
                ),
            )
            tensor_workflow = gr.Checkbox(
                label=t("Preserve native workflow tensor context"),
                value=False,
                info=t(
                    "Required automatically for layers, masks and references. Complex files can be replayed only with this app's original saved-context receipt."
                ),
            )
            latent_input = gr.File(
                label=t("Input native latent"),
                file_types=[".latent"],
                type="binary",
                elem_classes=["rf-compact-file"],
            )
            latent_denoise = gr.Slider(
                label=t("Native latent denoise"),
                minimum=0.01,
                maximum=1,
                step=0.01,
                value=0.5,
                info=t(
                    "Lower values retain more of the imported latent. Full denoise can discard it in flow models."
                ),
            )
            positive_input = gr.File(
                label=t("Input positive conditioning"),
                file_types=[".safetensors"],
                type="binary",
                elem_classes=["rf-compact-file"],
            )
            negative_input = gr.File(
                label=t("Input negative conditioning"),
                file_types=[".safetensors"],
                type="binary",
                elem_classes=["rf-compact-file"],
            )
        controls = {
            "native_tensor_family": tensor_family,
            "native_tensor_workflow": tensor_workflow,
            "native_latent_input": latent_input,
            "native_latent_denoise": latent_denoise,
            "native_positive_input": positive_input,
            "native_negative_input": negative_input,
            "native_export_latent": export_latent,
            "native_export_conditioning": export_conditioning,
            "native_export_handoff": export_handoff,
        }
    for name, control in controls.items():
        shared.add_ctrl(name, control)
    return controls
