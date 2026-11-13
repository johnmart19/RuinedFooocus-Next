"""Explicit generation inputs and exports for ordinary native tensor contracts."""

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui as t
from ruinedfooocus.features.generation.native_tensor_context import SAVED_FAMILIES


def create_ui():
    from ruinedfooocus.app import state as shared

    with gr.Accordion(t("Advanced latent and conditioning files"), open=False):
        gr.Markdown(
            t(
                "Use native files with their original model, codecs and text encoders. Saved workflow context preserves layers, masks, reference tensors and safe conditioning intervals. Reattach original source uploads for workflow replay. Files do not supply model companions. Exports are off until selected."
            )
        )
        controls = {
            "native_tensor_family": gr.Dropdown(
                label=t("Original tensor model family"),
                choices=["auto", *sorted(SAVED_FAMILIES)],
                value="auto",
                info=t(
                    "Choose the original family for external files without saved RuinedFooocus context."
                ),
            ),
            "native_tensor_workflow": gr.Checkbox(
                label=t("Preserve native workflow tensor context"),
                value=False,
                info=t(
                    "Required automatically for layers, masks and references. Complex files can be replayed only with this app's original saved-context receipt."
                ),
            ),
            "native_latent_input": gr.File(
                label=t("Input native latent"),
                file_types=[".latent"],
                type="binary",
            ),
            "native_latent_denoise": gr.Slider(
                label=t("Native latent denoise"),
                minimum=0.01,
                maximum=1,
                step=0.01,
                value=0.5,
                info=t(
                    "Lower values retain more of the imported latent. Full denoise can discard it in flow models."
                ),
            ),
            "native_positive_input": gr.File(
                label=t("Input positive conditioning"),
                file_types=[".safetensors"],
                type="binary",
            ),
            "native_negative_input": gr.File(
                label=t("Input negative conditioning"),
                file_types=[".safetensors"],
                type="binary",
            ),
            "native_export_latent": gr.Checkbox(
                label=t("Export sampled native latent"), value=False
            ),
            "native_export_conditioning": gr.Checkbox(
                label=t("Export native positive and negative conditioning"), value=False
            ),
            "native_export_handoff": gr.Checkbox(
                label=t("Export noisy SDXL refiner handoff"),
                value=False,
                info=t(
                    "Requires a staged SDXL base/refiner request. Resume this file with its original prompts, sources, seed and full schedule."
                ),
            ),
        }
    for name, control in controls.items():
        shared.add_ctrl(name, control)
    return controls
