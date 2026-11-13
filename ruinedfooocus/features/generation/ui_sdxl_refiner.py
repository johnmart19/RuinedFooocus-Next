"""Contextual controls for an explicit native SDXL base/refiner pairing."""

import gradio as gr
from pathlib import Path

from ruinedfooocus.core.translation_manager import translate_ui as t


def available(family):
    from ruinedfooocus.features.models.image_components import ALIASES

    return ALIASES.get(family, family) == "SDXL"


def model_choices(models):
    choices = []
    for name in models.get_names("checkpoints"):
        path = models.get_file("checkpoints", name)
        try:
            if (
                path is not None
                and Path(path).suffix.casefold() == ".safetensors"
                and models.detect_checkpoint_base(path) == "SDXLRefiner"
            ):
                choices.append(name)
        except (OSError, ValueError):
            continue
    return choices


def create_ui(base_model, *, preset_selection=None, main_model=None, model_refresh=None):
    import ruinedfooocus.app.state as shared
    from ruinedfooocus.features.image_editing.ui_guidance import family_for_checkpoint
    from ruinedfooocus.features.image_editing.image_guidance import effective_preset_checkpoint
    from ruinedfooocus.features.image_editing.ui_native_workflows import local_choices
    from ruinedfooocus.features.models.component_candidates import installed_candidates

    defaults = shared.settings.default_settings
    active = available(family_for_checkpoint(base_model.value))
    selected = defaults.get("sdxl_refiner_model") if active else None
    clip = defaults.get("sdxl_refiner_clip")
    encoders = [
        row["name"]
        for row in installed_candidates("clip_g", shared.path_manager)
        if row["compatibility"] != "incompatible"
    ]
    controls = {}
    with gr.Accordion(t("SDXL base → refiner"), open=False, visible=active) as group:
        gr.Markdown(
            t(
                "Choose a separate SDXL refiner and the step where the base hands it the noisy latent. Both stages share the total steps, sampler, scheduler, seed and denoise."
            )
        )
        controls["sdxl_refiner_model"] = gr.Dropdown(
            label=t("SDXL refiner checkpoint"),
            choices=[(t("Disabled"), None)] + local_choices(model_choices(shared.models), selected),
            value=selected,
        )
        with gr.Row():
            controls["sdxl_refiner_switch_step"] = gr.Number(
                label=t("Refiner switch step"),
                minimum=1,
                maximum=199,
                precision=0,
                value=defaults.get("sdxl_refiner_switch_step"),
                info=t(
                    "Required when a refiner is selected. Use one through total steps minus one."
                ),
                min_width=180,
            )
            controls["sdxl_refiner_cfg"] = gr.Number(
                label=t("Refiner CFG"),
                minimum=0,
                maximum=100,
                value=defaults.get("sdxl_refiner_cfg"),
                info=t("Blank uses the base CFG. Zero is an explicit override."),
                min_width=180,
            )
        controls["sdxl_refiner_clip"] = gr.Dropdown(
            label=t("Refiner CLIP-G"),
            choices=[(t("Bundled or automatic"), None)] + local_choices(encoders, clip),
            value=clip,
            info=t(
                "Used for split refiners. Final decoding uses the base model's selected SDXL VAE."
            ),
        )

    model = main_model if main_model is not None else base_model

    def refiner_context(name, current, preset=None):
        checkpoint = (
            effective_preset_checkpoint(name, preset, defaults.get("preset_mode", 7))
            if preset_selection is not None
            else name
        )
        if checkpoint is None:
            return gr.skip(), gr.skip()
        visible = available(family_for_checkpoint(checkpoint))
        return gr.update(visible=visible), gr.update(
            choices=[(t("Disabled"), None)] + local_choices(model_choices(shared.models), current),
            value=current if visible else None,
        )

    gr.on(
        triggers=[model.change]
        + ([preset_selection.change] if preset_selection is not None else [])
        + ([model_refresh.change] if model_refresh is not None else []),
        fn=refiner_context,
        inputs=[model, controls["sdxl_refiner_model"], preset_selection]
        if preset_selection is not None
        else [model, controls["sdxl_refiner_model"]],
        outputs=[group, controls["sdxl_refiner_model"]],
        trigger_mode="always_last",
        show_progress="hidden",
        show_progress_on=[],
        api_visibility="undocumented",
    )
    if model_refresh is not None:

        def refiner_encoder_choices(current):
            from ruinedfooocus.features.models.component_candidates import installed_candidates

            names = [
                row["name"]
                for row in installed_candidates("clip_g", shared.path_manager)
                if row["compatibility"] != "incompatible"
            ]
            return gr.update(
                choices=[(t("Bundled or automatic"), None)] + local_choices(names, current),
                value=current,
            )

        model_refresh.change(
            refiner_encoder_choices,
            inputs=controls["sdxl_refiner_clip"],
            outputs=controls["sdxl_refiner_clip"],
            show_progress="hidden",
            api_visibility="undocumented",
        )
    return controls
