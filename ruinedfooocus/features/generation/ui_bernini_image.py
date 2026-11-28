"""Still-image controls for the explicitly source-bound Bernini-R workflow."""

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui as t
from ruinedfooocus.features.generation.bernini_image import DEFAULTS, FIELDS, FAMILY, header_role


def low_choices(models):
    choices = [DEFAULTS["bernini_low_model"]]
    for name in models.get_names("checkpoints"):
        path = models.get_file("checkpoints", name)
        if path is not None and header_role(path) == "low" and name not in choices:
            choices.append(name)
    return choices


def create_ui(base_model, *, preset_selection=None, main_model=None, model_refresh=None):
    import ruinedfooocus.app.state as shared
    from ruinedfooocus.features.image_editing.ui_guidance import family_for_checkpoint
    from ruinedfooocus.features.image_editing.image_guidance import effective_preset_checkpoint

    defaults = shared.settings.default_settings
    controls = {}
    with gr.Accordion(
        t("Bernini-R image edit · two stages"),
        open=False,
        visible=family_for_checkpoint(base_model.value) == FAMILY,
    ) as panel:
        gr.Markdown(
            t(
                "Use Native image edit with a source image and optional ordered references. Standard: 40 steps, CFG 5, split 20. Lightx2v: 6 steps, CFG 1, split 3. Explicit sampling values remain yours. Required components are shown before Run; their download size is separate from runtime memory."
            )
        )
        saved = defaults.get("bernini_low_model") or DEFAULTS["bernini_low_model"]
        choices = low_choices(shared.models)
        if saved not in choices:
            choices.append(saved)
        controls["bernini_low_model"] = gr.Dropdown(
            label=t("Bernini low-noise stage"), choices=choices, value=saved
        )
        controls["bernini_lightx2v"] = gr.Checkbox(
            label=t("Use Bernini Lightx2v LoRA"), value=defaults.get("bernini_lightx2v", False)
        )
        controls["bernini_split_step"] = gr.Number(
            label=t("High-to-low switch step"),
            minimum=1,
            maximum=199,
            precision=0,
            value=defaults.get("bernini_split_step"),
            info=t("Blank uses 20 for standard or 3 for Lightx2v, bounded by the selected steps."),
        )
        with gr.Row():
            for key, label in (
                ("bernini_high_lora_strength", "High-stage Lightx2v strength"),
                ("bernini_low_lora_strength", "Low-stage Lightx2v strength"),
            ):
                controls[key] = gr.Number(
                    label=t(label),
                    minimum=-100,
                    maximum=100,
                    value=defaults.get(key, DEFAULTS[key]),
                )
        controls["bernini_ref_max_size"] = gr.Number(
            label=t("Bernini reference long edge"),
            minimum=16,
            maximum=8192,
            precision=0,
            value=defaults.get("bernini_ref_max_size", 848),
            info=t("Reference proportions are preserved; use a multiple of 16."),
        )
    model = main_model if main_model is not None else base_model

    def bernini_context(name, preset=None):
        checkpoint = (
            effective_preset_checkpoint(name, preset, defaults.get("preset_mode", 7))
            if preset_selection is not None
            else name
        )
        return (
            gr.skip()
            if checkpoint is None
            else gr.update(visible=family_for_checkpoint(checkpoint) == FAMILY)
        )

    gr.on(
        triggers=[model.change]
        + ([preset_selection.change] if preset_selection is not None else [])
        + ([model_refresh.change] if model_refresh is not None else []),
        fn=bernini_context,
        inputs=[model, preset_selection] if preset_selection is not None else [model],
        outputs=panel,
        trigger_mode="always_last",
        show_progress="hidden",
        show_progress_on=[],
        api_visibility="undocumented",
    )
    if model_refresh is not None:

        def refresh_bernini_choices(current):
            choices = low_choices(shared.models)
            if current and current not in choices:
                choices.append(current)
            return gr.update(choices=choices, value=current)

        model_refresh.change(
            refresh_bernini_choices,
            inputs=controls["bernini_low_model"],
            outputs=controls["bernini_low_model"],
            show_progress="hidden",
            api_visibility="undocumented",
        )
    return {key: controls[key] for key in FIELDS}
