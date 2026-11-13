"""Installed compatible control models beside the existing image processors."""

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui as t
from ruinedfooocus.features.image_editing.control_candidates import (
    CONTROL_KINDS,
    compatibility,
    local_candidates,
    model_family,
    selected_control,
)


def create_ui(selection, control_type, base_model, model_refresh=None):
    import ruinedfooocus.app.state as shared
    from ruinedfooocus.features.image_editing import controlnet
    from ruinedfooocus.features.image_editing.ui_guidance import family_for_checkpoint

    saved = shared.settings.default_settings.get("cn_model") or ""
    with gr.Group(visible=False) as panel:
        model = gr.Dropdown(
            label=t("ControlNet / T2I adapter"),
            choices=[(t("Automatic · stock SDXL preset"), "")]
            + ([(t("Saved control model"), saved)] if saved else []),
            value=saved,
            info=t(
                "Select a compatible local model. Automatic keeps the stock SDXL control presets."
            ),
        )
        status = gr.Markdown("")
        refresh = gr.Button(t("Refresh local control models"), size="sm")
    proof = gr.State(shared.settings.default_settings.get("cn_model_signature") if saved else None)
    shared.add_ctrl("cn_model", model)
    shared.add_ctrl("cn_model_signature", proof)

    def context(checkpoint, preset, kind, current, signature, force_refresh=False):
        family = family_for_checkpoint(checkpoint)
        kind = (
            kind.lower()
            if preset == controlnet.NEWCN
            else controlnet.cn_options.get(preset, {}).get("type")
        )
        visible = kind in CONTROL_KINDS and model_family(family) is not None
        if not visible:
            return gr.update(visible=False), gr.skip(), gr.skip()
        rows = local_candidates(shared.path_manager, family, refresh=force_refresh)
        choices = [
            (
                t("Automatic · stock SDXL preset")
                if model_family(family) == "SDXL"
                else t("Select a local control model"),
                "",
            )
        ]
        choices.extend(
            (
                row["name"]
                + (t(" · Compatibility unknown") if row["compatibility"] == "unknown" else ""),
                row["name"],
            )
            for row in rows
        )
        if current and current not in {value for _, value in choices}:
            choices.append((t("Selected: {name} · Unavailable").format(name=current), current))
        message = (
            t("Automatic uses this SDXL preset's published control model.")
            if model_family(family) == "SDXL" and not current
            else t("Choose a local control model before generating.")
        )
        if current:
            try:
                _, _, info = selected_control(shared.path_manager, current, family, signature)
                message = (
                    t("Compatibility unknown · explicitly selected local file.")
                    if compatibility(info, family) == "unknown"
                    else t("{kind} · compatible with the selected checkpoint.").format(
                        kind=info["kind"]
                    )
                )
            except (ValueError, OSError) as error:
                message = str(error)
        return gr.update(visible=True), gr.update(choices=choices, value=current), message

    inputs = [base_model, selection, control_type, model, proof]
    outputs = [panel, model, status]
    gr.on(
        triggers=[base_model.change, selection.change, control_type.change, proof.change]
        + ([model_refresh.change] if model_refresh is not None else []),
        fn=context,
        inputs=inputs,
        outputs=outputs,
        trigger_mode="always_last",
        show_progress="hidden",
        show_progress_on=[],
        api_visibility="undocumented",
    )

    def bind_selection(checkpoint, name):
        if not name:
            return None
        try:
            return selected_control(shared.path_manager, name, family_for_checkpoint(checkpoint))[1]
        except (ValueError, OSError) as error:
            raise gr.Error(str(error)) from None

    model.input(
        bind_selection,
        inputs=[base_model, model],
        outputs=proof,
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def preset_selected(preset):
        values = controlnet.cn_options.get(preset, {})
        if values.get("type") not in CONTROL_KINDS:
            return gr.skip(), gr.skip()
        return values.get("model") or "", values.get("model_signature")

    selection.input(
        preset_selected,
        inputs=selection,
        outputs=[model, proof],
        show_progress="hidden",
        api_visibility="undocumented",
    )
    refresh.click(
        lambda *values: context(*values, force_refresh=True),
        inputs=inputs,
        outputs=outputs,
        show_progress="hidden",
        show_progress_on=[],
        api_visibility="undocumented",
    )
    return model, proof
