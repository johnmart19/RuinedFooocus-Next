"""Image tools controls for the Qwen Image 2.1 Union model patch."""

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui as t
from ruinedfooocus.features.image_editing.qwen_image21_union import KIND, local_patch, patch_choices


def create_ui(selection, control_type, checkpoint, model_refresh, kind_for, features_for):
    from ruinedfooocus.app.state import add_ctrl, path_manager

    with gr.Group(visible=False) as panel:
        model = gr.Dropdown(
            label=t("Union model patch"),
            info=t("The first job may download the selected patch."),
        )
        mode = gr.Dropdown(
            label=t("Union guide type"),
            choices=[
                (t("Canny"), "canny"),
                (t("Depth"), "depth"),
                (t("Grayscale"), "grayscale"),
                (t("HED"), "hed"),
                (t("Lineart"), "lineart"),
                (t("MLSD"), "mlsd"),
                (t("Pose"), "pose"),
                (t("Scribble"), "scribble"),
                (t("Inpaint only"), "inpaint"),
            ],
            value="canny",
        )
        with gr.Group() as guide_group:
            guide = gr.Image(
                label=t("Union control guide"),
                type="pil",
                format="png",
                sources=["upload", "clipboard"],
            )
            gr.Markdown(
                t("Upload a prepared guide map for the selected type. Grayscale accepts a photo.")
            )
        with gr.Accordion(t("Union inpainting"), open=False) as inpaint:
            gr.Markdown(t("Add a source and mask to edit a region. White redraws and black keeps."))
            with gr.Row():
                source = gr.Image(label=t("Inpaint source"), type="pil", format="png")
                mask = gr.Image(label=t("Inpaint mask"), type="pil", image_mode="L", format="png")
    for name, control in (
        ("model", model),
        ("mode", mode),
        ("guide", guide),
        ("source", source),
        ("mask", mask),
    ):
        add_ctrl("qwen_union_" + name, control)
    signature = gr.State(None)
    add_ctrl("qwen_union_model_signature", signature)

    def context(preset, kind, name, current):
        if kind_for(preset, kind) != KIND or not features_for(name).get("qwen_union"):
            return gr.update(visible=False), gr.skip(), gr.skip()
        choices = patch_choices(path_manager)
        if current and current not in choices:
            choices.append((t("Selected: {name} · Unavailable").format(name=current), current))
        proof = gr.skip() if current else None
        current = current or next(iter(choices), None)
        return gr.update(visible=True), gr.update(choices=choices, value=current), proof

    gr.on(
        triggers=[selection.change, control_type.change, checkpoint.change]
        + ([model_refresh.change] if model_refresh is not None else []),
        fn=context,
        inputs=[selection, control_type, checkpoint, model],
        outputs=[panel, model, signature],
        api_visibility="undocumented",
        show_progress="hidden",
        show_progress_on=[],
        trigger_mode="always_last",
    )

    def bind_selection(name):
        try:
            selected = local_patch(path_manager, name) if name else None
            return list(selected[1]) if selected else None
        except (OSError, ValueError) as error:
            raise gr.Error(str(error)) from None

    model.input(
        bind_selection,
        inputs=model,
        outputs=signature,
        api_visibility="undocumented",
        show_progress="hidden",
    )
    mode.change(
        lambda selected: (
            gr.update(visible=selected != "inpaint"),
            gr.update(open=selected == "inpaint"),
        ),
        inputs=mode,
        outputs=[guide_group, inpaint],
        api_visibility="undocumented",
        show_progress="hidden",
    )
