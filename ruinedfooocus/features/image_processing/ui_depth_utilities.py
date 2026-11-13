"""Contextual Image tools controls for still depth, pose, faces and albedo."""

import gradio as gr

from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui
from ruinedfooocus.features.image_processing.depth_utilities import (
    DEFAULTS,
    KINDS,
    LABELS,
    component_choices,
)

MODEL_LABELS = {
    "lotus-depth": "Lotus depth model",
    "moge-depth": "MoGe model",
    "face-landmarks": "MediaPipe face landmarker",
    "pose-map": "SDPose checkpoint with bundled VAE",
    "marigold-albedo": "Qwen Image Edit2509 backbone",
}
HELP = {
    "lotus-depth": "Create a relative depth map. White is near. The source size and alpha are preserved.",
    "moge-depth": "Create a relative depth map from one photo. Invalid pixels are black. The source size and alpha are preserved.",
    "face-landmarks": "Draw facial landmarks on the source image. The source size and alpha are preserved.",
    "pose-map": "Create a whole-body pose map for one person in the full image. The source size and alpha are preserved.",
    "marigold-albedo": "Estimate sRGB albedo with a source-verified Edit2509 backbone and the official Marigold V2 albedo components. The source size and alpha are preserved.",
}


def choices(manager, kind, field, selected, *, locale=None):
    values = component_choices(manager, kind, field, locale=locale)
    if selected and selected not in {value for _, value in values}:
        values.append((localize("Unavailable: {name}", locale, name=selected), selected))
    return values


def create_ui(selection, control_type, input_image, model_refresh, kind_for):
    import ruinedfooocus.app.state as shared

    saved = shared.settings.default_settings
    kind = kind_for(selection.value, control_type.value)

    def picker(field, label, visible=True):
        selected = saved.get(field, DEFAULTS[field])
        return gr.Dropdown(
            label=translate_ui(label),
            choices=choices(shared.path_manager, kind, field, selected),
            value=selected or None,
            visible=visible,
            info=translate_ui("Select a model or component.")
            if field == "image_utility_model"
            else None,
        )

    with gr.Group(visible=kind in KINDS) as group:
        help_text = gr.Markdown(translate_ui(HELP.get(kind, HELP["lotus-depth"])))
        model = picker("image_utility_model", MODEL_LABELS.get(kind, MODEL_LABELS["lotus-depth"]))
        vae = picker(
            "image_utility_vae",
            "Marigold albedo VAE" if kind == "marigold-albedo" else "SD VAE",
            kind in ("lotus-depth", "marigold-albedo"),
        )
        lora = picker("image_utility_lora", "Marigold albedo LoRA", kind == "marigold-albedo")
        conditioning = picker(
            "image_utility_conditioning", "Marigold albedo conditioning", kind == "marigold-albedo"
        )
        with gr.Group(visible=kind == "moge-depth") as moge:
            detail = gr.Slider(
                label=translate_ui("Depth detail"),
                minimum=0,
                maximum=9,
                step=1,
                value=saved.get("moge_resolution_level", 2),
            )
            refine = gr.Slider(
                label=translate_ui("MoGe 3 refinement passes"),
                minimum=0,
                maximum=8,
                step=1,
                value=saved.get("moge_refine_steps", 3),
                info=translate_ui("Used by MoGe 3 models."),
            )
            from ruinedfooocus.features.image_processing.ui_utility_outputs import moge_controls

            moge_outputs = moge_controls(saved)
        with gr.Group(visible=kind == "face-landmarks") as face:
            variant = gr.Radio(
                label=translate_ui("Face detector range"),
                choices=[
                    (translate_ui("Short"), "short"),
                    (translate_ui("Full"), "full"),
                    (translate_ui("Both"), "both"),
                ],
                value=saved.get("face_detector_variant", "short"),
            )
            face_count = gr.Slider(
                label=translate_ui("Maximum faces"),
                minimum=1,
                maximum=16,
                step=1,
                value=saved.get("face_max_count", 1),
            )
            confidence = gr.Slider(
                label=translate_ui("Face confidence"),
                minimum=0,
                maximum=1,
                step=0.01,
                value=saved.get("face_min_confidence", 0.5),
            )
            from ruinedfooocus.features.image_processing.ui_utility_outputs import face_controls

            face_outputs = face_controls(saved)
        with gr.Group(visible=kind == "pose-map") as pose:
            pose_confidence = gr.Slider(
                label=translate_ui("Pose confidence"),
                minimum=0,
                maximum=1,
                step=0.01,
                value=saved.get("pose_score_threshold", 0.5),
            )

    components = {
        "image_utility_model": model,
        "image_utility_vae": vae,
        "image_utility_lora": lora,
        "image_utility_conditioning": conditioning,
        "moge_resolution_level": detail,
        "moge_refine_steps": refine,
        "face_detector_variant": variant,
        "face_max_count": face_count,
        "face_min_confidence": confidence,
        "pose_score_threshold": pose_confidence,
        **moge_outputs,
        **face_outputs,
    }
    for field, control in components.items():
        shared.add_ctrl(field, control)

    def utility_context(
        choice,
        tool_type,
        model_value,
        vae_value,
        lora_value,
        conditioning_value,
        request: gr.Request = None,
    ):
        selected = (model_value, vae_value, lora_value, conditioning_value)
        locale = resolve_locale(request=request)
        active = kind_for(choice, tool_type)
        pickers = []
        for field, selected_value in zip(tuple(components)[:4], selected):
            visible = (
                active in KINDS
                if field == "image_utility_model"
                else active in ("lotus-depth", "marigold-albedo")
                if field == "image_utility_vae"
                else active == "marigold-albedo"
            )
            update = dict(
                choices=choices(shared.path_manager, active, field, selected_value, locale=locale),
                value=selected_value or None,
                visible=visible,
            )
            if field == "image_utility_model":
                update["label"] = translate_ui(
                    MODEL_LABELS.get(active, MODEL_LABELS["lotus-depth"])
                )
            elif field == "image_utility_vae":
                update["label"] = translate_ui(
                    "Marigold albedo VAE" if active == "marigold-albedo" else "SD VAE"
                )
            pickers.append(gr.update(**update))
        return (
            gr.update(visible=active in KINDS),
            gr.update(value=translate_ui(HELP.get(active, HELP["lotus-depth"]))),
            *pickers,
            gr.update(visible=active == "moge-depth"),
            gr.update(visible=active == "face-landmarks"),
            gr.update(visible=active == "pose-map"),
        )

    gr.on(
        triggers=[selection.change, control_type.change, model_refresh.change],
        fn=utility_context,
        inputs=[selection, control_type, model, vae, lora, conditioning],
        outputs=[group, help_text, model, vae, lora, conditioning, moge, face, pose],
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
        api_visibility="undocumented",
    )
    return components
