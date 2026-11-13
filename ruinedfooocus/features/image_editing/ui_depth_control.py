from ruinedfooocus.core.translation_manager import translate_ui

"""Photo-to-depth options for compatible native depth ControlNets."""

import gradio as gr
import ruinedfooocus.app.state as shared
import ruinedfooocus.features.image_editing.controlnet as controlnet
from ruinedfooocus.features.image_editing.depth_control import (
    DEPTH_DEFAULTS,
    DEPTH_CHOICES,
    depth_settings,
)
from ruinedfooocus.features.image_editing.lotus_depth import local_models, local_vaes


def create_ui(selection, control_type, checkpoint, checkpoint_features):
    defaults = shared.settings.default_settings
    saved_model = defaults.get("cn_depth_lotus_model", "")
    saved_vae = defaults.get("cn_depth_lotus_vae", "")

    def initial_choices(names, current):
        choices = [(translate_ui("Select a local file"), "")] + [(name, name) for name in names]
        if current and current not in names:
            choices.append((translate_ui("Selected file · Unavailable") + ": " + current, current))
        return choices

    with gr.Group(visible=False) as panel:
        mode = gr.Dropdown(
            label=translate_ui("Depth input"),
            value=defaults.get("cn_depth_preprocess", "auto"),
            choices=[
                (translate_ui("Estimate from photo"), "auto"),
                (translate_ui("Use supplied depth map"), "none"),
            ],
            info=translate_ui("Estimate downloads once. Depth maps: white = near."),
        )
        model = gr.Dropdown(
            label=translate_ui("Depth model"),
            value=defaults.get("cn_depth_model", "auto"),
            choices=[
                (translate_ui("Auto"), "auto"),
                (translate_ui("DA3 Small (137 MB)"), "small"),
                (translate_ui("DA3 Base (542 MB)"), "base"),
                (translate_ui("Lotus · installed local model"), "lotus"),
            ],
            info=translate_ui("Auto: Base at 8+ GB VRAM; Small on CPU or smaller GPUs."),
        )
        with gr.Group(visible=False) as lotus:
            lotus_model = gr.Dropdown(
                label=translate_ui("Lotus depth checkpoint"),
                choices=initial_choices(
                    local_models(shared.path_manager) if saved_model else [], saved_model
                ),
                value=saved_model,
            )
            lotus_vae = gr.Dropdown(
                label=translate_ui("Lotus SD VAE"),
                choices=initial_choices(
                    local_vaes(shared.path_manager) if saved_vae else [], saved_vae
                ),
                value=saved_vae,
                info=translate_ui(
                    "Both local files are required. Lotus uses its fixed native one-step depth recipe."
                ),
            )
            refresh = gr.Button(translate_ui("Refresh local Lotus files"), size="sm")
        with gr.Accordion(translate_ui("DA3 options"), open=False) as advanced:
            resolution = gr.Slider(
                label=translate_ui("Depth resolution"),
                minimum=140,
                maximum=2520,
                step=14,
                value=504,
                info=translate_ui("More detail and VRAM; keeps input size."),
            )
            resize = gr.Dropdown(
                label=translate_ui("Depth resize method"),
                value="upper_bound_resize",
                choices=[
                    (translate_ui("Fit longest side"), "upper_bound_resize"),
                    (translate_ui("Fit shortest side (more VRAM)"), "lower_bound_resize"),
                ],
            )
            norm = gr.Dropdown(
                label=translate_ui("Depth contrast"),
                value="v2_style",
                choices=[
                    (translate_ui("Balanced"), "v2_style"),
                    (translate_ui("Full range"), "min_max"),
                ],
            )
            dtype = gr.Dropdown(
                label=translate_ui("Depth precision"),
                value="default",
                choices=list(DEPTH_CHOICES["depth_dtype"]),
                info=translate_ui("Default follows the native backend and device."),
            )
    controls = dict(
        depth_preprocess=mode,
        depth_model=model,
        depth_resolution=resolution,
        depth_resize_method=resize,
        depth_normalization=norm,
        depth_dtype=dtype,
        depth_lotus_model=lotus_model,
        depth_lotus_vae=lotus_vae,
    )
    for key, component in controls.items():
        shared.add_ctrl("cn_" + key, component)

    def context_changed(
        preset, kind, name, mode="auto", model="auto", current_model="", current_vae=""
    ):
        values = (
            DEPTH_DEFAULTS if preset == controlnet.NEWCN else controlnet.cn_options.get(preset, {})
        )
        is_depth = (kind.lower() if preset == controlnet.NEWCN else values.get("type")) == "depth"
        features = checkpoint_features(name)
        visible = (features["sdxl"] or features.get("control")) and is_depth
        estimate = visible and mode == "auto"
        use_lotus = estimate and model == "lotus"

        def choices(names, current):
            options = [(translate_ui("Select a local file"), "")] + [(name, name) for name in names]
            if current and current not in names:
                options.append(
                    (translate_ui("Selected file · Unavailable") + ": " + current, current)
                )
            return gr.update(choices=options, value=current)

        return [
            gr.update(visible=visible),
            gr.update(visible=estimate and not use_lotus),
            gr.update(visible=use_lotus),
            choices(local_models(shared.path_manager), current_model) if use_lotus else gr.skip(),
            choices(local_vaes(shared.path_manager), current_vae) if use_lotus else gr.skip(),
        ]

    def preset_values(preset):
        values = (
            DEPTH_DEFAULTS if preset == controlnet.NEWCN else controlnet.cn_options.get(preset, {})
        )
        if preset != controlnet.NEWCN and values.get("type") != "depth":
            return [gr.skip() for _ in controls]
        options = depth_settings(values)
        return [gr.update(value=options[key]) for key in controls]

    selection.input(
        preset_values,
        inputs=selection,
        outputs=list(controls.values()),
        show_progress="hidden",
        api_visibility="undocumented",
    )
    context_inputs = [selection, control_type, checkpoint, mode, model, lotus_model, lotus_vae]

    gr.on(
        triggers=[
            selection.change,
            control_type.change,
            checkpoint.change,
            mode.change,
            model.change,
            refresh.click,
        ],
        fn=context_changed,
        inputs=context_inputs,
        outputs=[panel, advanced, lotus, lotus_model, lotus_vae],
        concurrency_id="depth_context",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    return controls
