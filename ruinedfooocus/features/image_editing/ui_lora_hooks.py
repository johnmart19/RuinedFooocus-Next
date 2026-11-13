"""Contextual advanced controls for an additional scheduled local LoRA."""

import gradio as gr

from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui as t
from ruinedfooocus.features.image_editing import lora_hooks, lora_hook_sources


def local_choices(manager, mode, current, locale=None, source_type="lora"):
    values = [(t("Select a local file"), "")]
    if mode != "off":
        values.extend((name, name) for name in lora_hooks.choices(manager, source_type))
    if current and current not in {value for _, value in values}:
        values.append((localize("Unavailable: {name}", locale, name=current), current))
    return values


def create_ui(base_model, selection, tool_type, model_refresh, kind_for):
    from ruinedfooocus.app import state as shared
    from ruinedfooocus.features.image_editing.ui_guidance import family_for_checkpoint
    from ruinedfooocus.features.models.image_components import ALIASES, IMAGE_COMPONENTS

    saved = lora_hooks.replay_values(shared.settings.default_settings)

    def family(name):
        value = family_for_checkpoint(name)
        return ALIASES.get(value, value)

    initial_family = family(base_model.value)
    mode = saved["lora_hook_mode"]
    controls = {}
    with gr.Accordion(
        t("Scheduled LoRA"), open=False, visible=initial_family in IMAGE_COMPONENTS
    ) as panel:
        controls["lora_hook_mode"] = gr.Dropdown(
            label=t("LoRA schedule"),
            choices=[
                (t("Off"), "off"),
                (t("Manual keyframes"), "keyframes"),
                (t("Interpolated keyframes"), "interpolated"),
                (t("Strength list keyframes"), "floats"),
            ],
            value=mode,
        )
        with gr.Group(visible=mode != "off") as active:
            controls["lora_hook_source_type"] = gr.Dropdown(
                label=t("Scheduled source type"),
                choices=[(t("LoRA weights"), "lora"), (t("Checkpoint as LoRA"), "checkpoint")],
                value=saved["lora_hook_source_type"],
            )
            controls["lora_hook_model"] = gr.Dropdown(
                label=t("Primary scheduled source"),
                choices=local_choices(
                    shared.path_manager,
                    mode,
                    saved["lora_hook_model"],
                    source_type=saved["lora_hook_source_type"],
                ),
                value=saved["lora_hook_model"],
            )
            gr.Markdown(
                t(
                    "Scheduled sources apply after the selected and prompt LoRAs. Checkpoint sources use the native difference from the primary model weights. File compatibility is checked on Run."
                )
            )
            controls["lora_hook_model_strength"] = gr.Slider(
                label=t("Scheduled model strength"),
                minimum=-20,
                maximum=20,
                step=0.01,
                value=saved["lora_hook_model_strength"],
            )
            with gr.Group(
                visible=initial_family in lora_hooks.CLIP_FAMILIES
                or saved["lora_hook_clip_strength"] != 0
            ) as clip:
                clip_help = gr.Markdown(
                    t(
                        "This checkpoint has no external CLIP target. Set CLIP strength to 0 for model-only hooks."
                    ),
                    visible=initial_family not in lora_hooks.CLIP_FAMILIES,
                )
                controls["lora_hook_clip_strength"] = gr.Slider(
                    label=t("Scheduled CLIP strength"),
                    minimum=-20,
                    maximum=20,
                    step=0.01,
                    value=saved["lora_hook_clip_strength"],
                )
                controls["lora_hook_schedule_clip"] = gr.Checkbox(
                    label=t("Apply keyframes to CLIP encoding"),
                    value=saved["lora_hook_schedule_clip"],
                    info=t(
                        "When disabled, CLIP uses its constant strength. Model hooks always use the keyframes."
                    ),
                )
            with gr.Group(visible=mode == "keyframes") as manual:
                controls["lora_hook_keyframes"] = gr.Textbox(
                    label=t("LoRA keyframes JSON"),
                    value=saved["lora_hook_keyframes"],
                    lines=3,
                    max_lines=8,
                    info=t(
                        'Ordered start/strength pairs, beginning at 0. Example: [{"start":0,"strength":1},{"start":0.5,"strength":0}]'
                    ),
                )
            with gr.Group(visible=mode in ("interpolated", "floats")) as range_controls:
                for key, label in (
                    ("lora_hook_start", "First keyframe starts at"),
                    ("lora_hook_end", "Last keyframe starts at"),
                ):
                    controls[key] = gr.Slider(
                        label=t(label), minimum=0, maximum=1, step=0.001, value=saved[key]
                    )
            with gr.Group(visible=mode == "interpolated") as interpolated:
                controls["lora_hook_interpolation"] = gr.Dropdown(
                    label=t("Keyframe interpolation"),
                    choices=[
                        (t(label), value)
                        for label, value in (
                            ("Linear", "linear"),
                            ("Ease in", "ease_in"),
                            ("Ease out", "ease_out"),
                            ("Ease in and out", "ease_in_out"),
                        )
                    ],
                    value=saved["lora_hook_interpolation"],
                )
                for key, label, minimum, maximum in (
                    ("lora_hook_strength_start", "First keyframe multiplier", 0, 10),
                    ("lora_hook_strength_end", "Last keyframe multiplier", 0, 10),
                    ("lora_hook_keyframe_count", "LoRA keyframe count", 2, 100),
                ):
                    controls[key] = gr.Slider(
                        label=t(label),
                        minimum=minimum,
                        maximum=maximum,
                        step=1 if key.endswith("count") else 0.001,
                        value=saved[key],
                    )
            with gr.Group(visible=mode == "floats") as floats:
                controls["lora_hook_strengths"] = gr.Textbox(
                    label=t("LoRA strength list JSON"),
                    value=saved["lora_hook_strengths"],
                    lines=2,
                    info=t(
                        "One through 100 signed multipliers, evenly spaced between the first and last start."
                    ),
                )
            controls["lora_hook_sources"] = gr.Textbox(
                label=t("Additional scheduled sources JSON"),
                value=saved["lora_hook_sources"],
                lines=3,
                max_lines=10,
                info=t(
                    "At most 8 sources total. Each entry needs type (lora or checkpoint) and name; optional model_strength, clip_strength and keyframe settings override the controls above."
                ),
            )
            selected = getattr(shared, "get_ctrl", lambda _: None)("loras")
            if selected is not None:
                copy_selected = gr.Button(
                    t("Copy selected LoRAs into scheduled sources"), size="sm"
                )

                def copy_cards(loras, mode):
                    try:
                        values = lora_hook_sources.selected_sources(loras)
                    except (ValueError, TypeError, IndexError) as error:
                        raise gr.Error(str(error)) from None
                    return (
                        gr.update(value=""),
                        gr.update(value=values),
                        gr.update(value="keyframes" if mode == "off" else mode),
                    )

                copy_selected.click(
                    copy_cards,
                    inputs=[selected, controls["lora_hook_mode"]],
                    outputs=[
                        controls["lora_hook_model"],
                        controls["lora_hook_sources"],
                        controls["lora_hook_mode"],
                    ],
                    api_visibility="undocumented",
                    queue=False,
                )
            gr.Markdown(
                t(
                    "Keyframes hold until the next start. Before the first start, the model uses the first multiplier and scheduled CLIP uses its base strength. The last multiplier continues afterward. Steps and denoise affect which keyframes run."
                )
            )
    for key, component in controls.items():
        shared.add_ctrl(key, component)

    def context(
        checkpoint,
        mode,
        current,
        selection,
        tool,
        clip_strength,
        source_type,
        request: gr.Request = None,
    ):
        current_family = family(checkpoint)
        visible = current_family in IMAGE_COMPONENTS and kind_for(selection, tool) not in (
            "rembg",
            "upscale",
            "detect",
            "segment",
            "depth-map",
        )
        clip_supported = current_family in lora_hooks.CLIP_FAMILIES
        return (
            gr.update(visible=visible),
            gr.update(visible=mode != "off"),
            gr.update(
                choices=local_choices(
                    shared.path_manager, mode, current, resolve_locale(request=request), source_type
                ),
            ),
            gr.update(visible=clip_supported or clip_strength != 0),
            gr.update(visible=mode == "keyframes"),
            gr.update(visible=mode == "interpolated"),
            gr.update(visible=not clip_supported),
            gr.update(visible=mode in ("interpolated", "floats")),
            gr.update(visible=mode == "floats"),
        )

    triggers = [
        base_model.change,
        selection.change,
        tool_type.change,
        controls["lora_hook_mode"].change,
        controls["lora_hook_clip_strength"].input,
        controls["lora_hook_source_type"].change,
    ]
    if model_refresh is not None:
        triggers.append(model_refresh.change)
    gr.on(
        triggers=triggers,
        fn=context,
        inputs=[
            base_model,
            controls["lora_hook_mode"],
            controls["lora_hook_model"],
            selection,
            tool_type,
            controls["lora_hook_clip_strength"],
            controls["lora_hook_source_type"],
        ],
        outputs=[
            panel,
            active,
            controls["lora_hook_model"],
            clip,
            manual,
            interpolated,
            clip_help,
            range_controls,
            floats,
        ],
        show_progress="hidden",
        show_progress_on=[],
        trigger_mode="always_last",
        api_visibility="undocumented",
    )
    return controls
