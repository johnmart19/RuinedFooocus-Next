from ruinedfooocus.core.translation_manager import translate_ui

"""Current-image guidance controls; saved startup defaults live in Settings."""

import gradio as gr
from ruinedfooocus.features.image_editing.image_guidance import (
    FIELD_SCHEMA,
    capabilities,
    read_preset_metadata,
    effective_preset_checkpoint,
    restore_guidance_metadata,
    resolve_guidance,
)
from ruinedfooocus.features.generation.image_sampling import LCM_FIELDS
from ruinedfooocus.features.gallery.image_metadata import control_value


def family_for_checkpoint(name):
    import ruinedfooocus.app.state as shared

    info = shared.models.get_models_by_path("checkpoints", name, fetch=False)
    family = shared.models.get_model_base(info)
    path = shared.models.get_file("checkpoints", name)
    if path is not None:
        detected = shared.models.detect_checkpoint_base(path)
        if detected:
            family = detected
    return family


def control_visibility(family):
    available = capabilities(family)
    return (
        available["moss_edit"],
        available["moss_slg"],
        available["sampling_shift"],
        available["cfg_norm"],
    )


def references_visible(family, edit_format):
    available = capabilities(family)
    return available.get("multiple_refs", False) and (
        not available["moss_edit"] or edit_format != "Disabled"
    )


def create_slg_controls(defaults):
    controls = {}
    with gr.Row():
        for key, label in (
            ("moss_image_slg_double_layers", "Double-block layers"),
            ("moss_image_slg_single_layers", "Single-block layers"),
        ):
            controls[key] = gr.Textbox(
                label=label,
                min_width=200,
                value=defaults.get(key, FIELD_SCHEMA[key]["default"]),
                info=translate_ui("Comma-separated block numbers; blank skips none."),
            )
    with gr.Row():
        for key, label in (
            ("moss_image_slg_start", "SLG starts at"),
            ("moss_image_slg_end", "SLG ends at"),
        ):
            controls[key] = gr.Slider(
                label=label,
                minimum=0,
                maximum=1,
                step=0.001,
                min_width=200,
                value=defaults.get(key, FIELD_SCHEMA[key]["default"]),
                info=translate_ui("Fraction of sampling; start must not exceed end."),
            )
    return controls


def create_hidream_controls(defaults, model_control=None):
    controls = {}
    value = defaults.get("hidream_o1_noise_scale")
    controls["hidream_o1_noise_scale"] = gr.Textbox(
        label=translate_ui("Training noise scale"),
        placeholder=translate_ui("Auto"),
        value="" if value is None else str(value),
        info=translate_ui("Auto follows the model recipe. Dev: 7.6."),
    )
    enabled = gr.Dropdown(
        label=translate_ui("Reduce patch seams"),
        choices=[
            (translate_ui("Auto"), "auto"),
            (translate_ui("On"), "on"),
            (translate_ui("Off"), "off"),
        ],
        value=control_value("hidream_o1_seam_smoothing", defaults.get("hidream_o1_seam_smoothing")),
        info=translate_ui("Auto: full On, dev Off. Extra passes cost time."),
    )
    controls["hidream_o1_seam_smoothing"] = enabled

    def options_changed(value, name=None):
        active = resolve_guidance({"hidream_o1_seam_smoothing": value}, {}, "HiDreamO1", name)[
            "hidream_o1_seam_smoothing"
        ]
        return gr.update(visible=active)

    name = model_control.value if model_control is not None else None
    with gr.Accordion(
        translate_ui("Seam options"),
        open=False,
        visible=options_changed(enabled.value, name)["visible"],
    ) as options:
        with gr.Row():
            for key, label in (
                ("hidream_o1_seam_start", "Starts at"),
                ("hidream_o1_seam_end", "Ends at"),
            ):
                controls[key] = gr.Slider(
                    label=label,
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    min_width=200,
                    value=defaults.get(key, FIELD_SCHEMA[key]["default"]),
                    info=translate_ui("Fraction of sampling."),
                )
        for key, label, info in (
            ("hidream_o1_seam_pattern", "Shift pattern", ""),
            ("hidream_o1_seam_passes", "Passes", "More passes cost time and memory."),
            ("hidream_o1_seam_blend", "Blend method", ""),
        ):
            controls[key] = gr.Dropdown(
                label=label,
                choices=list(FIELD_SCHEMA[key]["choices"]),
                value=defaults.get(key, FIELD_SCHEMA[key]["default"]),
                info=info or None,
            )
        controls["hidream_o1_seam_strength"] = gr.Slider(
            label=translate_ui("Smoothing strength"),
            minimum=0,
            maximum=1,
            step=0.01,
            value=defaults.get("hidream_o1_seam_strength", 1),
            info=translate_ui("0 keeps the original; 1 applies full smoothing."),
        )
    enabled.change(
        options_changed,
        inputs=[enabled, model_control] if model_control is not None else enabled,
        outputs=options,
        api_visibility="undocumented",
    )
    return controls, options


def create_lcm_controls(defaults):
    controls = {}
    gr.Markdown(
        translate_ui("LCM only. Noise × training scale; clip 0 disables clipping. Blank: Auto.")
    )
    for key, label in (
        ("lcm_noise", translate_ui("Initial noise")),
        ("lcm_noise_end", translate_ui("Final noise")),
        ("lcm_noise_clip", translate_ui("Noise clip (std)")),
    ):
        controls[key] = gr.Textbox(
            label=label,
            placeholder=translate_ui("Auto"),
            value="" if defaults.get(key) is None else str(defaults[key]),
        )
    return controls


def normalization_auto_default(family, defaults):
    # Old profiles with an explicit blank/None strength retain their Off choice.
    return defaults.get("cfg_norm_auto", family == "Lens" and "cfg_norm_strength" not in defaults)


def qwen_schedule_for_family(value, family):
    if (
        capabilities(family)["qwen_image21"]
        and value in FIELD_SCHEMA["qwen_image21_schedule"]["choices"]
    ):
        return value
    return "auto"


def reset_model_overrides(values, defaults, family=None):
    values.update(
        qwen_image21_schedule=qwen_schedule_for_family(values.get("qwen_image21_schedule"), family),
        sampling_shift=None,
        cfg_norm_strength=None,
        cfg_norm_auto=normalization_auto_default(family, defaults),
        cfg_norm_pre_cfg=False,
        hidream_o1_noise_scale=None,
        hidream_o1_seam_smoothing=defaults.get("hidream_o1_seam_smoothing"),
    )
    values.update(dict.fromkeys(LCM_FIELDS))


def preset_control_state(
    current, defaults, metadata, mode, family, checkpoint, main_checkpoint, previous
):
    """Pure preset transition: never store reference images in backup state."""
    if metadata is None:
        return None
    current = {
        key: current.get(key, defaults.get(key, spec["default"]))
        for key, spec in FIELD_SCHEMA.items()
    }
    previous = previous or {"checkpoint": main_checkpoint, "backup": None}
    locked = bool(metadata and mode & 2)
    backup = previous.get("backup")
    values = dict(current)
    changed = previous.get("checkpoint") != checkpoint
    if locked:
        if backup is None:
            backup = {"checkpoint": main_checkpoint, "values": dict(current)}
        values = dict(backup["values"])
        if backup["checkpoint"] != checkpoint:
            reset_model_overrides(values, defaults, family)
        values.update(restore_guidance_metadata(metadata, family=family))
    elif backup is not None:
        values = (
            dict(backup["values"])
            if backup["checkpoint"] == main_checkpoint
            else {key: defaults.get(key, spec["default"]) for key, spec in FIELD_SCHEMA.items()}
        )
        if backup["checkpoint"] != main_checkpoint:
            reset_model_overrides(values, defaults, family)
        backup = None
    elif changed:
        reset_model_overrides(values, defaults, family)
    return {
        "values": values,
        "locked": locked,
        "clear_references": locked or changed or previous.get("backup") is not None,
        "state": {"checkpoint": checkpoint, "backup": backup},
    }


def create_ui(
    base_model,
    preset_selection=None,
    main_model=None,
    model_refresh=None,
    *,
    tool_selection=None,
    tool_type=None,
    kind_for=None,
):
    import ruinedfooocus.app.state as shared

    defaults = shared.settings.default_settings
    family = family_for_checkpoint(base_model.value)
    initial = control_visibility(family)
    tool_inputs = (
        [tool_selection, tool_type] if tool_selection is not None and tool_type is not None else []
    )

    def generation_active(selection=None, kind=None):
        return not tool_inputs or str(kind_for(selection, kind) or "").lower() not in (
            "upscale",
            "rembg",
            "detect",
            "segment",
        )

    preset_notice = gr.Markdown("", visible=False)
    with gr.Accordion(
        translate_ui("Image guidance"),
        open=False,
        visible=generation_active(*[control.value for control in tool_inputs]),
    ) as guidance_panel:
        gr.Markdown(
            translate_ui(
                "These controls apply to the current image without restarting. "
                "Select a checkpoint under Models to see its supported options."
            )
        )
        qwen_schedule = gr.Dropdown(
            label=translate_ui("Qwen sampling schedule"),
            choices=[
                (translate_ui("Auto"), "auto"),
                (translate_ui("Standard"), "standard"),
                (translate_ui("Turbo 8-step"), "turbo_8"),
            ],
            value=defaults.get("qwen_image21_schedule", "auto"),
            visible=capabilities(family)["qwen_image21"],
            info=translate_ui(
                "Auto follows Model Defaults. Standard uses the selected scheduler. "
                "Turbo 8-step requires 8 steps, CFG 1, Euler and Simple."
            ),
        )
        with gr.Group(visible=initial[0]) as moss_edit:
            edit_format = gr.Dropdown(
                label=translate_ui("MOSS edit format"),
                choices=list(FIELD_SCHEMA["moss_image_edit_format"]["choices"]),
                value=defaults.get("moss_image_edit_format", "Disabled"),
                info=translate_ui(
                    "Requires an input image and a matching edit LoRA. Disabled uses ordinary image generation."
                ),
            )
            strength = gr.Slider(
                label=translate_ui("Reference attention strength"),
                minimum=0,
                maximum=100,
                step=0.05,
                value=defaults.get("moss_image_edit_strength", 1.0),
                info=translate_ui(
                    "1 is normal; 0 omits vision conditioning. Applies to ai-toolkit formats."
                ),
            )
        slg = gr.Checkbox(
            label=translate_ui("MOSS skip-layer guidance"),
            visible=initial[1],
            value=defaults.get("moss_image_slg", False),
            info=translate_ui("Optional extra guidance. Off follows the standard MOSS workflow."),
        )
        with gr.Accordion(
            translate_ui("Skip-layer options"), open=False, visible=initial[1] and slg.value
        ) as slg_options:
            slg_controls = create_slg_controls(defaults)

        def slg_changed(name, enabled):
            return gr.update(
                visible=enabled and capabilities(family_for_checkpoint(name))["moss_slg"]
            )

        slg.change(
            slg_changed,
            inputs=[base_model, slg],
            outputs=slg_options,
            api_visibility="undocumented",
        )
        shift = gr.Textbox(
            label=translate_ui("Sampling shift"),
            placeholder=translate_ui("Auto"),
            value="",
            visible=initial[2],
            info=translate_ui(
                "Leave blank for the model default. An override applies only to this generation."
            ),
        )
        norm_auto = gr.Checkbox(
            label=translate_ui("Use model CFG normalization"),
            value=normalization_auto_default(family, defaults),
            visible=family == "Lens",
            info=translate_ui(
                "Use the selected model recipe. Enter a strength for a custom value; unchecked and blank disables normalization."
            ),
        )
        norm = gr.Textbox(
            label=translate_ui("CFG normalization strength"),
            placeholder=translate_ui("Off"),
            value="",
            visible=initial[3],
            info=translate_ui(
                "Custom native CFG normalization. Entering a value turns off model normalization; unchecked and blank disables it."
            ),
        )
        norm.input(
            lambda: False,
            outputs=norm_auto,
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )
        norm_auto.input(
            lambda active: gr.update(value="") if active else gr.update(),
            inputs=norm_auto,
            outputs=norm,
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )
        norm_pre = gr.Checkbox(
            label=translate_ui("Normalize before CFG"),
            visible=initial[3],
            value=defaults.get("cfg_norm_pre_cfg", False),
            info=translate_ui("Used when normalization strength is set. May amplify guidance."),
        )
        with gr.Accordion(
            translate_ui("HiDream O1"), open=False, visible=capabilities(family)["hidream_o1"]
        ) as hidream_panel:
            hidream_controls, hidream_options = create_hidream_controls(defaults, base_model)
        with gr.Accordion(translate_ui("LCM noise"), open=False, visible=initial[3]) as lcm_panel:
            lcm_controls = create_lcm_controls(defaults)
        from ruinedfooocus.features.image_editing.image_geometry import FRAMING_CHOICES

        framing = gr.Dropdown(
            label=translate_ui("Image framing"),
            choices=FRAMING_CHOICES,
            value=defaults.get("image_resize_mode", "preserve"),
            visible=initial[3],
            info=translate_ui(
                "Preserve keeps the source proportions and only downscales if needed. Pad keeps the whole source inside the exact selected size. Crop is an explicit framing choice; none of these modes stretch the image."
            ),
        )
        with gr.Accordion(
            translate_ui("VAE decoding"), open=False, visible=initial[3]
        ) as vae_panel:
            vae_tiled = gr.Checkbox(
                label=translate_ui("Always use tiled decoding"),
                value=defaults.get("vae_tiled", False),
                info=translate_ui(
                    "Off uses automatic VRAM fitting. Tiling can introduce slight seams."
                ),
            )
            with gr.Row():
                vae_size = gr.Textbox(
                    label=translate_ui("VAE tile size"),
                    placeholder=translate_ui("Auto"),
                    min_width=200,
                    value=""
                    if defaults.get("vae_tile_size") is None
                    else str(defaults["vae_tile_size"]),
                    info=translate_ui(
                        "Pixels (64–4096, step 32). Auto fits VRAM; smaller tiles save memory."
                    ),
                )
                vae_overlap = gr.Textbox(
                    label=translate_ui("VAE tile overlap"),
                    placeholder=translate_ui("Auto"),
                    min_width=200,
                    value=""
                    if defaults.get("vae_overlap") is None
                    else str(defaults["vae_overlap"]),
                    info=translate_ui(
                        "Pixels (0–4096, step 32). Auto blends edges; capped to fit the tile."
                    ),
                )
        references = gr.Gallery(
            label=translate_ui("Additional references (2–16, upload order)"),
            value=[],
            interactive=True,
            type="pil",
            file_types=["image"],
            sources=["upload"],
            columns=3,
            height=180,
            buttons=[],
            format="png",
            visible=references_visible(family, edit_format.value),
        )
        controls = dict(
            qwen_image21_schedule=qwen_schedule,
            moss_image_edit_format=edit_format,
            moss_image_edit_strength=strength,
            moss_image_slg=slg,
            **slg_controls,
            sampling_shift=shift,
            cfg_norm_strength=norm,
            cfg_norm_auto=norm_auto,
            cfg_norm_pre_cfg=norm_pre,
            image_resize_mode=framing,
            **hidream_controls,
            **lcm_controls,
            vae_tiled=vae_tiled,
            vae_tile_size=vae_size,
            vae_overlap=vae_overlap,
            reference_images=references,
        )
        for key, component in controls.items():
            shared.add_ctrl(key, component)

    def model_changed(name, current_format, current_slg=False, current_schedule="auto"):
        family = family_for_checkpoint(name)
        edit, use_slg, use_shift, use_norm = control_visibility(family)
        smoothing = defaults.get("hidream_o1_seam_smoothing")
        active = resolve_guidance({"hidream_o1_seam_smoothing": smoothing}, {}, "HiDreamO1", name)[
            "hidream_o1_seam_smoothing"
        ]
        # Do not carry model-specific numerical overrides into a different model.
        return (
            gr.update(visible=edit),
            gr.update(visible=use_slg),
            gr.update(visible=use_shift, value=""),
            gr.update(visible=use_norm, value=""),
            gr.update(visible=references_visible(family, current_format), value=[]),
            gr.update(visible=use_norm),
            gr.update(visible=use_norm),
            gr.update(visible=use_slg and current_slg),
            gr.update(visible=use_norm, value=False),
            gr.update(visible=capabilities(family)["hidream_o1"]),
            gr.update(visible=active),
            gr.update(value=""),
            gr.update(value=control_value("hidream_o1_seam_smoothing", smoothing)),
            gr.update(visible=use_norm),
            *[gr.update(value="") for key in LCM_FIELDS],
            gr.update(visible=family == "Lens", value=normalization_auto_default(family, defaults)),
            gr.update(
                visible=capabilities(family)["qwen_image21"],
                value=qwen_schedule_for_family(current_schedule, family),
            ),
        )

    if preset_selection is None:
        gr.on(
            triggers=[base_model.change]
            + ([model_refresh.change] if model_refresh is not None else []),
            fn=model_changed,
            inputs=[base_model, edit_format, slg, qwen_schedule],
            outputs=[
                moss_edit,
                slg,
                shift,
                norm,
                references,
                framing,
                vae_panel,
                slg_options,
                norm_pre,
                hidream_panel,
                hidream_options,
                hidream_controls["hidream_o1_noise_scale"],
                hidream_controls["hidream_o1_seam_smoothing"],
                lcm_panel,
                *[controls[key] for key in LCM_FIELDS],
                norm_auto,
                qwen_schedule,
            ],
            api_visibility="undocumented",
            show_progress="hidden",
        )
        if tool_inputs:
            gr.on(
                triggers=[control.change for control in tool_inputs]
                + ([model_refresh.change] if model_refresh is not None else []),
                fn=lambda selection, kind: gr.update(visible=generation_active(selection, kind)),
                inputs=tool_inputs,
                outputs=guidance_panel,
                show_progress="hidden",
                api_visibility="undocumented",
            )
    else:
        main_model = main_model or base_model
        state = gr.State({"checkpoint": main_model.value, "backup": None})
        fields = [controls[key] for key in FIELD_SCHEMA]
        outputs = (
            [guidance_panel, preset_notice, moss_edit]
            + fields
            + [references, vae_panel, slg_options, hidream_panel, hidream_options, lcm_panel, state]
        )

        def context_changed(main_checkpoint, preset, previous, *values):
            active_tool = generation_active(*values[-2:]) if tool_inputs else True
            values = values[:-2] if tool_inputs else values
            metadata = read_preset_metadata(preset)
            mode = shared.settings.default_settings.get("preset_mode", 7)
            checkpoint = effective_preset_checkpoint(main_checkpoint, preset, mode)
            if checkpoint is None:
                return [gr.update() for _ in outputs]
            family = family_for_checkpoint(checkpoint)
            transition = preset_control_state(
                dict(zip(FIELD_SCHEMA, values)),
                defaults,
                metadata,
                mode,
                family,
                checkpoint,
                main_checkpoint,
                previous,
            )
            if transition is None:
                return [gr.update() for _ in outputs]
            selected = transition["values"]
            available = capabilities(family)
            updates = [
                gr.update(visible=active_tool and not transition["locked"]),
                gr.update(
                    visible=active_tool and transition["locked"],
                    value="Preset controls image guidance. Clear the preset to edit these controls.",
                ),
                gr.update(visible=available["moss_edit"]),
            ]
            for key in FIELD_SCHEMA:
                visible = (
                    available["qwen_image21"]
                    if key == "qwen_image21_schedule"
                    else family == "Lens"
                    if key == "cfg_norm_auto"
                    else available["hidream_o1"]
                    if key.startswith("hidream_o1_")
                    else available["moss_slg"]
                    if key.startswith("moss_image_slg")
                    else available["sampling_shift"]
                    if key == "sampling_shift"
                    else available["cfg_norm"]
                    if key
                    in (
                        "cfg_norm_strength",
                        "cfg_norm_pre_cfg",
                        "image_resize_mode",
                        "vae_tiled",
                        "vae_tile_size",
                        "vae_overlap",
                        *LCM_FIELDS,
                    )
                    else True
                )
                updates.append(gr.update(value=control_value(key, selected[key]), visible=visible))
            updates.append(
                gr.update(
                    visible=not transition["locked"]
                    and references_visible(family, selected["moss_image_edit_format"]),
                    **({"value": []} if transition["clear_references"] else {}),
                )
            )
            updates.append(gr.update(visible=available["cfg_norm"]))
            updates.append(gr.update(visible=available["moss_slg"] and selected["moss_image_slg"]))
            updates.append(gr.update(visible=available["hidream_o1"]))
            active = resolve_guidance(
                {"hidream_o1_seam_smoothing": selected["hidream_o1_seam_smoothing"]},
                {},
                "HiDreamO1",
                checkpoint,
            )["hidream_o1_seam_smoothing"]
            updates.append(gr.update(visible=active))
            updates.append(gr.update(visible=available["cfg_norm"]))
            updates.append(transition["state"])
            return updates

        gr.on(
            triggers=[main_model.change, preset_selection.change]
            + ([model_refresh.change] if model_refresh is not None else [])
            + [control.change for control in tool_inputs],
            fn=context_changed,
            inputs=[main_model, preset_selection, state] + fields + tool_inputs,
            outputs=outputs,
            concurrency_id="image_guidance_context",
            concurrency_limit=1,
            api_visibility="undocumented",
            show_progress="hidden",
            show_progress_on=[],
        )

    def format_changed(name, value):
        visible = references_visible(family_for_checkpoint(name), value)
        return gr.update(visible=visible, **({} if visible else {"value": []}))

    edit_format.change(
        format_changed,
        inputs=[base_model, edit_format],
        outputs=references,
        api_visibility="undocumented",
    )
    return controls
