"""Contextual named native model patches and ordinary ControlNet recipes."""

import gradio as gr

from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui as t
from ruinedfooocus.features.image_editing import model_patches, native_control_recipes

LABELS = {
    "none": "None",
    "anima-lllite": "Anima LLLite",
    "qwen-diffsynth": "Qwen Image DiffSynth",
    "zimage-fun": "Z-Image Fun",
    "uso-style": "Flux USO style reference",
}


def choices(manager, recipe, selected="", *, vision=False, locale=None):
    options = [(t("Select a local file"), "")]
    if recipe != "none":
        options.extend(
            (
                name
                if status == "compatible"
                else localize("{name} · Native validation on Run", locale, name=name),
                name,
            )
            for name, status in model_patches.local_choices(manager, recipe, vision=vision)
        )
    if selected and selected not in {value for _, value in options}:
        options.append((localize("Unavailable: {name}", locale, name=selected), selected))
    return options


def create_ui(base_model, selection, tool_type, model_refresh, kind_for):
    from ruinedfooocus.app import state as shared
    from ruinedfooocus.features.image_editing.ui_guidance import family_for_checkpoint

    saved = shared.settings.default_settings
    available = model_patches.recipes_for_family(family_for_checkpoint(base_model.value))
    initial = saved.get("model_patch_recipe", "none")
    initial = initial if initial in available else "none"
    controls = {}
    with gr.Accordion(t("Native model patch"), open=False, visible=len(available) > 1) as panel:
        controls["model_patch_recipe"] = gr.Dropdown(
            label=t("Patch recipe"),
            choices=[(t(LABELS[value]), value) for value in available],
            value=initial,
        )
        controls["model_patch_model"] = gr.Dropdown(
            label=t("Model patch"),
            choices=choices(shared.path_manager, initial, saved.get("model_patch_model", "")),
            value=saved.get("model_patch_model", ""),
        )
        controls["model_patch_vision"] = gr.Dropdown(
            label=t("USO vision encoder"),
            choices=choices(
                shared.path_manager,
                initial if initial == "uso-style" else "none",
                saved.get("model_patch_vision", ""),
                vision=True,
            ),
            value=saved.get("model_patch_vision", ""),
            visible=initial == "uso-style",
        )
        with gr.Row(visible=initial != "uso-style") as interval:
            for key, label, minimum, maximum, default in (
                ("model_patch_strength", "Patch strength", -10, 10, 1),
                ("model_patch_start", "Patch starts at", 0, 1, 0),
                ("model_patch_stop", "Patch ends at", 0, 1, 1),
            ):
                controls[key] = gr.Slider(
                    label=t(label),
                    minimum=minimum,
                    maximum=maximum,
                    step=0.01,
                    value=saved.get(key, default),
                )
        controls["model_patch_image"] = gr.Image(
            label=t("Patch guide / style reference"),
            type="pil",
            image_mode="RGBA",
            format="png",
            height=200,
        )
        with gr.Group(visible=initial == "zimage-fun") as source_controls:
            controls["model_patch_inpaint_image"] = gr.Image(
                label=t("Patch inpaint source"),
                type="pil",
                image_mode="RGBA",
                format="png",
                height=200,
            )
            controls["model_patch_denoise"] = gr.Slider(
                label=t("Fun source denoise"),
                minimum=0.01,
                maximum=1.0,
                step=0.01,
                value=saved.get("model_patch_denoise", 1.0),
            )
        controls["model_patch_mask"] = gr.Image(
            label=t("Patch mask · white changes, black preserves"),
            type="pil",
            image_mode="L",
            format="png",
            visible=initial != "uso-style",
            height=180,
        )
        gr.Markdown(
            t(
                "Choose a matching local patch and add its own images. Generate runs the selected recipe."
            )
        )
    for key, component in controls.items():
        shared.add_ctrl(key, component)

    def context(
        checkpoint, recipe, current, vision, tool, control_type, request: gr.Request = None
    ):
        locale = resolve_locale(request=request)
        available = model_patches.recipes_for_family(family_for_checkpoint(checkpoint))
        visible = len(available) > 1 and kind_for(tool, control_type) in (None, "", "img2img")
        effective = recipe if recipe in available else "none"
        updates = [
            gr.update(visible=visible),
            gr.update(choices=[(t(LABELS[value]), value) for value in available], value=effective),
            gr.update(
                choices=choices(shared.path_manager, effective, current, locale=locale),
                value=current,
            ),
            gr.update(
                choices=choices(
                    shared.path_manager,
                    effective if effective == "uso-style" else "none",
                    vision,
                    vision=True,
                    locale=locale,
                ),
                value=vision,
                visible=effective == "uso-style",
            ),
            gr.update(visible=effective != "uso-style"),
            gr.update(visible=effective == "zimage-fun"),
            gr.update(visible=effective not in ("none", "uso-style")),
            *[
                gr.update(value=value) if effective == "uso-style" else gr.skip()
                for value in (1.0, 0.0, 1.0)
            ],
        ]
        return updates

    triggers = [
        base_model.change,
        controls["model_patch_recipe"].change,
        selection.change,
        tool_type.change,
    ]
    if model_refresh is not None:
        triggers.append(model_refresh.change)
    gr.on(
        triggers=triggers,
        fn=context,
        inputs=[
            base_model,
            controls["model_patch_recipe"],
            controls["model_patch_model"],
            controls["model_patch_vision"],
            selection,
            tool_type,
        ],
        outputs=[
            panel,
            controls["model_patch_recipe"],
            controls["model_patch_model"],
            controls["model_patch_vision"],
            interval,
            source_controls,
            controls["model_patch_mask"],
            controls["model_patch_strength"],
            controls["model_patch_start"],
            controls["model_patch_stop"],
        ],
        trigger_mode="always_last",
        show_progress="hidden",
        show_progress_on=[],
        api_visibility="undocumented",
    )

    control_controls = {}
    control_saved = native_control_recipes.replay_values(saved, clear_images=False)
    with gr.Accordion(t("Native ControlNet recipe"), open=False, visible=False) as control_panel:
        control_controls["control_recipe"] = gr.Dropdown(
            label=t("Control recipe"),
            choices=[
                (t("Ordinary ControlNet"), "ordinary"),
                (t("Union mode"), "union"),
                (t("Qwen AliMama inpaint / outpaint"), "alimama-inpaint"),
            ],
            value=control_saved["control_recipe"],
        )
        control_controls["control_union_type"] = gr.Dropdown(
            label=t("Union control type"),
            choices=list(native_control_recipes.union_types()),
            value=control_saved["control_union_type"],
            visible=False,
        )
        control_controls["control_inpaint_mode"] = gr.Radio(
            label=t("AliMama source"),
            choices=[(t("Painted / uploaded mask"), "mask"), (t("Extend canvas"), "outpaint")],
            value=control_saved["control_inpaint_mode"],
            visible=False,
        )
        control_controls["control_inpaint_image"] = gr.Image(
            label=t("AliMama source image"),
            type="pil",
            image_mode="RGBA",
            format="png",
            visible=False,
            height=200,
        )
        control_controls["control_inpaint_mask"] = gr.Image(
            label=t("AliMama mask · white changes, black preserves"),
            type="pil",
            image_mode="L",
            format="png",
            visible=False,
            height=180,
        )
        control_controls["control_inpaint_denoise"] = gr.Slider(
            label=t("AliMama source denoise"),
            minimum=0.01,
            maximum=1.0,
            step=0.01,
            value=control_saved["control_inpaint_denoise"],
            visible=False,
        )
        with gr.Row(visible=False) as outpaint:
            for key, label, maximum, default in (
                ("control_outpaint_left", "Extend left", 1024, 0),
                ("control_outpaint_top", "Extend top", 1024, 0),
                ("control_outpaint_right", "Extend right", 1024, 0),
                ("control_outpaint_bottom", "Extend bottom", 1024, 0),
                ("control_outpaint_feather", "Canvas edge blending", 128, 32),
            ):
                control_controls[key] = gr.Slider(
                    label=t(label),
                    minimum=0,
                    maximum=maximum,
                    step=1,
                    value=saved.get(key, default),
                )
        gr.Markdown(
            t(
                "Choose the matching local ControlNet above. Add this recipe's own source and mask or canvas edges."
            )
        )
    for key, component in control_controls.items():
        shared.add_ctrl(key, component)

    def control_context(checkpoint, tool, kind, recipe, mode):
        from ruinedfooocus.features.image_editing.control_candidates import (
            CONTROL_KINDS,
            model_family,
        )

        family = family_for_checkpoint(checkpoint)
        active = kind_for(tool, kind) in CONTROL_KINDS and model_family(family) is not None
        choices = [(t("Ordinary ControlNet"), "ordinary")]
        if model_family(family) in ("SDXL", "Flux"):
            choices.append((t("Union mode"), "union"))
        if model_family(family) == "QwenImage" and kind_for(tool, kind) == "control":
            choices.append((t("Qwen AliMama inpaint / outpaint"), "alimama-inpaint"))
        recipe = recipe if recipe in {value for _, value in choices} else "ordinary"
        alimama = recipe == "alimama-inpaint"
        return [
            gr.update(visible=active),
            gr.update(choices=choices, value=recipe),
            gr.update(visible=recipe == "union"),
            gr.update(visible=alimama),
            gr.update(visible=alimama),
            gr.update(visible=alimama and mode == "mask"),
            gr.update(visible=alimama and mode == "outpaint"),
            gr.update(visible=alimama),
        ]

    gr.on(
        triggers=[
            base_model.change,
            selection.change,
            tool_type.change,
            control_controls["control_recipe"].change,
            control_controls["control_inpaint_mode"].change,
        ],
        fn=control_context,
        inputs=[
            base_model,
            selection,
            tool_type,
            control_controls["control_recipe"],
            control_controls["control_inpaint_mode"],
        ],
        outputs=[
            control_panel,
            control_controls["control_recipe"],
            control_controls["control_union_type"],
            control_controls["control_inpaint_mode"],
            control_controls["control_inpaint_image"],
            control_controls["control_inpaint_mask"],
            outpaint,
            control_controls["control_inpaint_denoise"],
        ],
        show_progress="hidden",
        trigger_mode="always_last",
        api_visibility="undocumented",
    )
    return controls | control_controls
