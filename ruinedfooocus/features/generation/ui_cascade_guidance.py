"""Explicit native Stage C control and ordered vision reference controls."""

import json
from pathlib import Path
import gradio as gr

from ruinedfooocus.core.model_formats import NATIVE_TORCH_SUFFIXES
from ruinedfooocus.core.translation_manager import localize, translate_ui as t
from ruinedfooocus.features.generation.cascade_guidance import local_choices


def component_choices(manager, role, current):
    from ruinedfooocus.features.image_editing.ui_native_workflows import (
        local_choices as saved_choices,
    )

    names = local_choices(manager, role)
    return [
        (name + " · " + localize("Compatibility unknown"), name)
        if role == "vision" and Path(name).suffix.casefold() in NATIVE_TORCH_SUFFIXES
        else name
        for name in names
    ] + saved_choices([], current if current not in names else None)


def create_ui(base_model, *, preset_selection=None, main_model=None, model_refresh=None):
    import ruinedfooocus.app.state as shared
    from ruinedfooocus.features.image_editing.ui_guidance import family_for_checkpoint
    from ruinedfooocus.features.image_editing.image_guidance import effective_preset_checkpoint

    defaults = shared.settings.default_settings
    family = family_for_checkpoint(base_model.value)
    active = family in ("Stable_Cascade_C", "StableCascade_C")
    controls = {}
    with gr.Accordion(t("Cascade Stage C guidance"), open=False, visible=active) as group:
        gr.Markdown(
            t(
                "Super-resolution uses the native twofold Cascade stage grids. Explicit output sizes keep their exact target; the full source is fitted and padded. Vision references guide Stage C without replacing the prompt."
            )
        )
        controls["cascade_super_resolution"] = gr.Checkbox(
            label=t("Native Cascade super-resolution"),
            value=defaults.get("cascade_super_resolution", False) if active else False,
        )
        controls["cascade_super_resolution_model"] = gr.Dropdown(
            label=t("Cascade super-resolution adapter"),
            choices=component_choices(
                shared.path_manager,
                "control",
                defaults.get("cascade_super_resolution_model"),
            ),
            value=defaults.get("cascade_super_resolution_model"),
        )
        with gr.Row():
            controls["cascade_control_strength"] = gr.Number(
                label=t("Cascade control strength"),
                minimum=0,
                maximum=10,
                value=defaults.get("cascade_control_strength", 1),
                min_width=160,
            )
            controls["cascade_control_denoise"] = gr.Number(
                label=t("Super-resolution Stage C denoise"),
                minimum=0,
                maximum=1,
                value=defaults.get("cascade_control_denoise", 1),
                min_width=160,
            )
        with gr.Row():
            controls["cascade_control_start"] = gr.Number(
                label=t("Cascade control start"),
                minimum=0,
                maximum=1,
                value=defaults.get("cascade_control_start", 0),
                min_width=160,
            )
            controls["cascade_control_end"] = gr.Number(
                label=t("Cascade control end"),
                minimum=0,
                maximum=1,
                value=defaults.get("cascade_control_end", 1),
                min_width=160,
            )
        gr.Markdown(
            t(
                "Add the super-resolution source in Native image edit or Image to image. Ordinary Cascade compression and source denoise apply when super-resolution is disabled."
            )
        )
        controls["cascade_vision_enabled"] = gr.Checkbox(
            label=t("Cascade vision mixing"),
            value=defaults.get("cascade_vision_enabled", False) if active else False,
        )
        controls["cascade_vision_model"] = gr.Dropdown(
            label=t("Cascade CLIP vision model"),
            choices=component_choices(
                shared.path_manager, "vision", defaults.get("cascade_vision_model")
            ),
            value=defaults.get("cascade_vision_model"),
        )
        controls["cascade_vision_images"] = gr.Gallery(
            label=t("Ordered Cascade vision references"),
            value=[],
            type="pil",
            interactive=True,
            columns=4,
            height=180,
        )
        weights = defaults.get("cascade_vision_strengths", [])
        controls["cascade_vision_strengths"] = gr.Textbox(
            label=t("Vision strengths in image order"),
            value=json.dumps(weights) if isinstance(weights, (tuple, list)) else weights,
            info=t(
                "One number per image, separated by commas, from -10 through 10. Blank uses 1 for each image. Zero disables that image's contribution."
            ),
        )
        controls["cascade_vision_crop"] = gr.Dropdown(
            label=t("Native vision framing"),
            choices=[(t("Native full image"), "none"), (t("Native center crop"), "center")],
            value=defaults.get("cascade_vision_crop", "none"),
        )
        gr.Markdown(
            t(
                "Vision images are request inputs. Library setup restores the model and strengths; attach the reference images again in the same order."
            )
        )
    model = main_model if main_model is not None else base_model

    def cascade_guidance_context(name, sr, vision, preset=None):
        checkpoint = (
            effective_preset_checkpoint(name, preset, defaults.get("preset_mode", 7))
            if preset_selection is not None
            else name
        )
        if checkpoint is None:
            return gr.skip(), gr.skip(), gr.skip()
        active = family_for_checkpoint(checkpoint) in ("Stable_Cascade_C", "StableCascade_C")
        return gr.update(visible=active), sr if active else False, vision if active else False

    gr.on(
        triggers=[model.change]
        + ([preset_selection.change] if preset_selection is not None else []),
        fn=cascade_guidance_context,
        inputs=[
            model,
            controls["cascade_super_resolution"],
            controls["cascade_vision_enabled"],
            preset_selection,
        ]
        if preset_selection is not None
        else [model, controls["cascade_super_resolution"], controls["cascade_vision_enabled"]],
        outputs=[group, controls["cascade_super_resolution"], controls["cascade_vision_enabled"]],
        show_progress="hidden",
        api_visibility="undocumented",
        trigger_mode="always_last",
    )
    if model_refresh is not None:

        def refresh_cascade_components(control, vision):
            return tuple(
                gr.update(
                    choices=component_choices(shared.path_manager, role, selected),
                    value=selected,
                )
                for role, selected in (("control", control), ("vision", vision))
            )

        model_refresh.change(
            refresh_cascade_components,
            inputs=[controls["cascade_super_resolution_model"], controls["cascade_vision_model"]],
            outputs=[controls["cascade_super_resolution_model"], controls["cascade_vision_model"]],
            show_progress="hidden",
            api_visibility="undocumented",
        )
    return controls
