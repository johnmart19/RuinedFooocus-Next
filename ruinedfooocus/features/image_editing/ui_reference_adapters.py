"""Local image-prompt adapters within the existing Image tools panel."""

import gradio as gr
from pathlib import Path

from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui as t
from ruinedfooocus.core.model_formats import NATIVE_TORCH_SUFFIXES
from ruinedfooocus.features.image_editing.reference_adapters import (
    KINDS,
    local_choices,
    modes_for_family,
)

LABELS = {
    "none": "None",
    "style": "StyleAdapter · SD1",
    "redux": "Redux · Flux",
    "photomaker": "PhotoMaker · SDXL",
}


class ReferenceDropdown(gr.Dropdown):
    is_template = True

    def preprocess(self, payload):
        choices = [value for _, value in self.choices]
        if payload in KINDS and payload not in choices:
            payload = "none"
        return super().preprocess(payload)


def _choices(names, current, *, mode=None, locale=None):
    result = [(t("None"), "")] + [
        (
            name
            + (
                " · " + localize("Compatibility unknown", locale)
                if Path(name).suffix.lower() in NATIVE_TORCH_SUFFIXES
                else ""
            ),
            name,
        )
        for name in names
    ]
    if current and current not in names:
        result.append((localize("Selected file · Unavailable", locale) + f": {current}", current))
    return result


def create_ui(base_model, *, main_model=None, model_refresh=None):
    import ruinedfooocus.app.state as shared
    from ruinedfooocus.features.image_editing.ui_guidance import family_for_checkpoint

    family = family_for_checkpoint(base_model.value)
    modes = modes_for_family(family)
    defaults = shared.settings.default_settings
    selected = defaults.get("reference_adapter", "none")
    selected = selected if selected in modes else "none"
    model = defaults.get("reference_adapter_model", "") if selected != "none" else ""
    vision = defaults.get("reference_adapter_vision", "") if selected != "none" else ""
    controls = {}
    with gr.Accordion(t("Image prompt adapter"), open=False, visible=len(modes) > 1) as panel:
        gr.Markdown(t("Use with None or Image to image; add a separate reference image below."))
        controls["reference_adapter"] = ReferenceDropdown(
            choices=[(t(LABELS[mode]), mode) for mode in modes],
            value=selected,
            label=t("Reference adapter"),
        )
        controls["reference_adapter_model"] = gr.Dropdown(
            choices=_choices(
                local_choices(shared.path_manager, selected, "model"), model, mode=selected
            ),
            value=model,
            label=t("Adapter model"),
            visible=selected != "none",
        )
        controls["reference_adapter_vision"] = gr.Dropdown(
            choices=_choices(local_choices(shared.path_manager, selected, "vision"), vision),
            value=vision,
            label=t("Vision encoder"),
            visible=selected in ("style", "redux"),
        )
        controls["reference_adapter_strength"] = gr.Slider(
            minimum=0,
            maximum=10,
            step=0.01,
            value=defaults.get("reference_adapter_strength")
            if defaults.get("reference_adapter_strength") is not None
            else 1.0,
            label=t("Reference strength"),
            visible=selected in ("style", "redux"),
        )
        controls["reference_adapter_image"] = gr.Image(
            label=t("Reference image"),
            type="pil",
            image_mode="RGBA",
            format="png",
            visible=selected != "none",
            height="clamp(10rem, 24dvh, 20rem)",
        )
        help_text = gr.Markdown(
            t("PhotoMaker needs one photomaker token after the subject word in the prompt."),
            visible=selected == "photomaker",
        )
        refresh = gr.Button(t("Refresh local adapter files"), size="sm")
    for key, control in controls.items():
        shared.add_ctrl(key, control)
    controls["_panel"] = panel
    controls["_context_bound"] = False

    def update_mode(mode, model, vision, request: gr.Request = None):
        locale = resolve_locale(request=request)
        return (
            gr.update(
                choices=_choices(
                    local_choices(shared.path_manager, mode, "model"),
                    model,
                    mode=mode,
                    locale=locale,
                ),
                visible=mode != "none",
            ),
            gr.update(
                choices=_choices(
                    local_choices(shared.path_manager, mode, "vision"), vision, locale=locale
                ),
                visible=mode in ("style", "redux"),
            ),
            gr.update(visible=mode in ("style", "redux")),
            gr.update(visible=mode != "none"),
            gr.update(visible=mode == "photomaker"),
        )

    controls["reference_adapter"].input(
        lambda: ("", "", 1.0),
        outputs=[
            controls["reference_adapter_model"],
            controls["reference_adapter_vision"],
            controls["reference_adapter_strength"],
        ],
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    gr.on(
        triggers=[controls["reference_adapter"].change, refresh.click],
        fn=update_mode,
        inputs=[
            controls["reference_adapter"],
            controls["reference_adapter_model"],
            controls["reference_adapter_vision"],
        ],
        outputs=[
            controls["reference_adapter_model"],
            controls["reference_adapter_vision"],
            controls["reference_adapter_strength"],
            controls["reference_adapter_image"],
            help_text,
        ],
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def context(checkpoint, current):
        modes = modes_for_family(family_for_checkpoint(checkpoint))
        keep = current in modes
        return (
            gr.skip() if controls["_context_bound"] else gr.update(visible=len(modes) > 1),
            gr.update(
                choices=[(t(LABELS[mode]), mode) for mode in modes],
                value=current if keep else "none",
            ),
            *([gr.skip()] * 3 if keep else ["", "", None]),
        )

    model_control = base_model  # Effective checkpoint, including PNG preset overrides.
    gr.on(
        triggers=[model_control.change]
        + ([model_refresh.change] if model_refresh is not None else []),
        fn=context,
        inputs=[model_control, controls["reference_adapter"]],
        outputs=[
            panel,
            controls["reference_adapter"],
            controls["reference_adapter_model"],
            controls["reference_adapter_vision"],
            controls["reference_adapter_image"],
        ],
        show_progress="hidden",
        show_progress_on=[],
        trigger_mode="always_last",
        api_visibility="undocumented",
    )
    return controls


def bind_tool_context(controls, checkpoint, selection, kind, inpaint, resolve):
    """Hide inactive adapters without rewriting the user's saved adapter choices."""
    from ruinedfooocus.features.image_editing.ui_guidance import family_for_checkpoint

    controls["_context_bound"] = True

    def adapter_context(name, selected, tool_type, masked):
        tool = str(resolve(selected, tool_type) or "").lower()
        return gr.update(
            visible=not masked
            and tool in ("", "none", "img2img")
            and len(modes_for_family(family_for_checkpoint(name))) > 1
        )

    controls["_panel"].visible = adapter_context(
        checkpoint.value, selection.value, kind.value, inpaint.value
    )["visible"]
    gr.on(
        triggers=[checkpoint.change, selection.change, kind.change, inpaint.change],
        fn=adapter_context,
        inputs=[checkpoint, selection, kind, inpaint],
        outputs=controls["_panel"],
        trigger_mode="always_last",
        show_progress="hidden",
        api_visibility="undocumented",
    )
