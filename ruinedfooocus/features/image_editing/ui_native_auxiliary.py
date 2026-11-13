"""Opt-in native GLIGEN and hypernetwork controls with lazy file discovery."""

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui as t
from ruinedfooocus.features.image_editing.gligen_layout import FAMILIES
from ruinedfooocus.features.image_editing.native_auxiliary_assets import (
    local_choices,
    selected_asset,
)


def create_ui(base_model, model_refresh, inpaint_toggle):
    from ruinedfooocus.app import state as shared
    from ruinedfooocus.features.image_editing.ui import tool_kind
    from ruinedfooocus.features.models.image_components import ALIASES

    selection, control_type = shared.get_ctrl("cn_selection"), shared.get_ctrl("cn_type")
    defaults = shared.settings.default_settings

    def context_caps(name, selected, kind, painting):
        info = shared.models.get_models_by_path("checkpoints", name, fetch=False) if name else {}
        family = shared.models.get_model_base(info)
        ordinary = ALIASES.get(family, family) in FAMILIES
        operation = str(tool_kind(selected, kind) or "").lower()
        sampling = ordinary and operation in (
            "",
            "none",
            "edit",
            "img2img",
            "canny",
            "depth",
            "recolour",
            "sketch",
            "control",
        )
        return sampling, sampling and operation in ("", "none") and not painting

    active, layout_active = context_caps(
        base_model.value, selection.value, control_type.value, inpaint_toggle.value
    )
    initial_names = {
        "gligen": defaults.get("gligen_model") if layout_active else None,
        "hypernetwork": defaults.get("hypernetwork_model") if active else None,
    }

    def initial(role):
        value = initial_names[role]
        return [(t("Off"), None)] + ([(value, value)] if value else [])

    with gr.Accordion(t("Native layout and attention"), open=False, visible=active) as panel:
        gr.Markdown(
            t(
                "Guidance is off until a local model is selected. Expand or refresh to list local files."
            )
        )
        with gr.Group(visible=layout_active) as layout_group:
            gligen = gr.Dropdown(
                label=t("GLIGEN model"),
                choices=initial("gligen"),
                value=initial_names["gligen"],
            )
            layout = gr.Textbox(
                label=t("Layout boxes (JSON pixels)"),
                value=defaults.get("gligen_layout", "[]"),
                lines=3,
                max_lines=8,
            )
            gr.Markdown(
                t(
                    "Describe each box literally with text, x, y, width and height. Use ordered boxes and multiples of eight on a fresh image canvas."
                )
            )
            gr.Markdown(
                t(
                    "GLIGEN layout uses its matching SD checkpoint and CLIP. Keep image tools, staged refiners and automatic upscaling off."
                )
            )
        hypernetwork = gr.Dropdown(
            label=t("Hypernetwork model"),
            choices=initial("hypernetwork"),
            value=initial_names["hypernetwork"],
        )
        strength = gr.Slider(
            label=t("Hypernetwork strength"),
            minimum=-10,
            maximum=10,
            step=0.01,
            value=defaults.get("hypernetwork_strength", 1.0),
        )
        gr.Markdown(
            t(
                "Hypernetworks use matching SD attention dimensions on the primary checkpoint. A handoff refiner keeps its own attention. Signed strength controls the native attention transform."
            )
        )
        refresh_button = gr.Button(t("Refresh guidance files"), size="sm")
    gligen_signature = gr.State(
        defaults.get("gligen_model_signature") if initial_names["gligen"] else None
    )
    hypernetwork_signature = gr.State(
        defaults.get("hypernetwork_model_signature") if initial_names["hypernetwork"] else None
    )
    for key, control in (
        ("gligen_model", gligen),
        ("gligen_model_signature", gligen_signature),
        ("gligen_layout", layout),
        ("hypernetwork_model", hypernetwork),
        ("hypernetwork_model_signature", hypernetwork_signature),
        ("hypernetwork_strength", strength),
    ):
        shared.add_ctrl(key, control)

    def refresh(gligen_name, hypernetwork_name):
        updates = []
        for role, value in (("gligen", gligen_name), ("hypernetwork", hypernetwork_name)):
            names = local_choices(shared.path_manager, role)
            updates.append(
                gr.update(
                    choices=[(t("Off"), None)] + [(name, name) for name in names],
                    value=value if value in names else None,
                )
            )
        return updates

    quiet = dict(queue=False, show_progress="hidden", api_visibility="undocumented")
    gr.on(
        triggers=[panel.expand, refresh_button.click],
        fn=refresh,
        inputs=[gligen, hypernetwork],
        outputs=[gligen, hypernetwork],
        **quiet,
    )

    def bind_identity(component, state, role):
        def identity(name, expected):
            if not name:
                return None
            if isinstance(expected, (list, tuple)) and len(expected) == 4 and expected[0] == name:
                return expected
            return list(selected_asset(shared.path_manager, role, name).public_signature)

        component.change(fn=identity, inputs=[component, state], outputs=state, **quiet)

    bind_identity(gligen, gligen_signature, "gligen")
    bind_identity(hypernetwork, hypernetwork_signature, "hypernetwork")

    def context(name, selected, kind, painting, gligen_name, hypernetwork_name):
        enabled, layout_enabled = context_caps(name, selected, kind, painting)
        return (
            gr.update(visible=enabled),
            gr.update(visible=layout_enabled),
            gr.update(value=gligen_name if layout_enabled else None),
            gr.update(value=hypernetwork_name if enabled else None),
        )

    gr.on(
        triggers=[
            base_model.change,
            model_refresh.change,
            selection.change,
            control_type.change,
            inpaint_toggle.change,
        ],
        fn=context,
        inputs=[base_model, selection, control_type, inpaint_toggle, gligen, hypernetwork],
        outputs=[panel, layout_group, gligen, hypernetwork],
        trigger_mode="always_last",
        show_progress_on=[],
        **quiet,
    )
    return panel
