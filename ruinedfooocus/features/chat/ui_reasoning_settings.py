"""Persist requested reasoning preferences without claiming runtime support."""

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui as t

SAVED_PREFERENCE = "__saved__"


def conversation_reasoning(mode, effort, values, capability=None):
    """Omit inherited turns; merge partial overrides with supported saved values."""
    if mode == effort == SAVED_PREFERENCE:
        return None
    capability = capability or {}
    values = values or {}
    if mode == SAVED_PREFERENCE:
        saved = values.get("llm_chat_reasoning_mode", "Auto")
        mode = saved if saved in capability.get("modes", ("Auto",)) else "Auto"
    if effort == SAVED_PREFERENCE:
        saved = values.get("llm_chat_reasoning_effort", "Auto")
        effort = saved if saved in ("Auto", *capability.get("efforts", ())) else "Auto"
    return {"mode": mode, "effort": effort}


def preference_controls(capability, chat_mode, chat_effort, rewrite_mode, rewrite_effort):
    capability = capability or {}
    if rewrite_mode is None:
        rewrite_mode = ""
    modes = list(capability.get("modes", ("Auto",)))
    efforts = ["Auto", *capability.get("efforts", ())]
    supported = bool(capability.get("control"))
    computation_optional = supported and "Off" in modes

    def update(choices, value, enabled, *, legacy=False, visible=True):
        choices = list(choices)
        if legacy:
            choices.insert(0, (t("Keep saved rewrite behavior"), ""))
        if value not in {item[1] if isinstance(item, tuple) else item for item in choices}:
            choices.append((t("Saved preference (inactive)") + ": " + str(value), value))
        return gr.update(choices=choices, value=value, interactive=enabled, visible=visible)

    return (
        update(modes, chat_mode, computation_optional, visible=computation_optional),
        update(
            efforts,
            chat_effort,
            bool(capability.get("efforts")) and (chat_mode != "Off" or not computation_optional),
            visible=supported and bool(capability.get("efforts")),
        ),
        update(
            modes, rewrite_mode, computation_optional, legacy=True, visible=computation_optional
        ),
        update(
            efforts,
            rewrite_effort,
            bool(capability.get("efforts"))
            and (rewrite_mode not in ("", "Off") or not computation_optional),
            visible=supported and bool(capability.get("efforts")),
        ),
        gr.update(visible=supported and rewrite_mode == ""),
        gr.update(
            value=t(
                capability.get("reason", "Load a model in Chat to inspect its reasoning controls.")
            ),
            visible=supported,
        ),
        gr.update(visible=supported),
    )


def create_reasoning_settings(values, add_setting, capability):
    with gr.Accordion(
        t("Reasoning preferences"),
        open=False,
        visible=bool((capability.value or {}).get("control")),
    ) as group:
        gr.Markdown(t("Save defaults for future requests. Chat can override its own conversation."))
        controls = []
        for key, label, legacy in (
            ("llm_chat_reasoning_mode", "Chat reasoning computation", False),
            ("llm_chat_reasoning_effort", "Chat reasoning effort", False),
            ("llm_rewrite_reasoning_mode", "Rewrite reasoning computation", True),
            ("llm_rewrite_reasoning_effort", "Rewrite reasoning effort", False),
        ):
            choices = (
                ["Auto", "Off", "On"] if key.endswith("mode") else ["Auto", "Low", "Medium", "High"]
            )
            if legacy:
                choices.insert(0, (t("Keep saved rewrite behavior"), ""))
            control = gr.Dropdown(
                label=t(label),
                choices=choices,
                value=values.get(key) or ("" if legacy else "Auto"),
                interactive=False,
            )
            add_setting(key, control)
            controls.append(control)
        legacy_effort = gr.Dropdown(
            label=t("Saved rewrite effort"),
            choices=["Low", "Medium", "High"],
            value=values.get("llm_rewrite_reasoning", "Low"),
            visible=values.get("llm_rewrite_reasoning_mode") is None,
            info=t("Retains the earlier model-specific rewrite behavior."),
        )
        add_setting("llm_rewrite_reasoning", legacy_effort)
        status = gr.Markdown(t("Load a model in Chat to inspect its reasoning controls."))
        gr.on(
            triggers=[capability.change, controls[0].input, controls[2].input],
            fn=preference_controls,
            inputs=[capability, *controls],
            outputs=[*controls, legacy_effort, status, group],
            show_progress="hidden",
            queue=False,
            api_visibility="undocumented",
        )
    return controls
