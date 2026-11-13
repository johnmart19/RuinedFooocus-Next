"""Saved MoE cache preferences beside passive initialized runtime state."""

import gradio as gr

from ruinedfooocus.backends.llama.moe_cache import (
    DEFAULTS,
    moe_cache_cpu_share_capability,
    public_moe_cache_state,
)
from ruinedfooocus.core.translation_manager import (
    localize,
    localized_output,
    resolve_locale,
    translate_ui as t,
)

ENABLED = "llama_moe_cache_cpu_share_enabled"
SHARE = "llama_moe_cache_cpu_share"


def initialized_state():
    """Read an existing service only; never construct a pipeline or inspect a binary."""
    from ruinedfooocus.app.resource_lifecycle import service_pipeline

    state = getattr(service_pipeline("chat"), "moe_cache_cpu_share", None)
    return public_moe_cache_state(state) if isinstance(state, dict) else None


def _source_label(source, locale):
    if source == "native_cli":
        return localize("Extra llama.cpp arguments", locale)
    if source == "native_environment":
        return localize("Native environment", locale)
    return localize("RF preference", locale)


def status_text(runtime, enabled, share, arguments, profile, *, locale=None):
    """Keep draft, persisted intent and actual loaded state distinct."""
    values = dict(
        llm_runtime=runtime, llama_server_args=arguments, **{ENABLED: enabled, SHARE: share}
    )
    try:
        requested = moe_cache_cpu_share_capability(values)
        saved = moe_cache_cpu_share_capability((profile or {}).get("settings", {}))
    except ValueError as error:
        return localize("MoE cache preference is invalid: {error}", locale, error=str(error))
    on = localize("On", locale)
    off = localize("Off", locale)
    lines = [
        localize(
            "Form: {enabled} · share {share}. Saved: {saved_enabled} · share {saved_share}.",
            locale,
            enabled=on if requested["requested_enabled"] else off,
            share=f"{requested['requested_share']:g}",
            saved_enabled=on if saved["requested_enabled"] else off,
            saved_share=f"{saved['requested_share']:g}",
        ),
        localize(
            "Form selected share: {share} · source: {source}.",
            locale,
            share=f"{requested['selected_share']:g}",
            source=_source_label(requested["source"], locale),
        ),
    ]
    loaded = initialized_state()
    if loaded is None:
        lines.append(localize("Initialized Chat: no loaded MoE state is available.", locale))
        lines.append(localize(requested["reason"], locale))
    else:
        lines.append(
            localize(
                "Initialized Chat: {active} · effective share: {share} · source: {source}.",
                locale,
                active=localize("Active", locale)
                if loaded["active"]
                else localize("Inactive", locale),
                share=(
                    f"{loaded['effective_share']:g}"
                    if loaded["effective_share"] is not None
                    else localize("Inactive", locale)
                ),
                source=_source_label(loaded["source"], locale),
            )
        )
        lines.append(localize(loaded["reason"], locale))
    return "\n\n".join(lines)


def restore_moe_preferences(runtime, arguments, profile, request: gr.Request = None):
    """Initialize a fresh browser from its saved profile without resetting other fields."""
    from ruinedfooocus.core.profile_store import read_profile_pair

    profile = dict(profile or {})
    values = dict(profile.get("settings", {}))
    locale = resolve_locale(request=request)
    try:
        if "profile" in profile:
            persisted, _ = read_profile_pair(profile["profile"])
            values.update(persisted)
        saved = moe_cache_cpu_share_capability(values)
    except (OSError, ValueError) as error:
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            localize("MoE cache preference is invalid: {error}", locale, error=str(error)),
        )
    restored = dict(profile, settings=values)
    enabled, share = saved["requested_enabled"], saved["requested_share"]
    return (
        enabled,
        share,
        restored,
        status_text(runtime, enabled, share, arguments, restored, locale=locale),
    )


def create_moe_cache_settings(tab, values, add_setting, runtime, arguments, profile, *, on_load):
    enabled = gr.Checkbox(
        label=t("Enable MoE cache CPU share"),
        value=values.get(ENABLED, DEFAULTS[ENABLED]),
        info=t(
            "Save this preference, then reload Chat. Runtime support and the active share are shown below."
        ),
    )
    add_setting(ENABLED, enabled)
    with gr.Accordion(t("MoE cache share options"), open=False):
        share = gr.Number(
            label=t("CPU share of full-cache misses"),
            value=values.get(SHARE, DEFAULTS[SHARE]),
            minimum=0,
            maximum=1,
            step=0.01,
            info=t(
                "Fraction from 0 to 1. The 0.5 default is RF policy, not a tuning recommendation."
            ),
        )
        add_setting(SHARE, share)
        gr.Markdown(
            t(
                "Uses a supported native runtime, host-resident MoE experts and an existing nonzero MoE GPU cache. Explicit native arguments or environment values take precedence."
            )
        )
    inputs = [runtime, enabled, share, arguments, profile]
    status = gr.Markdown(status_text(*(control.value for control in inputs)))
    refresh = gr.Button(t("Refresh MoE status"), size="sm")
    on_load(
        restore_moe_preferences,
        inputs=[runtime, arguments, profile],
        outputs=[enabled, share, profile, status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def update_moe_status(
        runtime_value, enabled_value, share_value, extra, saved, request: gr.Request = None
    ):
        return status_text(
            runtime_value,
            enabled_value,
            share_value,
            extra,
            saved,
            locale=resolve_locale(request=request),
        )

    gr.on(
        triggers=[
            tab.select,
            enabled.input,
            share.input,
            runtime.change,
            arguments.input,
            profile.change,
            refresh.click,
        ],
        fn=update_moe_status,
        inputs=inputs,
        outputs=status,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    localized_output(lambda locale, *items: status_text(*items, locale=locale), inputs, [status])
    return dict(enabled=enabled, share=share, status=status, refresh=refresh)
