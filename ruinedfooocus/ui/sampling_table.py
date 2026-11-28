"""Unified inline-editable Sampling table payload and action handlers."""

from urllib.parse import urlparse

from ruinedfooocus.core.translation_manager import localize
from ruinedfooocus.features.generation.native_sampling import scheduler_choices
from ruinedfooocus.features.models.model_recommendations import FAMILY_PRESETS

SAMPLING_FIELDS = ("custom_steps", "cfg", "sampler_name", "scheduler", "clip_skip")

_SAFE_SOURCE_HOSTS = ("civitai.com", "huggingface.co")

_NUMERIC_BOUNDS = {
    "custom_steps": (1, 200, True),
    "cfg": (0.0, 20.0, False),
    "clip_skip": (1, 5, True),
}


def _safe_source_url(url):
    if not isinstance(url, str) or not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme == "https" and parsed.hostname in _SAFE_SOURCE_HOSTS:
        return url
    return None


def _coerce_value(field, raw, samplers, schedulers):
    if field in _NUMERIC_BOUNDS:
        lo, hi, is_int = _NUMERIC_BOUNDS[field]
        try:
            num = float(raw)
        except (TypeError, ValueError):
            return None
        clamped = min(hi, max(lo, num))
        return int(round(clamped)) if is_int else round(clamped, 1)
    text = str(raw) if raw is not None else ""
    if field == "sampler_name":
        return text if text in samplers else None
    if field == "scheduler":
        valid = set(scheduler_choices(schedulers))
        return text if text in valid else None
    return None


def _values_equal(field, a, b):
    if a is None or b is None:
        return False
    if field in _NUMERIC_BOUNDS:
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (TypeError, ValueError):
            return False
    return str(a) == str(b)


def _format_numeric(field, value):
    if value is None:
        return value
    if field == "cfg":
        try:
            num = round(float(value), 1)
            return int(num) if num.is_integer() else num
        except (TypeError, ValueError):
            return value
    if field in ("custom_steps", "clip_skip"):
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return value
    return value


def _row_source(key, is_custom, details, model_options, family_default, safe_url, locale):
    if is_custom:
        return dict(
            sourceKind="custom",
            sourceLabel=localize("Custom", locale),
            sourceTitle=localize("Custom", locale),
            sourceUrl=None,
        )
    if key in details.get("override", {}):
        return dict(
            sourceKind="override",
            sourceLabel=localize("Override", locale),
            sourceTitle=localize("Your override", locale),
            sourceUrl=None,
        )
    if key in details.get("fetched", {}):
        return dict(
            sourceKind="network",
            sourceLabel=localize("Network", locale),
            sourceTitle=details.get("source") or localize("Network", locale),
            sourceUrl=safe_url,
        )
    if key in details.get("recommended", {}):
        if details.get("native_recipe"):
            return dict(
                sourceKind="network",
                sourceLabel=localize("Model", locale),
                sourceTitle=details.get("source") or localize("Model", locale),
                sourceUrl=safe_url,
            )
        return dict(
            sourceKind="cached",
            sourceLabel=localize("Cached", locale),
            sourceTitle=details.get("source") or localize("Cached metadata", locale),
            sourceUrl=safe_url,
        )
    baseline = model_options.get("_recipe_baseline")
    title = (
        baseline
        if baseline
        else localize("Family preset", locale)
        if family_default
        else localize("App fallback", locale)
    )
    return dict(
        sourceKind="preset",
        sourceLabel=localize("Preset", locale),
        sourceTitle=title,
        sourceUrl=None,
    )


def build_sampling_table_payload(
    performance_settings,
    selection,
    family,
    checkpoint,
    values,
    samplers,
    schedulers,
    locale=None,
):
    """Build the localized JSON payload for the unified inline-editable Sampling table."""
    model_options = performance_settings.control_options(
        performance_settings.MODEL_PERFORMANCE, checkpoint, family
    )
    recipe_unavailable = model_options.get("_recipe_unavailable") or ""
    details = performance_settings.model_recommendations.details(checkpoint, samplers, schedulers)
    family_default = FAMILY_PRESETS.get(family, family) in performance_settings.performance_options
    safe_url = _safe_source_url(details.get("source_url"))
    is_custom_mode = selection == performance_settings.CUSTOM_PERFORMANCE

    sorted_samplers = sorted(samplers)
    sorted_schedulers = list(scheduler_choices(schedulers))

    specs = (
        dict(
            key="custom_steps",
            label=localize("Steps", locale),
            kind="number",
            min=1,
            max=200,
            step=1,
        ),
        dict(
            key="cfg",
            label=localize("CFG", locale),
            kind="number",
            min=0.0,
            max=20.0,
            step=0.1,
        ),
        dict(
            key="sampler_name",
            label=localize("Sampler", locale),
            kind="select",
            choices=sorted_samplers,
        ),
        dict(
            key="scheduler",
            label=localize("Scheduler", locale),
            kind="select",
            choices=sorted_schedulers,
        ),
        dict(
            key="clip_skip",
            label=localize("Clip skip", locale),
            kind="number",
            min=1,
            max=5,
            step=1,
            info=localize(
                "Used by ordinary text encoding. Native image-edit encoders choose their own layers.",
                locale,
            ),
        ),
    )

    rows = []
    for spec in specs:
        key = spec["key"]
        default_val = None if recipe_unavailable else model_options.get(key)
        raw_current = (
            values.get(key)
            if (is_custom_mode or recipe_unavailable or default_val is None)
            else default_val
        )
        current_val = _format_numeric(key, raw_current)
        default_formatted = _format_numeric(key, default_val)
        diverged = (
            not recipe_unavailable
            and default_formatted is not None
            and not _values_equal(key, current_val, default_formatted)
        )
        row_is_custom = is_custom_mode and (bool(recipe_unavailable) or diverged)
        row = dict(
            spec,
            value=current_val,
            defaultValue=default_formatted,
            canReset=bool(diverged),
            **_row_source(
                key,
                row_is_custom,
                details,
                model_options,
                family_default,
                safe_url,
                locale,
            ),
        )
        rows.append(row)

    if not recipe_unavailable and "sampling_shift" in model_options:
        rows.append(
            dict(
                key="sampling_shift",
                label=localize("Sampling shift", locale),
                kind="text",
                value=model_options["sampling_shift"],
                defaultValue=model_options["sampling_shift"],
                canReset=False,
                **_row_source(
                    "sampling_shift",
                    False,
                    details,
                    model_options,
                    family_default,
                    safe_url,
                    locale,
                ),
            )
        )

    return dict(
        headers=dict(
            setting=localize("Setting", locale),
            value=localize("Value", locale),
            source=localize("Source", locale),
        ),
        notice=recipe_unavailable,
        resetLabel=localize("Reset to default", locale),
        rows=rows,
    )


def apply_sampling_table_action(
    payload,
    performance_settings,
    selection,
    family,
    checkpoint,
    current_values,
    samplers,
    schedulers,
):
    """Apply an inline edit or per-row reset from #sampling-table and return (selection, values)."""
    updated = dict(current_values)
    if not isinstance(payload, dict):
        return selection, updated

    action = payload.get("action")
    field = payload.get("field")
    if field not in SAMPLING_FIELDS:
        return selection, updated

    model_options = performance_settings.control_options(
        performance_settings.MODEL_PERFORMANCE, checkpoint, family
    )
    recipe_available = not model_options.get("_recipe_unavailable")

    if action == "reset_field":
        if recipe_available and field in model_options:
            updated[field] = _format_numeric(field, model_options[field])
    elif action == "edit":
        coerced = _coerce_value(field, payload.get("value"), samplers, schedulers)
        if coerced is None:
            return selection, updated
        updated[field] = _format_numeric(field, coerced)
    else:
        return selection, updated

    if recipe_available and all(
        _values_equal(k, updated.get(k), model_options.get(k)) for k in SAMPLING_FIELDS
    ):
        new_selection = performance_settings.MODEL_PERFORMANCE
    else:
        new_selection = performance_settings.CUSTOM_PERFORMANCE

    return new_selection, updated
