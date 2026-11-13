"""Finite named source lists for native scheduled LoRA/checkpoint hooks."""

from dataclasses import dataclass
import json
import math

DEFAULTS = {
    "lora_hook_source_type": "lora",
    "lora_hook_sources": "[]",
    "lora_hook_strengths": "[1,0]",
}
SOURCE_TYPES = ("lora", "checkpoint")
SOURCE_FIELDS = {
    "model_strength": "lora_hook_model_strength",
    "clip_strength": "lora_hook_clip_strength",
    "mode": "lora_hook_mode",
    "keyframes": "lora_hook_keyframes",
    "strengths": "lora_hook_strengths",
    "interpolation": "lora_hook_interpolation",
    "start": "lora_hook_start",
    "end": "lora_hook_end",
    "strength_start": "lora_hook_strength_start",
    "strength_end": "lora_hook_strength_end",
    "count": "lora_hook_keyframe_count",
}


@dataclass(frozen=True)
class SourceSpec:
    kind: str
    name: str
    values: tuple
    keyframes: tuple

    def settings(self):
        return dict(self.values)


@dataclass(frozen=True)
class CapturedSource:
    spec: SourceSpec
    artifact: object

    def settings(self):
        return self.spec.settings()

    @property
    def keyframes(self):
        return self.spec.keyframes

    @property
    def category(self):
        return "checkpoints" if self.spec.kind == "checkpoint" else "loras"


def settings(values, validate, number, *, sources=True):
    result = {key: values.get(key, default) for key, default in DEFAULTS.items()}
    if result["lora_hook_source_type"] not in SOURCE_TYPES:
        raise ValueError("Choose LoRA weights or a checkpoint as the scheduled source.")
    text = result["lora_hook_sources"]
    if not isinstance(text, str) or len(text) > 131072:
        raise ValueError("Scheduled sources must be a bounded JSON list of at most 8 sources.")
    try:
        rows = json.loads(text) if sources else []
    except ValueError:
        raise ValueError(
            "Scheduled sources must be a JSON list of named LoRA/checkpoint sources."
        ) from None
    if not isinstance(rows, list) or len(rows) > 8:
        raise ValueError("Choose at most 8 scheduled sources including the primary source.")
    canonical = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) - {"type", "name", *SOURCE_FIELDS}
            or not {"type", "name"} <= set(row)
        ):
            raise ValueError(
                "Each scheduled source needs type/name and optional named strength/keyframe settings."
            )
        if row["type"] not in SOURCE_TYPES:
            raise ValueError("Each scheduled source type must be lora or checkpoint.")
        item = dict(values, lora_hook_model=row["name"])
        if item["lora_hook_mode"] == "off" and "mode" not in row:
            item["lora_hook_mode"] = "keyframes"
        for source, target in SOURCE_FIELDS.items():
            if source in row:
                item[target] = (
                    json.dumps(row[source]) if source in ("keyframes", "strengths") else row[source]
                )
        normalized, _ = validate(item, _sources=False)
        if not normalized["lora_hook_model"]:
            raise ValueError("Every scheduled source needs a local filename.")
        if normalized["lora_hook_mode"] == "off":
            raise ValueError("Choose a keyframe mode for each additional scheduled source.")
        canonical.append(dict(row, name=normalized["lora_hook_model"]))
    if len(rows) + bool(values.get("lora_hook_model")) > 8:
        raise ValueError("Choose at most 8 scheduled sources including the primary source.")
    if sources:
        result["lora_hook_sources"] = json.dumps(canonical, separators=(",", ":"))
    text = result["lora_hook_strengths"]
    if not isinstance(text, str) or len(text) > 8192:
        raise ValueError("LoRA strength values must be a JSON list of 1 to 100 numbers.")
    try:
        strengths = json.loads(text)
    except ValueError:
        raise ValueError("LoRA strength values must be a JSON list of finite numbers.") from None
    if not isinstance(strengths, list) or not 1 <= len(strengths) <= 100:
        raise ValueError("LoRA strength values require 1 to 100 numbers.")
    strengths = [number(value, -20, 20, "LoRA strength value") for value in strengths]
    result["lora_hook_strengths"] = json.dumps(strengths, separators=(",", ":"))
    return result


def specs(values, validate):
    result = []
    if values["lora_hook_model"]:
        normalized, keyframes = validate(values, _sources=False)
        result.append(
            SourceSpec(
                values["lora_hook_source_type"],
                values["lora_hook_model"],
                tuple(normalized.items()),
                keyframes,
            )
        )
    for row in json.loads(values["lora_hook_sources"]):
        item = dict(values, lora_hook_model=row["name"])
        for source, target in SOURCE_FIELDS.items():
            if source in row:
                item[target] = (
                    json.dumps(row[source]) if source in ("keyframes", "strengths") else row[source]
                )
        normalized, keyframes = validate(item, _sources=False)
        result.append(SourceSpec(row["type"], row["name"], tuple(normalized.items()), keyframes))
    return tuple(result)


def selected_sources(loras):
    """Copy names/weights from existing cards; leave card selection and prompt intact."""
    if not isinstance(loras, (list, tuple)) or not 1 <= len(loras) <= 8:
        raise ValueError("Select between 1 and 8 LoRA cards to copy into scheduled sources.")
    result = []
    for item in loras:
        caption = (
            item.get("caption")
            if isinstance(item, dict)
            else item[1]
            if isinstance(item, (list, tuple)) and len(item) == 2
            else None
        )
        if not isinstance(caption, str) or " - " not in caption:
            raise ValueError("Each selected LoRA card needs its saved weight and filename.")
        weight, name = caption.split(" - ", 1)
        weight = float(weight)
        if not math.isfinite(weight) or not -20 <= weight <= 20:
            raise ValueError("Copied scheduled LoRA strengths must be between -20 and 20.")
        if name and name != "None":
            result.append(dict(type="lora", name=name, model_strength=weight, clip_strength=0))
    if not 1 <= len(result) <= 8:
        raise ValueError("Select between 1 and 8 LoRA cards to copy into scheduled sources.")
    return json.dumps(result, separators=(",", ":"))
