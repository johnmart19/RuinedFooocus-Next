"""Model display names are presentation; filenames remain the selection identity."""

from collections import Counter
from pathlib import Path
import hashlib
import re

from ruinedfooocus.features.models.image_scope import retired_family
from ruinedfooocus.features.models.utility_models import is_image_utility
from ruinedfooocus.features.models.stage_companions import is_stage_companion


def model_label(name, metadata=None):
    metadata = metadata or {}
    model = metadata.get("model")
    title = model.get("name") if isinstance(model, dict) else None
    header = metadata.get("__metadata__", {})
    header = header if isinstance(header, dict) else {}
    title = (
        metadata.get("rf_display_name")
        or title
        or metadata.get("modelspec.title")
        or header.get("modelspec.title")
        or metadata.get("hf_repo_id")
    )
    if not isinstance(title, str) or not title.strip():
        stem = re.sub(r"(?:[ _.-]+[a-f0-9]{8,64})$", "", Path(name).stem, flags=re.I)
        return " ".join(stem.replace("_", " ").split()) or Path(name).stem
    title = " ".join(title.strip(" ✨★☆⭐").split())
    # Publisher titles often list every merge ancestor after the primary name.
    short = re.sub(r"\s+trained$", "", title.split(" + ", 1)[0], flags=re.IGNORECASE)
    short = short if len(short) <= 80 else short[:77] + "…"
    version = metadata.get("name")
    if isinstance(version, str) and version.strip() and version.casefold() not in short.casefold():
        version = " ".join(version.replace("_", " ").split())
        separator = " " if version.lower().startswith("v") else " · "
        short += separator + version
    return short


def checkpoint_records(names, metadata=None, query=""):
    records = []
    for name in names:
        info = metadata(name) if metadata else {}
        if (
            retired_family(info.get("baseModel"))
            or is_image_utility(info)
            or is_stage_companion(info)
        ):
            continue
        records.append({"name": name, "label": model_label(name, info)})
    counts = Counter(record["label"] for record in records)
    seen = Counter()
    for record in records:
        label = record["label"]
        seen[label] += 1
        if counts[record["label"]] > 1:
            record["label"] += f" · {seen[label]}"
        record["caption"] = resource_caption(record["label"], record["name"])
    search = (query or "").casefold().strip()
    return [r for r in records if search in (r["label"] + " " + r["name"]).casefold()]


def resource_caption(label, identity):
    # Hash only the selection identity string, never weights. The short visible
    # ID rejects queued clicks after a same-title file replaces a gallery item.
    short_id = hashlib.sha256(str(identity).encode("utf-8")).hexdigest()[:10]
    return f"{label} · {short_id}"


def checkpoint_choices(names, selected=None, metadata=None):
    names = list(names)
    choices = [(r["label"], r["name"]) for r in checkpoint_records(names, metadata)]
    if selected and selected not in [value for _, value in choices]:
        if metadata and (
            retired_family(metadata(selected).get("baseModel"))
            or is_image_utility(metadata(selected))
            or is_stage_companion(metadata(selected))
        ):
            return choices
        status = "outside image filter" if selected in names else "not installed"
        choices.append((f"{model_label(selected)} ({status})", selected))
    return choices


def primary_checkpoint_selection(selected, metadata=None, *, names=()):
    info = metadata(selected) if selected and metadata else {}
    if retired_family(info.get("baseModel")) or is_image_utility(info) or is_stage_companion(info):
        return next((record["name"] for record in checkpoint_records(names, metadata)), None)
    return selected


def selected_checkpoint(records, index, caption):
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(records):
        raise ValueError("The model list changed. Select the checkpoint again.")
    selected = records[index]
    if selected["caption"] != caption:
        raise ValueError("The model list changed. Select the checkpoint again.")
    return selected["name"]
