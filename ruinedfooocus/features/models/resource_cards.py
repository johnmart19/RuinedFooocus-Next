"""Offline presentation of scanned adapters; the caller supplies their verified kind.

No weights, hashes, remote URLs or model service are opened while building cards.
Metadata sidecars and the existing preview cache remain the source of publisher art.
"""

from collections import Counter
from functools import lru_cache
import hashlib
import json
import re
import threading
import io
from pathlib import Path
import unicodedata

from PIL import Image

from ruinedfooocus.core.paths import asset_path
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.model_previews import (
    SUFFIXES,
    is_warning,
    local_preview,
    valid_preview,
    write_preview,
)
from ruinedfooocus.features.models.selection import model_label, resource_caption


def _cache_dirs(cache_dir):
    if cache_dir is None:
        return []
    if isinstance(cache_dir, (str, Path)):
        cache_dir = [cache_dir]
    return [Path(directory) for directory in cache_dir]


def _read_metadata(path, signature):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    previous = data.get("_rf_file_signature")
    if previous is not None and previous != signature:
        return {}
    return data


def resource_metadata(path, cache_dir=None):
    """Read cached Civitai metadata, then let an explicit local sidecar override it."""
    path = Path(path)
    try:
        signature = file_signature(path)
    except OSError:
        return {}
    candidates = []
    for directory in reversed(_cache_dirs(cache_dir)):
        candidates.append((directory / path.name).with_suffix(".json"))
    candidates.extend((path.with_suffix(".json"), path.with_suffix(".civitai.info")))
    data = {}
    for candidate in dict.fromkeys(candidates):
        data.update(_read_metadata(candidate, signature))
    return data


@lru_cache(maxsize=256)
def _valid_preview(path, signature):
    return valid_preview(path) and not is_warning(path)


DEFAULT_PREVIEW_ROLES = {
    "image-model": ("Image model", "Image models"),
    "chat-model": ("Chat model", "Chat models"),
    "vision-projector": ("Vision / projectors", "Vision projector"),
    "text-encoder": ("Text encoder", "Text encoders"),
    # Detection (the face detector) reuses the lens-and-sensor artwork until it has its own.
    "vision-encoder": ("Vision encoder", "Vision encoders", "Detection"),
    "vae": ("VAE", "VAEs"),
    "lora": ("LoRA", "LoRAs"),
    "embedding": ("Embedding", "Embeddings"),
    "controlnet": ("ControlNet",),
    "model-patch": ("Model patch", "Model patches"),
    "upscaler": ("Upscaler", "Upscalers"),
    "latent-upscaler": ("Latent upscaler", "Latent upscalers"),
    "depth": ("Depth",),
    "background-removal": ("Background removal",),
    "face-restoration": ("Face restoration",),
    "model-component": ("Model component", "Component"),
}
_DEFAULT_ROLES = {
    label.casefold(): slug for slug, labels in DEFAULT_PREVIEW_ROLES.items() for label in labels
}


@lru_cache(maxsize=16)
def _bundled_default(path, signature):
    data = Path(path).read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def _default_artwork(kind):
    slug = _DEFAULT_ROLES.get(str(kind).casefold(), "model-component")
    path = asset_path(f"model-previews/{slug}.jpeg")
    data, revision = _bundled_default(str(path), tuple(file_signature(path)))
    return slug, data, revision


@lru_cache(maxsize=16)
def _placeholder_image(data):
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def _placeholder(kind):
    """Conceptual category artwork; the UI supplies its localized default label."""
    return _placeholder_image(_default_artwork(kind)[1])


_default_lock = threading.Lock()


def default_preview(kind, cache_dir):
    """Versioned local artwork, isolated from publisher caches and provenance."""
    slug, data, revision = _default_artwork(kind)
    destination = Path(cache_dir) / "default-previews" / f"{slug}-{revision[:16]}.jpeg"
    with _default_lock:
        if not destination.is_file() or not valid_preview(destination):
            write_preview(data, destination)
    return str(destination)


def resource_preview(path, kind, cache_dir=None):
    """Prefer local author artwork, then reuse the shared cached-thumbnail lookup."""
    if kind not in ("LoRA", "Embedding"):
        raise ValueError("Choose a verified LoRA or Embedding resource kind.")
    path = Path(path)
    local = local_preview(path)
    if local:
        return local
    candidates = []
    for directory in _cache_dirs(cache_dir):
        candidates.extend(
            (directory / path.name).with_suffix(suffix)
            for suffix in (".jpeg", ".jpg", ".png", ".webp", ".gif")
        )
    for candidate in candidates:
        try:
            signature = tuple(file_signature(candidate))
            if _valid_preview(str(candidate), signature):
                return str(candidate)
        except OSError:
            continue
    return _placeholder(kind).copy()


def resource_card(
    name, path, kind, cache_dir=None, preview_allowed=True, *, fallback_cache_dir=None
):
    """Build one card from a known scanner/header result, never infer type from its name."""
    if kind not in ("LoRA", "Embedding"):
        raise ValueError("Choose a verified LoRA or Embedding resource kind.")
    metadata = resource_metadata(path, cache_dir)
    display_metadata = dict(metadata)
    publisher = metadata.get("model")
    if isinstance(publisher, dict) and isinstance(publisher.get("name"), str):
        # Publisher banners often append marketing text after decorative emoji.
        title = publisher["name"].strip()
        title = title.lstrip(
            "".join(c for c in title if unicodedata.category(c) in ("So", "Sk")) + " "
        )
        end = next(
            (index for index, c in enumerate(title) if unicodedata.category(c) in ("So", "Sk")),
            len(title),
        )
        display_metadata["model"] = publisher | {"name": title[:end].strip() or publisher["name"]}
    label = model_label(name, display_metadata)
    if (
        kind == "Embedding"
        and isinstance(metadata.get("name"), str)
        and metadata["name"].strip()
        and not re.fullmatch(r"v?\d+(?:\.\d+)*(?:[a-z])?", metadata["name"].strip(), re.I)
    ):
        label = metadata["name"].strip()
    fallback = default_preview(kind, fallback_cache_dir) if fallback_cache_dir is not None else None
    preview = (
        resource_preview(path, kind, cache_dir) if preview_allowed else _placeholder(kind).copy()
    )
    preview_is_default = isinstance(preview, Image.Image)
    if preview_is_default and fallback:
        preview = fallback
    return {
        "name": name,
        "path": str(path),
        "kind": kind,
        "label": label,
        "caption": resource_caption(label, (kind, name, str(path))),
        "metadata": metadata,
        "preview": preview,
        "fallback": fallback,
        "preview_is_default": preview_is_default,
    }


def resource_records(items, kind, cache_dir=None, query="", *, fallback_cache_dir=None):
    """Cards for (native name, physical path) pairs; search retains filename support."""
    records = [
        resource_card(name, path, kind, cache_dir, fallback_cache_dir=fallback_cache_dir)
        for name, path in items
    ]
    counts = Counter(record["label"] for record in records)
    seen = Counter()
    for record in records:
        label = record["label"]
        seen[label] += 1
        if counts[label] > 1:
            record["label"] = f"{label} · {seen[label]}"
        record["caption"] = resource_caption(
            record["label"], (kind, record["name"], record["path"])
        )
    search = (query or "").casefold().strip()
    return [
        record
        for record in records
        if search in (record["label"] + " " + record["name"]).casefold()
    ]
