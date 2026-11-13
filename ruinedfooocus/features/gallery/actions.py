"""Validate Library selections and saved resources without loading model weights."""

import math
from pathlib import Path
import re

from ruinedfooocus.features.gallery.image_metadata import read_metadata
from ruinedfooocus.features.models.model_identity import checkpoint_hashes, file_signature


def _require_still_source(path):
    from ruinedfooocus.features.models.image_scope import require_image_path

    require_image_path(path)
    # Native EXR/AVIF imports already validate a single decoded still image,
    # including the AVIF auxiliary alpha stream. Ordinary raster formats use
    # Pillow's frame count so animations cannot silently become frame zero.
    if path.suffix.lower() not in (".exr", ".avif"):
        from PIL import Image

        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Choose a single still image.")


def source_record(path):
    path = Path(path).resolve()
    _require_still_source(path)
    return {"path": str(path), "caption": f"{path.name}\n{path}", "signature": file_signature(path)}


def source_path(record):
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("Select an image in Library first.")
    path = Path(record["path"])
    from ruinedfooocus.features.models.image_scope import require_image_path

    require_image_path(path)
    try:
        unchanged = file_signature(path) == record.get("signature")
    except OSError:
        unchanged = False
    if not unchanged:
        raise ValueError(
            "The selected file changed or was removed. Refresh Library and select it again."
        )
    _require_still_source(path)
    return path


def select_source(records, index, caption):
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(records):
        raise ValueError("The image list changed. Select the image again.")
    record = records[index]
    if record["caption"] != caption:
        raise ValueError("The image list changed. Select the image again.")
    source_path(record)
    return record.copy()


def resolve_resource(models, kind, name, recorded_hash=None):
    names = models.get_names(kind)
    if not isinstance(name, str) or not name:
        raise ValueError(f"The saved setup has no {kind} filename.")
    expected = (
        recorded_hash.lower()
        if isinstance(recorded_hash, str) and re.fullmatch(r"[a-fA-F0-9]{64}", recorded_hash)
        else None
    )
    normalized = name.replace("\\", "/").casefold()
    candidates = [n for n in names if n.replace("\\", "/").casefold() == normalized]
    if not candidates:
        candidates = [
            n
            for n in names
            if Path(n.replace("\\", "/")).name.casefold()
            == Path(name.replace("\\", "/")).name.casefold()
        ]

    def matches(candidate):
        if not models.get_file(kind, candidate):
            return False
        if not expected:
            return True
        info = models.get_models_by_path(kind, candidate, fetch=False)
        return checkpoint_hashes(info).get("SHA256", "").lower() == expected

    found = [candidate for candidate in candidates if matches(candidate)]
    if not found and expected:
        found = [
            candidate for candidate in names if candidate not in candidates and matches(candidate)
        ]
    if len(found) != 1:
        raise ValueError(
            f"Cannot uniquely match the saved {kind} resource: {name}. Install/refresh the recorded file or choose a model manually."
        )
    return found[0]


def resolve_setup(value, models):
    metadata = read_metadata(value)
    if metadata is None or not isinstance(metadata["prompt"], str):
        raise ValueError("Select an image with valid generation metadata first.")
    metadata.setdefault("negative", "")
    if not isinstance(metadata["negative"], str):
        raise ValueError("The saved negative prompt must be text.")
    checkpoint = resolve_resource(
        models, "checkpoints", metadata.get("base_model_name"), metadata.get("base_model_hash")
    )
    loras = []
    entries = metadata.get("loras") or []
    if not isinstance(entries, list):
        raise ValueError("Invalid saved LoRA list.")
    for entry in entries:
        if isinstance(entry, dict):
            name, weight, digest = entry.get("name"), entry.get("weight"), entry.get("hash")
        elif (
            isinstance(entry, (tuple, list))
            and len(entry) == 2
            and isinstance(entry[1], str)
            and " - " in entry[1]
        ):
            weight, name = entry[1].split(" - ", 1)
            digest = entry[0]
        else:
            raise ValueError("Invalid saved LoRA entry.")
        if name in (None, "None", ""):
            continue
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            raise ValueError("Saved LoRA weights must be finite numbers.") from None
        if not math.isfinite(weight):
            raise ValueError("Saved LoRA weights must be finite numbers.")
        loras.append((resolve_resource(models, "loras", name, digest), weight))
    return metadata, checkpoint, loras
