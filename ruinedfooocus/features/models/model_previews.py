from ruinedfooocus.core.paths import asset_path

"""Decode model artwork before publishing a bounded static thumbnail."""

import io
import json
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

SUFFIXES = (".gif", ".jpeg", ".jpg", ".png", ".webp")
VERSION = 3


def valid_preview(path):
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                return False
            image.verify()
        return True
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


def local_preview(path, *, ambiguous=False):
    """Exact author sidecars precede compatible, nonambiguous legacy names."""
    path = Path(path)
    candidates = [path.with_name(path.name + ".preview" + suffix) for suffix in SUFFIXES]
    if not ambiguous:
        for suffix in SUFFIXES:
            candidates.extend((path.with_suffix(".preview" + suffix), path.with_suffix(suffix)))
    return next(
        (
            str(candidate)
            for candidate in candidates
            if candidate.is_file() and valid_preview(candidate) and not is_warning(candidate)
        ),
        None,
    )


def is_warning(path):
    path = Path(path)
    for warning in (asset_path("warning.jpeg"), asset_path("warning.png")):
        if (
            warning.is_file()
            and path.is_file()
            and path.stat().st_size == warning.stat().st_size
            and path.read_bytes() == warning.read_bytes()
        ):
            return True
    return False


def write_preview(data, cache_path, caption=""):
    cache_path = Path(cache_path)
    with Image.open(io.BytesIO(data)) as source:
        if getattr(source, "n_frames", 1) != 1:
            # An animation fills an empty card with its first frame but never
            # replaces artwork that is already published.
            if valid_preview(cache_path.with_suffix(".jpeg")):
                raise ValueError("Publisher artwork must be a static image")
            source.seek(0)
        if source.width * source.height > 16 * 1024 * 1024:
            raise ValueError("Publisher artwork dimensions exceed size limit")
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        if caption:
            ImageDraw.Draw(image).text(
                (3, 3), caption, fill="white", stroke_width=1, stroke_fill="black"
            )
    destination = cache_path.with_suffix(".jpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, suffix=destination.suffix)
    os.close(descriptor)
    try:
        image.save(temporary, format="JPEG", quality=95, subsampling=0)
        os.replace(temporary, destination)
        # Publish first, then remove obsolete formats so the gallery cannot select
        # an older animation instead of the static thumbnail.
        for suffix in SUFFIXES:
            old = cache_path.with_suffix(suffix)
            if old != destination:
                old.unlink(missing_ok=True)
        cache_path.with_suffix(".preview.json").write_text(
            json.dumps({"version": VERSION}), encoding="utf-8"
        )
        return destination
    finally:
        Path(temporary).unlink(missing_ok=True)
