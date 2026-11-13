"""Cached scalar GGUF role metadata for the local Chat inventory."""

from functools import lru_cache
from pathlib import Path

from ruinedfooocus.features.models.model_identity import file_signature


@lru_cache(maxsize=128)
def _role_metadata(path, signature):
    from ruinedfooocus.features.models.gguf_metadata import read_scalar_strings

    return read_scalar_strings(path, ("general.architecture", "general.type"))


def role_metadata(path):
    signature = tuple(file_signature(path))
    result = _role_metadata(signature[0], signature)
    if signature != tuple(file_signature(path)):
        raise ValueError("Model changed during role inspection; refresh when stable.")
    return dict(result)


def is_projector(path):
    if "mmproj" in Path(path).stem.casefold():
        return True
    try:
        metadata = role_metadata(path)
    except (
        OSError,
        ValueError,
        TypeError,
        ImportError,
        AttributeError,
        KeyError,
        IndexError,
        OverflowError,
    ):
        return False
    return (
        metadata.get("general.architecture") in ("clip", "mmproj")
        or metadata.get("general.type") == "projector"
    )
