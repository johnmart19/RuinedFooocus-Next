"""Explicit owner-UI media roots when mutable data is outside the application."""

from pathlib import Path

from ruinedfooocus.core.cache_contract import configured_cache_root
from ruinedfooocus.core.paths import asset_path

PUBLIC_CACHE_MEDIA = (
    "card-previews",
    "catalogue-previews",
    "default-previews",
    "native-image-previews",
    "native-image-exports",
)


def owner_media_paths(manager, models, settings):
    """Retain existing media access and add only dedicated application media leaves."""
    cache = configured_cache_root(manager)
    roots = [asset_path("."), manager.model_paths["temp_outputs_path"]]
    roots.extend(models.cache_paths.values())
    roots.extend(settings.get("archive_folders", []))
    for name in PUBLIC_CACHE_MEDIA:
        candidate = (cache / name).resolve()
        # A redirected leaf must not turn this allowlist into access to private data.
        if candidate.is_relative_to(cache):
            roots.append(candidate)
    return list(dict.fromkeys(str(Path(root).expanduser().resolve()) for root in roots))
