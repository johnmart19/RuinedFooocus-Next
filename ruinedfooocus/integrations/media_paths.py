"""Explicit owner-UI media roots when mutable data is outside the application."""

from pathlib import Path
from urllib.parse import quote

from ruinedfooocus.core.cache_contract import configured_cache_root
from ruinedfooocus.core.paths import asset_path

PUBLIC_CACHE_MEDIA = (
    "card-previews",
    "catalogue-previews",
    "default-previews",
    "native-image-previews",
    "native-image-exports",
)
# Model artwork caches the owner interface serves through `Models.cache_paths`.
MODEL_CACHE_MEDIA = ("checkpoints", "loras", "embeddings")


def card_picture_roots(manager=None):
    """Cache folders whose pictures a card may name by address.

    The same leaves `owner_media_paths` serves, so an address never names a
    file the interface would refuse.
    """
    cache = configured_cache_root(manager)
    roots = []
    for name in (*PUBLIC_CACHE_MEDIA, *MODEL_CACHE_MEDIA):
        candidate = (cache / name).resolve()
        if candidate.is_relative_to(cache):
            roots.append(candidate)
    return tuple(roots)


def card_picture(path, roots):
    """A picture for an HTML card value: its address when served, else a file.

    Returning a file makes Gradio check and copy it on every update; for a page
    of cards that costs seconds. Pictures in a served cache folder are named by
    address instead; any other file (such as a preview beside a model) keeps the
    file route and its checks.
    """
    resolved = Path(path).resolve()
    if any(resolved.is_relative_to(root) for root in roots):
        return dict(
            path=str(resolved),
            url="/gradio_api/file=" + quote(resolved.as_posix(), safe="/:"),
        )
    import gradio as gr

    return gr.data_classes.FileData(path=str(path)).model_dump()


def owner_media_paths(manager, models, settings):
    """Retain existing media access and add only dedicated application media leaves."""
    from gradio.utils import get_upload_folder

    cache = configured_cache_root(manager)
    # Gradio already serves its own upload cache. Listing it first does not widen
    # access, but Gradio checks every returned file against these roots in order
    # (resolving each root again per file); Gallery thumbnails live in that cache,
    # so this turns about 16 path resolutions per thumbnail into one.
    roots = [get_upload_folder(), asset_path("."), manager.model_paths["temp_outputs_path"]]
    roots.extend(models.cache_paths.values())
    roots.extend(settings.get("archive_folders", []))
    for name in PUBLIC_CACHE_MEDIA:
        candidate = (cache / name).resolve()
        # A redirected leaf must not turn this allowlist into access to private data.
        if candidate.is_relative_to(cache):
            roots.append(candidate)
    return list(dict.fromkeys(str(Path(root).expanduser().resolve()) for root in roots))
