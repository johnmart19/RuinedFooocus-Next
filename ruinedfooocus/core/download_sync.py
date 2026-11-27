"""Process-wide synchronization across model download surfaces.

Every downloader that publishes through ``preparation_service`` (Models →
Download, Models → Popular now, Create → First model, Models → Components,
Chat, on-demand image preparation, and legacy ``load_file_from_url``) notifies
this coordinator so catalogues, Chat/Vision lists, native source receipts, and
``shared.models`` stay in sync regardless of which surface started the transfer.
"""

from collections.abc import Mapping
from pathlib import Path
import threading

_lock = threading.Lock()
_sync_version = 0


def sync_version() -> int:
    """Monotonic counter incremented whenever any prepared artifact is published."""
    with _lock:
        return _sync_version


def bump_sync_version() -> int:
    """Increment and return the global download-sync version."""
    global _sync_version
    with _lock:
        _sync_version += 1
        return _sync_version


def _is_under_roots(path: Path, roots) -> bool:
    for root in roots:
        try:
            if path.is_relative_to(Path(root).resolve()):
                return True
        except (OSError, ValueError):
            continue
    return False


def notify_artifact_published(destination, request=None, verified_signature=None):
    """Synchronize model inventories and caches after a verified publication."""
    bump_sync_version()
    try:
        installed = Path(destination).resolve()
    except (OSError, ValueError):
        return

    try:
        import ruinedfooocus.app.state as shared
    except Exception:
        shared = None

    manager = getattr(shared, "path_manager", None) if shared is not None else None
    models = getattr(shared, "models", None) if shared is not None else None

    # 1. Record native source verification receipts when available so models
    #    downloaded from Popular now, Create, Components, or on-demand image
    #    preparation receive the same provenance as Models → Download.
    if request is not None and isinstance(verified_signature, Mapping) and manager is not None:
        try:
            cache_path = getattr(manager, "model_paths", {}).get("cache_path")
            if cache_path:
                from ruinedfooocus.features.models.model_identity import (
                    record_prepared_native_source,
                )

                record_prepared_native_source(installed, request, verified_signature, cache_path)
        except Exception:
            pass

    # 2. Invalidate the Chat / Vision GGUF catalogue whenever a GGUF file or
    #    an artifact inside llm/vision folders is published.
    chat_changed = installed.suffix.lower() == ".gguf"
    if not chat_changed and manager is not None:
        try:
            chat_roots = [
                root for kind in ("llm", "vision") for root in manager.get_model_folders(kind)
            ]
            chat_changed = _is_under_roots(installed, chat_roots)
        except Exception:
            chat_changed = False
    if chat_changed:
        try:
            from ruinedfooocus.features.chat.llama_models import invalidate_model_catalogue

            invalidate_model_catalogue()
        except Exception:
            pass

    # 3. Refresh shared.models when the published file landed in a model folder
    #    (skipping unrelated cache files such as Chat URL embeds).
    if models is not None and hasattr(models, "update_all_models"):
        try:
            tracked_roots = []
            for folders in getattr(models, "model_dirs", {}).values():
                tracked_roots.extend(folders)
            if manager is not None and hasattr(manager, "get_model_folders"):
                for kind in (
                    "checkpoints",
                    "diffusion_models",
                    "loras",
                    "embeddings",
                    "clip",
                    "clip_vision",
                    "vae",
                    "controlnet",
                    "upscalers",
                    "latent_upscalers",
                    "geometry_estimation",
                    "detection",
                    "background_removal",
                    "faceswap",
                    "model_patches",
                ):
                    try:
                        tracked_roots.extend(manager.get_model_folders(kind))
                    except Exception:
                        continue
            if tracked_roots and _is_under_roots(installed, tracked_roots):
                models.update_all_models()
        except Exception:
            pass


def _on_preparation_complete(request, destination, verified_signature):
    notify_artifact_published(destination, request=request, verified_signature=verified_signature)


def install_default_sync(service):
    """Attach the default cross-surface sync listener to a PreparationService."""
    service.add_completion_listener(_on_preparation_complete)
