"""Ordered model roots shared by discovery, component inventory and native loaders."""

import os
from pathlib import Path
import stat

from ruinedfooocus.core.paths import APP_ROOT, ensure_app_directory

MODEL_DIRECTORIES = (
    "checkpoints",
    "diffusion_models",
    "diffusers",
    "diffusers_cache",
    "loras",
    "embeddings",
    "controlnet",
    "model_patches",
    "gligen",
    "hypernetworks",
    "geometry_estimation",
    "detection",
    "background_removal",
    "vae_approx",
    "vae",
    "faceswap",
    "upscale_models",
    "latent_upscale_models",
    "text_encoders",
    "clip_vision",
    "style_models",
    "photomaker",
    "llm",
    "vision",
    "inbox",
    "rag_embeddings",
)


def ensure_model_root(root, *, required=False):
    """Recover known owned categories only; never create a custom mount."""
    root = Path(root)
    configured = Path(os.environ.get("RF_DATA_DIR") or APP_ROOT).expanduser()
    base = Path(os.path.abspath(configured if configured.is_absolute() else APP_ROOT / configured))
    for name in MODEL_DIRECTORIES:
        owned = base / "models" / name
        # Keep the lexical path too: a link introduced after request capture
        # must not turn an owned category into an apparently custom destination.
        if root == owned or root == owned.resolve():
            return ensure_app_directory("models", name)
    if required and not stat.S_ISDIR(root.stat().st_mode):
        raise NotADirectoryError(f"Model folder is unavailable: {root}")
    return root


# Keep existing profile keys and catalog download identities stable.
FOLDER_SETTINGS = {
    "text_encoders": "clip",
    "unet": "diffusion_models",
    "embedding": "embeddings",
    "t2i_adapter": "controlnet",
    "upscale_models": "upscalers",
    "latent_upscale_models": "latent_upscalers",
}
LOCAL_ALIASES = {
    "clip": ("text_encoders", "clip"),
    "diffusion_models": ("diffusion_models", "unet"),
    "embeddings": ("embeddings", "embedding"),
    "controlnet": ("controlnet", "t2i_adapter"),
}


def canonical_model_root(folder, root, resolve):
    """Use canonical local destinations without rewriting custom profile roots."""
    folder = FOLDER_SETTINGS.get(folder, folder)
    aliases = [resolve(f"../models/{name}/") for name in LOCAL_ALIASES.get(folder, ())]
    return aliases[0] if root in aliases else root


def model_folder_lines(value):
    """Display legacy single roots and ordered root lists without changing meaning."""
    return "\n".join(value if isinstance(value, list) else [value])


def model_folders(paths, folder, resolve):
    """Preserve configured priority; pair standard local aliases without any I/O.

    Custom roots remain authoritative. In particular, a custom encoder folder
    does not silently fall back to a model in the application's local folders.
    """
    folder = FOLDER_SETTINGS.get(folder, folder)
    configured = paths.get("path_" + folder, [])
    configured = configured if isinstance(configured, list) else [configured]
    roots = list(dict.fromkeys(resolve(value) for value in configured if value))
    aliases = [resolve(f"../models/{name}/") for name in LOCAL_ALIASES.get(folder, ())]
    if any(root in aliases for root in roots):
        roots.extend(alias for alias in aliases if alias not in roots)
    return roots
