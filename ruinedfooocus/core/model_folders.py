"""Ordered model roots shared by discovery, component inventory and native loaders."""

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
