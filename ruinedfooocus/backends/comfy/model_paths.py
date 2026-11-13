"""Connect configured image model roots to the pinned native folder registry."""

from ruinedfooocus.core.model_folders import model_folders


NATIVE_IMAGE_FOLDERS = {
    "checkpoints": "checkpoints",
    "diffusion_models": "diffusion_models",
    "text_encoders": "clip",
    "clip_vision": "clip_vision",
    "style_models": "style_models",
    "photomaker": "photomaker",
    "loras": "loras",
    "embeddings": "embeddings",
    "vae": "vae",
    "vae_approx": "vae_approx",
    "controlnet": "controlnet",
    "model_patches": "model_patches",
    "gligen": "gligen",
    "hypernetworks": "hypernetworks",
    "upscale_models": "upscalers",
    "geometry_estimation": "geometry_estimation",
    "detection": "detection",
    "background_removal": "background_removal",
}


def register_image_model_paths(path_manager):
    import folder_paths

    # Prefer profile roots, retaining native defaults and their extension rules.
    # Registration never scans, moves, creates or downloads model files.
    for native, folder in NATIVE_IMAGE_FOLDERS.items():
        for root in reversed(model_folders(path_manager.paths, folder, path_manager.get_abspath)):
            folder_paths.add_model_folder_path(native, str(root), is_default=True)
