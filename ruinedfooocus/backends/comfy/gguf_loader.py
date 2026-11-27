"""Load local diffusion GGUFs through the backend's quantization-aware loader."""

from pathlib import Path

import folder_paths
from molbal_comfyui_gguf.nodes import UnetLoaderGGUFAdvanced

from ruinedfooocus.core.settings_schema import GGUF_DTYPE_CHOICES

DEFAULT_DEQUANT_DTYPE = "default"
DEFAULT_PATCH_DTYPE = "default"
DEFAULT_PATCH_ON_DEVICE = True


def _saved_settings():
    try:
        from ruinedfooocus.app import state as shared
    except Exception:
        return {}
    settings = getattr(shared, "settings", None)
    return getattr(settings, "default_settings", {}) or {}


def loader_options(saved=None, *, dequant_dtype=None, patch_dtype=None, patch_on_device=None):
    """Resolve GGUF diffusion loader tuning options with backward-compatible defaults."""
    source = _saved_settings() if saved is None else saved
    resolved_dequant = (
        source.get("gguf_dequant_dtype", DEFAULT_DEQUANT_DTYPE)
        if dequant_dtype is None
        else dequant_dtype
    )
    resolved_patch = (
        source.get("gguf_patch_dtype", DEFAULT_PATCH_DTYPE)
        if patch_dtype is None
        else patch_dtype
    )
    resolved_on_device = (
        source.get("gguf_patch_on_device", DEFAULT_PATCH_ON_DEVICE)
        if patch_on_device is None
        else patch_on_device
    )

    if resolved_dequant is None:
        resolved_dequant = DEFAULT_DEQUANT_DTYPE
    if resolved_patch is None:
        resolved_patch = DEFAULT_PATCH_DTYPE
    if resolved_dequant not in GGUF_DTYPE_CHOICES:
        raise ValueError(
            f"Unsupported GGUF dequant_dtype {resolved_dequant!r}; "
            f"choose one of {', '.join(GGUF_DTYPE_CHOICES)}."
        )
    if resolved_patch not in GGUF_DTYPE_CHOICES:
        raise ValueError(
            f"Unsupported GGUF patch_dtype {resolved_patch!r}; "
            f"choose one of {', '.join(GGUF_DTYPE_CHOICES)}."
        )
    if not isinstance(resolved_on_device, bool):
        raise ValueError("gguf_patch_on_device must be a boolean.")
    return {
        "dequant_dtype": resolved_dequant,
        "patch_dtype": resolved_patch,
        "patch_on_device": resolved_on_device,
    }


def load_diffusion_model(
    filename,
    *,
    dequant_dtype=None,
    patch_dtype=None,
    patch_on_device=None,
):
    path = Path(filename).resolve(strict=True)
    from ruinedfooocus.features.models.image_scope import require_image_checkpoint

    require_image_checkpoint(path)
    folder_paths.add_model_folder_path("diffusion_models", str(path.parent), is_default=True)
    options = loader_options(
        dequant_dtype=dequant_dtype,
        patch_dtype=patch_dtype,
        patch_on_device=patch_on_device,
    )
    return UnetLoaderGGUFAdvanced().load_unet(
        path.name,
        dequant_dtype=options["dequant_dtype"],
        patch_dtype=options["patch_dtype"],
        patch_on_device=options["patch_on_device"],
    )[0]
