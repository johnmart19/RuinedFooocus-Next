"""Nine-channel SD1/SD2 inpaint contract around the pinned native conditioner."""


def local_native_sd_inpaint(path):
    """Passive current-header capability; unreadable or unrecognized files remain false."""
    if path is None:
        return False
    from ruinedfooocus.features.models.model_architecture import inspect_model_file

    try:
        contract = inspect_model_file(path)
        return (
            contract.get("native_class") in ("SD15", "SD20")
            and contract.get("native_config", {}).get("in_channels") == 9
        )
    except (OSError, ValueError, TypeError, AttributeError, ImportError):
        return False


def is_native_sd_inpaint(model):
    config = getattr(model, "model_config", None)
    return (
        type(config).__name__ in ("SD15", "SD20")
        and getattr(config, "unet_config", {}).get("in_channels") == 9
    )


def require_native_sd_inpaint_source(model, inpainting):
    if is_native_sd_inpaint(model) and not inpainting:
        raise ValueError(
            "This nine-channel Stable Diffusion checkpoint requires Inpainting with a source image and painted mask."
        )


def native_sd_inpaint_conditioning(model, positive, negative, vae, pixels, mask, grow_mask_by=20):
    """Use source/mask concat conditioning, including native original latent and noise mask."""
    if not is_native_sd_inpaint(model):
        raise ValueError("Choose a native nine-channel SD1 or SD2 inpaint checkpoint.")
    if getattr(vae, "latent_channels", None) != 4 or vae.spacial_compression_encode() != 8:
        raise ValueError("Native SD1/SD2 inpainting requires its four-channel eightfold VAE.")
    from ruinedfooocus.features.image_editing.native_conditioning import flux_inpaint_conditioning

    # This existing adapter owns the generic InpaintModelConditioning node and GrowMask.
    return flux_inpaint_conditioning(positive, negative, vae, pixels, mask, grow_mask_by)
