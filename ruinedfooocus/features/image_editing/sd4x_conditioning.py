"""Native SD4x conditioning helpers; no loading, acquisition or routing."""


def is_sd4x(model):
    return type(getattr(model, "model_config", None)).__name__ == "SD_X4Upscaler"


def require_sd4x_source(model, control_type, source, *, inpainting=False):
    if not is_sd4x(model):
        return
    require_sd4x_input(control_type, source, inpainting=inpainting)


def require_sd4x_input(control_type, source, *, inpainting=False):
    if inpainting:
        raise ValueError("Masked inpainting is not available for this SD4x checkpoint.")
    if control_type != "edit" or source is None:
        raise ValueError("This SD4x checkpoint requires Diffusion 4× upscale with a source image.")


def require_sd4x_codec(vae):
    if vae.spacial_compression_decode() != 4:
        raise ValueError("This SD4x checkpoint requires its four-times VAE codec.")


def sd4x_size(image):
    from PIL import ImageOps

    image = ImageOps.exif_transpose(image)
    return image.width * 4, image.height * 4


def sd4x_pixels(image, request):
    import numpy as np
    import torch
    from PIL import ImageOps

    image = ImageOps.exif_transpose(image)
    if image.has_transparency_data:
        from PIL import Image

        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (128, 128, 128, 255)), rgba)
    image = image.convert("RGB")
    target = image.width * 4, image.height * 4
    specified = request.get("_sd4x_requested_size")
    if request.get("_metadata_prompt"):
        specified = request.get("width"), request.get("height")
    if specified is not None:
        if any(
            value is not None and value != expected for value, expected in zip(specified, target)
        ):
            raise ValueError(
                f"SD4x output must be {target[0]}×{target[1]} for this source. "
                "Choose that target size or resize the source explicitly."
            )
    return torch.from_numpy(np.asarray(image).copy()).float().unsqueeze(0) / 255.0


def restore_sd4x_alpha(pixels, source):
    """Keep generated RGB and reattach the source alpha resized uniformly by four."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(source)
    width, height = image.width * 4, image.height * 4
    if tuple(pixels.shape[1:]) != (height, width, 3):
        raise ValueError("SD4x decoded RGB does not match its four-times source dimensions.")
    if not image.has_transparency_data:
        return pixels
    alpha = image.convert("RGBA").getchannel("A").resize((width, height), Image.Resampling.BILINEAR)
    alpha = (
        torch.from_numpy(np.asarray(alpha).copy()).to(device=pixels.device, dtype=pixels.dtype)
        / 255.0
    )
    alpha = alpha[None, :, :, None].expand(pixels.shape[0], -1, -1, -1)
    return torch.cat((pixels, alpha), dim=-1)


def sd4x_conditioning(positive, negative, pixels, noise_augmentation=0.0):
    """Use native four-times geometry, normalized source and zero latent creation."""
    from comfy_extras.nodes_sdupscale import SD_4XUpscale_Conditioning

    if not 0.0 <= noise_augmentation <= 1.0:
        raise ValueError("SD4x noise augmentation must be between 0 and 1.")
    result = SD_4XUpscale_Conditioning.execute(
        images=pixels,
        positive=positive,
        negative=negative,
        scale_ratio=4.0,
        noise_augmentation=noise_augmentation,
    )
    return result[0], result[1], result[2]
