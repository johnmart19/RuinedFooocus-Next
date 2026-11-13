"""Operation-specific native image conditioning; no acquisition or model loading."""

INSTRUCT_FAMILIES = ("SD15_instructpix2pix", "SDXL_instructpix2pix")


def unclip_conditioning(conditioning, model, vision, pixels):
    """Use the checkpoint's native L/H vision tower for positive image guidance."""
    from nodes import CLIPVisionEncode, unCLIPConditioning

    config = getattr(model, "model_config", None)
    expected = {"SD21UnclipL": 768, "SD21UnclipH": 1024}.get(type(config).__name__)
    if expected is None:
        raise ValueError("Image embedding guidance requires native SD2.1 UnCLIP L or H.")
    if vision is None:
        raise ValueError("This UnCLIP workflow requires the checkpoint's bundled vision encoder.")
    encoded = CLIPVisionEncode().encode(vision, pixels, "center")[0]
    if encoded.image_embeds.shape[-1] != expected:
        raise ValueError("The vision embedding width does not match this UnCLIP L/H checkpoint.")
    return unCLIPConditioning().apply_adm(conditioning, encoded, 1.0, 0.0)[0]


def is_instruct_pix2pix(model):
    return type(getattr(model, "model_config", None)).__name__ in INSTRUCT_FAMILIES


def require_instruct_edit(model, control_type, source, *, inpainting=False):
    if not is_instruct_pix2pix(model):
        return
    if inpainting:
        raise ValueError(
            "Masked inpainting is not available for this instruct checkpoint. Use Native image edit."
        )
    if control_type != "edit":
        raise ValueError("This instruct checkpoint requires Native image edit with a source image.")
    if source is None:
        raise ValueError("Add a source image in Image tools for this instruct checkpoint.")


def instruct_conditioning(positive, negative, vae, pixels):
    """Attach the native source latent to both conditions, keeping native zero initialization."""
    from comfy_extras.nodes_ip2p import InstructPixToPixConditioning

    result = InstructPixToPixConditioning.execute(
        positive=positive, negative=negative, vae=vae, pixels=pixels
    )
    return result[0], result[1], result[2]


def is_flux_inpaint(model):
    """Identify the native 96-channel Flux variant, independently of display names."""
    config = getattr(getattr(model, "model_config", None), "unet_config", {})
    return config.get("image_model") == "flux" and config.get("in_channels") == 96


def require_inpaint_input(model, inpainting):
    if is_flux_inpaint(model) and not inpainting:
        raise ValueError(
            "This Flux inpaint checkpoint requires Inpainting with a Main image and painted mask."
        )


def flux_inpaint_conditioning(positive, negative, vae, pixels, mask, grow_mask_by=20):
    """Preserve native source/mask concat conditioning and the original sampling latent."""
    from nodes import InpaintModelConditioning
    from comfy_extras.nodes_mask import GrowMask

    if not 0 <= grow_mask_by <= 128:
        raise ValueError("Mask expansion must be between 0 and 128 pixels.")
    if grow_mask_by:
        mask = GrowMask.execute(mask, expand=grow_mask_by, tapered_corners=False)[0]
    return InpaintModelConditioning().encode(
        positive=positive,
        negative=negative,
        vae=vae,
        pixels=pixels,
        mask=mask,
        noise_mask=True,
    )


def refiner_text_conditioning(clip, text, width, height, *, negative=False):
    """Use native SDXL refiner geometry and positive/negative aesthetic defaults."""
    from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXLRefiner

    return CLIPTextEncodeSDXLRefiner.execute(
        clip=clip, text=text, width=width, height=height, ascore=2.5 if negative else 6.0
    )[0]
