"""PiD input codecs and native restoration conditioning, separate from its pixel output."""

import math
from pathlib import Path


def pid_vae_roles(latent_format):
    roles = {
        "flux": ("vae_flux", "vae_flux2"),
        "sd3": ("vae_sd3",),
        "sdxl": ("vae_sdxl",),
        "qwenimage": ("vae_qwen_image",),
    }
    if latent_format not in roles:
        raise ValueError("Choose Flux, SD3, SDXL or Qwen Image as the PiD input latent format.")
    return roles[latent_format]


def degrade_sigma(data):
    value = data.get("pid_degrade_sigma")
    if value in (None, ""):
        return 0.0
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("PiD degradation must be from 0 through 1.")
    return float(value)


def validate_input_codec(path, latent_format, checkpoint=None):
    from ruinedfooocus.features.models.component_candidates import inspect_local_component

    if Path(path).suffix.lower() not in (".safetensors", ".sft"):
        raise ValueError("PiD's input codec currently requires a local native safetensors VAE.")
    role = next(
        (
            role
            for role in pid_vae_roles(latent_format)
            if inspect_local_component(role, path).get("compatibility") == "compatible"
        ),
        None,
    )
    if role is None:
        raise ValueError("Select a verified compatible input VAE for PiD's chosen latent format.")
    if checkpoint is not None:
        from ruinedfooocus.features.models.model_architecture import inspect_model_file
        from ruinedfooocus.features.models.component_candidates import VAE_CHANNELS

        native = inspect_model_file(checkpoint)
        if (
            native.get("native_class") != "PiD"
            or native.get("native_config", {}).get("lq_latent_channels") != VAE_CHANNELS[role]
        ):
            raise ValueError(
                "The selected input VAE's channels do not match this native PiD variant."
            )


def pid_conditioning(model, positive, negative, pixels, data, manager):
    import comfy.sd
    import comfy.utils
    from nodes import VAEEncode
    from comfy_extras.nodes_pid import PiDConditioning
    from comfy_extras.nodes_chroma_radiance import EmptyChromaRadianceLatentImage
    from ruinedfooocus.features.generation.image_preparation import prepared_asset

    name = data.get("pid_input_vae")
    if not name:
        raise ValueError("Select PiD's input VAE before restoring an image.")
    layout = data.get("pid_latent_format")
    sigma = degrade_sigma(data)
    with prepared_asset(data, "pid_input_vae", name, manager, "vae") as path:
        validate_input_codec(path, layout)
        state, metadata = comfy.utils.load_torch_file(str(path), return_metadata=True)
        vae = comfy.sd.VAE(sd=state, metadata=metadata)
        vae.throw_exception_if_invalid()
        projection = model.diffusion_model.lq_proj
        if vae.latent_channels != projection.latent_channels:
            raise ValueError("This PiD checkpoint requires a different input latent channel count.")
        width, height = int(pixels.shape[2]), int(pixels.shape[1])
        if (data["width"], data["height"]) != (
            width * projection.sr_scale,
            height * projection.sr_scale,
        ):
            raise ValueError(
                "Select PiD's native output size: source dimensions multiplied by its native upscale factor."
            )
        latent = VAEEncode().encode(vae, pixels)[0]
        positive = PiDConditioning.execute(positive, latent, layout, sigma)[0]
        negative = PiDConditioning.execute(negative, latent, layout, sigma)[0]
    empty = EmptyChromaRadianceLatentImage.execute(
        width=data["width"], height=data["height"], batch_size=1
    )[0]
    data["pid_conditioning"] = dict(
        input_vae=Path(name).name,
        latent_format=layout,
        degrade_sigma=sigma,
        scale=projection.sr_scale,
    )
    return positive, negative, empty
