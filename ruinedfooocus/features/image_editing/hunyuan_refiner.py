"""Single-image Hunyuan refiner source encoding through the native folding contract."""

import math


def noise_augmentation(data):
    value = data.get("refiner_noise_augmentation")
    if value in (None, ""):
        return 0.10
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Refiner noise augmentation must be from 0 through 1.")
    return float(value)


def refiner_conditioning(positive, negative, vae, pixels, data):
    from nodes import VAEEncode
    from comfy_extras.nodes_hunyuan import HunyuanRefinerLatent

    if vae.latent_channels != 32 or vae.latent_dim != 3 or vae.spacial_compression_decode() != 16:
        raise ValueError(
            "Hunyuan Image refiner requires its dedicated 32-channel, single-frame 16x codec."
        )
    source = VAEEncode().encode(vae, pixels)[0]
    samples = source["samples"]
    if samples.ndim != 5 or samples.shape[1:3] != (32, 1):
        raise ValueError("Hunyuan refiner requires one encoded image frame with 32 codec channels.")
    value = noise_augmentation(data)
    data["refiner_noise_augmentation"] = value
    return tuple(HunyuanRefinerLatent.execute(positive, negative, source, value))
