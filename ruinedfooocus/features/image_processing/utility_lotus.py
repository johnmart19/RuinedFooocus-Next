"""Standalone Lotus depth reuses the existing official native Lotus workflow."""

from ruinedfooocus.features.image_editing import lotus_depth


def validate_model(path):
    from ruinedfooocus.features.models.model_identity import cached_safetensors_header

    lotus_depth.validate_model(path)
    if any("heatmap_head." in key for key in cached_safetensors_header(path)):
        raise ValueError("Select a Lotus depth model; SDPose is a separate pose utility.")


COMPONENTS = (
    (
        "image_utility_model",
        "lotus-depth",
        ("checkpoints", "diffusion_models", "geometry_estimation"),
        validate_model,
    ),
    ("image_utility_vae", "lotus-vae", ("vae",), lotus_depth.validate_vae),
)


def parameters(data):
    return {}


def render(owner, request, pixels, data, manager, check_cancelled):
    import torch.nn.functional as F

    width, height = request.size
    # The native SD VAE crops to multiples of eight. Pad and crop at the utility
    # boundary so odd-sized originals keep their pixels, dimensions and alpha.
    padded = F.pad(
        pixels.movedim(-1, 1), (0, -width % 8, 0, -height % 8), mode="replicate"
    ).movedim(1, -1)
    files = {item.field: item for item in request.files}
    options = {
        "depth_lotus_model": files["image_utility_model"].name,
        "depth_lotus_vae": files["image_utility_vae"].name,
    }
    depth = lotus_depth.prepare_lotus(
        owner, padded, options, data, manager, check_cancelled=check_cancelled
    )
    return depth[:, :height, :width, :], {
        "native": "LotusD",
        "convention": "white-near",
        "native_output": "IMAGE",
        "padding": "replicate-to-8-crop",
    }
