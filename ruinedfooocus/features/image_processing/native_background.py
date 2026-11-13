"""Explicit local BiRefNet removal through the pinned native loader and mask node."""

from pathlib import Path

from ruinedfooocus.features.models.model_identity import cached_safetensors_header, file_signature
from ruinedfooocus.core.model_folders import model_folders


def validate_model(path):
    if Path(path).suffix.lower() not in (".safetensors", ".sft"):
        raise ValueError(
            "Native background removal currently requires BiRefNet safetensors weights."
        )
    if "bb.layers.1.blocks.0.attn.relative_position_index" not in cached_safetensors_header(path):
        raise ValueError("The selected local file does not have the native BiRefNet contract.")


def local_models(manager):
    choices = {}
    for root in model_folders(manager.paths, "background_removal", manager.get_abspath):
        if root.is_dir():
            for path in sorted(root.iterdir()):
                if not path.is_file() or path.suffix.lower() not in (".safetensors", ".sft"):
                    continue
                try:
                    validate_model(path)
                except (OSError, ValueError):
                    continue
                choices.setdefault(path.name, str(path.resolve()))
    return choices


def clear_model(owner):
    cached = getattr(owner, "_native_background", None)
    if cached is not None:
        from comfy.model_management import unload_model_and_clones

        unload_model_and_clones(cached[1].patcher)
        owner._native_background = None


def remove_background(owner, image, data, manager, check_cancelled):
    import numpy as np
    import torch
    from PIL import Image, ImageOps, ImageChops
    from comfy.bg_removal_model import load
    from comfy_extras.nodes_bg_removal import RemoveBackground
    from ruinedfooocus.features.generation.image_preparation import prepared_asset

    image = ImageOps.exif_transpose(image).convert("RGBA")
    if not data.get("rembg_force", False) and image.getextrema()[3][0] < 255:
        return image
    check_cancelled()
    name = data["rembg_native_model"]
    with prepared_asset(data, "background-removal", name, manager, "background_removal") as path:
        validate_model(path)
        signature = file_signature(path)
        cached = getattr(owner, "_native_background", None)
        if cached is None or cached[0] != signature:
            clear_model(owner)
            model = load(str(path))
            if model is None:
                raise ValueError(
                    "The native loader did not recognize this background-removal model."
                )
            owner._native_background = (signature, model)
        model = owner._native_background[1]
        pixels = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0)[None,]
        check_cancelled()
        with torch.inference_mode():
            mask = RemoveBackground.execute(model, pixels)[0]
        check_cancelled()
        if tuple(mask.shape) != (1, image.height, image.width) or not torch.isfinite(mask).all():
            raise ValueError("The native background model returned an invalid foreground mask.")
        alpha = Image.fromarray(
            (mask[0].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        )
        image.putalpha(ImageChops.multiply(image.getchannel("A"), alpha))
        if file_signature(path) != signature:
            raise ValueError("The background-removal model changed. Submit the image again.")
        data["native_background_source"] = dict(
            name=Path(name).name,
            size=signature[1],
            native="BiRefNet",
            device=str(model.load_device),
        )
        data["rembg_runtime_status"] = f"Native BiRefNet: {model.load_device}"
    return image
