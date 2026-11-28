"""Typed assets and native conditioning for Qwen Image 2.1 Fun ControlNet Union."""

from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps

from ruinedfooocus.core.model_formats import is_safetensors
from ruinedfooocus.features.models.model_identity import cached_safetensors_header, file_signature


KIND = "qwen-union"
CATEGORY = "model_patches"
ROLE = "qwen-image21-union"
DEFAULT_MODEL = "qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors"
GUIDE_MODES = ("canny", "depth", "grayscale", "hed", "lineart", "mlsd", "pose", "scribble")
MODES = (*GUIDE_MODES, "inpaint")


def is_union_header(header):
    projection = header.get("control_img_in.weight", {}).get("shape", ())
    return (
        isinstance(projection, (tuple, list))
        and len(projection) == 2
        and tuple(projection) == (4096, 129)
        and "control_blocks.0.img_mlp.out.weight" in header
        and "control_blocks.0.attn.norm_q.weight" in header
        and all(f"control_blocks.{index}.after_proj.weight" in header for index in range(16))
        and "control_blocks.16.after_proj.weight" not in header
    )


def validate_patch(path):
    try:
        compatible = is_safetensors(path) and is_union_header(cached_safetensors_header(path))
    except (OSError, ValueError):
        compatible = False
    if not compatible:
        raise ValueError("Choose a Qwen Image 2.1 Fun ControlNet Union safetensors patch.")


def model_name(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Choose a Qwen Image 2.1 Union model patch.")
    name = PurePosixPath(value.replace("\\", "/"))
    if name.is_absolute() or ".." in name.parts or ":" in str(name) or not is_safetensors(name):
        raise ValueError("Select a Union patch from the configured model patches folder.")
    return name.as_posix()


def local_patch(manager, name):
    for folder in manager.get_model_folders(CATEGORY):
        root = Path(folder).resolve()
        path = root / name
        if path.is_file():
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("The selected Union patch is outside its configured folder.")
            signature = tuple(file_signature(resolved))
            validate_patch(resolved)
            if signature != tuple(file_signature(resolved)):
                raise ValueError("The selected Union patch changed while inspecting it.")
            return resolved, signature
    return None


def catalogue_patches(manager):
    names = (DEFAULT_MODEL, "qwen_image_2.1_fun_controlnet_union_bf16.safetensors")
    return {
        entry["filename"]: key
        for key, entry in manager.DOWNLOADABLE_FILES.items()
        if entry.get("path") == "path_" + CATEGORY and entry.get("filename") in names
    }


def patch_choices(manager):
    catalog = catalogue_patches(manager)
    choices = set()
    for filename in manager.get_folder_list(CATEGORY):
        try:
            name = model_name(filename)
            if local_patch(manager, name) or name in catalog:
                choices.add(name)
        except (OSError, ValueError):
            continue
    return sorted(choices, key=lambda name: (name != DEFAULT_MODEL, name.casefold()))


def settings(values):
    mode = values.get("mode", "canny")
    if mode not in MODES:
        raise ValueError("Choose a supported Qwen Image 2.1 Union guide type.")
    strength, start, stop = (
        values.get(key, default)
        for key, default in (("strength", 1.0), ("start", 0.0), ("stop", 1.0))
    )
    if any(type(number) not in (int, float) for number in (strength, start, stop)) or not (
        0 <= strength <= 2 and 0 <= start < stop <= 1
    ):
        raise ValueError("Union requires strength from 0 to 2 and 0 ≤ start < stop ≤ 1.")
    result = {
        "type": KIND,
        "model": model_name(values.get("model", DEFAULT_MODEL)),
        "mode": mode,
        "strength": float(strength),
        "start": float(start),
        "stop": float(stop),
    }
    signature = values.get("model_signature")
    if signature is not None:
        if (
            not isinstance(signature, (list, tuple))
            or len(signature) != 4
            or not isinstance(signature[0], str)
            or any(type(number) is not int or number < 0 for number in signature[1:])
        ):
            raise ValueError("The saved Union model signature is invalid. Select the patch again.")
        result["model_signature"] = list(signature)
    return result


def replay_values(metadata):
    """Restore the recipe controls while images remain request-owned uploads."""
    controls = metadata.get("controlnet")
    if not isinstance(controls, dict) or controls.get("type") != KIND:
        return {}
    options = settings(controls)
    return {
        "cn_selection": "Qwen Image 2.1 Union",
        "qwen_union_model": options["model"],
        "qwen_union_model_signature": options.get("model_signature"),
        "qwen_union_mode": options["mode"],
        "cn_strength": options["strength"],
        "cn_start": options["start"],
        "cn_stop": options["stop"],
    }


def _pixels(value, mode):
    if value is None:
        return None
    if not isinstance(value, Image.Image) or not 0 < value.width * value.height <= 16 * 1024**2:
        raise ValueError("Union inputs must be images of at most 16 megapixels.")
    return ImageOps.exif_transpose(value).convert(mode)


def inputs(data, mode):
    guide = None if mode == "inpaint" else _pixels(data.get("qwen_union_guide"), "RGB")
    source = _pixels(data.get("qwen_union_source"), "RGB")
    mask = _pixels(data.get("qwen_union_mask"), "L")
    if mode != "inpaint" and guide is None:
        raise ValueError("Add a Union control guide for the selected guide type.")
    if (source is None) != (mask is None) or mode == "inpaint" and source is None:
        raise ValueError("Add both an inpaint source image and its black-and-white mask.")
    if mask is not None and (source.size != mask.size or mask.getextrema()[1] < 128):
        raise ValueError("The Union mask must match the source and mark the edit area in white.")
    if mode == "grayscale":
        guide = ImageOps.grayscale(guide).convert("RGB")
    return guide, source, mask


def capture(data, manager, controls):
    from ruinedfooocus.features.generation.image_preparation import _artifact

    options = settings(controls)
    if (data.get("inpaint_toggle") or data.get("reference_images")) and (
        "_api_job" in data or data.get("task_type") == "tool_call"
    ):
        raise ValueError("Use the Union tool's own source and mask for this operation.")
    guide, source, mask = inputs(data, options["mode"])
    result = dict(
        data,
        controlnet=options,
        qwen_union_guide=guide,
        qwen_union_source=source,
        qwen_union_mask=mask,
        inpaint_toggle=False,
        reference_images=None,
    )
    name = options["model"]
    local = local_patch(manager, name)
    if local is not None:
        signature = list(local[1])
        if options.get("model_signature") not in (None, signature):
            raise ValueError("The selected Union patch changed. Select it again before generating.")
        options["model_signature"] = signature
    key = catalogue_patches(manager).get(name)
    if local is None and key is None:
        raise FileNotFoundError(f"Install the selected Union model patch: {name}.")
    artifact = _artifact(manager, ROLE, CATEGORY, name, key)
    if local is not None and (artifact.request.destination, artifact.signature) != local:
        raise ValueError("The selected Union patch changed during capture. Select it again.")
    return result, artifact


def apply_union(model, vae, data, manager):
    import numpy as np
    import torch
    import folder_paths
    from comfy.ldm.qwen_image21.model import QwenImage21FunControl, QwenImage21Transformer2DModel
    from comfy_extras.nodes_model_patch import ModelPatchLoader, ZImageFunControlnet

    from ruinedfooocus.features.generation.image_preparation import prepared_asset

    options = settings(data["controlnet"])
    target = model.get_model_object("diffusion_model")
    if (
        not isinstance(target, QwenImage21Transformer2DModel)
        or len(target.transformer_blocks) != 32
    ):
        raise ValueError("Union requires a Qwen Image 2.1 checkpoint with 32 transformer blocks.")
    if (
        vae.latent_channels != 64
        or vae.output_channels != 4
        or vae.spacial_compression_encode() != 16
    ):
        raise ValueError("Union requires the Qwen Image 2.1 64-channel 16× RGBA VAE.")
    guide, source, mask = inputs(data, options["mode"])

    def tensor(image):
        return (
            torch.from_numpy(np.array(image)).float().unsqueeze(0) / 255
            if image is not None
            else None
        )

    name = options["model"]
    with prepared_asset(data, ROLE, name, manager, CATEGORY) as path:
        validate_patch(path)
        native_path = Path(folder_paths.get_full_path_or_raise(CATEGORY, name)).resolve()
        if native_path != path.resolve():
            raise ValueError("The native Union patch path differs from the submitted model file.")
        cache_key = tuple(file_signature(path))
        if options.get("model_signature") not in (None, list(cache_key)):
            raise ValueError("The selected Union patch changed. Select it again before generating.")
        data["controlnet"] = dict(options, model_signature=list(cache_key))
        if not options["strength"]:
            return model
        cached = data.get("_qwen_union_patch")
        if cached is None or cached[0] != cache_key:
            patch = ModelPatchLoader().load_model_patch(name)[0]
            if not isinstance(patch.model, QwenImage21FunControl):
                raise ValueError("The selected file did not load as a Qwen Image 2.1 Union patch.")
            data["_qwen_union_patch"] = cache_key, patch
        else:
            patch = cached[1]

    return ZImageFunControlnet().diffsynth_controlnet(
        model=model,
        model_patch=patch,
        vae=vae,
        image=tensor(guide),
        inpaint_image=tensor(source),
        mask=tensor(mask),
        strength=options["strength"],
        start_percent=options["start"],
        end_percent=options["stop"],
    )[0]
