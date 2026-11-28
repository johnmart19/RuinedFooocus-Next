"""Ordinary Union mode and AliMama inpainting through distinct native callers."""

from dataclasses import dataclass
import hashlib
import math
import sys

from ruinedfooocus.features.models.model_identity import cached_safetensors_header, file_signature

KEY = "_native_control_recipe"
SOURCE_KEY = "_native_control_source"
OUTPAINT_FIELDS = tuple(
    "control_outpaint_" + edge for edge in ("left", "top", "right", "bottom", "feather")
)
DEFAULTS = dict(
    control_recipe="ordinary",
    control_union_type="auto",
    control_inpaint_mode="mask",
    control_inpaint_denoise=1.0,
    **{key: 32 if key.endswith("feather") else 0 for key in OUTPAINT_FIELDS},
)
FIELDS = tuple(DEFAULTS)
IMAGE_FIELDS = ("control_inpaint_image", "control_inpaint_mask")
# Saved/API values must be validated before native paths are initialized. The
# native-body test checks this finite enum against the pinned upstream table.
UNION_TYPES = (
    "auto",
    "openpose",
    "depth",
    "hed/pidi/scribble/ted",
    "canny/lineart/anime_lineart/mlsd",
    "normal",
    "segment",
    "tile",
    "repaint",
)


def union_types():
    return UNION_TYPES


def recipes_for_family(family):
    from ruinedfooocus.features.image_editing.control_candidates import model_family

    family = model_family(family)
    return [
        "ordinary",
        *(["union"] if family in ("SDXL", "Flux") else []),
        *(["alimama-inpaint"] if family == "QwenImage" else []),
    ]


def reset_values():
    return {"control_recipe": "ordinary", **dict.fromkeys(IMAGE_FIELDS)}


def replay_values(metadata, *, clear_images=True):
    values = {key: metadata.get(key, default) for key, default in DEFAULTS.items()}
    if values["control_recipe"] not in ("ordinary", "union", "alimama-inpaint"):
        raise ValueError("The saved native control recipe is unavailable.")
    if values["control_recipe"] == "union" and values["control_union_type"] not in union_types():
        raise ValueError("The saved Union mode is unavailable.")
    if values["control_inpaint_mode"] not in ("mask", "outpaint"):
        raise ValueError("The saved AliMama source mode is unavailable.")
    denoise = values["control_inpaint_denoise"]
    if type(denoise) not in (int, float) or not 0 < denoise <= 1:
        raise ValueError("AliMama source denoise must be greater than 0 and at most 1.")
    from ruinedfooocus.features.image_editing.inpaint_utils import outpaint_margins

    outpaint_margins({key.removeprefix("control_"): values[key] for key in OUTPAINT_FIELDS})
    feather = values["control_outpaint_feather"]
    if type(feather) not in (int, float) or not 0 <= feather <= 128 or int(feather) != feather:
        raise ValueError("Saved canvas edge blending must be whole pixels from 0 to 128.")
    if clear_images:
        values.update(dict.fromkeys(IMAGE_FIELDS))
    return values


def api_values(payload, decode_image, defaults=None):
    values = {key: (defaults or {}).get(key, default) for key, default in DEFAULTS.items()}
    values.update((key, payload[key]) for key in FIELDS if key in payload)
    values["control_recipe"] = payload.get("control_recipe", "ordinary")
    values = replay_values(values, clear_images=False)
    for key in IMAGE_FIELDS:
        values[key] = decode_image(payload[key], preserve_alpha=True) if payload.get(key) else None
    return values


def control_contract(header):
    """Native load_controlnet_state_dict discriminators and input widths."""

    def shape(key):
        entry = header.get(key)
        value = entry.get("shape") if isinstance(entry, dict) else None
        return tuple(value) if isinstance(value, (list, tuple)) else ()

    union = shape("task_embedding") or shape("control_model.task_embedding")
    flux_union = shape("controlnet_mode_embedder.weight")
    qwen = shape("controlnet_x_embedder.weight")
    return dict(
        union_modes=(union[0] if len(union) == 2 else flux_union[0] if len(flux_union) == 2 else 0),
        alimama=(
            len(qwen) == 2
            and qwen[1] == 68
            and "controlnet_blocks.0.weight" in header
            and "transformer_blocks.0.img_mlp.net.0.proj.weight" in header
        ),
    )


@dataclass(frozen=True)
class CapturedControlRecipe:
    recipe: str
    union_type: str
    checkpoint_signature: tuple
    control_signature: tuple
    kind: str
    inpaint_mode: str = "mask"


@dataclass(frozen=True)
class PreparedControlSource:
    signature: tuple
    pixels: object
    mask: object
    frame: dict


def _source(data):
    from ruinedfooocus.features.image_editing.model_patches import _image

    image = _image(data.get("control_inpaint_image"), "AliMama source")
    mode = data.get("control_inpaint_mode", "mask")
    if mode not in ("mask", "outpaint"):
        raise ValueError("Choose Mask or Extend canvas for AliMama inpainting.")
    mask = (
        _image(data.get("control_inpaint_mask"), "AliMama mask", mask=True)
        if mode == "mask"
        else None
    )
    if image is None or mode == "mask" and mask is None:
        raise ValueError("Add the AliMama source and a white-edit/black-preserve mask.")
    if mask is not None and mask.size != image.size:
        raise ValueError("The AliMama mask must match its source image dimensions.")
    return image, mask, mode


def _source_signature(image, mask, mode, data):
    return (
        image.size,
        hashlib.sha256(image.tobytes()).hexdigest(),
        hashlib.sha256(mask.tobytes()).hexdigest() if mask is not None else None,
        mode,
        *(data.get(key, 32 if key.endswith("feather") else 0) for key in OUTPAINT_FIELDS),
    )


def capture_source_size(data, controls=None):
    """Copy the dedicated source and freeze dimensions without loading native code."""
    result = dict(data)
    recipe = result.get("control_recipe") or "ordinary"
    controls = controls if isinstance(controls, dict) else result.get("controlnet") or {}
    kind = str(controls.get("type", "")).lower()
    explicit = "_api_job" in result or result.get("task_type") == "tool_call"
    from ruinedfooocus.features.image_editing.control_candidates import CONTROL_KINDS

    if recipe != "ordinary" and (
        kind not in CONTROL_KINDS or recipe == "alimama-inpaint" and kind != "control"
    ):
        if explicit:
            raise ValueError("Select the matching native ControlNet tool for this named recipe.")
        result["control_recipe"] = "ordinary"
        result.pop(SOURCE_KEY, None)
        return result
    if result.get("control_recipe") != "alimama-inpaint":
        result.pop(SOURCE_KEY, None)
        return result
    if not explicit:
        # These controls are hidden by the selected ordinary ControlNet context.
        # Keep their UI/profile choices; omit them only from this request.
        result.update(
            model_patch_recipe="none",
            reference_adapter="none",
            native_edit_recipe="auto",
            reference_images=None,
            inpaint_toggle=False,
        )
    image, mask, mode = _source(result)
    if (
        result.get("inpaint_toggle")
        or result.get("reference_images")
        or result.get("model_patch_recipe") not in (None, "", "none")
        or result.get("reference_adapter") not in (None, "", "none")
    ):
        raise ValueError(
            "AliMama uses its own source without another inpaint, reference or patch recipe."
        )
    if result.get("automatic_upscale", "Off") != "Off":
        raise ValueError("Set automatic upscale to Off for the dedicated AliMama source recipe.")
    if result.get("input_image") is not None and (
        "_api_job" in result or result.get("task_type") == "tool_call"
    ):
        raise ValueError("AliMama uses control_inpaint_image without a separate input_image.")
    result.update(control_inpaint_image=image, control_inpaint_mask=mask, input_image=None)
    if mode != "outpaint":
        result.pop(SOURCE_KEY, None)
        if result.get("image_resize_mode", "preserve") == "preserve":
            from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY

            result.pop(SIZE_KEY, None)
        return result
    from ruinedfooocus.features.image_editing.inpaint_utils import outpaint_margins, outpaint_size
    from ruinedfooocus.features.generation.size_plan import KEY as SIZE_KEY, SizePlan

    borders = {
        key.removeprefix("control_"): result.get(key, 32 if key.endswith("feather") else 0)
        for key in OUTPAINT_FIELDS
    }
    margins = outpaint_margins(borders)
    if not any(margins):
        raise ValueError("Extend at least one canvas edge for AliMama outpainting.")
    # Validate the saved numeric fields before either native preparation or queueing.
    replay_values(result, clear_images=False)
    signature = _source_signature(image, mask, mode, result)
    prepared = result.get(SOURCE_KEY)
    if not isinstance(prepared, PreparedControlSource) or prepared.signature != signature:
        result.pop(SOURCE_KEY, None)
    width, height = outpaint_size(image.size, margins)
    result.update(
        width=width,
        height=height,
        custom_width=width,
        custom_height=height,
        aspect_ratios_selection="Custom...",
        size_mode="target",
        resolution_auto=False,
    )
    result[SIZE_KEY] = SizePlan.create((width, height), 1, alignment=16)
    return result


def pre_size_capture(data, controls=None):
    """Prepare one native canvas after runtime initialization; admission stays pure."""
    result = capture_source_size(data, controls)
    if (
        result.get("control_recipe") != "alimama-inpaint"
        or result.get("control_inpaint_mode", "mask") != "outpaint"
    ):
        return result
    image, mask, mode = _source(result)
    signature = _source_signature(image, mask, mode, result)
    prepared = result.get(SOURCE_KEY)
    if not isinstance(prepared, PreparedControlSource) or prepared.signature != signature:
        if sys.modules.get("nodes") is None:
            raise RuntimeError(
                "Initialize the native image runtime before capturing an outpaint canvas."
            )
        import torch
        from ruinedfooocus.features.image_editing.inpaint_utils import prepare_outpaint

        borders = {
            key.removeprefix("control_"): result.get(key, 32 if key.endswith("feather") else 0)
            for key in OUTPAINT_FIELDS
        }
        with torch.device("cpu"):
            pixels, paint, frame = prepare_outpaint(image, borders)
        result[SOURCE_KEY] = PreparedControlSource(signature, pixels, paint, frame)
    return result


def capture(data, manager, models, controls):
    """Bind specialized control intent and images before ordinary asset capture."""
    from ruinedfooocus.features.image_editing.control_candidates import (
        CONTROL_KINDS,
        selected_control,
    )
    from ruinedfooocus.features.generation.image_preparation import _control_checkpoint_family

    result = dict(data)
    result.pop(KEY, None)
    result.pop("native_control_recipe", None)
    kind = str((controls or {}).get("type", "")).lower()
    if (
        "_api_job" not in result
        and result.get("task_type") != "tool_call"
        and kind not in CONTROL_KINDS
    ):
        result["control_recipe"] = "ordinary"
        return result
    recipe = result.get("control_recipe") or "ordinary"
    if recipe == "ordinary":
        return result
    if recipe not in ("union", "alimama-inpaint"):
        raise ValueError("Choose Ordinary ControlNet, Union mode or AliMama inpainting.")
    replay_values(result, clear_images=False)
    if kind not in CONTROL_KINDS or not controls.get("model"):
        raise ValueError("Select a local ControlNet in Image tools for this named control recipe.")
    strength, start, stop = (
        controls.get(key, default)
        for key, default in (("strength", 1.0), ("start", 0.0), ("stop", 1.0))
    )
    if (
        any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in (strength, start, stop)
        )
        or not 0 <= strength <= 2
        or not 0 <= start < stop <= 1
    ):
        raise ValueError("Named ControlNet requires 0 ≤ strength ≤ 2 and 0 ≤ start < stop ≤ 1.")
    if str(result.get("image_engine", "auto")).startswith("sdcpp-"):
        raise ValueError("Named ControlNet recipes require the native image engine.")
    if (
        result.get("inpaint_toggle")
        or result.get("model_patch_recipe") not in (None, "", "none")
        or result.get("reference_images")
        or result.get("reference_adapter") not in (None, "", "none")
        or result.get("native_edit_recipe") not in (None, "", "auto")
    ):
        raise ValueError(
            "Use this control recipe's own source and mask without another inpaint or model patch."
        )
    checkpoint = models.get_file("checkpoints", result.get("base_model_name"))
    if checkpoint is None:
        raise ValueError("Select an installed image checkpoint for the control recipe.")
    family = _control_checkpoint_family(models, result.get("base_model_name"))
    if recipe not in recipes_for_family(family):
        raise ValueError(
            "This named ControlNet recipe does not match the selected checkpoint family."
        )
    path, signature, info = selected_control(
        manager, controls["model"], family, controls.get("model_signature")
    )
    from ruinedfooocus.core.model_formats import is_safetensors

    contract = control_contract(cached_safetensors_header(path)) if is_safetensors(path) else None
    selection = result.get("control_union_type", "auto")
    mode = result.get("control_inpaint_mode", "mask")
    if recipe == "union":
        if selection not in union_types():
            raise ValueError("Choose a native Union control type.")
        if contract is not None and not contract["union_modes"]:
            raise ValueError("The selected ControlNet has no native Union mode embedding.")
        if (
            contract is not None
            and selection != "auto"
            and union_types().index(selection) - 1 >= contract["union_modes"]
        ):
            raise ValueError(
                "This Union control type is absent from the selected model's mode table."
            )
    else:
        if family != "QwenImage" or kind != "control":
            raise ValueError(
                "AliMama inpainting requires ordinary Qwen Image and the local ControlNet tool."
            )
        if contract is not None and not contract["alimama"]:
            raise ValueError("Select a Qwen AliMama 68-input inpaint ControlNet.")
        result = capture_source_size(result, controls)
    result[KEY] = CapturedControlRecipe(
        recipe, selection, tuple(file_signature(checkpoint)), tuple(signature), kind, mode
    )
    result["image_engine"] = "native"
    return result


def has_source(data):
    captured = data.get(KEY)
    return isinstance(captured, CapturedControlRecipe) and captured.recipe == "alimama-inpaint"


def prepare_source(data, target_size, resize_mode="preserve", *, multiple=16):
    """Use the existing source/mask geometry, including native canvas extension."""
    captured = data.get(KEY)
    if not isinstance(captured, CapturedControlRecipe) or captured.recipe != "alimama-inpaint":
        return None
    source = data["control_inpaint_image"]
    if captured.inpaint_mode == "outpaint":
        if SOURCE_KEY not in data:
            data[SOURCE_KEY] = pre_size_capture(data)[SOURCE_KEY]
        prepared = data.get(SOURCE_KEY)
        image, mask, mode = _source(data)
        if not isinstance(
            prepared, PreparedControlSource
        ) or prepared.signature != _source_signature(image, mask, mode, data):
            raise ValueError("Capture the dedicated AliMama outpaint canvas before its SizePlan.")
        return prepared.pixels, prepared.mask, prepared.frame
    from ruinedfooocus.features.image_editing.image_geometry import prepare_reference
    from ruinedfooocus.features.image_editing.inpaint_utils import prepare_mask

    image, plan = prepare_reference(source, target_size, resize_mode, multiple=multiple)
    mask = prepare_mask(data["control_inpaint_mask"], plan)
    return image, mask, plan


def source_latent(vae, data, target_size, resize_mode, check_cancelled):
    """Use the same source for native VAEEncode, noise masking and conditioning."""
    captured = data.get(KEY)
    if not isinstance(captured, CapturedControlRecipe) or captured.recipe != "alimama-inpaint":
        return None
    from ruinedfooocus.app.resource_lifecycle import require_compute_owner
    from nodes import VAEEncode, SetLatentNoiseMask

    require_compute_owner()
    check_cancelled()
    if (
        vae.latent_channels != 16
        or vae.spacial_compression_encode() != 8
        or vae.output_channels != 3
    ):
        raise ValueError("AliMama requires its 16-channel, 8× RGB VAE.")
    pixels, mask, frame = prepare_source(data, target_size, resize_mode)
    check_cancelled()
    latent = VAEEncode().encode(vae, pixels)[0]
    check_cancelled()
    # The pinned inpaint graph uses a latent noise mask. Its canvas-extension
    # graph encodes the full padded canvas and preserves source pixels at finish.
    if captured.inpaint_mode == "mask":
        latent = SetLatentNoiseMask().set_mask(latent, mask)[0]
    return latent, pixels, mask, frame


def finish(decoded, data, frame):
    """Finish the dedicated source frame once; callers skip generic restoration."""
    captured = data.get(KEY)
    if not isinstance(captured, CapturedControlRecipe) or captured.recipe != "alimama-inpaint":
        raise ValueError("Finish only a captured AliMama source recipe.")
    from ruinedfooocus.features.image_editing.image_geometry import restore_reference_frame

    decoded = restore_reference_frame(decoded, frame)
    if captured.inpaint_mode == "outpaint":
        from ruinedfooocus.features.image_editing.inpaint_utils import blend_outpaint

        prepared = data.get(SOURCE_KEY)
        if not isinstance(prepared, PreparedControlSource) or frame != prepared.frame:
            raise ValueError("The AliMama decoded frame differs from its captured full canvas.")
        return blend_outpaint(decoded, prepared.pixels, prepared.mask, frame)
    return decoded


def apply(owner, positive, negative, control_net, vae, image, data, check_cancelled, *, mask=None):
    """Apply only the selected named native node; callers retain ordinary controls."""
    captured = data.get(KEY)
    if captured is None:
        return None
    if not isinstance(captured, CapturedControlRecipe):
        raise ValueError("Capture the named ControlNet recipe before compute.")
    from ruinedfooocus.app.resource_lifecycle import require_compute_owner

    require_compute_owner()
    check_cancelled()
    if (
        tuple(getattr(owner, "base_file_signature", ())) != captured.checkpoint_signature
        or tuple(getattr(owner, "controlnet_file_signature", ())) != captured.control_signature
    ):
        raise ValueError("The named control recipe belongs to another checkpoint or ControlNet.")
    options = data["controlnet"]
    strength, start, stop = (float(options[key]) for key in ("strength", "start", "stop"))
    from comfy_extras.nodes_controlnet import (
        SetUnionControlNetType,
        ControlNetInpaintingAliMamaApply,
    )

    if captured.recipe == "union":
        count = getattr(control_net.control_model, "num_control_type", None) or getattr(
            control_net.control_model, "num_union_modes", 0
        )
        from comfy.cldm.control_types import UNION_CONTROLNET_TYPES

        if (
            not count
            or captured.union_type != "auto"
            and UNION_CONTROLNET_TYPES[captured.union_type] >= count
        ):
            raise ValueError("The loaded ControlNet has no matching native Union mode.")
        selected = SetUnionControlNetType.execute(control_net, captured.union_type)[0]
        from nodes import ControlNetApplyAdvanced

        result = ControlNetApplyAdvanced().apply_controlnet(
            positive, negative, selected, image, strength, start, stop, vae=vae
        )
    else:
        native = type(owner.xl_base_patched.unet.model.model_config).__name__
        if (
            native != "QwenImage"
            or not control_net.concat_mask
            or vae.latent_channels != 16
            or vae.spacial_compression_encode() != 8
            or vae.output_channels != 3
        ):
            raise ValueError(
                "AliMama needs ordinary Qwen Image, its masked ControlNet and 16-channel 8× RGB VAE."
            )
        if mask is None:
            raise ValueError("Prepare this AliMama source and mask in the same image frame.")
        result = ControlNetInpaintingAliMamaApply.execute(
            positive, negative, control_net, vae, image, mask, strength, start, stop
        )
    check_cancelled()
    data["native_control_recipe"] = dict(
        recipe=captured.recipe,
        union_type=captured.union_type if captured.recipe == "union" else None,
        inpaint_mode=captured.inpaint_mode if captured.recipe == "alimama-inpaint" else None,
        model=options["model"],
        source_sha256=hashlib.sha256(data["control_inpaint_image"].tobytes()).hexdigest()
        if captured.recipe == "alimama-inpaint"
        else None,
    )
    return result[0], result[1]
