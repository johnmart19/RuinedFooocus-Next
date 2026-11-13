"""Immutable frame-bound mask utilities using the pinned native mask nodes."""

from dataclasses import dataclass
import re

from ruinedfooocus.features.image_processing import mask_native as native
from ruinedfooocus.features.image_processing.image_geometry_tools import (
    FrozenPixels,
    NATIVE_REVISION,
    capture_source,
)
from ruinedfooocus.features.image_processing.native_vision import _number

KINDS = ("image-mask",)
LABELS = {"image-mask": "Edit mask"}
KEY = "_image_mask_request"
OPERATIONS = ("grow", "feather", "composite", "channel", "color", "threshold", "invert")
COMPOSITE_OPERATIONS = ("multiply", "add", "subtract", "and", "or", "xor")
CHANNELS = ("red", "green", "blue", "alpha")
DEFAULTS = dict(
    mask_operation="grow",
    mask_expand=0,
    mask_tapered_corners=True,
    mask_left=0,
    mask_top=0,
    mask_right=0,
    mask_bottom=0,
    mask_composite_operation="multiply",
    mask_channel="alpha",
    mask_color="#000000",
    mask_threshold=0.5,
    mask_invert=False,
)
FIELDS = tuple(DEFAULTS)
IMAGE_FIELDS = ("mask_image", "mask_reference")
NODES = dict(
    zip(
        OPERATIONS,
        (
            "GrowMask",
            "FeatherMask",
            "MaskComposite",
            "ImageToMask",
            "ImageColorToMask",
            "ThresholdMask",
            "InvertMask",
        ),
    )
)


@dataclass(frozen=True)
class MaskRequest:
    source: FrozenPixels
    mask: FrozenPixels | None
    reference: FrozenPixels | None
    parameters: tuple[tuple[str, object], ...]
    native_signatures: tuple
    output_settings: tuple[tuple[str, object], ...]


def parameters(data):
    values = {name: data.get(name, default) for name, default in DEFAULTS.items()}
    for name, choices in (
        ("mask_operation", OPERATIONS),
        ("mask_composite_operation", COMPOSITE_OPERATIONS),
        ("mask_channel", CHANNELS),
    ):
        if not isinstance(values[name], str) or values[name] not in choices:
            raise ValueError(f"Choose a supported {name.replace('_', ' ')}.")
    for name in ("mask_invert", "mask_tapered_corners"):
        if type(values[name]) is not bool:
            raise ValueError(f"{name.replace('_', ' ').capitalize()} must be true or false.")
    for name in ("mask_expand", "mask_left", "mask_top", "mask_right", "mask_bottom"):
        values[name] = _number(
            values[name],
            name.replace("_", " "),
            -16384 if name == "mask_expand" else 0,
            16384,
            integer=True,
        )
    values["mask_threshold"] = _number(values["mask_threshold"], "Mask threshold", 0, 1)
    if not isinstance(values["mask_color"], str) or not re.fullmatch(
        r"#[0-9a-fA-F]{6}", values["mask_color"]
    ):
        raise ValueError("Choose a six-digit RGB mask color.")
    values["mask_color"] = values["mask_color"].upper()
    return values


def _samples(source):
    import numpy as np

    array = np.frombuffer(source.pixels, dtype=source.dtype).reshape(
        source.size[1], source.size[0], source.channels
    )
    return array.astype(np.float32) / (
        255 if source.dtype == "uint8" else 65535 if source.dtype == "uint16" else 1
    )


def _mask_samples(source):
    import numpy as np

    values = _samples(source)
    if source.channels != 1 and not np.array_equal(values[..., :1], values[..., 1:2]):
        raise ValueError(
            "Mask inputs must contain grayscale values; extract an image channel explicitly first."
        )
    if source.channels != 1 and not np.array_equal(values[..., :1], values[..., 2:3]):
        raise ValueError(
            "Mask inputs must contain grayscale values; extract an image channel explicitly first."
        )
    values = values[..., 0]
    if values.min() < 0 or values.max() > 1:
        raise ValueError("Mask values must be between zero and one; white selects the image.")
    return values


def capture_assets(data, manager, models=None):
    from PIL import Image
    from ruinedfooocus.features.image_processing.native_image_formats import format_settings

    values = parameters(data)
    if (
        data.get("inpaint_toggle")
        or data.get("model_recipe_path")
        or data.get("reference_adapter") not in (None, "", "none")
    ):
        raise ValueError(
            "Turn off inpainting, generation recipes and reference adapters before editing a mask."
        )
    source = capture_source(data.get("input_image"), data.get("image_source"))
    operation = values["mask_operation"]
    mask = reference = None
    if operation in ("channel", "color"):
        if data.get("mask_image") is not None or data.get("mask_reference") is not None:
            raise ValueError(
                "Channel and color extraction use the main image; clear uploaded masks."
            )
        if operation == "channel":
            channel = CHANNELS.index(values["mask_channel"])
            if channel >= source.channels:
                raise ValueError("The source has no selected channel; choose an existing channel.")
            samples = _samples(source)[..., channel]
            if samples.min() < 0 or samples.max() > 1:
                raise ValueError("Selected mask channel values must be between zero and one.")
        elif source.channels < 3:
            raise ValueError("Color extraction requires an RGB or RGBA source image.")
    else:
        if data.get("mask_image") is None:
            raise ValueError("Add a grayscale mask matching the main image frame.")
        mask = capture_source(data["mask_image"])
        _mask_samples(mask)
        if mask.size != source.size:
            raise ValueError(
                "The mask must match the oriented main image frame; resize explicitly first."
            )
        if operation == "composite":
            if data.get("mask_reference") is None:
                raise ValueError("Add the second grayscale mask to combine.")
            reference = capture_source(data["mask_reference"])
            _mask_samples(reference)
            if reference.size != source.size:
                raise ValueError(
                    "Both masks must share the main image frame; no crop or stretch is implicit."
                )
        elif data.get("mask_reference") is not None:
            raise ValueError("Only Combine masks uses a second mask.")
    choice = format_settings(data)
    precision_source = source if operation in ("channel", "color") else mask
    if (
        any(frame is not None and frame.dtype != "uint8" for frame in (precision_source, reference))
        and operation != "color"
        and not (
            choice["output_format"] == "exr"
            or (choice["output_format"] == "png" and choice["output_bit_depth"] == "16-bit")
        )
    ):
        raise ValueError("Precision masks require PNG16 or EXR output.")
    if choice["output_color_space"] != "sRGB":
        raise ValueError(
            "Masks are scalar selections; choose sRGB output without color conversion."
        )
    if choice["output_format"] == "avif":
        raise ValueError("Save reusable selection masks as PNG or EXR.")
    request = MaskRequest(
        source,
        mask,
        reference,
        tuple(values.items()),
        native.source_signatures(),
        tuple(choice.items()),
    )
    result = dict(
        data, **values, controlnet={"type": KINDS[0]}, input_image=Image.new("L", source.size)
    )
    result[KEY] = request
    result.update(choice)
    for name in (*IMAGE_FIELDS, "image_utility_source"):
        result.pop(name, None)
    return result, ()


def worker_data(data):
    from PIL import Image
    from ruinedfooocus.features.generation.image_preparation import KEY as ASSET_KEY
    from ruinedfooocus.features.generation.request_metadata import SOURCE_GENERATION
    from ruinedfooocus.features.gallery.editor_session import KEY as EDITOR_KEY
    from ruinedfooocus.features.gallery.image_metadata import generation_provenance
    from ruinedfooocus.features.image_processing.image_filters import worker_data as common

    request = data.get(KEY)
    if not isinstance(request, MaskRequest):
        raise ValueError("Capture the mask before processing it.")
    result = {
        key: value
        for key, value in data.items()
        if key
        in (
            KEY,
            ASSET_KEY,
            EDITOR_KEY,
            SOURCE_GENERATION,
            "_image_workflow_step",
            "_api_job",
            "_job",
            "task_id",
            "task_type",
            "index",
            "silent",
            "show_preview",
            "output_format",
            "output_bit_depth",
            "output_color_space",
            "output_crf",
        )
    }
    provenance = generation_provenance(result.pop(SOURCE_GENERATION, None))
    if provenance is not None:
        result[SOURCE_GENERATION] = provenance
    result.update(dict(request.parameters))
    result.update(dict(request.output_settings))
    result.update(controlnet={"type": KINDS[0]}, input_image=Image.new("L", request.source.size))
    return common(result)


def _process_pixels(request, policy, check_cancelled):
    import torch

    values = dict(request.parameters)
    operation = values["mask_operation"]
    check_cancelled()
    ops = native.operations(request.native_signatures)
    with torch.device(policy.device), torch.inference_mode():
        if operation in ("channel", "color"):
            pixels = (
                torch.from_numpy(_samples(request.source).copy()).to(policy.device).unsqueeze(0)
            )
            result = (
                ops.ImageToMask(None, pixels, values["mask_channel"])[0]
                if operation == "channel"
                else ops.ImageColorToMask(None, pixels, int(values["mask_color"][1:], 16))[0]
            )
        else:
            result = (
                torch.from_numpy(_mask_samples(request.mask).copy()).to(policy.device).unsqueeze(0)
            )
        if values["mask_invert"]:
            result = ops.InvertMask(None, result)[0]
        if operation == "grow":
            # Each pass is native; checking between passes keeps large requests cancellable.
            expand = values["mask_expand"]
            for _ in range(abs(expand)):
                check_cancelled()
                result = ops.GrowMask(
                    None, result, -1 if expand < 0 else 1, values["mask_tapered_corners"]
                )[0]
        elif operation == "feather":
            result = ops.FeatherMask(
                None,
                result,
                *(values["mask_" + name] for name in ("left", "top", "right", "bottom")),
            )[0]
        elif operation == "composite":
            reference = (
                torch.from_numpy(_mask_samples(request.reference).copy())
                .to(policy.device)
                .unsqueeze(0)
            )
            result = ops.MaskComposite(
                None, result, reference, 0, 0, values["mask_composite_operation"]
            )[0]
        elif operation == "threshold":
            result = ops.ThresholdMask(None, result, values["mask_threshold"])[0]
        elif operation == "invert":
            result = ops.InvertMask(None, result)[0]
        result = ops.MaskToImage(None, result)[0]
    check_cancelled()
    if native.source_signatures() != request.native_signatures:
        raise ValueError("The native mask source changed during processing. Submit the mask again.")
    width, height = request.source.size
    if (
        not isinstance(result, torch.Tensor)
        or tuple(result.shape) != (1, height, width, 3)
        or not torch.isfinite(result).all()
        or result.min() < 0
        or result.max() > 1
    ):
        raise ValueError("The native mask operation returned invalid selection pixels.")
    return result[0].detach().cpu().numpy().copy()


def process(owner, data, manager, check_cancelled):
    import numpy as np
    from PIL import Image
    from ruinedfooocus.features.image_processing.native_image_formats import advanced_output
    from ruinedfooocus.features.image_processing.utility_device import run_utility

    request = data.get(KEY)
    data.pop("image_utility_source", None)
    if not isinstance(request, MaskRequest) or data.get("controlnet", {}).get("type") != KINDS[0]:
        raise ValueError("Capture this mask's inputs before processing. Submit them again.")
    values = dict(request.parameters)
    operation = values["mask_operation"]
    data.update(dict(request.output_settings))
    execution = run_utility(
        KINDS[0],
        lambda policy: _process_pixels(request, policy, check_cancelled),
        check_cancelled=check_cancelled,
        cpu_only_reason="Pinned GrowMask uses SciPy and NumPy" if operation == "grow" else None,
    )
    pixels = execution.value
    width, height = request.source.size
    information = dict(
        operation=KINDS[0],
        native_node=NODES[operation],
        native_revision=NATIVE_REVISION,
        width=width,
        height=height,
        source_sha256=request.source.source_sha256,
        source_dtype=request.source.dtype,
        source_color_space=request.source.color_space,
        mask_dtype=request.mask.dtype if request.mask else None,
        reference_dtype=request.reference.dtype if request.reference else None,
        mask_sha256=request.mask.source_sha256 if request.mask else None,
        reference_sha256=request.reference.source_sha256 if request.reference else None,
        mask_role="inpaint-selection",
        mask_values="white selects; black protects; soft grayscale retained",
        frame="oriented main image frame; no implicit crop or resize",
        working_dtype="float32",
        output_precision="float32 selection; export precision follows explicit output settings",
        color_key_policy="native RGB8 rounded equality" if operation == "color" else None,
        color_policy="scalar selection; no ICC or color-space conversion",
        compute=execution.metadata(),
        **values,
    )
    check_cancelled()
    data["image_utility_source"] = information
    if advanced_output(data):
        return [pixels]
    return [Image.fromarray((pixels * 255).round().astype(np.uint8), "RGB")]


def output_metadata(data):
    from copy import deepcopy

    request = data[KEY]
    return dict(
        software="RuinedFooocus",
        prompt="",
        negative="",
        controlnet={"type": KINDS[0]},
        width=request.source.size[0],
        height=request.source.size[1],
        image_utility_source=deepcopy(data["image_utility_source"]),
        **dict(request.parameters),
        **dict(request.output_settings),
    )


def replay_values(metadata):
    from ruinedfooocus.features.image_processing.native_image_formats import format_settings

    return dict(
        **parameters(metadata),
        **format_settings(metadata),
        **dict.fromkeys(IMAGE_FIELDS),
        input_image=None,
        image_source=None,
        inpaint_toggle=False,
        cn_selection=LABELS[KINDS[0]],
        prompt="",
        negative="",
        automatic_upscale="Off",
        automatic_upscale_target="Generated image",
        model_recipe_path="",
        reference_adapter="none",
    )
