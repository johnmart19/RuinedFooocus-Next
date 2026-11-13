"""Immutable, model-free still-image transforms using pinned native V3 nodes."""

from dataclasses import dataclass
import hashlib
import re

from ruinedfooocus.features.image_processing import geometry_native as native

KINDS = ("image-transform",)
LABELS = {"image-transform": "Transform image"}
KEY = "_image_transform_request"
NATIVE_REVISION = "989d93d0d968e704c589ce0146bb18c77f6b50cc"
MAX_PIXELS = 16777216
MAX_DIMENSION = 16384
INTERPOLATIONS = ("nearest-exact", "bilinear", "area", "bicubic")
DEFAULTS = dict(
    transform_operation="resize",
    transform_width=0,
    transform_height=0,
    transform_fit="contain",
    transform_interpolation="bicubic",
    transform_x=0,
    transform_y=0,
    transform_left=0,
    transform_top=0,
    transform_right=0,
    transform_bottom=0,
    transform_padding_style="transparent",
    transform_padding_color="#000000",
    transform_feather=0,
    transform_rotation="none",
    transform_flip="horizontal",
)
FIELDS = tuple(DEFAULTS)
IMAGE_FIELDS = ("mask_image",)


@dataclass(frozen=True)
class FrozenPixels:
    size: tuple[int, int]
    channels: int
    dtype: str
    pixels: bytes
    source_sha256: str
    color_space: str
    input_mode: str
    has_icc_profile: bool


@dataclass(frozen=True)
class TransformRequest:
    source: FrozenPixels
    parameters: tuple[tuple[str, object], ...]
    output_size: tuple[int, int]
    native_signatures: tuple
    output_settings: tuple[tuple[str, object], ...]
    selection: FrozenPixels | None = None
    crop_region: tuple[int, int, int, int] | None = None


def _integer(value, name, minimum=0):
    if (
        type(value) not in (int, float)
        or not minimum <= value <= MAX_DIMENSION
        or int(value) != value
    ):
        raise ValueError(f"{name} must be a whole number from {minimum} to {MAX_DIMENSION}.")
    return int(value)


def parameters(data):
    values = {name: data.get(name, default) for name, default in DEFAULTS.items()}
    choices = dict(
        transform_operation=("resize", "crop", "crop-to-mask", "pad", "rotate", "flip"),
        transform_fit=("contain", "stretch", "center-crop"),
        transform_interpolation=INTERPOLATIONS,
        transform_padding_style=("transparent", "color", "mirror"),
        transform_rotation=("none", "90 degrees", "180 degrees", "270 degrees"),
        transform_flip=("horizontal", "vertical"),
    )
    for name, options in choices.items():
        if values[name] not in options:
            raise ValueError(
                f"Choose a supported {name.replace('transform_', '').replace('_', ' ')}."
            )
    for name in ("width", "height", "x", "y", "left", "top", "right", "bottom", "feather"):
        field = "transform_" + name
        values[field] = _integer(values[field], name.capitalize())
    if not isinstance(values["transform_padding_color"], str) or not re.fullmatch(
        r"#[0-9A-Fa-f]{6}", values["transform_padding_color"]
    ):
        raise ValueError("Choose a six-digit RGB padding color.")
    values["transform_padding_color"] = values["transform_padding_color"].upper()
    return values


def output_size(source_size, values, crop_region=None):
    width, height = source_size
    operation = values["transform_operation"]
    target_w, target_h = values["transform_width"], values["transform_height"]
    if operation == "resize":
        if target_w or target_h:
            if not target_w:
                width, height = max(1, round(width * target_h / height)), target_h
            elif not target_h:
                width, height = target_w, max(1, round(height * target_w / width))
            elif values["transform_fit"] == "contain":
                scale = min(target_w / width, target_h / height)
                width, height = max(1, round(width * scale)), max(1, round(height * scale))
            else:
                width, height = target_w, target_h
    elif operation == "crop":
        x, y = values["transform_x"], values["transform_y"]
        width, height = target_w or width - x, target_h or height - y
        if x + width > source_size[0] or y + height > source_size[1]:
            raise ValueError("The crop rectangle must fit inside the oriented source image.")
    elif operation == "crop-to-mask":
        if crop_region is None:
            raise ValueError("Capture an exact-frame selection mask before cropping to it.")
        x, y, width, height = crop_region
        if min(x, y) < 0 or x + width > source_size[0] or y + height > source_size[1]:
            raise ValueError("The selection bounds must fit inside the oriented source image.")
    elif operation == "pad":
        width += values["transform_left"] + values["transform_right"]
        height += values["transform_top"] + values["transform_bottom"]
    elif operation == "rotate" and values["transform_rotation"] in ("90 degrees", "270 degrees"):
        width, height = height, width
    if min(width, height) < 1 or max(width, height) > MAX_DIMENSION or width * height > MAX_PIXELS:
        raise ValueError(
            "Transformed images must fit within 16 megapixels and 16384 pixels per side."
        )
    return width, height


def capture_source(value, provenance=None):
    import numpy as np
    from PIL import Image, ImageOps
    from ruinedfooocus.features.image_processing.native_image_formats import ImportedImage

    color, mode, profile = "sRGB", "array", False
    if isinstance(value, ImportedImage):
        color, mode = value.color_space, value.bit_depth
        array = np.asarray(value.pixels)
    elif isinstance(value, Image.Image):
        if getattr(value, "n_frames", 1) != 1 or value.width * value.height > MAX_PIXELS:
            raise ValueError("Choose one still source image of at most 16 megapixels.")
        if provenance and (
            provenance.get("original_dtype") not in (None, "uint8")
            or provenance.get("conversion") not in (None, "none")
        ):
            raise ValueError(
                "Use the native original for precision transforms; this source is an 8-bit working copy."
            )
        mode, profile = value.mode, bool(value.info.get("icc_profile"))
        oriented = ImageOps.exif_transpose(value)
        if mode in ("1", "L", "LA", "P", "RGB", "RGBA"):
            oriented = oriented.convert("RGBA")
        elif mode not in ("I;16", "I;16L", "F"):
            raise ValueError("Choose RGB, RGBA, grayscale or native precision image pixels.")
        array = np.asarray(oriented)
    elif isinstance(value, np.ndarray):
        array = value
    else:
        raise ValueError("Add one source image before transforming it.")
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError("Transform one grayscale, RGB or RGBA still image.")
    height, width, channels = array.shape
    if min(width, height) < 1 or max(width, height) > MAX_DIMENSION or width * height > MAX_PIXELS:
        raise ValueError("Choose one still source image of at most 16 megapixels.")
    if (
        array.dtype
        not in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("float16"), np.dtype("float32"))
        or not np.isfinite(array).all()
    ):
        raise ValueError("Transform pixels must be finite uint8, uint16 or float samples.")
    frozen = np.ascontiguousarray(array).tobytes()
    identity = f"{width},{height},{channels},{array.dtype},{color}:".encode()
    return FrozenPixels(
        (width, height),
        channels,
        str(array.dtype),
        frozen,
        hashlib.sha256(identity + frozen).hexdigest(),
        color,
        mode,
        profile,
    )


def capture_selection(value, source_size):
    """Capture a grayscale selection and its nonzero bounds without resizing it."""
    import numpy as np
    from PIL import Image

    if value is None:
        raise ValueError("Add a grayscale selection mask matching the main image frame.")
    selection = capture_source(value)
    if selection.size != source_size:
        raise ValueError("The selection mask must match the oriented main image frame exactly.")
    samples = np.frombuffer(selection.pixels, dtype=selection.dtype).reshape(
        selection.size[1], selection.size[0], selection.channels
    )
    maximum = 255 if selection.dtype == "uint8" else 65535 if selection.dtype == "uint16" else 1
    if samples.min() < 0 or samples.max() > maximum:
        raise ValueError(
            "Selection mask values must be between zero and one; white selects pixels."
        )
    if selection.channels != 1 and (
        not np.array_equal(samples[..., 0], samples[..., 1])
        or not np.array_equal(samples[..., 0], samples[..., 2])
    ):
        raise ValueError(
            "Use a grayscale selection mask; extract a color channel explicitly first."
        )
    if selection.channels == 4 and not (samples[..., 3] == maximum).all():
        raise ValueError("Use an opaque grayscale selection mask; extract alpha explicitly first.")
    bounds = Image.fromarray((samples[..., 0] > 0).astype(np.uint8)).getbbox()
    if bounds is None:
        raise ValueError("The selection mask is empty; select at least one source pixel.")
    left, top, right, bottom = bounds
    return selection, (left, top, right - left, bottom - top)


def capture_assets(data, manager, models=None):
    from PIL import Image
    from ruinedfooocus.features.image_processing.native_image_formats import format_settings

    values = parameters(data)
    source = capture_source(data.get("input_image"), data.get("image_source"))
    selection = crop_region = None
    if values["transform_operation"] == "crop-to-mask":
        selection, crop_region = capture_selection(data.get("mask_image"), source.size)
    elif data.get("mask_image") is not None:
        raise ValueError("Only Crop to selection mask uses an input selection mask.")
    size = output_size(source.size, values, crop_region)
    choice = format_settings(data)
    creates_alpha = (
        values["transform_operation"] == "pad"
        and values["transform_padding_style"] == "transparent"
    )
    if choice["output_format"] == "avif" and (source.channels == 4 or creates_alpha):
        raise ValueError("Native AVIF cannot preserve alpha. Choose PNG or EXR.")
    if source.dtype != "uint8" and not (
        choice["output_format"] == "exr"
        or (choice["output_format"] == "png" and choice["output_bit_depth"] == "16-bit")
    ):
        raise ValueError("Precision transforms require PNG16 or EXR output.")
    if choice["output_color_space"] != source.color_space:
        raise ValueError(
            "Transform output color space must match the source; convert color explicitly before transforming."
        )
    if source.color_space != "sRGB" and choice["output_format"] != "exr":
        raise ValueError("Linear and HDR source transforms require EXR output.")
    if choice["output_format"] != "exr" and source.dtype.startswith("float"):
        import numpy as np

        samples = np.frombuffer(source.pixels, dtype=source.dtype)
        if samples.min() < 0 or samples.max() > 1:
            raise ValueError("Out-of-range floating-point transforms require EXR output.")
    request = TransformRequest(
        source,
        tuple(values.items()),
        size,
        native.source_signatures(),
        tuple(choice.items()),
        selection,
        crop_region,
    )
    # The shared worker protocol needs dimensions only. Frozen source bytes above
    # are authoritative; this placeholder never supplies processing pixels.
    result = dict(
        data, **values, controlnet={"type": KINDS[0]}, input_image=Image.new("L", source.size)
    )
    result[KEY] = request
    result.update(choice)
    result.pop("mask_image", None)
    result.pop("image_utility_source", None)
    return result, ()


def worker_data(data):
    from PIL import Image
    from ruinedfooocus.features.generation.image_preparation import KEY as ASSET_KEY
    from ruinedfooocus.features.generation.request_metadata import SOURCE_GENERATION
    from ruinedfooocus.features.gallery.editor_session import KEY as EDITOR_KEY
    from ruinedfooocus.features.gallery.image_metadata import generation_provenance
    from ruinedfooocus.features.image_processing.image_filters import worker_data as common

    request = data.get(KEY)
    if not isinstance(request, TransformRequest):
        raise ValueError("Capture the transform source before processing it.")
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
        )
    }
    if SOURCE_GENERATION in result:
        provenance = generation_provenance(result[SOURCE_GENERATION])
        if provenance is None:
            result.pop(SOURCE_GENERATION)
        else:
            result[SOURCE_GENERATION] = provenance
    result.update(dict(request.parameters))
    result.update(dict(request.output_settings))
    result.update(controlnet={"type": KINDS[0]}, input_image=Image.new("L", request.source.size))
    result = common(result)
    result["custom_width"], result["custom_height"] = request.output_size
    return result


def _process_pixels(request, policy, check_cancelled):
    import numpy as np
    import torch

    check_cancelled()
    ops = native.operations(request.native_signatures)
    source, values = request.source, dict(request.parameters)
    array = np.frombuffer(source.pixels, dtype=source.dtype).reshape(
        source.size[1], source.size[0], source.channels
    )
    pixels = (
        torch.from_numpy(array.copy()).to(device=policy.device, dtype=torch.float32).unsqueeze(0)
    )
    if source.dtype in ("uint8", "uint16"):
        pixels.div_(255 if source.dtype == "uint8" else 65535)
    width, height = request.output_size
    operation = values["transform_operation"]
    with torch.inference_mode():
        if operation == "resize":
            pixels = ops.resize(
                None,
                pixels,
                values["transform_interpolation"],
                dict(
                    resize_type="scale dimensions",
                    width=width,
                    height=height,
                    crop="center" if values["transform_fit"] == "center-crop" else "disabled",
                ),
            )[0]
        elif operation == "crop":
            pixels = ops.crop(
                None,
                pixels,
                dict(x=values["transform_x"], y=values["transform_y"], width=width, height=height),
            )[0]
        elif operation == "crop-to-mask":
            region = ops.bounding_box(None, *request.crop_region)[0]
            pixels = ops.crop(None, pixels, region)[0]
        elif operation == "pad":
            padding = dict(
                padding_type="pad dimensions",
                **{
                    name: values["transform_" + name] for name in ("left", "top", "right", "bottom")
                },
            )
            style = values["transform_padding_style"]
            options = dict(
                padding_style="mirror image" if style == "mirror" else "color",
                color=values["transform_padding_color"],
                feathering=values["transform_feather"],
            )
            original = pixels
            pixels = ops.pad(None, pixels, padding, options)[0]
            if style == "transparent":
                check_cancelled()
                alpha = (
                    original[..., 3:]
                    if source.channels == 4
                    else torch.ones_like(original[..., :1])
                )
                alpha = ops.pad(
                    None,
                    alpha,
                    padding,
                    dict(
                        padding_style="color",
                        color="#000000",
                        feathering=values["transform_feather"],
                    ),
                )[0]
                rgb = pixels[..., :3] if source.channels != 1 else pixels.expand(-1, -1, -1, 3)
                pixels = torch.cat((rgb, alpha), dim=-1)
        elif operation == "rotate":
            pixels = ops.rotate(None, pixels, values["transform_rotation"])[0]
        else:
            pixels = ops.flip(
                None,
                pixels,
                "y-axis: horizontally"
                if values["transform_flip"] == "horizontal"
                else "x-axis: vertically",
            )[0]
    check_cancelled()
    if native.source_signatures() != request.native_signatures:
        raise ValueError(
            "The native geometry source changed during processing. Submit the image again."
        )
    channels = (
        4
        if operation == "pad" and values["transform_padding_style"] == "transparent"
        else source.channels
    )
    if (
        not isinstance(pixels, torch.Tensor)
        or tuple(pixels.shape) != (1, height, width, channels)
        or not torch.isfinite(pixels).all()
    ):
        raise ValueError("Native transform returned invalid dimensions or non-finite pixels.")
    # Include synchronization/host transfer in the attempt so a device failure
    # can restart from the immutable source under the shared CPU fallback policy.
    return pixels[0].detach().cpu().numpy().copy()


def process(owner, data, manager, check_cancelled):
    import numpy as np
    from PIL import Image
    from ruinedfooocus.features.image_processing.native_image_formats import advanced_output
    from ruinedfooocus.features.image_processing.utility_device import run_utility

    request = data.get(KEY)
    data.pop("image_utility_source", None)
    if (
        not isinstance(request, TransformRequest)
        or data.get("controlnet", {}).get("type") != KINDS[0]
    ):
        raise ValueError("Capture the transform source before processing. Submit it again.")
    data.update(dict(request.output_settings))
    execution = run_utility(
        KINDS[0],
        lambda policy: _process_pixels(request, policy, check_cancelled),
        check_cancelled=check_cancelled,
    )
    result = execution.value
    source, values = request.source, dict(request.parameters)
    width, height = request.output_size
    node = {
        "resize": "ResizeImageMaskNode",
        "crop": "ImageCropV2",
        "crop-to-mask": "ImageCropV2",
        "pad": "PadImageNode",
        "rotate": "ImageRotate",
        "flip": "ImageFlip",
    }[values["transform_operation"]]
    information = dict(
        operation=KINDS[0],
        native_node=node,
        native_revision=NATIVE_REVISION,
        source_sha256=source.source_sha256,
        input_mode=source.input_mode,
        source_dtype=source.dtype,
        color_space=source.color_space,
        has_icc_profile=source.has_icc_profile,
        width=width,
        height=height,
        source_width=source.size[0],
        source_height=source.size[1],
        alpha="native geometry on alpha; transparent pad uses native zero-filled alpha",
        compute=execution.metadata(),
        **values,
    )
    if request.selection is not None:
        information.update(
            selection_sha256=request.selection.source_sha256,
            selection_policy="nonzero grayscale bounds; crop all source channels without masking or resizing",
            crop_region=dict(zip(("x", "y", "width", "height"), request.crop_region)),
            native_nodes=["PrimitiveBoundingBox", "ImageCropV2"],
        )
    check_cancelled()
    data["image_utility_source"] = information
    if advanced_output(data):
        return [result[..., 0] if result.shape[-1] == 1 else result]
    byte = (np.clip(result, 0, 1) * 255).round().astype(np.uint8)
    return [Image.fromarray(byte[..., 0] if byte.shape[-1] == 1 else byte)]


def output_metadata(data):
    from copy import deepcopy

    request = data[KEY]
    return dict(
        software="RuinedFooocus",
        prompt="",
        negative="",
        controlnet={"type": KINDS[0]},
        width=request.output_size[0],
        height=request.output_size[1],
        image_utility_source=deepcopy(data["image_utility_source"]),
        **dict(request.parameters),
        **dict(request.output_settings),
    )


def replay_values(metadata):
    from ruinedfooocus.features.image_processing.native_image_formats import format_settings

    return dict(
        **parameters(metadata),
        **format_settings(metadata),
        input_image=None,
        image_source=None,
        mask_image=None,
        inpaint_toggle=False,
        cn_selection=LABELS[KINDS[0]],
        prompt="",
        negative="",
        automatic_upscale="Off",
        automatic_upscale_target="Generated image",
        model_recipe_path="",
        reference_adapter="none",
    )
