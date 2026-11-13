"""Small operation dispatcher for generation-independent still-image utilities."""

from copy import deepcopy

from ruinedfooocus.features.image_processing import (
    depth_utilities,
    image_filters,
    image_shaders,
    image_geometry_tools,
    image_mask_tools,
    image_color_tools,
    wevisone,
)
from ruinedfooocus.features.image_processing.depth_utilities import kind_from

ADJUSTMENTS = (image_geometry_tools, image_mask_tools, image_color_tools)
ADJUSTMENT_KINDS = tuple(kind for owner in ADJUSTMENTS for kind in owner.KINDS)
ADJUSTMENT_DEFAULTS = {key: value for owner in ADJUSTMENTS for key, value in owner.DEFAULTS.items()}
ADJUSTMENT_IMAGES = tuple(
    dict.fromkeys(field for owner in ADJUSTMENTS for field in getattr(owner, "IMAGE_FIELDS", ()))
)
ADJUSTMENT_LABEL = "Adjust image"
OWNERS = (depth_utilities, image_filters, image_shaders, wevisone, *ADJUSTMENTS)
KINDS = tuple(kind for owner in OWNERS for kind in owner.KINDS)
FIELDS = tuple(field for owner in OWNERS for field in owner.FIELDS)
RGB8_KINDS = (*image_filters.KINDS, *image_shaders.KINDS, *wevisone.KINDS)


def is_utility(kind):
    return str(kind).lower() in KINDS


def adjustment_kind(selection, control_type):
    if selection != ADJUSTMENT_LABEL:
        return None
    kind = str(control_type or "").lower()
    return kind if kind in ADJUSTMENT_KINDS else "image-transform"


def read_adjustment_source(value):
    """File widgets retain original samples; display pixels are never inputs."""
    from pathlib import Path

    if not isinstance(value, (str, Path)):
        return value
    from PIL import Image, ImageOps
    from ruinedfooocus.features.image_processing.native_image_formats import (
        MAX_BYTES,
        MAX_PIXELS,
        STILL_SUFFIXES,
        needs_preview,
        read_image,
    )

    path = Path(value)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("Choose a bounded existing still image.")
    if path.suffix.lower() in STILL_SUFFIXES and needs_preview(path):
        return read_image(path)
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1 or not 0 < image.width * image.height <= MAX_PIXELS:
            raise ValueError("Choose one still image of at most 16 megapixels.")
        return ImageOps.exif_transpose(image).convert("RGBA").copy()


def active_adjustment_images(kind, data):
    if kind == "image-transform":
        return ("mask_image",) if data.get("transform_operation") == "crop-to-mask" else ()
    if kind == "image-mask":
        operation = data.get("mask_operation", "grow")
        return (
            ()
            if operation in ("channel", "color")
            else ("mask_image", "mask_reference")
            if operation == "composite"
            else ("mask_image",)
        )
    if kind == "image-color":
        return (
            ("color_reference",)
            if data.get("color_operation", "blend") in ("blend", "transfer")
            else ()
        )
    return ()


def uses_owned_selection(kind, data):
    return "mask_image" in active_adjustment_images(kind, data)


def _handler(kind):
    for owner in OWNERS:
        if kind in owner.KINDS:
            return owner
    raise ValueError("Choose a supported standalone still-image operation.")


def capture_assets(data, manager, models=None):
    kind = kind_from(data)
    if kind in ADJUSTMENT_KINDS:
        if data.get("cn_selection") == ADJUSTMENT_LABEL and kind != adjustment_kind(
            data["cn_selection"], data.get("cn_type")
        ):
            raise ValueError("The adjustment group changed before capture. Submit it again.")
        handler = _handler(kind)
        data = dict(data)
        if data.get("cn_selection") == ADJUSTMENT_LABEL:
            active = active_adjustment_images(kind, data)
            for field in ADJUSTMENT_IMAGES:
                if field not in active:
                    data.pop(field, None)
        for field in ("input_image", *getattr(handler, "IMAGE_FIELDS", ())):
            if data.get(field) is not None:
                data[field] = read_adjustment_source(data[field])
    if data.get("output_format") == "avif" and kind_from(data) not in ADJUSTMENT_KINDS:
        raise ValueError(
            "Source-preserving image tools return RGBA. Choose PNG or EXR; native still AVIF cannot preserve alpha."
        )
    return _handler(kind_from(data)).capture_assets(data, manager, models)


def worker_data(data):
    return _handler(kind_from(data)).worker_data(data)


def clear_model(owner):
    if (
        getattr(owner, "_lotus_depth", None) is not None
        or getattr(owner, "_depth_utility_cache", None) is not None
    ):
        depth_utilities.clear_model(owner)


def process(owner, data, manager, check_cancelled):
    handler = _handler(kind_from(data))
    if handler not in (depth_utilities, wevisone):
        clear_model(owner)
    return handler.process(owner, data, manager, check_cancelled)


def output_metadata(data):
    return _handler(kind_from(data)).output_metadata(data)


def replay_values(data):
    resets = {
        **deepcopy(image_filters.DEFAULTS),
        **deepcopy(image_shaders.DEFAULTS),
        **depth_utilities.REPLAY_RESETS,
        **deepcopy(wevisone.DEFAULTS),
        "wevisone_model": None,
        **deepcopy(ADJUSTMENT_DEFAULTS),
        **dict.fromkeys(ADJUSTMENT_IMAGES),
    }
    kind = kind_from(data)
    result = {**resets, **_handler(kind).replay_values(data)}
    if kind in ADJUSTMENT_KINDS:
        result.update(
            cn_selection=ADJUSTMENT_LABEL, cn_type=kind.capitalize(), _native_main_source=None
        )
        result.pop("prompt", None)
        result.pop("negative", None)
    return result
