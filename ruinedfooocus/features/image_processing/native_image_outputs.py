"""Request-local original/display transport for precision generation results."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class NativeImageOutput:
    original: str
    display: str
    signature: tuple = field(init=False)

    def __post_init__(self):
        from ruinedfooocus.features.models.model_identity import file_signature

        object.__setattr__(self, "signature", tuple(file_signature(self.original)))


def output_role(metadata):
    """Classify terminal semantics for the Latest shortcut; never hide/delete outputs."""
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (TypeError, ValueError):
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    data = {str(key).lower(): value for key, value in metadata.items()}
    utility = data.get("image_utility_source") or {}
    if not isinstance(utility, dict):
        utility = {}
    operation = utility.get("operation", "") if isinstance(utility, dict) else ""
    control = data.get("controlnet") or {}
    kind = control.get("type", "") if isinstance(control, dict) else ""
    if operation == "wevisone-matte":
        return (
            "image"
            if utility.get("output") == "cutout"
            else "mask"
            if utility.get("output") == "mask"
            else "guide"
        )
    if operation == "wevisone-detect":
        return "guide"
    if str(data.get("data", "")).lower() in ("preview grid", "preview", "placeholder"):
        return "preview"
    if operation in ("moge-depth", "lotus-depth", "pose-map", "marigold-albedo") or kind in (
        "depth-map",
        "detect",
        "normal-map",
        "pose-map",
        "marigold-albedo",
    ):
        return "guide"
    if kind == "segment" and data.get("segment_output", "cutout") == "cutout":
        return "image"
    if operation == "face-landmarks" and utility.get("mask_role") != "inpaint-selection":
        return "guide"
    if (
        operation in ("image-mask", "face-landmarks")
        or utility.get("mask_role")
        or kind in ("segment", "segmentation", "mask")
    ):
        return "mask"
    if data.get("native_vision_source"):
        return "guide"
    if data.get("output_role") in ("mask", "guide", "helper", "preview", "placeholder"):
        return data["output_role"]
    return "image"


def latest_generated_result(records, index=None):
    """A selected completed final image keeps its exact batch and original identity."""
    from ruinedfooocus.features.gallery.actions import source_path

    if index is None:
        index = next(
            (index for index, record in enumerate(records) if record.get("output_role") == "image"),
            -1,
        )
    if type(index) is not int or not 0 <= index < len(records):
        return None
    record = records[index]
    if record.get("output_role") != "image":
        return None
    source_path(record)
    return dict(record, batch_index=index)


def result_presentation(product):
    from ruinedfooocus.features.gallery.actions import source_record
    from ruinedfooocus.features.image_processing.native_image_formats import image_metadata

    displays, downloads, records = [], [], []
    metadata = {"Data": "Preview Grid"}
    for value in product:
        precision = isinstance(value, NativeImageOutput)
        original = value.original if precision else str(value)
        display = value.display if precision else str(value)
        record = dict(source_record(original), display=display, precision=precision)
        if precision:
            if tuple(record["signature"]) != value.signature:
                raise ValueError("The generated original changed before its result was displayed.")
            record["display_sha256"] = hashlib.sha256(Path(display).read_bytes()).hexdigest()
            downloads.append(original)
            information = image_metadata(original)
        else:
            with Image.open(original) as image:
                information = image.info
        record["output_role"] = output_role(information.get("parameters", {}))
        record["metadata"] = information.get("parameters", {})
        record["batch_index"] = len(records)
        if not records:
            metadata = information.get("parameters", metadata)
        records.append(record)
        displays.append(display)
    return displays, downloads, records, metadata


def selected_result(records, index):
    from ruinedfooocus.features.gallery.actions import source_path
    from ruinedfooocus.features.image_processing.native_image_formats import image_metadata

    if type(index) is not int or not 0 <= index < len(records):
        raise ValueError("The generation results changed. Select an image again.")
    record = records[index]
    original = source_path(record)
    if record.get("precision"):
        metadata = image_metadata(original).get("parameters", {"Data": "Image"})
    else:
        with Image.open(original) as image:
            metadata = image.info.get("parameters", {"Data": "Preview Grid"})
    return record["display"], metadata, record if record.get("precision") else None


def capture_original_source(data):
    """Read a selected original for a follow-up, never the cached display pixels."""
    record = data.get("_native_main_source")
    if not isinstance(record, dict) or not isinstance(data.get("main_view"), (str, Path)):
        return data
    candidate = Path(data["main_view"])
    if str(candidate) != record.get("display"):
        from ruinedfooocus.features.image_processing.native_image_formats import MAX_BYTES

        if (
            not candidate.is_file()
            or candidate.stat().st_size > MAX_BYTES
            or hashlib.sha256(candidate.read_bytes()).hexdigest() != record.get("display_sha256")
        ):
            return data
    from ruinedfooocus.features.gallery.actions import source_path
    from ruinedfooocus.features.image_processing.native_image_formats import (
        convert_color,
        read_image,
    )

    original = source_path(record)
    imported = read_image(original)
    source_path(record)
    from ruinedfooocus.features.image_processing.still_image_utilities import (
        ADJUSTMENT_KINDS,
        adjustment_kind,
    )

    kind = (data.get("controlnet") or {}).get("type")
    if kind is None:
        kind = adjustment_kind(data.get("cn_selection"), data.get("cn_type"))
    if kind in ADJUSTMENT_KINDS:
        return dict(
            data,
            main_view=imported,
            input_image=imported,
            image_source=dict(
                filename=Path(original).name,
                original_dtype=str(imported.pixels.dtype),
                original_color_space=imported.color_space,
                conversion="none",
            ),
        )
    pixels = convert_color(imported.pixels, imported.color_space, "sRGB")
    array = (pixels.clamp(0, 1).numpy() * 255).astype(np.uint8)
    working = Image.fromarray(array[..., 0] if array.shape[-1] == 1 else array)
    result = dict(data, main_view=working)
    result["image_source"] = {
        "filename": Path(original).name,
        "original_dtype": str(imported.pixels.dtype),
        "original_color_space": imported.color_space,
        "width": working.width,
        "height": working.height,
        "working_dtype": "uint8",
        "conversion": "native color conversion from original; image tools use an 8-bit working image",
    }
    return result
