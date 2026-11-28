"""A bounded saved-latent operation with no checkpoint or generation request."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re

from ruinedfooocus.features.image_processing import latent_composition_native as native

KIND = "latent-compose"
KEY = "_latent_composition_capture"
INPUTS = ("latent_composition_primary", "latent_composition_secondary")
MASK = "latent_composition_mask"
DEFAULTS = dict(
    latent_composition_mode="replace",
    latent_composition_x=0,
    latent_composition_y=0,
    latent_composition_feather=0,
)
FIELDS = tuple(DEFAULTS)
MAX_MASK_PIXELS = 16 * 1024**2
MAX_INLINE_CHARS = 2 * (4 * ((32 * 1024**2 + 2) // 3))


def handles(kind):
    return kind == KIND


def ui_request(data):
    """Inactive controls retain their form values without entering this operation."""
    result = dict(data)
    if result.get("latent_composition_mode", "replace") == "masked":
        result["latent_composition_feather"] = 0
    else:
        result.pop(MASK, None)
    return result


def omit_inactive(data):
    fields = {*INPUTS, MASK, *FIELDS, KEY, "native_latent_composition_source"}
    return {key: value for key, value in data.items() if key not in fields}


def settings(data, *, operation=False):
    values = {key: data.get(key, default) for key, default in DEFAULTS.items()}
    if values["latent_composition_mode"] not in ("replace", "masked"):
        raise ValueError("Choose native latent replacement or masked replacement.")
    for key in FIELDS[1:]:
        value = values[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or int(value) != value
            or not 0 <= value <= 16384
        ):
            raise ValueError(
                "Latent coordinates and feathering must be whole pixels from zero through 16384."
            )
        values[key] = int(value)
    if (
        operation
        and values["latent_composition_mode"] == "masked"
        and values["latent_composition_feather"]
    ):
        raise ValueError("Masked latent composition uses its mask, without edge feathering.")
    return values


def request_values(payload):
    control = payload.get("controlnet") or {}
    if not isinstance(control, dict):
        raise ValueError("Image control settings must be a mapping.")
    if not handles(control.get("type")):
        if any(payload.get(key) is not None for key in (*INPUTS, MASK)):
            raise ValueError("Saved latent composition inputs require latent-compose.")
        return None
    if set(control) != {"type"}:
        raise ValueError("Latent composition control settings accept only their type.")
    if payload.get("n", 1) != 1:
        raise ValueError("Latent composition returns exactly one saved latent.")
    if any(not isinstance(payload.get(key), str) or not payload[key] for key in INPUTS):
        raise ValueError("Provide both saved latent files as bounded base64 data.")
    if sum(len(payload[key]) for key in INPUTS) > MAX_INLINE_CHARS:
        raise ValueError("The two encoded latent inputs exceed their combined bound.")
    if any(
        payload.get(key) is not None
        for key in ("input_image", "mask", "borders", "reference_adapter_image")
    ) or payload.get("reference_images"):
        raise ValueError(
            "Latent composition uses only its two saved files and optional selection mask."
        )
    if payload.get("automatic_upscale", "Off") != "Off":
        raise ValueError("Latent composition preserves its saved canvas without pixel upscaling.")
    if (
        payload.get(MASK) is not None
        and payload.get("latent_composition_mode", "replace") != "masked"
    ):
        raise ValueError("An optional selection mask requires masked latent composition.")
    if any(
        payload.get(key) is not None
        for key in ("native_latent_input", "native_positive_input", "native_negative_input")
    ) or any(
        payload.get(key) is True
        for key in ("native_export_latent", "native_export_conditioning", "native_export_handoff")
    ):
        raise ValueError("Choose standalone latent composition without generation tensor inputs.")
    return settings(payload, operation=True)


@dataclass(frozen=True)
class Mask:
    size: tuple
    pixels: bytes
    sha256: str


@dataclass(frozen=True)
class Composition:
    parameters: tuple
    scale: int
    pixel_size: tuple
    shape: tuple
    mask: Mask | None
    native_sources: tuple


def geometry(primary, values):
    layout = dict(primary.context).get("layout", {})
    scale = layout.get("scale")
    shape = primary.shape
    if scale not in (8, 16) or len(shape) != 4 or shape[0] != 1:
        raise ValueError("Compose matching rank-four still latents at their original codec scale.")
    width, height = shape[-1] * scale, shape[-2] * scale
    x, y, feather = (values[key] for key in FIELDS[1:])
    if any(value % scale for value in (x, y, feather)):
        raise ValueError("Latent coordinates and feathering must align to the original codec grid.")
    if x >= width or y >= height:
        raise ValueError("Place the source inside the saved destination canvas.")
    if feather > min(width - x, height - y):
        raise ValueError("Latent feathering must fit the visible overlay region.")
    return scale, (width, height)


def capture(data, manager):
    from PIL import Image, ImageOps
    from ruinedfooocus.features.generation import native_tensor_io as tensors

    values = settings(data, operation=True)
    if values["latent_composition_mode"] != "masked" and data.get(MASK) is not None:
        raise ValueError("An optional selection mask requires masked latent composition.")
    # Only lifecycle values enter E's owner-private bytes-only capture. Generation
    # prompts, model choices, paths and settings never enter this operation.
    lifecycle = {
        key: data[key]
        for key in ("_api_job", "_job", "task_id", "task_type", "index", "silent")
        if key in data
    }
    result = tensors.capture_composition_inputs(
        data.get(INPUTS[0]), data.get(INPUTS[1]), manager, data=lifecycle
    )
    primary, secondary = result[tensors.KEY].inputs
    tensors.assert_composable(primary, secondary)
    scale, size = geometry(primary, values)
    mask = None
    if data.get(MASK) is not None:
        image = data[MASK]
        if not isinstance(image, Image.Image) or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Choose one still selection mask for the saved latent canvas.")
        if image.width * image.height > MAX_MASK_PIXELS:
            raise ValueError("The latent selection mask may contain at most 16 megapixels.")
        image = ImageOps.exif_transpose(image).convert("L").copy()
        if image.size != size:
            raise ValueError("The latent mask must match the complete saved image canvas.")
        pixels = image.tobytes()
        mask = Mask(image.size, pixels, hashlib.sha256(pixels).hexdigest())
    result[KEY] = Composition(
        tuple(values.items()), scale, size, primary.shape, mask, native.source_signatures()
    )
    result.update(values)
    result["controlnet"] = dict(type=KIND)
    return result


def captured(data):
    value = data.get(KEY)
    if not isinstance(value, Composition):
        raise ValueError("Capture both original saved latent files before composition.")
    return value


def process(data, check_cancelled):
    import numpy as np
    import torch
    from ruinedfooocus.features.generation import native_tensor_io as tensors

    packet = captured(data)
    primary, secondary = data[tensors.KEY].inputs
    tensors.assert_composable(primary, secondary)
    values = dict(packet.parameters)
    if (
        geometry(primary, values) != (packet.scale, packet.pixel_size)
        or primary.shape != packet.shape
    ):
        raise ValueError("The captured latent layout changed before composition.")
    check_cancelled()
    operations = native.operations(packet.native_sources)
    # Both E inputs and the actual native formulas use CPU F32 working samples.
    # Original uploaded bytes and dtype remain in the source provenance.
    with torch.device("cpu"), torch.inference_mode():
        destination = tensors.load_composition_latent(data, primary)
        check_cancelled()
        source = tensors.load_composition_latent(data, secondary)
        check_cancelled()
        x, y = (values[key] // packet.scale * 8 for key in FIELDS[1:3])
        if values["latent_composition_mode"] == "masked":
            mask = None
            if packet.mask is not None:
                mask = torch.from_numpy(
                    np.frombuffer(packet.mask.pixels, dtype=np.uint8)
                    .copy()
                    .reshape(packet.mask.size[1], packet.mask.size[0])
                    .astype(np.float32)
                    / 255
                )[None]
            result = operations.masked(None, destination, source, x, y, False, mask)[0]
            node = "LatentCompositeMasked"
        else:
            feather = values["latent_composition_feather"] // packet.scale * 8
            result = operations.plain(None, destination, source, x, y, feather=feather)[0]
            node = "LatentComposite"
        check_cancelled()
        metadata = dict(
            native_node=node,
            parameters=values,
            x_pixels=values["latent_composition_x"],
            y_pixels=values["latent_composition_y"],
            feather_pixels=values["latent_composition_feather"],
            codec_scale=packet.scale,
            native_x=x,
            native_y=y,
            destination_pixels=list(packet.pixel_size),
            preserved_shape=list(packet.shape),
            resize_source=False,
            framing="destination canvas retained; native offset clips the overlay to its visible region",
            mask_sha256=packet.mask.sha256 if packet.mask is not None else None,
            mask_mapping="native bilinear canvas-to-latent-grid"
            if packet.mask is not None
            else None,
        )
        exported = tensors.save_composed_latent(data, primary, result, composition=metadata)
    check_cancelled()
    data["native_latent_composition_source"].update(metadata)
    return tensors.TensorGenerationArtifacts([], (exported,))


def validate_result(data, result):
    from ruinedfooocus.features.generation import native_tensor_io as tensors
    from ruinedfooocus.features.models.model_identity import file_signature

    captured(data)
    plan = data.get(tensors.KEY)
    files = getattr(result, "tensor_files", ())
    if (
        not isinstance(plan, tensors.TensorPlan)
        or not isinstance(result, tensors.TensorGenerationArtifacts)
        or len(result) != 0
        or len(files) != 1
        or tuple(data.get(tensors.EXPORT_KEY, ())) != files
    ):
        raise RuntimeError("Latent composition did not complete exactly one owned saved tensor.")
    artifact = files[0]
    if (
        not isinstance(artifact, tensors.TensorExport)
        or artifact.kind != "latent-composite"
        or artifact.family != plan.family
        or Path(artifact.path).resolve().parent != plan.directory.resolve()
        or not re.fullmatch(r"[0-9a-f]{64}", artifact.sha256)
        or tuple(file_signature(artifact.path)) != artifact.signature
    ):
        raise RuntimeError("The saved composition artifact changed or lost its request ownership.")
    return result


def replay_values(metadata, data):
    source = metadata.get("native_latent_composition_source") or metadata.get("layout", {}).get(
        "composition"
    )
    if not isinstance(source, dict):
        raise ValueError("This file has no saved latent composition source.")
    from ruinedfooocus.features.generation.native_tensor_io import _raw

    for field, digest in zip(
        INPUTS, (source.get("primary_sha256"), source.get("secondary_sha256"))
    ):
        if data.get(field) is None or hashlib.sha256(_raw(data[field])).hexdigest() != digest:
            raise ValueError(
                "Re-upload both original saved latent inputs to replay this composition."
            )
    if source.get("mask_sha256"):
        from PIL import Image, ImageOps

        mask = data.get(MASK)
        if (
            not isinstance(mask, Image.Image)
            or hashlib.sha256(ImageOps.exif_transpose(mask).convert("L").tobytes()).hexdigest()
            != source["mask_sha256"]
        ):
            raise ValueError("Re-upload the original selection mask to replay this composition.")
    values = source.get("parameters", metadata)
    return {"controlnet": dict(type=KIND), **settings(values, operation=True)}
