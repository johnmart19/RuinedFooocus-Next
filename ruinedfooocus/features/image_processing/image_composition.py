"""Bounded, immutable still-image grids and native masked/alpha composition."""

from dataclasses import dataclass
import hashlib
import math

from ruinedfooocus.features.image_processing import composition_native as native

KINDS = ("image-grid", "image-composite")
KEY = "_image_composition_capture"
MAX_IMAGES = 32
MAX_PIXELS = 16 * 1024 * 1024
MAX_INPUT_PIXELS = 32 * 1024 * 1024
DEFAULTS = dict(
    grid_columns=2,
    grid_cell_width=512,
    grid_cell_height=512,
    grid_padding=8,
    composition_mode="SRC_OVER",
    composition_x=0,
    composition_y=0,
    composition_width=0,
    composition_height=0,
)
FIELDS = tuple(DEFAULTS)
IMAGE_FIELDS = ("composition_images", "composition_mask")
MAX_INLINE_CHARS = 56_000_000


def request_values(payload):
    """Validate the public operation before decoding or looking up model state."""
    control = payload.get("controlnet") or {}
    if not isinstance(control, dict):
        raise ValueError("Image control settings must be a mapping.")
    kind = control.get("type")
    uploads = payload.get("composition_images")
    if not handles(kind):
        if uploads is not None or payload.get("composition_mask") is not None:
            raise ValueError("Ordered image inputs require image-grid or image-composite.")
        return None
    if not isinstance(uploads, list) or not 1 <= len(uploads) <= MAX_IMAGES:
        raise ValueError("Add one through 32 inline ordered images.")
    if kind == "image-composite" and len(uploads) != 2:
        raise ValueError("Compose images uses exactly two ordered images: base, then overlay.")
    mask = payload.get("composition_mask")
    if kind == "image-grid" and mask is not None:
        raise ValueError("An image grid does not use a composition mask.")
    inline = [*uploads, *([mask] if mask is not None else [])]
    if any(not isinstance(value, str) or not value for value in inline):
        raise ValueError("Provide inline images for the ordered inputs and mask.")
    if sum(map(len, inline)) > MAX_INLINE_CHARS:
        raise ValueError("The combined inline image inputs exceed the 56 MB limit.")
    if payload.get("n", 1) != 1:
        raise ValueError("Image grids and composition return exactly one image.")
    for key in (
        "input_image",
        "mask",
        "borders",
        "reference_images",
        "reference_adapter_image",
        "reference_adapter_model",
        "reference_adapter_vision",
        "reference_adapter_strength",
        "model_recipe_path",
        "native_vision_model",
        "rembg_native_model",
        "qwen_union_guide",
        "qwen_union_source",
        "qwen_union_mask",
    ):
        if payload.get(key) is not None and payload[key] != "" and payload[key] != []:
            raise ValueError(f"{key} is not used by image grids or composition.")
    if (
        payload.get("reference_adapter") not in (None, "", "none")
        or payload.get("clip_layer_source") == "recipe"
        or payload.get("native_edit_recipe", "auto") != "auto"
        or payload.get("automatic_upscale", "Off") != "Off"
    ):
        raise ValueError("Choose only the image grid or composition operation.")
    if any(key != "type" for key in control):
        raise ValueError("Image grid and composition controlnet accepts only its type.")
    return settings(payload)


def handles(kind):
    return str(kind).lower() in KINDS


def _integer(value, label, minimum, maximum):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or int(value) != value
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{label} must be a whole number from {minimum} through {maximum}.")
    return int(value)


def settings(data):
    values = {key: data.get(key, default) for key, default in DEFAULTS.items()}
    for key, low, high in (
        ("grid_columns", 1, 20),
        ("grid_cell_width", 32, 2048),
        ("grid_cell_height", 32, 2048),
        ("grid_padding", 0, 50),
        ("composition_x", 0, 16384),
        ("composition_y", 0, 16384),
        ("composition_width", 0, 16384),
        ("composition_height", 0, 16384),
    ):
        values[key] = _integer(values[key], key, low, high)
    if values["composition_mode"] not in ("masked", *native.modes()):
        raise ValueError("Choose a native composition mode.")
    return values


def replay_values(metadata, *, clear_images=True):
    values = settings(metadata)
    if clear_images:
        values.update(composition_images=[], composition_mask=None)
    return values


def reset_values():
    return dict(composition_images=[], composition_mask=None)


@dataclass(frozen=True)
class OwnedImage:
    size: tuple
    mode: str
    pixels: bytes
    sha256: str

    def image(self):
        from PIL import Image

        return Image.frombytes(self.mode, self.size, self.pixels)


def own_image(value, *, mask=False):
    from PIL import Image, ImageOps

    if (
        not isinstance(value, Image.Image)
        or getattr(value, "n_frames", 1) != 1
        or not 0 < value.width * value.height <= MAX_PIXELS
    ):
        raise ValueError("Choose a single still image of at most 16 megapixels.")
    image = ImageOps.exif_transpose(value).convert("L" if mask else "RGBA").copy()
    raw = image.tobytes()
    return OwnedImage(image.size, image.mode, raw, hashlib.sha256(raw).hexdigest())


def source_images(uploaded):
    """Validate Gallery/API pixels before copying or returning media to the UI."""
    from PIL import Image

    if not isinstance(uploaded, (list, tuple)) or not 1 <= len(uploaded) <= MAX_IMAGES:
        raise ValueError("Add one through 32 ordered images; composition needs exactly two.")
    values, total = [], 0
    for item in uploaded:
        value = item[0] if isinstance(item, (list, tuple)) and len(item) == 2 else item
        if not isinstance(value, Image.Image):
            raise ValueError("Ordered inputs must be uploaded still images.")
        if getattr(value, "n_frames", 1) != 1 or not 0 < value.width * value.height <= MAX_PIXELS:
            raise ValueError("Choose a single still image of at most 16 megapixels.")
        total += value.width * value.height
        if total > MAX_INPUT_PIXELS:
            raise ValueError("The ordered inputs may contain at most 32 megapixels in total.")
        values.append(value)
    return values


@dataclass(frozen=True)
class CapturedComposition:
    kind: str
    images: tuple
    mask: OwnedImage | None
    parameters: tuple
    output_size: tuple
    native_sources: tuple


def capture(data, kind):
    """No checkpoint, prompt, model, path or inference work enters this capture."""
    kind = str(kind).lower()
    if not handles(kind):
        raise ValueError("Choose Make image grid or Compose images.")
    from ruinedfooocus.features.gallery.editor_session import KEY as EDITOR_KEY

    if EDITOR_KEY in data:
        data = bind_editor_inputs(data)
    values = settings(data)
    uploaded = data.get("composition_images") or []
    from PIL import Image

    uploads = source_images(uploaded)
    total = sum(image.width * image.height for image in uploads)
    images = tuple(own_image(value) for value in uploads)
    mask_value = data.get("composition_mask")
    if (
        isinstance(mask_value, Image.Image)
        and total + mask_value.width * mask_value.height > MAX_INPUT_PIXELS
    ):
        raise ValueError("The ordered inputs and mask may contain at most 32 megapixels in total.")
    mask = own_image(mask_value, mask=True) if mask_value is not None else None
    if kind == "image-grid":
        if mask is not None:
            raise ValueError("An image grid does not use a composition mask.")
        columns = values["grid_columns"]
        rows = math.ceil(len(images) / columns)
        size = (
            columns * values["grid_cell_width"] + (columns - 1) * values["grid_padding"],
            rows * values["grid_cell_height"] + (rows - 1) * values["grid_padding"],
        )
    else:
        if len(images) != 2:
            raise ValueError("Compose images uses exactly two ordered images: base, then overlay.")
        size = images[0].size
        x, y = values["composition_x"], values["composition_y"]
        if x >= size[0] or y >= size[1]:
            raise ValueError("The overlay must start inside the base image.")
        width = values["composition_width"] or size[0] - x
        height = values["composition_height"] or size[1] - y
        if x + width > size[0] or y + height > size[1]:
            raise ValueError(
                "The overlay rectangle must stay inside the base; no framing is cropped."
            )
        values.update(composition_width=width, composition_height=height)
        if mask is not None and mask.size != images[1].size:
            raise ValueError("The composition mask must match the oriented overlay image.")
    if size[0] * size[1] > MAX_PIXELS:
        raise ValueError("The composed output may contain at most 16 megapixels.")
    if data.get("output_format") == "avif":
        if kind == "image-composite":
            raise ValueError(
                "RGBA composition needs PNG or EXR output; native still AVIF has no alpha."
            )
        if min(size) < 64:
            raise ValueError("Native still AVIF needs dimensions of at least 64 pixels per side.")
    result = dict(data)
    result.update(values)
    result[KEY] = CapturedComposition(
        kind, images, mask, tuple(values.items()), size, native.source_signatures()
    )
    result.pop("composition_images", None)
    result.pop("composition_mask", None)
    return worker_data(result)


def captured(data):
    value = data.get(KEY)
    if not isinstance(value, CapturedComposition) or not handles(value.kind):
        raise ValueError("Capture the ordered image inputs before composing them.")
    return value


def worker_data(data):
    from ruinedfooocus.features.generation.image_preparation import KEY as ASSET_KEY
    from ruinedfooocus.features.gallery.editor_session import KEY as EDITOR_KEY

    packet = captured(data)
    # Only operation-owned pixels, transport and lifecycle cross this boundary.
    # Unknown generation state must not become utility metadata or preparation.
    result = {
        key: data[key]
        for key in (
            KEY,
            ASSET_KEY,
            EDITOR_KEY,
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
        if key in data
    }
    width, height = packet.output_size
    result.update(dict(packet.parameters))
    result.update(
        controlnet=dict(type=packet.kind),
        input_image=None,
        prompt="",
        negative="",
        loras=[],
        style_selection=[],
        base_model_name="",
        base_model_hash="",
        custom_steps=1,
        cfg=0,
        sampler_name="",
        scheduler="",
        clip_skip=1,
        image_number=1,
        generate_forever=False,
        seed=0,
        aspect_ratios_selection="Custom...",
        custom_width=width,
        custom_height=height,
        width=width,
        height=height,
        automatic_upscale="Off",
        automatic_upscale_target="Generated image",
        _sampling_captured=False,
    )
    return result


def bind_editor_inputs(data, *, owner=None, store=None):
    """Use validated owned artifacts for composition, with explicit working conversion."""
    from ruinedfooocus.features.gallery.actions import source_record
    from ruinedfooocus.features.gallery.editing_workflow import working_source
    from ruinedfooocus.features.gallery.editor_session import validate_request

    ticket = validate_request(data, owner=owner, store=store)
    if ticket is None:
        return data
    step = data.get("_image_workflow_step")
    if ticket.operation == "compose":
        from ruinedfooocus.features.image_editing.image_workflows import (
            WorkflowStep,
            composition_inputs,
        )

        if not isinstance(step, WorkflowStep) or step.ticket != ticket or step.action != "compose":
            raise ValueError("The composition has no matching owned workflow step.")
        if store is None:
            from ruinedfooocus.features.gallery.editor_session import default_store

            store = default_store()
        images = composition_inputs(step, store, owner=owner)
    else:
        images = [
            working_source(source_record(ref.path))[0]
            for ref in (ticket.source, *ticket.additional_images)
        ]
    mask = None
    if ticket.mask is not None:
        from PIL import Image, ImageOps

        with Image.open(ticket.mask.path) as original:
            mask = ImageOps.exif_transpose(original).convert("L").copy()
        if ticket.mask.inverted:
            mask = ImageOps.invert(mask)
    validate_request(data, owner=owner, store=store)
    return dict(data, composition_images=images, composition_mask=mask)


def _fit(image, size, *, background, check_cancelled):
    from PIL import Image, ImageOps

    check_cancelled()
    fitted = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    left, top = (size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2
    canvas = Image.new(image.mode, size, background)
    canvas.paste(fitted, (left, top))
    return canvas, dict(
        source=list(image.size),
        scaled=list(fitted.size),
        left=left,
        top=top,
        width=size[0],
        height=size[1],
    )


def process(data, check_cancelled):
    import numpy as np
    from PIL import Image
    import torch
    from ruinedfooocus.features.image_processing.native_image_formats import advanced_output

    packet = captured(data)
    values = dict(packet.parameters)
    ops = native.operations(packet.native_sources, check_cancelled)
    check_cancelled()
    frames = []
    with torch.device("cpu"), torch.inference_mode():
        if packet.kind == "image-grid":
            cells = []
            size = values["grid_cell_width"], values["grid_cell_height"]
            for source in packet.images:
                image, frame = _fit(
                    source.image(), size, background=(0, 0, 0, 0), check_cancelled=check_cancelled
                )
                # Native ImageGrid is RGB. Composite source alpha onto its black
                # background explicitly; hidden RGB never becomes visible.
                flattened = Image.new("RGBA", size, (0, 0, 0, 255))
                flattened.alpha_composite(image)
                cells.append(ops.from_pil(flattened.convert("RGB")))
                frames.append(frame)
            output = ops.grid(None, cells, values["grid_columns"], *size, values["grid_padding"])[0]
            operation = "ImageGrid._group_process"
            alpha_policy = "opaque RGB on black"
        else:
            base = packet.images[0].image()
            overlay = packet.images[1].image()
            size = values["composition_width"], values["composition_height"]
            layer, frame = _fit(
                overlay, size, background=(0, 0, 0, 0), check_cancelled=check_cancelled
            )
            mask = (
                packet.mask.image()
                if packet.mask is not None
                else Image.new("L", overlay.size, 255)
            )
            # Source and mask use the identical contain/pad geometry.
            mask = mask.resize(tuple(frame["scaled"]), Image.Resampling.LANCZOS)
            fitted_mask = Image.new("L", size, 0)
            fitted_mask.paste(mask, (frame["left"], frame["top"]))
            destination = ops.from_pil(base)
            source = ops.from_pil(layer)
            selection = torch.from_numpy(np.array(fitted_mask, dtype=np.float32) / 255)[None]
            x, y = values["composition_x"], values["composition_y"]
            if values["composition_mode"] == "masked":
                output = ops.masked(None, destination, source, x, y, False, selection)[0][0]
                operation = "ImageCompositeMasked.execute"
            else:
                full = torch.zeros_like(destination)
                full_mask = torch.zeros_like(destination[..., 0])
                full[:, y : y + size[1], x : x + size[0]] = source
                full_mask[:, y : y + size[1], x : x + size[0]] = selection
                source_alpha = full[..., 3] * full_mask
                rgb, inverted = ops.porter(
                    None,
                    full[..., :3],
                    1 - source_alpha,
                    destination[..., :3],
                    1 - destination[..., 3],
                    values["composition_mode"],
                )
                output = torch.cat((rgb, (1 - inverted).unsqueeze(-1)), dim=-1)[0]
                operation = "PorterDuffImageComposite.execute"
            frames.append(frame)
            alpha_policy = "native RGBA; mask white selects overlay"
        check_cancelled()
        pixels = output.detach().cpu().float().numpy().copy()
    if (
        tuple(pixels.shape[:2]) != (packet.output_size[1], packet.output_size[0])
        or not np.isfinite(pixels).all()
        or np.min(pixels) < 0
        or np.max(pixels) > 1
    ):
        raise ValueError("The native image operation returned an invalid output frame.")
    data["native_composition_source"] = dict(
        kind=packet.kind,
        native=operation,
        alpha_policy=alpha_policy,
        fitting="uniform contain with padding; no crop",
        output_size=list(packet.output_size),
        inputs=[
            dict(
                index=index,
                size=list(image.size),
                sha256=image.sha256,
                working_dtype="uint8",
                mode=image.mode,
            )
            for index, image in enumerate(packet.images)
        ],
        mask_sha256=packet.mask.sha256 if packet.mask is not None else None,
        frames=frames,
    )
    check_cancelled()
    return (
        pixels
        if advanced_output(data)
        else Image.fromarray((pixels * 255).round().astype(np.uint8))
    )
