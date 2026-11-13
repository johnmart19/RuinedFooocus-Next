"""Image-only preparation shared by the inpaint editor and pipeline."""

from PIL import Image, ImageOps
import numpy as np


OUTPAINT_MARGINS = ("outpaint_left", "outpaint_top", "outpaint_right", "outpaint_bottom")
OUTPAINT_MAX_MARGIN = 1024
OUTPAINT_MAX_SIDE = 4096
OUTPAINT_MAX_FEATHER = 128
OUTPAINT_DEFAULT_FEATHER = 32


def inpaint_supported(family):
    from ruinedfooocus.features.models.image_components import ALIASES

    return family in ("FluxInpaint", "QwenImage21") or ALIASES.get(family, family) == "SDXL"


def outpaint_margins(data):
    margins = tuple(data.get(key, 0) for key in OUTPAINT_MARGINS)
    if any(
        type(value) not in (int, float)
        or not 0 <= value <= OUTPAINT_MAX_MARGIN
        or int(value) != value
        for value in margins
    ):
        raise ValueError("Canvas extension margins must be whole pixels from 0 to 1024.")
    return tuple(map(int, margins))


def outpaint_size(source_size, margins):
    left, top, right, bottom = margins
    width, height = source_size[0] + left + right, source_size[1] + top + bottom
    if max(width, height) > OUTPAINT_MAX_SIDE:
        raise ValueError(
            f"The extended canvas must be at most {OUTPAINT_MAX_SIDE} pixels on each side."
        )
    return width, height


def prepare_outpaint(source, data):
    """Create the native image and border mask, then align them for VAE encoding."""
    import torch
    from torch.nn.functional import pad
    from nodes import ImagePadForOutpaint
    from ruinedfooocus.features.image_editing.image_geometry import canvas_plan

    left, top, right, bottom = outpaint_margins(data)
    feather = data.get("outpaint_feather", OUTPAINT_DEFAULT_FEATHER)
    if (
        type(feather) not in (int, float)
        or not 0 <= feather <= OUTPAINT_MAX_FEATHER
        or int(feather) != feather
    ):
        raise ValueError("Canvas edge blending must be a whole number from 0 to 128 pixels.")
    # The native node ignores feathering when it reaches half either dimension.
    feather = min(int(feather), (min(source.size) - 1) // 2)
    width, height = outpaint_size(source.size, (left, top, right, bottom))
    plan = canvas_plan((width, height))
    plan.update(
        mode="outpaint",
        source_width=source.width,
        source_height=source.height,
        content_left=left,
        content_top=top,
        content_right=right,
        content_bottom=bottom,
        feather=feather,
    )
    image = torch.from_numpy(np.array(source)).float().unsqueeze(0) / 255.0
    pixels, mask = ImagePadForOutpaint().expand_image(image, left, top, right, bottom, feather)
    x, y = plan["left"], plan["top"]
    borders = (x, plan["canvas_width"] - width - x, y, plan["canvas_height"] - height - y)
    return (
        pad(pixels.movedim(-1, 1), borders, mode="replicate").movedim(1, -1),
        pad(mask[:, None], borders, mode="replicate")[:, 0],
        plan,
    )


def blend_outpaint(decoded, pixels, mask, plan):
    """Keep the source pixels outside the native edge blend after VAE decoding."""
    from ruinedfooocus.features.image_editing.image_geometry import restore_reference_frame

    decoded = decoded.float()
    original = restore_reference_frame(pixels, plan).to(decoded)
    blend = restore_reference_frame(mask.unsqueeze(-1), plan).to(decoded)
    return original.lerp(decoded, blend)


def load_source(value, *, preserve_alpha=False):
    if value is None:
        raise ValueError("Select a source image before enabling inpainting.")
    try:
        if isinstance(value, Image.Image):
            return ImageOps.exif_transpose(value).convert("RGBA" if preserve_alpha else "RGB")
        with Image.open(value) as source:
            return ImageOps.exif_transpose(source).convert("RGBA" if preserve_alpha else "RGB")
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("Select a readable source image for inpainting.") from error


def editor_payload(value, *, preserve_alpha=False):
    image = load_source(value, preserve_alpha=preserve_alpha)
    if image.height > 600:
        image = image.resize(
            (max(1, round(image.width * 600 / image.height)), 600), Image.Resampling.LANCZOS
        )
    return {"background": image, "layers": [Image.new("RGBA", image.size)], "composite": None}


def source_and_mask(value, editor, *, preserve_alpha=False):
    source = load_source(value, preserve_alpha=preserve_alpha)
    layers = editor.get("layers") if isinstance(editor, dict) else None
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError("Paint a mask before running inpainting.")
    layer = layers[0]
    if isinstance(layer, Image.Image):
        if layer.mode != "RGBA":
            raise ValueError("The inpaint mask must have an alpha channel.")
    else:
        pixels = np.asarray(layer)
        if pixels.ndim != 3 or pixels.shape[2] != 4 or pixels.dtype != np.uint8:
            raise ValueError("The inpaint mask must be an RGBA image.")
        layer = Image.fromarray(pixels)
    mask = layer.getchannel("A")
    if mask.getbbox() is None:
        raise ValueError("Paint a mask before running inpainting.")
    resample = Image.Resampling.BILINEAR if preserve_alpha else Image.Resampling.LANCZOS
    return source, mask.resize(source.size, resample)


def prepare_inpaint(source, mask, target_size, mode="preserve"):
    """Use the same content transform for source and mask; keep padding unpainted."""
    from ruinedfooocus.features.image_editing.image_geometry import prepare_reference

    pixels, plan = prepare_reference(source, target_size, mode)
    return pixels, prepare_mask(mask, plan), plan


def prepare_mask(mask, plan, resample=Image.Resampling.LANCZOS):
    """Map a mask into an existing image frame with zero coverage in padding."""
    import torch

    mode = plan["mode"]
    size = plan["width"], plan["height"]
    if mode == "pad":
        content_size = plan["content_width"], plan["content_height"]
        left, top = plan["content_left"], plan["content_top"]
        mask = ImageOps.expand(
            mask.resize(content_size, resample),
            (left, top, size[0] - content_size[0] - left, size[1] - content_size[1] - top),
            fill=0,
        )
    mask = ImageOps.fit(mask, size, resample) if mode == "crop" else mask.resize(size, resample)
    mask = ImageOps.expand(
        mask,
        (
            plan["left"],
            plan["top"],
            plan["canvas_width"] - size[0] - plan["left"],
            plan["canvas_height"] - size[1] - plan["top"],
        ),
        fill=0,
    )
    return torch.from_numpy(np.asarray(mask).copy()).float().unsqueeze(0) / 255
