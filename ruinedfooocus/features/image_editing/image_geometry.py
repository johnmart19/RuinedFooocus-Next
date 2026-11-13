"""Preserve image proportions while fitting the model's working pixel budget."""

import math

FRAMING_CHOICES = [
    ("Preserve proportions", "preserve"),
    ("Fit selected size (pad)", "pad"),
    ("Crop to selected size", "crop"),
]


def canvas_plan(size, multiple=16):
    width, height = map(int, size)
    if min(width, height, multiple) < 1:
        raise ValueError("Image dimensions must be positive.")
    canvas_w = math.ceil(width / multiple) * multiple
    canvas_h = math.ceil(height / multiple) * multiple
    return dict(
        width=width,
        height=height,
        canvas_width=canvas_w,
        canvas_height=canvas_h,
        left=(canvas_w - width) // 2,
        top=(canvas_h - height) // 2,
    )


def reference_plan(source_size, target_size, mode="preserve", multiple=16):
    if mode not in ("preserve", "pad", "crop"):
        raise ValueError("Choose Preserve proportions, Fit selected size or Crop to selected size.")
    source_w, source_h = map(int, source_size)
    target_w, target_h = map(int, target_size)
    if min(source_w, source_h, target_w, target_h, multiple) < 1:
        raise ValueError("Image dimensions must be positive.")
    if mode in ("crop", "pad"):
        width, height = target_w, target_h
    else:
        scale = min(1.0, math.sqrt(target_w * target_h / (source_w * source_h)))
        width, height = max(1, round(source_w * scale)), max(1, round(source_h * scale))
    plan = dict(
        source_width=source_w,
        source_height=source_h,
        **canvas_plan((width, height), multiple),
        mode=mode,
    )
    if mode == "pad":
        scale = min(target_w / source_w, target_h / source_h)
        content_w = min(target_w, max(1, round(source_w * scale)))
        content_h = min(target_h, max(1, round(source_h * scale)))
        plan.update(
            content_width=content_w,
            content_height=content_h,
            content_left=(target_w - content_w) // 2,
            content_top=(target_h - content_h) // 2,
        )
    return plan


def prepare_reference(image, target_size, mode="preserve", *, multiple=16, preserve_alpha=False):
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(image).convert("RGBA" if preserve_alpha else "RGB")
    plan = reference_plan(image.size, target_size, mode, multiple)
    size = plan["width"], plan["height"]
    content_size = (plan["content_width"], plan["content_height"]) if mode == "pad" else size
    image = (
        ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        if mode == "crop"
        else image.resize(content_size, Image.Resampling.LANCZOS)
    )
    pixels = torch.from_numpy(np.asarray(image).copy()).float().unsqueeze(0) / 255.0
    if mode == "pad":
        left, top = plan["content_left"], plan["content_top"]
        pixels = functional.pad(
            pixels.movedim(-1, 1),
            (left, size[0] - content_size[0] - left, top, size[1] - content_size[1] - top),
            # Match native ImagePadForOutpaint's neutral image fill, without
            # allocating its unused paint mask. Replication creates streaks in
            # wide content borders; only hidden alignment borders replicate.
            mode="constant",
            value=0.0 if preserve_alpha else 0.5,
        ).movedim(1, -1)
    left, top = plan["left"], plan["top"]
    right, bottom = plan["canvas_width"] - size[0] - left, plan["canvas_height"] - size[1] - top
    pixels = functional.pad(
        pixels.movedim(-1, 1), (left, right, top, bottom), mode="replicate"
    ).movedim(1, -1)
    return pixels, plan


def restore_reference_frame(pixels, plan):
    if plan is None:
        return pixels
    if tuple(pixels.shape[1:3]) != (plan["canvas_height"], plan["canvas_width"]):
        raise ValueError(
            "Decoded image does not match its reference canvas; refusing to stretch it."
        )
    top, left = plan["top"], plan["left"]
    return pixels[:, top : top + plan["height"], left : left + plan["width"], :]
