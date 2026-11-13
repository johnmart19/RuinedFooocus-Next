"""Immutable output intent, native canvas padding and exact pixel restoration.

Dimensions describe image pixels, never a memory-capacity estimate. Native
alignment and hard limits are supplied by the consuming model, not a recipe.
"""

from dataclasses import asdict, dataclass
import math
from numbers import Real

from ruinedfooocus.features.image_editing.image_geometry import canvas_plan, restore_reference_frame

KEY = "_image_size_plan"
VERSION = 1


def dimension(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or int(value) != value
        or value < 1
    ):
        raise ValueError("Image dimensions must be positive whole pixels.")
    return int(value)


@dataclass(frozen=True)
class SizePlan:
    version: int
    target_width: int
    target_height: int
    scale: int
    content_width: int
    content_height: int
    canvas_width: int
    canvas_height: int
    left: int
    top: int
    alignment: int
    restoration: str

    @classmethod
    def create(cls, target, scale=1, *, alignment=8, minimum=1, maximum=None):
        width, height = map(dimension, target)
        scale, alignment, minimum = map(dimension, (scale, alignment, minimum))
        content = (math.ceil(width / scale), math.ceil(height / scale))
        if min(content) < minimum or (maximum is not None and max(content) > maximum):
            alternative = f"{minimum * scale} pixels or more on each side"
            if maximum is not None:
                alternative += f", at most {maximum * scale} pixels per side"
            raise ValueError(
                f"{scale}× needs {content[0]}×{content[1]} content; this model requires "
                f"{minimum}–{maximum or 'unlimited'} pixels. Choose a target of {alternative}, "
                "a different factor or another model."
            )
        frame = canvas_plan(content, alignment)
        if maximum is not None and max(frame["canvas_width"], frame["canvas_height"]) > maximum:
            raise ValueError("Aligned native canvas exceeds this model's maximum side.")
        return cls(
            VERSION,
            width,
            height,
            scale,
            *content,
            frame["canvas_width"],
            frame["canvas_height"],
            frame["left"],
            frame["top"],
            alignment,
            "uniform_fit_pad" if width % scale or height % scale else "padding_only",
        )

    @classmethod
    def from_metadata(cls, value):
        if (
            not isinstance(value, dict)
            or type(value.get("version")) is not int
            or value["version"] != VERSION
        ):
            raise ValueError("Unsupported image size plan; choose an output size again.")
        if any(
            type(value.get(field)) is not int
            for field in (
                "target_width",
                "target_height",
                "scale",
                "content_width",
                "content_height",
                "canvas_width",
                "canvas_height",
                "left",
                "top",
                "alignment",
            )
        ):
            raise ValueError("Image size plan needs whole-pixel integer geometry.")
        try:
            result = cls.create(
                (value["target_width"], value["target_height"]),
                value["scale"],
                alignment=value["alignment"],
            )
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid image size plan.") from error
        if result.metadata() != value:
            raise ValueError("Image size plan geometry is inconsistent.")
        return result

    @property
    def target(self):
        return self.target_width, self.target_height

    @property
    def content(self):
        return self.content_width, self.content_height

    @property
    def canvas(self):
        return self.canvas_width, self.canvas_height

    def metadata(self):
        return asdict(self)

    def aligned(self, alignment, *, minimum=1, maximum=None):
        return self.create(
            self.target, self.scale, alignment=alignment, minimum=minimum, maximum=maximum
        )

    def restore_canvas(self, pixels):
        return restore_reference_frame(
            pixels,
            dict(
                width=self.content_width,
                height=self.content_height,
                canvas_width=self.canvas_width,
                canvas_height=self.canvas_height,
                left=self.left,
                top=self.top,
            ),
        )

    def finish(self, pixels):
        """Divisible targets are pixel-exact; odd targets fit uniformly and pad."""
        expected = self.content_height * self.scale, self.content_width * self.scale
        if tuple(pixels.shape[1:3]) != expected:
            raise ValueError("Native upscaler output disagrees with its captured size plan.")
        return fit_tensor(pixels, self.target)


def fit_tensor(pixels, target):
    """Uniformly fit BHWC pixels; fill RGB neutrally and alpha transparently."""
    import torch.nn.functional as functional

    width, height = map(dimension, target)
    source_h, source_w = pixels.shape[1:3]
    if (source_w, source_h) == (width, height):
        return pixels
    scale = min(width / source_w, height / source_h)
    fitted = (
        min(width, max(1, round(source_w * scale))),
        min(height, max(1, round(source_h * scale))),
    )
    image = functional.interpolate(
        pixels.movedim(-1, 1),
        size=(fitted[1], fitted[0]),
        mode="bicubic",
        align_corners=False,
    ).clamp(0, 1)
    left, top = (width - fitted[0]) // 2, (height - fitted[1]) // 2
    padding = left, width - fitted[0] - left, top, height - fitted[1] - top
    if image.shape[1] == 4:
        rgb = functional.pad(image[:, :3], padding, value=0.5)
        alpha = functional.pad(image[:, 3:], padding, value=0.0)
        import torch

        return torch.cat((rgb, alpha), 1).movedim(1, -1)
    return functional.pad(image, padding, value=0.5).movedim(1, -1)


def replay_size(metadata):
    """New PNGs own final targets; old PNGs own their recorded sampling size."""
    if "size_plan" in metadata:
        return SizePlan.from_metadata(metadata["size_plan"]).target
    return (
        metadata.get("generation_width", metadata.get("width")),
        metadata.get("generation_height", metadata.get("height")),
    )


def capture_size_plan(data, resolutions, *, family=None, controls=None):
    """Capture only explicit new target intent, before waiting for any resources."""
    result = dict(data)
    from ruinedfooocus.features.image_processing.image_composition import handles

    if handles((controls or result.get("controlnet") or {}).get("type")):
        result.pop(KEY, None)
        return result
    if isinstance(result.get(KEY), SizePlan):
        return result
    from ruinedfooocus.features.gallery.image_metadata import read_metadata

    imported = read_metadata(result.get("prompt"))
    if imported is not None:
        if "size_plan" not in imported:
            result["size_mode"] = "legacy_base"
            return result
        plan = SizePlan.from_metadata(imported["size_plan"])
        from ruinedfooocus.features.generation.automatic_upscale import scale_for

        native_scale = 2 if imported.get("cascade_super_resolution") is True else 1
        if scale_for(imported.get("automatic_upscale", "Off")) * native_scale != plan.scale:
            raise ValueError("PNG target plan and recorded native upscale factor disagree.")
        result[KEY] = plan
        return result
    if result.get("size_mode") != "target":
        return result
    if family == "SD_X4Upscaler":
        return result
    kind = str((controls or {}).get("type", "")).lower()
    input_only = result.get("automatic_upscale_target") == "Input image only"
    if input_only and result.get("resolution_auto") is True:
        return result
    from ruinedfooocus.features.image_processing.still_image_utilities import (
        KINDS as DEPTH_UTILITIES,
    )

    if kind in ("rembg", "detect", "segment", "depth-map", "restore-seedvr2", *DEPTH_UTILITIES) or (
        kind == "upscale" and not input_only
    ):
        return result
    if not input_only and (
        result.get("inpaint_toggle")
        or (
            (
                result.get("input_image") is not None
                or kind in ("", "none")
                and result.get("model_patch_recipe") == "zimage-fun"
                and result.get("model_patch_inpaint_image") is not None
                or kind == "control"
                and result.get("control_recipe") == "alimama-inpaint"
                and result.get("control_inpaint_mode", "mask") == "mask"
                and result.get("control_inpaint_image") is not None
            )
            and result.get("image_resize_mode", "preserve") == "preserve"
        )
    ):
        return result
    from ruinedfooocus.features.generation.automatic_upscale import scale_for

    selected = result.get("aspect_ratios_selection")
    target = (
        (result["width"], result["height"])
        if result.get("width") is not None and result.get("height") is not None
        else (
            getattr(resolutions, "aspect_ratios", {}).get(selected)
            or (result.get("custom_width", 1024), result.get("custom_height", 1024))
        )
    )
    scale = scale_for(result.get("automatic_upscale", "Off"))
    cascade_sr = (
        family in ("Stable_Cascade_C", "StableCascade_C")
        and result.get("cascade_super_resolution") is True
    )
    if cascade_sr:
        scale *= 2
    if max(map(dimension, target)) > 32768 or math.prod(target) > 67108864:
        raise ValueError(
            "Output target exceeds 32,768 pixels per side or the 64-megapixel application limit."
        )
    hidream = family in ("HiDreamO1", "HiDream O1") and not input_only
    result[KEY] = SizePlan.create(
        target,
        scale,
        alignment=1 if input_only else 32 if hidream or cascade_sr else 8,
        minimum=64 if hidream else 1,
        maximum=4096 if hidream else None,
    )
    return result
