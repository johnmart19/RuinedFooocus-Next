"""Bounded still-image inputs and outputs for the pinned native SAM3 detector."""

import ast
from functools import lru_cache
import json
import math
import re

MAX_INSTANCES = 64
MAX_MASK_PIXELS = 64 * 1024 * 1024
MAX_OUTPUT_BYTES = 256 * 1024 * 1024
FIELDS = ("segment_text", "segment_boxes", "segment_individual_masks", "segment_max_instances")
DEFAULTS = dict(
    segment_text="", segment_boxes="[]", segment_individual_masks=False, segment_max_instances=16
)


@lru_cache(maxsize=1)
def _text_parser():
    """Reuse native prompt syntax without importing its tensor/device modules."""
    from ruinedfooocus.core.paths import APP_ROOT

    path = APP_ROOT / "repositories/ComfyUI/comfy/text_encoders/sam3_clip.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parser = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_prompts"
    )
    namespace = {"re": re, "__builtins__": {"max": max, "round": round, "float": float}}
    exec(compile(ast.Module(body=[parser], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_parse_prompts"]


def _finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    return float(value)


def _boxes(value, image):
    if not isinstance(value, str) or len(value) > 32768:
        raise ValueError("Boxes must be bounded JSON with x, y, width and height in source pixels.")
    try:
        boxes = json.loads(value)
    except (ValueError, RecursionError) as error:
        raise ValueError("Use box JSON with x, y, width and height in source pixels.") from error
    if not isinstance(boxes, list) or len(boxes) > MAX_INSTANCES:
        raise ValueError("Use at most 64 source-image boxes.")
    result = []
    for box in boxes:
        if not isinstance(box, dict) or set(box) != {"x", "y", "width", "height"}:
            raise ValueError("Each box needs exactly x, y, width and height.")
        current = {key: _finite(value, "Box " + key) for key, value in box.items()}
        if not (
            0 <= current["x"] < image.width
            and 0 <= current["y"] < image.height
            and 0 < current["width"] <= image.width - current["x"]
            and 0 < current["height"] <= image.height - current["y"]
        ):
            raise ValueError(
                "Boxes must have positive size and stay inside the oriented source image."
            )
        result.append(current)
    return result


def capture(data, image):
    """Validate literal native prompts and freeze the one-run output limit."""
    from ruinedfooocus.features.generation.image_output_batches import OutputPlan, with_output_plan
    from ruinedfooocus.features.image_processing.native_image_formats import advanced_output

    result = dict(data)
    text = result.get("segment_text", "")
    if (
        not isinstance(text, str)
        or len(text) > 1024
        or any(ord(char) < 32 and char not in "\n\r\t" for char in text)
    ):
        raise ValueError("SAM text must be at most 1024 literal characters without control codes.")
    text = text.strip()
    try:
        categories = _text_parser()(text) if text else []
    except (ValueError, OverflowError) as error:
        raise ValueError("Use native SAM categories such as person:3, dog:2.") from error
    if text and not categories or len(categories) > 16:
        raise ValueError("Use one to 16 native SAM text categories.")
    maximum = result.get("segment_max_instances", DEFAULTS["segment_max_instances"])
    if (
        type(maximum) not in (int, float)
        or not math.isfinite(maximum)
        or not 1 <= maximum <= MAX_INSTANCES
        or int(maximum) != maximum
    ):
        raise ValueError("Maximum instances must be an integer between 1 and 64.")
    maximum = int(maximum)
    individual = result.get("segment_individual_masks", False)
    if type(individual) is not bool:
        raise ValueError("Individual masks must be true or false.")
    boxes = _boxes(result.get("segment_boxes", "[]"), image)
    has_positive = result.get("segment_positive_points", "[]") != "[]"
    has_points = has_positive or result.get("segment_negative_points", "[]") != "[]"
    if not (has_positive or boxes or categories):
        raise ValueError(
            "Add a positive point, a source-image box or a text category for segmentation."
        )
    pixel_count = image.width * image.height
    channels = 4 if result.get("segment_output", "cutout") == "cutout" else 1
    bytes_per_pixel = channels * (4 if advanced_output(result) else 1)
    limit = min(
        maximum,
        MAX_MASK_PIXELS // pixel_count,
        MAX_OUTPUT_BYTES // (pixel_count * bytes_per_pixel),
    )
    if any(type(count) is not int or not 1 <= count <= MAX_INSTANCES for _, count in categories):
        raise ValueError("Each native SAM text category must request one to 64 detections.")
    masks = int(has_points) + (sum(count for _, count in categories) if categories else len(boxes))
    if masks > limit:
        raise ValueError(
            f"These prompts can return {masks} masks; at most {limit} fit the selected instance and aggregate pixel limits."
        )
    result.update(
        segment_text=text,
        segment_boxes=json.dumps(boxes, separators=(",", ":")),
        segment_individual_masks=individual,
        segment_max_instances=maximum,
        _sam_mask_limit=limit,
    )
    if individual:
        result = with_output_plan(
            result,
            OutputPlan(
                kind="sam-instances",
                min_per_generation=int(has_points) + (len(boxes) if not categories else 0),
                max_per_generation=limit,
            ),
        )
    return result


def validate_text_encoder(path):
    """The selected SAM source itself must contain its paired language backbone."""
    from ruinedfooocus.features.models.model_identity import cached_safetensors_header

    header = cached_safetensors_header(path)
    for prefix in ("detector.backbone.language_backbone.", "backbone.language_backbone."):
        token = header.get(prefix + "encoder.token_embedding.weight", {})
        position = header.get(prefix + "encoder.positional_embedding", {})
        if token.get("shape") == [49408, 1024] and position.get("shape") == [32, 1024]:
            return
    raise ValueError(
        "Text segmentation needs the selected SAM checkpoint's paired language encoder."
    )


def detect(model, clip, pixels, data, check_cancelled):
    """Call native text encoding and SAM; native code owns all mask semantics."""
    import torch
    from comfy_extras.nodes_sam3 import SAM3_Detect

    conditioning = None
    if data.get("segment_text"):
        from nodes import CLIPTextEncode

        if clip is None:
            raise ValueError("The SAM checkpoint did not load its paired language encoder.")
        conditioning = CLIPTextEncode().encode(clip, data["segment_text"])[0]
        check_cancelled()
    output = SAM3_Detect.execute(
        model,
        pixels,
        conditioning=conditioning,
        bboxes=json.loads(data.get("segment_boxes", "[]")) or None,
        positive_coords=data["segment_positive_points"],
        negative_coords=data["segment_negative_points"],
        threshold=data["vision_threshold"],
        refine_iterations=data["segment_refine_iterations"],
        individual_masks=data.get("segment_individual_masks", False),
    )
    masks = output[0]
    individual = data.get("segment_individual_masks", False)
    if (
        not isinstance(masks, torch.Tensor)
        or masks.ndim != 3
        or not masks.is_floating_point()
        or tuple(masks.shape[1:]) != tuple(pixels.shape[1:3])
        or masks.shape[0] > (data["_sam_mask_limit"] if individual else 1)
        or not individual
        and masks.shape[0] != 1
        or masks.numel() > MAX_MASK_PIXELS
        or not torch.isfinite(masks).all()
    ):
        raise ValueError("Native segmentation returned invalid or oversized masks.")
    boxes = output[1]
    if (
        not isinstance(boxes, list)
        or len(boxes) != 1
        or not isinstance(boxes[0], list)
        or len(boxes[0]) > data["_sam_mask_limit"]
    ):
        raise ValueError("Native segmentation returned invalid or oversized boxes.")
    detected = []
    for box in boxes[0]:
        keys = ("x", "y", "width", "height", "score")
        if not isinstance(box, dict) or not set(keys) <= box.keys():
            raise ValueError("Native segmentation returned invalid boxes.")
        item = {key: _finite(box[key], "Detected box " + key) for key in keys}
        if item["width"] < 0 or item["height"] < 0 or not 0 <= item["score"] <= 1:
            raise ValueError("Native segmentation returned invalid boxes.")
        detected.append(item)
    check_cancelled()
    return masks, detected


def render(masks, image, *, cutout, precision, check_cancelled):
    """Apply the native mask to original alpha; retain float output when requested."""
    import numpy as np
    from PIL import Image, ImageChops

    if masks.numel() * (4 if cutout else 1) * (4 if precision else 1) > MAX_OUTPUT_BYTES:
        raise ValueError("Native segmentation exceeds the combined output byte budget.")
    results = []
    for mask in masks:
        check_cancelled()
        scalar = mask.detach().float().cpu().clamp(0, 1).numpy().copy()
        if precision:
            result = np.asarray(image, dtype=np.float32) / 255 if cutout else scalar
            if cutout:
                result[..., 3] *= scalar
        else:
            alpha = Image.fromarray((scalar * 255).round().astype(np.uint8))
            result = image.copy() if cutout else alpha
            if cutout:
                result.putalpha(ImageChops.multiply(image.getchannel("A"), alpha))
        results.append(result)
    return tuple(results)
