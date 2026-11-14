"""Native detection, point segmentation and monocular depth on the image lane."""

import json
import math
import ast
from functools import lru_cache
from pathlib import Path

from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.image_processing.sam_prompts import (
    DEFAULTS as SAM_DEFAULTS,
    FIELDS as SAM_FIELDS,
)

KINDS = {
    "detect": ("RT_DETR_v4",),
    "segment": ("SAM3", "SAM31"),
    "depth-map": ("DepthAnything3",),
}
FIELDS = (
    "native_vision_model",
    "vision_threshold",
    "detect_class",
    "detect_max_count",
    "segment_positive_points",
    "segment_negative_points",
    "segment_refine_iterations",
    "segment_output",
    *SAM_FIELDS,
)


DEFAULTS = {
    "vision_threshold": 0.5,
    "detect_class": "all",
    "detect_max_count": 100,
    "segment_positive_points": "[]",
    "segment_negative_points": "[]",
    "segment_refine_iterations": 2,
    "segment_output": "cutout",
    **SAM_DEFAULTS,
}


@lru_cache(maxsize=1)
def coco_classes():
    """Native class names accepted by RTDETR_detect, besides ``all``."""
    from ruinedfooocus.core.paths import APP_ROOT

    source = APP_ROOT / "repositories/ComfyUI/comfy/ldm/rt_detr/rtdetr_v4.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    value = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "COCO_CLASSES" for target in node.targets
        )
    )
    names = ast.literal_eval(value)
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("The pinned native detection class list is unavailable.")
    return tuple(names)


def validate_model(path, kind):
    from ruinedfooocus.features.models.utility_models import passive_utility_file

    if kind not in KINDS or Path(path).suffix.lower() not in (".safetensors", ".sft"):
        raise ValueError("Select an installed native vision safetensors model for this image tool.")
    native = passive_utility_file(path)
    from ruinedfooocus.features.models.model_architecture import native_detector_ready

    if native is None and native_detector_ready():
        from ruinedfooocus.features.models.model_architecture import inspect_model_file

        native = inspect_model_file(path).get("native_class")
    if native not in KINDS[kind]:
        raise ValueError("The selected model does not support this native image tool.")
    return native


def _roots(manager, kind):
    categories = (
        ("geometry_estimation",) if kind == "depth-map" else ("checkpoints", "diffusion_models")
    )
    return tuple(
        dict.fromkeys(
            Path(root).resolve()
            for category in categories
            for root in manager.get_model_folders(category)
        )
    )


def native_model_choices(manager, kind):
    """Header-only local choices; no weights, network or traversal outside roots."""
    choices, seen = [], set()
    for root in _roots(manager, kind):
        pending, visited = [root], set()
        while pending:
            folder = pending.pop()
            resolved = folder.resolve()
            if not resolved.is_relative_to(root) or resolved in visited:
                continue
            visited.add(resolved)
            try:
                children = sorted(folder.iterdir())
            except OSError:
                continue
            for path in children:
                if path.is_dir():
                    pending.append(path)
                elif path.is_file() and path.resolve().is_relative_to(root):
                    resolved_file = str(path.resolve())
                    if resolved_file in seen:
                        continue
                    seen.add(resolved_file)
                    try:
                        native = validate_model(path, kind)
                    except (OSError, ValueError, TypeError):
                        continue
                    choices.append(
                        (f"{path.relative_to(root).as_posix()} · {native}", resolved_file)
                    )
    return sorted(choices, key=lambda item: item[0].casefold())


def installed_tool_choice(path, manager):
    """Resolve a supported installed utility by its header, without runtime startup."""
    path = Path(path).resolve()
    before = tuple(file_signature(path))
    for kind in ("detect", "segment"):
        if not any(path.is_relative_to(root) for root in _roots(manager, kind)):
            continue
        try:
            validate_model(path, kind)
        except (ValueError, TypeError, OSError):
            continue
        if before != tuple(file_signature(path)):
            raise ValueError("The installed utility changed. Refresh and select it again.")
        return dict(kind=kind, resource="native_vision_model", value=str(path))
    return None


def _number(value, label, minimum, maximum, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    if not minimum <= value <= maximum or integer and int(value) != value:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return int(value) if integer else float(value)


def _points(value, image):
    if not isinstance(value, str) or len(value) > 16384:
        raise ValueError("Point prompts must be bounded JSON arrays of pixel coordinates.")
    try:
        points = json.loads(value)
    except (ValueError, RecursionError) as error:
        raise ValueError('Use point JSON such as [{"x":100,"y":80}].') from error
    if not isinstance(points, list) or len(points) > 256:
        raise ValueError("Use at most 256 points per point prompt.")
    result = []
    for point in points:
        if not isinstance(point, dict) or set(point) != {"x", "y"}:
            raise ValueError("Each point needs exactly x and y pixel coordinates.")
        result.append(
            dict(
                x=_number(point["x"], "Point x", 0, image.width - 1),
                y=_number(point["y"], "Point y", 0, image.height - 1),
            )
        )
    return json.dumps(result, separators=(",", ":"))


def capture_assets(data, manager, kind):
    from PIL import Image, ImageOps
    from ruinedfooocus.features.generation.image_preparation import selected_image_artifact

    if kind not in KINDS:
        raise ValueError("Choose a supported native image tool.")
    from ruinedfooocus.features.image_processing.utility_defaults import default_components

    result = default_components(manager, kind, data)
    image = result.get("input_image")
    if (
        not isinstance(image, Image.Image)
        or min(image.size) < 1
        or image.width * image.height > 16777216
    ):
        raise ValueError("Add an image of at most 16 megapixels in Image tools.")
    if result.get("inpaint_toggle"):
        raise ValueError("Turn off inpainting before running this image tool.")
    result["input_image"] = image = ImageOps.exif_transpose(image).convert("RGBA").copy()
    name = result.get("native_vision_model")
    if not isinstance(name, str) or not name.strip() or len(name) > 1024:
        raise ValueError(_missing_model(kind))
    try:
        item = selected_image_artifact(
            manager,
            name,
            "native-vision",
            ("geometry_estimation",)
            if kind == "depth-map"
            else ("checkpoints", "diffusion_models"),
            identities=(*KINDS[kind], "DA3-SMALL", "DA3-BASE")
            if kind == "depth-map"
            else KINDS[kind],
            source_roles=("depth_estimation",) if kind == "depth-map" else ("image_utility_model",),
            allowed_roots=_roots(manager, kind),
        )
    except ValueError as error:
        raise ValueError(f"{_missing_model(kind)} {error}") from error
    path, before = item.request.destination, item.signature
    if before is not None:
        validate_model(path, kind)
    result["native_vision_model"] = str(path)
    if kind in ("detect", "segment"):
        result["vision_threshold"] = _number(result.get("vision_threshold", 0.5), "Threshold", 0, 1)
    if kind == "detect":
        label = result.get("detect_class", "all")
        if not isinstance(label, str) or label != "all" and label not in coco_classes():
            raise ValueError("Choose all or an exact native COCO class name.")
        result["detect_class"] = label
        result["detect_max_count"] = _number(
            result.get("detect_max_count", 100), "Maximum detections", 1, 1000, integer=True
        )
    elif kind == "segment":
        from ruinedfooocus.features.image_processing import sam_prompts

        for key in ("segment_positive_points", "segment_negative_points"):
            result[key] = _points(result.get(key, "[]"), image)
        result = sam_prompts.capture(result, image)
        if result["segment_text"] and before is not None:
            sam_prompts.validate_text_encoder(path)
        result["segment_refine_iterations"] = _number(
            result.get("segment_refine_iterations", 2), "Refinement", 0, 5, integer=True
        )
        if result.get("segment_output", "cutout") not in ("cutout", "mask"):
            raise ValueError("Choose a segmentation cutout or mask.")
        result["segment_output"] = result.get("segment_output", "cutout")
    if before is not None and before != tuple(file_signature(path)):
        raise ValueError("The native vision model changed during capture. Select it again.")
    return result, item


def validate_prepared(data):
    """Validate newly acquired bytes before the request can enter compute."""
    kind = str(data["controlnet"]["type"]).lower()
    path = Path(data["native_vision_model"])
    before = tuple(file_signature(path))
    validate_model(path, kind)
    if kind == "segment" and data.get("segment_text"):
        from ruinedfooocus.features.image_processing.sam_prompts import validate_text_encoder

        validate_text_encoder(path)
    if before != tuple(file_signature(path)):
        raise ValueError("The native vision model changed during preparation. Submit it again.")
    return data


def _missing_model(kind):
    if kind == "depth-map":
        return "Select an installed Depth Anything 3 model in Image tools. DA3 models are available in Models."
    return "Select an installed local model for this image tool."


def worker_data(data):
    """Populate only the shared output/progress protocol, without model recipes."""
    from ruinedfooocus.features.generation.image_preparation import omit_generation_conditioning

    result = omit_generation_conditioning(data)
    result.pop("_prompt_preparation", None)
    result.pop("_sdcpp_snapshot", None)
    result.pop("_sdcpp_prepared", None)
    width, height = result["input_image"].size
    result.update(
        prompt="",
        negative="",
        loras=[],
        style_selection=[],
        base_model_name=Path(result["native_vision_model"]).name,
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
        automatic_upscale="Off",
        _sampling_captured=False,
    )
    return result


def _release_models(model, clip):
    from comfy.model_management import unload_model_and_clones

    try:
        if model is not None:
            unload_model_and_clones(model)
    finally:
        if clip is not None:
            unload_model_and_clones(clip.patcher)


def clear_model(owner):
    cached = getattr(owner, "_native_vision", None)
    if cached is not None:
        owner._native_vision = None
        _release_models(cached[2], cached[3] if len(cached) > 3 else None)


def process(owner, data, manager, check_cancelled):
    import numpy as np
    import torch
    from PIL import Image
    import comfy.sd
    from ruinedfooocus.features.generation.image_preparation import prepared_asset

    kind = str(data["controlnet"]["type"]).lower()
    name = data["native_vision_model"]
    image = data["input_image"].convert("RGBA")
    category = "geometry_estimation" if kind == "depth-map" else "checkpoints"
    with prepared_asset(data, "native-vision", name, manager, category) as path:
        signature = tuple(file_signature(path))
        native = validate_model(path, kind)
        check_cancelled()
        cached = getattr(owner, "_native_vision", None)
        needs_text = kind == "segment" and bool(data.get("segment_text"))
        if (
            cached is None
            or cached[:2] != (signature, kind)
            or needs_text
            and (len(cached) < 4 or cached[3] is None)
        ):
            clear_model(owner)
            clip = None
            if needs_text:
                model, clip, _, _ = comfy.sd.load_checkpoint_guess_config(
                    str(path), output_vae=False, output_clip=True, output_clipvision=False
                )
            else:
                model = comfy.sd.load_diffusion_model(str(path))
            if model is None or type(model.model.model_config).__name__ not in KINDS[kind]:
                _release_models(model, clip)
                raise ValueError("The native loader returned an incompatible vision model.")
            if needs_text and clip is None:
                _release_models(model, clip)
                raise ValueError("The SAM checkpoint did not load its paired language encoder.")
            owner._native_vision = (signature, kind, model, clip)
        model = owner._native_vision[2]
        pixels = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255)[None]
        check_cancelled()
        with torch.inference_mode():
            if kind == "depth-map":
                from ruinedfooocus.features.image_editing.depth_control import (
                    DEPTH_DEFAULTS,
                    render_da3,
                )

                depth = render_da3(model, pixels, DEPTH_DEFAULTS, check_cancelled=check_cancelled)
                if not torch.isfinite(depth).all():
                    raise ValueError("Native depth returned non-finite image data.")
                result = Image.fromarray(
                    (depth[0, :, :, 0].detach().cpu().clamp(0, 1).numpy() * 255)
                    .round()
                    .astype(np.uint8)
                )
                precision_result = depth[0, :, :, 0].detach().cpu().numpy().copy()
            elif kind == "detect":
                from comfy_extras.nodes_rtdetr import RTDETR_detect, DrawBBoxes, COCO_CLASSES

                label = data.get("detect_class", "all")
                if label != "all" and label not in COCO_CLASSES:
                    raise ValueError("Choose all or an exact native COCO class name.")
                boxes = RTDETR_detect.execute(
                    model, pixels, data["vision_threshold"], label, data["detect_max_count"]
                )[0]
                detections = [
                    {key: box[key] for key in ("label", "score", "x", "y", "width", "height")}
                    for box in boxes[0]
                ]
                output = DrawBBoxes.execute(boxes, pixels)[0]
                if tuple(output.shape) != tuple(pixels.shape) or not torch.isfinite(output).all():
                    raise ValueError("Native detection returned invalid image data.")
                result = Image.fromarray(
                    (output[0].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
                )
                result.putalpha(image.getchannel("A"))
                precision_result = np.concatenate(
                    (
                        output[0].detach().cpu().numpy(),
                        np.asarray(image.getchannel("A"), dtype=np.float32)[..., None] / 255,
                    ),
                    axis=-1,
                )
            else:
                from ruinedfooocus.features.image_processing import sam_prompts
                from ruinedfooocus.features.image_processing.native_image_formats import (
                    advanced_output,
                )

                masks, segment_boxes = sam_prompts.detect(
                    model, owner._native_vision[3], pixels, data, check_cancelled
                )
                segment_outputs = sam_prompts.render(
                    masks,
                    image,
                    cutout=data.get("segment_output", "cutout") == "cutout",
                    precision=advanced_output(data),
                    check_cancelled=check_cancelled,
                )
                if not data.get("segment_individual_masks"):
                    result = precision_result = segment_outputs[0]
        check_cancelled()
        if signature != tuple(file_signature(path)):
            raise ValueError("The vision model changed. Submit the image again.")
        # File name and size only: PNG metadata must not carry local folder paths.
        data["native_vision_source"] = dict(native=native, name=Path(name).name, size=signature[1])
        if kind == "detect":
            data["native_vision_source"].update(count=len(detections), detections=detections[:100])
        elif kind == "segment":
            data["native_vision_source"].update(
                output=data.get("segment_output", "cutout"),
                mask_role="inpaint-selection"
                if data.get("segment_output", "cutout") == "mask"
                else None,
                individual_masks=data.get("segment_individual_masks", False),
                output_count=len(segment_outputs),
                mask_order="native-sam3",
                detected_boxes=segment_boxes,
            )
            if data.get("segment_individual_masks"):
                data["native_vision_source"]["instance_count"] = len(segment_outputs)
        elif kind == "depth-map":
            data["native_vision_source"].update(
                resolution=DEPTH_DEFAULTS["depth_resolution"],
                resize_method=DEPTH_DEFAULTS["depth_resize_method"],
                normalization=DEPTH_DEFAULTS["depth_normalization"],
                convention="white-near",
                format="png-gray8",
            )
        from ruinedfooocus.features.image_processing.native_image_formats import (
            advanced_output,
            format_settings,
        )

        if advanced_output(data):
            data["native_vision_source"]["format"] = format_settings(data)
        if kind == "segment" and data.get("segment_individual_masks"):
            from ruinedfooocus.features.generation.image_output_batches import ImageOutputBatch

            return ImageOutputBatch(
                outputs=segment_outputs,
                plan=data["_native_image_output_plan"],
                metadata=tuple(
                    {
                        "native_vision_source": dict(
                            data["native_vision_source"], instance_index=index
                        )
                    }
                    for index in range(len(segment_outputs))
                ),
            )
        if advanced_output(data):
            result = precision_result
        return result
