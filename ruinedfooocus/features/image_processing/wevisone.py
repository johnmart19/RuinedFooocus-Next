"""Owned native WeVisOne detection and matting for one still image."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from ruinedfooocus.features.image_processing import depth_utilities as common
from ruinedfooocus.features.models.model_identity import cached_safetensors_header, file_signature

KINDS = ("wevisone-detect", "wevisone-matte")
LABELS = {"wevisone-detect": "WeVisOne detection", "wevisone-matte": "WeVisOne matting"}
KEY = "_wevisone_request"
DEFAULTS = {"wevisone_query": "person", "wevisone_max_length": 4096, "wevisone_output": "cutout"}
FIELDS = ("wevisone_model", *DEFAULTS)
NATIVE_TYPES = frozenset(("WEVISONE_2B", "WEVISONE_4B"))
MAX_PIXELS = 16_777_216
MAX_MASK_ELEMENTS = 64 * 1024**2
MAX_BOXES = 64
MAX_ANSWER_CHARS = 131072
NATIVE_FILES = ("comfy/sd.py", "comfy/text_encoders/wevisone.py", "comfy_extras/nodes_wevisone.py")


@dataclass(frozen=True)
class WeVisRequest(common.UtilityRequest):
    native_type: str
    native_sources: tuple


class WeVisOneArtifacts(list):
    def __init__(self, images, source):
        super().__init__(images)
        self.wevisone = deepcopy(source)


def parameters(data):
    query = data.get("wevisone_query", "person")
    if (
        not isinstance(query, str)
        or not query.strip()
        or len(query) > 4096
        or sum(bool(line.strip()) for line in query.splitlines()) > 64
        or any(ord(char) < 32 and char not in "\n\r\t" for char in query)
    ):
        raise ValueError(
            "Describe the subject or enter at most 64 object labels in 4096 characters."
        )
    length = data.get("wevisone_max_length", 4096)
    if type(length) is not int or not 1 <= length <= 32768:
        raise ValueError("WeVisOne answer length must be a whole number from 1 through 32768.")
    output = data.get("wevisone_output", "cutout")
    if output not in ("cutout", "mask", "overlay"):
        raise ValueError("Choose a WeVisOne cutout, grayscale matte or overlay.")
    values = dict(wevisone_query=query, wevisone_max_length=length, wevisone_output=output)
    if common.kind_from(data) == "wevisone-detect":
        values.pop("wevisone_output")
    return values


def native_source_signatures():
    from ruinedfooocus.core.paths import APP_ROOT

    try:
        return tuple(
            tuple(file_signature(APP_ROOT / "repositories/ComfyUI" / name)) for name in NATIVE_FILES
        )
    except OSError as error:
        raise ValueError(
            "WeVisOne requires the native ComfyUI 0.45 backend. Update the backend first."
        ) from error


def validate_model(path):
    from ruinedfooocus.features.models.native_text_identity import passive_native_text_type

    header = cached_safetensors_header(path)
    shapes = {name: row["shape"] for name, row in header.items() if name != "__metadata__"}
    native_type = passive_native_text_type(shapes)
    if native_type not in NATIVE_TYPES:
        raise ValueError(
            "Choose a WeVisOne-VL 2B or 4B image encoder, including its native mask decoder."
        )
    return native_type


def model_choices(manager):
    choices, seen, visited = [], set(), set()
    scanned = 0
    for root in common.roots(manager, ("clip",)):
        pending = [root]
        while pending:
            folder = pending.pop()
            try:
                current = folder.resolve()
                if not current.is_relative_to(root) or current in visited:
                    continue
                visited.add(current)
                for path in sorted(folder.iterdir()):
                    scanned += 1
                    if scanned > 16384:
                        return sorted(choices)
                    if path.is_dir():
                        pending.append(path)
                    elif path.suffix.lower() in (".safetensors", ".sft"):
                        resolved, name = common.selected_file(manager, str(path), ("clip",))
                        if resolved in seen:
                            continue
                        seen.add(resolved)
                        try:
                            native_type = validate_model(resolved)
                        except (OSError, ValueError, TypeError, KeyError):
                            continue
                        choices.append((f"{name} · {native_type}", str(resolved)))
            except (OSError, ValueError):
                continue
    return sorted(choices)


def installed_tool_choice(path, manager):
    path, _ = common.selected_file(manager, str(Path(path).resolve()), ("clip",))
    signature = tuple(file_signature(path))
    validate_model(path)
    if tuple(file_signature(path)) != signature:
        raise ValueError("The WeVisOne encoder changed. Refresh its installed record.")
    return dict(kind="wevisone-matte", resource="wevisone_model", value=str(path))


def capture_assets(data, manager, models=None):
    from PIL import Image, ImageOps
    from ruinedfooocus.core.preparation import PreparationRequest
    from ruinedfooocus.features.generation.image_preparation import ImageArtifact

    kind = common.kind_from(data)
    if kind not in KINDS:
        raise ValueError("Choose WeVisOne detection or matting.")
    source = data.get("input_image")
    if (
        not isinstance(source, Image.Image)
        or getattr(source, "n_frames", 1) != 1
        or min(source.size) < 1
        or source.width * source.height > MAX_PIXELS
        or source.mode not in ("1", "L", "LA", "P", "RGB", "RGBA")
    ):
        raise ValueError("Add one eight-bit source image of at most 16 megapixels for WeVisOne.")
    if (
        data.get("inpaint_toggle")
        or data.get("model_recipe_path")
        or data.get("reference_adapter") not in (None, "", "none")
    ):
        raise ValueError(
            "Turn off inpainting, generation recipes and reference adapters for WeVisOne."
        )
    image = ImageOps.exif_transpose(source).convert("RGBA").copy()
    values = parameters(data)
    path, name = common.selected_file(manager, data.get("wevisone_model"), ("clip",))
    signature = tuple(file_signature(path))
    native_type = validate_model(path)
    sources = native_source_signatures()
    if tuple(file_signature(path)) != signature:
        raise ValueError("The WeVisOne encoder changed during capture. Select it again.")
    item = common.UtilityFile("wevisone_model", "wevisone-clip", "clip", name, path, signature)
    result = dict(
        data, input_image=image, controlnet={"type": kind}, wevisone_model=str(path), **values
    )
    result[KEY] = WeVisRequest(
        kind, image.size, image.tobytes(), (item,), tuple(values.items()), native_type, sources
    )
    result.pop("image_utility_source", None)
    return result, (ImageArtifact(item.role, str(path), PreparationRequest("", path), signature),)


def worker_data(data):
    from ruinedfooocus.features.image_processing.native_vision import (
        worker_data as common_worker_data,
    )

    result = common_worker_data(dict(data, native_vision_model=data["wevisone_model"]))
    result.pop("native_vision_model", None)
    return result


def _boxes(boxes, width, height):
    if not isinstance(boxes, list) or len(boxes) > MAX_BOXES:
        raise ValueError(
            "WeVisOne returned too many objects. Narrow the subject or crop the source."
        )
    for box in boxes:
        if (
            not isinstance(box, dict)
            or not isinstance(box.get("label"), str)
            or len(box["label"]) > 4096
        ):
            raise ValueError("WeVisOne returned an invalid object label.")
        numbers = [box.get(key) for key in ("x", "y", "width", "height")]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in numbers):
            raise ValueError("WeVisOne returned non-finite box coordinates.")
        x, y, w, h = numbers
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width + 1e-5 or y + h > height + 1e-5:
            raise ValueError("WeVisOne returned a box outside the source image.")
    return boxes


class CheckedClip:
    """Delegate native CLIP, bounding generated mask allocation before node use."""

    def __init__(self, clip, parse_shapes, size, check_cancelled):
        self.clip, self.parse_shapes, self.size, self.check = (
            clip,
            parse_shapes,
            size,
            check_cancelled,
        )

    def __getattr__(self, name):
        return getattr(self.clip, name)

    def tokenize(self, *args, **kwargs):
        return common.native_call(self.check, self.clip.tokenize, *args, **kwargs)

    def generate(self, *args, **kwargs):
        return common.native_call(self.check, self.clip.generate, *args, **kwargs)

    def decode(self, *args, **kwargs):
        text = common.native_call(self.check, self.clip.decode, *args, **kwargs)
        if not isinstance(text, str) or len(text) > MAX_ANSWER_CHARS:
            raise ValueError("WeVisOne returned an oversized or invalid answer.")
        width, height = self.size
        boxes, _, objects = self.parse_shapes(text.strip(), width, height)
        _boxes(boxes, width, height)
        if max(1, len(boxes) if objects else 1) * width * height > MAX_MASK_ELEMENTS:
            raise ValueError(
                "WeVisOne instance mattes exceed the pixel budget. Narrow the subject or crop the source."
            )
        self.check()
        return text


def _load_clip(path):
    import comfy.sd
    import folder_paths
    from comfy.text_encoders.wevisone import WeVisOneTEModel, WeVisOneTokenizer

    clip = comfy.sd.load_clip(
        ckpt_paths=[str(path)],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
    )
    try:
        if not isinstance(clip.cond_stage_model, WeVisOneTEModel) or not isinstance(
            clip.tokenizer, WeVisOneTokenizer
        ):
            raise ValueError(
                "The native loader did not return a WeVisOne image encoder and tokenizer."
            )
    except BaseException:
        from comfy.model_management import unload_model_and_clones

        unload_model_and_clones(clip.patcher)
        raise
    return clip, (clip.patcher,)


def process(owner, data, manager, check_cancelled):
    import numpy as np
    import torch
    from PIL import Image
    from comfy_extras.nodes_wevisone import WeVisOneDetect, parse_shapes
    from ruinedfooocus.features.generation.image_preparation import prepared_asset
    from ruinedfooocus.features.image_processing.native_image_formats import advanced_output
    from ruinedfooocus.features.image_processing.sam_prompts import render

    request = data.get(KEY)
    if not isinstance(request, WeVisRequest) or request.kind != common.kind_from(data):
        raise ValueError("Capture the WeVisOne source and encoder before processing.")
    data.pop("image_utility_source", None)
    check_cancelled()
    if native_source_signatures() != request.native_sources:
        raise ValueError("The native WeVisOne source changed. Submit the image again.")
    item = request.files[0]
    image = Image.frombytes("RGBA", request.size, request.rgba)
    pixels = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255)[None]
    values = dict(request.parameters)
    with prepared_asset(data, item.role, str(item.path), manager, item.category) as path:
        if (
            path.resolve() != item.path
            or tuple(file_signature(path)) != item.signature
            or validate_model(path) != request.native_type
        ):
            raise ValueError("The captured WeVisOne encoder changed. Select it again.")
        clip = common.cached_model(
            owner,
            request,
            lambda: _load_clip(path),
            check_cancelled,
            source_identity=request.native_sources,
        )
        proxy = CheckedClip(clip, parse_shapes, request.size, check_cancelled)
        task = "detection" if request.kind == "wevisone-detect" else "segmentation"
        with torch.inference_mode():
            masks, frame_boxes, points, answer = common.native_call(
                check_cancelled,
                WeVisOneDetect.execute,
                proxy,
                pixels,
                task,
                values["wevisone_query"],
                False,
                max_length=values["wevisone_max_length"],
            )
            if not isinstance(frame_boxes, list) or len(frame_boxes) != 1:
                raise ValueError("WeVisOne returned invalid still-image detections.")
            boxes = _boxes(frame_boxes[0], image.width, image.height)
            if (
                not isinstance(masks, torch.Tensor)
                or not masks.is_floating_point()
                or tuple(masks.shape) != (1, image.height, image.width)
                or not torch.isfinite(masks).all()
                or masks.min() < 0
                or masks.max() > 1
            ):
                raise ValueError("WeVisOne returned an invalid native alpha matte.")
            if (
                not isinstance(points, str)
                or len(points) > MAX_ANSWER_CHARS
                or not isinstance(answer, str)
                or len(answer) > MAX_ANSWER_CHARS
            ):
                raise ValueError("WeVisOne returned invalid text or point metadata.")
            points = json.loads(points)
            if not isinstance(points, list) or len(points) > MAX_BOXES:
                raise ValueError("WeVisOne returned too many points.")
            for point in points:
                if (
                    not isinstance(point, dict)
                    or not isinstance(point.get("label"), str)
                    or len(point["label"]) > 4096
                    or point.get("frame") != 0
                    or any(
                        type(point.get(key)) not in (int, float)
                        or not math.isfinite(point[key])
                        or not 0 <= point[key] <= side
                        for key, side in (("x", image.width), ("y", image.height))
                    )
                ):
                    raise ValueError("WeVisOne returned an invalid still-image point.")
            precision = advanced_output(data)
            output_kind = "annotated" if task == "detection" else values["wevisone_output"]
            if output_kind in ("annotated", "overlay"):
                if output_kind == "annotated":
                    from comfy_extras.nodes_rtdetr import DrawBBoxes

                    output = common.native_call(
                        check_cancelled, DrawBBoxes.execute, frame_boxes, pixels
                    )[0]
                else:
                    from comfy_extras.nodes_mask import ImageCompositeMasked

                    tint = torch.zeros_like(pixels)
                    tint[..., 1] = 1
                    output = common.native_call(
                        check_cancelled,
                        ImageCompositeMasked.execute,
                        pixels,
                        tint,
                        0,
                        0,
                        False,
                        masks * 0.35,
                    )[0]
                if tuple(output.shape) != tuple(pixels.shape) or not torch.isfinite(output).all():
                    raise ValueError(
                        "WeVisOne drawing changed the source geometry or returned non-finite pixels."
                    )
                rgba = np.concatenate(
                    (
                        output[0].detach().float().cpu().numpy(),
                        np.asarray(image.getchannel("A"), dtype=np.float32)[..., None] / 255,
                    ),
                    axis=-1,
                )
                outputs = [
                    rgba
                    if precision
                    else Image.fromarray(
                        (np.clip(rgba, 0, 1) * 255).round().astype(np.uint8), "RGBA"
                    )
                ]
            else:
                outputs = list(
                    render(
                        masks,
                        image,
                        cutout=output_kind == "cutout",
                        precision=precision,
                        check_cancelled=check_cancelled,
                    )
                )
        check_cancelled()
        if (
            tuple(file_signature(path)) != item.signature
            or native_source_signatures() != request.native_sources
        ):
            raise ValueError("A WeVisOne source changed during processing. Submit the image again.")
        data["image_utility_source"] = dict(
            operation=request.kind,
            native_node="WeVisOneDetect",
            native_task=task,
            native_type=request.native_type,
            components=[dict(role=item.role, name=item.name, size=item.signature[1])],
            width=image.width,
            height=image.height,
            source_pixel_sha256=hashlib.sha256(request.rgba).hexdigest(),
            output=output_kind,
            alpha="source multiplied by native matte"
            if output_kind == "cutout"
            else "native mask"
            if output_kind == "mask"
            else "source",
            boxes=deepcopy(boxes),
            points=points,
            answer=answer,
            **values,
        )
    return outputs


def output_metadata(data):
    request = data[KEY]
    return dict(
        software="RuinedFooocus",
        prompt="",
        negative="",
        controlnet={"type": request.kind},
        width=request.size[0],
        height=request.size[1],
        image_utility_source=deepcopy(data["image_utility_source"]),
        **dict(request.parameters),
    )


def replay_values(data):
    return dict(
        DEFAULTS,
        **parameters(data),
        wevisone_model=None,
        input_image=None,
        image_source=None,
        inpaint_toggle=False,
        cn_selection=LABELS[common.kind_from(data)],
        prompt="",
        negative="",
        automatic_upscale="Off",
        automatic_upscale_target="Generated image",
        model_recipe_path="",
        reference_adapter="none",
        native_latent_input=None,
        native_positive_input=None,
        native_negative_input=None,
        native_export_latent=False,
        native_export_conditioning=False,
        native_export_handoff=False,
        model_patch_recipe="none",
        control_recipe="ordinary",
    )
