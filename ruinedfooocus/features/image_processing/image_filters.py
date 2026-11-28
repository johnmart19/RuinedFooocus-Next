"""Source-sized still filters using native tensor nodes, without a model recipe."""

from dataclasses import dataclass

from ruinedfooocus.features.image_processing import image_histogram
from ruinedfooocus.features.image_processing.native_vision import _number

KINDS = ("image-blur", "image-sharpen", "adjust-brightness", "adjust-contrast")
LABELS = dict(zip(KINDS, ("Blur image", "Sharpen image", "Adjust brightness", "Adjust contrast")))
KEY = "_image_filter_request"
DEFAULTS = {"filter_radius": 1, "filter_sigma": 1.0, "filter_strength": 1.0, "filter_factor": 1.0}
FIELDS = tuple(DEFAULTS)
NODES = dict(zip(KINDS, ("ImageBlur", "ImageSharpen", "AdjustBrightness", "AdjustContrast")))


@dataclass(frozen=True)
class FilterRequest:
    kind: str
    source: image_histogram.HistogramRequest
    parameters: tuple[tuple[str, object], ...]


def parameters(kind, data):
    if kind not in KINDS:
        raise ValueError("Choose a supported still-image filter.")
    if kind in ("adjust-brightness", "adjust-contrast"):
        return {"filter_factor": _number(data.get("filter_factor", 1.0), "Filter factor", 0, 2)}
    result = {
        "filter_radius": _number(
            data.get("filter_radius", 1), "Filter radius", 1, 31, integer=True
        ),
        "filter_sigma": _number(data.get("filter_sigma", 1.0), "Filter sigma", 0.1, 10),
    }
    if kind == "image-sharpen":
        result["filter_strength"] = _number(
            data.get("filter_strength", 1.0), "Sharpen strength", 0, 5
        )
    return result


def capture_assets(data, manager, models=None):
    from PIL import Image
    from ruinedfooocus.features.image_processing.depth_utilities import kind_from

    kind = kind_from(data)
    if (
        data.get("inpaint_toggle")
        or data.get("model_recipe_path")
        or data.get("reference_adapter") not in (None, "", "none")
    ):
        raise ValueError(
            "Turn off inpainting, generation recipes and reference adapters for this filter."
        )
    source = image_histogram.capture(data.get("input_image"))
    options = parameters(kind, data)
    if "filter_radius" in options and options["filter_radius"] >= min(source.size):
        raise ValueError("Filter radius must be smaller than both source dimensions.")
    result = dict(
        data,
        input_image=Image.frombytes("RGBA", source.size, source.rgba),
        controlnet={"type": kind},
    )
    result.update(options)
    result[KEY] = FilterRequest(kind, source, tuple(options.items()))
    result.pop("image_utility_source", None)
    return result, ()


def worker_data(data):
    from ruinedfooocus.features.image_processing.native_vision import worker_data as common

    result = common(dict(data, native_vision_model=data["controlnet"]["type"]))
    result.pop("native_vision_model", None)
    return result


def process(owner, data, manager, check_cancelled):
    import numpy as np
    import torch
    from PIL import Image
    from ruinedfooocus.features.image_processing.depth_utilities import kind_from, native_call
    from ruinedfooocus.features.image_processing.native_image_formats import advanced_output

    request = data.get(KEY)
    data.pop("image_utility_source", None)
    if not isinstance(request, FilterRequest) or request.kind != kind_from(data):
        raise ValueError("Capture this filter's source before processing. Submit it again.")
    source = request.source
    width, height = source.size
    pixels = np.frombuffer(source.rgba, dtype=np.uint8).reshape(height, width, 4)
    tensor = torch.from_numpy(pixels[..., :3].copy()).to(dtype=torch.float32).div_(255).unsqueeze(0)
    options = dict(request.parameters)
    check_cancelled()
    with torch.inference_mode():
        if request.kind in ("image-blur", "image-sharpen"):
            from comfy_extras.nodes_post_processing import Blur, Sharpen

            node = Blur if request.kind == "image-blur" else Sharpen
            arguments = [tensor, options["filter_radius"], options["filter_sigma"]]
            if request.kind == "image-sharpen":
                arguments.append(options["filter_strength"])
            output = native_call(check_cancelled, node.execute, *arguments)[0]
        else:
            from comfy_extras.nodes_dataset import AdjustBrightnessNode, AdjustContrastNode

            node = (
                AdjustBrightnessNode if request.kind == "adjust-brightness" else AdjustContrastNode
            )
            output = native_call(check_cancelled, node._process, tensor, options["filter_factor"])
    if tuple(output.shape) != tuple(tensor.shape) or not torch.isfinite(output).all():
        raise ValueError("The native filter returned invalid dimensions or non-finite pixels.")
    rgb = output[0].detach().float().cpu().numpy().copy()
    alpha = pixels[..., 3:4].astype(np.float32) / 255
    result = np.concatenate((rgb, alpha), axis=-1)
    check_cancelled()
    data["image_utility_source"] = {
        "operation": request.kind,
        "native_node": NODES[request.kind],
        "native_revision": image_histogram.NATIVE_REVISION,
        "width": width,
        "height": height,
        "source_sha256": source.source_sha256,
        "source_sha256_layout": "oriented RGBA8 bytes with width and height",
        "input_mode": source.input_mode,
        "has_alpha": source.has_alpha,
        "has_icc_profile": source.has_icc_profile,
        "working_mode": "encoded RGB8 to native float32",
        "color_policy": "no ICC conversion or linearization",
        "alpha": "source bytes preserved",
        **options,
    }
    if advanced_output(data):
        return [result]
    return [Image.fromarray((np.clip(result, 0, 1) * 255).round().astype(np.uint8), "RGBA")]


def output_metadata(data):
    from copy import deepcopy

    request = data[KEY]
    return {
        "software": "RuinedFooocus",
        "prompt": "",
        "negative": "",
        "controlnet": {"type": request.kind},
        "width": request.source.size[0],
        "height": request.source.size[1],
        "image_utility_source": deepcopy(data["image_utility_source"]),
        **dict(request.parameters),
    }


def replay_values(metadata):
    from ruinedfooocus.features.image_processing.depth_utilities import kind_from

    kind = kind_from(metadata)
    return {
        **DEFAULTS,
        **parameters(kind, metadata),
        "input_image": None,
        "image_source": None,
        "inpaint_toggle": False,
        "cn_selection": LABELS[kind],
        "prompt": "",
        "negative": "",
        "automatic_upscale": "Off",
        "automatic_upscale_target": "Generated image",
        "model_recipe_path": "",
        "reference_adapter": "none",
    }
