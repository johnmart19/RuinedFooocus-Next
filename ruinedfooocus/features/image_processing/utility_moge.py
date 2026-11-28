"""Native MoGe perspective depth/normals and equirectangular panorama depth."""

from ruinedfooocus.features.models.model_identity import cached_safetensors_header
from ruinedfooocus.features.image_processing.depth_utilities import (
    cached_model,
    clear_model,
    native_call,
)
from ruinedfooocus.features.image_processing.native_vision import _number


def validate_model(path):
    header = cached_safetensors_header(path)
    # These are the independent v1/v2 builder contracts from the pinned native
    # loader. DA3 also uses DINO, so a backbone marker alone is insufficient.
    v1 = any(
        key in header for key in ("backbone.cls_token", "backbone.embeddings.cls_token")
    ) and all(
        any(key.startswith(prefix) for key in header)
        for prefix in ("head.upsample_blocks.", "head.output_block.")
    )
    v2 = any(
        key in header
        for key in ("encoder.backbone.cls_token", "encoder.backbone.embeddings.cls_token")
    ) and all(
        any(key.startswith(prefix) for key in header)
        for prefix in (
            "encoder.output_projections.",
            "neck.input_blocks.",
            "points_head.output_blocks.",
            "mask_head.output_blocks.",
            "scale_head.",
        )
    )
    if not (v1 or v2):
        raise ValueError("Select a native MoGe v1, v2 or v3 safe tensor model.")
    return (
        "MoGe3" if any(key.startswith("refiner.") for key in header) else "MoGe2" if v2 else "MoGe1"
    )


COMPONENTS = (("image_utility_model", "moge-model", ("geometry_estimation",), validate_model),)
OUTPUTS = ("depth", "normal")
PROJECTIONS = ("perspective", "panorama")
NORMAL_CONVENTIONS = ("opengl", "directx")


def _choice(data, name, choices, default):
    value = data.get(name, default)
    if value not in choices:
        raise ValueError(f"{name}: choose one of {', '.join(choices)}.")
    return value


def parameters(data):
    result = {
        "moge_output": _choice(data, "moge_output", OUTPUTS, "depth"),
        "moge_projection": _choice(data, "moge_projection", PROJECTIONS, "perspective"),
        "moge_normal_convention": _choice(
            data, "moge_normal_convention", NORMAL_CONVENTIONS, "opengl"
        ),
        "moge_resolution_level": _number(
            data.get("moge_resolution_level", 2), "MoGe detail", 0, 9, integer=True
        ),
        "moge_refine_steps": _number(
            data.get("moge_refine_steps", 3), "MoGe refinement", 0, 8, integer=True
        ),
        "moge_split_resolution": _number(
            data.get("moge_split_resolution", 512),
            "MoGe panorama split resolution",
            256,
            1024,
            integer=True,
        ),
        "moge_merge_resolution": _number(
            data.get("moge_merge_resolution", 1920),
            "MoGe panorama merge resolution",
            256,
            8192,
            integer=True,
        ),
        "moge_panorama_batch_size": _number(
            data.get("moge_panorama_batch_size", 1),
            "MoGe panorama batch size",
            1,
            12,
            integer=True,
        ),
    }
    if result["moge_projection"] == "panorama" and result["moge_output"] != "depth":
        raise ValueError("MoGe panorama supports depth output; choose perspective for normals.")
    return result


class _RequestModel:
    """Add cooperative checks between native view batches without modifying cached weights."""

    def __init__(self, model, check_cancelled):
        self._model, self._check_cancelled = model, check_cancelled

    def __getattr__(self, name):
        return getattr(self._model, name)

    def infer(self, *args, **kwargs):
        return native_call(self._check_cancelled, self._model.infer, *args, **kwargs)


def render(owner, request, pixels, data, manager, check_cancelled):
    import torch
    from ruinedfooocus.features.image_processing.utility_device import run_utility

    def run(policy):
        try:
            output, details = _render(owner, request, pixels, check_cancelled, policy)
            if tuple(output.shape) != tuple(pixels.shape) or not torch.isfinite(output).all():
                raise ValueError("The native utility returned invalid image dimensions or values.")
            # Include the actual host transfer in recovery: asynchronous device
            # errors can surface here after native inference/rendering returned.
            output = native_call(check_cancelled, lambda: output.detach().float().cpu())
            return output, details
        except BaseException:
            clear_model(owner)
            raise

    result = run_utility(
        "MoGe image utility", run, check_cancelled=check_cancelled, native_model_placement=True
    )
    output, details = result.value
    return output, {**details, "device_policy": result.metadata()}


def _render(owner, request, pixels, check_cancelled, policy):
    import folder_paths
    from comfy_extras.nodes_moge import LoadMoGeModel, MoGeInference, MoGeRender

    (item,) = request.files

    def load():
        folder_paths.add_model_folder_path(
            "geometry_estimation", str(item.path.parent), is_default=True
        )
        model = LoadMoGeModel.execute(item.path.name)[0]
        # Only fresh request-owned patchers are retargeted for explicit CPU or
        # a genuine failed accelerator attempt. Native offload and dtype stay.
        if str(getattr(model, "load_device", policy.device)) != str(policy.device):
            model.patcher.load_device = policy.device
        model.load_device = policy.device
        return model, (model.patcher,)

    options = parameters(dict(request.parameters))
    model = _RequestModel(
        cached_model(
            owner,
            request,
            load,
            check_cancelled,
            source_identity=("native-device", str(policy.device)),
        ),
        check_cancelled,
    )
    panorama = options["moge_projection"] == "panorama"
    if panorama:
        from comfy_extras.nodes_moge import MoGePanoramaInference

        geometry = native_call(
            check_cancelled,
            MoGePanoramaInference.execute,
            model,
            pixels,
            resolution_level=options["moge_resolution_level"],
            split_resolution=options["moge_split_resolution"],
            merge_resolution=options["moge_merge_resolution"],
            batch_size=options["moge_panorama_batch_size"],
            refine_steps=options["moge_refine_steps"],
        )[0]
    else:
        geometry = native_call(
            check_cancelled,
            MoGeInference.execute,
            model,
            pixels,
            resolution_level=options["moge_resolution_level"],
            fov_x_degrees=0.0,
            batch_size=1,
            force_projection=True,
            apply_mask=True,
            refine_steps=options["moge_refine_steps"],
        )[0]
    normal = options["moge_output"] == "normal"
    if normal and "normal" not in geometry and min(pixels.shape[1:3]) < 2:
        raise ValueError("This MoGe model needs an image at least 2 × 2 pixels to derive normals.")
    mode = "normal_" + options["moge_normal_convention"] if normal else "depth"
    output = native_call(check_cancelled, MoGeRender.execute, geometry, output=mode)[0]
    details = {
        "native": validate_model(item.path),
        "native_output": "MOGE_GEOMETRY → IMAGE",
        "native_inference": "MoGePanoramaInference" if panorama else "MoGeInference",
        "native_render_mode": mode,
        "projection": "equirectangular" if panorama else "perspective",
        "inference_device": str(model.load_device),
        "inference_dtype": str(getattr(model, "dtype", "unknown")),
        "output_device": "cpu",
    }
    if normal:
        details.update(
            convention="native-" + options["moge_normal_convention"],
            normalization="native-normal*0.5+0.5",
            normal_source="native-predicted" if "normal" in geometry else "native-point-derived",
            normal_frame="native-camera",
        )
    else:
        details.update(
            convention="white-near; uncovered-native-far-fill"
            if panorama
            else "white-near; invalid-black",
            normalization="native-disparity-0.1/99.9-percentile",
            depth_measure="native-radial-distance" if panorama else "native-camera-depth",
        )
    return output, details
