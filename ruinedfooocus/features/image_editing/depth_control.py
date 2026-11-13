"""Depth ControlNet preparation through the pinned native geometry nodes."""

import math
import sys

DEPTH_MODELS = {size: f"depth_anything_3_{size}.safetensors" for size in ("small", "base")}
DEPTH_DEFAULTS = dict(
    depth_preprocess="auto",
    depth_model="auto",
    depth_resolution=504,
    depth_resize_method="upper_bound_resize",
    depth_normalization="v2_style",
    depth_dtype="default",
    depth_lotus_model="",
    depth_lotus_vae="",
)
DEPTH_CHOICES = dict(
    depth_preprocess=("auto", "none"),
    depth_model=("auto", "small", "base", "lotus"),
    depth_resize_method=("upper_bound_resize", "lower_bound_resize"),
    depth_normalization=("v2_style", "min_max"),
    depth_dtype=("default", "fp16", "bf16", "fp32"),
)


def depth_settings(values):
    # Old presets/metadata supplied a depth map directly. Only new defaults opt in.
    options = {
        key: values.get(key, "none" if key == "depth_preprocess" else default)
        for key, default in DEPTH_DEFAULTS.items()
    }
    for key, choices in DEPTH_CHOICES.items():
        if options[key] not in choices:
            raise ValueError(f"{key}: choose a supported depth option.")
    resolution = options["depth_resolution"]
    if isinstance(resolution, bool):
        raise ValueError("Depth resolution must be a multiple of 14 from 140 to 2520.")
    try:
        resolution = float(resolution)
    except (TypeError, ValueError):
        raise ValueError("Depth resolution must be a multiple of 14 from 140 to 2520.") from None
    if not math.isfinite(resolution) or not 140 <= resolution <= 2520 or resolution % 14:
        raise ValueError("Depth resolution must be a multiple of 14 from 140 to 2520.")
    options["depth_resolution"] = int(resolution)
    for key in ("depth_lotus_model", "depth_lotus_vae"):
        if not isinstance(options[key], str):
            raise ValueError(f"{key}: choose an installed local file.")
        if (
            options["depth_preprocess"] == "auto"
            and options["depth_model"] == "lotus"
            and not options[key]
        ):
            raise ValueError("Select both the local Lotus depth checkpoint and its SD VAE.")
    return options


def depth_model_name(selection):
    return captured_depth_name(selection)


def captured_depth_name(selection, *, force_cpu=False):
    """Read the initialized native capacity snapshot without a device query."""
    if selection != "auto":
        return DEPTH_MODELS[selection]
    if force_cpu:
        return DEPTH_MODELS["small"]
    memory = sys.modules.get("comfy.model_management")
    states = getattr(memory, "CPUState", None)
    state = getattr(memory, "cpu_state", None)
    if states is None or state is None:
        raise ValueError(
            "Depth Auto selection is pending native initialization. Select Small or Base explicitly."
        )
    if state == states.CPU:
        return DEPTH_MODELS["small"]
    capacity = getattr(memory, "total_vram", None)
    if (
        state not in (states.GPU, states.MPS)
        or type(capacity) not in (int, float)
        or not math.isfinite(capacity)
        or capacity <= 0
    ):
        raise ValueError("Depth Auto capacity is unavailable. Select Small or Base explicitly.")
    return DEPTH_MODELS["base" if capacity >= 8192 else "small"]


def prepare_depth(owner, image, options, gen_data=None):
    options = depth_settings(options)
    if options["depth_preprocess"] == "none":
        return image
    if options["depth_model"] == "lotus":
        from ruinedfooocus.app.state import path_manager
        from ruinedfooocus.features.image_editing.lotus_depth import prepare_lotus

        return prepare_lotus(owner, image, options, gen_data, path_manager)
    import folder_paths
    from ruinedfooocus.app.state import path_manager
    from ruinedfooocus.features.models.model_identity import file_signature
    from comfy_extras.nodes_depth_anything_3 import LoadDA3Model

    from ruinedfooocus.features.generation.image_preparation import KEY, ImageAssets, prepared_asset

    captured = (gen_data or {}).get(KEY)
    model_name = (
        captured.depth_name
        if isinstance(captured, ImageAssets)
        else depth_model_name(options["depth_model"])
    )
    if model_name is None:
        raise ValueError("Depth preprocessing was not captured before compute.")
    with prepared_asset(
        gen_data, "depth-preprocessor", model_name, path_manager, "geometry_estimation"
    ) as path:
        signature = (file_signature(path), options["depth_dtype"])
        if signature != getattr(owner, "depth_model_signature", None):
            folder_paths.add_model_folder_path(
                "geometry_estimation", str(path.parent), is_default=True
            )
            model = LoadDA3Model.execute(model_name=path.name, weight_dtype=options["depth_dtype"])[
                0
            ]
            owner.depth_model = model
            owner.depth_model_signature = signature
    owner.depth_model_name = model_name
    return render_da3(owner.depth_model, image, options)


def render_da3(model, image, options, *, check_cancelled=None):
    """Render native monocular depth at the original input dimensions."""
    from comfy_extras.nodes_depth_anything_3 import DA3Inference, DA3Render

    if check_cancelled is not None:
        check_cancelled()
    geometry = DA3Inference.execute(
        da3_model=model,
        image=image,
        resolution=options["depth_resolution"],
        resize_method=options["depth_resize_method"],
        mode={"mode": "mono"},
    )[0]
    if check_cancelled is not None:
        check_cancelled()
    depth = DA3Render.execute(
        da3_geometry=geometry,
        output=dict(
            output="depth", normalization=options["depth_normalization"], apply_sky_clip=False
        ),
    )[0]
    if tuple(depth.shape) != tuple(image.shape):
        raise ValueError(
            "Depth preprocessing changed the input dimensions; refusing to stretch it."
        )
    return depth
