"""Optional model-based image enlargement after native VAE decoding."""

from functools import lru_cache
import json
import sys

from ruinedfooocus.core.paths import resource_path

PRESETS = {
    "Off": (None, 1),
    "2× Anime (Futsuu)": ("2x-Futsuu-Anime.pth", 2),
    "2× Anime cleanup (AniScale)": ("2x-AniScale.pth", 2),
    "2× Anime sharp (AnimeSharp V4 Fast)": ("2x-AnimeSharpV4_Fast_RCAN_PU.safetensors", 2),
    "2× General (RealESRGAN)": ("RealESRGAN_x2plus.pth", 2),
    "4× Anime (RealESRGAN)": ("RealESRGAN_x4plus_anime_6B.pth", 4),
    "4× General (RealESRGAN)": ("RealESRGAN_x4plus.pth", 4),
    "4× UltraSharp": ("4x-UltraSharp.pth", 4),
}
# Saved selections from the first version remain valid without family guessing.
ALIASES = {"2×": "2× General (RealESRGAN)", "4×": "4× General (RealESRGAN)"}


@lru_cache(maxsize=1)
def registered_upscalers():
    """Read only pixel-model facts; never scan weights or qualification records."""
    from ruinedfooocus.features.models.model_registry import load_registry

    registry = load_registry()
    catalogue = {}
    for path in resource_path("model_catalogue").glob("*.json"):
        catalogue.update(json.loads(path.read_text(encoding="utf-8")))
    result = {}
    for key, download in catalogue.items():
        if download.get("path") != "path_upscalers":
            continue
        artifact = registry.annotations_for(key, download)
        if artifact is None or artifact["role"] != "pixel_upscaler":
            continue
        profile = registry.profile(artifact["profile_id"])
        scale = profile.get("native_scale")
        if type(scale) is int and scale > 0:
            result[download["filename"]] = dict(
                native_scale=scale,
                input_channels=profile.get("input_channels"),
                output_channels=profile.get("output_channels"),
                source="registered native model",
            )
    return result


def native_metadata(name, manager=None):
    metadata = registered_upscalers().get(name)
    if metadata is not None:
        return dict(metadata)
    from ruinedfooocus.core.cache_contract import (
        configured_cache_root,
        file_identity,
        read_record,
        record_key,
    )

    manager = manager or getattr(sys.modules.get("ruinedfooocus.app.state"), "path_manager", None)
    if manager is None:
        return None
    path = next(
        (root / name for root in manager.get_model_folders("upscalers") if (root / name).is_file()),
        None,
    )
    if path is None:
        return None
    identity = file_identity(path)
    cache = (
        configured_cache_root(manager)
        / "upscaler-metadata"
        / (record_key("native-upscaler", identity) + ".json")
    )
    value = read_record(cache, "native-upscaler", identity)
    if (
        not isinstance(value, dict)
        or type(value.get("native_scale")) is not int
        or value["native_scale"] < 1
    ):
        return None
    return value


def remember_native_metadata(path, model, manager):
    """Publish descriptor facts only after the native loader has inspected weights."""
    from ruinedfooocus.core.cache_contract import (
        configured_cache_root,
        file_identity,
        record_key,
        write_record,
    )

    if any(
        type(getattr(model, key, None)) is not int
        for key in ("scale", "input_channels", "output_channels")
    ):
        return
    identity = file_identity(path)
    value = dict(
        native_scale=int(model.scale),
        input_channels=int(model.input_channels),
        output_channels=int(model.output_channels),
        source="native model inspection",
    )
    cache = (
        configured_cache_root(manager)
        / "upscaler-metadata"
        / (record_key("native-upscaler", identity) + ".json")
    )
    write_record(cache, "native-upscaler", identity, value)


def choices(factor=None, *, advanced=True):
    manager = getattr(sys.modules.get("ruinedfooocus.app.state"), "path_manager", None)
    names = set(registered_upscalers()) | set(getattr(manager, "upscaler_filenames", ()))

    def compatible(name):
        metadata = native_metadata(name, manager)
        return (
            metadata is not None
            and metadata.get("input_channels") in (None, 3)
            and (
                metadata["native_scale"] == factor
                if factor is not None
                else advanced or metadata["native_scale"] in (2, 4)
            )
        )

    presets = [label for label, (name, _) in PRESETS.items() if name is None or compatible(name)]
    preset_models = {model for model, _ in PRESETS.values()}
    return presets + sorted(
        name for name in names if name not in preset_models and compatible(name)
    )


def filename_scale(name):
    """Compatibility name: scale is descriptor data, never a filename hint."""
    metadata = native_metadata(name)
    return metadata["native_scale"] if metadata else None


def resolve(selection):
    selection = ALIASES.get(selection, selection) or "Off"
    if selection == "Off":
        return None, 1
    name = PRESETS[selection][0] if selection in PRESETS else selection
    metadata = native_metadata(name)
    if metadata is None:
        raise ValueError(
            "Choose a registered upscale model, or inspect the local model in Image tools first."
        )
    if metadata.get("input_channels") not in (None, 3):
        raise ValueError("This native upscaler does not accept RGB images.")
    return name, metadata["native_scale"]


def scale_for(selection):
    return resolve(selection)[1]


def describe_size(
    selection, resolution, width, height, resolutions, *, target_mode=False, alignment=16
):
    scale = scale_for(selection)
    if resolution in resolutions:
        width, height = resolutions[resolution]
    width, height = int(width), int(height)
    if target_mode:
        from ruinedfooocus.features.generation.size_plan import SizePlan

        plan = SizePlan.create((width, height), scale, alignment=alignment)
        return (
            (
                f"{plan.content_width} × {plan.content_height} content → "
                f"{plan.canvas_width} × {plan.canvas_height} native canvas → "
                f"{width} × {height} target ({scale}×)."
                + (
                    " Odd target: uniform fit with padding."
                    if plan.restoration == "uniform_fit_pad"
                    else ""
                )
            )
            if scale > 1
            else f"{width} × {height} pixels"
        )
    return f"{width * scale} × {height * scale} pixels" + (
        f" (from {width} × {height}, {scale}× upscale)" if scale > 1 else ""
    )


def model_for(selection):
    return resolve(selection)[0]


def upscale_decoded(owner, image, gen_data):
    gen_data.pop("automatic_upscale_model", None)
    from ruinedfooocus.features.generation.image_preparation import KEY, ImageAssets

    captured = gen_data.get(KEY)
    name, scale = (
        (captured.automatic_upscale_name, captured.automatic_upscale_scale)
        if isinstance(captured, ImageAssets)
        else resolve(gen_data.get("automatic_upscale"))
    )
    if name is None:
        from ruinedfooocus.features.generation.size_plan import KEY, SizePlan

        plan = gen_data.get(KEY)
        if isinstance(plan, SizePlan):
            image = plan.finish(image)
            gen_data["size_plan"] = plan.metadata()
        return image
    from ruinedfooocus.features.image_processing.upscale_pipeline import (
        pipeline,
        InterruptibleUpscaler,
    )
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
    import ruinedfooocus.app.async_worker as worker

    worker.check_interrupt(gen_data)
    worker.add_result(gen_data["task_id"], "preview", (-1, "Automatic upscaling ...", None))
    if getattr(owner, "automatic_upscale_pipeline", None) is None:
        owner.automatic_upscale_pipeline = pipeline()
    model = owner.automatic_upscale_pipeline.load_upscaler_model(name, gen_data)
    if model.scale != scale:
        raise ValueError(f"Unexpected scale for automatic upscaler {name}: {model.scale}")
    if getattr(model, "input_channels", 3) != 3 or getattr(model, "output_channels", 3) != 3:
        raise ValueError(
            "Automatic upscaling requires a native RGB upscaler; source alpha is preserved separately."
        )
    result = ImageUpscaleWithModel.execute(InterruptibleUpscaler(model, gen_data), image)[0]
    from ruinedfooocus.features.generation.size_plan import KEY, SizePlan

    plan = gen_data.get(KEY)
    if isinstance(plan, SizePlan):
        result = plan.finish(result)
        gen_data["size_plan"] = plan.metadata()
    worker.check_interrupt(gen_data)
    gen_data["automatic_upscale_model"] = name
    return result
