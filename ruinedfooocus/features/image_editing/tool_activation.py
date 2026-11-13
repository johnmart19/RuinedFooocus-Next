"""Atomic installed-tool handoffs, without model loads or profile writes."""

from pathlib import Path


def installed_tool(row, manager):
    """Resolve only callable local resources; a basename match is insufficient."""
    from ruinedfooocus.features.models.acquisition_actions import validate_local_record

    category = row.get("category")
    if category not in (
        "upscalers",
        "background_removal",
        "geometry_estimation",
        "checkpoints",
        "diffusion_models",
        "clip",
        "text_encoders",
    ):
        return None
    path = validate_local_record(row).resolve()
    if category in ("clip", "text_encoders"):
        from ruinedfooocus.features.image_processing.wevisone import installed_tool_choice

        try:
            return installed_tool_choice(path, manager)
        except (ValueError, OSError):
            return None
    if category in ("checkpoints", "diffusion_models"):
        from ruinedfooocus.features.models.utility_models import passive_utility_file

        if passive_utility_file(path) == "SeedVR2":
            from ruinedfooocus.features.image_processing.native_seedvr2 import (
                _selected,
                validate_model,
            )

            path = _selected(manager, str(path))
            validate_model(path)
            return dict(kind="restore-seedvr2", resource="seedvr2_model", value=str(path))
        from ruinedfooocus.features.image_processing.native_vision import installed_tool_choice

        return installed_tool_choice(path, manager)
    if category == "upscalers":
        from ruinedfooocus.features.generation.automatic_upscale import native_metadata

        name = next(
            (
                path.relative_to(root).as_posix()
                for folder in manager.get_model_folders(category)
                if path.is_relative_to(root := Path(folder).resolve())
            ),
            None,
        )
        if name is None:
            return None
        first = next(
            (
                (Path(root) / name).resolve()
                for root in manager.get_model_folders(category)
                if (Path(root) / name).is_file()
            ),
            None,
        )
        metadata = native_metadata(name, manager)
        if first != path or not metadata or metadata.get("input_channels") not in (None, 3):
            return None
        return dict(kind="upscale", resource="cn_upscale", value=name)
    if category == "background_removal":
        from ruinedfooocus.features.image_processing.native_background import local_models

        if Path(local_models(manager).get(path.name, ".")).resolve() != path:
            return None
        return dict(kind="rembg", resource="rembg_native_model", value=path.name)
    from ruinedfooocus.features.image_processing.native_vision import native_model_choices

    if str(path) not in {value for _, value in native_model_choices(manager, "depth-map")}:
        return None
    return dict(kind="depth-map", resource="native_vision_model", value=str(path))


def activation_values(kind, resource=None, *, presets, custom, features=None):
    """Return one preset/type/resource transaction for Hub and Library callers."""
    from ruinedfooocus.features.image_processing.image_composition import handles
    from ruinedfooocus.features.image_processing.still_image_utilities import (
        ADJUSTMENT_KINDS,
        ADJUSTMENT_LABEL,
    )

    if kind in ADJUSTMENT_KINDS:
        if resource is not None or ADJUSTMENT_LABEL not in presets:
            raise ValueError("Choose the built-in adjustment without a model resource.")
        return dict(
            cn_selection=ADJUSTMENT_LABEL,
            cn_type=kind.capitalize(),
            automatic_upscale_target="Generated image",
        )

    if kind not in (
        "img2img",
        "edit",
        "upscale",
        "rembg",
        "depth-map",
        "detect",
        "segment",
        "restore-seedvr2",
        "wevisone-detect",
        "wevisone-matte",
    ) and not handles(kind):
        raise ValueError("This image tool has no supported activation route.")
    if features is not None and kind == "edit" and not features.get("native_edit"):
        raise ValueError("This image tool is not supported by the selected checkpoint.")
    selection = next(
        (
            name
            for name, options in presets.items()
            if options.get("type") == kind
            and name != "Image to image (preserve details)"
            and kind != "upscale"
        ),
        None,
    )
    native_utility = kind in (
        "depth-map",
        "detect",
        "segment",
        "restore-seedvr2",
        "wevisone-detect",
        "wevisone-matte",
    ) or handles(kind)
    if native_utility and selection is None:
        raise ValueError("This image tool has no saved activation route.")
    values = dict(
        cn_selection=selection or custom,
        cn_type=kind.capitalize(),
        automatic_upscale_target="Generated image",
    )
    if native_utility:
        # Standalone utility presets are intentionally absent from custom type choices.
        values.pop("cn_type")
    if resource is not None:
        expected = {
            "upscale": "cn_upscale",
            "rembg": "rembg_native_model",
            "depth-map": "native_vision_model",
            "detect": "native_vision_model",
            "segment": "native_vision_model",
            "restore-seedvr2": "seedvr2_model",
            "wevisone-detect": "wevisone_model",
            "wevisone-matte": "wevisone_model",
        }.get(kind)
        if resource.get("kind") != kind or resource.get("resource") != expected:
            raise ValueError("The selected resource does not support this image tool.")
        values[expected] = resource["value"]
    return values
