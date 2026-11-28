"""Passive defaults for the ready-to-run native still-image tools.

These are preferences among existing catalogue entries, not new download or
model contracts. Capture and preparation retain their normal role validation.
"""

from pathlib import Path

# Prefer the smaller supported variant when no compatible local file exists.
PREFERENCES = {
    "depth-map": {
        "native_vision_model": (
            "geometry_estimation/depth_anything_3_small.safetensors",
            ("DepthAnything3", "DA3-SMALL"),
            ("depth_estimation",),
        ),
    },
    "detect": {
        "native_vision_model": (
            "diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors",
            ("RT_DETR_v4",),
            ("image_utility_model",),
        ),
    },
    "segment": {
        "native_vision_model": (
            "checkpoints/sam3.1_multiplex_fp16.safetensors",
            ("SAM31",),
            ("image_utility_model",),
        ),
    },
    "restore-seedvr2": {
        "seedvr2_model": (
            "diffusion_models/seedvr2_3b_int8_convrot.safetensors",
            ("SeedVR2",),
            ("image_utility_model",),
        ),
        "seedvr2_vae": (
            "vae/seedvr2_ema_vae_fp16.safetensors",
            ("SeedVR2 codec",),
            ("image_codec",),
        ),
    },
    "face-landmarks": {
        "image_utility_model": (
            "detection/mediapipe_face_fp32.safetensors",
            ("FaceLandmarker",),
            ("image_utility_model",),
        ),
    },
    "pose-map": {
        "image_utility_model": (
            "checkpoints/sdpose_wholebody_fp16.safetensors",
            ("SDPose",),
            ("image_utility_model",),
        ),
    },
}


def component_choices(manager, kind, field, *, locale=None):
    """Return validated local choices, followed by a trusted first-use choice.

    No runtime initialization, acquisition or model-weight reads happen here.
    A missing/changed catalogue contract simply supplies no automatic choice.
    """
    spec = PREFERENCES.get(kind, {}).get(field)
    if spec is None:
        return []
    if kind == "restore-seedvr2":
        from ruinedfooocus.features.image_processing.native_seedvr2 import model_choices

        values = model_choices(manager, codec=field == "seedvr2_vae")
    elif kind in ("depth-map", "detect", "segment"):
        from ruinedfooocus.features.image_processing.native_vision import native_model_choices

        values = native_model_choices(manager, kind)
    else:
        from ruinedfooocus.features.image_processing.depth_utilities import (
            component_choices as local_choices,
        )

        values = local_choices(manager, kind, field, locale=locale)
    # The face picker also includes a first-use row. Recheck its source below,
    # and always rank real compatible local files before a missing destination.
    values = [row for row in values if Path(row[1]).is_file()]
    key, identities, source_roles = spec
    entry = getattr(manager, "DOWNLOADABLE_FILES", {}).get(key)
    if entry is None:
        return values
    from ruinedfooocus.features.generation.image_preparation import selected_image_artifact
    from ruinedfooocus.core.translation_manager import localize

    category = entry.get("path", "").removeprefix("path_")
    try:
        item = selected_image_artifact(
            manager, key, field, (category,), identities=identities, source_roles=source_roles
        )
        path = item.request.destination
        # A present file must have passed the native header validator above;
        # do not relabel a corrupt or incompatible local file as downloadable.
        if not path.exists() and str(path) not in {value for _, value in values}:
            values.append(
                (
                    localize("Download on first use: {name}", locale, name=path.name),
                    str(path),
                )
            )
    except (OSError, ValueError, TypeError):
        pass
    return values


def default_components(manager, kind, values=None):
    """Fill blank component selections, preserving every explicit selection.

    UI callers can use the same values as capture, including while offline:
    the existing readiness/preparation path explains missing files on Run.
    """
    result = dict(values or {})
    for field in PREFERENCES.get(kind, {}):
        selected = result.get(field)
        if selected is None or isinstance(selected, str) and not selected.strip():
            choices = component_choices(manager, kind, field)
            if choices:
                result[field] = choices[0][1]
    return result


def picker_value(manager, kind, field, selected, choices):
    """Choose a tool-compatible UI default; retain unknown missing saved paths.

    Switching a shared selector from another known tool must not carry that
    tool's model into the new action. Capture itself never replaces explicit
    selections and remains the authoritative validation boundary.
    """
    if field not in PREFERENCES.get(kind, {}):
        return selected
    if selected and selected not in {value for _, value in choices}:
        known_other = False
        for other, fields in PREFERENCES.items():
            if other == kind or field not in fields:
                continue
            key = fields[field][0]
            if key in getattr(manager, "DOWNLOADABLE_FILES", {}):
                try:
                    known_other |= str(manager.preparation_request(key).destination) == selected
                except (OSError, ValueError, TypeError):
                    continue
        if Path(selected).is_file() or known_other:
            selected = None
    return selected or (choices[0][1] if choices else None)
