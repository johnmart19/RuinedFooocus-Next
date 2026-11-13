"""Replay a typed still utility from a selected original, without a checkpoint."""

import json
from pathlib import Path
import re

from ruinedfooocus.features.image_processing import native_seedvr2 as native


def restoration_metadata(value):
    if isinstance(value, str):
        if len(value) > 65536:
            return None
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            return None
    if not isinstance(value, dict):
        return None
    control, source = value.get("controlnet"), value.get("native_seedvr2_source")
    if (
        not isinstance(control, dict)
        or control.get("type") != native.KIND
        or not isinstance(source, dict)
        or source.get("native") != "SeedVR2"
    ):
        return None
    return value


def replay_settings(value, manager):
    from ruinedfooocus.core.cache_contract import configured_cache_root
    from ruinedfooocus.features.models.model_identity import cached_file_sha256

    metadata = restoration_metadata(value)
    if metadata is None:
        raise ValueError("Select an image with saved SeedVR2 restoration settings.")
    source, values = metadata["native_seedvr2_source"], {}
    for field, key, codec in (("seedvr2_model", "model", False), ("seedvr2_vae", "codec", True)):
        digest, size = source.get(key + "_sha256"), source.get(key + "_bytes")
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[a-fA-F0-9]{64}", digest)
            or type(size) is not int
            or size < 1
        ):
            raise ValueError(
                "The saved restoration has no verified component identity. Select its model and codec manually."
            )
        matches = [
            path
            for _, path in native.model_choices(manager, codec)
            if Path(path).stat().st_size == size
            and cached_file_sha256(path, configured_cache_root(manager)).lower() == digest.lower()
        ]
        if len(matches) != 1:
            raise ValueError(
                "The saved SeedVR2 component cannot be uniquely matched. Select its model and codec manually."
            )
        values[field] = matches[0]
    scale, color = (
        metadata.get("seedvr2_scale", source.get("scale")),
        metadata.get("seedvr2_color", source.get("color")),
    )
    native.geometry((2, 2), scale)
    if color not in native.COLORS:
        raise ValueError("The saved SeedVR2 color correction is unavailable.")
    seed = metadata.get("seed", 0)
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("Invalid saved SeedVR2 seed.")
    values.update(seedvr2_scale=scale, seedvr2_color=color, seed=seed, seed_random=False)
    return values


def prepare_record(record, metadata, manager, *, presets, custom, resets):
    from ruinedfooocus.features.gallery.editing_workflow import working_source
    from ruinedfooocus.features.image_editing.tool_activation import activation_values

    values = replay_settings(metadata, manager)
    image, details = working_source(record)
    native.geometry(image.size, values["seedvr2_scale"])
    values = dict(resets) | values
    values.update(
        activation_values(native.KIND, presets=presets, custom=custom),
        input_image=image.copy(),
        main_view=image.copy(),
        image_source=details,
        preset_selection=None,
        inpaint_toggle=False,
        automatic_upscale="Off",
    )
    return values
