"""Shared Create/Library UI request construction before worker admission."""

import json

from PIL import Image


def build_ui_request(values, *, models, settings, preset_mode=7, family_for=None):
    """Resolve UI-only input and PNG preset intent without enqueueing any work."""
    from ruinedfooocus.features.image_editing import controlnet
    from ruinedfooocus.features.image_processing.still_image_utilities import is_utility
    from ruinedfooocus.features.image_processing.image_composition import handles

    data = dict(values)
    kind = controlnet.get_settings(data).get("type")
    if handles(kind):
        data.update(generate_forever=False, image_number=1, preset_selection=None)
        return data
    depth_map = kind in ("depth-map", "restore-seedvr2") or is_utility(kind)
    if (
        not depth_map
        and data.get("automatic_upscale", "Off") != "Off"
        and data.get("automatic_upscale_target") == "Input image only"
        and (data.get("controlnet") or {}).get("type") != "upscale"
    ):
        data["input_image"] = data.get("upscale_upload")
    data["generate_forever"] = int(data.get("image_number", 1)) == 0
    if depth_map or not data.get("preset_selection"):
        return data
    with Image.open(data["preset_selection"]) as image:
        saved = json.loads(image.info.get("parameters", ""))
    if saved.get("software") != "RuinedFooocus":
        return data
    if preset_mode & 0b100:
        from ruinedfooocus.features.models.component_profiles import apply_preset_choices

        data.update(
            base_model_name=saved["base_model_name"],
            base_model_hash=saved.get("base_model_hash"),
            loras=saved["loras"],
        )
        data = apply_preset_choices(data, saved, models, settings)
        from ruinedfooocus.features.image_editing.native_auxiliary_assets import (
            replay_values as replay_auxiliary,
        )

        data.update(replay_auxiliary(saved))
    if preset_mode & 0b010:
        from ruinedfooocus.features.image_editing.image_guidance import restore_guidance_metadata

        family = (
            family_for(data["base_model_name"])
            if family_for is not None
            else models.get_model_base(
                models.get_models_by_path("checkpoints", data["base_model_name"], fetch=False)
            )
        )
        for key in ("steps", "cfg", "sampler_name", "scheduler", "clip_skip"):
            data[key] = saved[key]
        data.update(restore_guidance_metadata(saved, family=family))
    if preset_mode & 0b001:
        from ruinedfooocus.features.generation.size_plan import replay_size

        data["width"], data["height"] = replay_size(saved)
        data["size_mode"] = "target" if "size_plan" in saved else "legacy_base"
        data["automatic_upscale"] = saved.get("automatic_upscale", "Off")
        data["automatic_upscale_target"] = "Generated image"
    return data
