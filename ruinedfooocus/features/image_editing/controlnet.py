from ruinedfooocus.core.paths import data_path, resource_path
import os
import shutil
import json
from contextlib import contextmanager
from ruinedfooocus.app.state import path_manager
from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.features.image_processing.still_image_utilities import KINDS as DEPTH_UTILITIES

DEFAULT_CNSETTINGS_FILE = resource_path("defaults", "powerup.default")
CNSETTINGS_FILE = data_path("settings", "powerup.json")
NEWCN = "Custom..."

# https://huggingface.co/stabilityai/control-lora/tree/main/control-LoRAs-rank128
from ruinedfooocus.features.generation.image_preparation import (
    CONTROL_MODELS,
    prepared_asset,
    KEY,
    ImageAssets,
)

controlnet_models = {
    **CONTROL_MODELS,
    "img2img": None,
    "edit": None,
    "control": None,
    "qwen-union": None,
    "upscale": None,
    "rembg": None,
    "detect": None,
    "segment": None,
    "depth-map": None,
    "restore-seedvr2": None,
    "image-grid": None,
    "image-composite": None,
    **dict.fromkeys(DEPTH_UTILITIES),
    "latent-compose": None,
}


def load_cnsettings():
    settings = {}
    if not os.path.isfile(CNSETTINGS_FILE):
        shutil.copy(DEFAULT_CNSETTINGS_FILE, CNSETTINGS_FILE)

    if os.path.exists(CNSETTINGS_FILE):
        with open(CNSETTINGS_FILE) as f:
            settings.update(json.load(f))

    with open(DEFAULT_CNSETTINGS_FILE) as f:
        default_settings = json.load(f)

    # Update settings with any missing keys from default_settings
    settings_updated = False
    # The old stock preset caused large redraws. Keep customized presets intact.
    if settings.get("Img2Img") == {"type": "img2img", "start": 0.06, "denoise": 0.64}:
        settings["Img2Img"] = dict(default_settings["Img2Img"])
        settings_updated = True
    for key in ("Depth (low)", "Depth (high)"):
        updated = default_settings.get(key, {})
        previous = {key: value for key, value in updated.items() if key != "depth_preprocess"}
        if updated and settings.get(key) == previous:
            settings[key] = dict(updated)
            settings_updated = True
    for key, value in default_settings.items():
        if key not in settings:
            settings[key] = value
            settings_updated = True

    # If settings were updated, write them back to the file
    if settings_updated:
        save_json(CNSETTINGS_FILE, settings)

    # Ignore unknown settings
    for key in list(settings.keys()):
        if settings[key]["type"] not in controlnet_models:
            del settings[key]

    return settings


def save_cnsettings(cn_save_options):
    global controlnet_settings, cn_options
    save_json(CNSETTINGS_FILE, cn_save_options)
    controlnet_settings = cn_save_options
    cn_options = {f"{k}": v for k, v in controlnet_settings.items()}


@contextmanager
def model_asset(type, gen_data=None):
    explicit = get_settings(gen_data or {}).get("model")
    if explicit and not isinstance((gen_data or {}).get(KEY), ImageAssets):
        raise ValueError("Capture the selected control model before native loading.")
    if (
        type not in controlnet_models
        or controlnet_models[type] is None
        and not (type == "control" and explicit)
    ):
        yield None
        return
    with prepared_asset(gen_data, "controlnet", type, path_manager, "controlnet") as filename:
        yield filename


def get_settings(gen_data):
    from ruinedfooocus.features.image_processing.still_image_utilities import (
        KINDS as DEPTH_UTILITIES,
    )

    explicit = gen_data.get("controlnet")
    from ruinedfooocus.features.image_processing.still_image_utilities import adjustment_kind

    adjustment = adjustment_kind(gen_data.get("cn_selection"), gen_data.get("cn_type"))
    if explicit is None and adjustment is not None:
        return {"type": adjustment}
    if explicit is None:
        selected = controlnet_settings.get(gen_data.get("cn_selection"), {})
    else:
        selected = explicit
    from ruinedfooocus.features.image_processing.image_composition import handles

    if selected.get("type") in (
        "depth-map",
        "restore-seedvr2",
        "latent-compose",
        *DEPTH_UTILITIES,
    ) or handles(selected.get("type")):
        return {"type": selected["type"]}
    if gen_data.get("automatic_upscale_target") == "Input image only":
        from ruinedfooocus.features.generation.automatic_upscale import model_for

        model = model_for(gen_data.get("automatic_upscale"))
        if model is None:
            raise ValueError("Choose an upscale model before upscaling the input image.")
        return {"type": "upscale", "upscaler": model}
    if explicit is not None:
        return dict(explicit)
    if "cn_selection" not in gen_data:
        return {}
    if gen_data["cn_selection"] == NEWCN:
        options = {
            "type": gen_data["cn_type"].lower(),
            "edge_low": gen_data["cn_edge_low"],
            "edge_high": gen_data["cn_edge_high"],
            "start": gen_data["cn_start"],
            "stop": gen_data["cn_stop"],
            "strength": gen_data["cn_strength"],
            "upscaler": gen_data["cn_upscale"],
        }
    else:
        options = dict(controlnet_settings.get(gen_data["cn_selection"], {}))
        if options.get("type") == "img2img" and "cn_strength" in gen_data:
            options["denoise"] = gen_data["cn_strength"]
        elif options.get("type") in ("canny", "depth", "sketch", "recolour", "control"):
            for key in ("strength", "start", "stop"):
                if "cn_" + key in gen_data:
                    options[key] = gen_data["cn_" + key]
            if options["type"] == "canny":
                for key in ("edge_low", "edge_high"):
                    if "cn_" + key in gen_data:
                        options[key] = gen_data["cn_" + key]
        elif options.get("type") == "upscale" and "cn_upscale" in gen_data:
            options["upscaler"] = gen_data["cn_upscale"]
    if options.get("type") == "qwen-union":
        from ruinedfooocus.features.image_editing.qwen_image21_union import settings

        options.update(
            (key, gen_data[field])
            for key, field in (
                ("model", "qwen_union_model"),
                ("model_signature", "qwen_union_model_signature"),
                ("mode", "qwen_union_mode"),
                ("strength", "cn_strength"),
                ("start", "cn_start"),
                ("stop", "cn_stop"),
            )
            if field in gen_data
        )
        return settings(options)
    if options.get("type") in ("canny", "depth", "sketch", "recolour", "control"):
        if "cn_model" in gen_data:
            options["model"] = gen_data["cn_model"]
            options["model_signature"] = gen_data.get("cn_model_signature")
    if options.get("type") == "depth":
        from ruinedfooocus.features.image_editing.depth_control import (
            DEPTH_DEFAULTS,
            depth_settings,
        )

        options.update(
            depth_settings(
                {
                    **options,
                    **{
                        key: gen_data["cn_" + key]
                        for key in DEPTH_DEFAULTS
                        if "cn_" + key in gen_data
                    },
                }
            )
        )
    return options


controlnet_settings = load_cnsettings()
cn_options = {f"{k}": v for k, v in controlnet_settings.items()}
