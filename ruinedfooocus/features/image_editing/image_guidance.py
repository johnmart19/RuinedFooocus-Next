"""GPU-free image guidance schema and request-scoped native model patches."""

import math
import re
from ruinedfooocus.features.models.image_components import IMAGE_COMPONENTS, ALIASES

MOSS_EDIT_FORMATS = ("Disabled", "ai-toolkit", "ai-toolkit (kv_cache off)", "Moss-Image-Edit")
STARTUP_SHIFT_LABELS = {
    "auraflow_shift": "AuraFlow",
    "lumina2_shift": "Lumina 2 / Z Image",
    "hidream_shift": "HiDream",
    "sd3_shift": "SD3",
    "qwen_image_shift": "Qwen Image",
    "newbieimage_shift": "NewBieImage",
    "anima_image_shift": "Anima",
}
FIELD_SCHEMA = {
    "qwen_image21_schedule": {"choices": ("auto", "standard", "turbo_8"), "default": "auto"},
    "moss_image_edit_format": {"choices": MOSS_EDIT_FORMATS, "default": "Disabled"},
    "moss_image_edit_strength": {"minimum": 0.0, "maximum": 100.0, "step": 0.05, "default": 1.0},
    "moss_image_slg": {"default": False},
    "moss_image_slg_double_layers": {"default": ""},
    "moss_image_slg_single_layers": {"default": "22"},
    "moss_image_slg_start": {"minimum": 0.0, "maximum": 1.0, "step": 0.001, "default": 0.0},
    "moss_image_slg_end": {"minimum": 0.0, "maximum": 1.0, "step": 0.001, "default": 0.5},
    "sampling_shift": {"minimum": 0.01, "maximum": 100.0, "step": 0.01, "default": None},
    "cfg_norm_strength": {"minimum": 0.0, "maximum": 100.0, "step": 0.01, "default": None},
    "cfg_norm_auto": {"default": False},
    "cfg_norm_pre_cfg": {"default": False},
    "hidream_o1_noise_scale": {"minimum": 0.0, "maximum": 64.0, "step": 0.01, "default": None},
    "hidream_o1_seam_smoothing": {"default": None},
    "hidream_o1_seam_start": {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "default": 0.8},
    "hidream_o1_seam_end": {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "default": 1.0},
    "hidream_o1_seam_pattern": {
        "choices": ("single_shift", "symmetric"),
        "default": "single_shift",
    },
    "hidream_o1_seam_passes": {
        "choices": ("2", "4", "ramp_2_4", "ramp_2_4_8"),
        "default": "ramp_2_4",
    },
    "hidream_o1_seam_blend": {"choices": ("average", "window", "median"), "default": "median"},
    "hidream_o1_seam_strength": {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "default": 1.0},
    "lcm_noise": {"minimum": 0.0, "maximum": 64.0, "step": 0.01, "default": None},
    "lcm_noise_end": {"minimum": 0.0, "maximum": 64.0, "step": 0.01, "default": None},
    "lcm_noise_clip": {"minimum": 0.0, "maximum": 10.0, "step": 0.01, "default": None},
    "image_resize_mode": {"choices": ("preserve", "pad", "crop"), "default": "preserve"},
    "vae_tiled": {"default": False},
    "vae_tile_size": {"minimum": 64, "maximum": 4096, "step": 32, "default": None},
    "vae_overlap": {"minimum": 0, "maximum": 4096, "step": 32, "default": None},
}
GUIDANCE_RANGES = (
    ("moss_image_slg_start", "moss_image_slg_end", "SLG"),
    ("hidream_o1_seam_start", "hidream_o1_seam_end", "Seam smoothing"),
)
SHIFT_FAMILIES = {
    "ChromaRadiance": "AuraFlow",
    "Anima": "AuraFlow",
    "AuraFlow": "AuraFlow",
    "Lumina2": "AuraFlow",
    "ZImage": "AuraFlow",
    "HiDream": "SD3",
    "SD3": "SD3",
    "NewBieImage": "AuraFlow",
    "QwenImage": "AuraFlow",
    "MOSSImage": "AuraFlow",
    "MageFlow": "AuraFlow",
    "MingImage": "AuraFlow",
}


def capabilities(family):
    family = ALIASES.get(family, family)
    if family == "BerniniImage":
        return {
            "qwen_image21": False,
            "moss_edit": False,
            "moss_slg": False,
            "hidream_o1": False,
            "sampling_shift": False,
            "cfg_norm": False,
            "multiple_refs": True,
        }
    return {
        "qwen_image21": family == "QwenImage21",
        "moss_edit": family == "MOSSImage",
        "moss_slg": family == "MOSSImage",
        "hidream_o1": family == "HiDreamO1",
        "sampling_shift": family in SHIFT_FAMILIES,
        "cfg_norm": family in IMAGE_COMPONENTS,
        "multiple_refs": family
        in (
            "Boogu",
            "Krea2",
            "MiniMaxH3Image",
            "MOSSImage",
            "MageFlow",
            "Fibo2",
            "HiDreamO1",
            "QwenImage",
            "QwenImage21",
            "Flux2",
            "Flux2Klein4B",
            "Flux2Klein9B",
        ),
    }


def normalize_reference_images(references, primary, family, edit_format="Disabled"):
    """Accept PIL images/Gallery entries, never paths or remote URLs."""
    from PIL import Image, ImageOps

    if references is None:
        return []
    if not isinstance(references, (list, tuple)) or len(references) > 15:
        raise ValueError("Use at most 15 additional reference images.")
    if not references:
        return []
    family = ALIASES.get(family, family)
    if family in ("QwenImage", "Krea2") and len(references) > 2:
        raise ValueError("This native recipe accepts at most two additional references.")
    if primary is None:
        raise ValueError("Additional references require a primary input image.")
    if not isinstance(primary, Image.Image):
        raise ValueError("The primary reference must be an image.")
    if not capabilities(family)["multiple_refs"] or (
        family == "MOSSImage" and edit_format == "Disabled"
    ):
        raise ValueError("Additional references require a supported native edit workflow.")
    result = []
    for entry in references:
        image = entry[0] if isinstance(entry, (tuple, list)) and len(entry) == 2 else entry
        if (
            not isinstance(image, Image.Image)
            or min(image.size) <= 0
            or image.width * image.height > 16777216
        ):
            raise ValueError("Each additional reference must be an image of at most 16 megapixels.")
        try:
            result.append(
                ImageOps.exif_transpose(image).convert(
                    "RGBA" if family in ("QwenImage21", "MiniMaxH3Image") else "RGB"
                )
            )
        except (OSError, ValueError) as error:
            raise ValueError("Invalid additional reference image.") from error
    return result


def reference_image_map(primary, references, transform):
    return {
        "image_1": primary,
        **{f"image_{index}": transform(image) for index, image in enumerate(references, start=2)},
    }


def validate_guidance(values):
    result = {key: value for key, value in values.items() if key in FIELD_SCHEMA}
    for key, value in result.items():
        if key == "qwen_image21_schedule":
            if value not in FIELD_SCHEMA[key]["choices"]:
                raise ValueError("Choose Auto, Standard or Turbo 8-step for the Qwen schedule.")
        elif key == "moss_image_edit_format":
            if value not in MOSS_EDIT_FORMATS:
                raise ValueError("Choose a supported MOSS Image edit format.")
        elif key == "image_resize_mode":
            if value not in FIELD_SCHEMA[key]["choices"]:
                raise ValueError(
                    "Choose Preserve proportions, Fit selected size or Crop to selected size."
                )
        elif key == "hidream_o1_seam_smoothing":
            if isinstance(value, str) and value in ("auto", "on", "off"):
                value = {"auto": None, "on": True, "off": False}[value]
            if value is not None and type(value) is not bool:
                raise ValueError("Seam smoothing: choose Auto, On or Off.")
            result[key] = value
        elif key in ("moss_image_slg", "cfg_norm_pre_cfg", "cfg_norm_auto", "vae_tiled"):
            if type(value) is not bool:
                raise ValueError(f"{key}: choose on or off.")
        elif key.startswith("hidream_o1_") and "choices" in FIELD_SCHEMA[key]:
            if value not in FIELD_SCHEMA[key]["choices"]:
                raise ValueError(f"{key}: choose a supported option.")
        elif key in ("moss_image_slg_double_layers", "moss_image_slg_single_layers"):
            if (
                not isinstance(value, str)
                or len(value) > 1024
                or not re.fullmatch(r"\s*(?:[0-9]+\s*(?:,\s*[0-9]+\s*)*)?", value)
            ):
                raise ValueError("SLG layers: enter comma-separated block numbers, or leave blank.")
            result[key] = value.strip()
        else:
            if FIELD_SCHEMA[key]["default"] is None and value in (None, ""):
                result[key] = None
                continue
            if isinstance(value, bool):
                raise ValueError(f"{key}: enter a finite number.")
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key}: enter a finite number.") from None
            spec = FIELD_SCHEMA[key]
            if not math.isfinite(number) or not spec["minimum"] <= number <= spec["maximum"]:
                raise ValueError(
                    f"{key}: value must be between {spec['minimum']} and {spec['maximum']}."
                )
            if key in ("vae_tile_size", "vae_overlap"):
                if not number.is_integer() or number % 32:
                    raise ValueError("VAE tile size and overlap must be whole multiples of 32.")
                number = int(number)
            result[key] = number
    for start, end, label in GUIDANCE_RANGES:
        if start in result and end in result and result[start] > result[end]:
            raise ValueError(f"{label}: start must not exceed end.")
    return result


def resolve_guidance(request, defaults, family, checkpoint=None):
    """Missing fields inherit defaults; explicit None/blank means Auto or Off."""
    family = ALIASES.get(family, family)
    values = {
        key: request[key] if key in request else defaults.get(key, spec["default"])
        for key, spec in FIELD_SCHEMA.items()
    }
    if family != "QwenImage21":
        values["qwen_image21_schedule"] = "auto"
    if family != "MOSSImage":
        values.update(
            {
                key: spec["default"]
                for key, spec in FIELD_SCHEMA.items()
                if key.startswith("moss_image_")
            }
        )
    if family != "HiDreamO1":
        values.update(
            {
                key: spec["default"]
                for key, spec in FIELD_SCHEMA.items()
                if key.startswith("hidream_o1_")
            }
        )
        values["hidream_o1_seam_smoothing"] = False
    values = validate_guidance(values)
    recipe = request.get("_model_default_guidance") or {}
    if family == "Krea2" and request.get("_native_krea_style") is not None:
        values["_model_sampling_flux"] = dict(
            max_shift=1.15, base_shift=0.5, width=request["width"], height=request["height"]
        )
    if family == "Lens" and values["cfg_norm_auto"] and values["cfg_norm_strength"] is None:
        strength = recipe.get("cfg_norm_strength")
        if strength is not None:
            values["cfg_norm_strength"] = validate_guidance({"cfg_norm_strength": strength})[
                "cfg_norm_strength"
            ]
    if (
        family == "MingImage"
        and values["sampling_shift"] is None
        and recipe.get("model_sampling_flux")
    ):
        width, height = request.get("width"), request.get("height")
        if any(type(side) is not int or side <= 0 for side in (width, height)):
            raise ValueError(
                "Ming Image's model recipe requires the actual target width and height."
            )
        values["_model_sampling_flux"] = dict(
            recipe["model_sampling_flux"], width=width, height=height
        )
    if family == "HiDreamO1" and values["hidream_o1_seam_smoothing"] is None:
        from ruinedfooocus.features.generation.image_sampling import hidream_o1_dev

        values["hidream_o1_seam_smoothing"] = not hidream_o1_dev(checkpoint)
    if values["sampling_shift"] is not None and family not in SHIFT_FAMILIES:
        raise ValueError("Sampling shift is not supported by this image model family.")
    return values


def request_model(base, values, family, nodes=None, checkpoint=None):
    """Always start from a clone of the cached, unmodified LoRA-patched model."""
    family = ALIASES.get(family, family)
    model = base.clone()
    if nodes is None:
        from comfy_extras.nodes_model_advanced import (
            ModelSamplingAuraFlow,
            ModelSamplingSD3,
            ModelSamplingFlux,
            ModelNoiseScale,
        )
        from comfy_extras.nodes_hidream_o1 import HiDreamO1PatchSeamSmoothing
        from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
        from comfy_extras.nodes_cfg import CFGNorm

        nodes = {
            "AuraFlow": ModelSamplingAuraFlow().patch_aura,
            "SD3": ModelSamplingSD3().patch,
            "Flux": ModelSamplingFlux().patch,
            "slg": SkipLayerGuidanceDiTSimple.execute,
            "cfg_norm": CFGNorm.execute,
            "noise_scale": ModelNoiseScale().patch,
            "seam": HiDreamO1PatchSeamSmoothing.execute,
        }
    if family == "HiDreamO1":
        from ruinedfooocus.features.generation.image_sampling import hidream_o1_dev

        noise_scale = values["hidream_o1_noise_scale"]
        if noise_scale is None and hidream_o1_dev(checkpoint):
            # Comfy-Org/workflow_templates: image_hidream_o1_dev, ModelNoiseScale.
            noise_scale = 7.6
        if noise_scale is not None:
            model = nodes["noise_scale"](model=model, noise_scale=noise_scale)[0]
        if values["hidream_o1_seam_smoothing"]:
            model = nodes["seam"](
                model=model,
                **{
                    arg: values["hidream_o1_seam_" + field]
                    for arg, field in (
                        ("start_percent", "start"),
                        ("end_percent", "end"),
                        ("pattern", "pattern"),
                        ("passes", "passes"),
                        ("blend", "blend"),
                        ("strength", "strength"),
                    )
                },
            )[0]
    if values["sampling_shift"] is not None:
        model = nodes[SHIFT_FAMILIES[family]](model=model, shift=values["sampling_shift"])[0]
    elif family in ("MingImage", "Krea2") and values.get("_model_sampling_flux"):
        model = nodes["Flux"](model=model, **values["_model_sampling_flux"])[0]
    if values["moss_image_slg"]:
        model = nodes["slg"](
            model=model,
            double_layers=values["moss_image_slg_double_layers"],
            single_layers=values["moss_image_slg_single_layers"],
            start_percent=values["moss_image_slg_start"],
            end_percent=values["moss_image_slg_end"],
        )[0]
    if values["cfg_norm_strength"] is not None:
        model = nodes["cfg_norm"](
            model=model, strength=values["cfg_norm_strength"], pre_cfg=values["cfg_norm_pre_cfg"]
        )[0]
    return model


def effective_guidance(values, model, family):
    result = dict(values)
    result["sampling_shift_supported"] = capabilities(family)["sampling_shift"]
    sampling = model.get_model_object("model_sampling")
    shift = getattr(sampling, "shift", None)
    noise = getattr(sampling, "noise_scale", None)
    result["effective_hidream_o1_noise_scale"] = (
        float(noise)
        if capabilities(family)["hidream_o1"]
        and isinstance(noise, (int, float))
        and math.isfinite(noise)
        else None
    )
    result["effective_sampling_shift"] = (
        float(shift) if isinstance(shift, (int, float)) and math.isfinite(shift) else None
    )
    sampling_types = {
        base.__name__
        for base in type(sampling).__mro__
        if base.__module__ == "comfy.model_sampling"
    }
    result["effective_sampling_shift_mode"] = next(
        (
            mode
            for name, mode in (("ModelSamplingDiscreteFlow", "flow"), ("ModelSamplingFlux", "flux"))
            if name in sampling_types
        ),
        None,
    )
    return result


def restore_guidance_metadata(metadata, family=None):
    """Restore only validated scalar guidance, never reference pixels or paths.

    Flat fields take precedence over grouped metadata. Replay effective flow
    shifts only in the override node's units. Native exponential shifts stay Auto.
    Legacy metadata without guidance contributes no overrides.
    """
    if not isinstance(metadata, dict):
        return {}
    group = metadata.get("image_guidance", {})
    group = group if isinstance(group, dict) else {}
    recorded = {
        key: metadata[key] if key in metadata else group[key]
        for key in FIELD_SCHEMA
        if key in metadata or key in group
    }
    restored = {}
    for key, value in recorded.items():
        try:
            restored.update(validate_guidance({key: value}))
        except ValueError:
            continue  # Invalid metadata is not a trusted generation override.
    if restored.get("qwen_image21_schedule", "auto") == "auto" and (
        family is None or capabilities(family)["qwen_image21"]
    ):
        from ruinedfooocus.features.models.model_recommendations import QWEN_IMAGE21_TURBO_SIGMAS

        sigmas = metadata.get("effective_sampling_sigmas", group.get("effective_sampling_sigmas"))
        if (
            isinstance(sigmas, list)
            and all(type(value) in (int, float) for value in sigmas)
            and sigmas == list(QWEN_IMAGE21_TURBO_SIGMAS)
        ):
            restored["qwen_image21_schedule"] = "turbo_8"
    for start, end, label in GUIDANCE_RANGES:
        try:
            validate_guidance(
                {key: restored.get(key, FIELD_SCHEMA[key]["default"]) for key in (start, end)}
            )
        except ValueError:
            restored.pop(start, None)
            restored.pop(end, None)
    from ruinedfooocus.features.generation.image_sampling import LCM_FIELDS

    for key in LCM_FIELDS:
        actual = metadata.get("effective_" + key, group.get("effective_" + key))
        if recorded.get(key) in (None, "") and actual is not None:
            try:
                restored.update(validate_guidance({key: actual}))
            except ValueError:
                pass
    if recorded.get("hidream_o1_noise_scale") in (None, "") and (
        family is None or capabilities(family)["hidream_o1"]
    ):
        actual = metadata.get(
            "effective_hidream_o1_noise_scale", group.get("effective_hidream_o1_noise_scale")
        )
        if actual is not None:
            try:
                restored.update(validate_guidance({"hidream_o1_noise_scale": actual}))
            except ValueError:
                pass
    supported = metadata.get("sampling_shift_supported", group.get("sampling_shift_supported"))
    mode = metadata.get("effective_sampling_shift_mode", group.get("effective_sampling_shift_mode"))
    # Qwen's native Flux sampler uses exponential shift; the AuraFlow override
    # uses a direct ratio. Untyped older Qwen Auto metadata is ambiguous too.
    compatible = mode == "flow" or mode is None and ALIASES.get(family, family) != "QwenImage"
    can_promote = (
        compatible
        and supported is not False
        and (family is None or capabilities(family)["sampling_shift"])
    )
    if recorded.get("sampling_shift") in (None, "") and can_promote:
        actual = metadata.get("effective_sampling_shift", group.get("effective_sampling_shift"))
        if actual not in (None, ""):
            try:
                restored.update(validate_guidance({"sampling_shift": actual}))
            except ValueError:
                pass
    return restored


def read_preset_metadata(filename):
    if not filename:
        return {}
    import json
    from PIL import Image

    try:
        with Image.open(filename) as image:
            metadata = json.loads(image.info.get("parameters", ""))
        if not isinstance(metadata, dict) or metadata.get("software") != "RuinedFooocus":
            return None
        return metadata
    except (OSError, ValueError, TypeError):
        return None


def effective_preset_checkpoint(checkpoint, preset, mode):
    metadata = read_preset_metadata(preset)
    if metadata is None:
        return None
    selected = metadata.get("base_model_name", checkpoint) if preset and mode & 4 else checkpoint
    return selected if isinstance(selected, str) and selected else None
