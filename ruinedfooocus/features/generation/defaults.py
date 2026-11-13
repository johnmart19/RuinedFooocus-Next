"""Save deliberate startup choices without storing uploads or generation state."""

import copy
import math

from ruinedfooocus.features.image_processing.depth_utilities import EXTRA_UTILITY_DEFAULTS
from ruinedfooocus.features.image_processing import latent_composition
from ruinedfooocus.features.generation.bernini_image import DEFAULTS as BERNINI_DEFAULTS
from ruinedfooocus.features.image_editing import model_patches, native_control_recipes
from ruinedfooocus.features.image_editing import lora_hooks
from ruinedfooocus.features.image_editing import krea_style
from ruinedfooocus.features.image_processing import wevisone
from ruinedfooocus.features.image_processing.still_image_utilities import ADJUSTMENT_DEFAULTS
from ruinedfooocus.backends.llama.moe_cache import DEFAULTS as MOE_CACHE_DEFAULTS

# Fresh profiles are neutral; saved style/LoRA choices remain untouched.
STARTUP_DEFAULTS = {
    **krea_style.DEFAULTS,
    **MOE_CACHE_DEFAULTS,
    **latent_composition.DEFAULTS,
    "moss_image_slg": False,
    "moss_image_edit_format": "Disabled",
    "moss_image_edit_strength": 1.0,
    "image_number": 1,
    "seed_random": True,
    "seed": 0,
    "style": [],
    "prompt": "",
    "negative_prompt": "",
    "performance": "Model Defaults",
    "resolution": "1152x896 (4:3)",
    "resolution_auto": True,
    "base_model": "sd_xl_base_1.0_0.9vae.safetensors",
    "model_recipe_path": "",
    "clip_layer_source": "request",
    "reference_adapter": "none",
    "reference_adapter_model": "",
    "reference_adapter_vision": "",
    "reference_adapter_strength": 1.0,
    "gligen_model": None,
    "gligen_model_signature": None,
    "gligen_layout": "[]",
    "hypernetwork_model": None,
    "hypernetwork_model_signature": None,
    "hypernetwork_strength": 1.0,
    **model_patches.DEFAULTS,
    **native_control_recipes.DEFAULTS,
    **lora_hooks.DEFAULTS,
    **wevisone.DEFAULTS,
    **ADJUSTMENT_DEFAULTS,
    **EXTRA_UTILITY_DEFAULTS,
    "lora_1_model": "None",
    "lora_1_weight": 0.5,
    "lora_2_model": "None",
    "lora_2_weight": 0.5,
    "lora_3_model": "None",
    "lora_3_weight": 0.5,
    "lora_4_model": "None",
    "lora_4_weight": 0.5,
    "lora_5_model": "None",
    "lora_5_weight": 0.5,
    "auto_negative_prompt": False,
    **BERNINI_DEFAULTS,
}

DEFAULT_CONTROLS = {
    **{key: key for key in krea_style.FIELDS},
    "gligen_model": "gligen_model",
    "gligen_model_signature": "gligen_model_signature",
    "gligen_layout": "gligen_layout",
    "hypernetwork_model": "hypernetwork_model",
    "hypernetwork_model_signature": "hypernetwork_model_signature",
    "hypernetwork_strength": "hypernetwork_strength",
    "qwen_layer_count": "qwen_layer_count",
    "curve_master": "curve_master",
    "curve_red": "curve_red",
    "curve_green": "curve_green",
    "curve_blue": "curve_blue",
    "curve_interpolation": "curve_interpolation",
    "shader_effect": "shader_effect",
    "shader_amount": "shader_amount",
    "shader_levels": "shader_levels",
    "filter_radius": "filter_radius",
    "filter_sigma": "filter_sigma",
    "filter_strength": "filter_strength",
    "filter_factor": "filter_factor",
    "image_utility_model": "image_utility_model",
    "image_utility_vae": "image_utility_vae",
    "image_utility_lora": "image_utility_lora",
    "image_utility_conditioning": "image_utility_conditioning",
    "moge_resolution_level": "moge_resolution_level",
    "moge_refine_steps": "moge_refine_steps",
    "face_detector_variant": "face_detector_variant",
    "face_max_count": "face_max_count",
    "face_min_confidence": "face_min_confidence",
    "pose_score_threshold": "pose_score_threshold",
    "grid_columns": "grid_columns",
    "grid_cell_width": "grid_cell_width",
    "grid_cell_height": "grid_cell_height",
    "grid_padding": "grid_padding",
    "composition_mode": "composition_mode",
    "composition_x": "composition_x",
    "composition_y": "composition_y",
    "composition_width": "composition_width",
    "composition_height": "composition_height",
    "rembg_native_model": "rembg_native_model",
    "native_vision_model": "native_vision_model",
    "vision_threshold": "vision_threshold",
    "detect_class": "detect_class",
    "detect_max_count": "detect_max_count",
    "segment_refine_iterations": "segment_refine_iterations",
    "segment_output": "segment_output",
    "seedvr2_model": "seedvr2_model",
    "seedvr2_vae": "seedvr2_vae",
    "seedvr2_scale": "seedvr2_scale",
    "seedvr2_color": "seedvr2_color",
    "base_model": "base_model_name",
    "performance": "performance_selection",
    "resolution": "aspect_ratios_selection",
    "resolution_auto": "resolution_auto",
    "prompt": "prompt",
    "negative_prompt": "negative",
    **{key: key for key in latent_composition.FIELDS},
    "style": "style_selection",
    "seed": "seed",
    "seed_random": "seed_random",
    "image_number": "image_number",
    "auto_negative_prompt": "auto_negative",
    "custom_width": "custom_width",
    "custom_height": "custom_height",
    "cn_model": "cn_model",
    "cn_model_signature": "cn_model_signature",
    "cn_depth_preprocess": "cn_depth_preprocess",
    "cn_depth_model": "cn_depth_model",
    "cn_depth_lotus_model": "cn_depth_lotus_model",
    "cn_depth_lotus_vae": "cn_depth_lotus_vae",
    "model_recipe_path": "model_recipe_path",
    "model_recipe_sha256": "model_recipe_sha256",
    "model_recipe_overrides": "model_recipe_overrides",
    "clip_layer_source": "clip_layer_source",
    "cascade_stage_b": "cascade_stage_b",
    "cascade_b_steps": "cascade_b_steps",
    "cascade_b_cfg": "cascade_b_cfg",
    "cascade_b_sampler": "cascade_b_sampler",
    "cascade_b_scheduler": "cascade_b_scheduler",
    "cascade_compression": "cascade_compression",
    "cascade_source_denoise": "cascade_source_denoise",
    "cascade_super_resolution": "cascade_super_resolution",
    "cascade_super_resolution_model": "cascade_super_resolution_model",
    "cascade_control_strength": "cascade_control_strength",
    "cascade_control_start": "cascade_control_start",
    "cascade_control_end": "cascade_control_end",
    "cascade_control_denoise": "cascade_control_denoise",
    "cascade_vision_enabled": "cascade_vision_enabled",
    "cascade_vision_model": "cascade_vision_model",
    "cascade_vision_strengths": "cascade_vision_strengths",
    "cascade_vision_crop": "cascade_vision_crop",
    "sdxl_refiner_model": "sdxl_refiner_model",
    "sdxl_refiner_switch_step": "sdxl_refiner_switch_step",
    "sdxl_refiner_cfg": "sdxl_refiner_cfg",
    "sdxl_refiner_clip": "sdxl_refiner_clip",
    "refiner_noise_augmentation": "refiner_noise_augmentation",
    "pid_input_vae": "pid_input_vae",
    "pid_latent_format": "pid_latent_format",
    "pid_degrade_sigma": "pid_degrade_sigma",
    "reference_adapter": "reference_adapter",
    "reference_adapter_model": "reference_adapter_model",
    "reference_adapter_vision": "reference_adapter_vision",
    "reference_adapter_strength": "reference_adapter_strength",
    **{key: key for key in ADJUSTMENT_DEFAULTS},
    **{key: key for key in EXTRA_UTILITY_DEFAULTS},
    **{key: key for key in (*model_patches.FIELDS, *native_control_recipes.FIELDS)},
    **{key: key for key in lora_hooks.FIELDS},
    **{key: key for key in wevisone.DEFAULTS},
    **{key: key for key in BERNINI_DEFAULTS},
}
SAMPLING_KEYS = ("custom_steps", "cfg", "sampler_name", "scheduler", "clip_skip")


def startup_loras(settings):
    if "startup_loras" in settings:
        return [(entry["name"], entry["weight"]) for entry in settings["startup_loras"]]
    return [
        (settings[f"lora_{i}_model"], settings.get(f"lora_{i}_weight", 0.5))
        for i in range(1, 6)
        if settings.get(f"lora_{i}_model") not in (None, "", "None")
    ]


def generation_defaults(values):
    from ruinedfooocus.core.settings_validation import validate_settings

    if values.get("preset_selection"):
        raise ValueError("Clear the generation preset before saving current controls as defaults.")
    result = {
        key: copy.deepcopy(values[control])
        for key, control in DEFAULT_CONTROLS.items()
        if control in values
    }
    # Old styles were staged, not active. Only explicit new saves opt into applying them.
    result["style_application"] = 1
    # Forever is an action, not a safe startup batch count.
    result["image_number"] = max(1, int(result.get("image_number", 1)))
    loras = []
    for item in values.get("loras") or []:
        caption = item.get("caption") if isinstance(item, dict) else item[1]
        weight, name = caption.split(" - ", 1)
        weight = float(weight)
        if not math.isfinite(weight):
            raise ValueError("LoRA weights must be finite numbers.")
        if name and name != "None":
            loras.append({"name": name, "weight": weight})
    result["startup_loras"] = loras
    sampling = {key: values[key] for key in SAMPLING_KEYS if key in values}
    validate_settings(sampling)
    result["custom_sampling"] = sampling
    return validate_settings(result)
