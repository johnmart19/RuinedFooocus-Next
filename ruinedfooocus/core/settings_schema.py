"""Pure validation shared by early startup and the Settings editor.

Unknown fields survive unchanged. Feature-specific validation remains with its
owner; this module never imports application state, Torch or device discovery.
"""

import math
import re
from decimal import Decimal, InvalidOperation
from ipaddress import ip_address

# Stable saved values per component; hardware filtering happens in hardware_inventory.
IMAGE_BACKENDS = ("auto", "cuda", "rocm", "xpu", "cpu", "directml")
# Background removal: native ONNX Runtime builds run in an isolated worker.
ONNX_BACKENDS = ("Auto", "CPU", "CUDA", "MIGraphX", "OpenVINO", "Windows GPU")
# The xllamacpp ROCm 6.4 index stopped at an older release; ROCm 7.2 is Linux-only.
XLLAMACPP_BACKENDS = ("Auto", "CPU", "Vulkan", "CUDA 12.8", "CUDA 13.2", "ROCm 7.2")
RAG_EMBEDDING_DEVICES = ("Auto", "CPU")

COMPUTE_CHOICES = {
    "image_device": ("Auto", "CPU"),
    "image_backend": IMAGE_BACKENDS,
    "image_deterministic": ("Auto", "Enabled"),
    "image_smart_memory": ("Auto", "Disabled"),
    "image_pinned_memory": ("Auto", "Disabled"),
    "image_mmap": ("Auto", "Enabled", "Disabled"),
    "image_allocator": ("Auto", "Native", "Async"),
    "image_dynamic_vram": ("Auto", "Enabled", "Disabled"),
    "image_async_offload": ("Auto", "Disabled", "1", "2", "4"),
    "image_fast_disk": ("Auto", "Enabled", "Disabled"),
    "image_vae_device": ("Auto", "CPU"),
    "image_preview_method": ("auto", "none", "latent2rgb", "taesd"),
    "image_unet_precision": ("Auto", "FP32", "FP16", "BF16"),
    "image_text_precision": ("Auto", "FP32", "FP16", "BF16"),
    "image_vae_precision": ("Auto", "FP32", "FP16", "BF16"),
    "image_attention": ("Auto", "PyTorch", "Split", "Quad"),
    "image_attention_upcast": ("Auto", "Enabled", "Disabled"),
    "image_compiler": ("Auto", "Disabled"),
    "image_cuda_graphs": ("Auto", "Disabled"),
}
COMPUTE_MEMORY_FIELDS = ("image_reserve_vram", "image_vram_headroom")


INTEGER_RANGES = {
    "qwen_layer_count": (1, 16),
    "shader_levels": (2, 32),
    "filter_radius": (1, 31),
    "moge_resolution_level": (0, 9),
    "moge_refine_steps": (0, 8),
    "face_max_count": (1, 16),
    "image_number": (0, None),
    "image_number_max": (1, None),
    "seed": (-1, None),
    "preset_mode": (1, 7),
    "images_per_page": (1, 1000),
    "hint_chance": (0, 100),
    "llm_n_predict": (-1, None),
    "llm_n_ctx": (0, None),
    "llm_n_gpu_layers": (-1, None),
    "llm_chat_history": (0, None),
    "llm_hp_max_tokens": (1, None),
    "startup_port": (1, 65535),
}
FLOAT_RANGES = {
    "krea_style_strength": (-10, 10),
    "llama_moe_cache_cpu_share": (0, 1),
    "shader_amount": (0, 1),
    "filter_sigma": (0.1, 10),
    "filter_strength": (0, 5),
    "filter_factor": (0, 2),
    "face_min_confidence": (0, 1),
    "pose_score_threshold": (0, 1),
    "reference_adapter_strength": (0, 10),
    "update_interval": (0.01, None),
    "hypernetwork_strength": (-10, 10),
    "moss_image_edit_strength": (0, 100),
}
BOOLEAN_FIELDS = {
    "llama_moe_cache_cpu_share_enabled",
    "resolution_auto",
    "seed_random",
    "auto_negative_prompt",
    "local_model_metadata",
    "preload_image_model",
    "cpu_everywhere",
    "llm_use_model_defaults",
    "save_metadata",
    "vae_tiled",
    "moss_image_slg",
    "cfg_norm_pre_cfg",
    "cfg_norm_auto",
    "startup_browser",
    "startup_offline",
    "startup_skip_dependency_install",
    "startup_api",
    "startup_mcp",
}
ENUM_FIELDS = {
    "curve_interpolation": ("linear", "monotone_cubic"),
    "shader_effect": ("invert", "bt709-grayscale", "posterize"),
    "face_detector_variant": ("short", "full", "both"),
    "reference_adapter": ("none", "style", "redux", "photomaker"),
    "clip_layer_source": ("request", "recipe"),
    "image_engine": ("auto", "native", "sdcpp-vulkan", "sdcpp-cpu"),
    "rembg_backend": ONNX_BACKENDS,
    "xllamacpp_backend": XLLAMACPP_BACKENDS,
    "rembg_provider": ("Auto", "Windows ML", "DirectML"),
    "rembg_winml_type": ("Auto", "CPU", "GPU", "NPU"),
    "llm_rewrite_reasoning": ("Low", "Medium", "High"),
    "llm_chat_reasoning_mode": ("Auto", "Off", "On"),
    "llm_chat_reasoning_effort": ("Auto", "Low", "Medium", "High"),
    "llm_rewrite_reasoning_mode": (None, "Auto", "Off", "On"),
    "llm_rewrite_reasoning_effort": ("Auto", "Low", "Medium", "High"),
    "rag_embedding_device": RAG_EMBEDDING_DEVICES,
    "image_resize_mode": ("preserve", "pad", "crop"),
}
STARTUP_CONTROLS = (
    (
        "startup_browser",
        "Open browser on startup",
        True,
        "Restart required. --nobrowser takes priority.",
    ),
    (
        "startup_offline",
        "Offline startup",
        False,
        "Skip startup updates and downloads. RF_OFFLINE and --offline take priority.",
    ),
    (
        "startup_skip_dependency_install",
        "Use installed packages",
        False,
        "Skip dependency installation at startup; source updates remain allowed when online.",
    ),
)


def validate_known(values):
    result = dict(values)
    from ruinedfooocus.features.image_processing.image_composition import FIELDS, settings

    if any(key in result for key in FIELDS):
        validated = settings(result)
        result.update((key, validated[key]) for key in FIELDS if key in result)
    from ruinedfooocus.features.image_processing import latent_composition

    if any(key in result for key in latent_composition.FIELDS):
        validated = latent_composition.settings(result)
        result.update((key, validated[key]) for key in latent_composition.FIELDS if key in result)
    if "image_component_profiles" in result:
        from ruinedfooocus.features.models.component_profiles import validate_profiles

        result["image_component_profiles"] = validate_profiles(result["image_component_profiles"])
    for key, value in result.items():
        if key == "model_recipe_path" and (not isinstance(value, str) or len(value) > 4096):
            raise ValueError("model_recipe_path: choose a local recipe or leave blank.")
        if (
            key == "model_recipe_sha256"
            and value is not None
            and (not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value))
        ):
            raise ValueError("model_recipe_sha256: invalid saved recipe identity.")
        if key == "sdcpp_vulkan_device" and (
            not isinstance(value, str) or not re.fullmatch(r"Vulkan[0-9]+", value)
        ):
            raise ValueError("sdcpp_vulkan_device: choose an exact backend name such as Vulkan0.")
        if key in COMPUTE_CHOICES and value not in COMPUTE_CHOICES[key]:
            raise ValueError(f"{key}: choose a listed option.")
        if key in COMPUTE_MEMORY_FIELDS:
            if value in (None, ""):
                result[key] = None
                continue
            if isinstance(value, bool):
                raise ValueError(f"{key}: enter a number, not a boolean.")
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key}: enter a non-negative number or leave blank.") from None
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{key}: enter a finite non-negative number.")
            result[key] = number
        if key == "startup_listen":
            if value in (None, ""):
                result[key] = None
                continue
            if not isinstance(value, str):
                raise ValueError("startup_listen: enter an IP address, localhost, or leave blank.")
            value = value.strip()
            if value != "localhost":
                try:
                    ip_address(value)
                except ValueError:
                    raise ValueError(
                        "startup_listen: enter an IP address, localhost, or leave blank."
                    ) from None
            result[key] = value
        if key in ENUM_FIELDS and value not in ENUM_FIELDS[key]:
            raise ValueError(f"{key}: choose a listed option.")
        if key in BOOLEAN_FIELDS and type(value) is not bool:
            raise ValueError(f"{key}: choose enabled or disabled (a boolean).")
        if key in INTEGER_RANGES or key in FLOAT_RANGES:
            if key == "startup_port" and (
                value is None or value == "" or (type(value) is int and value == 0)
            ):
                # The port field shows 0 for "automatic"; store it as unset.
                result[key] = None
                continue
            if isinstance(value, bool):
                raise ValueError(f"{key}: enter a number, not a boolean.")
            try:
                # Integer seeds and decimal text can exceed float's exact range.
                if key in INTEGER_RANGES and isinstance(value, (int, str)):
                    number = Decimal(value) if isinstance(value, str) else value
                else:
                    number = float(value)
            except (TypeError, ValueError, InvalidOperation):
                raise ValueError(f"{key}: enter a number.") from None
            finite = (
                number.is_finite() and math.isfinite(number)
                if isinstance(number, Decimal)
                else (isinstance(number, int) or math.isfinite(number))
            )
            if not finite:
                raise ValueError(f"{key}: enter a finite number.")
            low, high = (INTEGER_RANGES | FLOAT_RANGES)[key]
            if (low is not None and number < low) or (high is not None and number > high):
                raise ValueError(f"{key}: number is outside the supported range.")
            if key in INTEGER_RANGES and number != int(number):
                raise ValueError(f"{key}: enter a whole number.")
            result[key] = int(number) if key in INTEGER_RANGES else number
        if key == "language" and (not isinstance(value, str) or not value.strip()):
            raise ValueError("language: choose a supported language.")
    from ruinedfooocus.features.image_editing import model_patches, native_control_recipes

    for feature in (model_patches, native_control_recipes):
        if any(key in result for key in feature.FIELDS):
            validated = feature.replay_values(result, clear_images=False)
            result.update((key, validated[key]) for key in feature.FIELDS if key in result)
    from ruinedfooocus.features.image_editing import krea_style, lora_hooks

    if any(key in result for key in krea_style.FIELDS):
        validated = krea_style.values(result)
        if "krea_style_lora" in result:
            result["krea_style_lora"] = validated.name
        if "krea_style_strength" in result:
            result["krea_style_strength"] = validated.strength

    if any(key in result for key in lora_hooks.FIELDS):
        validated = lora_hooks.replay_values(result)
        result.update((key, validated[key]) for key in lora_hooks.FIELDS if key in result)
    from ruinedfooocus.features.image_processing import wevisone

    if any(key in result for key in wevisone.DEFAULTS):
        validated = wevisone.parameters(dict(result, controlnet={}))
        result.update((key, validated[key]) for key in wevisone.DEFAULTS if key in result)
    from ruinedfooocus.features.image_processing.still_image_utilities import ADJUSTMENTS

    for feature in ADJUSTMENTS:
        if any(key in result for key in feature.DEFAULTS):
            validated = feature.parameters(result)
            result.update((key, validated[key]) for key in feature.DEFAULTS if key in result)
    from ruinedfooocus.features.image_processing.depth_utilities import (
        MOGE_OUTPUT_DEFAULTS,
        FACE_OUTPUT_DEFAULTS,
    )
    from ruinedfooocus.features.image_processing import utility_moge, utility_face

    for defaults, owner in (
        (MOGE_OUTPUT_DEFAULTS, utility_moge),
        (FACE_OUTPUT_DEFAULTS, utility_face),
    ):
        if any(key in result for key in defaults):
            validated = owner.parameters(result)
            result.update((key, validated[key]) for key in defaults if key in result)
    return result
