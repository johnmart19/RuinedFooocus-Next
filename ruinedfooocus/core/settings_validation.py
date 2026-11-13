"""Validate Settings form values before changing or writing configuration."""

import math
from ruinedfooocus.core.settings_schema import validate_known
from ruinedfooocus.core.profile_store import profile_name
from ruinedfooocus.features.image_editing.image_guidance import (
    MOSS_EDIT_FORMATS,
    STARTUP_SHIFT_LABELS,
    validate_guidance,
)


LIST_FIELDS = {
    "archive_folders",
    "path_checkpoints",
    "path_diffusion_models",
    "path_clip",
    "path_loras",
    "path_embeddings",
    "path_gligen",
    "path_hypernetworks",
    "path_wildcards",
}
# Reasoning effort for local prompt rewrites (gpt-oss reasoning_effort; Low also disables Qwen3 thinking).
REWRITE_REASONING = ("Low", "Medium", "High")


def validate_settings(values):
    values = dict(values)
    # Number components submit None when cleared. Keep that removal sentinel
    # for prospective_profile, but do not pass it through integer validation.
    # Raw persisted/startup values still use validate_known's existing contract.
    automatic_chat = {
        key: values.pop(key)
        for key in ("llm_n_ctx", "llm_n_predict")
        if key in values and values[key] is None
    }
    if values.get("llm_rewrite_reasoning_mode") == "":
        # The empty UI choice preserves the pre-existing model-specific policy.
        # prospective_profile removes this optional key rather than writing Auto.
        values["llm_rewrite_reasoning_mode"] = None
    result = validate_known(values)
    result.update(automatic_chat)
    from ruinedfooocus.features.chat.model_profiles import SAMPLERS, validate_sampling

    # Optional form fields become one saved override object; blank means automatic.
    sampling_fields = [key for key in result if key.startswith("llm_sampling__")]
    if sampling_fields:
        overrides = {}
        for key in sampling_fields:
            name, value = key.removeprefix("llm_sampling__"), result.pop(key)
            if name not in SAMPLERS:
                raise ValueError(f"Unsupported Chat sampling option: {name}")
            if isinstance(value, str):
                value = value.strip()
                if value:
                    try:
                        value = float(value)
                    except ValueError:
                        raise ValueError(f"{name}: enter a number or leave blank.") from None
                    if name == "top_k" and value.is_integer():
                        value = int(value)
            if value is not None and value != "":
                overrides[name] = value
        result["llm_sampling"] = overrides

    if "llm_sampling" in result:
        result["llm_sampling"] = validate_sampling(result["llm_sampling"])
    from ruinedfooocus.core.settings_schema import ONNX_BACKENDS

    if result.get("rembg_backend", "Auto") not in ONNX_BACKENDS:
        raise ValueError("Choose a listed background-removal runtime.")
    from ruinedfooocus.backends.onnx.windows_ml import PROVIDERS, CATALOG_PROVIDERS, DEVICE_TYPES

    if result.get("rembg_provider", "Auto") not in PROVIDERS:
        raise ValueError("Choose Auto, Windows ML or DirectML for the background provider.")
    if result.get("rembg_winml_provider", "Auto") not in ("Auto", *CATALOG_PROVIDERS):
        raise ValueError("Choose a listed Windows ML execution provider.")
    if result.get("rembg_winml_type", "Auto") not in DEVICE_TYPES:
        raise ValueError("Choose Auto, CPU, GPU or NPU for Windows ML.")
    device = result.get("rembg_winml_device", "Auto")
    if not isinstance(device, str) or (
        device != "Auto" and (len(device) != 24 or any(c not in "0123456789abcdef" for c in device))
    ):
        raise ValueError("Choose Auto or a discovered Windows ML device identity.")
    result.update(validate_guidance(result))
    if result.get("moss_image_edit_format", "Disabled") not in MOSS_EDIT_FORMATS:
        raise ValueError("Choose a supported MOSS Image edit format.")
    if result.get("llm_rewrite_reasoning", "Low") not in REWRITE_REASONING:
        raise ValueError("Choose Low, Medium or High prompt rewrite reasoning.")
    for key, value in result.items():
        # Primitive profile fields were validated and normalized by validate_known.
        # These bounds belong only to image guidance and LoRA feature overrides.
        bounds = None
        if key.endswith("_shift"):
            bounds = (0.01, 100 if key in STARTUP_SHIFT_LABELS else None)
        if (key.startswith("lora_") and key.endswith("_weight")) or key in ("lora_min", "lora_max"):
            bounds = (None, None)
        if bounds is not None and value in (None, "") and not key.endswith("_shift"):
            raise ValueError(f"{key}: enter a number.")
        if bounds is not None and value not in (None, ""):
            if isinstance(value, bool):
                raise ValueError(f"{key}: enter a number, not a boolean.")
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key}: enter a number.") from None
            if not math.isfinite(number):
                raise ValueError(f"{key}: enter a finite number.")
            low, high = bounds
            if (low is not None and number < low) or (high is not None and number > high):
                raise ValueError(
                    f"{key}: value must be between {low} and {high or 'no upper limit'}."
                )
            result[key] = number
        if key in LIST_FIELDS:
            lines = value.splitlines() if isinstance(value, str) else (value or [])
            result[key] = [line.strip() for line in lines if line.strip()]
            if key in ("path_checkpoints", "path_loras", "path_clip") and not result[key]:
                raise ValueError(f"{key}: provide at least one folder.")
        if (
            key
            in (
                "path_outputs",
                "path_inbox",
                "path_geometry_estimation",
                "path_detection",
                "path_model_patches",
            )
            and not str(value or "").strip()
        ):
            raise ValueError(f"{key}: provide a folder.")
    if result.get("image_number", 1) > result.get("image_number_max", 50):
        raise ValueError("Image Number cannot exceed Image Number Max.")
    if result.get("lora_min", 0) >= result.get("lora_max", 2):
        raise ValueError("LoRA weight min must be smaller than max.")
    name = result.get("ui_settings_name")
    if name:
        profile_name(name)
    return result
