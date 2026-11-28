"""Captured model-only Krea2 style LoRA; native loaders own weight application."""

from dataclasses import dataclass
from functools import wraps
import math
from pathlib import Path, PurePosixPath

KEY = "_native_krea_style"
DEFAULT_LORA = "krea2_style_reference.safetensors"
FIELDS = ("krea_style_lora", "krea_style_strength")
DEFAULTS = dict(krea_style_lora=DEFAULT_LORA, krea_style_strength=1.0)


@dataclass(frozen=True)
class KreaStyle:
    name: str
    strength: float


def values(data):
    name = data.get("krea_style_lora")
    if name is None or name == "":
        name = DEFAULT_LORA
    if not isinstance(name, str):
        raise ValueError("Choose a Krea style LoRA by its relative local filename.")
    name = name.replace("\\", "/")
    path = PurePosixPath(name)
    if path.is_absolute() or ":" in name or ".." in path.parts or len(name) > 512:
        raise ValueError("Choose a Krea style LoRA by its relative local filename.")
    strength = data.get("krea_style_strength")
    if strength is None:
        strength = 1.0
    if (
        type(strength) not in (int, float)
        or not math.isfinite(strength)
        or not -10 <= strength <= 10
    ):
        raise ValueError("Krea style LoRA strength must be finite and between -10 and 10.")
    return KreaStyle(name, float(strength))


def capture_assets(data, manager, models, controls):
    from ruinedfooocus.features.generation.image_preparation import (
        _artifact,
        _control_checkpoint_family,
    )
    from ruinedfooocus.features.models.image_components import ALIASES

    result = dict(data)
    result.pop(KEY, None)
    result.pop("native_krea_style", None)
    if (controls or {}).get("type") != "edit":
        if any(result.get(key) is not None for key in FIELDS):
            raise ValueError("Krea style LoRA settings require the Krea2 style-reference recipe.")
        return result, ()
    family = _control_checkpoint_family(models, result.get("base_model_name"))
    family = ALIASES.get(family, family)
    active = (
        family == "Krea2"
        and (controls or {}).get("type") == "edit"
        and result.get("native_edit_recipe", "auto") in ("auto", "krea2-style")
    )
    if not active:
        if any(result.get(key) is not None for key in FIELDS):
            raise ValueError("Krea style LoRA settings require the Krea2 style-reference recipe.")
        return result, ()
    if result.get("input_image") is None or result.get("inpaint_toggle"):
        raise ValueError("Krea style references require a primary image without an inpaint mask.")
    if str(result.get("image_engine", "auto")).startswith("sdcpp-"):
        raise ValueError("Krea style references require the native image engine.")
    plan = values(result)
    item = _artifact(manager, "krea-style-lora", "loras", plan.name)
    result.update(krea_style_lora=plan.name, krea_style_strength=plan.strength)
    result[KEY] = plan
    return result, (item,)


def scoped(function):
    """Hold the captured adapter lease and restore the exact original model on exit."""

    @wraps(function)
    def run(owner, gen_data=None, *args, **kwargs):
        plan = (gen_data or {}).get(KEY)
        if plan is None:
            if (gen_data or {}).get("native_edit_recipe") == "krea2-style":
                raise ValueError("Capture the Krea style LoRA before compute.")
            return function(owner, gen_data, *args, **kwargs)
        if not isinstance(plan, KreaStyle) or plan != values(gen_data):
            raise ValueError("The captured Krea style settings changed after submission.")
        from ruinedfooocus.app.resource_lifecycle import require_compute_owner
        from ruinedfooocus.features.generation.image_preparation import prepared_asset
        from ruinedfooocus.backends.comfy.gguf_lora import patch_count
        from ruinedfooocus.features.models.model_identity import file_signature
        from ruinedfooocus.app import async_worker
        from ruinedfooocus.app.state import path_manager
        import folder_paths
        from nodes import LoraLoaderModelOnly

        require_compute_owner()
        original = owner.xl_base_patched
        if original is None or type(original.unet.model).__name__ != "Krea2":
            raise ValueError("The Krea style recipe requires the captured Krea2 checkpoint.")
        with prepared_asset(gen_data, "krea-style-lora", plan.name, path_manager, "loras") as path:
            if str(path.resolve()) != str(
                Path(folder_paths.get_full_path_or_raise("loras", plan.name)).resolve()
            ):
                raise ValueError("The native style LoRA path differs from its captured file.")
            stamp = file_signature(path)
            async_worker.check_interrupt(gen_data)
            model = LoraLoaderModelOnly().load_lora_model_only(
                original.unet, plan.name, plan.strength
            )[0]
            if plan.strength and patch_count(model, None) <= patch_count(original.unet, None):
                raise ValueError(
                    "The selected style LoRA has no weights for this Krea2 checkpoint."
                )
            if file_signature(path) != stamp:
                raise ValueError("The Krea style LoRA changed while loading. Submit again.")
            async_worker.check_interrupt(gen_data)
            owner.xl_base_patched = owner.StableDiffusionModel(
                unet=model, clip=original.clip, vae=original.vae, clip_vision=original.clip_vision
            )
            gen_data["native_krea_style"] = {
                "native_node": "LoraLoaderModelOnly",
                "name": plan.name,
                "strength": plan.strength,
                "file_stat": list(stamp[1:]),
            }
            try:
                return function(owner, gen_data, *args, **kwargs)
            finally:
                owner.xl_base_patched = original

    return run
