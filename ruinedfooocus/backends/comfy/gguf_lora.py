"""Apply the pinned GGUF adapter importer to an already resolved local file."""

import math

from ruinedfooocus.features.models.model_identity import file_signature


def patch_count(model, clip):
    return sum(
        len(rows)
        for patcher in (model, getattr(clip, "patcher", None))
        for rows in getattr(patcher, "patches", {}).values()
    )


def _unchanged(source, path, signature):
    try:
        return file_signature(source) == signature and file_signature(path) == signature
    except OSError:
        return False


def _target_shape(target, modules):
    from comfy.utils import get_attr

    if isinstance(target, tuple):
        if len(target) != 2:
            raise ValueError("GGUF LoRA target transforms are unsupported.")
        key, offset = target
    else:
        key, offset = target, None

    matches = []
    for module in modules:
        try:
            matches.append(get_attr(module, key))
        except AttributeError:
            continue
    shape = list(getattr(matches[0], "shape", ())) if len(matches) == 1 else []
    if len(shape) != 2:
        raise ValueError(f"GGUF LoRA needs an unambiguous 2-D target for {key}.")
    if offset is not None:
        dimension, start, length = offset
        if (
            not 0 <= dimension < len(shape)
            or start < 0
            or not 0 < length <= shape[dimension] - start
        ):
            raise ValueError(f"GGUF LoRA target slice is out of bounds for {key}.")
        shape[dimension] = length
    return tuple(shape)


def apply_gguf_lora(source, signature, model, clip, weight):
    from comfy.sd import load_lora_for_models
    from molbal_comfyui_gguf.lora import load_gguf_lora
    from molbal_comfyui_gguf.nodes import _gguf_lora_key_map, _remap_gguf_lora_for_comfy
    from ruinedfooocus.app.resource_lifecycle import current_job
    from ruinedfooocus.core.preparation import preparation_service

    with preparation_service.artifact_lease(
        signature[0], cancelled=getattr(current_job(), "cancelled", None)
    ) as path:
        if not _unchanged(source, path, signature):
            raise ValueError("The GGUF LoRA changed before loading. Submit the request again.")
        _, targets, metadata = load_gguf_lora(str(path))
        if not _unchanged(source, path, signature):
            raise ValueError("The GGUF LoRA changed during loading. Submit the request again.")

    if metadata["alpha"] is not None and not math.isfinite(metadata["alpha"]):
        raise ValueError("GGUF LoRA alpha must be finite.")
    key_map = _gguf_lora_key_map(model, clip)
    lora = _remap_gguf_lora_for_comfy(targets, key_map)
    factor_suffix = ".lora_A.weight"
    names = [key.removesuffix(factor_suffix) for key in lora if key.endswith(factor_suffix)]
    mapped = [key_map[name] for name in names]
    if len(mapped) != len(targets) or len(set(mapped)) != len(mapped):
        raise ValueError("GGUF LoRA factors map to duplicate native targets.")
    modules = [
        patcher
        for patcher in (getattr(model, "model", None), getattr(clip, "cond_stage_model", None))
        if patcher is not None
    ]
    for name, target in zip(names, mapped):
        down, up = lora[name + factor_suffix], lora[name + ".lora_B.weight"]
        if not down.shape[0] or (up.shape[0], down.shape[1]) != _target_shape(target, modules):
            raise ValueError(
                f"GGUF LoRA factor dimensions do not match the native target for {name}."
            )

    before = patch_count(model, clip)
    patched_model, patched_clip = load_lora_for_models(
        model, clip, lora, weight, weight, lora_metadata={"gguf_lora": metadata}
    )
    if patch_count(patched_model, patched_clip) - before != len(targets):
        raise ValueError("The native patcher did not apply all GGUF LoRA targets.")
    return patched_model, patched_clip
