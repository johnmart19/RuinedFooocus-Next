"""Shared LLM acquisition preference and explicit xllamacpp parameter controls."""

_GGML_KV_TYPES = {
    "f32": "GGML_TYPE_F32",
    "f16": "GGML_TYPE_F16",
    "bf16": "GGML_TYPE_BF16",
    "q8_0": "GGML_TYPE_Q8_0",
    "q5_1": "GGML_TYPE_Q5_1",
    "q5_0": "GGML_TYPE_Q5_0",
    "q4_1": "GGML_TYPE_Q4_1",
    "q4_0": "GGML_TYPE_Q4_0",
}

_FLASH_ATTN_TYPES = {
    "Auto": "LLAMA_FLASH_ATTN_TYPE_AUTO",
    "On": "LLAMA_FLASH_ATTN_TYPE_ENABLED",
    "Off": "LLAMA_FLASH_ATTN_TYPE_DISABLED",
}

_LOAD_MODES = {
    "Auto": "LLAMA_LOAD_MODE_AUTO",
    "Enabled": "LLAMA_LOAD_MODE_MMAP",
    "Disabled": "LLAMA_LOAD_MODE_NONE",
}


def preferred_backend(system, choice="Auto", *, gpu=True):
    """Prefer Vulkan independently of image Torch; activation must verify devices."""
    if not gpu or choice == "CPU":
        return "CPU"
    if choice == "Auto" and system in ("Windows", "Linux"):
        return "Vulkan"
    return choice


def _resolve_enum(engine, enum_name, member_name):
    """Resolve a Cython cpdef enum member from either the enum class or module."""
    if engine is None:
        return None
    enum_cls = getattr(engine, enum_name, None)
    if enum_cls is not None and hasattr(enum_cls, member_name):
        return getattr(enum_cls, member_name)
    if hasattr(engine, member_name):
        return getattr(engine, member_name)
    return None


def _parse_flag_int(arguments, names):
    """Extract the last integer value for a CLI flag in ``arguments``."""
    tokens = tuple(arguments or ())
    value = None
    for index, token in enumerate(tokens):
        flag, sep, inline = str(token).partition("=")
        if flag in names:
            raw = inline if sep else (tokens[index + 1] if index + 1 < len(tokens) else None)
            if raw is not None:
                try:
                    value = int(raw)
                except (TypeError, ValueError):
                    pass
    return value


def configure_xllamacpp_offload(params, settings, *, cpu=False):
    """Enforce CPU even when a GPU-capable extension is already cached/imported."""
    params.n_gpu_layers = 0 if cpu else int(settings.get("llm_n_gpu_layers", -1))
    if cpu:
        params.no_kv_offload = True
        params.no_op_offload = True
        params.mmproj_use_gpu = False


def configure_xllamacpp_params(
    params, settings, *, engine=None, cpu=False, projector=None, native_defaults=()
):
    """Map RuinedFooocus settings cleanly onto xllamacpp 2026.11.12361 CommonParams.

    Handles flash attention, KV cache quantization, batch sizing, memory-mapping
    load_mode (replacing removed use_mmap), MoE cache CPU share, multimodal
    projector flags, reasoning format/budget, and CPU offload enforcement.
    """
    predict = settings.get("llm_n_predict", -1)
    params.n_predict = int(-1 if predict is None else predict)
    params.n_ctx = int(settings.get("llm_n_ctx", 8192))
    if hasattr(params, "n_parallel"):
        params.n_parallel = 1

    configure_xllamacpp_offload(params, settings, cpu=cpu)

    if projector:
        params.mmproj.path = str(projector)
        if hasattr(params, "no_mmproj"):
            params.no_mmproj = False
        if not cpu and hasattr(params, "mmproj_use_gpu"):
            params.mmproj_use_gpu = True
    elif hasattr(params, "no_mmproj"):
        params.no_mmproj = True

    kv_type = str(settings.get("llm_kv_cache_type", "Auto") or "Auto")
    flash_attn = str(settings.get("llm_flash_attn", "Auto") or "Auto")
    kv_member = _GGML_KV_TYPES.get(kv_type.lower())
    if kv_member is not None:
        kv_enum = _resolve_enum(engine, "ggml_type", kv_member)
        if kv_enum is not None:
            if hasattr(params, "cache_type_k"):
                params.cache_type_k = kv_enum
            if hasattr(params, "cache_type_v"):
                params.cache_type_v = kv_enum
        if flash_attn == "Auto" and kv_type.lower() not in ("f16", "bf16", "f32") and not cpu:
            flash_attn = "On"

    fa_member = _FLASH_ATTN_TYPES.get(flash_attn)
    if fa_member is not None and flash_attn in ("On", "Off") and hasattr(params, "flash_attn_type"):
        fa_enum = _resolve_enum(engine, "llama_flash_attn_type", fa_member)
        if fa_enum is not None:
            params.flash_attn_type = fa_enum

    batch_size = int(settings.get("llm_batch_size") or 0)
    if batch_size > 0 and hasattr(params, "n_batch"):
        params.n_batch = batch_size
        if hasattr(params, "n_ubatch"):
            current_ubatch = int(getattr(params, "n_ubatch", 0) or 0)
            if current_ubatch > batch_size:
                params.n_ubatch = batch_size

    mmap_mode = str(settings.get("llm_mmap", "Auto") or "Auto")
    if mmap_mode in ("Enabled", "Disabled"):
        if hasattr(params, "load_mode"):
            mode_enum = _resolve_enum(engine, "llama_load_mode", _LOAD_MODES[mmap_mode])
            if mode_enum is not None:
                params.load_mode = mode_enum
        elif hasattr(params, "use_mmap"):
            params.use_mmap = mmap_mode == "Enabled"

    if hasattr(params, "moe_cache_cpu_share"):
        if cpu:
            params.moe_cache_cpu_share = 0.0
        elif settings.get("llama_moe_cache_cpu_share_enabled", False) is True:
            try:
                share = float(settings.get("llama_moe_cache_cpu_share", 0.5))
            except (TypeError, ValueError):
                share = None
            if share is not None and 0.0 <= share <= 1.0:
                params.moe_cache_cpu_share = share

    if hasattr(params, "reasoning_format"):
        rf_enum = _resolve_enum(engine, "common_reasoning_format", "COMMON_REASONING_FORMAT_AUTO")
        if rf_enum is not None:
            params.reasoning_format = rf_enum
    if hasattr(params, "enable_reasoning"):
        params.enable_reasoning = -1

    reasoning_budget = _parse_flag_int(native_defaults, {"--reasoning-budget"})
    if (
        reasoning_budget is not None
        and hasattr(params, "sampling")
        and hasattr(params.sampling, "reasoning_budget_tokens")
    ):
        params.sampling.reasoning_budget_tokens = reasoning_budget

    params.ctx_shift = True
    params.endpoint_metrics = False
    params.use_jinja = True


def admit_xllamacpp_devices(engine, choice="Auto", *, cpu=False):
    """Verify initialized devices before creating a model server.

    Query failures propagate; they are not silently converted into a CPU retry.
    A CPU-only result under Auto has an explicit reason and CPU offload controls.
    """
    import re

    if cpu or choice == "CPU":
        return "CPU", True, ""
    devices = engine.get_device_info()
    labels = tuple(
        dict.fromkeys(
            re.sub(r"\d+$", "", item["name"])
            for item in devices
            if re.match(r"^(Vulkan|CUDA|ROCm|SYCL|Metal)\d*$", item["name"])
        )
    )
    if choice != "Auto" and choice.split()[0] not in labels:
        raise RuntimeError(
            f"xllamacpp has no usable {choice} device; select another backend explicitly"
        )
    if not labels:
        return (
            "CPU",
            True,
            "No usable GPU is visible to the cached xllamacpp wheel",
        )
    return "/".join(labels), False, ""
