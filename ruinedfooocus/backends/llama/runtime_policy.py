"""Shared LLM acquisition preference and explicit CPU offload controls."""


def preferred_backend(system, choice="Auto", *, gpu=True):
    """Prefer Vulkan independently of image Torch; activation must verify devices."""
    if not gpu or choice == "CPU":
        return "CPU"
    if choice == "Auto" and system in ("Windows", "Linux"):
        return "Vulkan"
    return choice


def configure_xllamacpp_offload(params, settings, *, cpu=False):
    """Enforce CPU even when a GPU-capable extension is already cached/imported."""
    params.n_gpu_layers = 0 if cpu else int(settings.get("llm_n_gpu_layers", -1))
    if cpu:
        params.no_kv_offload = True
        params.no_op_offload = True
        params.mmproj_use_gpu = False


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
