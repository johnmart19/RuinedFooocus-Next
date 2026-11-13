"""Compatibility with the pinned ComfyUI backend."""

import atexit
import os
import platform
import logging
import math
import subprocess
import sys
import json
from functools import lru_cache

from ruinedfooocus.core.device_fallback import warn_cpu_fallback


_runtime_platform = None
_aimdo_log_detach_registered = False


def initialize_backend(launch_args):
    """Configure once before metadata scanning can import backend/GPU modules."""
    global _runtime_platform
    if _runtime_platform is not None:
        return _runtime_platform
    os_platform = platform.system()
    if launch_args.cpu:
        torch_platform = "cpu"
    else:
        import torchruntime
        from ruinedfooocus.packaging.runtime_support import (
            inspect_installed_rocm,
            inspect_installed_cuda,
            select_torch_platform,
        )

        torch_platform = select_torch_platform(
            torchruntime,
            os_platform,
            inspect_installed_rocm(os_platform),
            inspect_installed_cuda(os_platform),
        )
        if torch_platform == "cpu":
            warn_cpu_fallback(
                "ComfyUI image processing", "no compatible PyTorch GPU backend was detected"
            )
    if torch_platform.startswith("rocm"):
        # AIMDO reserves a large virtual address space; match native startup
        # before Torch or the controller loads the ROCm runtime.
        os.environ["OCL_SET_SVM_SIZE"] = "4194304"
    import comfy.cli_args

    from ruinedfooocus.backends.comfy.image_compute import finalize_compute

    finalize_compute(launch_args, cpu=torch_platform == "cpu")
    configure_comfy_runtime(launch_args, comfy.cli_args.args)
    if torch_platform == "cpu":
        comfy.cli_args.args.cpu = True
    if launch_args.directml is not None:
        comfy.cli_args.args.directml = launch_args.directml
    elif torch_platform == "directml":
        comfy.cli_args.args.directml = -1
    configure_torch_allocator(torch_platform, os_platform, launch_args, native=True)
    if reason := unavailable_accelerator(torch_platform):
        warn_cpu_fallback("ComfyUI image processing", reason)
        torch_platform = "cpu"
        finalize_compute(launch_args, cpu=True)
        configure_comfy_runtime(launch_args, comfy.cli_args.args)
    configure_torch_compatibility()
    if torch_platform == "cpu":
        from ruinedfooocus.backends.comfy.cpu_startup import initialize_cpu_model_management

        initialize_cpu_model_management()
    initialize_comfy_memory()
    _runtime_platform = (torch_platform, os_platform)
    return _runtime_platform


def unavailable_accelerator(torch_platform):
    """Check the installed backend after applying the launcher's device masks."""
    if torch_platform.startswith(("cu", "rocm")):
        import torch

        native = torch.version.hip if torch_platform.startswith("rocm") else torch.version.cuda
        if not native:
            return f"the installed PyTorch build does not provide {torch_platform}"
        if not torch.cuda.is_available():
            return f"the installed {torch_platform} runtime reports no visible GPU"
    elif torch_platform in ("xpu", "mps"):
        import torch

        runtime = getattr(torch, "xpu", None) if torch_platform == "xpu" else torch.backends.mps
        if runtime is None or not runtime.is_available():
            return f"the installed {torch_platform} runtime reports no visible GPU"
    return None


VISIBILITY_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ONEAPI_DEVICE_SELECTOR",
    "ZE_AFFINITY_MASK",
    "GGML_VK_VISIBLE_DEVICES",
)


def _native_mask_backend(args):
    from ruinedfooocus.packaging.hardware_inventory import installed_torch_kind

    requested = os.environ.get("TORCH_PLATFORM", "")
    if getattr(args, "rocm10", False) or requested.startswith("rocm"):
        return "hip"
    if (
        getattr(args, "legacy", False)
        or getattr(args, "cuda_nightly", False)
        or requested.startswith("cu")
    ):
        return "cuda"
    kind = installed_torch_kind()
    return {"rocm": "hip", "cuda": "cuda"}.get(kind)


def _physical_device_count(backend):
    """Bounded child enumeration; the live parent's inherited masks never change."""
    environment = dict(os.environ)
    native_keys = ("CUDA_VISIBLE_DEVICES",)
    if backend == "hip":
        native_keys += ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
    for key in native_keys:
        environment.pop(key, None)
    script = (
        "import json,sys; backend=sys.argv[1]; caps=(); "
        "\nif backend == 'cuda':\n from ruinedfooocus.packaging.cuda_selection import nvidia_devices\n caps,_=nvidia_devices()"
        "\nif caps:\n print(json.dumps({'count':len(caps),'backend':'cuda'}))"
        "\nelse:\n import torch\n print(json.dumps({'count':torch.cuda.device_count(),'backend':'hip' if torch.version.hip else 'cuda'}))"
    )
    try:
        output = subprocess.check_output(
            [sys.executable, "-c", script, backend],
            env=environment,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        result = json.loads(output.strip().splitlines()[-1])
        count = result["count"]
        if result["backend"] != backend or type(count) is not int or count < 0:
            raise ValueError("Child enumeration differs from the selected native backend")
        if backend == "cuda" and count > 1 and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
            raise ValueError(
                "--default-device on multiple NVIDIA GPUs requires CUDA_DEVICE_ORDER=PCI_BUS_ID."
            )
        return count
    except (OSError, subprocess.SubprocessError, KeyError, ValueError, IndexError) as error:
        raise ValueError(
            "Cannot enumerate physical GPUs for the selected native backend; preserve inherited masks and select an explicit compatible runtime/device before startup. "
            + str(error)
        ) from error


def configure_device_environment(args):
    """Apply GPU visibility before importing Torch, matching backend startup."""
    for name in ("reserve_vram", "vram_headroom", "async_offload"):
        value = getattr(args, name, None)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative.")
    inherited = {key: os.environ[key] for key in VISIBILITY_KEYS if key in os.environ}
    report = {"inherited": inherited, "requested": {}, "effective": dict(inherited)}
    args._device_environment_status = report
    status = getattr(args, "_compute_status", None)
    if status is not None:
        status.setdefault("backend", {})["device_environment"] = report
    if getattr(args, "cpu", False) or os.environ.get("TORCH_PLATFORM") == "cpu":
        # The image backend's CPU choice must not enumerate adapters or rewrite
        # visibility inherited by independent Chat/ONNX services.
        return
    devices = args.cuda_device
    if devices is not None and devices != "all":
        if not all(part.isdecimal() for part in devices.split(",")):
            raise ValueError("--cuda-device expects comma-separated GPU indices or all")
    if args.default_device is not None and not 0 <= args.default_device < 32:
        raise ValueError("--default-device expects an index from 0 to 31")
    selecting = (
        devices not in (None, "all")
        or args.gpu_device_id is not None
        or args.default_device is not None
    )
    backend = _native_mask_backend(args) if selecting else None
    if selecting and backend is None:
        raise ValueError(
            "Select an installed/requested CUDA or ROCm runtime before using its numeric GPU selectors"
        )
    if (selecting or args.oneapi_device_selector is not None) and (
        _runtime_platform is not None
        or hasattr(sys.modules.get("comfy.model_management"), "cpu_state")
    ):
        raise RuntimeError(
            "Native device visibility is already initialized; restart to apply a different selection"
        )
    if devices is None and args.gpu_device_id is None and args.default_device is not None:
        count = _physical_device_count(backend)
        if args.default_device >= count:
            raise ValueError(
                f"--default-device {args.default_device} is unavailable; found {count} GPU(s)."
            )
        order = list(range(count))
        order.remove(args.default_device)
        devices = ",".join(map(str, [args.default_device, *order]))
    if args.gpu_device_id is not None:
        if args.gpu_device_id < 0:
            raise ValueError("--gpu-device-id expects a non-negative index")
        devices = str(args.gpu_device_id)
    if devices is not None and devices != "all":
        report["requested"][
            "HIP_VISIBLE_DEVICES" if backend == "hip" else "CUDA_VISIBLE_DEVICES"
        ] = devices
    if args.oneapi_device_selector is not None:
        report["requested"]["ONEAPI_DEVICE_SELECTOR"] = args.oneapi_device_selector
    conflicts = [
        key
        for key, value in report["requested"].items()
        if key in inherited and inherited[key] != value
    ]
    if backend == "hip":
        for key in ("ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
            if (
                key in inherited
                and report["requested"].get("HIP_VISIBLE_DEVICES") != inherited[key]
            ):
                conflicts.append(key)
    if conflicts:
        report["conflicts"] = tuple(conflicts)
        raise ValueError(
            "Inherited device-mask conflict: "
            + ", ".join(conflicts)
            + ". Resolve it before startup; no live masks were changed."
        )
    os.environ.update(report["requested"])
    report["effective"].update(report["requested"])
    # Match native startup defaults only after request/conflict validation.
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    if platform.system() == "Windows":
        os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
    elif "microsoft" in platform.release().lower():
        os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def child_device_environment(environment=None, *, service):
    """Undo only proven RF image-startup masks in an independent child copy."""
    environment = dict(os.environ if environment is None else environment)
    arguments = sys.modules.get("ruinedfooocus.app.arguments")
    args = getattr(arguments, "args", None)
    report = getattr(args, "_device_environment_status", {})
    inherited = report.get("inherited", {})
    for key, generated in report.get("requested", {}).items():
        if environment.get(key) != generated:
            raise RuntimeError(
                f"{service}: device mask {key} changed after native startup; explicitly restart/verify its process policy"
            )
        if key in inherited:
            environment[key] = inherited[key]
        else:
            environment.pop(key, None)
    report.setdefault("children", {})[service] = {
        key: environment[key] for key in VISIBILITY_KEYS if key in environment
    }
    return environment


def configure_comfy_runtime(launch_args, comfy_args):
    """Apply embedded-backend flags before model management is imported."""
    comfy_args.offline = launch_args.offline or os.environ.get("RF_OFFLINE") == "1"
    if comfy_args.offline:
        comfy_args.disable_partner_nodes = True
    from ruinedfooocus.backends.comfy.comfy_args import COMFY_DESTINATIONS
    from comfy.cli_args import PerformanceFeature, LatentPreviewMethod

    # The embedded backend defaults to no previews; the app displays them.
    comfy_args.preview_method = LatentPreviewMethod.from_string(
        getattr(launch_args, "preview_method", "auto")
    )

    for name in COMFY_DESTINATIONS:
        setattr(comfy_args, name, getattr(launch_args, name))
    if launch_args.normalvram:
        comfy_args.disable_dynamic_vram = True
        comfy_args.enable_dynamic_vram = False
    if comfy_args.force_fp16:
        comfy_args.fp16_unet = True
    if comfy_args.high_ram:
        comfy_args.cache_classic = True
    if comfy_args.disable_comfy_compiler:
        comfy_args.disable_cuda_graphs = True
    features = launch_args.fast
    comfy_args.fast = (
        set()
        if features is None
        else set(PerformanceFeature)
        if not features
        else {PerformanceFeature(x.value) for x in features}
    )


def initialize_comfy_memory():
    """Initialize the pinned DynamicVRAM controller without starting its server."""
    from comfy.cli_args import args, enables_dynamic_vram

    if args.cpu or args.directml is not None or not enables_dynamic_vram():
        return False
    import comfy_aimdo.control as control

    headroom = None if args.reserve_vram is None else int(args.reserve_vram * 1024**3)
    try:
        if not control.init(
            simple_vram_headroom=headroom, nvml_pressure=not args.disable_nvml_pressure
        ):
            logging.warning("DynamicVRAM controller unavailable; using legacy model loading.")
            return False
    except Exception as error:
        logging.warning("DynamicVRAM controller unavailable; using legacy model loading: %s", error)
        return False

    import comfy.model_management as management
    import comfy.memory_management as memory
    import comfy.model_patcher as patcher

    supported = (
        args.directml is None
        and not args.cpu
        and (management.is_nvidia() or (management.is_amd() and management.rocm_version >= (7, 14)))
    )
    if not supported or management.torch_version_numeric < (2, 8):
        logging.warning(
            "DynamicVRAM requires NVIDIA or ROCm HIP 7.14+, and Torch 2.8+; using legacy model loading."
        )
        return False
    try:
        devices = [
            (device.index, int(args.vram_headroom * 1024**3))
            for device in management.get_all_torch_devices()
        ]
        if not devices:
            logging.warning(
                "DynamicVRAM found no visible devices in PyTorch. Check the selected GPU "
                "and device visibility settings. Using legacy model loading."
            )
            return False
        initialized = control.init_devices(devices)
    except Exception as error:
        logging.warning(
            "DynamicVRAM device initialization failed; using legacy model loading: %s", error
        )
        return False
    if not initialized:
        logging.warning(
            "The DynamicVRAM controller rejected PyTorch device indices %s. "
            "Check the preceding comfy-aimdo messages. Using legacy model loading.",
            ", ".join(str(index) for index, _ in devices),
        )
        return False
    set_aimdo_log_level(control, getattr(args, "verbose", []))
    patcher.CoreModelPatcher = patcher.ModelPatcherDynamic
    memory.aimdo_enabled = True
    register_aimdo_log_shutdown(control, platform.system(), management.is_amd())
    logging.info("DynamicVRAM support detected and enabled")
    return True


def set_aimdo_log_level(control, verbose):
    """Match the controller's native log level to the backend console level, as native startup does."""
    import comfy.cli_args

    levels = getattr(comfy.cli_args, "get_console_log_level", lambda outputs: "INFO")
    setter = {
        "DEBUG": "set_log_debug",
        "DETAIL": "set_log_detail",
        "CRITICAL": "set_log_critical",
        "ERROR": "set_log_error",
        "WARNING": "set_log_warning",
    }.get(levels(verbose), "set_log_info")
    # Older controllers lack DETAIL; native startup falls back to INFO as well.
    setter = getattr(control, setter, None) or getattr(control, "set_log_info", None)
    if setter is not None:
        setter()


def register_aimdo_log_shutdown(control, os_platform, is_amd):
    """Detach Windows ROCm's Python log callback before late solver teardown.

    rocBLAS/hipSOLVER handles can release memory during native Py_Exit, after
    Python module globals (including control._log_callback) have been cleared.
    Keep the callback alive until atexit and clear only the native callback
    pointer; AIMDO device and memory owners must retain their normal lifetimes.
    """
    global _aimdo_log_detach_registered
    if _aimdo_log_detach_registered or os_platform != "Windows" or not is_amd:
        return False
    library = getattr(control, "lib", None)
    callback_type = getattr(control, "_LOG_CALLBACK", None)
    active_callback = getattr(control, "_log_callback", None)
    if library is None or callback_type is None or active_callback is None:
        return False
    import ctypes

    setter = library.set_log_callback
    null_callback = ctypes.cast(None, callback_type)

    # Capture dependencies now: late native logging must never enter a freed
    # CFUNCTYPE thunk, even if control's module globals are cleared afterwards.
    def detach(setter=setter, null_callback=null_callback, keepalive=active_callback):
        setter(null_callback)

    atexit.register(detach)
    _aimdo_log_detach_registered = True
    return True


def configure_torch_allocator(torch_platform, os_platform, launch_args=None, *, native=False):
    if torch_platform == "cpu" and launch_args is not None:
        from ruinedfooocus.backends.comfy.image_compute import finalize_compute

        # Bootstrap can resolve CPU after the saved preferences were applied.
        # Suppress those flags before they alter the allocator environment.
        finalize_compute(launch_args, cpu=True)
    if launch_args is not None and (launch_args.cuda_malloc or launch_args.disable_cuda_malloc):
        backend = "cudaMallocAsync" if launch_args.cuda_malloc else "native"
        config = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_ALLOC_CONF", ""))
        parts = [
            part
            for part in config.split(",")
            if part and not part.startswith(("backend:", "expandable_segments:"))
        ]
        parts.extend([f"backend:{backend}", "expandable_segments:False"])
        os.environ["PYTORCH_ALLOC_CONF"] = ",".join(parts)
        if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ["PYTORCH_ALLOC_CONF"]
        return
    if torch_platform.startswith("cu"):
        if "PYTORCH_CUDA_ALLOC_CONF" in os.environ or "PYTORCH_ALLOC_CONF" in os.environ:
            return
        if not native:
            # Bootstrap has not configured Comfy's arguments/source path yet.
            return
        import runpy
        from ruinedfooocus.core.paths import APP_ROOT

        # Reuse the pinned policy: CUDA >=13 defaults, legacy GPU blacklist,
        # autotune conflicts and explicit disable handling all stay upstream.
        runpy.run_path(str(APP_ROOT / "repositories/ComfyUI/cuda_malloc.py"))
        generated = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        os.environ["PYTORCH_ALLOC_CONF"] = generated or "expandable_segments:True"
        return
    # ROCDXG cannot transfer model-sized tensors using expandable allocations.
    wsl_rocm = (
        os_platform == "Linux"
        and torch_platform.startswith("rocm")
        and "microsoft" in platform.release().lower()
    )
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ.setdefault(
            "PYTORCH_ALLOC_CONF",
            "expandable_segments:False" if wsl_rocm else "expandable_segments:True",
        )


def configure_rocm_attention():
    """Keep the gfx11 wide-head safeguard without overriding other AMD GPUs.

    Torch 2.14 + ROCm 10.1 on gfx11 dispatches Flash/efficient kernels for these
    shapes, which then fail to launch (hipErrorProfilerNotInitialized). Gemma 4's
    512-wide global attention (LTX 2.5's text encoder) hits this; CUDA already
    falls back to math. Smaller heads keep the fast kernels.
    """
    import torch
    import torch.nn.functional as functional

    if not torch.version.hip or getattr(
        functional.scaled_dot_product_attention, "_rf_rocm_head_dim", False
    ):
        return False
    from torch.nn.attention import sdpa_kernel, SDPBackend

    original = functional.scaled_dot_product_attention

    @lru_cache(maxsize=32)
    def needs_math(device):
        # Like native ComfyUI's VAE guard, key on the tensor's actual device.
        # Encoders also call Torch SDPA directly, outside that VAE policy.
        arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
        return not arch or arch.startswith("gfx11")

    def scaled_dot_product_attention(query, key, value, *args, **kwargs):
        if query.shape[-1] > 256 and query.is_cuda and needs_math(query.device):
            with sdpa_kernel(SDPBackend.MATH):
                return original(query, key, value, *args, **kwargs)
        return original(query, key, value, *args, **kwargs)

    scaled_dot_product_attention._rf_rocm_head_dim = True
    functional.scaled_dot_product_attention = scaled_dot_product_attention
    return True


def configure_torch_compatibility():
    import torch
    import typing

    configure_rocm_attention()
    if not (2, 4) <= tuple(int(v) for v in torch.__version__.split(".")[:2]) <= (2, 6):
        return
    # Torch 2.4–2.6 understand typing.List but not list[T] in custom-op schemas.
    # Newer comfy-kitchen uses the latter; both describe the same tensor operation.
    from torch._library.infer_schema import (
        SUPPORTED_PARAM_TYPES,
        SUPPORTED_RETURN_TYPES,
    )

    for supported in (SUPPORTED_PARAM_TYPES, SUPPORTED_RETURN_TYPES):
        for annotation, schema in list(supported.items()):
            if typing.get_origin(annotation) is list:
                supported.setdefault(list[typing.get_args(annotation)[0]], schema)
    if tuple(int(v) for v in torch.__version__.split(".")[:2]) == (2, 4):
        # DirectML's torch predates resolution of postponed custom-op annotations.
        import torch._custom_op.impl as custom_ops

        if not getattr(custom_ops.infer_schema, "_rf_annotations", False):
            original = custom_ops.infer_schema

            def infer_schema(fn, *args, **kwargs):
                annotations = fn.__annotations__
                try:
                    fn.__annotations__ = typing.get_type_hints(fn)
                    if fn.__annotations__.get("return") is type(None):
                        fn.__annotations__["return"] = None
                    return original(fn, *args, **kwargs)
                finally:
                    fn.__annotations__ = annotations

            infer_schema._rf_annotations = True
            custom_ops.infer_schema = infer_schema
