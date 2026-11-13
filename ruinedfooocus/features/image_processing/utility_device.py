"""Worker-only native Torch policy for request-local image utility operations."""

from dataclasses import dataclass
import logging
import os
import sys

from ruinedfooocus.core.device_fallback import warn_cpu_fallback


@dataclass(frozen=True)
class UtilityPolicy:
    device: object
    intermediate_device: object
    intermediate_dtype: object
    requested: str = "auto"
    fallback_reason: str = ""
    cpu_only_reason: str = ""

    def metadata(self):
        return {
            "requested": self.requested,
            "device": str(self.device),
            "backend": "cpu" if _is_cpu(self.device) else "native-torch",
            "intermediate_device": str(self.intermediate_device),
            "intermediate_dtype": str(self.intermediate_dtype),
            "fallback": bool(self.fallback_reason),
            "fallback_reason": self.fallback_reason,
            "cpu_only_reason": self.cpu_only_reason,
        }


@dataclass(frozen=True)
class UtilityResult:
    value: object
    policy: UtilityPolicy

    def metadata(self):
        return self.policy.metadata()


def _is_cpu(device):
    return str(getattr(device, "type", device)).split(":", 1)[0] == "cpu"


def _explicit_cpu():
    """Read applied startup authority without importing arguments or discovering devices."""
    if "--cpu" in sys.argv:
        return True
    launch = getattr(sys.modules.get("ruinedfooocus.app.arguments"), "args", None)
    status = getattr(launch, "_compute_status", {})
    backend = status.get("backend", {})
    if (
        "--cpu" in status.get("cli", ())
        or backend.get("cpu_origin") in ("cli", "environment", "saved")
        and getattr(launch, "cpu", False)
    ):
        return True
    if getattr(launch, "cpu", False) and backend.get("cpu_origin") == "automatic":
        return False
    if getattr(launch, "cpu", False) and status:
        return bool(backend.get("cpu_applied") or backend.get("cpu_everywhere"))
    if status:
        return False
    if os.environ.get("TORCH_PLATFORM", "").lower() == "cpu":
        return True
    for name in ("ruinedfooocus.app.arguments", "comfy.cli_args"):
        args = getattr(sys.modules.get(name), "args", None)
        if getattr(args, "cpu", False):
            return True
    return False


def _cpu_policy(*, requested="auto", fallback_reason="", cpu_only_reason=""):
    import torch

    return UtilityPolicy(
        torch.device("cpu"),
        torch.device("cpu"),
        torch.float32,
        requested=requested,
        fallback_reason=fallback_reason,
        cpu_only_reason=cpu_only_reason,
    )


def _native_model_cpu_reason(model_management):
    state = getattr(model_management, "vram_state", None)
    name = getattr(state, "name", str(state) if state is not None else "unknown")
    if name in ("LOW_VRAM", "NO_VRAM", "DISABLED"):
        return f"native model placement selected CPU under the {name} memory policy"
    if getattr(model_management, "FORCE_FP32", False):
        return "native model placement selected CPU under the forced FP32 text-encoder precision policy"
    return f"native text-encoder precision/capacity policy selected CPU (VRAM policy: {name})"


def resolve_policy(*, cpu_only_reason=None, native_model_placement=False):
    """Resolve inside a worker; optionally retain native model placement policy."""
    if _explicit_cpu():
        return _cpu_policy(requested="cpu", cpu_only_reason=cpu_only_reason or "")
    if cpu_only_reason:
        return _cpu_policy(cpu_only_reason=cpu_only_reason)
    try:
        import torch
        from comfy import model_management

        device = model_management.get_torch_device()
        if _is_cpu(device):
            return _cpu_policy(fallback_reason="the configured native PyTorch runtime selected CPU")
        if native_model_placement:
            device = model_management.text_encoder_device()
            if _is_cpu(device):
                return _cpu_policy(cpu_only_reason=_native_model_cpu_reason(model_management))
        return UtilityPolicy(
            device,
            model_management.intermediate_device(),
            torch.float32,
        )
    except (ImportError, OSError, RuntimeError, NotImplementedError) as error:
        reason = f"{type(error).__name__}: {error}"
    return _cpu_policy(fallback_reason=reason)


def _device_failure(error):
    if isinstance(error, NotImplementedError) or type(error).__name__ == "OutOfMemoryError":
        return True
    if isinstance(error, OSError) and (
        error.errno is not None
        or isinstance(
            error, (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError)
        )
    ):
        return False
    message = str(error).lower()
    return any(
        token in message
        for token in (
            "cuda",
            "hip error",
            "rocm",
            "directml",
            "privateuse",
            "mps",
            "xpu",
            "out of memory",
            "cudnn",
            "cublas",
            "miopen",
            "hipblas",
            "kernel image",
            "illegal memory access",
            "gpu runtime",
            "gpu memory",
            "gpu allocation",
            "gpu device",
            "device lost",
            "device-side",
            "unsupported device",
            "device type",
            "same device",
        )
    )


def _require_cpu_model_offload(policy):
    if not _is_cpu(policy.device):
        return
    from comfy import model_management

    offload = model_management.text_encoder_offload_device()
    if not _is_cpu(offload):
        raise RuntimeError(
            "CPU recovery is unavailable while native model offload remains "
            f"on {offload}. Disable GPU-only and use Auto, or launch with --cpu."
        )


def run_utility(
    feature, operation, *, check_cancelled, cpu_only_reason=None, native_model_placement=False
):
    """Execute a fresh callback; retry one failed accelerator attempt on CPU.

    The callback must reconstruct its inputs from captured data and include output
    validation and host transfer, where asynchronous device errors can surface.
    The helper never resets global backend state or shared caches. Native-model
    callbacks own cleanup of failed request-owned model caches before rebuilding.
    """
    check_cancelled()
    policy = (
        resolve_policy(cpu_only_reason=cpu_only_reason, native_model_placement=True)
        if native_model_placement
        else resolve_policy(cpu_only_reason=cpu_only_reason)
    )
    check_cancelled()
    if native_model_placement:
        _require_cpu_model_offload(policy)
    if policy.fallback_reason:
        warn_cpu_fallback(feature, policy.fallback_reason)
    elif policy.cpu_only_reason and policy.requested != "cpu":
        logging.warning("%s is running on CPU: %s.", feature, policy.cpu_only_reason)
    try:
        value = operation(policy)
    except (OSError, RuntimeError, NotImplementedError) as error:
        if _is_cpu(policy.device) or not _device_failure(error):
            raise
        check_cancelled()
        reason = f"{type(error).__name__}: {error}"
    else:
        check_cancelled()
        return UtilityResult(value, policy)
    # Leave the exception scope before rebuilding inputs, releasing its traceback
    # and any failed attempt's request-local tensors without global cache resets.
    policy = _cpu_policy(fallback_reason=reason)
    if native_model_placement:
        _require_cpu_model_offload(policy)
    warn_cpu_fallback(feature, reason)
    check_cancelled()
    value = operation(policy)
    check_cancelled()
    return UtilityResult(value, policy)
