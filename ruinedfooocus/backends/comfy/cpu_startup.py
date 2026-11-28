"""Suppress accelerator discovery only while embedding native CPU startup."""

from contextlib import contextmanager, ExitStack
import importlib
import sys
from functools import wraps


@contextmanager
def cpu_discovery_disabled():
    """Restore every temporary change before startup can create service threads.

    The pinned model manager discovers XPU/MPS/NPU/MLU before applying args.cpu.
    Its CPU option therefore needs this narrow import boundary; CPU tensor APIs
    and later independent services retain the original Torch behavior.
    """
    import torch

    missing = object()
    with ExitStack() as restore:
        for owner, name, value in (
            (torch.cuda, "is_available", False),
            (torch.cuda, "device_count", 0),
            (getattr(torch, "xpu", None), "is_available", False),
            (getattr(torch, "xpu", None), "device_count", 0),
            (getattr(torch.backends, "mps", None), "is_available", False),
        ):
            if owner is not None and hasattr(owner, name):
                original = getattr(owner, name)

                @wraps(original)
                def disabled(*args, _value=value, **kwargs):
                    return _value

                # Dynamo reads this cached MPS query's undecorated identity.
                if hasattr(original, "__wrapped__"):
                    disabled.__wrapped__ = original.__wrapped__
                restore.callback(setattr, owner, name, original)
                setattr(owner, name, disabled)
        for name in ("torch_npu", "torch_mlu"):
            original = sys.modules.get(name, missing)
            if original is missing:
                restore.callback(sys.modules.pop, name, None)
            else:
                restore.callback(sys.modules.__setitem__, name, original)
            # None refuses import even when the extension was already cached.
            sys.modules[name] = None
        yield


def initialize_cpu_model_management():
    """Called once by serialized embedded startup, before any service threads."""
    from comfy.cli_args import args

    if not args.cpu:
        raise RuntimeError("Native CPU startup requires the finalized CPU image selection.")
    # comfy-kitchen imports Dynamo while importing the model manager. Its static
    # function-identity registry must capture real Torch callables, not the
    # temporary discovery wrappers used below. This import performs no probing.
    importlib.import_module("torch._dynamo")
    with cpu_discovery_disabled():
        management = importlib.import_module("comfy.model_management")
        if management.cpu_state != management.CPUState.CPU:
            raise RuntimeError(
                "The native image runtime is already initialized for a GPU; restart in CPU mode."
            )
        # ops has its own unconditional CUDA availability check during import.
        # Import it here too, so later feature imports cannot trigger that probe.
        importlib.import_module("comfy.ops")
        # Native attention asks comfy-kitchen about optional GPU kernels too.
        importlib.import_module("comfy.ldm.modules.attention")
    return management
