"""Saved compute-backend choices and their precedence at startup.

Order of authority: explicit CLI device flags, then ``TORCH_PLATFORM`` /
``LLAMA_SERVER``, then the saved profile, then automatic selection. This module
is pure policy: it never probes a device, imports PyTorch or installs anything
unless a caller runs ``apply_saved_image_backend`` during bootstrap.
"""

import os

from ruinedfooocus.core.settings_schema import IMAGE_BACKENDS
from ruinedfooocus.packaging.hardware_inventory import installed_torch_kind

# Flags that already express an image-device decision; a saved choice never overrides them.
CLI_DEVICE_ATTRIBUTES = (
    "directml",
    "rocm10",
    "cuda_nightly",
    "legacy",
    "gpu_device_id",
    "cuda_device",
    "default_device",
    "oneapi_device_selector",
    "gpu_only",
    "highvram",
    "lowvram",
    "novram",
    "normalvram",
)


def saved_image_backend(values):
    """The saved image backend; the old ``image_device: CPU`` only counts when none is saved."""
    backend = values.get("image_backend")
    if backend in IMAGE_BACKENDS:
        return backend
    return "cpu" if values.get("image_device") == "CPU" else "auto"


def cpu_everywhere(values):
    return values.get("cpu_everywhere") is True


def saved_cpu_requested(values):
    return cpu_everywhere(values) or saved_image_backend(values) == "cpu"


def effective_backends(values):
    """Backends that apply now. ``cpu_everywhere`` hides, but never rewrites, the saved choices."""
    forced = cpu_everywhere(values)
    return {
        "forced_cpu": forced,
        "image": "cpu" if forced else saved_image_backend(values),
        "llama": "CPU" if forced else values.get("llama_backend", "Auto"),
        "rembg": "CPU" if forced else values.get("rembg_backend", "Auto"),
        "xllamacpp": "CPU" if forced else values.get("xllamacpp_backend", "Auto"),
    }


def cli_overrides(args, environment=None):
    """Names of explicit startup choices that win over a saved image backend."""
    names = [
        name
        for name in CLI_DEVICE_ATTRIBUTES
        if getattr(args, name, None) is not None and getattr(args, name, None) is not False
    ]
    if "--cpu" in getattr(args, "_compute_status", {}).get("cli", ()):
        names.insert(0, "--cpu")
    torch_platform = (
        getattr(args, "_user_torch_platform", os.environ.get("TORCH_PLATFORM"))
        if environment is None
        else environment.get("TORCH_PLATFORM")
    )
    if torch_platform:
        names.append("TORCH_PLATFORM")
    return names


def torch_switch_decision(values, installed_kind, *, reinstall_pending, overrides=()):
    """Decide what a saved CUDA/ROCm choice means for this startup.

    Returns ``{"action": ..., "backend": ..., "note": ...}`` where action is one of
    ``none`` (nothing to do), ``overridden``, ``pending`` (switch not requested yet)
    or ``apply`` (reinstall marker present: resolve the platform and switch).
    """
    backend = "cpu" if cpu_everywhere(values) else saved_image_backend(values)
    if backend in ("auto", "cpu"):
        return {"action": "none", "backend": backend, "note": ""}
    if overrides:
        return {
            "action": "overridden",
            "backend": backend,
            "note": f"Saved image backend {backend} is overridden by {', '.join(overrides)}.",
        }
    if backend == installed_kind:
        return {"action": "none", "backend": backend, "note": ""}
    if backend == "directml":
        return {"action": "apply", "backend": backend, "note": ""}
    if not reinstall_pending:
        return {
            "action": "pending",
            "backend": backend,
            "note": (
                f"Saved image backend {backend} differs from the installed "
                f"{installed_kind or 'missing'} torch build. Keeping it: use Settings > "
                "Hardware & backends to request the switch for the next online restart."
            ),
        }
    return {"action": "apply", "backend": backend, "note": ""}


def record_decision(args, **decision):
    status = getattr(args, "_compute_status", None)
    if status is not None:
        status.setdefault("backend", {}).update(decision)


def apply_saved_image_backend(args, os_platform, *, reinstall):
    """Bootstrap hook: map a saved CUDA/ROCm/DirectML choice to ``TORCH_PLATFORM``.

    CPU needs no platform change (installed torch runs on CPU) and is applied while
    arguments are parsed. A different GPU build is only installed when the user
    already requested a reinstall.
    """
    from ruinedfooocus.core.profile_store import read_profile
    from ruinedfooocus.packaging.startup_options import saved_profile_path

    # This hook runs after application arguments are initialized. Unlike the
    # pre-install updater policy, backend decisions retain full profile validation.
    values = read_profile(saved_profile_path())
    decision = torch_switch_decision(
        values,
        installed_torch_kind(),
        reinstall_pending=reinstall,
        overrides=cli_overrides(args),
    )
    record_decision(args, **decision)
    if decision["note"]:
        print(decision["note"])
    if decision["action"] != "apply":
        return decision
    backend = decision["backend"]
    if backend == "directml":
        from ruinedfooocus.packaging.runtime_support import installed_directml_matches_requirements

        if installed_directml_matches_requirements(os_platform):
            os.environ["TORCH_PLATFORM"] = "directml"
        return decision
    if backend == "xpu":
        os.environ["TORCH_PLATFORM"] = "xpu"
        print("Saved image backend xpu: using torch platform xpu.")
        return decision
    if backend == "cuda":
        import torchruntime
        from ruinedfooocus.packaging.cuda_selection import select_cuda

        platform_name = select_cuda(torchruntime.device_db.get_gpus())
    else:
        from ruinedfooocus.packaging.rocm_installer import ROCM_PLATFORM, automatic_rocm_plan

        plan = automatic_rocm_plan(os_platform, ROCM_PLATFORM, force=True)
        if plan is None:
            print(
                "Saved ROCm image backend is unavailable on this system; keeping automatic selection."
            )
            return decision
        platform_name = plan.platform
    os.environ["TORCH_PLATFORM"] = platform_name
    print(f"Saved image backend {backend}: using torch platform {platform_name}.")
    return decision


def force_cpu_for_auxiliary(args):
    """Distinguish an image-only CPU selection from an explicit global override."""
    if not getattr(args, "cpu", False):
        return False
    status = getattr(args, "_compute_status", {})
    backend = status.get("backend", {})
    if "--cpu" in status.get("cli", ()) or backend.get("cpu_everywhere"):
        return True
    scope = backend.get("cpu_scope")
    if scope in ("global", "image"):
        return scope == "global"
    if backend.get("cpu_origin") == "automatic":
        return False
    # A recorded startup has provenance: args.cpu may have been set later by
    # automatic image fallback. It cannot widen into a global CPU request.
    if status and not backend.get("cpu_applied"):
        return False
    # Older callers without provenance retain the conservative CPU contract.
    return not (backend.get("saved_image_backend") == "cpu" and backend.get("cpu_applied"))


def cpu_selection_scope(*, cli_cpu=False, environment=None, values=None, automatic_image_cpu=False):
    """Resolve CPU origin/scope before mutating bootstrap arguments.

    CLI wins over TORCH_PLATFORM, which wins over saved preferences. Automatic
    image fallback is image-only even when bootstrap also sets ``args.cpu``.
    """
    environment = {} if environment is None else environment
    values = {} if values is None else values
    if cli_cpu:
        return {"cpu_scope": "global", "cpu_origin": "cli"}
    if environment.get("TORCH_PLATFORM"):
        return {
            "cpu_scope": "image" if environment["TORCH_PLATFORM"] == "cpu" else "none",
            "cpu_origin": "environment",
        }
    if cpu_everywhere(values):
        return {"cpu_scope": "global", "cpu_origin": "saved"}
    if saved_image_backend(values) == "cpu":
        return {"cpu_scope": "image", "cpu_origin": "saved"}
    return {"cpu_scope": "image" if automatic_image_cpu else "none", "cpu_origin": "automatic"}
