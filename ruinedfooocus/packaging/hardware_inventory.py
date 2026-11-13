"""Detected hardware and the backend choices valid for it.

Nothing here initialises CUDA/HIP, loads a model or imports PyTorch: GPUs come
from a PCI scan (``torchruntime.device_db``), and installed runtimes are read from package metadata
and the llama.cpp cache directory.
"""

import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess

from ruinedfooocus.core.settings_schema import ONNX_BACKENDS, XLLAMACPP_BACKENDS

VENDORS = {"10de": "nvidia", "1002": "amd", "8086": "intel"}
VENDOR_LABELS = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel", "other": "Other"}

_gpus = None
_gpu_scan_state = "unknown"
# Windows host query used from WSL, where lspci shows only the virtual adapter.
_HOST_QUERY = (
    "Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID | ConvertTo-Json"
)
_WINDOWS_POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"


def _vendor(vendor_id):
    return VENDORS.get(str(vendor_id or "").lower(), "other")


def is_wsl(system=None, release=None):
    system = system or platform.system()
    release = (release if release is not None else platform.release()).lower()
    return system == "Linux" and ("microsoft" in release or "wsl" in release)


def os_kind(system=None, release=None):
    """Windows, WSL, Linux, or the reported system name; WSL is a separate install target."""
    system = system or platform.system()
    return "WSL" if is_wsl(system, release) else system


def identify_adapters(gpus, *, location="native", source="pci"):
    """Preserve duplicate cards and keep inventory IDs distinct from runtime IDs.

    A PCI product ID is not a physical identity. Ordinal fallback IDs explicitly
    have ``identity_stable=False`` and must be rebound at runtime verification.
    """
    found, occurrences = [], {}
    for item in gpus:
        gpu = dict(item)
        gpu.setdefault("location", location)
        gpu.setdefault("source", source)
        gpu.setdefault("kind", "gpu")
        key = (gpu["location"], gpu.get("pci_id", "unknown"))
        ordinal = occurrences.get(key, 0)
        occurrences[key] = ordinal + 1
        hardware_id = gpu.get("hardware_id")
        gpu.setdefault(
            "adapter_id", f"{gpu['location']}:{hardware_id or key[1] + ':' + str(ordinal)}"
        )
        gpu.setdefault("identity_stable", bool(hardware_id))
        if gpu["location"] == "host":
            gpu.setdefault("guest_available", None)
        found.append(gpu)
    return found


def parse_host_gpus(text):
    """Parse ``Win32_VideoController`` JSON; virtual and basic display adapters are skipped."""
    try:
        rows = json.loads(text or "[]")
    except ValueError:
        return []
    rows = [rows] if isinstance(rows, dict) else rows if isinstance(rows, list) else []
    found = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        match = re.search(
            r"VEN_([0-9a-f]{4})&DEV_([0-9a-f]{4})", str(row.get("PNPDeviceID") or ""), re.I
        )
        if not match or match[1].lower() not in VENDORS:
            continue
        found.append(
            {
                "vendor": _vendor(match[1]),
                "name": str(row.get("Name") or "GPU"),
                "pci_id": f"{match[1]}:{match[2]}".lower(),
                "hardware_id": str(row.get("PNPDeviceID")),
            }
        )
    return identify_adapters(found, location="host", source="windows-host")


def wsl_host_gpus():
    """GPUs of the Windows host as seen from WSL, through Windows interop; [] if unavailable."""
    executable = shutil.which("powershell.exe") or (
        _WINDOWS_POWERSHELL if os.path.exists(_WINDOWS_POWERSHELL) else None
    )
    if not executable:
        return []
    try:
        result = subprocess.run(
            [executable, "-NoProfile", "-NonInteractive", "-Command", _HOST_QUERY],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return parse_host_gpus(result.stdout) if result.returncode == 0 else []


def detect_gpus(refresh=False):
    """Cached passive adapter inventory; never runs nvidia-smi or compute checks."""
    global _gpus, _gpu_scan_state
    if _gpus is not None and not refresh:
        return [dict(gpu) for gpu in _gpus]
    found = []
    try:
        from torchruntime.device_db import get_gpus

        for gpu in get_gpus():
            hardware_id = getattr(gpu, "instance_id", None) or getattr(gpu, "pci_bus_id", None)
            found.append(
                {
                    "vendor": _vendor(gpu.vendor_id),
                    "name": str(gpu.device_name),
                    "pci_id": f"{gpu.vendor_id}:{gpu.device_id}".lower(),
                    **({"hardware_id": str(hardware_id)} if hardware_id else {}),
                }
            )
        _gpu_scan_state = "detected" if found else "absent"
    except Exception:
        # Detection is informational; an unreadable PCI list means "unknown".
        _gpu_scan_state = "unknown"
    found = identify_adapters(found, location="guest" if is_wsl() else "native")
    if is_wsl():
        # WSL shows only the Microsoft virtual adapter; the host lists the real GPUs.
        found = [gpu for gpu in found if not gpu["pci_id"].startswith("1414:")]
        # A host card is useful information, not proof of guest availability.
        # Do not deduplicate product IDs: identical adapters remain independent.
        found += identify_adapters(wsl_host_gpus(), location="host", source="windows-host")
        if found:
            _gpu_scan_state = "partial" if _gpu_scan_state == "unknown" else "detected"
    _gpus = found
    return [dict(gpu) for gpu in found]


def gpu_inventory(refresh=False):
    """Keep an unreadable scan distinct from an empty, successful inventory."""
    adapters = detect_gpus(refresh)
    return {"state": _gpu_scan_state, "adapters": adapters}


def has_vendor(gpus, vendor):
    return any(gpu.get("vendor") == vendor for gpu in gpus)


def installed_torch_kind(version=None):
    """Classify the installed torch build from its version string: cuda, rocm, cpu or None."""
    if version is None:
        try:
            version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            return None
    version = version.lower()
    if "+rocm" in version:
        return "rocm"
    if "+cu" in version:
        return "cuda"
    if "+xpu" in version:
        return "xpu"
    if "+cpu" in version:
        return "cpu"
    return "other"


def installed_runtimes(os_platform=None):
    """Cheap facts about the installed image runtime (no torch import)."""
    os_platform = os_platform or platform.system()
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        version = None
    directml = False
    if os_platform in ("Windows", "Linux"):
        try:
            from ruinedfooocus.packaging.runtime_support import (
                installed_directml_matches_requirements,
            )

            directml = installed_directml_matches_requirements(os_platform)
        except Exception:
            directml = False
    return {"torch": version, "kind": installed_torch_kind(version), "directml": directml}


def image_backend_choices(gpus, os_platform=None, installed=None, machine=None, *, decisions=None):
    """Installation catalogue; with decisions, only exact eligible routes.

    Without artifact decisions vendor matches are candidates, not usable devices.
    Saved unavailable choices must be retained by their profile/UI caller.
    """
    os_platform = os_platform or platform.system()
    machine = (machine or platform.machine()).lower()
    installed = installed or {}
    kind = installed.get("kind")
    supported_os = os_platform in ("Windows", "Linux")
    x64 = machine in ("amd64", "x86_64")
    choices = ["auto"]
    if decisions is not None:
        return choices + [
            name
            for name in ("cuda", "rocm", "xpu", "cpu", "directml")
            if any(
                row["operation"] == "image" and row["backend"] == name and row["eligible"]
                for row in decisions
            )
        ]
    if supported_os and (has_vendor(gpus, "nvidia") or kind == "cuda"):
        choices.append("cuda")
    # The exact GFX target is validated when the switch is applied; the UI never
    # calls the AMD installer helpers because they may install a detection package.
    if supported_os and ((has_vendor(gpus, "amd") and x64) or kind == "rocm"):
        choices.append("rocm")
    # Intel Arc / Core Ultra GPUs use the official PyTorch XPU wheels (x64 only).
    if supported_os and ((has_vendor(gpus, "intel") and x64) or kind == "xpu"):
        choices.append("xpu")
    choices.append("cpu")
    if installed.get("directml"):
        choices.append("directml")
    return choices


def llm_backend_choices(gpus, system=None, machine=None, *, decisions=None):
    """llama.cpp backends from the installer, filtered to what this hardware can use."""
    if decisions is not None:
        return ["Auto"] + list(
            dict.fromkeys(
                row["backend"]
                for row in decisions
                if row["operation"] == "llama" and row["eligible"]
            )
        )
    from ruinedfooocus.backends.llama.llama_installer import backend_choices

    choices = backend_choices(system, machine)
    nvidia, amd = has_vendor(gpus, "nvidia"), has_vendor(gpus, "amd")
    intel = has_vendor(gpus, "intel")
    result = []
    for choice in choices:
        if choice in ("Auto", "CPU"):
            result.append(choice)
        elif choice == "Vulkan":
            if gpus:
                result.append(choice)
        elif choice == "ROCm":
            if amd:
                result.append(choice)
        elif choice == "SYCL":
            if intel:
                result.append(choice)
        elif "CUDA" in choice:
            # The local-build label is already filtered to a finished build.
            if nvidia:
                result.append(choice)
        else:
            result.append(choice)
    return result


def _x64(machine=None):
    return (machine or platform.machine()).lower() in ("amd64", "x86_64")


def onnx_backend_choices(gpus=(), system=None, machine=None, *, decisions=None):
    """Background-removal runtimes this hardware can use (stable values).

    Vendor ONNX Runtime builds (CUDA, MIGraphX, OpenVINO) are x64 wheels installed
    into an isolated folder; Windows GPU covers Windows ML and the deprecated DirectML.
    """
    if decisions is not None:
        return ["Auto"] + list(
            dict.fromkeys(
                row["backend"]
                for row in decisions
                if row["operation"] == "onnx" and row["eligible"]
            )
        )
    system = system or platform.system()
    native = system in ("Windows", "Linux") and _x64(machine)
    allowed = {"Auto", "CPU"}
    if native and has_vendor(gpus, "nvidia"):
        allowed.add("CUDA")
    # MIGraphX needs a native ROCm install; AMD does not support it on WSL.
    if native and system == "Linux" and not is_wsl(system) and has_vendor(gpus, "amd"):
        allowed.add("MIGraphX")
    if native and has_vendor(gpus, "intel"):
        allowed.add("OpenVINO")
    if system == "Windows":
        allowed.add("Windows GPU")
    return [choice for choice in ONNX_BACKENDS if choice in allowed]


# xllamacpp publishes one wheel index per backend; ROCm wheels are Linux-only.
XLLAMACPP_INDEXES = {
    "CPU": None,
    "Vulkan": "vulkan",
    "CUDA 12.8": "cu128",
    "CUDA 13.2": "cu132",
    "ROCm 7.2": "rocm-7.2.4",
}


def rocm10_torch(version=None):
    """True when the installed PyTorch is a ROCm 10 build (from metadata, no import)."""
    if version is None:
        try:
            version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            return False
    return bool(re.search(r"\+rocm1\d", version))


def xllamacpp_backend_choices(
    gpus=(), system=None, machine=None, torch_version=None, *, decisions=None
):
    """xllamacpp wheel backends valid for this hardware (stable values).

    The newest xllamacpp ROCm wheel is built for ROCm 7.2. Loaded beside ROCm 10
    PyTorch in the same process it crashes, so it is offered only without ROCm 10
    PyTorch; native llama.cpp has the ROCm 10 build.
    """
    if decisions is not None:
        return ["Auto"] + list(
            dict.fromkeys(
                row["backend"]
                for row in decisions
                if row["operation"] == "xllamacpp" and row["eligible"]
            )
        )
    system = system or platform.system()
    native = system in ("Windows", "Linux") and _x64(machine)
    allowed = {"Auto", "CPU"}
    if native and gpus:
        allowed.add("Vulkan")
    if native and has_vendor(gpus, "nvidia"):
        allowed.update(("CUDA 12.8", "CUDA 13.2"))
    if native and system == "Linux" and has_vendor(gpus, "amd") and not rocm10_torch(torch_version):
        allowed.add("ROCm 7.2")
    return [choice for choice in XLLAMACPP_BACKENDS if choice in allowed]


def xllamacpp_installed():
    """``{"version", "backend"}`` of the installed xllamacpp wheel, or None.

    The backend comes from the index recorded at installation; an untracked install
    reports ``None`` as its backend.
    """
    try:
        version = importlib.metadata.version("xllamacpp")
    except importlib.metadata.PackageNotFoundError:
        return None
    backend = None
    try:
        from ruinedfooocus.core.paths import data_path

        marker = data_path("cache", "llama.cpp", "xllamacpp-backend.json")
        index = json.loads(marker.read_text(encoding="utf-8")).get("index", "")
        suffix = index.rstrip("/").rsplit("/", 1)[-1]
        backend = next(
            (name for name, value in XLLAMACPP_INDEXES.items() if value == suffix),
            "CPU" if "pypi.org" in index else None,
        )
    except (OSError, ValueError, AttributeError):
        pass
    return {"version": version, "backend": backend}


def llama_cached_servers(system=None, machine=None):
    """Mapping of backend name to cached ``llama-server`` path."""
    from ruinedfooocus.backends.llama import llama_installer as installer

    system = system or platform.system()
    machine = machine or platform.machine()
    executable = "llama-server.exe" if system == "Windows" else "llama-server"
    found = {}
    for backend in installer.backend_choices(system, machine):
        if backend in ("Auto",) or backend.startswith("CUDA (local"):
            continue
        try:
            package = installer.package_name(system, machine, True, False, backend)
        except RuntimeError:
            continue
        directory = installer.package_directory(package)
        if directory.is_dir():
            matches = list(directory.rglob(executable))
            if matches:
                found[backend] = str(matches[0].resolve())
    return found


def llama_installed(system=None, machine=None):
    """Backends whose official llama.cpp package already sits in the cache."""
    return list(llama_cached_servers(system, machine))


def component_status(os_platform=None):
    """Installed or active state per component, without loading models or probing devices."""
    os_platform = os_platform or platform.system()
    installed = installed_runtimes(os_platform)
    try:
        llama_paths = llama_cached_servers(os_platform)
    except Exception:
        llama_paths = {}
    try:
        onnx = importlib.metadata.version("onnxruntime")
    except importlib.metadata.PackageNotFoundError:
        onnx = None
    return {
        "image": {
            "torch": installed["torch"],
            "kind": installed["kind"],
            "directml": installed["directml"],
            "platform_override": os.environ.get("TORCH_PLATFORM") or None,
        },
        "llm": {
            "server_override": os.environ.get("LLAMA_SERVER") or None,
            "installed": list(llama_paths),
            "paths": llama_paths,
        },
        "background": {"onnxruntime": onnx, "windows_ml": windows_ml_provider()},
    }


def windows_ml_provider():
    """Name of the installed Windows ML provider from saved evidence, else None."""
    try:
        from ruinedfooocus.backends.onnx.windows_ml import installed_providers

        record = next(iter(installed_providers(verify_payload=False)), None)
    except Exception:
        return None
    return record["name"] if record else None
