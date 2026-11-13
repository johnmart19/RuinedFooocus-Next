"""CUDA selection using GPU architecture, driver and this interpreter's wheel tags."""

import re
import csv
import os
import shutil
from pathlib import Path
import subprocess
import sys
from functools import lru_cache
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen
from urllib.error import URLError

from packaging.tags import sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from ruinedfooocus.packaging.runtime_bundles import BUNDLES, bundle_for


CUDA_PACKAGES = {
    runtime: (bundle.torch, bundle.vision)
    for runtime in BUNDLES
    if runtime.startswith("cu")
    for bundle in (bundle_for(runtime),)
}


def cuda_index(runtime, nightly=False):
    channel = f"nightly/{runtime}" if nightly else runtime
    return f"https://download.pytorch.org/whl/{channel}"


@lru_cache(maxsize=32)
def wheel_available(index, package, version=None):
    tags = set(sys_tags())
    try:
        with urlopen(f"{index.rstrip('/')}/{package}/", timeout=30) as response:
            page = response.read().decode("utf-8")
    except (URLError, OSError) as error:
        raise RuntimeError(
            f"Could not check {package} wheels at {index}: {error}. Retry when the index is reachable."
        ) from error
    for link in re.findall(r"href=[\"\']([^\"\']+)", page):
        filename = unquote(urlsplit(link).path).rsplit("/", 1)[-1]
        try:
            _, release, _, wheel_tags = parse_wheel_filename(filename)
        except InvalidWheelFilename:
            continue
        if (version is None or release.base_version == version) and tags.intersection(wheel_tags):
            return True
    return False


def cuda_wheels_available(runtime, nightly=False):
    versions = (None, None) if nightly else CUDA_PACKAGES.get(runtime, (None, None))
    index = cuda_index(runtime, nightly)
    return all(
        wheel_available(index, name, version)
        for name, version in zip(("torch", "torchvision"), versions)
    )


def select_cuda_nightly(os_platform, override=None):
    if os_platform not in ("Windows", "Linux"):
        raise RuntimeError("--cuda-nightly requires Windows or Linux.")
    capabilities, driver = nvidia_devices()
    capability = min(capabilities) if capabilities else None
    if capability is None or driver is None:
        raise RuntimeError(
            "--cuda-nightly requires an NVIDIA GPU and working nvidia-smi detection."
        )
    if capability < 7.5:
        raise RuntimeError(
            "Current CUDA nightlies require Turing or newer. Keep CUDA 12.6/12.4 for GTX 1080 Ti and other Pascal GPUs."
        )
    candidates = ["cu134", "cu132", "cu130"]
    if override:
        if override not in candidates:
            raise RuntimeError(
                "TORCH_PLATFORM conflicts with --cuda-nightly; remove it or select cu130, cu132 or cu134."
            )
        candidates = [override]
    for candidate in candidates:
        if (int(candidate[2:-1]), int(candidate[-1])) <= driver and cuda_wheels_available(
            candidate, nightly=True
        ):
            print(
                f"CUDA nightly: {candidate}; Python {sys.version.split()[0]}; GPU capability {capability}"
            )
            return candidate
    raise RuntimeError(
        f"No supported CUDA nightly bundle matches Python {sys.version.split()[0]}, this OS/CPU architecture and driver CUDA {driver}. Existing packages were not replaced."
    )


def select_cuda_legacy(os_platform, override=None, *, install=True):
    """Explicit compatibility line, checked before any package replacement."""
    if os_platform not in ("Windows", "Linux"):
        raise RuntimeError("--legacy requires NVIDIA on Windows or Linux/WSL.")
    if override and override not in ("cu126", "cu124"):
        raise RuntimeError("TORCH_PLATFORM conflicts with --legacy; use cu126/cu124 or unset it.")
    if not install:
        import importlib.metadata

        from packaging.version import Version

        try:
            version = Version(importlib.metadata.version("torch"))
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(
                "--legacy needs installed CUDA 12.6/12.4 wheels when installation is disabled."
            ) from None
        if version.local not in ("cu126", "cu124") or version >= Version("2.15"):
            raise RuntimeError(
                "Installed Torch is outside the legacy CUDA profile. Restart online without --skip-dependency-install/freezetorch to install it."
            )
        if override and override != version.local:
            raise RuntimeError(
                "Installed legacy CUDA differs from TORCH_PLATFORM and installation is disabled."
            )
        override = version.local
    capabilities, driver = nvidia_devices()
    if not capabilities or driver is None or any(value < 5 for value in capabilities):
        raise RuntimeError(
            "--legacy requires detected NVIDIA GPUs with compute capability 5.0 or newer and a working driver."
        )
    candidates = [override] if override else ["cu126", "cu124"]
    for runtime in candidates:
        if all(runtime in cuda_candidates(value, driver) for value in capabilities) and (
            not install or cuda_wheels_available(runtime)
        ):
            return runtime
    raise RuntimeError(
        "No legacy CUDA bundle matches the selected GPUs, driver and Python wheel tags. No packages were replaced."
    )


def nvidia_devices():
    executable = shutil.which("nvidia-smi")
    if executable is None:
        # Driver utilities are not always on PATH, especially in WSL.
        candidates = []
        if sys.platform == "win32":
            candidates = [
                Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/nvidia-smi.exe",
                Path(
                    os.environ.get(
                        "ProgramW6432", os.environ.get("ProgramFiles", "C:/Program Files")
                    )
                )
                / "NVIDIA Corporation/NVSMI/nvidia-smi.exe",
            ]
        elif sys.platform == "linux":
            candidates = [Path("/usr/lib/wsl/lib/nvidia-smi")]
        executable = next((str(path) for path in candidates if path.is_file()), "nvidia-smi")
    try:
        output = subprocess.check_output(
            [executable], text=True, timeout=15, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.SubprocessError):
        return (), None
    cuda = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", output)
    driver = (int(cuda[1]), int(cuda[2])) if cuda else None
    try:
        output = subprocess.check_output(
            [executable, "--query-gpu=uuid,pci.bus_id,compute_cap", "--format=csv,noheader"],
            text=True,
            timeout=15,
            stderr=subprocess.DEVNULL,
        )
        devices = []
        for row in csv.reader(output.splitlines()):
            if not row:
                continue
            if len(row) != 3:
                raise RuntimeError(
                    "Unexpected nvidia-smi device response. Update the NVIDIA driver."
                )
            uuid, bus, capability = (value.strip() for value in row)
            # An actual adapter with unavailable capability must not become CPU.
            capability = float(capability) if re.fullmatch(r"\d+\.\d+", capability) else 0.0
            devices.append((uuid, bus, capability))
        if not devices:
            return (), driver
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            selectors = [value.strip() for value in visible.split(",")]
            if visible.strip() in ("", "-1"):
                raise RuntimeError(
                    "CUDA_VISIBLE_DEVICES hides NVIDIA GPUs. Use --cpu explicitly or unset it."
                )
            selected = []
            for selector in selectors:
                if selector.startswith("GPU-"):
                    matches = [device for device in devices if device[0].startswith(selector)]
                elif selector.isdigit():
                    if len(devices) > 1 and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
                        raise RuntimeError(
                            "For multiple NVIDIA GPUs, use GPU UUIDs in CUDA_VISIBLE_DEVICES "
                            "or set CUDA_DEVICE_ORDER=PCI_BUS_ID before selecting numeric devices."
                        )
                    ordered = sorted(devices, key=lambda device: device[1])
                    index = int(selector)
                    matches = ordered[index : index + 1]
                else:
                    matches = []
                if len(matches) != 1:
                    raise RuntimeError(
                        "Cannot resolve CUDA_VISIBLE_DEVICES to one NVIDIA GPU per selector. "
                        "Use full GPU UUIDs from nvidia-smi -L; MIG setup needs a custom environment."
                    )
                selected.extend(matches)
            devices = selected
        return tuple(device[2] for device in devices), driver
    except (OSError, subprocess.SubprocessError):
        # Older drivers may not expose compute_cap. Keep actual NVIDIA presence
        # distinct from a leftover driver header, especially without PCI in WSL.
        try:
            identities = subprocess.check_output(
                [executable, "--query-gpu=uuid", "--format=csv,noheader"],
                text=True,
                timeout=15,
                stderr=subprocess.DEVNULL,
            )
            return tuple(
                0.0 for line in identities.splitlines() if line.strip().startswith("GPU-")
            ), driver
        except (OSError, subprocess.SubprocessError):
            return (), driver


def nvidia_info():
    capabilities, driver = nvidia_devices()
    return min(capabilities) if capabilities else None, driver


def cuda_candidates(capability, driver_cuda=None):
    if capability < 5:
        return []
    candidates = ["cu126", "cu124"] if capability < 7.5 else ["cu134", "cu132", "cu130", "cu128"]
    if 7.5 <= capability < 10:
        candidates += ["cu126", "cu124"]
    if capability == 11.0 or capability >= 12.1:
        candidates = [
            "cu134",
            "cu132",
            "cu130",
        ]  # Includes GB10/Spark; Python tags decide ARM wheel availability.
    if driver_cuda:
        candidates = [c for c in candidates if (int(c[2:-1]), int(c[-1])) <= driver_cuda]
    return candidates


def validate_cuda_runtime(runtime, nightly=False):
    """Validate explicit reinstall choices before pip can replace working wheels."""
    capabilities, driver = nvidia_devices()
    if not capabilities or driver is None or any(capability == 0 for capability in capabilities):
        raise RuntimeError(
            "Cannot validate the selected CUDA runtime without NVIDIA GPU and driver information."
        )
    if nightly:
        supported = min(capabilities) >= 7.5 and (int(runtime[2:-1]), int(runtime[-1])) <= driver
    else:
        supported = all(
            runtime in cuda_candidates(capability, driver) for capability in capabilities
        )
    if not supported:
        raise RuntimeError(
            f"{runtime} is incompatible with the selected NVIDIA GPUs or driver. "
            "Remove TORCH_PLATFORM to select a compatible bundle; no packages were replaced."
        )


def select_cuda(gpus):
    capabilities, driver = nvidia_devices()
    if not capabilities or any(capability == 0 for capability in capabilities):
        raise RuntimeError(
            "Cannot determine every selected NVIDIA GPU's compute capability. "
            "Check nvidia-smi and update the driver; no packages were replaced."
        )
    if min(capabilities) < 5:
        raise RuntimeError(
            "NVIDIA GPUs older than Maxwell need a legacy environment; "
            "this application's CUDA bundles require compute capability 5.0 or newer."
        )
    if driver is None:
        raise RuntimeError(
            "Cannot determine the NVIDIA driver's CUDA version. Check nvidia-smi before installing."
        )
    print(
        f"NVIDIA compute capabilities: {capabilities}; driver CUDA: {driver}; Python: {sys.version.split()[0]}"
    )
    candidates = cuda_candidates(capabilities[0], driver)
    candidates = [
        candidate
        for candidate in candidates
        if all(candidate in cuda_candidates(capability, driver) for capability in capabilities)
    ]
    if not candidates:
        raise RuntimeError(
            "No CUDA bundle supports all selected NVIDIA GPUs and this driver. "
            "Select compatible GPUs with CUDA_VISIBLE_DEVICES (GPU UUIDs) or update the driver."
        )
    for candidate in candidates:
        if cuda_wheels_available(candidate):
            return candidate
    raise RuntimeError(
        f"No compatible CUDA wheel bundle for Python {sys.version.split()[0]}, "
        f"GPU compute capabilities {capabilities}, driver CUDA {driver}. "
        "Use a supported Python environment or update the NVIDIA driver."
    )
