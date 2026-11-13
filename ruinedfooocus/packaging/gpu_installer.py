"""PyTorch and llama.cpp package selection."""

import importlib.metadata
import base64
import hashlib
import json
import os
import re
import sys
import subprocess
import sysconfig
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from ruinedfooocus.packaging.runtime_support import installed_rocm_platform
from ruinedfooocus.packaging.runtime_support import installed_cuda_platform
from ruinedfooocus.packaging.runtime_bundles import bundle_for


def preserve_installed_cuda(bundle, torch_platform, reinstall=False):
    from packaging.version import Version

    return bool(
        bundle
        and not reinstall
        and installed_cuda_platform(bundle) == torch_platform
        and Version(bundle["versions"]["torch"]) >= Version("2.6.0")
    )


def installed_torch_constraints():
    versions = {}
    packages = ["torch", "torchvision"]
    if os.environ.get("TORCH_PLATFORM") == "directml":
        packages += ["numpy", "scipy", "torch-directml"]
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    # Keep the wheel-owned vendor runtime closure, not just the visible
    # Torch companions. An overlay must not upgrade HIP/CUDA libraries behind
    # an unchanged Torch version or redirect them to an older system SDK.
    for distribution in importlib.metadata.distributions():
        name = (distribution.metadata.get("Name") or "").lower().replace("_", "-")
        if name.startswith(("rocm-sdk-", "nvidia-", "pytorch-triton", "triton")) or name == "rocm":
            versions[name] = distribution.version
    # Transformers 5.17 disables its torch integration on Torch 2.4, which
    # Microsoft DirectML still requires. Preserve the last verified integration.
    if versions.get("torch", "").split(".")[:2] == ["2", "4"]:
        versions["transformers"] = "5.13.0"
    return {"versions": versions} if versions else None


def preserve_installed_rocm(installed_rocm, torch_platform, reinstall=False):
    if not installed_rocm or reinstall:
        return False
    # An explicit different ROCm version is a request to change environments.
    return torch_platform == installed_rocm_platform(installed_rocm)


@contextmanager
def torch_constraints(installed_rocm):
    """Keep vendor builds intact when resolving transitive dependencies."""
    if not installed_rocm:
        yield ""
        return
    with TemporaryDirectory(prefix="ruinedfooocus-torch-") as directory:
        path = Path(directory) / "constraints.txt"
        path.write_text(
            "".join(f"{name}=={version}\n" for name, version in installed_rocm["versions"].items()),
            encoding="utf-8",
        )
        yield f' --constraint "{path}"'


def torch_install_commands(torchruntime, torch_platform, os_platform, nightly=False):
    if torch_platform == "xpu" and sys.version_info[:2] < bundle_for("xpu").python_min:
        raise RuntimeError("Intel XPU PyTorch wheels need Python 3.11 or newer.")
    if torch_platform in ("cpu", "xpu"):
        # Pinned like CUDA/ROCm; torchruntime would install the newest XPU wheels.
        bundle = bundle_for(torch_platform)
        return [
            [
                f"torch=={bundle.torch}",
                f"torchvision=={bundle.vision}",
                "--index-url",
                bundle.index,
            ]
        ]
    if torch_platform == "directml":
        import platform

        if os_platform not in ("Windows", "Linux") or not (3, 10) <= sys.version_info[:2] <= (
            3,
            12,
        ):
            raise RuntimeError("DirectML requires Python 3.10–3.12 on Windows or WSL.")
        if platform.machine().lower() not in ("amd64", "x86_64"):
            raise RuntimeError("The published DirectML wheels require x86_64 Python.")
        if os_platform == "Linux" and "microsoft" not in platform.release().lower():
            raise RuntimeError(
                "DirectML requires Windows or WSL, not native Linux. Use --cpu or your GPU's native backend."
            )
        return [
            [
                "torch==2.4.1+cpu",
                "torchvision==0.19.1+cpu",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            ["--no-deps", "torch-directml==0.2.5.dev240914", "numpy==1.26.4", "scipy==1.15.3"],
        ]
    if torch_platform == "cu124" and not (3, 10) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError(
            "CUDA 12.4 requires Python 3.10–3.13 in RuinedFooocus. Use Python 3.10 or 3.12 for GTX 1080 Ti."
        )
    # Keep the selected CUDA runtime: newer wheels may drop older GPU support.
    from ruinedfooocus.packaging.cuda_selection import (
        CUDA_PACKAGES,
        cuda_index,
        cuda_wheels_available,
        validate_cuda_runtime,
    )

    if os_platform not in ("Windows", "Linux") or torch_platform not in (*CUDA_PACKAGES, "cu134"):
        return torchruntime.installer.get_install_commands(torch_platform, [])
    validate_cuda_runtime(torch_platform, nightly=nightly)
    if not cuda_wheels_available(torch_platform, nightly=nightly):
        raise RuntimeError(
            f"No {torch_platform} wheel bundle matches Python {sys.version.split()[0]} and this OS/CPU architecture."
        )
    if nightly:
        return [
            [
                "--pre",
                "--upgrade",
                "--force-reinstall",
                "--only-binary=:all:",
                "torch",
                "torchvision",
                "--index-url",
                cuda_index(torch_platform, nightly),
            ],
        ]
    torch, vision = CUDA_PACKAGES[torch_platform]
    packages = [f"torch=={torch}+{torch_platform}", f"torchvision=={vision}+{torch_platform}"]
    commands = [packages + ["--index-url", f"https://download.pytorch.org/whl/{torch_platform}"]]
    # Windows Triton is optional; an unbounded version breaks older torch builds.
    return commands




























