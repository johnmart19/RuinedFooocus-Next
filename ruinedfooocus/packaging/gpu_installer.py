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






























