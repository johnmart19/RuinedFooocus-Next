"""Host prerequisites for GPU JIT kernels; no accelerator imports or probes."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


def ensure_gpu_jit_toolchain(*, enabled):
    if not enabled or sys.platform != "linux":
        return
    from importlib.util import find_spec

    if find_spec("triton") is None:
        return
    compiler = os.environ.get("CC")
    found = shutil.which(compiler) if compiler else shutil.which("gcc") or shutil.which("clang")
    if compiler and not found:
        raise RuntimeError(f"The configured GPU JIT C compiler is unavailable: {compiler}")
    headers = Path(sysconfig.get_path("include")) / "Python.h"
    packages = []
    if not found:
        packages.extend(("gcc", "libc6-dev"))
    if not headers.is_file():
        packages.append(f"python{sys.version_info.major}.{sys.version_info.minor}-dev")
    if not packages:
        return
    if not shutil.which("apt-get"):
        raise RuntimeError(
            "GPU JIT kernels require a C compiler and development headers matching "
            f"{sys.executable}. Install the equivalents of: {' '.join(packages)}"
        )
    elevate = [] if os.geteuid() == 0 else ["sudo"]
    print("Installing GPU JIT prerequisites (administrator access required): " + " ".join(packages))
    subprocess.run([*elevate, "apt-get", "update"], check=True)
    subprocess.run([*elevate, "apt-get", "install", "-y", *packages], check=True)
    found = shutil.which(compiler) if compiler else shutil.which("gcc") or shutil.which("clang")
    if not found or not headers.is_file():
        raise RuntimeError(
            "GPU JIT prerequisites are still missing for the selected Python; "
            f"check its compiler and header path {headers}"
        )
