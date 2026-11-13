"""Inspect installed chat runtimes without downloading or loading models."""

import json
from os import scandir
from ruinedfooocus.core.paths import data_path
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ruinedfooocus.core.bounded_subprocess import run_bounded


MAX_BUILDS = 8
MAX_OUTPUT = 64 * 1024
REFRESH_SECONDS = 45


def runtime_choices():
    from ruinedfooocus.app.state import state

    pipeline = state.get("pipeline")
    loaded = getattr(pipeline, "runtime_name", None) if getattr(pipeline, "llm", None) else None
    label = getattr(pipeline, "runtime_label", None) or "loading"
    return [
        (f"{name} [{label if name == loaded else 'not loaded'}]", name)
        for name in ("llama.cpp", "xllamacpp")
    ]


def _run(command, env=None):
    stdout, stderr = run_bounded(
        command,
        timeout=15,
        max_output=16 * 1024,
        env=env,
    )
    return (stdout + b"\n" + stderr).decode("utf-8", errors="replace")


def _installed_binaries():
    override = os.environ.get("LLAMA_SERVER")
    if override:
        return [Path(shutil.which(override) or override)], False
    root = data_path("cache", "llama.cpp")
    executable = "llama-server.exe" if os.name == "nt" else "llama-server"
    pending = [(root, 0)]
    found = []
    inspected = 0
    truncated = False
    while pending and len(found) <= MAX_BUILDS and inspected < 1024:
        directory, depth = pending.pop()
        try:
            with scandir(directory) as entries:
                for entry in entries:
                    inspected += 1
                    if inspected > 1024:
                        truncated = True
                        break
                    if entry.is_symlink() or entry.name == "build-work":
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if depth < 5 and (depth > 0 or entry.name.startswith("b")):
                            pending.append((Path(entry.path), depth + 1))
                    elif entry.name == executable and entry.is_file(follow_symlinks=False):
                        found.append(Path(entry.path))
                        if len(found) > MAX_BUILDS:
                            break
        except OSError:
            continue
    return sorted(found[:MAX_BUILDS]), truncated or bool(pending) or len(found) > MAX_BUILDS


def _build_label(binary):
    if os.environ.get("LLAMA_SERVER"):
        return "Custom llama.cpp: " + binary.name
    root = data_path("cache", "llama.cpp")
    try:
        relative = binary.relative_to(root)
        build = relative.parts[0]
    except ValueError:
        build = "cached build"
    backend = (
        "CUDA (local build)" if binary.parent.parent.name == "local-cuda" else binary.parent.name
    )
    return build + " / " + backend


def runtime_metadata():
    """Package/cache identity only, safe for explicit global CPU mode."""
    binaries, truncated = _installed_binaries()
    lines = [
        "**llama.cpp:** "
        + (", ".join(_build_label(binary) for binary in binaries) or "not installed")
    ]
    try:
        from ruinedfooocus.packaging.hardware_inventory import xllamacpp_installed

        installed = xllamacpp_installed()
        lines.append(
            f"**xllamacpp:** {installed['version']}; recorded wheel backend: {installed['backend'] or 'unknown'}"
            if installed
            else "**xllamacpp:** not installed."
        )
    except (OSError, ValueError):
        lines.append("**xllamacpp:** package identity unavailable.")
    if truncated:
        lines.append("Cached build listing limited to eight entries.")
    return "\n\n".join(lines)


def runtime_info(*, cpu=False, check_cancelled=None):
    """One owned refresh, including discovery/SDK queries, has a fixed budget."""
    if check_cancelled is not None:
        check_cancelled()
    if cpu:
        return runtime_metadata() + "\n\nCPU authority is active; device queries were not run."
    root = str(Path(__file__).resolve().parents[3])
    script = f"import sys; sys.path.insert(0, {root!r}); from ruinedfooocus.backends.llama.llama_runtime_info import _inspect_runtime_info; print(_inspect_runtime_info())"
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    output, _ = run_bounded(
        [sys.executable, "-I", "-c", script],
        env=environment,
        timeout=REFRESH_SECONDS,
        max_output=MAX_OUTPUT,
        check_cancelled=check_cancelled,
    )
    return output.decode("utf-8", errors="replace")


def _inspect_runtime_info():
    sections = []
    override = os.environ.get("LLAMA_SERVER")
    binaries, truncated = _installed_binaries()
    if not binaries:
        sections.append("**llama.cpp:** not installed; downloads on first use.")
    for binary in binaries:
        label = _build_label(binary)
        try:
            env = os.environ.copy()
            if not override and "rocm" in str(binary):
                if os.name == "nt":
                    from ruinedfooocus.backends.llama.llama_installer import (
                        windows_rocm_environment,
                    )

                    env = windows_rocm_environment(env, install=False, query=_run)
                else:
                    from ruinedfooocus.backends.llama.llama_installer import (
                        active_rocm_directory,
                    )

                    directory = active_rocm_directory(data_path("cache", "llama.cpp", "rocm-10.0"))
                    sites = directory.glob("lib/python*/site-packages")
                    paths = [
                        str(site / package / "lib")
                        for site in sites
                        for package in ("_rocm_sdk_core", "_rocm_sdk_libraries")
                        if (site / package / "lib").is_dir()
                    ]
                    if Path("/dev/dxg").exists():
                        from ruinedfooocus.backends.llama.wsl_bridge import library_directory

                        paths.append(str(library_directory(acquire=False)))
                    if Path("/usr/lib/wsl/lib").is_dir():
                        paths.append("/usr/lib/wsl/lib")
                    env["LD_LIBRARY_PATH"] = ":".join(
                        paths + [env.get("LD_LIBRARY_PATH", "")]
                    ).rstrip(":")
            version = _run([str(binary.resolve()), "--version"], env).strip()
            devices = _run([str(binary.resolve()), "--list-devices"], env).strip()
            sections.append(
                f"**{label}**\n```text\n{version.replace('```', chr(39) * 3)}\nCPU; GPU devices reported by this build:\n{devices.replace('```', chr(39) * 3)}\n```"
            )
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError, TypeError):
            sections.append(f"**{label}:** installed, but could not query this build's devices.")
    try:
        script = (
            "import sys\n"
            "if sys.platform == 'linux':\n"
            "    import torch\n"
            "import json, xllamacpp as x; from importlib.metadata import version; "
            "print('RF_RUNTIME_INFO=' + json.dumps({'version':version('xllamacpp'), "
            "'devices':[{'name':d['name'],'description':d['description']} for d in x.get_device_info()[:32]]}))"
        )
        output = _run([sys.executable, "-c", script])
        line = next(line for line in output.splitlines() if line.startswith("RF_RUNTIME_INFO="))
        info = json.loads(line.split("=", 1)[1])
        devices = "\n".join(f"{d['name']}: {d['description']}" for d in info["devices"])
        from ruinedfooocus.packaging.hardware_inventory import xllamacpp_installed

        installed = xllamacpp_installed() or {}
        backend = installed.get("backend") or "unknown"
        sections.append(
            f"**xllamacpp {info['version']} / recorded wheel backend: {backend}**\n```text\n{devices.replace('```', chr(39) * 3) or 'No devices reported'}\n```"
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired, StopIteration, ValueError, KeyError):
        from importlib.metadata import PackageNotFoundError, version

        try:
            sections.append(
                f"**xllamacpp {version('xllamacpp')}:** installed; device information unavailable."
            )
        except PackageNotFoundError:
            sections.append("**xllamacpp:** not installed.")
    if truncated:
        sections.append("Cached build queries limited to eight entries.")
    return "\n\n".join(sections)
