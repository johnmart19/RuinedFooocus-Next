"""Owned, cancellable child processes for explicit optional ONNX acquisition."""

import subprocess
import tempfile
import time
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit
import zipfile


def current_target():
    from packaging.tags import sys_tags
    from ruinedfooocus.packaging.runtime_support import RuntimeTarget
    from ruinedfooocus.packaging.hardware_inventory import is_wsl

    return RuntimeTarget(
        platform.system(),
        platform.machine(),
        platform.python_version(),
        f"cp{sys.version_info.major}{sys.version_info.minor}",
        frozenset(str(tag) for tag in sys_tags()),
        environment="wsl" if is_wsl() else "native",
    )


from ruinedfooocus.packaging.vendor_admission import linux_amd_adapters


def artifacts_from_report(document):
    from ruinedfooocus.packaging.runtime_support import RuntimeArtifact

    artifacts = []
    for item in document.get("install", []):
        metadata, download = item["metadata"], item["download_info"]
        checksum = download.get("archive_info", {}).get("hashes", {}).get("sha256", "")
        artifacts.append(
            RuntimeArtifact(
                metadata["name"],
                metadata["version"],
                filename=unquote(Path(urlsplit(download["url"]).path).name),
                available=True,
                requires_python=metadata.get("requires_python", ""),
                requirements=tuple(metadata.get("requires_dist", ())),
                source=download["url"],
                sha256=checksum,
            )
        )
    return tuple(artifacts)


@contextmanager
def prepare_migraphx(adapter=None, *, check_cancelled=None):
    """Explicit installation action: resolve/download/audit the worker's exact closure.

    Discovery/device binding and canary are later active stages. This producer is
    never called by Settings rendering, status queries or automatic startup.
    """
    from packaging.utils import canonicalize_name
    from ruinedfooocus.backends.onnx.provider_recipes import (
        MIGRAPHX_PACKAGES,
        MIGRAPHX_INDEX,
        MIGRAPHX_LIBS_INDEX,
        MIGRAPHX_GFX,
        migraphx_decision,
    )

    target = current_target()
    if (
        target.system != "Linux"
        or target.environment != "native"
        or target.python_abi != "cp312"
        or target.machine.lower() not in ("x86_64", "amd64")
    ):
        raise RuntimeError(
            "The MIGraphX plugin acquisition route requires native Linux Python 3.12; WSL/HIP/Windows use separate recipes."
        )
    adapters = linux_amd_adapters()
    if adapter is not None:
        matches = [
            item
            for item in adapters
            if item["adapter_id"] == adapter.get("adapter_id")
            or item["hardware_id"] == adapter.get("hardware_id")
        ]
    else:
        matches = adapters
    if len(matches) != 1:
        raise RuntimeError(
            "Select one physical AMD KFD/PCI adapter before MIGraphX acquisition; detected "
            + str(len(matches))
            + " matching adapters."
        )
    physical = dict(matches[0])
    physical["physical_adapter_count"] = len(adapters)
    if adapter and adapter.get("onnx_device_id"):
        physical["onnx_device_id"] = adapter["onnx_device_id"]
    if physical["gfx"] not in MIGRAPHX_GFX:
        raise RuntimeError("ROCm 10.1 MIGraphX does not qualify " + physical["gfx"])
    with tempfile.TemporaryDirectory(prefix="rf-migraphx-acquire-") as directory:
        root = Path(directory)
        report = root / "pip-report.json"
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--ignore-installed",
                "--only-binary=:all:",
                "--report",
                str(report),
                "--extra-index-url",
                MIGRAPHX_INDEX,
                "--extra-index-url",
                MIGRAPHX_LIBS_INDEX,
                *MIGRAPHX_PACKAGES,
            ],
            check_cancelled=check_cancelled,
            timeout=300,
            capture_output=True,
            text=True,
        )
        artifacts = artifacts_from_report(json.loads(report.read_text(encoding="utf-8")))
        metadata_decision = migraphx_decision(
            target, physical, artifacts=artifacts, closure_complete=True, native_payload=None
        )
        pending = {
            "Driver compatibility has not been verified",
            "Provider-specific MIGraphX device identity is unknown",
            "Native runtime payload has not been verified",
        }
        if not set(metadata_decision["reasons"]).issubset(pending):
            raise RuntimeError(
                "MIGraphX resolver rejected the artifact metadata: "
                + "; ".join(metadata_decision["reasons"])
            )
        urls = []
        for artifact in artifacts:
            if len(artifact.sha256) != 64 or any(
                c not in "0123456789abcdef" for c in artifact.sha256.lower()
            ):
                raise RuntimeError("The resolved wheel has no SHA-256: " + artifact.name)
            urls.append(artifact.source + "#sha256=" + artifact.sha256)
        wheels = root / "wheels"
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--only-binary=:all:",
                "--no-deps",
                "--dest",
                str(wheels),
                *urls,
            ],
            check_cancelled=check_cancelled,
            timeout=600,
            capture_output=True,
            text=True,
        )
        paths, payloads = {}, {}
        for artifact in artifacts:
            path = wheels / artifact.filename
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum != artifact.sha256:
                raise RuntimeError(
                    "The downloaded wheel differs from its resolver hash: " + artifact.name
                )
            with zipfile.ZipFile(path) as archive:
                payloads[canonicalize_name(artifact.name)] = any(
                    re.search(r"\.so(?:\.\d+)*$", name) for name in archive.namelist()
                )
            paths[canonicalize_name(artifact.name)] = str(path)
        required_native = ("onnxruntime-ep-migraphx", "onnxruntime", "migraphx-libs")
        if not all(payloads.get(name) for name in required_native):
            raise RuntimeError(
                "MIGraphX closure has no real native payload (possibly a placeholder wheel)."
            )
        yield SimpleNamespace(
            target=target,
            adapter=physical,
            evidence=dict(
                artifacts=artifacts,
                closure_complete=True,
                native_payload=True,
                artifact_paths=paths,
            ),
        )


def run(command, *, check_cancelled=None, timeout=None, capture_output=False, check=True, **kwargs):
    from ruinedfooocus.backends.comfy.comfy_compat import child_device_environment

    if "env" not in kwargs:
        kwargs["env"] = child_device_environment(service="onnx-acquisition")
    if check_cancelled is None:
        return subprocess.run(
            command, check=check, timeout=timeout, capture_output=capture_output, **kwargs
        )
    from ruinedfooocus.core.child_process import start_process

    check_cancelled()
    started = time.monotonic()
    if capture_output and ("stdout" in kwargs or "stderr" in kwargs):
        raise ValueError("stdout and stderr cannot be used with capture_output")
    if any(kwargs.get(key) == subprocess.PIPE for key in ("stdin", "stdout", "stderr")):
        raise ValueError(
            "Cancellable acquisition does not support interactive pipes; use bounded file output"
        )
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as diagnostic:
        if "input" in kwargs:
            raise ValueError(
                "Cancellable acquisition requires a noninteractive command; input is unsupported"
            )
        spawn = dict(kwargs)
        spawn.setdefault("stdout", output)
        spawn.setdefault("stderr", diagnostic)
        spawn.setdefault("creationflags", getattr(subprocess, "CREATE_NO_WINDOW", 0))
        process, job = start_process(command, **spawn)
        try:
            while process.poll() is None:
                check_cancelled()
                if timeout is not None and time.monotonic() - started >= timeout:
                    raise subprocess.TimeoutExpired(command, timeout)
                time.sleep(0.1)
            check_cancelled()
            output.seek(0)
            diagnostic.seek(0)
            encoding = kwargs.get("encoding") or "utf-8"
            errors = kwargs.get("errors") or "replace"
            stdout = (
                output.read().decode(encoding, errors=errors) if spawn["stdout"] is output else None
            )
            stderr = (
                diagnostic.read().decode(encoding, errors=errors)
                if spawn["stderr"] is diagnostic
                else None
            )
            if check and process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        finally:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            finally:
                if job:
                    job.close()
