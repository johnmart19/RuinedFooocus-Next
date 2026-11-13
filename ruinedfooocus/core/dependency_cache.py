"""Cache successful startup package maintenance using only the standard library."""

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig

from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.core.paths import APP_ROOT, data_path

FINGERPRINT_VERSION = 2
CONTRACT_PATTERNS = (
    "requirements*.txt",
    "pip/**/*.txt",
    "entry_with_update.py",
    "RuinedFooocus.bat",
    "RuinedFooocus.sh",
    "ruinedfooocus/app/bootstrap.py",
    "ruinedfooocus/app/arguments.py",
    "ruinedfooocus/packaging/**/*.py",
    "ruinedfooocus/core/**/*.py",
    "ruinedfooocus/backends/**/*.py",
)


def _startup_arguments():
    """Retain install/device intent before argparse applies it to the environment."""
    flags = (
        "--cpu",
        "--directml",
        "--gpu-device-id",
        "--cuda-device",
        "--default-device",
        "--oneapi-device-selector",
        "--rocm10",
        "--cuda-nightly",
        "--legacy",
        "--settings",
        "--api",
    )
    options, selected = sys.argv[1:], []
    for index, option in enumerate(options):
        spelling = option.split("=", 1)[0]
        if spelling.startswith("--") and any(flag.startswith(spelling) for flag in flags):
            selected.append(option)
            if "=" not in option and index + 1 < len(options):
                value = options[index + 1]
                if not value.startswith("--"):
                    selected.append(value)
    return tuple(selected)


# Keep the caller's context stable when bootstrap applies its own device masks
# or selects TORCH_PLATFORM. A subsequent process captures a changed selection.
STARTUP_ARGUMENTS = _startup_arguments()
STARTUP_ENVIRONMENT = tuple(
    (name, os.environ.get(name))
    for name in (
        "TORCH_PLATFORM",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "ASCEND_RT_VISIBLE_DEVICES",
        "ONEAPI_DEVICE_SELECTOR",
        "SYCL_DEVICE_FILTER",
        "ZE_AFFINITY_MASK",
        "GGML_VK_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
        "INDEX_URL",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
    )
)


def cache_path(root):
    root = Path(root).resolve()
    return (
        data_path("cache", "dependency-checks.json")
        if root == APP_ROOT
        else root / "cache/dependency-checks.json"
    )


def fingerprint(root):
    root = Path(root)
    digest = hashlib.sha256()
    digest.update(
        repr(
            (
                FINGERPRINT_VERSION,
                sys.executable,
                sys.prefix,
                sys.version,
                sys.implementation.cache_tag,
                sysconfig.get_config_var("SOABI"),
                sysconfig.get_config_var("Py_GIL_DISABLED"),
                sys.platform,
                platform.machine(),
                STARTUP_ARGUMENTS,
                STARTUP_ENVIRONMENT,
            )
        ).encode()
    )
    paths = {path for pattern in CONTRACT_PATTERNS for path in root.glob(pattern)}
    for path in sorted(paths):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(path.read_bytes())
    manifest = root / "distribution.json"
    if manifest.is_file():
        raw = manifest.read_bytes()
        try:
            contract = json.loads(raw)
            if isinstance(contract, dict):
                contract.pop("source_commit", None)
            digest.update(json.dumps(contract, sort_keys=True).encode())
        except (ValueError, UnicodeError):
            digest.update(raw)
    for directory in sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")}):
        for path in sorted(Path(directory).glob("*.dist-info")):
            metadata = path / "METADATA"
            if metadata.is_file():
                stat = metadata.stat()
                digest.update(repr((str(path), stat.st_size, stat.st_mtime_ns)).encode())
    return digest.hexdigest()


def needed(root, stage, force=False):
    if force:
        return True
    try:
        saved = json.loads(cache_path(root).read_text())
        return not isinstance(saved, dict) or saved.get(stage) != fingerprint(root)
    except (OSError, ValueError):
        return True


def record(root, stage):
    path = cache_path(root)
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError):
        saved = {}
    if not isinstance(saved, dict):
        saved = {}
    saved[stage] = fingerprint(root)
    save_json(path, saved)
