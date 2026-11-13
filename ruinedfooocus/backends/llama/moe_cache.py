"""Passive native MoE CPU-share capability and launch policy.

No help/version subprocess, device discovery, acquisition or model inspection.
RF's enabled midpoint is an app policy, not an upstream tuning recommendation.
"""

from functools import lru_cache
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shlex
import threading

DEFAULTS = {
    "llama_moe_cache_cpu_share_enabled": True,
    "llama_moe_cache_cpu_share": 0.5,
}
_FLAG = "--moe-cache-cpu-share"
_TOKENS = (_FLAG.encode(), b"LLAMA_ARG_MOE_CACHE_CPU_SHARE")
_WARNED = set()
_WARNING_LOCK = threading.Lock()
_MAX_BINARY = 128 * 1024**2
_MAX_MANIFEST = 64 * 1024
# Exact files from the hash-verified publisher archive, not arbitrary neighbors.
# Other modular assets stay unknown until their file identities are reviewed.
_MODULAR_PARSERS = {
    ("b12327", "llama-b12327-bin-win-vulkan-x64.zip"): (
        "5f5ee83bc362f92c6cff05a24071d2e2057f47de689913edc64be2274705b504",
        "33789ad472dfcaeba7ddbb8a58083c3f6f1bfff2965803ddba5ae2e538a8ecb7",
        "451ff89e6d00b9c0c6f28b078b01516c2b562665fe26a4374861a379f67dcfd0",
    ),
    ("b12366", "llama-b12366-bin-win-vulkan-x64.zip"): (
        "1708e3125afc80bcad84b571a3d1563fc776cad39acb9f2f3937f9bfbba50028",
        "1f768bd4b9d58a211152227bbe48782ebe9d706afd92723dd35ed793a5cb8f16",
        "8e999279b1c898eac3da45ddd38a42972f49e0e005584eb8be3c1d0e7da673d0",
    ),
}


def _signature(path):
    path = Path(path).resolve()
    value = path.stat()
    return str(path), value.st_size, value.st_mtime_ns, value.st_ino, value.st_ctime_ns


@lru_cache(maxsize=32)
def _binary_evidence(signature):
    """Bounded passive bytes bound to one unchanged executable or parser DLL."""
    path = Path(signature[0])
    if signature[1] > _MAX_BINARY:
        return None, False
    digest = hashlib.sha256()
    found = set()
    tail = b""
    try:
        with path.open("rb") as stream:
            remaining = signature[1]
            while remaining:
                chunk = stream.read(min(1024**2, remaining))
                if not chunk:
                    return None, False
                remaining -= len(chunk)
                digest.update(chunk)
                data = tail + chunk
                found.update(token for token in _TOKENS if token in data)
                tail = data[-256:]
            if stream.read(1) or _signature(path) != signature:
                return None, False
        return digest.hexdigest(), len(found) == len(_TOKENS)
    except OSError:
        return None, False


@lru_cache(maxsize=32)
def _modular_parser_option(executable, library, marker, registered):
    """Only a registered official package can lend its known parser DLL."""
    try:
        if marker[1] > _MAX_MANIFEST:
            return False
        with Path(marker[0]).open("rb") as stream:
            data = json.loads(stream.read(_MAX_MANIFEST + 1))
        version, package = data.get("version"), data.get("package")
        expected = dict(registered).get((version, package))
        if expected is None:
            return False
        archive, launcher_hash, parser_hash = expected
        if data.get("archives") != [
            {
                "name": package,
                "sha256": archive,
                "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{version}/{package}",
            }
        ]:
            return False
        files = data.get("files", {})
        if (
            files.get("llama-server.exe") != launcher_hash
            or files.get("llama-common.dll") != parser_hash
        ):
            return False
        actual_launcher, _ = _binary_evidence(executable)
        actual_parser, supported = _binary_evidence(library)
        return (
            actual_launcher == launcher_hash
            and actual_parser == parser_hash
            and supported
            and all(
                _signature(signature[0]) == signature for signature in (executable, library, marker)
            )
        )
    except (OSError, ValueError, TypeError, AttributeError, RecursionError):
        return False


def parser_capability(executable):
    try:
        signature = _signature(executable)
        if _binary_evidence(signature)[1]:
            return True
        path = Path(executable)
        library, marker = path.with_name("llama-common.dll"), path.with_name("rf-runtime.json")
        if path.name != "llama-server.exe" or any(
            item.is_symlink() for item in (path, library, marker)
        ):
            return False
        signatures = signature, _signature(library), _signature(marker)
        if len({Path(value[0]).parent for value in signatures}) != 1:
            return False
        return _modular_parser_option(*signatures, tuple(_MODULAR_PARSERS.items()))
    except (OSError, TypeError, ValueError):
        return False


def _number(value, label, *, integer=False):
    if isinstance(value, bool):
        raise ValueError(label + " requires a number.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(label + " requires a number.") from None
    if not math.isfinite(number) or (integer and not number.is_integer()):
        raise ValueError(label + " requires a finite " + ("integer." if integer else "number."))
    return int(number) if integer else number


def _options(tokens, names, *, remove=False):
    values, retained = [], []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        key, equals, value = token.partition("=")
        if key not in names:
            retained.append(token)
            index += 1
            continue
        if not equals:
            if index + 1 >= len(tokens):
                raise ValueError(key + " requires a value.")
            value = tokens[index + 1]
        values.append(value)
        count = 1 if equals else 2
        if not remove:
            retained.extend(tokens[index : index + count])
        index += count
    return values, retained


def moe_cache_cpu_share_capability(
    settings, executable=None, *, gpu=None, arguments=None, environment=None, model_is_moe=None
):
    """Requested/effective state plus safe native arguments, without execution.

    Unsupported requests remain saved but inactive. Explicit native CLI wins;
    unsupported occurrences are omitted from this launch before the process can
    reject an unknown option. Other native arguments remain byte-for-byte tokens.
    """
    enabled = settings.get("llama_moe_cache_cpu_share_enabled", True)
    if type(enabled) is not bool:
        raise ValueError("MoE cache CPU share enabled must be true or false.")
    requested_share = _number(settings.get("llama_moe_cache_cpu_share", 0.5), _FLAG)
    if not 0 <= requested_share <= 1:
        raise ValueError(_FLAG + " must be between 0 and 1.")
    extra = list(
        shlex.split(settings.get("llama_server_args", "")) if arguments is None else arguments
    )
    environment = os.environ if environment is None else environment
    overrides, filtered = _options(extra, {_FLAG}, remove=True)
    for item in overrides:
        if not 0 <= _number(item, _FLAG) <= 1:
            raise ValueError(_FLAG + " must be between 0 and 1.")
    override = overrides[-1] if overrides else environment.get("LLAMA_ARG_MOE_CACHE_CPU_SHARE")
    selected_share = requested_share if override is None else _number(override, _FLAG)
    if not 0 <= selected_share <= 1:
        raise ValueError(_FLAG + " must be between 0 and 1.")
    native = settings.get("llm_runtime", "llama.cpp") == "llama.cpp"
    supported = native and executable is not None and parser_capability(executable)
    source = (
        "native_cli"
        if overrides
        else "native_environment"
        if override is not None
        else "app_setting"
    )
    state = {
        "requested_enabled": enabled,
        "requested_share": requested_share,
        "selected_share": selected_share,
        "source": source,
        "supported": bool(supported),
        "active": False,
        "applied": False,
        "observed": None,
        "effective_share": None,
        "arguments": (),
        "extra_arguments": tuple(extra if supported else filtered),
        "reason": "Load a native runtime exposing this parser option. The pinned b12110 runtime lacks it.",
    }
    if not native:
        return {
            **state,
            "reason": "No equivalent CPU-share setting is established for xllamacpp; the saved native preference is inactive.",
        }
    if not supported:
        return state
    caches, _ = _options(extra, {"--moe-cache-mib"})
    cache = _number(
        caches[-1] if caches else environment.get("LLAMA_ARG_MOE_CACHE_MIB", 0),
        "--moe-cache-mib",
        integer=True,
    )
    if cache < 0:
        raise ValueError("--moe-cache-mib cannot be negative.")
    counts, _ = _options(extra, {"-ncmoe", "--n-cpu-moe"})
    count = _number(
        counts[-1] if counts else environment.get("LLAMA_ARG_N_CPU_MOE", 0),
        "--n-cpu-moe",
        integer=True,
    )
    tensors, _ = _options(extra, {"-ot", "--override-tensor"})
    tensors.insert(0, environment.get("LLAMA_ARG_OVERRIDE_TENSOR", ""))
    expert_rules = [
        assignment.rpartition("=")
        for group in tensors
        for assignment in group.split(",")
        if "=" in assignment
    ]
    host = (
        "-cmoe" in extra
        or "--cpu-moe" in extra
        or environment.get("LLAMA_ARG_CPU_MOE", "").lower() in ("1", "true", "on")
        or count > 0
        or any(
            "exps" in pattern.lower() and buffer.strip().upper() == "CPU"
            for pattern, separator, buffer in expert_rules
        )
    )
    layers, _ = _options(extra, {"-ngl", "--gpu-layers", "--n-gpu-layers"})
    devices, _ = _options(extra, {"--device", "-dev"})
    selected_layers = (
        layers[-1]
        if layers
        else settings.get("llm_n_gpu_layers", environment.get("LLAMA_ARG_N_GPU_LAYERS", -1))
    )
    zero_layers = (
        str(selected_layers).strip() not in ("auto", "all")
        and _number(selected_layers, "--n-gpu-layers", integer=True) == 0
    )
    selected_devices = devices[-1] if devices else environment.get("LLAMA_ARG_DEVICE", "")
    actual_gpu = (
        gpu is True
        and not zero_layers
        and selected_devices.strip().lower() != "none"
        and not settings.get("cpu_everywhere", False)
        and settings.get("llama_backend") != "CPU"
    )
    requested = enabled or override is not None
    if not requested:
        reason = "Disabled in RF; no app CPU-share option is passed."
    elif not actual_gpu or not (host or model_is_moe is True) or cache <= 0:
        reason = "Inactive: requires GPU execution, host-resident MoE experts and a nonzero --moe-cache-mib cache."
    elif selected_share == 0:
        reason = "The selected native share is 0; cache misses are uploaded to the device."
    else:
        reason = "CPU computes the selected share of cache misses only after the MoE GPU cache is full. Reload Chat to apply changes."
        state.update(active=True, effective_share=selected_share)
    if state["active"] and override is None:
        state["arguments"] = (_FLAG, format(selected_share, ".12g"))
    return {**state, "reason": reason}


def warn_unsupported(state, executable):
    # The default preference is deliberately dormant for ordinary full-GPU and
    # dense loads. Warn only when a user explicitly supplied an unsupported flag.
    if state["supported"] or state["source"] == "app_setting":
        return
    try:
        identity = _signature(executable)
    except (OSError, TypeError, ValueError):
        identity = (str(executable),)
    with _WARNING_LOCK:
        if identity in _WARNED:
            return
        if len(_WARNED) >= 128:
            _WARNED.clear()
        _WARNED.add(identity)
    logging.getLogger(__name__).warning("MoE cache CPU share remains inactive: %s", state["reason"])


def public_moe_cache_state(state):
    return {
        key: value for key, value in state.items() if key not in ("arguments", "extra_arguments")
    }
