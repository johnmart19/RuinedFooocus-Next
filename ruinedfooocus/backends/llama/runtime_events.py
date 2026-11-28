"""Request-owned native load events and bounded startup evidence, without probes."""

from concurrent.futures import CancelledError
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import uuid


PHASES = frozenset(
    (
        "resolving_runtime",
        "downloading_runtime",
        "verifying_runtime",
        "staging_runtime",
        "probing_device",
        "loading_model",
        "ready",
        "cancelled",
        "failed",
    )
)
MAX_LOG_BYTES = 2 * 1024**2
MAX_LINE = 16 * 1024
MAX_ROWS = 256


class RuntimeLoad:
    """One identity fences callbacks and completion; invalidation never starts compute."""

    def __init__(self, load_id=None, callback=None, check_cancelled=None):
        self.load_id = str(load_id or uuid.uuid4().hex)
        self.callback = callback
        self.check_cancelled = check_cancelled
        self._cancelled = threading.Event()
        self._lock = threading.RLock()
        self.state = None

    def cancel(self):
        with self._lock:
            self._cancelled.set()

    def check(self):
        if self._cancelled.is_set():
            raise CancelledError("A newer model load replaced this request")
        if self.check_cancelled is not None:
            self.check_cancelled()

    def emit(self, event=None, *, phase=None, **fields):
        with self._lock:
            event = dict(event or {}, **fields)
            phase = phase or event.pop("phase", None)
            transfer = {
                key: event.pop(key) for key in ("received", "total", "rate") if key in event
            }
            if transfer:
                event["transfer"] = transfer
            if phase not in PHASES:
                raise ValueError("Unknown runtime load phase")
            if phase not in ("cancelled", "failed"):
                self.check()
            elif self._cancelled.is_set():
                return
            self.state = deepcopy({**event, "load_id": self.load_id, "phase": phase})
            if self.callback is not None:
                self.callback(deepcopy(self.state))

    def snapshot(self):
        with self._lock:
            return deepcopy(self.state)


def file_identity(path, *, checksum=False, check_cancelled=None):
    path = Path(path).resolve()
    stat = path.stat()
    signature = [str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    result = {"path": str(path), "signature": signature}
    if checksum and stat.st_size <= 128 * 1024**2:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024**2):
                if check_cancelled is not None:
                    check_cancelled()
                digest.update(chunk)
        current = path.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        ):
            raise ValueError("Runtime changed while recording its identity")
        result["sha256"] = digest.hexdigest()
    return result


def runtime_identity(executable, backend, *, check_cancelled=None, custom=False):
    """Identify the child and bounded verified-install receipt; never expose raw CLI."""
    try:
        result = file_identity(executable, checksum=True, check_cancelled=check_cancelled)
    except OSError:
        result = {"path": str(Path(executable).resolve()), "signature": None}
    result["backend"] = backend
    for directory in list(Path(executable).resolve().parents)[:4]:
        receipt = directory / "rf-runtime.json"
        try:
            with receipt.open("rb") as stream:
                raw = stream.read(512 * 1024 + 1)
            data = json.loads(raw) if len(raw) <= 512 * 1024 else {}
            if not isinstance(data, dict):
                continue
            result.update(tag=data.get("version"), package=data.get("package"))
            expected = data.get("files", {}).get(
                Path(executable).resolve().relative_to(directory).as_posix()
            )
            result["manifest_verified"] = bool(
                not custom and expected and expected == result.get("sha256")
            )
            break
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return result


def native_request(settings, tokens, *, gpu=True, environment=None):
    """Only approved memory-policy fields, using native last-CLI-value precedence."""
    names = {
        "--ctx-size": "context",
        "-c": "context",
        "--n-gpu-layers": "gpu_layers",
        "--gpu-layers": "gpu_layers",
        "-ngl": "gpu_layers",
        "--device": "device",
        "-dev": "device",
        "--fit": "fit",
        "-fit": "fit",
        "--fit-target": "fit_margin",
        "-fitt": "fit_margin",
        "--fit-ctx": "fit_context_min",
        "-fitc": "fit_context_min",
        "--cache-type-k": "cache_type_k",
        "-ctk": "cache_type_k",
        "--cache-type-v": "cache_type_v",
        "-ctv": "cache_type_v",
        "--override-tensor": "tensor_overrides",
        "-ot": "tensor_overrides",
        "--n-cpu-moe": "cpu_expert_layers",
        "-ncmoe": "cpu_expert_layers",
        "--moe-cache-mib": "moe_cache_mib",
    }
    result = {
        "backend": settings.get("llama_backend", "Auto"),
        "context": str(int(settings.get("llm_n_ctx", 8192))),
        "gpu_layers": "auto"
        if int(settings.get("llm_n_gpu_layers", -1)) == -1
        else "all"
        if int(settings.get("llm_n_gpu_layers", -1)) < -1
        else str(settings.get("llm_n_gpu_layers", -1)),
        "fit": "on",
        "kv_offload": True,
        "op_offload": True,
        "cpu_experts": False,
    }
    environment = environment or {}
    result["cpu_experts"] = str(environment.get("LLAMA_ARG_CPU_MOE", "")).lower() in (
        "1",
        "true",
        "on",
    )
    for variable, key in (
        ("LLAMA_ARG_FIT", "fit"),
        ("LLAMA_ARG_FIT_TARGET", "fit_margin"),
        ("LLAMA_ARG_FIT_CTX", "fit_context_min"),
        ("LLAMA_ARG_DEVICE", "device"),
        ("LLAMA_ARG_CACHE_TYPE_K", "cache_type_k"),
        ("LLAMA_ARG_CACHE_TYPE_V", "cache_type_v"),
        ("LLAMA_ARG_N_CPU_MOE", "cpu_expert_layers"),
        ("LLAMA_ARG_OVERRIDE_TENSOR", "tensor_overrides"),
    ):
        if variable in environment:
            result[key] = str(environment[variable])[:1024]
    index = 0
    while index < len(tokens):
        key, equal, value = tokens[index].partition("=")
        if key in names:
            if not equal and index + 1 < len(tokens):
                index += 1
                value = tokens[index]
            if value:
                if names[key] == "tensor_overrides" and result.get("tensor_overrides"):
                    result["tensor_overrides"] = (result["tensor_overrides"] + "," + value)[:1024]
                else:
                    result[names[key]] = value[:1024]
        elif key in ("--cpu-moe", "-cmoe"):
            result["cpu_experts"] = True
        elif key == "--no-kv-offload":
            result["kv_offload"] = False
        elif key == "--no-op-offload":
            result["op_offload"] = False
        index += 1
    if not gpu:
        result.update(gpu_layers="0", device="none", kv_offload=False, op_offload=False)
    result["context_policy"] = "full_model" if result["context"] == "0" else "explicit"
    return result


def _host(name):
    return bool(re.search(r"CPU|HOST|MAPPED", name, re.I))


def model_header_evidence(path, *, check_cancelled=None):
    """Reuse bounded GGUF descriptors; never map or read tensor payloads."""
    from ruinedfooocus.features.models.gguf_metadata import read_gguf_header
    from gguf.constants import GGML_QUANT_SIZES

    if check_cancelled is not None:
        check_cancelled()
    try:
        tensors, architecture = read_gguf_header(path, include_storage=True)
        tensor = tensors.get("token_embd.weight", {})
        block, width = GGML_QUANT_SIZES[tensor["gguf_type"]]
        size = math.prod(tensor["storage_shape"]) * width // block
        return {
            "architecture": architecture,
            "input_tensor": "token_embd.weight",
            "input_bytes": size,
            "moe": any("_exps" in name for name in tensors),
            "source": "bounded_gguf_tensor_descriptors",
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None
    finally:
        if check_cancelled is not None:
            check_cancelled()


def parse_startup(text, *, truncated=False, host_input=None):
    """Parse native fields, never timing/provider labels as placement proof.

    Full layer totals with nonzero host weights remain unknown unless bounded
    tensor descriptors explain the mandatory native input buffer.
    Native fit rows are estimates and never establish initialized placement.
    """
    encoded = text.encode("utf-8", errors="replace")
    truncated = truncated or len(encoded) > MAX_LOG_BYTES
    text = encoded[:MAX_LOG_BYTES].decode("utf-8", errors="replace")
    result = {
        "state": "unknown",
        "layers": None,
        "model_buffers": [],
        "kv_buffers": [],
        "compute_buffers": [],
        "moe_cache_buffers": [],
        "tensors": [],
        "fit_estimates": [],
        "context": None,
        "quantization": None,
        "architecture": None,
        "host_experts": None,
        "truncated": truncated,
        "complete": False,
        "evidence_source": "native_startup_log",
        "reason": "Initialized tensor placement was not established.",
    }
    pending_tensors = []
    for line in text.splitlines():
        if len(line) > MAX_LINE:
            result["truncated"] = True
            continue
        if line.startswith("{"):
            try:
                event = json.loads(line)
                if event.get("type") == "fit_memory_breakdown":
                    rows = event.get("data", {}).get("rows", [])
                    if len(rows) > MAX_ROWS:
                        result["truncated"] = True
                    result["fit_estimates"] = [
                        {
                            key: value
                            for key, value in row.items()
                            if key
                            in (
                                "kind",
                                "name",
                                "total",
                                "free",
                                "model",
                                "context",
                                "compute",
                                "unaccounted",
                            )
                            and isinstance(value, (str, int, float))
                        }
                        for row in rows[:MAX_ROWS]
                        if isinstance(row, dict)
                    ]
                line = str(event.get("msg", ""))
            except (ValueError, AttributeError, TypeError, RecursionError):
                continue
        match = re.search(r"offloaded (\d{1,6})/(\d{1,6}) layers to GPU", line)
        if match:
            result["layers"] = {"offloaded": int(match[1]), "total": int(match[2])}
            # A subsequent model initialization replaces dry-run/older evidence.
            result["model_buffers"] = []
            result["moe_cache_buffers"] = []
            result["complete"] = False
            result["tensors"] = pending_tensors
            result["host_experts"] = (
                True
                if any("exps" in tensor["name"] and tensor["host"] for tensor in pending_tensors)
                else None
            )
            pending_tensors = []
        match = re.search(r"([\w.:-]+)\s+(model|KV|compute) buffer size\s*=\s*([\d.]+) MiB", line)
        if match:
            try:
                size = float(match[3])
            except ValueError:
                result["truncated"] = True
                continue
            key = {"model": "model_buffers", "KV": "kv_buffers", "compute": "compute_buffers"}[
                match[2]
            ]
            if math.isfinite(size) and size >= 0 and len(result[key]) < MAX_ROWS:
                result[key].append({"buffer": match[1], "mib": size, "host": _host(match[1])})
                if key == "model_buffers":
                    result["complete"] = False
            else:
                result["truncated"] = True
        if re.search(r"\bmodel loaded\s*$", line):
            result["complete"] = True
        match = re.search(
            r"([\w.:-]+) MoE cache size\s*=\s*([\d.]+) MiB for ([\d.]+) MiB of host experts, ([\d.]+)% of the misses on the CPU",
            line,
        )
        if match:
            try:
                size, experts, share = (float(match[index]) for index in (2, 3, 4))
                if (
                    not all(math.isfinite(value) and value >= 0 for value in (size, experts, share))
                    or share > 100
                    or len(result["moe_cache_buffers"]) >= MAX_ROWS
                ):
                    raise ValueError()
                result["moe_cache_buffers"].append(
                    {
                        "buffer": match[1],
                        "mib": size,
                        "host_experts_mib": experts,
                        "configured_cpu_share": share / 100,
                    }
                )
                if experts > 0:
                    result["host_experts"] = True
            except ValueError:
                result["truncated"] = True
        match = re.search(r"(?:n_ctx|n_ctx_per_seq)\s*=\s*(\d{1,10})(?!\d)", line)
        if match:
            result["context"] = int(match[1])
        for label, key in (
            ("model ftype", "quantization"),
            ("file type", "quantization"),
            ("general.architecture", "architecture"),
        ):
            match = re.search(re.escape(label) + r"(?:\s+\w+)?\s*=\s*([\w .+()/-]{1,80})", line)
            if match:
                result[key] = match[1].strip()
        # Explicit native tensor overrides are positive evidence, never an
        # exhaustive inventory. They can disprove full offload but cannot prove it.
        match = re.search(
            r"tensor [\"']?([\w.]+)[\"']?.*?buffer type (?:changed from \S+ to |overridden to |[=:] ?)([\w.:-]+)",
            line,
        )
        if match and len(pending_tensors) < MAX_ROWS:
            pending_tensors.append({"name": match[1], "buffer": match[2], "host": _host(match[2])})
        elif match:
            result["truncated"] = True
    if not result["complete"] or result["truncated"]:
        return result
    buffers = result["model_buffers"]
    device = any(not row["host"] and row["mib"] > 0 for row in buffers)
    host = any(row["host"] and row["mib"] > 0 for row in buffers)
    layers = result["layers"]
    if device and (result["host_experts"] or (layers and layers["offloaded"] < layers["total"])):
        result.update(
            state="mixed", reason="Native initialized weights include host and device placement."
        )
    elif (
        device
        and layers
        and layers["offloaded"] == layers["total"]
        and not host
        and not result["truncated"]
        and result["complete"]
    ):
        result.update(
            state="full",
            host_experts=False,
            reason="All offloadable layers and observed model buffers are on the device.",
        )
    elif (
        host
        and not device
        and result["complete"]
        and not result["truncated"]
        and (layers is None or layers["offloaded"] == 0)
    ):
        result.update(
            state="cpu", reason="Native initialized model buffers and layers are on the host."
        )
    elif device and host and layers and layers["offloaded"] == layers["total"]:
        result["reason"] = (
            "All layers were offloaded, but host buffers may include input weights or MoE experts; tensor placement is unconfirmed."
        )
        host_rows = [row for row in buffers if row["host"]]
        expected = host_input.get("input_bytes") if isinstance(host_input, dict) else None
        # Native logs round each buffer to .01 MiB; allow that rounding and one
        # alignment page per buffer, never a percentage of model/VRAM capacity.
        tolerance = len(host_rows) * (0.005 * 1024**2 + 4096)
        actual = sum(row["mib"] * 1024**2 for row in host_rows)
        if (
            expected
            and abs(actual - expected) <= tolerance
            and not result["truncated"]
            and result["complete"]
        ):
            result.update(
                state="full",
                host_experts=False,
                mandatory_host_input=deepcopy(host_input),
                reason="All offloadable layers are on the device; host model bytes match the native input tensor within log rounding/alignment.",
            )
        elif expected and actual > expected + tolerance:
            result.update(
                state="mixed",
                reason="Initialized host model buffers exceed the mandatory native input tensor.",
            )
    return result


def startup_evidence(path, *, host_input=None):
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_LOG_BYTES + 1)
        return parse_startup(
            raw.decode("utf-8", errors="replace"),
            truncated=len(raw) > MAX_LOG_BYTES,
            host_input=host_input,
        )
    except OSError:
        return parse_startup("")
