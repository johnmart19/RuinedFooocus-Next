"""Exact original-Bria qualification identity; acquisition is never qualification."""

from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import inspect

BRIA_SHA256 = "5b486f08200f513f460da46dd701db5fbb47d79b4be4b708a19444bcd4e79958"


@lru_cache(maxsize=16)
def _file_hash(path, size, modified, changed):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_hash(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _file_hash(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def current_model():
    """Resolve an existing native model only; never invoke download_models."""
    import rembg

    if importlib.metadata.version("rembg") != "2.0.87":
        return None
    if inspect.signature(rembg.new_session).parameters["model_name"].default != "bria-rmbg":
        return None
    from rembg.sessions.bria_rmbg import BriaRmBgSession

    path = BriaRmBgSession.resolve_existing("bria-rmbg.onnx")
    if path is None:
        return None
    return dict(model="bria-rmbg", sha256=file_hash(path), preprocessing="native-rembg-2.0.87")


def runtime_identity(ort=None):
    try:
        return _runtime_identity(ort)
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        importlib.metadata.PackageNotFoundError,
    ):
        return None


def _runtime_identity(ort=None):
    root = (
        Path(ort.__file__).parent
        if ort is not None
        else Path(importlib.util.find_spec("onnxruntime").origin).parent
    )
    binaries = [
        path
        for path in (root / "capi").glob("onnxruntime_pybind11_state*")
        if path.suffix in (".pyd", ".so")
    ]
    if len(binaries) != 1:
        return None
    native = [
        path
        for path in (root / "capi").iterdir()
        if path.is_file() and (path.suffix in (".dll", ".pyd") or ".so" in path.name)
    ]
    return dict(
        ort=ort.__version__ if ort is not None else importlib.metadata.version("onnxruntime"),
        binaries={path.name: file_hash(path) for path in sorted(native)},
        python=platform.python_version(),
        abi=f"cp{sys.version_info.major}{sys.version_info.minor}",
        system=platform.system(),
    )


def driver_identity():
    from ruinedfooocus.packaging.vendor_admission import windows_driver_inventory

    inventory = windows_driver_inventory()
    rows = inventory.get("drivers", [])
    rows = [rows] if isinstance(rows, dict) else rows
    if (
        not inventory.get("build")
        or not rows
        or any(not row.get("DeviceID") or not row.get("DriverVersion") for row in rows)
    ):
        return None
    return dict(
        build=str(inventory["build"]),
        drivers=sorted(
            [dict(id=row["DeviceID"], version=row["DriverVersion"]) for row in rows],
            key=lambda row: row["id"],
        ),
    )


def current_contract(
    record, device_id, device_type="GPU", *, model=None, runtime=None, driver=None
):
    # Display-driver evidence cannot qualify an NPU. Named NPU selection remains
    # available for explicit qualification until its driver identity is supplied.
    if device_type not in ("GPU", "CPU") or not isinstance(device_id, str) or len(device_id) != 24:
        return None
    try:
        model = current_model() if model is None else model
        runtime = runtime_identity() if runtime is None else runtime
        driver = driver_identity() if driver is None else driver
        library_sha256 = file_hash(record["library_path"])
        if (
            not model
            or model.get("model") != "bria-rmbg"
            or model.get("sha256") != BRIA_SHA256
            or model.get("preprocessing") != "native-rembg-2.0.87"
            or not runtime
            or not driver
            or library_sha256 != record["sha256"]
        ):
            return None
        return dict(
            model_sha256=model["sha256"],
            preprocessing=model["preprocessing"],
            shape=[1, 3, 1024, 1024],
            dtype="float32",
            provider=record["name"],
            provider_library_sha256=library_sha256,
            catalog_version=record.get("catalog_version", "2.3.0"),
            runtime=runtime,
            device=device_id,
            device_type=device_type,
            driver=driver,
            options={},
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        importlib.metadata.PackageNotFoundError,
    ):
        return None


def fingerprint(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def evidence_valid(evidence, contract):
    if (
        not isinstance(contract, dict)
        or contract.get("model_sha256") != BRIA_SHA256
        or contract.get("preprocessing") != "native-rembg-2.0.87"
        or contract.get("shape") != [1, 3, 1024, 1024]
        or contract.get("dtype") != "float32"
        or not isinstance(contract.get("provider"), str)
        or not contract.get("runtime")
        or not contract.get("driver")
        or not isinstance(evidence, dict)
        or evidence.get("schema") != 1
        or evidence.get("contract") != contract
        or evidence.get("cache_identity") != fingerprint(contract)
    ):
        return False
    required = (
        "qualified",
        "real_model",
        "original_bria",
        "numerical_passed",
        "visual_passed",
        "stability_passed",
        "hidden_fallback_disabled",
    )
    if not all(evidence.get(key) is True for key in required):
        return False
    assignment = evidence.get("assignment", {})
    if not isinstance(assignment, dict) or not isinstance(assignment.get("nodes"), dict):
        return False
    nodes = assignment["nodes"].get(contract["provider"], 0)
    if (
        assignment.get("scope") != "first_inference_executed_nodes"
        or not isinstance(nodes, int)
        or isinstance(nodes, bool)
        or nodes <= 0
    ):
        return False
    costs = [
        evidence.get(key)
        for key in ("cold_end_to_end_ms", "warm_end_to_end_ms", "cpu_end_to_end_ms")
    ]
    return (
        all(
            isinstance(value, (float, int))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
            for value in costs
        )
        and costs[1] < costs[2]
    )


def matching_record(record, *, device_id=None, device_type=None):
    evidence = record.get("workload_evidence", {})
    if not isinstance(evidence, dict) or evidence.get("schema") != 1:
        return False
    saved = evidence.get("contract", {})
    if not isinstance(saved, dict):
        return False
    current = current_contract(
        record, device_id or saved.get("device"), device_type or saved.get("device_type")
    )
    return evidence_valid(evidence, current)
