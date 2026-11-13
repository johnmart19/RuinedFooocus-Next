"""Explicit optional-runtime activation and evidence-bound workload receipts."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import platform
import threading

from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.core.paths import data_path
from . import artifacts, runtime
from .contracts import (
    RELEASE,
    SOURCE_REVISION,
    GenerationRequest,
    _Cancellation,
    assess,
    inspect_model,
    with_model_hash,
)

_ACTIVE = {}
_LOCK = threading.Lock()


@dataclass(frozen=True)
class DeviceIdentity:
    adapter_id: str = ""
    driver_version: str = ""
    driver_identity: str = ""
    location: str = ""
    stable: bool = False

    @property
    def complete(self):
        return self.stable and all(
            (self.adapter_id, self.driver_version, self.driver_identity, self.location)
        )


@dataclass(frozen=True)
class Activation:
    candidate: object
    identity: DeviceIdentity


def root_path():
    return data_path("cache", "sdcpp")


def artifact(backend, *, system=None, machine=None):
    item = artifacts.artifact_for(
        system or platform.system(), machine or platform.machine(), backend
    )
    if item is None:
        raise ValueError("No pinned optional image runtime exists for this platform")
    return item


def runtime_directory(item):
    return root_path() / RELEASE / item.filename.removesuffix(".zip")


def status(backend, *, system=None, machine=None):
    """Passive UI facts: no hashing, loading, downloading or device enumeration."""
    try:
        item = artifact(backend, system=system, machine=machine)
    except ValueError as error:
        return {"backend": backend, "prepared": False, "available": False, "reason": str(error)}
    directory = runtime_directory(item)
    with _LOCK:
        active = [value for key, value in _ACTIVE.items() if key[0] == str(directory)]
    return {
        "backend": backend,
        "available": True,
        "revision": SOURCE_REVISION,
        "artifact": item.filename,
        "artifact_sha256": item.sha256,
        "platform_contract": item.platform_contract,
        "directory": str(directory),
        "prepared": (directory / "rf-runtime.json").is_file(),
        "devices": tuple(value.candidate.device for value in active),
        "identity_complete": any(value.identity.complete for value in active),
        "qualification": "Explicit model/operation acceptance required",
    }


def install(
    backend,
    *,
    check_cancelled,
    offline=False,
    skip_dependency_install=False,
    progress=None,
    system=None,
    machine=None,
):
    """Explicit Settings action. Offline can verify/reuse an existing bundle only."""
    item = artifact(backend, system=system, machine=machine)
    directory = runtime_directory(item)
    if directory.exists():
        return artifacts.open_runtime(directory, item)
    if offline or skip_dependency_install:
        raise RuntimeError(
            "Optional image runtime is not prepared; acquisition is disabled in offline/skip-install mode"
        )
    return artifacts.prepare_runtime(
        item,
        directory,
        archive=root_path() / "archives" / item.filename,
        check_cancelled=check_cancelled,
        progress=progress,
    )


def verify(backend, device, *, check_cancelled, identity=None, system=None, machine=None):
    """Explicit active Settings action; no acquisition and no workload qualification."""
    item = artifact(backend, system=system, machine=machine)
    directory = runtime_directory(item)
    candidate = artifacts.open_runtime(directory, item, device=device)
    candidate = runtime.verify_device(candidate, check_cancelled=check_cancelled)
    identity = identity or DeviceIdentity()
    if not isinstance(identity, DeviceIdentity):
        raise TypeError("Device identity must come from trusted physical/driver inventory")
    activation = Activation(candidate, identity)
    with _LOCK:
        _ACTIVE[(str(directory), device)] = activation
    return activation


def current(backend, device, *, system=None, machine=None):
    """Return only this process's explicit activation; a saved index is insufficient."""
    item = artifact(backend, system=system, machine=machine)
    directory = runtime_directory(item)
    with _LOCK:
        activation = _ACTIVE.get((str(directory), device))
    if activation is None:
        return None
    runtime._unchanged(activation.candidate)
    if (
        runtime._device_environment(runtime._environment(activation.candidate))
        != activation.candidate.device_environment
    ):
        raise RuntimeError("Optional image device environment changed; verify it again")
    return activation


def _payload_identity(candidate):
    manifest = Path(candidate.executable).parent / "rf-runtime.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    return hashlib.sha256(json.dumps(value["files"], sort_keys=True).encode()).hexdigest()


def _key(activation):
    candidate = activation.candidate
    return {
        "revision": candidate.revision,
        "archive_sha256": candidate.artifact_sha256,
        "payload_sha256": _payload_identity(candidate),
        "backend": candidate.backend,
        "device": candidate.device,
        "description": candidate.device_description,
        "environment": [list(item) for item in candidate.device_environment],
        "physical_driver": asdict(activation.identity),
    }


def receipts_path():
    return root_path() / "acceptance.json"


def cached_model_hash(path):
    """Read explicit local verification evidence without hashing model weights."""
    from ruinedfooocus.features.models.model_identity import file_signature

    try:
        values = json.loads((root_path() / "model-identities.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ""
    signature = file_signature(path)
    entry = values.get(signature[0], {})
    return entry.get("sha256", "") if entry.get("signature") == signature else ""


def verify_model(path, *, architecture, check_cancelled, variant="unknown"):
    """Explicit local identity action, usable offline and outside compute admission.

    Hashing reuses the project's signature cache; variant remains a caller-supplied
    preparation descriptor until actual acceptance records its provenance.
    """
    from ruinedfooocus.core.preparation import preparation_service
    from ruinedfooocus.features.models.model_identity import cached_file_sha256, file_signature
    from ruinedfooocus.packaging.install_lock import installation_lock

    check_cancelled()
    with preparation_service.artifact_lease(path, cancelled=_Cancellation(check_cancelled)):
        model = inspect_model(path, architecture=architecture, variant=variant)
        checksum = cached_file_sha256(path, root_path() / "model-hashes")
        check_cancelled()
        model = with_model_hash(model, checksum)
        if tuple(file_signature(path)) != model.signature:
            raise ValueError("Checkpoint changed during explicit identity verification")
        destination = root_path() / "model-identities.json"
        with installation_lock(destination):
            try:
                values = json.loads(destination.read_text(encoding="utf-8"))
            except FileNotFoundError:
                values = {}
            values[model.signature[0]] = {
                "signature": list(model.signature),
                "sha256": model.sha256,
            }
            save_json(destination, values)
        return model


def _read_receipts():
    try:
        document = json.loads(receipts_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    if (
        not isinstance(document, dict)
        or document.get("schema") != 1
        or not isinstance(document.get("receipts"), list)
    ):
        raise RuntimeError("Optional image acceptance store is invalid; retain it for diagnosis")
    return document["receipts"]


def record_acceptance(activation, model, operation, *, evidence):
    """Called only by explicit actual acceptance, never by install or enumeration.

    The parent supplies inspected visual/placement/cancel evidence. This function
    records it; it does not claim that a nonempty string proves that execution.
    """
    from ruinedfooocus.packaging.install_lock import installation_lock

    if not activation.identity.complete:
        raise ValueError(
            "Stable physical adapter and complete driver identity are required for automatic qualification"
        )
    if (
        not model.sha256
        or model.variant != "base"
        or operation not in ("txt2img", "img2img")
        or not evidence
    ):
        raise ValueError(
            "Actual acceptance needs an exact standard-base model hash, operation and evidence reference"
        )
    source_reasons = assess(
        GenerationRequest(
            model,
            "",
            "",
            64,
            64,
            0,
            operation=operation,
            init_image=Path("acceptance-input.png") if operation == "img2img" else None,
        ),
        activation.candidate,
        allow_unqualified=True,
    )
    if source_reasons:
        raise ValueError(
            "Cannot record unsupported/unverified acceptance: " + "; ".join(source_reasons)
        )
    runtime._unchanged(activation.candidate, model)
    record = {
        "runtime": _key(activation),
        "model_sha256": model.sha256.lower(),
        "architecture": model.architecture,
        "format": model.format,
        "weight_types": list(model.weight_types),
        "quantization": model.quantization,
        "variant": model.variant,
        "operation": operation,
        "evidence": str(evidence),
    }
    with installation_lock(receipts_path()):
        records = _read_receipts()
        records = [
            item
            for item in records
            if not (
                item.get("runtime") == record["runtime"]
                and item.get("model_sha256") == record["model_sha256"]
                and item.get("operation") == operation
            )
        ]
        save_json(receipts_path(), {"schema": 1, "receipts": [*records, record]})
    return qualified(activation, model, operation)


def known_variant(sha256):
    # This publisher hash is provenance, independent of filename and remote
    # release ordering. Other models require a recorded actual acceptance.
    if sha256.lower() == "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b":
        return "base"
    for record in _read_receipts():
        if record.get("model_sha256") == sha256.lower() and record.get("variant") == "base":
            return "base"
    return "unknown"


def qualified(activation, model, operation):
    candidate = activation.candidate
    if not activation.identity.complete:
        return candidate
    key = _key(activation)
    for record in _read_receipts():
        if (
            record.get("runtime") == key
            and record.get("model_sha256") == model.sha256.lower()
            and record.get("operation") == operation
            and record.get("variant") == model.variant
            and record.get("architecture") == model.architecture
            and record.get("format") == model.format
            and record.get("weight_types") == list(model.weight_types)
            and record.get("quantization") == model.quantization
            and record.get("evidence")
        ):
            return replace(
                candidate,
                qualified=True,
                qualification=record["evidence"],
                accepted_models=(model.sha256.lower(),),
                accepted_operations=(operation,),
            )
    return candidate
