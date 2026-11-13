"""Identify the local checkpoint without assuming a release's first file is it."""

from functools import lru_cache
from collections.abc import Mapping
import hashlib
import re
import json
from pathlib import Path

from ruinedfooocus.core.cache_contract import (
    file_identity,
    canonical_path_key,
    read_record,
    record_key,
    write_record,
)
from ruinedfooocus.core.model_formats import is_safetensors
from ruinedfooocus.core.preparation import preparation_service


def cached_file_sha256(filename, cache_dir):
    """Hash unchanged weights once, outside picker/render paths; never write beside them."""
    filename = Path(filename)
    signature = file_signature(filename)
    identity = file_identity(filename) | {"parser_version": 1}
    cache = Path(cache_dir) / (record_key("model-sha256", identity) + ".json")
    # Callers can already hold model read leases. Do not nest a second source
    # lease behind a waiting model writer; source stability is checked below.
    with preparation_service.artifact_lease(cache, write=True):
        payload = read_record(cache, "model-sha256", identity)
        checksum = payload.get("sha256") if isinstance(payload, dict) else None
        if isinstance(checksum, str) and re.fullmatch(r"[A-F0-9]{64}", checksum):
            return checksum
        # Compatibility with the original unversioned signature-keyed cache.
        legacy = Path(cache_dir) / (
            hashlib.sha256(json.dumps(signature).encode()).hexdigest() + ".json"
        )
        try:
            with legacy.open("rb") as source:
                data = source.read(64 * 1024 + 1)
            cached = json.loads(data) if len(data) <= 64 * 1024 else {}
            checksum = cached.get("sha256")
            if not (
                cached.get("signature") == signature
                and isinstance(checksum, str)
                and re.fullmatch(r"[A-F0-9]{64}", checksum)
            ):
                checksum = None
        except (OSError, ValueError, AttributeError):
            checksum = None
        if checksum is None:
            digest = hashlib.sha256()
            with open(filename, "rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            checksum = digest.hexdigest().upper()
        if signature != file_signature(filename):
            raise OSError("Model changed while hashing; retry when the file is stable.")
        try:
            write_record(cache, "model-sha256", identity, {"sha256": checksum})
        except OSError as error:
            print(f"Could not cache model hash: {error}")
        return checksum


def file_signature(path):
    path = Path(path)
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _source_record(path, cache_root):
    identity = file_identity(path) | {"parser_version": 1}
    record = (
        Path(cache_root) / "model-sources" / (record_key("model-native-source", identity) + ".json")
    )
    return identity, record


def record_prepared_native_source(path, request, verified_signature, cache_root):
    """Reuse a service SHA-verified receipt; browsing never creates this proof."""
    from ruinedfooocus.features.models.model_registry import load_registry

    if verified_signature is None:
        return None
    if not isinstance(verified_signature, Mapping):
        raise ValueError("Prepared source identity must be an immutable completion receipt.")
    path = Path(path)
    current = file_identity(path)
    if dict(verified_signature) != current:
        raise ValueError("Prepared model changed after verification; refresh its source identity.")
    if canonical_path_key(request.destination) != current["canonical_path"]:
        raise ValueError("Prepared source receipt belongs to a different destination.")
    checksum = getattr(request, "sha256", None)
    if not isinstance(checksum, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", checksum):
        return None
    matches = [
        row
        for row in load_registry().entries(role="image_diffusion_checkpoint")
        if is_native_checkpoint_source(row)
        and row["artifact"]["sha256"] == checksum.lower()
        and row["artifact"]["bytes"] == current["size"]
        and row["artifact"].get("pinned_url") == request.url
    ]
    if not matches:
        return None
    contracts = {
        (
            row["native_family"],
            row.get("native_class", row.get("architecture")),
            row.get("behavioral_variant"),
            row.get("evidence", {}).get("default_recipe_id"),
        )
        for row in matches
    }
    if request.expected_size != current["size"] or len(contracts) != 1:
        raise ValueError("Prepared model receipt conflicts with its published native source.")
    row = matches[0]
    signature = file_signature(path)
    payload = dict(
        baseModel=row["native_family"],
        _rf_local_sha256=checksum.lower(),
        _rf_file_signature=signature,
        _rf_verified_native_source=dict(
            artifact=row["id"], family=row["native_family"], sha256=checksum.lower()
        ),
        _rf_source_verification="preparation_service_sha256",
    )
    identity, record = _source_record(path, cache_root)
    with preparation_service.artifact_lease(record, write=True):
        if current != file_identity(path):
            raise ValueError("Prepared model changed before source metadata publication.")
        write_record(record, "model-native-source", identity, payload)
        if current != file_identity(path):
            raise ValueError("Prepared model changed during source metadata publication.")
    return payload


def read_prepared_native_source(path, cache_root):
    """An invalid/stale optional source record is a miss, including offline."""
    try:
        identity, record = _source_record(path, cache_root)
        payload = read_record(record, "model-native-source", identity)
        if not isinstance(payload, dict) or verified_native_source(path, payload) is None:
            return {}
        return payload
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def is_native_checkpoint_source(row):
    return (
        row.get("role") == "image_diffusion_checkpoint"
        and isinstance(row.get("catalogue_key"), str)
        and row["catalogue_key"].startswith(("checkpoints/", "diffusion_models/"))
        and isinstance(row.get("native_family"), str)
        and bool(row["native_family"])
    )


def verified_native_source(path, metadata, *, current_signature=None):
    """Reuse only a current explicit verification of a packaged source identity."""
    from ruinedfooocus.features.models.model_registry import load_registry, RegistryError

    if not isinstance(metadata, dict):
        return None
    proof = metadata.get("_rf_verified_native_source")
    if not isinstance(proof, dict):
        return None
    try:
        signature = file_signature(path)
        registry = load_registry()
        artifact = getattr(registry, "artifact", None)
        entries = (artifact(proof.get("artifact")),) if callable(artifact) else registry.entries()
    except (OSError, RegistryError):
        return None
    if current_signature is not None and current_signature != signature:
        return None
    if metadata.get("_rf_file_signature") != signature or metadata.get(
        "_rf_local_sha256"
    ) != proof.get("sha256"):
        return None
    for row in entries:
        if (
            row is not None
            and row["id"] == proof.get("artifact")
            and is_native_checkpoint_source(row)
            and row.get("native_family") == proof.get("family")
            and row["artifact"]["sha256"] == proof.get("sha256")
            and row["artifact"]["bytes"] == signature[1]
        ):
            return row
    return None


def files_unchanged(signatures):
    if signatures is None:
        return False
    try:
        return all(signature == file_signature(signature[0]) for signature in signatures)
    except OSError:
        return False


def read_safetensors_header(path):
    """Bounded shape/metadata inspection without importing the tensor runtime."""
    path = Path(path)
    if not is_safetensors(path):
        raise ValueError("Select a safetensors checkpoint")
    with path.open("rb") as source:
        raw = source.read(8)
        size = int.from_bytes(raw, "little")
        if len(raw) != 8 or not 2 <= size <= min(16 * 1024**2, path.stat().st_size - 8):
            raise ValueError("Invalid or oversized safetensors header")
        try:
            result = json.loads(source.read(size))
        except RecursionError as error:
            raise ValueError(
                "Safetensors header metadata exceeds the JSON nesting limit."
            ) from error
    if not isinstance(result, dict):
        raise ValueError("Invalid safetensors header")
    if any(not isinstance(value, dict) for key, value in result.items() if key != "__metadata__"):
        raise ValueError("Invalid tensor entries in safetensors header")
    return result


@lru_cache(maxsize=256)
def _cached_header(path, signature):
    return read_safetensors_header(path)


def cached_safetensors_header(path):
    """Header of an unchanged file is parsed once; a replaced file is read again."""
    path = Path(path)
    return _cached_header(str(path), tuple(file_signature(path)[1:]))


def checkpoint_hashes(metadata):
    files = metadata.get("files", [])
    files = [item for item in files if isinstance(item, dict)] if isinstance(files, list) else []
    local = metadata.get("_rf_local_sha256")
    if isinstance(local, str) and re.fullmatch(r"[a-fA-F0-9]{64}", local):
        for item in files:
            hashes = item.get("hashes", {})
            if isinstance(hashes, dict) and str(hashes.get("SHA256", "")).lower() == local.lower():
                return hashes | {"SHA256": local.upper()}
        return {"SHA256": local.upper()}
    if len(files) == 1 and isinstance(files[0].get("hashes"), dict):
        return files[0]["hashes"]
    return {}
