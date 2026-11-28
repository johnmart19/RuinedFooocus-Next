"""Pinned native capability dispositions joined to exact registered artifacts."""

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

from ruinedfooocus.features.models.model_registry import (
    RegistryError,
    _freeze,
    _read_document,
    _require,
    metadata_digest,
)

KINDS = ("configurations", "roles", "clip_types", "te_models")


@lru_cache(maxsize=32)
def _read(path, modified, size, digest=None):
    _require(size <= 1024 * 1024, "Oversized native catalogue resource.")
    document = _read_document(path)
    _require(
        isinstance(document, dict)
        and type(document.get("schema_version")) is int
        and document["schema_version"] == 1,
        "Unsupported native catalogue schema.",
    )
    _require(
        digest is None or metadata_digest(document) == digest, "Changed native catalogue resource."
    )
    return _freeze(document)


def _resource(root, name, digest=None):
    _require(isinstance(name, str), "Invalid native catalogue resource path.")
    path = (root / name).resolve()
    _require(path.is_relative_to(root) and path != root, "Native catalogue resource escapes root.")
    try:
        stat = path.stat()
        return _read(str(path), stat.st_mtime_ns, stat.st_size, digest)
    except OSError as error:
        raise RegistryError(f"Native catalogue unavailable: {path}") from error


def _artifacts(row, registry):
    identities = row.get("available_artifact_ids", ())
    _require(
        isinstance(identities, (tuple, list))
        and all(isinstance(identity, str) for identity in identities)
        and len(identities) == len(set(identities)),
        "Invalid native catalogue artifact references.",
    )
    groups = {}
    for identity in identities:
        artifact = registry.artifact(identity)
        _require(artifact is not None, f"Unknown native catalogue artifact: {identity}")
        _require(
            artifact["state"] == "existing"
            and artifact["role"]
            in (
                "image_diffusion_checkpoint",
                "depth_estimation",
                "image_utility_model",
            ),
            f"Invalid native model role: {identity}",
        )
        consumers = (registry.profile(artifact["profile_id"]) or {}).get("consumers", ())
        _require(
            artifact.get("native_family") == row["id"]
            or artifact.get("native_class") == row["id"]
            or row["id"] in consumers
            or (row["id"] == "DepthAnything3" and artifact["role"] == "depth_estimation"),
            f"Native catalogue family mismatch: {identity}",
        )
        groups.setdefault(artifact.get("behavioral_variant") or "unspecified", []).append(artifact)
    return [
        {
            "behavioral_variant": variant,
            "artifact_ids": tuple(
                row["id"]
                for row in sorted(
                    entries,
                    key=lambda row: (
                        row["artifact"]["bytes"] is None,
                        row["artifact"]["bytes"] or 0,
                        row["id"],
                    ),
                )
            ),
            "selection_basis": "Smallest published payload within this behavioral variant; device fit and output quality remain unqualified.",
        }
        for variant, entries in groups.items()
    ]


def dispositions(registry, resource_root, *, kind="configurations", media=None):
    """No model discovery, weights, downloads or qualification history reads."""
    if kind not in KINDS:
        raise RegistryError(f"Unknown native catalogue row kind: {kind}")
    if resource_root is None:
        return ()
    base = Path(resource_root).resolve()
    root = (base / "native_coverage").resolve()
    _require(root.is_relative_to(base), "Native catalogue directory escapes resource root.")
    if not (root / "index.json").exists():
        return ()  # Legacy/custom registries need not ship this optional evidence.
    index = _resource(root, "index.json")
    _require(
        isinstance(index.get("groups"), Mapping) and isinstance(index.get("shards"), Mapping),
        "Invalid native catalogue index.",
    )
    names = index["groups"].get(kind)
    _require(
        isinstance(names, (tuple, list)) and all(isinstance(name, str) for name in names),
        "Invalid native catalogue group.",
    )
    rows, identities = [], set()
    for name in names:
        descriptor = index["shards"].get(name)
        _require(
            isinstance(descriptor, Mapping)
            and descriptor.get("kind") == kind
            and isinstance(descriptor.get("sha256"), str),
            "Invalid native catalogue shard reference.",
        )
        if (
            media is not None
            and descriptor.get("media") is not None
            and descriptor["media"] != media
        ):
            continue
        document = _resource(root, name, descriptor["sha256"])
        entries = document.get("rows")
        _require(
            document.get("kind") == kind
            and isinstance(entries, (tuple, list))
            and len(entries) == descriptor.get("rows"),
            "Invalid native catalogue rows.",
        )
        for row in entries:
            _require(
                isinstance(row, Mapping)
                and isinstance(row.get("id"), str)
                and row["id"] not in identities,
                "Duplicate or invalid native catalogue identity.",
            )
            identities.add(row["id"])
            _require(
                all(
                    isinstance(row.get(key), str) and bool(row[key])
                    for key in (
                        "media",
                        "reason",
                        "source_support",
                        "installed_status",
                        "qualification",
                    )
                ),
                "Missing native catalogue disposition.",
            )
            if media is None or row["media"] == media:
                result = dict(row)
                if kind == "configurations":
                    _require(
                        all(
                            isinstance(row.get(key), str) and bool(row[key])
                            for key in (
                                "app_route",
                                "download_disposition",
                                "download_reason",
                                "native_source_url",
                            )
                        ),
                        "Missing native configuration route.",
                    )
                    result["artifact_choices"] = _artifacts(row, registry)
                rows.append(_freeze(result))
    return tuple(rows)
