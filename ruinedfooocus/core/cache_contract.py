"""Small, versioned derived-cache records; source files remain authoritative."""

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.core.paths import data_path

SCHEMA_VERSION = 1
MAX_RECORD_BYTES = 64 * 1024
LIBRARY_THUMBNAILS = "library-thumbnails"
LIBRARY_THUMBNAIL_MAX_BYTES = 256 * 1024 * 1024
LIBRARY_THUMBNAIL_MAX_FILES = 2048
_THUMBNAIL_NAME = re.compile(r"[0-9a-f]{64}\.webp")


def library_thumbnail_root(cache_root):
    """The Library owns only derived grid copies in this direct child."""
    return Path(cache_root) / LIBRARY_THUMBNAILS


def owned_library_thumbnail(path):
    """Do not treat manual files, directories, symlinks or reparse points as ours."""
    path = Path(path)
    if _THUMBNAIL_NAME.fullmatch(path.name) is None:
        return False
    try:
        status = path.lstat()
        return stat.S_ISREG(status.st_mode) and not (
            getattr(status, "st_file_attributes", 0) & 0x400
        )
    except OSError:
        return False


def resolved_path(path):
    """Windows final-path APIs can return either DOS or extended DOS/UNC names."""
    value = str(Path(path).resolve())
    if os.name == "nt":
        if value.lower().startswith("\\\\?\\unc\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
    return Path(value)


def canonical_path_key(path):
    """Comparison/lease identity only; never use a normalized key as an output path."""
    return os.path.normcase(str(resolved_path(path)))


def configured_cache_root(manager=None):
    """Resolve the configured root lazily, without importing application state."""
    if manager is None:
        manager = getattr(sys.modules.get("ruinedfooocus.app.state"), "path_manager", None)
    value = getattr(manager, "model_paths", {}).get("cache_path")
    return Path(value).expanduser().resolve() if value is not None else data_path("cache")


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "canonical_path": canonical_path_key(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def record_key(namespace, identity):
    value = {"schema_version": SCHEMA_VERSION, "namespace": namespace, "identity": identity}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_record(path, namespace, identity, *, max_bytes=MAX_RECORD_BYTES):
    """Unknown, oversized, partial or stale records are misses, including offline."""
    try:
        with Path(path).open("rb") as stream:
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            return None
        value = json.loads(data)
        if (
            isinstance(value, dict)
            and type(value.get("schema_version")) is int
            and value["schema_version"] == SCHEMA_VERSION
            and value.get("namespace") == namespace
            and value.get("identity") == identity
        ):
            return value.get("payload")
    except (OSError, ValueError, UnicodeError):
        pass
    return None


def write_record(path, namespace, identity, payload):
    """The caller holds the artifact write lease; replacement preserves old data on failure."""
    save_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "namespace": namespace,
            "identity": identity,
            "payload": payload,
        },
    )
