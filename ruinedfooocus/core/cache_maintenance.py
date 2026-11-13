"""Bounded, allowlisted cleanup of derived records, never runtime/input trees."""

from dataclasses import dataclass
from contextlib import closing
import json
from itertools import islice
from pathlib import Path
import re
import stat

from ruinedfooocus.core.cache_contract import (
    canonical_path_key,
    configured_cache_root,
    library_thumbnail_root,
    owned_library_thumbnail,
    resolved_path,
)
from ruinedfooocus.core.paths import data_path
from ruinedfooocus.core.preparation import PreparationBusy, preparation_service

MAX_ENTRIES = 5000
_DIGEST = re.compile(r"[0-9a-fA-F]{64}")


@dataclass(frozen=True)
class Category:
    id: str
    label: str
    paths: tuple
    protected: bool
    reason: str


def cache_roots(manager=None):
    result = {}
    for path in (configured_cache_root(manager), data_path("cache")):
        result.setdefault(canonical_path_key(path), path)
    return tuple(result.values())


def _categories(manager):
    configured = configured_cache_root(manager)
    data = data_path("cache")
    roots = cache_roots(manager)
    return (
        Category(
            "model-hashes",
            "Model hash records",
            (configured / "hashes",),
            False,
            "Regenerated from local model files. Legacy records are retained.",
        ),
        Category(
            "library-index",
            "Library index",
            (configured / "images.db",),
            False,
            "Refresh Library to rebuild metadata. Original images are preserved.",
        ),
        Category(
            "library-thumbnails",
            "Library thumbnails",
            (library_thumbnail_root(configured),),
            False,
            "Regenerated when browsing Library. Original images are preserved.",
        ),
        Category(
            "startup-checks",
            "Startup verification",
            (data / "dependency-checks.json",),
            False,
            "Recheck the current environment on next startup.",
        ),
        Category(
            "artwork",
            "Model artwork and metadata",
            tuple(
                root / name
                for root in roots
                for name in ("checkpoints", "loras", "embeddings", "card-previews")
            ),
            True,
            "Retained for offline browsing and user artwork.",
        ),
        Category(
            "model-sources",
            "Verified model source records",
            tuple(root / "model-sources" for root in roots),
            True,
            "Retained to reuse verified source identities without rehashing model files.",
        ),
        Category(
            "catalogue-previews",
            "Publisher model artwork",
            tuple(root / "catalogue-previews" for root in roots),
            True,
            "Retained for current Model Hub views and offline browsing.",
        ),
        Category(
            "native-image-previews",
            "Native image display previews",
            tuple(root / "native-image-previews" for root in roots),
            True,
            "May be used by active Library views and image workflows.",
        ),
        Category(
            "native-image-exports",
            "Exported image results",
            tuple(root / "native-image-exports" for root in roots),
            True,
            "User export results may still be displayed, downloaded or saved.",
        ),
        Category(
            "inputs",
            "Uploads and Chat inputs",
            tuple(root / name for root in roots for name in ("uploads", "embeds", "rag")),
            True,
            "May be required by active or saved workflows.",
        ),
        Category(
            "pending",
            "Downloads and Inbox state",
            tuple(root / name for root in roots for name in ("preparation", "inbox")),
            True,
            "Retains active downloads and resumable work.",
        ),
        Category(
            "runtimes",
            "Installed runtimes and recovery",
            tuple(
                root / name
                for root in roots
                for name in (
                    "llama.cpp",
                    "sdcpp",
                    "onnx",
                    "installer-tools",
                    "package-plans",
                    "clip-interrogator",
                    "qwen-image-2.1-pe",
                    "themes",
                    "chat_sources",
                )
            ),
            True,
            "Managed by the matching runtime, model or source action.",
        ),
    )


def _protected_roots(manager):
    roots = [data_path("settings"), data_path("models"), data_path("outputs")]
    for key, value in getattr(manager, "model_paths", {}).items():
        if key in {"cache_path", "diffusers_cache_path"}:
            continue
        for path in value if isinstance(value, (tuple, list)) else (value,):
            if isinstance(path, (str, Path)):
                roots.append(Path(path).expanduser().resolve())
    profile = getattr(manager, "settings_path", None)
    if profile:
        roots.append(Path(profile).resolve().parent)
    return roots


def _linked(path):
    status = path.lstat()
    return stat.S_ISLNK(status.st_mode) or bool(getattr(status, "st_file_attributes", 0) & 0x400)


def _safe_path(path, root, protected):
    """Do not follow child symlinks/junctions or aliases into protected stores."""
    original, root = Path(path).absolute(), resolved_path(root)
    try:
        cursor = original
        root_key = canonical_path_key(root)
        while True:
            if _linked(cursor):
                return False
            if canonical_path_key(cursor) == root_key:
                break
            if cursor.parent == cursor:
                return False
            cursor = cursor.parent
        resolved = resolved_path(original)
        if not resolved.is_relative_to(root):
            return False
        if any(
            resolved.is_relative_to(parent) or parent.is_relative_to(resolved)
            for parent in protected
        ):
            return False
        return True
    except OSError:
        return False


def _files(paths, limit=MAX_ENTRIES):
    """One bounded snapshot; links are counted as skipped, never traversed."""
    visited = 0
    stack = list(paths)
    while stack:
        path = stack.pop()
        try:
            if not path.exists() or _linked(path):
                continue
            if path.is_dir():
                for child in path.iterdir():
                    visited += 1
                    if visited > limit:
                        return
                    stack.append(child)
            elif path.is_file():
                visited += 1
                if visited > limit:
                    return
                yield path
        except OSError:
            continue


def maintenance_categories(manager=None):
    rows = []
    for category in _categories(manager):
        files = total = 0
        for path in _files(category.paths):
            try:
                total += path.stat().st_size
                files += 1
            except OSError:
                pass
        rows.append(
            {
                "id": category.id,
                "label": category.label,
                "bytes": total,
                "files": files,
                "protected": category.protected,
                "reason": category.reason,
                "bounded": True,
            }
        )
    return rows


def _owned_json(path, category):
    try:
        with path.open("rb") as source:
            data = source.read(64 * 1024 + 1)
        if len(data) > 64 * 1024:
            return False
        value = json.loads(data)
        if not isinstance(value, dict):
            return False
        if category == "model-hashes":
            from ruinedfooocus.core.cache_contract import record_key

            identity = value.get("identity")
            payload = value.get("payload")
            return (
                type(value.get("schema_version")) is int
                and value["schema_version"] == 1
                and value.get("namespace") == "model-sha256"
                and isinstance(identity, dict)
                and set(identity)
                == {"canonical_path", "size", "mtime_ns", "ctime_ns", "parser_version"}
                and isinstance(identity.get("canonical_path"), str)
                and bool(identity["canonical_path"])
                and all(
                    type(identity.get(key)) is int and identity[key] >= 0
                    for key in ("size", "mtime_ns", "ctime_ns")
                )
                and type(identity.get("parser_version")) is int
                and identity["parser_version"] == 1
                and isinstance(payload, dict)
                and isinstance(payload.get("sha256"), str)
                and _DIGEST.fullmatch(payload["sha256"]) is not None
                and path.name == record_key("model-sha256", identity) + ".json"
            )
        return bool(value) and all(
            isinstance(key, str) and isinstance(digest, str) and _DIGEST.fullmatch(digest)
            for key, digest in value.items()
        )
    except (OSError, ValueError, UnicodeError):
        return False


def clear_category(category_id, manager=None, *, service=preparation_service):
    categories = {item.id: item for item in _categories(manager)}
    category = categories.get(category_id)
    if category is None or category.protected:
        raise ValueError("Choose a derived cache category that can be cleared")
    protected = _protected_roots(manager)
    result = {"removed": 0, "bytes": 0, "skipped": 0, "errors": []}
    root = data_path("cache") if category_id == "startup-checks" else configured_cache_root(manager)
    # Only direct owned directories or exact marker/index are eligible; never recurse.
    if category_id in {"model-hashes", "library-thumbnails"}:
        directory = category.paths[0]
        if not directory.is_dir():
            return result
        if not _safe_path(directory, root, protected):
            result["skipped"] += 1
            return result
        if category_id == "library-thumbnails":
            try:
                with service.artifact_lease(directory, write=True, blocking=False):
                    return _clear_thumbnails(directory, root, protected)
            except PreparationBusy:
                result["skipped"] += 1
                return result
            except OSError as error:
                result["errors"].append(f"{directory.name}: {error}")
                return result
        candidates = list(islice(directory.glob("*.json"), MAX_ENTRIES))
    else:
        candidates = category.paths
    for path in candidates:
        if not path.exists():
            continue
        if not _safe_path(path, root, protected):
            result["skipped"] += 1
            continue
        try:
            with service.artifact_lease(path, write=True, blocking=False):
                if not _safe_path(path, root, protected):
                    result["skipped"] += 1
                    continue
                if category_id == "library-index":
                    import sqlite3

                    with closing(sqlite3.connect(path, timeout=0)) as connection, connection:
                        if connection.execute("PRAGMA user_version").fetchone()[0] not in (0, 2):
                            result["skipped"] += 1
                            continue
                        columns = {
                            row[1] for row in connection.execute("PRAGMA table_info(images)")
                        }
                        if not {"fullpath", "path", "json"} <= columns:
                            result["skipped"] += 1
                            continue
                        connection.execute("DELETE FROM images")
                        if connection.execute(
                            "SELECT 1 FROM sqlite_master WHERE name='library_snapshot'"
                        ).fetchone():
                            connection.execute(
                                "UPDATE library_snapshot SET revision=revision+1,roots='[]' WHERE id=1"
                            )
                    result["removed"] += 1
                elif _owned_json(path, category_id):
                    size = path.stat().st_size
                    path.unlink()
                    result["removed"] += 1
                    result["bytes"] += size
                else:
                    result["skipped"] += 1
        except PreparationBusy:
            result["skipped"] += 1
        except Exception as error:
            result["errors"].append(f"{path.name}: {error}")
    return result


def _clear_thumbnails(directory, root, protected):
    result = {"removed": 0, "bytes": 0, "skipped": 0, "errors": []}
    for path in islice(directory.iterdir(), MAX_ENTRIES):
        if not _safe_path(path, root, protected) or not owned_library_thumbnail(path):
            result["skipped"] += 1
            continue
        try:
            size = path.stat().st_size
            path.unlink()
            result["removed"] += 1
            result["bytes"] += size
        except OSError as error:
            result["errors"].append(f"{path.name}: {error}")
    return result
