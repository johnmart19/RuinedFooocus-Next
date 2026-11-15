"""Previewed, idle-only resets using the existing cache and profile owners."""

from collections import defaultdict
from concurrent.futures import CancelledError
from contextlib import ExitStack, closing, contextmanager
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
from types import SimpleNamespace

from ruinedfooocus.core import cache_maintenance, profile_store
from ruinedfooocus.core.cache_contract import canonical_path_key, owned_library_thumbnail
from ruinedfooocus.core.paths import data_path, ensure_app_directory
from ruinedfooocus.core.preparation import PreparationBusy, preparation_service

ACTIONS = frozenset({"cache", "cache-outputs", "factory-profile"})
MAX_ENTRIES = 20000
CACHE_CATEGORIES = ("model-hashes", "library-index", "library-thumbnails", "startup-checks")


@dataclass(frozen=True)
class Target:
    path: str
    kind: str
    signature: tuple | None
    bytes: int = 0
    rows: int = 0


@dataclass(frozen=True)
class ResetPlan:
    action: str
    profile: str | None
    targets: tuple[Target, ...]
    retained: tuple[str, ...]
    fingerprint: str


def _check(cancelled):
    if cancelled is not None and cancelled.is_set():
        raise CancelledError("Maintenance was cancelled")


def _signature(path):
    try:
        value = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(value.st_mode) or getattr(value, "st_file_attributes", 0) & 0x400:
        raise ValueError(f"Linked maintenance target is not eligible: {path}")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _safe_target(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Maintenance target is outside its owned folder")
    cursor = path
    while cursor != root:
        _signature(cursor)
        cursor = cursor.parent
    _signature(root)


def _index_state(path):
    try:
        return _read_index_state(path)
    except sqlite3.DatabaseError as error:
        if getattr(error, "sqlite_errorcode", None) in (
            sqlite3.SQLITE_NOTADB,
            sqlite3.SQLITE_CORRUPT,
        ):
            return None
        raise OSError(str(error)) from error


def _read_index_state(path):
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0)) as connection:
        if connection.execute("PRAGMA user_version").fetchone()[0] not in (0, 2):
            return None
        columns = {row[1] for row in connection.execute("PRAGMA table_info(images)")}
        if not {"fullpath", "path", "json"} <= columns:
            return None
        rows = connection.execute("SELECT count(*) FROM images").fetchone()[0]
        revision = (
            connection.execute("SELECT revision FROM library_snapshot WHERE id=1").fetchone()[0]
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='library_snapshot'"
            ).fetchone()
            else None
        )
    return rows, revision


def _cache_targets(cancelled):
    root = ensure_app_directory("cache")
    view = SimpleNamespace(model_paths={"cache_path": root})
    protected = cache_maintenance._protected_roots(view)
    targets, retained = [], []
    categories = {item.id: item for item in cache_maintenance._categories(view)}
    leaves = {categories[key].paths[0].name for key in CACHE_CATEGORIES}
    for count, path in enumerate(root.iterdir(), 1):
        if count > MAX_ENTRIES:
            raise ValueError("Cache preview limit exceeded; no reset was performed")
        if path.name not in leaves:
            retained.append(str(path))
    for category in CACHE_CATEGORIES:
        path = categories[category].paths[0]
        if not os.path.lexists(path):
            continue
        if not cache_maintenance._safe_path(path, root, protected):
            retained.append(str(path))
            continue
        candidates = (
            path.iterdir() if category in {"model-hashes", "library-thumbnails"} else (path,)
        )
        for count, candidate in enumerate(candidates, 1):
            _check(cancelled)
            if count > cache_maintenance.MAX_ENTRIES:
                raise ValueError("Cache preview limit exceeded; no reset was performed")
            if not cache_maintenance._safe_path(candidate, root, protected):
                retained.append(str(candidate))
                continue
            valid = (
                owned_library_thumbnail(candidate)
                if category == "library-thumbnails"
                else candidate.is_file() and cache_maintenance._owned_json(candidate, category)
                if category != "library-index"
                else candidate.is_file()
            )
            state = _index_state(candidate) if valid and category == "library-index" else None
            if not valid or category == "library-index" and state is None:
                retained.append(str(candidate))
                continue
            signature = _signature(candidate)
            # Row/revision state detects changes still resident in SQLite's WAL.
            if state is not None:
                signature += state
            targets.append(
                Target(
                    str(candidate),
                    category,
                    signature,
                    candidate.stat().st_size,
                    state[0] if state else 0,
                )
            )
    return targets, retained


def _output_targets(cancelled):
    root = ensure_app_directory("outputs")
    pending, targets, count = [root], [], 0
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            _check(cancelled)
            count += 1
            if count > MAX_ENTRIES:
                raise ValueError("Output preview limit exceeded; no reset was performed")
            _safe_target(path, root)
            signature = _signature(path)
            if path.is_dir():
                targets.append(Target(str(path), "output-directory", signature))
                pending.append(path)
            elif path.is_file():
                targets.append(Target(str(path), "output-file", signature, path.stat().st_size))
            else:
                raise ValueError(f"Unsupported output entry is not eligible: {path}")
    return targets


def preview_reset(action, manager, settings=None, *, cancelled=None):
    if action not in ACTIONS:
        raise ValueError("Choose a supported maintenance action")
    _check(cancelled)
    targets, retained, profile = [], [], None
    if action == "factory-profile":
        if settings is None:
            raise ValueError("The current profile is unavailable")
        profile = profile_store.profile_name(settings.name)
        folder = profile_store.ensure_profile_folder(profile)
        profile_store.recover_profile(folder)
        for name in ("settings.json", "paths.json"):
            path = folder / name
            signature = _signature(path)
            if signature is not None:
                signature += (hashlib.sha256(path.read_bytes()).hexdigest(),)
            targets.append(
                Target(
                    str(path),
                    "profile-document",
                    signature,
                    path.stat().st_size if path.exists() else 0,
                )
            )
    else:
        targets, retained = _cache_targets(cancelled)
        if action == "cache-outputs":
            targets.extend(_output_targets(cancelled))
        for key, default in (
            ("cache_path", data_path("cache")),
            ("temp_outputs_path", data_path("outputs")),
        ):
            configured = Path(manager.model_paths[key])
            if configured != default:
                retained.append(str(configured))
    targets.sort(key=lambda target: (target.kind, target.path))
    value = {
        "action": action,
        "profile": profile,
        "targets": [target.__dict__ for target in targets],
    }
    digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return ResetPlan(action, profile, tuple(targets), tuple(sorted(set(retained))), digest)


@contextmanager
def _idle(registry, admission, service, settings, action):
    with ExitStack() as stack:
        locks = [registry.condition, admission.condition, profile_store._lock]
        if settings is not None:
            locks.append(settings._lock)
        for lock in locks:
            if not lock.acquire(blocking=False):
                raise PreparationBusy("The application is busy; retry after current work finishes")
            stack.callback(lock.release)
        if (
            any(not item.finished for item in registry.jobs.values())
            or admission.held
            or admission.waiters
        ):
            raise PreparationBusy("The application is busy; finish or stop current work first")
        if action == "factory-profile":
            scope = [profile_store.profile_folder(settings.name)]
        else:
            scope = [
                data_path("cache", name)
                for name in ("hashes", "images.db", "library-thumbnails", "dependency-checks.json")
            ]
            if action == "cache-outputs":
                scope.append(data_path("outputs"))
        stack.enter_context(service.quiescent(scope))
        yield


def execute_reset(
    plan,
    manager,
    settings=None,
    *,
    confirmed=False,
    cancelled=None,
    service=preparation_service,
    registry=None,
    admission=None,
):
    if not isinstance(plan, ResetPlan) or plan.action not in ACTIONS:
        raise ValueError("Preview the maintenance action first")
    if plan.action != "cache" and confirmed is not True:
        raise ValueError("Confirm the previewed destructive action first")
    if registry is None or admission is None:
        from ruinedfooocus.app.resource_lifecycle import (
            registry as live_registry,
            admission as live_admission,
        )

        registry = live_registry if registry is None else registry
        admission = live_admission if admission is None else admission
    result = {
        "complete": False,
        "cancelled": False,
        "removed": 0,
        "bytes": 0,
        "errors": [],
        "retained": list(plan.retained),
        "profile": plan.profile,
    }
    with _idle(registry, admission, service, settings, plan.action):
        _check(cancelled)
        current = preview_reset(plan.action, manager, settings, cancelled=cancelled)
        if current.fingerprint != plan.fingerprint:
            raise ValueError("Targets changed after preview; recheck them before confirming")
        # Execute only re-derived owner data, never caller-supplied paths/profile.
        plan = current
        result["profile"] = current.profile
        result["retained"] = list(current.retained)
        if plan.action == "factory-profile":
            factory = copy.deepcopy(settings._defaults)
            factory_paths = copy.deepcopy(manager.DEFAULT_PATHS)

            def publish():
                manager.paths = factory_paths
                manager.name = plan.profile
                manager.set_settings_path(plan.profile)

            settings.save_profile(factory, factory_paths, profile=plan.profile, on_publish=publish)
            result["profile_reset"] = True
            result["complete"] = True
            return result
        view = SimpleNamespace(model_paths={"cache_path": data_path("cache")})
        grouped = defaultdict(list)
        for target in current.targets:
            grouped[target.kind].append(Path(target.path))
        try:
            for category in CACHE_CATEGORIES:
                _check(cancelled)
                selected = grouped[category]
                if not selected:
                    continue
                cleared = cache_maintenance.clear_category(
                    category, view, service=service, selected_paths=selected, cancelled=cancelled
                )
                result["removed"] += cleared["removed"]
                result["bytes"] += cleared["bytes"]
                result["errors"].extend(cleared["errors"])
                if cleared.get("cancelled"):
                    raise CancelledError("Maintenance was cancelled")
                if cleared["errors"] or cleared["removed"] != len(selected):
                    raise OSError("Some previewed cache targets could not be reset")
            for path in grouped["output-file"]:
                _check(cancelled)
                _safe_target(path, data_path("outputs"))
                with service.artifact_lease(path, write=True, blocking=False):
                    _safe_target(path, data_path("outputs"))
                    size = path.stat().st_size
                    path.unlink()
                    result["removed"] += 1
                    result["bytes"] += size
            for path in sorted(
                grouped["output-directory"], key=lambda value: len(value.parts), reverse=True
            ):
                _check(cancelled)
                _safe_target(path, data_path("outputs"))
                path.rmdir()
            if plan.action == "cache-outputs":
                # Existing provider invalidates stale output rows on the next Gallery read.
                manager._output_storage_generation = (
                    getattr(manager, "_output_storage_generation", 0) + 1
                )
            result["complete"] = True
        except CancelledError:
            result["cancelled"] = True
        except (OSError, PreparationBusy, ValueError) as error:
            result["errors"].append(str(error))
    return result


def recheck_folders(manager, settings=None):
    manager.ensure_default_storage()
    if settings is not None:
        settings.default_settings
    return [str(ensure_app_directory(name)) for name in ("cache", "outputs", "settings")]
