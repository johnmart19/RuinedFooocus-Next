"""Persistent native installations and explicit non-destructive cache migration.

Call migration only while native workers/installers are stopped. Discovery never
migrates; a lock cannot stop an older app writing its legacy cache.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
from tempfile import TemporaryDirectory

from ruinedfooocus.core.paths import data_path
from ruinedfooocus.packaging.install_lock import installation_lock, publish_runtime_directory

FAMILIES = ("llama.cpp", "onnx", "sdcpp")
MAX_ENTRIES = 100_000
MAX_BYTES = 64 * 1024**3


def native_runtime_root(name, *parts, data_root=None):
    """Resolve installed payloads/receipts without creating or moving anything."""
    if name not in FAMILIES:
        raise ValueError(f"Unknown native runtime family: {name}")
    root = Path(data_root) if data_root is not None else data_path()
    base = root / "repositories" / name
    result = base.joinpath(*parts)
    if not result.resolve().is_relative_to(base.resolve()):
        raise ValueError("Native runtime path escapes its repository")
    return result


def _plain(path):
    status = path.lstat()
    if path.is_symlink() or getattr(status, "st_file_attributes", 0) & 0x400:
        raise RuntimeError(f"Runtime migration refuses symlink/reparse paths: {path}")
    if not (stat.S_ISDIR(status.st_mode) or stat.S_ISREG(status.st_mode)):
        raise RuntimeError(f"Runtime migration refuses special files: {path}")
    return status


def _ancestors(path):
    for candidate in (path, *path.parents):
        if os.path.lexists(candidate):
            _plain(candidate)


def _inventory(root, check_cancelled=None):
    _ancestors(root)
    pending, files = [root], {}
    entries = total = 0
    while pending:
        with os.scandir(pending.pop()) as children:
            for child in children:
                if check_cancelled:
                    check_cancelled()
                entries += 1
                if entries > MAX_ENTRIES:
                    raise RuntimeError("Native runtime migration exceeds its entry limit")
                path = Path(child.path)
                status = _plain(path)
                if stat.S_ISDIR(status.st_mode):
                    pending.append(path)
                else:
                    total += status.st_size
                    if total > MAX_BYTES:
                        raise RuntimeError("Native runtime migration exceeds its byte limit")
                    files[path.relative_to(root)] = (
                        status.st_size,
                        status.st_mtime_ns,
                        status.st_ctime_ns,
                    )
    return files


def plan_legacy_runtime_migration(data_root, *, families=FAMILIES):
    """Read-only counts; explicit root prevents migrating the wrong checkout."""
    root = Path(data_root).expanduser().absolute()
    _ancestors(root)
    plans = []
    for name in families:
        target = native_runtime_root(name, data_root=root)
        source = root / "cache" / name
        _ancestors(target)
        files = _inventory(source) if os.path.lexists(source) else {}
        plans.append(
            {
                "family": name,
                "source": str(source),
                "destination": str(target),
                "status": ("collision" if files else "existing")
                if os.path.lexists(target)
                else "ready"
                if files
                else "missing",
                "files": len(files),
                "bytes": sum(value[0] for value in files.values()),
            }
        )
    return plans


def _hash(path, check_cancelled):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            if check_cancelled:
                check_cancelled()
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    if path.stat().st_size > 16 * 1024**2:
        raise RuntimeError(f"Runtime receipt is oversized: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Runtime receipt is invalid: {path}")
    return value


def _validate_receipts(root, files, check_cancelled):
    """Preserve existing integrity evidence; never manufacture qualification."""
    for relative in files:
        if relative.name not in ("rf-runtime.json", "verified.json"):
            continue
        receipt = root / relative
        document = _json(receipt)
        if relative.name == "rf-runtime.json":
            payload = document.get("files")
            if isinstance(payload, dict):
                records = list(payload.items())
            elif isinstance(payload, list):
                records = [
                    (row[0], row[2]) for row in payload if isinstance(row, list) and len(row) == 3
                ]
                if len(records) != len(payload):
                    raise RuntimeError(f"Invalid runtime payload receipt: {receipt}")
            else:
                raise RuntimeError(f"Missing runtime payload receipt: {receipt}")
            actual = {
                str(path.relative_to(relative.parent)).replace("\\", "/")
                for path in files
                if path.is_relative_to(relative.parent) and path != relative
            }
            if set(name for name, _ in records) != actual:
                raise RuntimeError(f"Incomplete runtime payload: {receipt}")
        else:
            records = [(document.get("binary"), document.get("sha256"))]
            records += [
                (row.get("path"), row.get("sha256")) for row in document.get("plugin_libraries", [])
            ]
        for name, digest in records:
            if (
                not isinstance(name, str)
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise RuntimeError(f"Invalid runtime file identity: {receipt}")
            path = receipt.parent / name
            if not path.resolve().is_relative_to(receipt.parent.resolve()) or not path.is_file():
                raise RuntimeError(f"Missing or escaping runtime payload: {receipt}")
            if _hash(path, check_cancelled) != digest:
                raise RuntimeError(f"Runtime integrity differs from its receipt: {path}")
    for relative in files:
        if relative.name in ("llama-server", "llama-server.exe", "sd-cli", "sd-cli.exe"):
            parents = (relative.parent, *relative.parent.parents)
            if not any(
                parent / "rf-runtime.json" in files or parent / "ready.json" in files
                for parent in parents
            ):
                raise RuntimeError(
                    f"Unverified legacy runtime; retain for manual recovery: {relative}"
                )
    if root.name == "onnx":
        for abi in root.glob("cp[0-9]*"):
            for bundle in abi.iterdir():
                if bundle.is_dir() and not (bundle / "verified.json").is_file():
                    raise RuntimeError(f"Incomplete legacy ONNX runtime: {bundle}")


def migrate_legacy_runtime_storage(data_root, *, families=FAMILIES, check_cancelled=None):
    """Copy exact bytes and atomically publish; never remove/merge old runtimes.

    Destinations, even incomplete ones, are never replaced. Only owned SDK active
    pointers are rewritten. External custom runtime/provider paths stay external.
    No package manager, native executable or device API is invoked.
    """
    plans = plan_legacy_runtime_migration(data_root, families=families)
    if any(plan["status"] == "collision" for plan in plans):
        raise RuntimeError(
            "Native repository destination already exists; preserve it and reconcile explicitly"
        )
    results = []
    for plan in plans:
        if plan["status"] != "ready":
            results.append(plan)
            continue
        source, target = Path(plan["source"]), Path(plan["destination"])
        with installation_lock(target, timeout=0):
            before = _inventory(source, check_cancelled)
            _validate_receipts(source, before, check_cancelled)
            _ancestors(target)
            if os.path.lexists(target):
                raise RuntimeError("Native repository destination appeared; nothing was replaced")
            target.parent.mkdir(parents=True, exist_ok=True)
            if (
                shutil.disk_usage(target.parent).free
                < sum(item[0] for item in before.values()) + 64 * 1024**2
            ):
                raise RuntimeError(
                    "Insufficient free space to preserve both legacy and persistent runtimes"
                )
            with TemporaryDirectory(prefix="rf-runtime-migration-", dir=target.parent) as temporary:
                stage = Path(temporary) / source.name
                stage.mkdir()
                for relative in before:
                    if check_cancelled:
                        check_cancelled()
                    original, copied = source / relative, stage / relative
                    _ancestors(original)
                    copied.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original, copied)
                    if _hash(original, check_cancelled) != _hash(copied, check_cancelled):
                        raise RuntimeError("Native runtime changed while copying; source retained")
                if _inventory(source, check_cancelled) != before:
                    raise RuntimeError(
                        "Legacy native runtime changed during migration; source retained"
                    )
                _validate_receipts(stage, _inventory(stage, check_cancelled), check_cancelled)
                if plan["family"] == "llama.cpp":
                    for pointer in stage.glob("rocm-*.active.json"):
                        saved = _json(pointer)
                        active = Path(saved.get("directory", ""))
                        if active.is_absolute() and active.resolve().parent == source.resolve():
                            saved["directory"] = str(target / active.name)
                            pointer.write_text(json.dumps(saved, indent=2), encoding="utf-8")
                if check_cancelled:
                    check_cancelled()
                _ancestors(target)
                if os.path.lexists(target):
                    raise RuntimeError(
                        "Native repository destination appeared; nothing was replaced"
                    )

                def validate_source():
                    if _inventory(source, check_cancelled) != before:
                        raise RuntimeError("Legacy native runtime changed before publication")

                publish_runtime_directory(
                    stage, target, check_cancelled=check_cancelled, validate=validate_source
                )
            results.append(plan | {"status": "copied-source-retained"})
    return results
