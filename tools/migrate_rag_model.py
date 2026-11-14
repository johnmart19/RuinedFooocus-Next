"""Explicit idle, copy-only migration of the eight pinned legacy RAG model files."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ruinedfooocus.core.paths import data_path
from ruinedfooocus.core.preparation import PreparationRequest, preparation_service, verify_file
from ruinedfooocus.features.chat.rag_preparation import MODEL_FILES, MODEL_ID, MODEL_REVISION


def _safe(path):
    """Reject directory links/reparse points before resolving or creating anything."""
    for item in (path, *path.parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Migration refuses a link/reparse path: {item}")
    return path.resolve()


def migrate(data_root=None, *, apply=False, idle=False, service=preparation_service):
    if apply and not idle:
        raise ValueError("Apply requires --idle after the application owner confirms it is idle.")
    root = Path(data_root).expanduser().absolute() if data_root is not None else data_path()
    source = _safe(root / "cache" / "rag" / MODEL_REVISION)
    destination = _safe(root / "models" / "rag_embeddings" / MODEL_REVISION)
    rows = []
    for name, size, checksum in MODEL_FILES:
        before, after = _safe(source / name), _safe(destination / name)
        if not before.is_relative_to(source) or not after.is_relative_to(destination):
            raise ValueError("RAG migration paths must stay within their exact model roots.")
        for path in (before, after):
            if path.exists() and (not path.is_file() or path.stat().st_size != size):
                raise IOError(f"Preserved conflicting or incomplete model file: {path}")
        rows.append((before, after, size, checksum))
    report = dict(
        model=MODEL_ID,
        revision=MODEL_REVISION,
        source=str(source),
        destination=str(destination),
        apply=apply,
        source_deleted=False,
        overwritten=False,
        files=[
            dict(name=str(b.relative_to(destination)), present=b.is_file()) for _, b, _, _ in rows
        ],
    )
    missing = [a for a, b, _, _ in rows if not b.is_file() and not a.is_file()]
    if missing:
        return report | {"status": "missing-source", "missing": [str(p) for p in missing]}
    if not apply:
        return report | {"status": "ready"}
    with ExitStack() as leases:
        for before, after, _, _ in rows:
            if before.is_file():
                leases.enter_context(service.artifact_lease(before, blocking=False))
            leases.enter_context(service.artifact_lease(after, write=True, blocking=False))
        # Verify the complete available closure before publishing any new file.
        for before, after, size, checksum in rows:
            selected = after if after.is_file() else before
            verify_file(selected, PreparationRequest("", selected, size, checksum))
        copied = []
        for before, after, size, checksum in rows:
            if after.is_file():
                continue
            signature = before.stat()
            _safe(after)
            after.parent.mkdir(parents=True, exist_ok=True)
            _safe(after)
            handle, temporary = tempfile.mkstemp(prefix=".rag-copy-", dir=after.parent)
            temporary = Path(temporary)
            try:
                with os.fdopen(handle, "wb") as target, before.open("rb") as original:
                    shutil.copyfileobj(original, target, 1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                verify_file(temporary, PreparationRequest("", temporary, size, checksum))
                current = before.stat()
                if (signature.st_size, signature.st_mtime_ns, signature.st_ctime_ns) != (
                    current.st_size,
                    current.st_mtime_ns,
                    current.st_ctime_ns,
                ):
                    raise IOError("Legacy RAG source changed during copying; source retained.")
                # Hard-link publication fails if another writer created the final
                # file. Unlike replace/rename, it never overwrites on any OS.
                os.link(temporary, after)
                copied.append(str(after.relative_to(destination)))
            finally:
                temporary.unlink(missing_ok=True)
    return report | {"status": "copied" if copied else "already-present", "copied": copied}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--apply", action="store_true", help="Copy the pinned files; never delete/overwrite."
    )
    parser.add_argument(
        "--idle", action="store_true", help="The app owner has confirmed the app is idle."
    )
    args = parser.parse_args()
    try:
        report = migrate(args.data_root, apply=args.apply, idle=args.idle)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps(report, indent=2))
    return 2 if report["status"] == "missing-source" else 0


if __name__ == "__main__":
    raise SystemExit(main())
