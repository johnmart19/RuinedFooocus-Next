"""Bound optional Git network work without leaving a live background fetch."""

from pathlib import Path
import os
import subprocess
import sys

FETCH_TIMEOUT = 10
CLONE_TIMEOUT = 300


def bounded_git_operation(operation, path, source, *, timeout=None):
    """Only the child imports libgit2; terminating it also stops blocked I/O.

    Fetch updates objects/remote refs. Worktree/branch publication remains in the
    caller, after its local-change checks. Clone callers supply a staging path.
    """
    if operation not in {"fetch", "clone"}:
        raise ValueError(f"Unknown Git network operation: {operation}")
    if timeout is None:
        timeout = FETCH_TIMEOUT if operation == "fetch" else CLONE_TIMEOUT
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), operation, str(path), source],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"Git {operation} timed out after {timeout:g} seconds.") from error
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise RuntimeError(f"Git {operation} failed: {detail or 'worker exited unexpectedly'}")


def _worker(operation, path, source):
    import pygit2

    pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)
    repo = None
    try:
        if operation == "clone":
            repo = pygit2.clone_repository(source, path)
        elif operation == "fetch":
            repo = pygit2.Repository(path)
            repo.remotes[source].fetch()
        else:
            raise ValueError(f"Unknown Git network operation: {operation}")
    finally:
        if repo is not None:
            repo.free()


if __name__ == "__main__":
    try:
        _worker(*sys.argv[1:])
    except Exception as error:
        print(str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
