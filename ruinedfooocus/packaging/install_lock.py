"""Cross-process installer exclusion without importing a compute runtime."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import stat
import threading
import time
from tempfile import gettempdir

_guard = threading.Lock()
_locks = {}
_WINDOWS = os.name == "nt"
_PUBLICATION_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)


def _plain_directory(path):
    for directory in (path, *path.parents):
        status = directory.lstat()
        if (
            not stat.S_ISDIR(status.st_mode)
            or stat.S_ISLNK(status.st_mode)
            or getattr(status, "st_file_attributes", 0) & 0x400
        ):
            raise RuntimeError(f"Runtime publication requires a plain directory: {directory}")
    return path.stat()


def publish_runtime_directory(source, destination, *, check_cancelled=None, validate=None):
    """Publish a staged directory without replacing an existing installation.

    The caller holds its installation lock and validates payloads before calling.
    Windows can transiently deny publication after fresh extraction/copying. Retry
    only access/sharing violations, at most six attempts over 1.55s of waits.
    Persistent failures retain their original exception; no permissions change.
    """
    source, destination = Path(source), Path(destination)
    original = _plain_directory(source)
    identity = original.st_dev, original.st_ino
    for attempt in range(len(_PUBLICATION_DELAYS) + 1):
        if check_cancelled:
            check_cancelled()
        current = _plain_directory(source)
        if (current.st_dev, current.st_ino) != identity:
            raise RuntimeError("Staged runtime directory changed before publication")
        _plain_directory(destination.parent)
        if os.path.lexists(destination):
            raise FileExistsError(
                f"Runtime destination appeared; nothing was replaced: {destination}"
            )
        if validate:
            validate()
        try:
            # Windows rename is atomic and fails if the destination exists.
            os.rename(source, destination)
            return
        except OSError as error:
            if (
                not _WINDOWS
                or getattr(error, "winerror", None) not in (5, 32)
                or attempt == len(_PUBLICATION_DELAYS)
            ):
                raise
            remaining = _PUBLICATION_DELAYS[attempt]
            while remaining > 0:
                if check_cancelled:
                    check_cancelled()
                pause = min(0.05, remaining)
                time.sleep(pause)
                remaining -= pause


@contextmanager
def installation_lock(target, *, timeout=120):
    """Serialize cooperating installers for a canonical environment or artifact.

    OS locks are released after a crashed process; lock files carry no ownership
    state and can remain on disk. They are outside the destination being staged.
    """
    identity = os.path.normcase(str(Path(target).resolve()))
    key = hashlib.sha256(identity.encode()).hexdigest()
    with _guard:
        lock = _locks.setdefault(key, threading.Lock())
    if not lock.acquire(timeout=timeout):
        raise RuntimeError("Another installer owns this runtime; retry when it finishes.")
    try:
        # Profiles may share an interpreter/artifact. RF_DATA_DIR must not split
        # their OS lock domains; the user temp directory is shared across those
        # profiles and remains writable for read-only application installs.
        path = Path(gettempdir()) / "ruinedfooocus-installer-locks" / (key + ".lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as stream:
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"0")
                stream.flush()
            deadline = time.monotonic() + timeout
            while True:
                try:
                    stream.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Another installer owns this runtime; retry later.")
                    time.sleep(0.1)
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)
    finally:
        lock.release()
