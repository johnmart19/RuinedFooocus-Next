"""Cross-process installer exclusion without importing a compute runtime."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import threading
import time
from tempfile import gettempdir

_guard = threading.Lock()
_locks = {}


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
