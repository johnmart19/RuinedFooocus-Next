"""Bounded file preparation, independent of model loading and compute ownership.

Leases and single-flight are process-local. Every cooperating importer, remover or
loader must use this service's artifact lease for the captured canonical path.
"""

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import dataclass, field
import errno
import hashlib
import os
from pathlib import Path
import threading
import time
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from urllib.parse import urlparse

from ruinedfooocus.core.cache_contract import (
    canonical_path_key,
    configured_cache_root,
    resolved_path,
    file_identity,
)


class PreparationBusy(RuntimeError):
    """The bounded preparation queue is full."""


class PreparationConflict(RuntimeError):
    """A destination already has an incompatible or cancelling preparation."""


@dataclass(frozen=True)
class PreparationRequest:
    url: str
    destination: Path
    expected_size: int | None = None
    sha256: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "destination", resolved_path(self.destination))
        if self.expected_size is not None and self.expected_size <= 0:
            raise ValueError("Expected size must be positive")
        if self.sha256 is not None:
            digest = self.sha256.lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("Expected a SHA256 hexadecimal digest")
            object.__setattr__(self, "sha256", digest)


@dataclass
class _Artifact:
    users: int = 0
    readers: int = 0
    writer: bool = False
    waiting_writers: int = 0


@dataclass
class _Job:
    request: PreparationRequest
    subscribers: dict = field(default_factory=dict)
    cancelled: threading.Event = field(default_factory=threading.Event)
    progress: tuple = (0, None, 0)
    verified_signature: object = None
    future: object = None


class PreparationSubscription:
    """One caller's result/cancel handle; cancelling never cancels another caller."""

    def __init__(self, service, job, progress):
        self._service = service
        self._job = job
        self._progress = progress
        self._context = copy_context()
        self._future = Future()
        self._callback_lock = threading.RLock()

    @property
    def request(self):
        return self._job.request

    @property
    def verified_signature(self):
        """Immutable identity captured after SHA verification under the write lease."""
        if (
            not self._future.done()
            or self._future.cancelled()
            or self._future.exception() is not None
        ):
            return None
        return self._job.verified_signature

    def result(self, timeout=None):
        return self._future.result(timeout)

    def done(self):
        return self._future.done()

    def cancel(self):
        # Return only after an in-progress callback has finished.
        with self._callback_lock, self._service._condition:
            if self._future.done():
                return False
            self._future.cancel()
            self._job.subscribers.pop(id(self), None)
            if not self._job.subscribers:
                self._job.cancelled.set()
                # A cancelled waiting transfer must return its admission slot
                # immediately, even while another transfer is blocked in I/O.
                future = self._job.future
                if future is not None and future.cancel():
                    key = self._service._key(self.request.destination)
                    if self._service._jobs.get(key) is self._job:
                        self._service._jobs.pop(key)
                self._service._condition.notify_all()
            return True

    def _report(self, received, total, speed):
        with self._callback_lock:
            if self._future.done() or self._progress is None:
                return
            try:
                self._context.run(self._progress, received, total, speed)
            except Exception:
                # A disconnected UI subscriber cannot fail a shared transfer.
                pass


class PreparationService:
    def __init__(self, max_workers=2, max_jobs=8, transport=None, *, staging_root=None):
        if max_workers < 1 or max_jobs < max_workers:
            raise ValueError("Require 1 <= max_workers <= max_jobs")
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="prepare")
        self._condition = threading.Condition(threading.RLock())
        self._jobs = {}
        self._artifacts = {}
        self._max_jobs = max_jobs
        self._transport = transport or transfer_file
        self._staging_root = Path(staging_root).resolve() if staging_root is not None else None
        self._closed = False

    @staticmethod
    def _key(destination):
        return canonical_path_key(destination)

    def _staging_directory(self, request):
        key = self._key(request.destination)
        identity = hashlib.sha256((key + "\0" + request.url).encode("utf-8")).hexdigest()
        root = resolved_path(self._staging_root or configured_cache_root() / "preparation")
        staging = root / identity
        if not resolved_path(staging).is_relative_to(root):
            raise ValueError("Preparation staging path escapes its configured cache root")
        return staging

    def transfer_status(self, request):
        """Nonblocking local observation; never acquire an artifact lease or fetch.

        HTTP resume eligibility comes from the same exact receipt validator used
        by transfer. It remains conditional on the server's next response. SDK
        partials stay unverified until the SDK negotiates their source identity.
        """
        key = self._key(request.destination)
        result = dict(
            state="absent", received_bytes=0, total_bytes=request.expected_size, resumable=False
        )
        with self._condition:
            job = self._jobs.get(key)
            if job is not None:
                if job.request != request:
                    return dict(
                        result, state="conflict", reason="Another request owns this destination."
                    )
                received, total, speed = job.progress
                return dict(
                    result,
                    state="cancelling" if job.cancelled.is_set() else "running",
                    received_bytes=received,
                    total_bytes=total or request.expected_size,
                    speed_bytes=speed,
                    resumable=None,
                )
        try:
            staging = self._staging_directory(request)
            part, receipt = staging / "hub-http.part", staging / "hub-http.json"
            if any(not resolved_path(path).is_relative_to(staging) for path in (part, receipt)):
                return dict(
                    result, state="unsafe", reason="Partial cache path is outside its owner."
                )
            if part.is_file():
                from ruinedfooocus.core.hub_download import _validated_resume_offset

                identity = {
                    "url": request.url,
                    "etag": f'"{request.sha256}"',
                    "size": request.expected_size,
                }
                offset = (
                    _validated_resume_offset(part, receipt, identity)
                    if request.sha256 is not None and request.expected_size is not None
                    else None
                )
                return dict(
                    result,
                    state="partial" if offset else "stale_partial",
                    received_bytes=part.stat().st_size,
                    resumable=bool(offset),
                    reason="Resume rechecks the publisher response; partial bytes are not verified."
                    if offset
                    else "Partial identity differs or is unknown; retry lets the transport validate it.",
                )
            # Bounded traversal; do not ask the SDK for paths (that API creates
            # directories) or follow junctions/symlinks outside this owner.
            pending, inspected, count, size = [staging / ".cache/huggingface/download"], 0, 0, 0
            while pending and inspected < 128:
                directory = pending.pop()
                if not resolved_path(directory).is_relative_to(staging):
                    return dict(
                        result, state="unsafe", reason="Partial cache path is outside its owner."
                    )
                if not directory.is_dir():
                    continue
                with os.scandir(directory) as entries:
                    for entry in entries:
                        inspected += 1
                        if inspected > 128:
                            break
                        path = Path(entry.path)
                        if not resolved_path(path).is_relative_to(staging):
                            return dict(
                                result,
                                state="unsafe",
                                reason="Partial cache path is outside its owner.",
                            )
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(path)
                        elif entry.is_file(follow_symlinks=False) and path.suffix == ".incomplete":
                            count += 1
                            size += entry.stat(follow_symlinks=False).st_size
            if count:
                return dict(
                    result,
                    state="partial_unverified",
                    received_bytes=size,
                    resumable=None,
                    reason="SDK partial cache exists; the SDK must validate identity and resume support.",
                )
        except (OSError, ValueError):
            return dict(
                result, state="unknown", reason="Partial cache could not be inspected safely."
            )
        return result

    def submit(self, request, progress=None):
        key = self._key(request.destination)
        with self._condition:
            if self._closed:
                raise RuntimeError("Preparation service is closed")
            job = self._jobs.get(key)
            if job is not None:
                if job.request != request or job.cancelled.is_set():
                    raise PreparationConflict(f"Destination is already being prepared: {key}")
            else:
                if len(self._jobs) >= self._max_jobs:
                    raise PreparationBusy("Preparation queue is full")
                job = _Job(request)
                self._jobs[key] = job
            subscriber = PreparationSubscription(self, job, progress)
            job.subscribers[id(subscriber)] = subscriber
            if len(job.subscribers) == 1:
                try:
                    job.future = self._executor.submit(self._run, key, job)
                except BaseException:
                    self._jobs.pop(key, None)
                    job.subscribers.pop(id(subscriber), None)
                    subscriber._future.cancel()
                    raise
            return subscriber

    @contextmanager
    def artifact_lease(self, destination, *, write=False, cancelled=None, blocking=True):
        """Read leases protect loaded files; write leases cover import/delete/replace.

        Acquire outside compute admission; do not submit/wait for preparation while
        holding a lease on that destination. Cancellation is cooperative.
        """
        destination = Path(destination).resolve()
        key = self._key(destination)
        acquired = False
        with self._condition:
            if not blocking and write and key in self._jobs:
                raise PreparationBusy("Artifact has a pending preparation; retry after it finishes")
            artifact = self._artifacts.setdefault(key, _Artifact())
            artifact.users += 1
            if write:
                artifact.waiting_writers += 1
            self._condition.notify_all()
            try:
                while artifact.writer or (artifact.readers if write else artifact.waiting_writers):
                    if not blocking:
                        raise PreparationBusy(
                            "Artifact is in use; retry after its operation finishes"
                        )
                    _check_cancelled(cancelled)
                    self._condition.wait(0.1 if cancelled is not None else None)
                _check_cancelled(cancelled)
                if write:
                    artifact.writer = True
                else:
                    artifact.readers += 1
                acquired = True
            except BaseException:
                artifact.users -= 1
                if not artifact.users:
                    self._artifacts.pop(key)
                raise
            finally:
                if write:
                    artifact.waiting_writers -= 1
                self._condition.notify_all()
        try:
            yield destination
        finally:
            with self._condition:
                if acquired and write:
                    artifact.writer = False
                elif acquired:
                    artifact.readers -= 1
                artifact.users -= 1
                if not artifact.users:
                    self._artifacts.pop(key)
                self._condition.notify_all()

    def _report(self, job, received, total, speed):
        _check_cancelled(job.cancelled)
        with self._condition:
            job.progress = (received, total, speed)
            subscribers = list(job.subscribers.values())
        for subscriber in subscribers:
            subscriber._report(received, total, speed)
        _check_cancelled(job.cancelled)

    def _run(self, key, job):
        result = error = None
        try:
            request = job.request
            destination = request.destination
            with self.artifact_lease(destination, write=True, cancelled=job.cancelled):
                if destination.exists():
                    verify_file(destination, request, job.cancelled)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    # SDK-complete filenames must never appear under model roots.
                    staging = self._staging_directory(request)
                    staging.mkdir(parents=True, exist_ok=True)
                    prepared = Path(
                        self._transport(
                            request,
                            staging,
                            lambda *values: self._report(job, *values),
                            job.cancelled,
                        )
                    ).resolve()
                    if not prepared.is_relative_to(staging.resolve()):
                        raise ValueError(
                            "Transport returned an artifact outside its staging directory"
                        )
                    try:
                        verify_file(prepared, request, job.cancelled)
                        self._publish(prepared, request, job.cancelled)
                    finally:
                        # Complete transfers have separate paths from resumable partials.
                        # Remove rejected and published staged files.
                        prepared.unlink(missing_ok=True)
                if request.sha256 is not None:
                    job.verified_signature = MappingProxyType(file_identity(destination))
                result = destination
                size = destination.stat().st_size
                self._report(job, size, size, 0)
        except BaseException as caught:
            job.verified_signature = None
            error = caught
        with self._condition:
            if self._jobs.get(key) is job:
                self._jobs.pop(key)
            subscribers = list(job.subscribers.values())
        for subscriber in subscribers:
            with subscriber._callback_lock:
                if not subscriber._future.done():
                    if error is None:
                        subscriber._future.set_result(result)
                    else:
                        subscriber._future.set_exception(error)

    def _publish(self, prepared, request, cancelled):
        destination = request.destination

        def publish(source):
            # Cancellation and final publication have one linearization point.
            with self._condition:
                _check_cancelled(cancelled)
                _publish_no_clobber(source, destination)

        try:
            publish(prepared)
        except OSError as error:
            if error.errno != errno.EXDEV and getattr(error, "winerror", None) != 17:
                raise
            # External cache may be on another volume. Only a non-loadable .part
            # exists in the model root until the copied artifact is fully verified.
            temporary = None
            try:
                expected_size = prepared.stat().st_size
                with (
                    prepared.open("rb") as source,
                    NamedTemporaryFile(
                        dir=destination.parent, suffix=".part", delete=False
                    ) as output,
                ):
                    temporary = Path(output.name)
                    copied = 0
                    while chunk := source.read(1024 * 1024):
                        _check_cancelled(cancelled)
                        copied += output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if copied != expected_size:
                    raise IOError("Prepared artifact changed while copying for publication")
                verify_file(temporary, request, cancelled)
                publish(temporary)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def close(self, *, wait=True):
        """Stop accepting jobs and cancel subscriptions; retain Hub resume state."""
        with self._condition:
            self._closed = True
            subscribers = [sub for job in self._jobs.values() for sub in job.subscribers.values()]
        for subscriber in subscribers:
            subscriber.cancel()
        self._executor.shutdown(wait=wait)


def _check_cancelled(event):
    if event is not None and event.is_set():
        raise CancelledError()


def _publish_no_clobber(source, destination):
    if os.name == "nt":
        # Windows rename is atomic and fails if destination exists.
        os.rename(source, destination)
    else:
        os.link(source, destination)


def verify_file(path, request, cancelled=None):
    _check_cancelled(cancelled)
    size = path.stat().st_size
    if not path.is_file() or size <= 0:
        raise IOError("Prepared artifact must be a nonempty regular file")
    if request.expected_size is not None and size != request.expected_size:
        raise IOError("Prepared artifact size does not match expected size")
    if request.sha256 is not None:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                _check_cancelled(cancelled)
                digest.update(chunk)
        if digest.hexdigest() != request.sha256:
            raise IOError("Prepared artifact SHA256 does not match")
    _check_cancelled(cancelled)


def transfer_file(request, staging, progress, cancelled):
    """Transport contract: return a complete staged file, never write the final path."""
    if urlparse(request.url).hostname == "huggingface.co":
        from ruinedfooocus.core.hub_download import _transfer_hub_file

        return _transfer_hub_file(
            request.url,
            staging,
            progress,
            cancelled,
            expected_size=request.expected_size,
            sha256=request.sha256,
        )

    import requests

    temporary = staging / "artifact.part"
    try:
        with requests.get(request.url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            received = 0
            started = last_update = time.monotonic()
            progress(0, total, 0)
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    _check_cancelled(cancelled)
                    received += output.write(chunk)
                    now = time.monotonic()
                    if now - last_update >= 0.5:
                        progress(received, total, received / max(now - started, 0.001))
                        last_update = now
                output.flush()
                os.fsync(output.fileno())
            if total and received != total:
                raise IOError("Incomplete model download; retry to download again")
        _check_cancelled(cancelled)
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


# No threads are started until the first submission.
preparation_service = PreparationService()
