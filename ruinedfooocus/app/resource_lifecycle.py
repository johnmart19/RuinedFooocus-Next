"""Job-local channels and conservative ownership for actual worker callers."""

from collections import OrderedDict, deque
from collections.abc import MutableMapping
from concurrent.futures import CancelledError
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import sys
import threading
import uuid

_CURRENT = ContextVar("rf_current_job", default=None)
_REQUEST_OWNER = ContextVar("rf_request_owner", default=None)


@dataclass
class UIRequest:
    owner: tuple
    channel: str
    version: int
    cancelled: threading.Event = field(default_factory=threading.Event)
    started: bool = False
    finished: bool = False
    preparation: object = None
    source: object = None


class LatestUIRequests:
    """Per-owner latest-input fencing, independent of the serial compute domain."""

    def __init__(self, capacity=256):
        self.capacity = capacity
        self._lock = threading.RLock()
        self._requests = OrderedDict()

    @staticmethod
    def _cancel(request):
        request.cancelled.set()
        if request.preparation is not None:
            request.preparation.cancel()

    def begin(self, owner, channel, version):
        if owner is None:
            raise ValueError("A UI request needs an authenticated session owner")
        if type(version) is not int or not 0 < version < 2**53:
            raise ValueError("Invalid UI request version")
        key = (owner, channel)
        with self._lock:
            previous = self._requests.get(key)
            request = UIRequest(owner, channel, version)
            if previous is not None and version <= previous.version:
                request.cancelled.set()
                return request
            if previous is None and len(self._requests) >= self.capacity:
                expired = next(
                    (key for key, value in self._requests.items() if value.finished), None
                )
                if expired is None:
                    raise RuntimeError("UI preparation queue is full; retry later")
                self._requests.pop(expired)
            if previous is not None:
                self._cancel(previous)
            self._requests[key] = request
            self._requests.move_to_end(key)
            return request

    def current(self, request):
        with self._lock:
            return (
                not request.cancelled.is_set()
                and self._requests.get((request.owner, request.channel)) is request
            )

    def get(self, owner, channel, version, *, start=False):
        with self._lock:
            request = self._requests.get((owner, channel))
            if request is None or request.version != version or not self.current(request):
                raise CancelledError("A newer UI request replaced this result")
            if start:
                if request.started:
                    raise CancelledError("This UI request is already being prepared")
                request.started = True
            return request

    def peek(self, owner, channel):
        with self._lock:
            request = self._requests.get((owner, channel))
            if request is None or not self.current(request):
                raise CancelledError("The UI selection changed")
            return request

    def attach(self, request, future):
        """Return a superseded waiting preparation's bounded slot promptly."""
        with self._lock:
            if not self.current(request):
                future.cancel()
                raise CancelledError("A newer UI request replaced this work")
            request.preparation = future
        return future

    def publish(self, request, function):
        """Check and publish/commit under the same lock as invalidation."""
        with self._lock:
            if not self.current(request):
                raise CancelledError("A newer UI request replaced this result")
            return function()

    def cancel(self, owner, channel):
        with self._lock:
            request = self._requests.get((owner, channel))
            if request is None:
                return False
            self._cancel(request)
            return True

    def cancel_owner(self, prefix):
        """Invalidate all tabs/channels for the server-authenticated login prefix."""
        with self._lock:
            matching = [
                request
                for request in self._requests.values()
                if request.owner[: len(prefix)] == prefix
            ]
            for request in matching:
                self._cancel(request)
                request.finished = True
            return len(matching)

    def finish(self, request):
        with self._lock:
            request.finished = True
            request.preparation = None


latest_ui_requests = LatestUIRequests()


def current_job():
    return _CURRENT.get()


def chat_operation(function):
    """Lazy worker boundary, safe while the pipeline registry is being imported."""

    @wraps(function)
    def run(*args, **kwargs):
        from ruinedfooocus.app.async_worker import chat_stage

        with chat_stage():
            return function(*args, **kwargs)

    return run


def current_owner():
    """Use authenticated UI request context, never an owner supplied in job JSON."""
    job = current_job()
    if job is not None:
        return job.owner
    module = sys.modules.get("gradio.context")
    context = getattr(module, "LocalContext", None)
    request = context.request.get(None) if context is not None else None
    if request is not None and getattr(request, "session_hash", None):
        from ruinedfooocus.integrations.web_access import authenticated_owner

        prefix = authenticated_owner(getattr(request, "request", None))
        return (prefix or ("ui", str(getattr(request, "username", None) or ""), "local")) + (
            str(request.session_hash),
        )
    return _REQUEST_OWNER.get()


def ui_request_owner(request=None):
    """Use real Gradio/server context; JSON values never choose another owner."""
    owner = current_owner()
    if owner is not None:
        return owner
    if request is None or not getattr(request, "session_hash", None):
        raise ValueError("A UI request needs a browser session")
    from ruinedfooocus.integrations.web_access import authenticated_owner

    prefix = authenticated_owner(getattr(request, "request", None))
    return (prefix or ("ui", str(getattr(request, "username", None) or ""), "local")) + (
        str(request.session_hash),
    )


@dataclass(frozen=True)
class JobIdentity:
    task_id: int
    token: str
    owner: tuple


@dataclass
class WorkItem:
    identity: JobIdentity
    cancelled: threading.Event
    state: dict = field(default_factory=dict)
    events: deque = field(default_factory=deque)
    terminal: bool = False
    started: bool = False
    finished: bool = False
    preparation: object = None
    external: object = None
    on_finish: object = None

    @property
    def task_id(self):
        return self.identity.task_id

    @property
    def owner(self):
        return self.identity.owner

    def check(self):
        if self.cancelled.is_set():
            raise CancelledError("This job was cancelled")


class JobRegistry:
    """Bounded submissions with private channels; terminal events release once."""

    def __init__(self, capacity=32, retained=128, channel_capacity=64):
        self.capacity, self.retained, self.channel_capacity = capacity, retained, channel_capacity
        self.condition = threading.Condition(threading.RLock())
        self.jobs = OrderedDict()
        self.ready = deque()
        self.sequence = 0

    def create(self, *, owner=None, cancelled=None):
        with self.condition:
            if sum(not item.finished for item in self.jobs.values()) >= self.capacity:
                raise RuntimeError("Compute queue is full; retry when a job finishes")
            self.sequence += 1
            identity = JobIdentity(
                self.sequence,
                uuid.uuid4().hex,
                owner or current_owner() or ("internal", uuid.uuid4().hex),
            )
            item = WorkItem(identity, cancelled if cancelled is not None else threading.Event())
            self.jobs[item.task_id] = item
            self._prune()
            return item

    def _prune(self):
        while len(self.jobs) > self.retained:
            key = next((key for key, item in self.jobs.items() if item.finished), None)
            if key is None:
                break
            self.jobs.pop(key)

    def get(self, task_id, *, owner=None, trusted=False):
        with self.condition:
            item = self.jobs.get(task_id)
            if item is None:
                raise ValueError("The requested job is no longer available")
            observed = current_owner() if owner is None else owner
            if (
                not trusted
                and observed != item.owner
                and not (observed is None and item.owner[0] == "internal")
            ):
                raise PermissionError("This job belongs to a different session")
            return item

    def enqueue(self, item, data):
        with self.condition:
            if item.finished:
                return False
            if item.cancelled.is_set():
                self.publish(
                    item.task_id, "results", {"error": "The operation was stopped. You can retry."}
                )
                self.finish(item)
                return False
            self.ready.append((item, data))
            self.condition.notify_all()
            return True

    def take(self, timeout=0.25):
        with self.condition:
            if not self.ready:
                self.condition.wait(timeout)
            if not self.ready:
                return None
            item, data = self.ready.popleft()
            item.started = True
            return item, data

    def publish(self, task_id, flag, product):
        with self.condition:
            item = self.jobs.get(task_id)
            if item is None or item.terminal:
                return False
            if flag in ("preview", "download", "reasoning"):
                item.events = deque(event for event in item.events if event[0] != flag)
            while len(item.events) >= self.channel_capacity:
                item.events.popleft()
            item.events.append((flag, product))
            if flag == "results":
                item.terminal = True
            self.condition.notify_all()
            return True

    def result(self, task_id, *, owner=None, timeout=0.25):
        item = self.get(task_id, owner=owner)
        with self.condition:
            if not item.events:
                self.condition.wait(timeout)
            if item.events:
                return item.events.popleft()
            if item.finished:
                raise RuntimeError("The job ended without another result")
            return None

    def cancel(self, task_id=None, *, owner=None):
        observed = current_owner() if owner is None else owner
        with self.condition:
            if task_id is None:
                matches = [
                    item
                    for item in self.jobs.values()
                    if item.owner == observed and not item.finished
                ]
                if observed is None or not matches:
                    return False
                active = [item for item in matches if item.started]
                item = (active or matches)[-1]
            else:
                item = self.get(task_id, owner=owner)
            item.cancelled.set()
            if item.preparation is not None:
                item.preparation.cancel()
            if not item.started:
                self.ready = deque(entry for entry in self.ready if entry[0] is not item)
                self.publish(
                    item.task_id, "results", {"error": "The operation was stopped. You can retry."}
                )
                self.finish(item)
            self.condition.notify_all()
            return True

    def cancel_owner(self, prefix):
        with self.condition:
            matching = [
                item
                for item in self.jobs.values()
                if not item.finished and item.owner[: len(prefix)] == prefix
            ]
        for item in matching:
            self.cancel(item.task_id, owner=item.owner)
        return len(matching)

    def finish(self, item):
        with self.condition:
            if item.finished:
                return False
            item.finished = True
            item.state.pop("preview_grid", None)
            if item.on_finish is not None:
                item.on_finish()
            item.preparation = None
            item.external = None
            self.condition.notify_all()
            self._prune()
            return True

    def fail_ready(self, message):
        """A worker death fails stranded admitted jobs; future submissions can restart."""
        with self.condition:
            stranded = list(self.ready)
            self.ready.clear()
        for item, _ in stranded:
            if item.external is not None:
                item.external.fail(message)
            self.publish(item.task_id, "results", {"error": message})
            self.finish(item)


class FairAdmission:
    """Fair per-resource exclusion, reentrant only for the same immutable job."""

    def __init__(self, capacity=32):
        self.capacity = capacity
        self.condition = threading.Condition(threading.RLock())
        self.waiters = deque()
        self.held = {}

    @contextmanager
    def lease(self, item, resources=("compute:shared",), *, on_wait=None):
        resources = frozenset(resources)
        ticket = (object(), item.identity.token, resources)
        acquired = False
        with self.condition:
            if len(self.waiters) >= self.capacity:
                raise RuntimeError("Resource admission queue is full")
            self.waiters.append(ticket)
            try:
                while True:
                    item.check()
                    earlier = list(self.waiters)[: list(self.waiters).index(ticket)]
                    owned = all(
                        key not in self.held or self.held[key][0] == item.identity.token
                        for key in resources
                    )
                    already_owned = {
                        key for key, value in self.held.items() if value[0] == item.identity.token
                    }
                    if already_owned and not owned:
                        raise RuntimeError(
                            "A compound stage cannot wait for another owner while holding resources"
                        )
                    fresh = {key for key in resources if key not in self.held}
                    fair = not any(
                        old[1] != item.identity.token
                        and old[2] & fresh
                        and not old[2] & already_owned
                        for old in earlier
                    )
                    if owned and fair:
                        self.waiters.remove(ticket)
                        for key in resources:
                            _, count = self.held.get(key, (item.identity.token, 0))
                            self.held[key] = (item.identity.token, count + 1)
                        acquired = True
                        break
                    if on_wait:
                        on_wait()
                    self.condition.wait(0.1)
            finally:
                if not acquired:
                    self.waiters.remove(ticket)
                    self.condition.notify_all()
        try:
            yield
        finally:
            with self.condition:
                for key in resources:
                    token, count = self.held[key]
                    if count == 1:
                        self.held.pop(key)
                    else:
                        self.held[key] = (token, count - 1)
                self.condition.notify_all()


registry = JobRegistry()
admission = FairAdmission()
_PIPELINES = {}
_RESIDENT = None
_SERVICE = ContextVar("rf_current_service", default=None)


@contextmanager
def job_scope(item):
    token = _CURRENT.set(item)
    try:
        yield item
    finally:
        _CURRENT.reset(token)


def service_pipeline(service):
    return _PIPELINES.get(service)


class OwnershipError(PermissionError):
    pass


def require_compute_owner():
    item = current_job()
    with admission.condition:
        held = admission.held.get("compute:shared")
        if item is None or held is None or held[0] != item.identity.token:
            raise OwnershipError(
                "This native operation requires an admitted owning job; no other service was released"
            )
    item.check()
    return item


def release_native_for_external():
    require_compute_owner()
    if _SERVICE.get() in ("image", "upscale", "caption"):
        _release_native()


def remember_pipeline(pipeline):
    service = _SERVICE.get()
    if service is not None:
        _PIPELINES[service] = pipeline


def _release_native():
    module = sys.modules.get("comfy.model_management")
    if module is not None:
        module.unload_all_models()
        module.soft_empty_cache()


def _release_service(service):
    selected = _PIPELINES.get(service)
    if service == "chat":
        if selected is not None:
            selected.unload()
    elif service in ("image", "upscale", "caption"):
        _release_native()
    elif service == "rembg":
        from ruinedfooocus.features.image_processing.auxiliary_resources import release_except

        release_except()


@contextmanager
def service_scope(service, state, *, item=None):
    """Actual callers share one conservative compute domain, including nested stages.

    Resource-specific admission is ready for separately qualified isolated lanes;
    production heavy compute remains serialized instead of assuming VRAM headroom.
    """
    global _RESIDENT
    item = item or current_job()
    if item is None:
        raise RuntimeError("Compute requires an owned job context")
    with admission.lease(item, ("compute:shared", "service:" + service)):
        previous_service = _SERVICE.get()
        previous_pipeline = state.get("pipeline")
        if _RESIDENT is not None and _RESIDENT != service:
            from ruinedfooocus.features.image_processing.auxiliary_resources import release_except

            release_except("rembg" if service == "rembg" else None)
            _release_service(_RESIDENT)
        _RESIDENT = service
        state["pipeline"] = _PIPELINES.get(service)
        token = _SERVICE.set(service)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            try:
                _release_service(service)
            finally:
                _PIPELINES.pop(service, None)
                _RESIDENT = None
            raise
        finally:
            selected = state.get("pipeline")
            if selected is not None and not failed:
                _PIPELINES[service] = selected
            _SERVICE.reset(token)
            if previous_service is not None:
                if previous_service != service and _RESIDENT == service:
                    from ruinedfooocus.features.image_processing.auxiliary_resources import (
                        release_except,
                    )

                    release_except("rembg" if previous_service == "rembg" else None)
                    _release_service(service)
                    _RESIDENT = previous_service
                state["pipeline"] = previous_pipeline


def cleanup_interrupt(item):
    """Only the owner of native cancellation may reset the native global flag."""
    if item.state.get("native_interrupt") and current_job() is not item:
        raise OwnershipError("Only the active job may clear its native interrupt state")
    if item.state.pop("native_interrupt", False):
        module = sys.modules.get("comfy.model_management")
        if module is not None:
            module.interrupt_current_processing(False)


class RuntimeState(MutableMapping):
    """Keep per-job preview/cancellation state out of the process-wide UI map."""

    SCOPED = frozenset(
        ("preview_grid", "preview_count", "preview_total", "interrupted", "last_image")
    )

    def __init__(self, initial):
        self.global_state = dict(initial)
        self.owners = OrderedDict()
        self._owners_lock = threading.RLock()

    def _mapping(self, key):
        if key not in self.SCOPED:
            return self.global_state
        job = current_job()
        owner = job.owner if job is not None else current_owner()
        if owner is None:
            return self.global_state
        with self._owners_lock:
            if job is not None:
                mapping = job.state
                self.owners[owner] = mapping
            else:
                mapping = self.owners.setdefault(owner, {})
            while len(self.owners) > 256:
                self.owners.popitem(last=False)
            return mapping

    def __getitem__(self, key):
        return self._mapping(key)[key]

    def __setitem__(self, key, value):
        self._mapping(key)[key] = value

    def __delitem__(self, key):
        del self._mapping(key)[key]

    def __iter__(self):
        return iter(set(self.global_state) | set(self._mapping("interrupted")))

    def __len__(self):
        return len(set(iter(self)))
