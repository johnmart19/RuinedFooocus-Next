"""Release external GPU allocations before the serial worker changes tasks."""

import atexit
from contextlib import contextmanager
from functools import wraps
import threading
from weakref import WeakSet

_resources = WeakSet()
_active = {}
_lock = threading.RLock()


def register(resource):
    with _lock:
        _resources.add(resource)


def using(function):
    """Fence an actual session call against bulk disposal, even during failure."""

    @wraps(function)
    def run(resource, *args, **kwargs):
        with _lock:
            _active[resource] = _active.get(resource, 0) + 1
        try:
            return function(resource, *args, **kwargs)
        finally:
            with _lock:
                count = _active[resource] - 1
                if count:
                    _active[resource] = count
                else:
                    _active.pop(resource)

    return run


def release_except(group=None, *, shutdown=False):
    with _lock:
        selected = [
            resource for resource in _resources if group is None or resource.resource_group != group
        ]
        if any(resource in _active for resource in selected) and not shutdown:
            raise RuntimeError(
                "An auxiliary model is in use; wait for its owning job before changing compute service"
            )
        for resource in selected:
            if resource not in _active:
                resource.close()


atexit.register(lambda: release_except(shutdown=True))
