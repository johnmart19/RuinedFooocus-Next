"""Bounded memory-only API job facts; request/result contents are never accepted."""

from collections import Counter, OrderedDict
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from uuid import uuid4


_KINDS = {"chat", "image", "upscale", "rembg", "caption", "depth", "sdcpp"}
_TERMINAL = {"completed", "failed", "cancelled"}


@dataclass
class _Entry:
    kind: str
    created: float
    started: float | None = None
    ended: float | None = None
    state: str = "waiting"


class ApiActivity:
    def __init__(self, capacity=128, retention_seconds=3600, clock=monotonic):
        if not 1 <= capacity <= 128 or not 0 < retention_seconds <= 3600:
            raise ValueError("Activity retention must fit the bounded local history")
        self.capacity = capacity
        self.retention_seconds = retention_seconds
        self.clock = clock
        self.lock = RLock()
        self.entries = OrderedDict()

    def _prune(self, now):
        for token, entry in list(self.entries.items()):
            if (
                now - (entry.ended if entry.ended is not None else entry.created)
                > self.retention_seconds
            ):
                self.entries.pop(token)

    def begin(self, kind):
        kind = kind if isinstance(kind, str) and kind in _KINDS else "other"
        with self.lock:
            now = self.clock()
            self._prune(now)
            while len(self.entries) >= self.capacity:
                self.entries.popitem(last=False)
            token = uuid4().hex
            self.entries[token] = _Entry(kind, now)
            return token

    def started(self, token):
        with self.lock:
            entry = self.entries.get(token)
            if entry is not None and entry.state == "waiting":
                entry.started = self.clock()
                entry.state = "running"

    def finished(self, token, state):
        if state not in _TERMINAL:
            raise ValueError("Invalid terminal activity state")
        with self.lock:
            entry = self.entries.get(token)
            if entry is not None and entry.state not in _TERMINAL:
                entry.ended = self.clock()
                entry.state = state

    def snapshot(self):
        with self.lock:
            now = self.clock()
            self._prune(now)
            counts = dict(Counter(entry.state for entry in self.entries.values()))
            rows = []
            for entry in list(self.entries.values())[-20:][::-1]:
                ended = entry.ended if entry.ended is not None else now
                started = entry.started if entry.started is not None else ended
                rows.append(
                    {
                        "kind": entry.kind,
                        "state": entry.state,
                        "wait_ms": round(max(0, started - entry.created) * 1000, 1),
                        "run_ms": round(max(0, ended - started) * 1000, 1),
                        "total_ms": round(max(0, ended - entry.created) * 1000, 1),
                    }
                )
            return {"retained": len(self.entries), "counts": counts, "rows": rows}


activity = ApiActivity()
