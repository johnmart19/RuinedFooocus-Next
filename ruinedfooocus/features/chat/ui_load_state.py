"""Pure presentation of captured native load identities and progress."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from html import escape
from threading import Event
from uuid import uuid4

from ruinedfooocus.core.translation_manager import localize


def same_initialized_instance(accepted, resident):
    """A path/configuration match cannot establish the accepted native instance."""
    if not accepted or not resident or not accepted.get("load_id"):
        return False
    return all(
        accepted.get(key) == resident.get(key)
        for key in (
            "load_id",
            "selection_key",
            "model_identity",
            "runtime_identity",
            "requested",
            "effective",
        )
    )


@dataclass
class LoadOperation:
    current_selection: object
    load_id: str = field(default_factory=lambda: uuid4().hex)
    cancelled: Event = field(default_factory=Event)
    task_id: object = None

    def current(self):
        return not self.cancelled.is_set() and self.current_selection()


@contextmanager
def load_operation(session, current_selection):
    operation = LoadOperation(current_selection)
    with session.lock:
        previous = getattr(session, "load_operation", None)
        if previous is not None:
            previous.cancelled.set()
        session.load_operation = operation
    try:
        yield operation
    finally:
        with session.lock:
            if getattr(session, "load_operation", None) is operation:
                session.load_operation = None


def progress_text(event, *, locale=None):
    """Unknown totals stay indeterminate; phase/identity come from the backend."""
    runtime = event.get("runtime") or {}
    name = " · ".join(
        str(runtime[key]) for key in ("tag", "backend", "package") if runtime.get(key)
    )
    message = (
        event.get("message") or str(event.get("phase", "Loading")).replace("_", " ").capitalize()
    )
    transfer = event.get("transfer") or event
    received, total = transfer.get("received"), transfer.get("total")
    percent = (
        min(100, max(0, received * 100 / total))
        if isinstance(received, (int, float)) and isinstance(total, (int, float)) and total > 0
        else None
    )
    value = f' value="{percent:.1f}"' if percent is not None else ""
    amount = (
        f" · {percent:.0f}%"
        if percent is not None
        else f" · {received / 1000000:.1f} MB"
        if isinstance(received, (int, float))
        else ""
    )
    return (
        escape(localize(str(message), locale)) + (" · " + escape(name) if name else "") + amount,
        f'<progress max="100"{value} aria-label="Model loading progress" style="width:100%"></progress>',
    )


def initialized_text(snapshot, *, locale=None):
    model = snapshot.get("model_identity") or {}
    runtime = snapshot.get("runtime_identity") or {}
    from pathlib import Path

    identity = " · ".join(
        str(value)
        for value in (
            Path(model.get("path") or "").name,
            model.get("quantization"),
            runtime.get("backend"),
            runtime.get("tag"),
        )
        if value
    )
    placement = snapshot.get("placement") or {}
    observed = placement.get("state", "unknown")
    detail = (
        localize("Placement unavailable", locale)
        if observed == "unknown"
        else localize("Observed placement: {placement}", locale, placement=observed)
    )
    fallback = snapshot.get("fallback_reason")
    return escape(identity) + "\n\n" + detail + (" · " + escape(str(fallback)) if fallback else "")
