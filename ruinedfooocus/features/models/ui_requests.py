"""Bounded latest-result model projections; native model state is never admitted here."""

from concurrent.futures import CancelledError
import json

import gradio as gr
from ruinedfooocus.core.translation_manager import localize, resolve_locale

from ruinedfooocus.app.preparation_admission import preparation_admission
from ruinedfooocus.app.resource_lifecycle import current_owner, latest_ui_requests, ui_request_owner


def version_js(channel):
    return """(...values) => {
        const channel = %s;
        const versions = (window.__rfModelUIRequests ||= {});
        values[values.length - 1] = versions[channel] = Math.max(versions[channel] || 0, Number(values[values.length - 1]) || 0) + 1;
        return values;
    }""" % json.dumps(channel)


def stamp(payload, channel, version):
    return {**payload, "requestIdentity": {"channel": channel, "version": version}}


def project(channel, version, request, build, *, outputs, source=None, prepare=True, validate=None):
    """Pure construction callers have no browser context; events always have one.

    Browser-triggered calls supply a monotonic version. Localization/read observers
    reuse the current ticket and cannot reactivate an older source.
    """
    if request is None and current_owner() is None:
        return build(None)
    owner = ui_request_owner(request)
    ticket = None
    try:
        if version is None:
            ticket = latest_ui_requests.peek(owner, channel)
            if source is not None and ticket.source != source:
                raise CancelledError("The model selection changed")
        else:
            ticket = latest_ui_requests.begin(owner, channel, int(version))
            ticket.source = source
        if not latest_ui_requests.current(ticket):
            raise CancelledError("A newer model request replaced this result")
        if validate is not None:
            validate()
        if prepare:
            try:
                future = preparation_admission.submit_work(
                    lambda: build(ticket), cancelled=ticket.cancelled
                )
            except RuntimeError as error:
                raise gr.Error(str(error)) from None
            if version is not None:
                latest_ui_requests.attach(ticket, future)
            result = future.result()
        else:
            result = build(ticket)

        def publish():
            if validate is not None:
                validate()
            return result

        return latest_ui_requests.publish(ticket, publish)
    except CancelledError:
        if ticket is not None and version is not None:
            ticket.cancelled.set()
        return tuple(gr.skip() for _ in range(outputs))
    except Exception:
        if ticket is not None and not latest_ui_requests.current(ticket):
            return tuple(gr.skip() for _ in range(outputs))
        raise
    finally:
        if ticket is not None and version is not None:
            latest_ui_requests.finish(ticket)


def current_source(channel, request=None, default=None):
    if request is None and current_owner() is None:
        return default
    try:
        return latest_ui_requests.peek(ui_request_owner(request), channel).source
    except CancelledError:
        return default


def current_identity(channel, request=None):
    if request is None and current_owner() is None:
        return None
    try:
        ticket = latest_ui_requests.peek(ui_request_owner(request), channel)
        return {"channel": channel, "version": ticket.version}
    except CancelledError:
        return None


def validate_view(rows, request=None, channel="acquisition-view", initial_version=None):
    if request is None and current_owner() is None:
        return
    owner = ui_request_owner(request)
    try:
        view = latest_ui_requests.peek(owner, channel)
    except CancelledError:
        if initial_version is None or not rows or rows[0].get("_view_version") != initial_version:
            raise
        # The local checkpoint browser creates this immutable initial State on
        # the server. A greater/cancelled existing ticket cannot be revived.
        view = latest_ui_requests.begin(owner, channel, initial_version)
        latest_ui_requests.finish(view)
        if not latest_ui_requests.current(view):
            raise CancelledError("The model list changed")
    if not rows or rows[0].get("_view_version") != view.version:
        raise CancelledError("The model list changed")


def validate_source(key, rows, request=None):
    if request is None and current_owner() is None:
        return
    owner = ui_request_owner(request)
    selection = latest_ui_requests.peek(owner, "acquisition-selection")
    view = latest_ui_requests.peek(owner, "acquisition-view")
    row = next((row for row in rows if row["key"] == key), None)
    if selection.source != key or row is None or row.get("_view_version") != view.version:
        raise CancelledError()


def validate_selection(key, rows, request=None):
    if key is None:
        if current_source("acquisition-selection", request) is not None:
            raise CancelledError("The model selection changed")
    else:
        validate_source(key, rows, request)


def require_source(key, rows, request=None):
    try:
        validate_source(key, rows, request)
    except CancelledError:
        raise gr.Error(
            localize(
                "The model selection changed. Select the file again.",
                resolve_locale(request=request),
            )
        ) from None
