"""Run consecutive listeners of one component event as a single request.

Gradio's client runs every listener of an event strictly one after another:
gather inputs, one HTTP round trip, apply outputs, re-render, next. A selection
that thirty feature panels observe therefore takes thirty round trips before
its own options appear. Each panel registers its listener on its own, which is
the right ownership, so the listeners are combined here, after the whole
interface is built, instead of in the features.

Order is never changed. Listeners are taken in dispatch order and grouped into
batches of neighbours; a listener starts a new batch when it reads or writes a
component that an earlier listener of the current batch writes, because it must
then see that result first. A listener with client-side JavaScript, a chained
`.then`, a progress tracker, cancellation, batching or streaming, or a generator,
stays on its own and also ends the batch. Each batch of two or more becomes one
request in the place of its first member.
"""

import inspect

import gradio as gr

MINIMUM = 2


def _plain(function, chained):
    """A callback the combined request can call exactly as Gradio would."""
    target = inspect.unwrap(function.fn) if function.fn is not None else None
    return bool(
        target is not None
        and function.js is None
        and function.preprocess
        and function.postprocess
        and not function.batch
        and not function.cancels
        and not function.is_cancel_function
        and function.trigger_after is None
        and function._id not in chained
        and not function.tracks_progress
        and not function.collects_event_data
        and not function.component_prop_inputs
        and function.connection == "sse"
        and function.time_limit is None
        and function.renderable is None
        and function.rendered_in is None
        and function.validator is None
        and not function.is_validator_function
        and function.key is None
        and not inspect.isgeneratorfunction(target)
        and not inspect.iscoroutinefunction(target)
        and not inspect.isasyncgenfunction(target)
    )


def _ids(components):
    return {component._id for component in components}


def batches(functions, key, chained):
    """Consecutive listeners of `key` that can share one request, in dispatch order."""
    groups, current, written = [], [], set()
    for function in functions:
        if key not in function.targets:
            continue
        if not _plain(function, chained):
            groups.append(current)
            current, written = [], set()
            continue
        reads, writes = _ids(function.inputs), _ids(function.outputs)
        if (reads | writes) & written:
            groups.append(current)
            current, written = [], set()
        current.append(function)
        written |= writes
    groups.append(current)
    return [group for group in groups if group]


def _results(function, result):
    """Map one callback's return value to its outputs, as Gradio does."""
    outputs = list(function.outputs)
    if isinstance(result, dict) and result and all(hasattr(item, "_id") for item in result):
        return {component._id: value for component, value in result.items()}
    if not outputs:
        return {}
    if len(outputs) == 1:
        return {outputs[0]._id: result}
    values = list(result)
    if len(values) != len(outputs):
        raise ValueError(
            f"{getattr(function.fn, '__name__', 'A callback')} returned {len(values)} values "
            f"for {len(outputs)} outputs."
        )
    return {component._id: value for component, value in zip(outputs, values)}


def combined(members):
    """One callback, its inputs and its outputs for `members`."""
    from gradio.helpers import special_args

    inputs = [component for function in members for component in function.inputs]
    outputs = [component for function in members for component in function.outputs]

    def run(request: gr.Request, *values):
        values, updates, start = list(values), {}, 0
        for function in members:
            count = len(function.inputs)
            arguments = values[start : start + count]
            start += count
            if function.inputs_as_dict:
                arguments = [dict(zip(function.inputs, arguments))]
            arguments = special_args(function.fn, arguments, request, None)[0]
            updates.update(_results(function, function.fn(*arguments)))
        return [updates.get(component._id, gr.skip()) for component in outputs]

    run.__name__ = (
        "combined_"
        + "_".join(
            dict.fromkeys(getattr(function.fn, "__name__", "listener") for function in members)
        )[:80]
    )
    return run, inputs, outputs


def coalesce(app, components, event="change", minimum=MINIMUM):
    """Combine neighbouring `event` listeners of each component in `app`.

    Returns {component id: [size of each combined batch]}. Call it after every
    listener is registered and before other passes wrap the callbacks.
    """
    report = {}
    for component in components:
        key = (component._id, event)
        functions = list(app.fns.values())
        chained = {function.trigger_after for function in functions} - {None}
        for members in batches(functions, key, chained):
            if len(members) < minimum:
                continue
            run, inputs, outputs = combined(members)
            first = members[0]._id
            with app:
                gr.on(
                    triggers=[getattr(component, event)],
                    fn=run,
                    inputs=inputs,
                    outputs=outputs,
                    queue=False,
                    show_progress="hidden",
                    # A newer selection must not be dropped while one is in flight.
                    trigger_mode="always_last",
                    api_visibility="private",
                )
            for function in members:
                function.targets = [target for target in function.targets if target != key]
            # Dispatch order is registration order: stand where the first member stood.
            items = list(app.fns.items())
            added = items.pop()
            position = next(index for index, (identity, _) in enumerate(items) if identity == first)
            items.insert(position, added)
            app.fns.clear()
            app.fns.update(items)
            report.setdefault(component._id, []).append(len(members))
    if report:
        app.config = app.get_config_file()
    return report
