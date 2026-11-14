"""Publish prompt edits in the browser only while the validated draft is current."""

import json

import gradio as gr

from ruinedfooocus.core.paths import asset_path
from ruinedfooocus.core.translation_manager import localize, resolve_locale


def capture_draft(prompt, inputs, operation, *, context_inputs=()):
    """Capture the shared browser action serial before a slow builder starts."""
    if not prompt.elem_id:
        prompt.elem_id = f"rf-prompt-draft-{prompt._id}"
    guard = gr.JSON({}, visible="hidden")
    contexts = []
    for component in [*inputs, *context_inputs]:
        if component is prompt or isinstance(component, (gr.State, gr.JSON)):
            continue
        if not component.elem_id:
            component.elem_id = f"rf-prompt-context-{component._id}"
        if component.elem_id not in contexts:
            contexts.append(component.elem_id)
    config = json.dumps(dict(prompt=inputs.index(prompt), history_input=None, contexts=contexts))
    implementation = asset_path("prompt_draft_fence.js").read_text(encoding="utf-8")
    begin = f"(...values) => {{ {implementation}; return fence.capture({json.dumps(prompt.elem_id)}, {json.dumps(operation)}, values, {config}); }}"
    return guard, begin


def publish_draft(event, result, *, prompt, outputs):
    """Publish an already captured async result through the same browser fence."""
    status = gr.Markdown("", visible=False)
    implementation = asset_path("prompt_draft_fence.js").read_text(encoding="utf-8")
    config = json.dumps(dict(history_output=None))
    publish = f"(result) => {{ {implementation}; return fence.publish({json.dumps(prompt.elem_id)}, result, {len(outputs)}, {config}); }}"
    return event.success(
        fn=None,
        inputs=result,
        outputs=[*outputs, status],
        js=publish,
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )


def fence_event(
    method,
    callback,
    *,
    prompt,
    inputs,
    outputs,
    on_stale=None,
    history_output=None,
    history_input=None,
    history_revision=2,
    guard_input=None,
    guard_index=4,
    with_request=False,
    discard_context_stale=False,
    revocation_input=None,
    revoke=False,
    image_contexts=(),
    **options,
):
    """Keep server validation, then fence the actual client-side output update.

    Gradio queue ordering cannot protect a keystroke after server validation.
    A result envelope carries the browser's capture through validation; only the
    browser success handler may update the prompt and its related controls.
    """
    if not prompt.elem_id:
        prompt.elem_id = f"rf-prompt-draft-{prompt._id}"
    capture = gr.JSON({}, visible="hidden") if guard_input is None else None
    result = gr.JSON(None, visible="hidden")
    status = gr.Markdown("", visible=False)
    implementation = asset_path("prompt_draft_fence.js").read_text(encoding="utf-8")

    def validated(*values, request: gr.Request = None):
        if guard_input is None:
            *arguments, guard = values
        else:
            arguments = values
            ticket = arguments[guard_input]
            guard = ticket[guard_index] if ticket and len(ticket) > guard_index else None
        updates = callback(*arguments, request=request) if with_request else callback(*arguments)
        return {
            "updates": list(updates),
            "stale": list(on_stale(*arguments)) if on_stale else None,
            "guard": guard,
            "message": localize(
                "The prompt changed after this preview. Preview again before applying it.",
                resolve_locale(request=request),
            ),
        }

    validated.__name__ = callback.__name__
    operation = f"prompt-edit-{result._id}"
    contexts = []
    for component in inputs:
        if component is prompt or isinstance(component, (gr.State, gr.JSON)):
            continue
        if not component.elem_id:
            component.elem_id = f"rf-prompt-context-{component._id}"
        contexts.append(
            {"id": component.elem_id, "image": True}
            if component in image_contexts
            else component.elem_id
        )
    config = json.dumps(
        dict(
            prompt=inputs.index(prompt),
            history_input=history_input,
            history_output=history_output,
            history_revision=history_revision,
            discard_context_stale=discard_context_stale,
            revocation_input=revocation_input,
            revoke=revoke,
            contexts=contexts,
        )
    )
    begin = f"(...values) => {{ {implementation}; return fence.capture({json.dumps(prompt.elem_id)}, {json.dumps(operation)}, values, {config}); }}"
    publish = f"(result) => {{ {implementation}; return fence.publish({json.dumps(prompt.elem_id)}, result, {len(outputs)}, {config}); }}"
    event = method(
        validated,
        inputs=[*inputs, capture] if guard_input is None else inputs,
        outputs=result,
        js=begin if guard_input is None else None,
        **options,
    )
    return event.success(
        fn=None,
        inputs=result,
        outputs=[*outputs, status],
        js=publish,
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )
