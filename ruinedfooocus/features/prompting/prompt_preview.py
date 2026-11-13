"""Optimistic prompt edits shared by explicit preview tools."""

from functools import wraps
from dataclasses import dataclass
import json

import gradio as gr

from ruinedfooocus.features.prompting.ui_draft_fence import capture_draft, fence_event

from ruinedfooocus.core.translation_manager import translate_ui
from ruinedfooocus.features.prompting.prompt_batches import join_prompt_batch


def apply_preview(current, original, proposed, *, append=False):
    if current != original:
        raise ValueError("The prompt changed after this preview. Preview again before applying it.")
    if not proposed or not proposed.strip():
        raise ValueError("Preview a nonempty prompt first.")
    result = join_prompt_batch((current, proposed)) if append else proposed
    return result, (current, result)


def undo_preview(current, previous):
    if previous is None:
        raise ValueError("There is no prompt edit to undo.")
    if current != previous[1]:
        raise ValueError(
            "The prompt changed after applying the preview. Undo would overwrite your edits."
        )
    return previous[0]


def prompt_result(current, revision, ticket, *, explicit=False):
    """Decide against the latest draft, including edits that restore identical text."""
    if not ticket:
        raise ValueError("Generate a prompt first.")
    original, started_revision, proposed, append = ticket[:4]
    if not proposed or not proposed.strip():
        raise ValueError("The generated prompt is empty.")
    if not explicit and (current != original or revision != started_revision):
        return None
    return apply_preview(current, current, proposed, append=append)


@dataclass(frozen=True)
class ChatPromptAction:
    prompt: object
    ticket: object
    permissions: object
    context_inputs: tuple = ()

    def capture(self, inputs, operation):
        return capture_draft(self.prompt, inputs, operation, context_inputs=self.context_inputs)

    @property
    def invalidate_js(self):
        """Invalidate even an already validated envelope before cancellation queues."""
        from ruinedfooocus.core.paths import asset_path

        implementation = asset_path("prompt_draft_fence.js").read_text(encoding="utf-8")
        return f"(...values) => {{ {implementation}; fence.capture({json.dumps(self.prompt.elem_id)}, 'chat-invalidate', [null, {{}}], {{prompt: 0, history_input: null, contexts: [], revoke: true}}); return values; }}"

    def stage(self, result, permissions, guard):
        """Keep authority in private State; JSON carries only a bounded proposal."""
        from ruinedfooocus.features.chat.image_tool_requests import ChatToolPermissions

        if not isinstance(permissions, ChatToolPermissions) or not permissions.update_create_prompt:
            raise ValueError("Updating the Create prompt is Off for this turn.")
        permissions.check()
        proposal = result.get("prompt_proposal", {})
        if proposal.get("turn_id") != permissions.turn_id:
            raise ValueError("The prompt proposal belongs to a different Chat turn.")
        text = proposal.get("prompt")
        if not isinstance(text, str) or not text.strip() or len(text) > 12000 or "\x00" in text:
            raise ValueError("The prompt proposal is empty or exceeds its limit.")
        if (
            not isinstance(guard, dict)
            or not isinstance(guard.get("text"), str)
            or type(guard.get("revision")) is not int
            or type(guard.get("serial")) is not int
        ):
            raise ValueError("Capture the Create draft before requesting its prompt update.")
        return [guard["text"], guard["revision"], text, False, guard, permissions.turn_id]


def create_chat_prompt_action(prompt, *, owner_for_request, context_inputs=()):
    """Connect Chat proposals to the canonical draft fence and Apply/Undo flow."""
    from ruinedfooocus.features.chat.image_tool_requests import ChatToolPermissions

    ticket = gr.JSON(None, visible="hidden")
    permissions = gr.State(None)
    previous = gr.JSON(None, visible="hidden")
    with gr.Group(visible=False) as pending:
        proposed = gr.Textbox(label=translate_ui("Pending prompt"), lines=3, interactive=False)
        with gr.Row(elem_classes=["rf-prompt-actions"], equal_height=True):
            apply = gr.Button(translate_ui("Apply result"))
            dismiss = gr.Button(translate_ui("Dismiss result"))
    undo = gr.Button(translate_ui("Undo"), size="sm")
    outputs = [prompt, previous, proposed, pending, ticket]

    def validate(result, policy, request):
        if (
            not isinstance(policy, ChatToolPermissions)
            or policy.owner != owner_for_request(request)
            or not policy.update_create_prompt
        ):
            raise gr.Error("Updating the Create prompt is Off for this turn.")
        policy.check()
        if not isinstance(result, list) or len(result) != 6 or result[5] != policy.turn_id:
            raise gr.Error("The prompt proposal belongs to a different Chat turn.")

    def retain(current, result, policy, *_contexts):
        if result is None:
            return tuple(gr.skip() for _ in outputs)
        return gr.skip(), gr.skip(), result[2], gr.update(visible=True), gr.skip()

    def publish_chat_prompt(current, result, policy, *_contexts, request=None):
        if result is None:
            return tuple(gr.skip() for _ in outputs)
        validate(result, policy, request)
        applied = prompt_result(current, result[1], result)
        if applied is None:
            return retain(current, result, policy)
        return *applied, "", gr.update(visible=False), None

    def apply_chat_prompt(current, result, policy, *_contexts, request=None):
        validate(result, policy, request)
        applied = prompt_result(current, result[1], result, explicit=True)
        return *applied, "", gr.update(visible=False), None

    def undo_chat_prompt(current, saved):
        return undo_preview(current, saved), None

    def dismiss_chat_prompt(current):
        return gr.skip(), gr.skip(), "", gr.update(visible=False), None

    fence_event(
        ticket.change,
        publish_chat_prompt,
        prompt=prompt,
        inputs=[prompt, ticket, permissions, *context_inputs],
        outputs=outputs,
        on_stale=retain,
        history_output=1,
        guard_input=1,
        with_request=True,
        discard_context_stale=True,
        queue=False,
        api_visibility="undocumented",
    )
    fence_event(
        apply.click,
        apply_chat_prompt,
        prompt=prompt,
        inputs=[prompt, ticket, permissions, *context_inputs],
        outputs=outputs,
        history_output=1,
        with_request=True,
        discard_context_stale=True,
        revocation_input=1,
        queue=False,
        api_visibility="undocumented",
    )
    fence_event(
        undo.click,
        undo_chat_prompt,
        prompt=prompt,
        inputs=[prompt, previous],
        outputs=[prompt, previous],
        history_input=1,
        queue=False,
        api_visibility="undocumented",
    )
    fence_event(
        dismiss.click,
        dismiss_chat_prompt,
        prompt=prompt,
        inputs=[prompt],
        outputs=outputs,
        revoke=True,
        queue=False,
        api_visibility="undocumented",
    )
    return ChatPromptAction(prompt, ticket, permissions, tuple(context_inputs))


def create_prompt_action(prompt, *, advanced=None):
    """One prompt-only action with a pending result when the draft changed."""
    revision = gr.State(0)
    ticket = gr.JSON(None, visible="hidden")
    previous = gr.JSON(None, visible="hidden")
    with gr.Row(elem_classes=["rf-prompt-actions"], equal_height=True):
        generate = gr.Button(translate_ui("Generate prompt"), variant="primary")
        undo = gr.Button(translate_ui("Undo"))
    with gr.Group(visible=False) as pending:
        gr.Markdown(translate_ui("Your draft changed. Apply this result when ready."))
        proposed = gr.Textbox(label=translate_ui("Pending prompt"), lines=3, interactive=False)
        with gr.Row(elem_classes=["rf-prompt-actions"], equal_height=True):
            apply = gr.Button(translate_ui("Apply result"))
            dismiss = gr.Button(translate_ui("Dismiss result"))
    with gr.Accordion(translate_ui("Prompt options"), open=False):
        append = gr.Checkbox(label=translate_ui("Append as another prompt"), value=False)
        if advanced is not None:
            advanced()

    prompt.change(
        lambda value: value + 1,
        inputs=revision,
        outputs=revision,
        queue=False,
        api_visibility="undocumented",
    )

    def undo_prompt(current, saved):
        try:
            return undo_preview(current, saved), None
        except ValueError as error:
            raise gr.Error(str(error)) from error

    fence_event(
        undo.click,
        undo_prompt,
        prompt=prompt,
        inputs=[prompt, previous],
        outputs=[prompt, previous],
        history_input=1,
        queue=False,
        api_visibility="undocumented",
    )
    dismiss.click(
        lambda: (None, "", gr.update(visible=False)),
        outputs=[ticket, proposed, pending],
        queue=False,
        api_visibility="undocumented",
    )

    def bind(builder, inputs, *, clear_controls=()):
        def build_prompt(current, version, append_mode, *values):
            *arguments, guard = values
            result = builder(*arguments)
            if not result or not result.strip():
                raise gr.Error("The generated prompt is empty.")
            return [current, version, result, append_mode, guard]

        def publish_prompt(current, version, result):
            try:
                applied = prompt_result(current, version, result)
            except ValueError as error:
                raise gr.Error(str(error)) from error
            if applied is None:
                return (
                    gr.skip(),
                    gr.skip(),
                    result[2],
                    gr.update(visible=True),
                    gr.skip(),
                    *(gr.skip() for _ in clear_controls),
                )
            return *applied, "", gr.update(visible=False), None, *(False for _ in clear_controls)

        def apply_result(current, version, result):
            try:
                applied = prompt_result(current, version, result, explicit=True)
            except ValueError as error:
                raise gr.Error(str(error)) from error
            return *applied, "", gr.update(visible=False), None, *(False for _ in clear_controls)

        generation_inputs = [prompt, revision, append, *inputs]
        generation_guard, begin = capture_draft(
            prompt, generation_inputs, f"prompt-build-{ticket._id}"
        )
        event = generate.click(
            build_prompt,
            inputs=[*generation_inputs, generation_guard],
            outputs=ticket,
            js=begin,
            api_visibility="undocumented",
        )

        def pending_result(current, version, result):
            return (
                gr.skip(),
                gr.skip(),
                result[2],
                gr.update(visible=True),
                gr.skip(),
                *(gr.skip() for _ in clear_controls),
            )

        fence_event(
            event.success,
            publish_prompt,
            prompt=prompt,
            inputs=[prompt, revision, ticket],
            outputs=[prompt, previous, proposed, pending, ticket, *clear_controls],
            on_stale=pending_result,
            history_output=1,
            guard_input=2,
            queue=False,
            api_visibility="undocumented",
        )
        fence_event(
            apply.click,
            apply_result,
            prompt=prompt,
            inputs=[prompt, revision, ticket],
            outputs=[prompt, previous, proposed, pending, ticket, *clear_controls],
            history_output=1,
            queue=False,
            api_visibility="undocumented",
        )

    return bind


def bind_explicit_generation(button, prepare, inputs, outputs, run_event, *, clear_controls=()):
    """Publish reviewed text/settings before requesting generation on success."""
    callback = prepare
    if clear_controls:

        @wraps(prepare)
        def callback(*values):
            return (*prepare(*values), *(False for _ in clear_controls))

    def publish_reviewed(*values):
        *arguments, event = values
        return (*callback(*arguments), event + 1)

    publish_reviewed.__name__ = prepare.__name__
    return fence_event(
        button.click,
        publish_reviewed,
        prompt=inputs[0],
        inputs=[*inputs, run_event],
        outputs=[*outputs, *clear_controls, run_event],
        history_output=len(outputs) - 1,
        api_visibility="undocumented",
    )
