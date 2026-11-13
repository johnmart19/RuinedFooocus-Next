import hashlib
import gradio as gr
import re
from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui
from ruinedfooocus.features.prompting.local_model import run_prompt
from ruinedfooocus.features.prompting.presets import rewrite_presets
from ruinedfooocus.features.prompting.ui_draft_fence import fence_event


RESOURCE = re.compile(
    r"<(?:lora|style|embedding):[^<>]+>|"
    r"\(embedding:[\w./\\-]+:[+-]?(?:\d+(?:\.\d*)?|\.\d+)\)|"
    r"\bembedding:[\w./\\-]+",
    re.I,
)
# Apostrophes inside words belong to prose, including inside single-quoted text.
REWRITE_LITERALS = re.compile(
    r'<[^<>]*>|__[^\n]*?__|"(?:\\.|[^"\\])*"|'
    r"“[^”]*”|«[^»]*»|(?<!\w)'(?:\\.|[^'\\]|(?<=\w)'(?=\w))*'(?!\w)"
)


def safe_rewrite(original, proposed):
    """Do not accept invented adapters or altered literal lettering from a model."""
    supplied = list(dict.fromkeys(match.group() for match in RESOURCE.finditer(original)))
    proposed = RESOURCE.sub(
        lambda match: match.group() if match.group() in supplied else "", proposed
    )
    missing = []
    for match in REWRITE_LITERALS.finditer(original):
        literal = match.group()
        if literal in proposed:
            continue
        if literal.startswith(("<", "__")):
            missing.append(literal)
        else:
            raise gr.Error("The rewrite changed quoted text. Your original prompt is preserved.")
    missing.extend(value for value in supplied if value not in proposed and value not in missing)
    result = " ".join([proposed.rstrip(), *missing]) if missing else proposed
    if not result.strip():
        raise gr.Error("The rewrite is empty. Your original prompt is preserved.")
    return result


def add_rewrite_controls(prompt):
    def suggest(system_file, current):
        if not current or not current.strip():
            raise gr.Error("Enter a prompt before rewriting it.")
        if not system_file:
            raise gr.Error("Choose a rewrite preset first.")
        result = safe_rewrite(current, run_prompt(current, system_file))
        return gr.update(value=result, visible=True), current, gr.update(interactive=True)

    def apply_rewrite(current, original, proposed):
        if current != original:
            raise gr.Error(
                "The prompt changed after this preview. Preview a new rewrite before applying it."
            )
        if not proposed or not proposed.strip():
            raise gr.Error("The rewrite is empty.")
        return (
            proposed,
            (current, proposed),
            gr.update(interactive=True),
            gr.update(interactive=False),
        )

    def undo_rewrite(current, previous):
        if previous is None:
            return gr.skip(), None, gr.update(interactive=False)
        if current != previous[1]:
            raise gr.Error(
                "The prompt changed after applying the rewrite. Undo would overwrite your edits."
            )
        return gr.update(value=previous[0]), None, gr.update(interactive=False)

    def refresh_presets(selected):
        choices = rewrite_presets()
        # Keep the selection unless its preset file was removed.
        return gr.update(
            choices=choices, value=selected if selected in [path for _, path in choices] else None
        )

    # Choose first, then act: the full-width preset keeps long names readable.
    with gr.Group():
        llama_select = gr.Dropdown(
            choices=rewrite_presets(),
            label=translate_ui("Rewrite preset"),
            show_label=True,
        )
        llama_btn = gr.Button(value=translate_ui("Preview rewrite"), size="sm")
    previous_prompt = gr.JSON(None, visible="hidden")
    original_prompt = gr.State(None)
    preview = gr.Textbox(label=translate_ui("Suggested prompt"), lines=4, visible=False)
    with gr.Row(equal_height=True, elem_classes=["rf-prompt-actions"]):
        apply_btn = gr.Button(value=translate_ui("Apply rewrite"), interactive=False)
        undo_btn = gr.Button(value=translate_ui("Undo rewrite"), size="sm", interactive=False)
    with gr.Accordion(translate_ui("Rewrite options"), open=False):
        refresh_btn = gr.Button(value=translate_ui("Refresh presets"), size="sm")
    gr.Markdown(
        translate_ui(
            "Uses the local model selected in **Chat**. Preview the suggestion, then apply it explicitly."
        )
    )
    llama_btn.click(
        suggest,
        api_visibility="undocumented",
        inputs=[llama_select, prompt],
        outputs=[preview, original_prompt, apply_btn],
    )
    fence_event(
        apply_btn.click,
        apply_rewrite,
        prompt=prompt,
        inputs=[prompt, original_prompt, preview],
        outputs=[prompt, previous_prompt, undo_btn, apply_btn],
        history_output=1,
        api_visibility="undocumented",
        queue=False,
    )
    fence_event(
        undo_btn.click,
        undo_rewrite,
        prompt=prompt,
        inputs=[prompt, previous_prompt],
        outputs=[prompt, previous_prompt, undo_btn],
        history_input=1,
        api_visibility="undocumented",
        queue=False,
    )
    refresh_btn.click(
        refresh_presets,
        inputs=llama_select,
        outputs=llama_select,
        api_visibility="undocumented",
        queue=False,
    )


def add_qwen_rewrite_controls(prompt, checkpoint):
    from ruinedfooocus.features.image_editing.ui_guidance import family_for_checkpoint
    from ruinedfooocus.features.prompting.prompt_preview import apply_preview, undo_preview
    from ruinedfooocus.features.prompting.qwen_image21_enhancement import (
        DEFAULT_TOKENS,
        MAX_TOKENS,
        reference_images,
        rewrite_prompt,
    )

    def available(name):
        return bool(name and family_for_checkpoint(name) == "QwenImage21")

    with gr.Accordion(
        translate_ui("Qwen Image 2.1 prompt enhancement"),
        open=False,
        visible=available(checkpoint.value),
        elem_id="qwen-image-prompt-enhancer",
    ) as panel:
        gr.Markdown(
            translate_ui(
                "Uses Qwen's official rewriter for text-to-image or Native image edit. "
                "The first preview may download the 9.47 GB model for that mode."
            )
        )
        max_tokens = gr.Slider(
            minimum=512,
            maximum=MAX_TOKENS,
            step=64,
            value=DEFAULT_TOKENS,
            label=translate_ui("Maximum rewrite tokens"),
        )
        preview_btn = gr.Button(translate_ui("Preview Qwen rewrite"))
        preview = gr.Textbox(
            label=translate_ui("Suggested prompt"), lines=5, interactive=False, visible=False
        )
        framing = gr.Textbox(
            label=translate_ui("Suggested aspect ratio"), interactive=False, visible=False
        )
        with gr.Row(equal_height=True, elem_classes=["rf-prompt-actions"]):
            apply_btn = gr.Button(translate_ui("Apply rewrite"), interactive=False)
            undo_btn = gr.Button(translate_ui("Undo rewrite"), interactive=False)

    ticket = gr.State(None)
    previous = gr.JSON(None, visible="hidden")
    checkpoint.change(
        lambda name: gr.update(visible=available(name)),
        inputs=checkpoint,
        outputs=panel,
        queue=False,
        api_visibility="undocumented",
    )

    def bind(primary, additional, tool_selection, tool_type, inpaint, *, clear_controls=()):
        from ruinedfooocus.features.image_editing.ui import tool_kind

        context = [checkpoint, tool_selection, tool_type, inpaint, primary, additional]

        def active_references(name, selected, kind, inpainting, image, extras):
            if not available(name):
                raise ValueError("Select a Qwen Image 2.1 checkpoint first.")
            if inpainting:
                raise ValueError("Preview a rewrite in Native image edit or text-to-image mode.")
            return reference_images(image, extras) if tool_kind(selected, kind) == "edit" else ()

        def source_identity(references):
            return tuple(
                (image.size, hashlib.sha256(image.tobytes()).digest()) for image in references
            )

        def preview_qwen(
            current,
            tokens,
            name,
            selected,
            kind,
            inpainting,
            image,
            extras,
            request: gr.Request = None,
        ):
            try:
                references = active_references(name, selected, kind, inpainting, image, extras)
                result = rewrite_prompt(current or "", references, max_tokens=int(tokens))
                suggested = safe_rewrite(current, result.prompt)
            except (ValueError, RuntimeError, OSError) as error:
                raise gr.Error(str(error)) from error
            ratio = result.ratio
            if result.follow_reference is not None:
                ratio = localize(
                    "Follow reference image {index}",
                    resolve_locale(request=request),
                    index=result.follow_reference,
                )
            return (
                gr.update(value=suggested, visible=True),
                gr.update(value=ratio, visible=True),
                (current, name, source_identity(references), suggested),
                gr.update(interactive=True),
            )

        def apply_qwen(current, saved, *values):
            try:
                references = active_references(*values)
                if not saved or saved[1:3] != (values[0], source_identity(references)):
                    raise ValueError("The edit context changed. Preview a new Qwen rewrite first.")
                updated, undo_value = apply_preview(current, saved[0], saved[3])
            except ValueError as error:
                raise gr.Error(str(error)) from error
            return (
                updated,
                undo_value,
                gr.update(interactive=True),
                gr.update(interactive=False),
                *(False for _ in clear_controls),
            )

        def undo_qwen(current, saved):
            try:
                return undo_preview(current, saved), None, gr.update(interactive=False)
            except ValueError as error:
                raise gr.Error(str(error)) from error

        preview_btn.click(
            preview_qwen,
            inputs=[prompt, max_tokens, *context],
            outputs=[preview, framing, ticket, apply_btn],
            api_visibility="undocumented",
        )
        fence_event(
            apply_btn.click,
            apply_qwen,
            prompt=prompt,
            inputs=[prompt, ticket, *context],
            outputs=[prompt, previous, undo_btn, apply_btn, *clear_controls],
            history_output=1,
            queue=False,
            api_visibility="undocumented",
        )
        fence_event(
            undo_btn.click,
            undo_qwen,
            prompt=prompt,
            inputs=[prompt, previous],
            outputs=[prompt, previous, undo_btn],
            history_input=1,
            queue=False,
            api_visibility="undocumented",
        )

    return bind
