"""Explicit caption-to-prompt action using the existing serial caption service."""

import hashlib
import json
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps

from ruinedfooocus.core.paths import asset_path
from ruinedfooocus.core.translation_manager import translate_ui
from ruinedfooocus.features.prompting.prompt_preview import undo_preview
from ruinedfooocus.features.prompting.ui_draft_fence import fence_event


def describe_image(current, image, native_source=None):
    from ruinedfooocus.app.state import settings
    from ruinedfooocus.features.gallery.actions import source_path, source_record
    from ruinedfooocus.features.gallery.editing_workflow import working_source
    from ruinedfooocus.features.image_processing.interrogate import looks, queued_caption

    if not isinstance(image, (str, Path, Image.Image)):
        raise gr.Error("Choose or upload an image to describe.")
    record = None
    if isinstance(image, Image.Image):
        from ruinedfooocus.features.image_processing.native_image_formats import MAX_PIXELS

        if getattr(image, "n_frames", 1) != 1 or not 0 < image.width * image.height <= MAX_PIXELS:
            raise gr.Error("Choose a single still image to describe.")
        pixels = ImageOps.exif_transpose(image).copy()
    else:
        path = Path(image)
        placeholder = asset_path("init_image.png")
        if path.resolve() == placeholder.resolve() or (
            path.is_file()
            and path.stat().st_size == placeholder.stat().st_size
            and hashlib.sha256(path.read_bytes()).digest()
            == hashlib.sha256(placeholder.read_bytes()).digest()
        ):
            raise gr.Error("Choose or upload an image to describe.")
    if native_source is not None and not isinstance(image, Image.Image):
        from ruinedfooocus.features.image_processing.native_image_outputs import (
            capture_original_source,
        )

        captured = capture_original_source(dict(main_view=image, _native_main_source=native_source))
        if isinstance(captured.get("main_view"), (str, Path)):
            raise gr.Error("The original image changed. Select it again.")
        record, pixels = native_source, captured["main_view"].copy()
    elif not isinstance(image, Image.Image):
        record = source_record(path)
        pixels, _ = working_source(record)
    kind = settings.default_settings.get("interrogator", "florence")
    if kind not in looks:
        raise gr.Error("Choose an image description engine in Settings.")
    description = queued_caption(pixels, kind, gr)
    if record is not None:
        source_path(record)
    if not isinstance(description, str) or not description.strip():
        raise gr.Error("The image description was empty. Your prompt is unchanged.")
    return description, (current, description), gr.update(interactive=True)


def add_describe_controls(
    prompt, image_source, native_source=None, *, input_source=None, source_kind=None
):
    previous = gr.JSON(None, visible="hidden")
    with gr.Row(elem_classes=["rf-prompt-actions"], equal_height=True):
        describe = gr.Button(translate_ui("Describe image"))
        undo = gr.Button(translate_ui("Undo description"), size="sm", interactive=False)
    gr.Markdown(translate_ui("Uses the image description engine selected in Settings."))
    inputs = [prompt, image_source] + ([native_source] if native_source is not None else [])
    inputs += [input_source] if input_source is not None else []
    inputs += [source_kind] if source_kind is not None else []

    def describe_current(current, image, *context):
        values = iter(context)
        native = next(values) if native_source is not None else None
        uploaded = next(values) if input_source is not None else None
        selected = next(values) if source_kind is not None else None
        if selected == "input":
            image, native = uploaded, None
        return describe_image(current, image, native)

    fence_event(
        describe.click,
        describe_current,
        prompt=prompt,
        inputs=inputs,
        outputs=[prompt, previous, undo],
        history_output=1,
        image_contexts=(image_source,) + ((input_source,) if input_source is not None else ()),
        discard_context_stale=True,
        api_visibility="undocumented",
        concurrency_id="image_description",
        concurrency_limit=1,
        show_progress_on=describe,
    )
    if source_kind is not None:
        source_kind.change(
            fn=None,
            inputs=[],
            outputs=[],
            js=f"() => {{ const node = document.getElementById({json.dumps(image_source.elem_id)}); if (node?.rfPromptContext) node.rfPromptContext.revision++; return []; }}",
            queue=False,
            api_visibility="private",
        )

    def undo_description(current, saved):
        try:
            return undo_preview(current, saved), None, gr.update(interactive=False)
        except ValueError as error:
            raise gr.Error(str(error)) from error

    fence_event(
        undo.click,
        undo_description,
        prompt=prompt,
        inputs=[prompt, previous],
        outputs=[prompt, previous, undo],
        history_input=1,
        api_visibility="undocumented",
        queue=False,
        show_progress="hidden",
    )
    return describe, undo
