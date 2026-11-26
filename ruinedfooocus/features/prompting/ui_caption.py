"""Image-to-Prompt studio and caption-to-prompt actions over the serial worker."""

import hashlib
import json
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps

from ruinedfooocus.core.paths import asset_path
from ruinedfooocus.core.translation_manager import translate_ui
from ruinedfooocus.features.prompting.image_to_prompt import (
    DEFAULT_STYLE,
    ENGINE_CHOICES,
    PROMPT_STYLES,
    installed_vision_models,
)
from ruinedfooocus.features.prompting.prompt_preview import undo_preview
from ruinedfooocus.features.prompting.ui_draft_fence import fence_event


def _vision_model_choices():
    choices = [("Auto (best installed)", "Auto")]
    try:
        for label, path in installed_vision_models():
            choices.append((label, path))
    except Exception:
        pass
    return choices


def _char_count_markdown(text):
    count = len(text or "")
    return f"*{count:,} characters*" if count else ""


def describe_image(
    current,
    image,
    native_source=None,
    *,
    engine=None,
    style=None,
    extra_instructions="",
    vision_model=None,
    detail="standard",
):
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
    kind = engine or settings.default_settings.get("interrogator", "florence")
    if kind not in looks:
        raise gr.Error("Choose an image description engine in Settings.")
    extra_kwargs = {}
    if style is not None:
        extra_kwargs["style"] = style
    if extra_instructions:
        extra_kwargs["extra_instructions"] = extra_instructions
    if vision_model and vision_model != "Auto":
        extra_kwargs["vision_model"] = vision_model
    if detail and detail != "standard":
        extra_kwargs["detail"] = detail
    description = queued_caption(pixels, kind, gr, **extra_kwargs)
    if record is not None:
        source_path(record)
    if not isinstance(description, str) or not description.strip():
        raise gr.Error("The image description was empty. Your prompt is unchanged.")
    return description, (current, description), gr.update(interactive=True)


def add_describe_controls(
    prompt,
    image_source,
    native_source=None,
    *,
    input_source=None,
    source_kind=None,
    allow_upload=False,
    run_event=None,
    embedded_source=False,
    id_prefix="describe",
):
    previous = gr.JSON(None, visible=False)
    describe_upload = None
    style_picker = None
    engine_picker = None
    vision_picker = None
    detail_picker = None
    extra_box = None
    result_box = None
    result_meta = None
    append_btn = None
    generate_btn = None

    if allow_upload:
        from ruinedfooocus.app.state import settings

        if not embedded_source:
            gr.Markdown(
                "### "
                + str(translate_ui("Image to Prompt"))
                + "\n"
                + str(
                    translate_ui(
                        "Upload an image or use the current canvas image to generate a structured prompt."
                    )
                )
            )
            describe_upload = gr.Image(
                label=translate_ui("Image to describe"),
                height=240,
                type="pil",
                image_mode="RGBA",
                format="png",
                elem_id=f"{id_prefix}-image-input",
                buttons=["download", "fullscreen"],
            )
        style_picker = gr.Radio(
            label=translate_ui("Prompt style"),
            choices=list(PROMPT_STYLES),
            value=DEFAULT_STYLE,
            elem_classes=["rf-chips"],
            elem_id=f"{id_prefix}-prompt-style",
        )
        default_engine = settings.default_settings.get("interrogator", "auto")
        if default_engine not in {value for _, value in ENGINE_CHOICES}:
            default_engine = "auto"
        with gr.Row():
            engine_picker = gr.Dropdown(
                label=translate_ui("Engine"),
                choices=list(ENGINE_CHOICES),
                value=default_engine,
                interactive=True,
                elem_id=f"{id_prefix}-engine",
            )
            vision_picker = gr.Dropdown(
                label=translate_ui("Vision model"),
                choices=_vision_model_choices(),
                value="Auto",
                interactive=True,
                visible=default_engine in ("auto", "vlm"),
                elem_id=f"{id_prefix}-vision-model",
            )
        with gr.Accordion(translate_ui("Prompt options"), open=False):
            extra_box = gr.Textbox(
                label=translate_ui("Extra instructions (optional)"),
                placeholder="e.g. Focus on outfit, pose, and lighting; ignore background",
                lines=1,
                elem_id=f"{id_prefix}-extra-instructions",
            )
            detail_picker = gr.Radio(
                label=translate_ui("Vision detail"),
                choices=[("Standard (1024 px)", "standard"), ("High detail (2048 px)", "high")],
                value="standard",
                elem_classes=["rf-chips"],
                elem_id=f"{id_prefix}-vision-detail",
            )
        with gr.Row(elem_classes=["rf-actions", "rf-actions-compact"], equal_height=True):
            describe = gr.Button(
                translate_ui("Describe image"),
                variant="primary",
                size="sm",
                elem_id=f"{id_prefix}-generate-prompt",
            )
            append_btn = gr.Button(
                translate_ui("Append to prompt"),
                size="sm",
                interactive=False,
                elem_id=f"{id_prefix}-append-prompt",
            )
            undo = gr.Button(translate_ui("Undo description"), size="sm", interactive=False)
        result_box = gr.Textbox(
            label=translate_ui("Generated prompt"),
            placeholder="Generated prompt will appear here. You can edit, copy, or send it straight to Generate.",
            lines=5,
            max_lines=10,
            buttons=["copy"],
            elem_id=f"{id_prefix}-result-prompt",
        )
        result_meta = gr.Markdown("", elem_id=f"{id_prefix}-result-meta")
        if run_event is not None:
            generate_btn = gr.Button(
                translate_ui("Generate image"),
                variant="primary",
                size="md",
                elem_id=f"{id_prefix}-generate-image",
            )
    else:
        with gr.Row(elem_classes=["rf-actions", "rf-actions-compact"], equal_height=True):
            describe = gr.Button(translate_ui("Describe image"), size="sm")
            undo = gr.Button(translate_ui("Undo description"), size="sm", interactive=False)

    inputs = [prompt, image_source] + ([native_source] if native_source is not None else [])
    inputs += [input_source] if input_source is not None else []
    inputs += [source_kind] if source_kind is not None else []
    if allow_upload:
        if describe_upload is not None:
            inputs += [describe_upload]
        inputs += [
            engine_picker,
            style_picker,
            extra_box,
            vision_picker,
            detail_picker,
        ]

    def describe_current(current, image, *context):
        values = iter(context)
        native = next(values) if native_source is not None else None
        uploaded = next(values) if input_source is not None else None
        selected = next(values) if source_kind is not None else None
        direct_upload = next(values) if describe_upload is not None else None
        engine = next(values) if allow_upload else None
        style = next(values) if allow_upload else None
        extra = next(values) if allow_upload else ""
        vision_model = next(values) if allow_upload else None
        detail = next(values) if allow_upload else "standard"
        if direct_upload is not None:
            image, native = direct_upload, None
        elif selected == "input":
            image, native = uploaded, None
        description, history, undo_update = describe_image(
            current,
            image,
            native,
            engine=engine,
            style=style,
            extra_instructions=extra,
            vision_model=vision_model,
            detail=detail,
        )
        if allow_upload:
            return (
                description,
                history,
                undo_update,
                description,
                _char_count_markdown(description),
                gr.update(interactive=True),
            )
        return description, history, undo_update

    image_contexts = (image_source,)
    if input_source is not None:
        image_contexts += (input_source,)
    if describe_upload is not None:
        image_contexts += (describe_upload,)

    outputs = [prompt, previous, undo]
    if allow_upload:
        outputs += [result_box, result_meta, append_btn]

    fence_event(
        describe.click,
        describe_current,
        prompt=prompt,
        inputs=inputs,
        outputs=outputs,
        history_output=1,
        image_contexts=image_contexts,
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

    if allow_upload:
        engine_picker.change(
            lambda engine: gr.update(
                visible=engine in ("auto", "vlm"),
                choices=_vision_model_choices(),
            ),
            inputs=[engine_picker],
            outputs=[vision_picker],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        result_box.change(
            lambda text: (_char_count_markdown(text), gr.update(interactive=bool((text or "").strip()))),
            inputs=[result_box],
            outputs=[result_meta, append_btn],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

        def append_generated_prompt(current, generated):
            addition = (generated or "").strip()
            if not addition:
                raise gr.Error("Generate a prompt first.")
            base = (current or "").strip()
            combined = f"{base}, {addition}" if base else addition
            return combined, (current, combined), gr.update(interactive=True)

        append_btn.click(
            append_generated_prompt,
            inputs=[prompt, result_box],
            outputs=[prompt, previous, undo],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

        if generate_btn is not None and run_event is not None:

            def apply_and_generate(current, generated, counter):
                text = (generated or "").strip() or (current or "").strip()
                if not text:
                    raise gr.Error("Generate or enter a prompt first.")
                return text, (counter or 0) + 1

            generate_btn.click(
                apply_and_generate,
                inputs=[prompt, result_box, run_event],
                outputs=[prompt, run_event],
                queue=False,
                show_progress="hidden",
                api_visibility="private",
            )

    return describe, undo
