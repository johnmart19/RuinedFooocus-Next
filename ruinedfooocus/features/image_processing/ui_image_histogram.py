"""Explicit image analysis beside the source, independent of generation controls."""

import asyncio
import queue

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui
from ruinedfooocus.features.image_processing import image_histogram

ANALYSIS_TIMEOUT = 1800
NON_HISTOGRAM_KINDS = frozenset(
    ("image-to-prompt", "latent-compose", "qwen-union", "image-grid", "image-composite")
)
NON_HISTOGRAM_SELECTIONS = frozenset(
    (
        "Image to prompt",
        "Compose saved latents",
        "Qwen Image 2.1 Union",
        "Make image grid",
        "Compose images",
    )
)


def histogram_visible(selection, image, control_type=None, kind_for=None):
    if selection in NON_HISTOGRAM_SELECTIONS:
        return False
    if kind_for is not None:
        kind = str(kind_for(selection, control_type) or "").lower()
        if kind in NON_HISTOGRAM_KINDS:
            return False
    return bool((selection and selection != "None") or image is not None)


async def analyze_source(image):
    from ruinedfooocus.app import api_runtime

    try:
        request = image_histogram.capture(image)
    except ValueError as error:
        raise gr.Error(translate_ui(str(error))) from None
    job = api_runtime.submit(image_histogram.operation(request))
    try:
        async with asyncio.timeout(ANALYSIS_TIMEOUT):
            while True:
                try:
                    kind, result = await asyncio.to_thread(job.events.get, True, 0.25)
                except queue.Empty:
                    continue
                if kind == "result":
                    return image_histogram.chart(result), result
                if kind in ("error", "invalid_request"):
                    raise gr.Error(translate_ui(str(result)))
    except TimeoutError:
        raise gr.Error(translate_ui("Histogram analysis timed out.")) from None
    finally:
        job.cancelled.set()


def create_ui(input_image, *, tool_selection=None, control_type=None, kind_for=None):
    with gr.Accordion(
        translate_ui("Histogram analysis"), open=False, visible=tool_selection is None
    ) as section:
        gr.Markdown(
            translate_ui(
                "Analyze this source without a model. Shows 256 bins for RGB, BT.709 luminance and each channel. "
                "All pixels count, including transparent pixels. Colors use their stored RGB values."
            )
        )
        with gr.Row():
            analyze = gr.Button(
                translate_ui("Analyze histogram"), interactive=input_image.value is not None
            )
            cancel = gr.Button(translate_ui("Cancel analysis"), interactive=False)
        plot = gr.HTML()
        data = gr.JSON(label=translate_ui("Histogram bins and source details"), open=False)
    if tool_selection is not None:
        vis_inputs = [tool_selection, input_image]
        vis_triggers = [tool_selection.change, input_image.change]
        if control_type is not None:
            vis_inputs.append(control_type)
            vis_triggers.append(control_type.change)

        def section_visibility(sel, img, ctrl_type=None):
            return gr.update(visible=histogram_visible(sel, img, ctrl_type, kind_for))

        gr.on(
            triggers=vis_triggers,
            fn=section_visibility,
            inputs=vis_inputs,
            outputs=[section],
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )

    def sync_analyze_button(img):
        return gr.update(interactive=img is not None)

    def enable_cancel():
        return gr.update(interactive=True)

    def disable_cancel():
        return gr.update(interactive=False)

    input_image.change(
        sync_analyze_button,
        inputs=input_image,
        outputs=analyze,
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    analyze.click(
        enable_cancel,
        outputs=cancel,
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    event = analyze.click(
        analyze_source,
        inputs=input_image,
        outputs=[plot, data],
        api_visibility="undocumented",
        concurrency_id="image-histogram",
        concurrency_limit=1,
    )
    event.then(
        disable_cancel,
        outputs=cancel,
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    cancel.click(
        disable_cancel,
        outputs=cancel,
        cancels=[event],
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    input_image.change(
        lambda: ("", None),
        outputs=[plot, data],
        cancels=[event],
        queue=False,
        api_visibility="undocumented",
    )
    return plot, data
