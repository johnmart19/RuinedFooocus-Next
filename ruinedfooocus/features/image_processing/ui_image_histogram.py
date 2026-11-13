"""Explicit image analysis beside the source, independent of generation controls."""

import asyncio
import queue

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui
from ruinedfooocus.features.image_processing import image_histogram

ANALYSIS_TIMEOUT = 1800


async def analyze_source(image):
    from ruinedfooocus.app import api_runtime

    request = image_histogram.capture(image)
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


def create_ui(input_image):
    with gr.Accordion(translate_ui("Histogram analysis"), open=False):
        gr.Markdown(
            translate_ui(
                "Analyze this source without a model. Shows 256 bins for RGB, BT.709 luminance and each channel. "
                "All pixels count, including transparent pixels. Colors use their stored RGB values."
            )
        )
        with gr.Row():
            analyze = gr.Button(translate_ui("Analyze histogram"))
            cancel = gr.Button(translate_ui("Cancel analysis"))
        plot = gr.HTML()
        data = gr.JSON(label=translate_ui("Histogram bins and source details"), open=False)
    event = analyze.click(
        analyze_source,
        inputs=input_image,
        outputs=[plot, data],
        api_visibility="undocumented",
        concurrency_id="image-histogram",
        concurrency_limit=1,
    )
    cancel.click(fn=None, cancels=[event], api_visibility="undocumented")
    input_image.change(
        lambda: ("", None),
        outputs=[plot, data],
        cancels=[event],
        queue=False,
        api_visibility="undocumented",
    )
    return plot, data
