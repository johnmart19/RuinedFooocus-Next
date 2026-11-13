"""Ordered owned uploads and bounded controls for no-checkpoint image tools."""

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui as t
from ruinedfooocus.features.image_processing import image_composition as composition
from ruinedfooocus.features.image_processing.composition_native import modes


def move(images, index, direction):
    values = list(images or [])
    if values:
        composition.source_images(values)
    if type(index) is not int or not 0 <= index < len(values):
        return values, index
    target = index + direction
    if 0 <= target < len(values):
        values[index], values[target] = values[target], values[index]
        index = target
    return values, index


def create_ui(selection, tool_type, kind_for):
    from ruinedfooocus.app import state as shared

    saved = shared.settings.default_settings
    kind = kind_for(selection.value, tool_type.value)
    controls = {}
    with gr.Group(visible=composition.handles(kind)) as panel:
        gr.Markdown(
            t(
                "Add ordered images. A grid uses every image; composition uses base first, overlay second. Images fit with padding, never cropping. Working copies are 8-bit sRGB; originals are retained."
            )
        )
        controls["composition_images"] = gr.Gallery(
            label=t("Ordered input images"),
            value=[],
            type="pil",
            file_types=["image"],
            format="png",
            interactive=True,
            columns=2,
            height=280,
        )
        selected = gr.State(None)
        with gr.Row():
            earlier = gr.Button(t("Move selected earlier"), size="sm")
            later = gr.Button(t("Move selected later"), size="sm")
            clear = gr.Button(t("Clear image inputs"), size="sm")
        with gr.Group(visible=kind == "image-grid") as grid:
            gr.Markdown(
                t(
                    "Grid output is opaque RGB on black. Transparent source pixels and empty cells use black."
                )
            )
            for key, label, low, high in (
                ("grid_columns", "Columns", 1, 20),
                ("grid_cell_width", "Cell width", 32, 2048),
                ("grid_cell_height", "Cell height", 32, 2048),
                ("grid_padding", "Cell spacing", 0, 50),
            ):
                controls[key] = gr.Number(
                    label=t(label),
                    minimum=low,
                    maximum=high,
                    precision=0,
                    value=saved.get(key, composition.DEFAULTS[key]),
                )
        with gr.Group(visible=kind == "image-composite") as composite:
            controls["composition_mode"] = gr.Dropdown(
                label=t("Composition mode"),
                choices=[(t("Masked replace"), "masked"), *modes()],
                value=saved.get("composition_mode", "SRC_OVER"),
            )
            controls["composition_mask"] = gr.Image(
                label=t("Overlay selection mask (optional): white selects, black preserves"),
                type="pil",
                image_mode="L",
                format="png",
                height=160,
            )
            gr.Markdown(
                t(
                    "The base defines the output canvas. Zero rectangle width or height uses the available area. The overlay and its mask fit the same rectangle; source alpha is retained."
                )
            )
            with gr.Row():
                for key, label in (
                    ("composition_x", "Overlay left"),
                    ("composition_y", "Overlay top"),
                    ("composition_width", "Overlay rectangle width (0 = available)"),
                    ("composition_height", "Overlay rectangle height (0 = available)"),
                ):
                    controls[key] = gr.Number(
                        label=t(label),
                        minimum=0,
                        maximum=16384,
                        precision=0,
                        value=saved.get(key, 0),
                    )
        gr.Markdown(
            t(
                "These image tools run once and do not use a generation checkpoint, prompt, LoRA or sampling settings."
            )
        )
    for key, control in controls.items():
        shared.add_ctrl(key, control)
    quiet = dict(queue=False, show_progress="hidden", api_visibility="undocumented")

    def select(event: gr.SelectData):
        return event.index if isinstance(event.index, int) and event.selected else None

    def validate_uploads(images):
        try:
            if images:
                composition.source_images(images)
        except ValueError as error:
            raise gr.Error(str(error)) from None

    controls["composition_images"].upload(
        validate_uploads, inputs=controls["composition_images"], **quiet
    )
    controls["composition_images"].select(select, outputs=selected, **quiet)
    earlier.click(
        lambda images, index: move(images, index, -1),
        inputs=[controls["composition_images"], selected],
        outputs=[controls["composition_images"], selected],
        **quiet,
    )
    later.click(
        lambda images, index: move(images, index, 1),
        inputs=[controls["composition_images"], selected],
        outputs=[controls["composition_images"], selected],
        **quiet,
    )
    clear.click(
        lambda: ([], None, None),
        outputs=[controls["composition_images"], controls["composition_mask"], selected],
        **quiet,
    )

    def context(choice, kind):
        active = kind_for(choice, kind)
        return (
            gr.update(visible=composition.handles(active)),
            gr.update(visible=active == "image-grid"),
            gr.update(visible=active == "image-composite"),
        )

    gr.on(
        triggers=[selection.change, tool_type.change],
        fn=context,
        inputs=[selection, tool_type],
        outputs=[panel, grid, composite],
        show_progress_on=[],
        **quiet,
    )
    return controls
