"""Contextual finishing controls and an explicit mirror of native EXR settings."""

import gradio as gr
from ruinedfooocus.core.translation_manager import translate_ui as t
from ruinedfooocus.features.image_processing import image_color_tools as color

HDR_MIRROR = "_adjustment_hdr_precision"


def create_controls(field, visibility, initial_operation):
    with gr.Group(visible=initial_operation == "text-overlay") as text:
        gr.Markdown(
            t(
                "Text requires an 8-bit sRGB source. Text adds its own opacity; the original image is preserved."
            )
        )
        field("color_text", "Text", lines=3, max_lines=8, max_length=2048)
        field("color_text_font_size", "Text size (% of image height)", low=0.5, high=50, step=0.5)
        field("color_text_color", "Text color (#RRGGBB)")
        field("color_text_position", "Text position", choices=color.TEXT_POSITIONS)
        field("color_text_align", "Text alignment", choices=color.TEXT_ALIGNMENTS)
        field("color_text_outline", "Text outline")
    with gr.Group(visible=initial_operation == "morphology") as morphology:
        field(
            "color_morphology_operation",
            "Morphology operation",
            choices=color.MORPHOLOGY_OPERATIONS,
        )
        field("color_morphology_kernel", "Kernel size", low=3, high=999)
    with gr.Group(visible=initial_operation == "sdr-to-hdr") as hdr:
        gr.Markdown(
            t("SDR-to-HDR requires EXR output. Choose its precision explicitly before running.")
        )
        field("color_hdr_luminance", "HDR luminance (nits; 0 uses calibration)", high=10000, step=1)
        precision = gr.Dropdown(
            label=t("HDR EXR output"),
            choices=[("16-bit float", "16-bit float"), ("32-bit float", "32-bit float")],
            value=None,
        )
    for panel, operation in (
        (text, "text-overlay"),
        (morphology, "morphology"),
        (hdr, "sdr-to-hdr"),
    ):
        visibility.append((panel, lambda k, o, m, c, operation=operation: c == operation))
    return precision


def bind_output_controls():
    from ruinedfooocus.app import state as shared
    from ruinedfooocus.features.image_processing.native_image_formats import OPTIONS

    mirror = shared.get_ctrl(HDR_MIRROR)
    output_format, depth, space, crf = [
        shared.get_ctrl(name)
        for name in ("output_format", "output_bit_depth", "output_color_space", "output_crf")
    ]

    def choose_hdr_output(value):
        if value not in OPTIONS["exr"][0]:
            raise gr.Error("Choose 16-bit or 32-bit floating point EXR output.")
        return (
            gr.update(value="exr"),
            gr.update(choices=list(OPTIONS["exr"][0]), value=value),
            gr.update(choices=list(OPTIONS["exr"][1]), value="HDR"),
            gr.update(visible=False),
        )

    mirror.input(
        choose_hdr_output,
        inputs=mirror,
        outputs=[output_format, depth, space, crf],
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def reflect_hdr_output(kind, precision, color_space):
        return gr.update(
            value=precision
            if kind == "exr" and color_space == "HDR" and precision in OPTIONS["exr"][0]
            else None
        )

    gr.on(
        triggers=[output_format.change, depth.change, space.change],
        fn=reflect_hdr_output,
        inputs=[output_format, depth, space],
        outputs=mirror,
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )
