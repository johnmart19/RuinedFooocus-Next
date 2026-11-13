"""One adjustment surface over the three immutable native utility requests."""

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui as t
from ruinedfooocus.features.image_processing import image_geometry_tools as transform
from ruinedfooocus.features.image_processing import image_mask_tools as mask
from ruinedfooocus.features.image_processing import image_color_tools as color
from ruinedfooocus.features.image_processing.still_image_utilities import (
    ADJUSTMENT_KINDS,
    ADJUSTMENT_DEFAULTS,
)


def create_ui(selection, control_type, kind_for):
    import ruinedfooocus.app.state as shared

    saved = shared.settings.default_settings
    initial = kind_for(selection.value, control_type.value)
    controls, visibility = {}, []

    def field(
        name,
        label,
        *,
        choices=None,
        low=0,
        high=16384,
        step=1,
        info=None,
        when=None,
        **widget_options,
    ):
        default = ADJUSTMENT_DEFAULTS[name]
        options = dict(label=t(label), value=saved.get(name, default))
        options.update(widget_options)
        if when is not None:
            options["visible"] = bool(
                when(
                    initial,
                    saved.get("transform_operation", "resize"),
                    saved.get("mask_operation", "grow"),
                    saved.get("color_operation", "blend"),
                )
            )
        if info:
            options["info"] = t(info)
        if choices:
            component = gr.Dropdown(
                choices=[
                    (
                        t(
                            {
                                "histogram": "Histogram (256-bin)",
                                "sdr-to-hdr": "SDR to HDR",
                                "crop-to-mask": "Crop to selection",
                            }[value]
                            if value in ("histogram", "sdr-to-hdr", "crop-to-mask")
                            else value.replace("-", " ").replace("_", " ").capitalize()
                        ),
                        value,
                    )
                    for value in choices
                ],
                **options,
            )
        elif isinstance(default, bool):
            component = gr.Checkbox(**options)
        elif isinstance(default, str):
            component = gr.Textbox(**options)
        else:
            component = gr.Number(
                minimum=low,
                maximum=high,
                step=step,
                precision=0 if isinstance(default, int) else None,
                **options,
            )
        controls[name] = component
        if when is not None:
            visibility.append((component, when))
        return component

    with gr.Group(visible=initial in ADJUSTMENT_KINDS) as panel:
        gr.Markdown(
            t(
                "Adjust one source image. Choose an operation, then run the image action. No model or prompt is needed. Precision originals require PNG16 or EXR output."
            )
        )
        family = gr.Radio(
            choices=[
                (t("Transform"), "image-transform"),
                (t("Mask"), "image-mask"),
                (t("Color"), "image-color"),
            ],
            value=initial if initial in ADJUSTMENT_KINDS else "image-transform",
            label=t("Adjustment group"),
            elem_classes=["rf-chips"],
        )

        def mask_visible(kind, operation, mask_operation, color_operation):
            return (
                kind == "image-transform"
                and operation == "crop-to-mask"
                or kind == "image-mask"
                and mask_operation not in ("channel", "color")
            )

        controls["mask_image"] = gr.File(
            label=t("Input selection mask"),
            type="filepath",
            file_types=[".png", ".exr", ".avif"],
            visible=mask_visible(
                initial,
                saved.get("transform_operation", "resize"),
                saved.get("mask_operation", "grow"),
                saved.get("color_operation", "blend"),
            ),
        )
        visibility.append((controls["mask_image"], mask_visible))
        with gr.Group(visible=initial == "image-transform") as transform_panel:
            operation = field(
                "transform_operation",
                "Transform operation",
                choices=("resize", "crop", "pad", "rotate", "flip", "crop-to-mask"),
            )
            field(
                "transform_width",
                "Target width",
                info="0 keeps the source dimension or derives it from the other dimension.",
                when=lambda k, o, m, c: k == "image-transform" and o in ("resize", "crop"),
            )
            field(
                "transform_height",
                "Target height",
                when=lambda k, o, m, c: k == "image-transform" and o in ("resize", "crop"),
            )
            field(
                "transform_fit",
                "Resize fit",
                choices=("contain", "stretch", "center-crop"),
                when=lambda k, o, m, c: o == "resize",
            )
            field(
                "transform_interpolation",
                "Interpolation",
                choices=transform.INTERPOLATIONS,
                when=lambda k, o, m, c: o == "resize",
            )
            for name in ("x", "y"):
                field("transform_" + name, "Crop " + name, when=lambda k, o, m, c: o == "crop")
            for name in ("left", "top", "right", "bottom"):
                field("transform_" + name, "Padding " + name, when=lambda k, o, m, c: o == "pad")
            field(
                "transform_padding_style",
                "Padding style",
                choices=("transparent", "color", "mirror"),
                when=lambda k, o, m, c: o == "pad",
            )
            field(
                "transform_padding_color",
                "Padding color (#RRGGBB)",
                when=lambda k, o, m, c: o == "pad",
            )
            field("transform_feather", "Padding feather", when=lambda k, o, m, c: o == "pad")
            field(
                "transform_rotation",
                "Rotation",
                choices=("none", "90 degrees", "180 degrees", "270 degrees"),
                when=lambda k, o, m, c: o == "rotate",
            )
            field(
                "transform_flip",
                "Flip direction",
                choices=("horizontal", "vertical"),
                when=lambda k, o, m, c: o == "flip",
            )
        with gr.Group(visible=initial == "image-mask") as mask_panel:
            mask_operation = field("mask_operation", "Mask operation", choices=mask.OPERATIONS)
            gr.Markdown(
                t(
                    "White selects pixels. Extract a channel or color from the source, or upload an exact-frame grayscale mask. The mask result can be attached to its source in the image workspace."
                )
            )
            controls["mask_reference"] = gr.File(
                label=t("Second selection mask"),
                type="filepath",
                file_types=[".png", ".exr", ".avif"],
                visible=False,
            )
            visibility.append((controls["mask_reference"], lambda k, o, m, c: m == "composite"))
            field(
                "mask_expand",
                "Mask growth (negative shrinks)",
                low=-16384,
                when=lambda k, o, m, c: m == "grow",
            )
            field("mask_tapered_corners", "Taper mask corners", when=lambda k, o, m, c: m == "grow")
            for name in ("left", "top", "right", "bottom"):
                field("mask_" + name, "Feather " + name, when=lambda k, o, m, c: m == "feather")
            field(
                "mask_composite_operation",
                "Combine masks",
                choices=mask.COMPOSITE_OPERATIONS,
                when=lambda k, o, m, c: m == "composite",
            )
            field(
                "mask_channel",
                "Source channel",
                choices=mask.CHANNELS,
                when=lambda k, o, m, c: m == "channel",
            )
            field("mask_color", "Selected color (#RRGGBB)", when=lambda k, o, m, c: m == "color")
            field(
                "mask_threshold",
                "Mask threshold",
                high=1,
                step=0.01,
                when=lambda k, o, m, c: m == "threshold",
            )
            field("mask_invert", "Invert input selection")
        with gr.Group(visible=initial == "image-color") as color_panel:
            gr.Markdown(
                t(
                    "Blend clamps RGB to 0–1. Transfer requires sRGB in 0–1 and also clamps its result. Quantize uses the native RGB8 palette conversion. These three operations preserve source alpha."
                )
            )
            color_operation = field("color_operation", "Color operation", choices=color.OPERATIONS)
            controls["color_reference"] = gr.File(
                label=t("Color reference image"), type="filepath", file_types=["image", ".exr"]
            )
            visibility.append(
                (controls["color_reference"], lambda k, o, m, c: c in ("blend", "transfer"))
            )
            field(
                "color_blend_mode",
                "Blend mode",
                choices=color.BLEND_MODES,
                when=lambda k, o, m, c: c == "blend",
            )
            field(
                "color_blend_factor",
                "Blend amount",
                high=1,
                step=0.01,
                when=lambda k, o, m, c: c == "blend",
            )
            field(
                "color_transfer_method",
                "Color transfer method",
                choices=color.TRANSFER_METHODS,
                when=lambda k, o, m, c: c == "transfer",
            )
            field(
                "color_transfer_strength",
                "Color transfer strength",
                high=10,
                step=0.01,
                when=lambda k, o, m, c: c == "transfer",
            )
            field(
                "color_quantize_colors",
                "Palette colors",
                low=1,
                high=256,
                when=lambda k, o, m, c: c == "quantize",
            )
            field(
                "color_quantize_dither",
                "Palette dither",
                choices=color.DITHERS,
                when=lambda k, o, m, c: c == "quantize",
            )
            from ruinedfooocus.features.image_processing.ui_color_operations import (
                create_controls,
                HDR_MIRROR,
            )

            hdr_precision = create_controls(
                field, visibility, saved.get("color_operation", "blend")
            )

    for name, component in controls.items():
        shared.add_ctrl(name, component)
    shared.add_ctrl(HDR_MIRROR, hdr_precision)

    def choose_adjustment(kind):
        if kind not in ADJUSTMENT_KINDS:
            raise gr.Error("Choose Transform, Mask or Color.")
        choices = list(control_type.choices)
        choices += [
            (candidate.capitalize(), candidate.capitalize())
            for candidate in ADJUSTMENT_KINDS
            if candidate.capitalize() not in {value for _, value in choices}
        ]
        return gr.update(value=kind.capitalize(), choices=choices)

    family.input(
        choose_adjustment,
        inputs=family,
        outputs=control_type,
        queue=False,
        api_visibility="undocumented",
    )

    def adjustment_context(choice, tool, transform_op, mask_op, color_op):
        kind = kind_for(choice, tool)
        return [
            gr.update(visible=kind in ADJUSTMENT_KINDS),
            gr.update(value=kind if kind in ADJUSTMENT_KINDS else "image-transform"),
            *[gr.update(visible=kind == candidate) for candidate in ADJUSTMENT_KINDS],
            *[
                gr.update(visible=bool(predicate(kind, transform_op, mask_op, color_op)))
                for _, predicate in visibility
            ],
        ]

    gr.on(
        triggers=[
            selection.change,
            control_type.change,
            operation.change,
            mask_operation.change,
            color_operation.change,
        ],
        fn=adjustment_context,
        inputs=[selection, control_type, operation, mask_operation, color_operation],
        outputs=[
            panel,
            family,
            transform_panel,
            mask_panel,
            color_panel,
            *[component for component, _ in visibility],
        ],
        queue=False,
        api_visibility="undocumented",
        show_progress="hidden",
    )
    return controls
