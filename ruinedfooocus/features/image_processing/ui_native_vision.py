"""Image tools controls for native RT-DETR detection and SAM 3 segmentation.

Points are added by clicking the tool image. They are pixel coordinates of that
image, so they are cleared whenever the image changes.
"""

import json
import math

import gradio as gr

from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui
from ruinedfooocus.features.image_processing.native_vision import (
    DEFAULTS,
    KINDS,
    coco_classes,
    native_model_choices,
)

POSITIVE, NEGATIVE = "positive", "negative"
PREVIEW_SIDE = 768


def model_choices(manager, kind, selected=""):
    """Header-validated local files for the active tool; an absent saved file stays visible."""
    values = native_model_choices(manager, kind) if kind in KINDS else []
    if selected and selected not in {value for _, value in values}:
        values.append((translate_ui("Unavailable: ") + selected, selected))
    return values


def model_prompt(kind):
    if kind == "depth-map":
        return dict(
            label=translate_ui("Depth Anything 3 model"),
            info=translate_ui(
                "Choose an installed Depth Anything 3 model. Generate saves a grayscale map at the source size. White is near."
            ),
        )
    return dict(
        label=translate_ui("Detection / segmentation model"),
        info=translate_ui(
            "Choose a local RT-DETR model for detection or SAM 3 / 3.1 for segmentation."
        ),
    )


def parse_points(text):
    """Lenient reader for the UI only; submission validates strictly against the image."""
    if not isinstance(text, str) or len(text) > 16384:
        return []
    try:
        points = json.loads(text or "[]")
    except (ValueError, RecursionError):
        return []
    if not isinstance(points, list) or len(points) > 256:
        return []
    return [
        point
        for point in (points if isinstance(points, list) else [])
        if isinstance(point, dict)
        and all(
            type(point.get(axis)) in (int, float) and math.isfinite(point[axis])
            for axis in ("x", "y")
        )
    ]


def dump_points(points):
    return json.dumps(
        [{"x": int(point["x"]), "y": int(point["y"])} for point in points], separators=(",", ":")
    )


def summary(positive, negative, locale=None):
    included, excluded = len(parse_points(positive)), len(parse_points(negative))
    if not included and not excluded:
        return localize("Click the image above to mark what to keep.", locale)
    return localize(
        "{included} points to keep, {excluded} points to exclude.",
        locale,
        included=included,
        excluded=excluded,
    )


def preview(image, positive, negative):
    """Downscaled copy with markers; the submitted image itself is never drawn on."""
    points = [(point, True) for point in parse_points(positive)]
    points += [(point, False) for point in parse_points(negative)]
    if image is None or not points:
        return None
    from PIL import ImageDraw

    scale = min(1, PREVIEW_SIDE / max(image.size))
    result = image.convert("RGB")
    if scale < 1:
        result = result.resize((round(image.width * scale), round(image.height * scale)))
    draw = ImageDraw.Draw(result)
    radius = max(5, round(max(result.size) / 110))
    for point, keep in points:
        if not 0 <= point["x"] < image.width or not 0 <= point["y"] < image.height:
            continue
        x, y = point["x"] * scale, point["y"] * scale
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(40, 200, 90) if keep else (225, 60, 60),
            outline=(255, 255, 255),
            width=2,
        )
    return result


def create_ui(selection, control_type, input_image, model_refresh, kind_for):
    import ruinedfooocus.app.state as shared

    saved = shared.settings.default_settings
    kind = kind_for(selection.value, control_type.value)
    saved_model = saved.get("native_vision_model") or ""
    with gr.Group(visible=kind in KINDS) as group:
        model = gr.Dropdown(
            choices=model_choices(shared.path_manager, kind, saved_model),
            value=saved_model or None,
            **model_prompt(kind),
        )
        threshold = gr.Slider(
            label=translate_ui("Confidence threshold"),
            minimum=0,
            maximum=1,
            step=0.01,
            value=saved.get("vision_threshold", DEFAULTS["vision_threshold"]),
            info=translate_ui("Minimum confidence for object detection or text segmentation."),
            visible=kind in ("detect", "segment"),
        )
        with gr.Row(visible=kind == "detect") as detect_row:
            detect_class = gr.Dropdown(
                label=translate_ui("Object class"),
                choices=[(translate_ui("All classes"), "all"), *coco_classes()],
                value=saved.get("detect_class", DEFAULTS["detect_class"]),
                min_width=160,
                info=translate_ui("All classes, or one of the 80 COCO classes."),
            )
            detect_max_count = gr.Slider(
                label=translate_ui("Maximum detections"),
                minimum=1,
                maximum=1000,
                step=1,
                value=saved.get("detect_max_count", DEFAULTS["detect_max_count"]),
                min_width=160,
            )
        with gr.Group(visible=kind == "segment") as segment_group:
            segment_text = gr.Textbox(
                label=translate_ui("Text categories (optional)"),
                value="",
                max_lines=3,
                info=translate_ui(
                    "For example: person:3, dog:2. Use text, points or boxes. Text needs the SAM model's language encoder."
                ),
            )
            point_mode = gr.Radio(
                label=translate_ui("Clicking the image adds"),
                choices=[
                    (translate_ui("A point to keep"), POSITIVE),
                    (translate_ui("A point to exclude"), NEGATIVE),
                ],
                value=POSITIVE,
            )
            point_summary = gr.Markdown(translate_ui("Click the image above to mark what to keep."))
            point_preview = gr.Image(
                label=translate_ui("Marked points"),
                visible=False,
                interactive=False,
                buttons=[],
                height="clamp(10rem, 26dvh, 20rem)",
            )
            with gr.Row():
                undo_point = gr.Button(translate_ui("Undo last point"), size="sm", min_width=120)
                clear_points = gr.Button(
                    translate_ui("Clear points"), size="sm", variant="stop", min_width=120
                )
            with gr.Row():
                individual_masks = gr.Checkbox(
                    label=translate_ui("Separate result for each object"),
                    value=saved.get(
                        "segment_individual_masks", DEFAULTS["segment_individual_masks"]
                    ),
                )
                max_instances = gr.Slider(
                    label=translate_ui("Maximum instances"),
                    minimum=1,
                    maximum=64,
                    step=1,
                    value=saved.get("segment_max_instances", DEFAULTS["segment_max_instances"]),
                    info=translate_ui("Large images also have a combined mask pixel limit."),
                    min_width=160,
                )
            with gr.Accordion(translate_ui("Source-image boxes"), open=False):
                segment_boxes = gr.Textbox(
                    label=translate_ui("Boxes (JSON pixels)"),
                    value="[]",
                    max_lines=4,
                    info=translate_ui('Example: [{"x":10,"y":20,"width":100,"height":80}]'),
                )
            with gr.Row():
                segment_output = gr.Radio(
                    label=translate_ui("Result"),
                    choices=[
                        (translate_ui("Cutout (transparent background)"), "cutout"),
                        (translate_ui("Mask (black and white)"), "mask"),
                    ],
                    value=saved.get("segment_output", DEFAULTS["segment_output"]),
                    min_width=200,
                )
                segment_refine = gr.Slider(
                    label=translate_ui("Mask refinement passes"),
                    minimum=0,
                    maximum=5,
                    step=1,
                    value=saved.get(
                        "segment_refine_iterations", DEFAULTS["segment_refine_iterations"]
                    ),
                    min_width=160,
                )
            with gr.Accordion(translate_ui("Exact point coordinates"), open=False):
                positive_points = gr.Textbox(
                    label=translate_ui("Points to keep (JSON pixels)"),
                    value="[]",
                    max_lines=3,
                    info=translate_ui('Example: [{"x":100,"y":80}]'),
                )
                negative_points = gr.Textbox(
                    label=translate_ui("Points to exclude (JSON pixels)"),
                    value="[]",
                    max_lines=3,
                )
    # Click order, so Undo removes the most recent point of either kind.
    history = gr.State([])
    for name, control in (
        ("native_vision_model", model),
        ("vision_threshold", threshold),
        ("detect_class", detect_class),
        ("detect_max_count", detect_max_count),
        ("segment_positive_points", positive_points),
        ("segment_negative_points", negative_points),
        ("segment_refine_iterations", segment_refine),
        ("segment_output", segment_output),
        ("segment_text", segment_text),
        ("segment_boxes", segment_boxes),
        ("segment_individual_masks", individual_masks),
        ("segment_max_instances", max_instances),
    ):
        shared.add_ctrl(name, control)
    quiet = dict(queue=False, show_progress="hidden", api_visibility="undocumented")

    def context(choice, tool_type, selected, selected_class):
        active = kind_for(choice, tool_type)
        classes = gr.skip()
        if active == "detect":
            classes = gr.update(
                choices=[(translate_ui("All classes"), "all"), *coco_classes()],
                value=selected_class,
            )
        choices = model_choices(shared.path_manager, active, selected)
        return [
            gr.update(visible=active in KINDS),
            gr.update(visible=active == "detect"),
            gr.update(visible=active == "segment"),
            gr.update(choices=choices, value=selected or None, **model_prompt(active))
            if active in KINDS
            else gr.skip(),
            classes,
            gr.update(visible=active in ("detect", "segment")),
        ]

    triggers = [selection.change, control_type.change]
    if model_refresh is not None:
        triggers.append(model_refresh.change)
    gr.on(
        triggers=triggers,
        fn=context,
        inputs=[selection, control_type, model, detect_class],
        outputs=[group, detect_row, segment_group, model, detect_class, threshold],
        show_progress_on=[],
        **quiet,
    )
    point_outputs = [positive_points, negative_points, history, point_summary, point_preview]

    def render(image, positive, negative, order, request):
        marked = preview(image, positive, negative)
        return [
            positive,
            negative,
            order,
            summary(positive, negative, resolve_locale(request=request)),
            gr.update(value=marked, visible=marked is not None),
        ]

    def add_point(
        image,
        choice,
        tool_type,
        mode,
        positive,
        negative,
        order,
        request: gr.Request,
        event: gr.SelectData,
    ):
        index = event.index
        if (
            image is None
            or kind_for(choice, tool_type) != "segment"
            or not isinstance(index, (list, tuple))
            or len(index) != 2
        ):
            return [gr.skip()] * len(point_outputs)
        point = {
            "x": min(max(int(index[0]), 0), image.width - 1),
            "y": min(max(int(index[1]), 0), image.height - 1),
        }
        keep, exclude = parse_points(positive), parse_points(negative)
        target = keep if mode == POSITIVE else exclude
        if len(target) >= 256:
            raise gr.Error(
                localize(
                    "Use at most 256 points per point prompt.", resolve_locale(request=request)
                )
            )
        target.append(point)
        return render(
            image, dump_points(keep), dump_points(exclude), [*(order or []), mode], request
        )

    def undo(image, positive, negative, order, request: gr.Request):
        keep, exclude = parse_points(positive), parse_points(negative)
        order = list(order or [])
        target = order.pop() if order else (POSITIVE if keep else NEGATIVE)
        points = keep if target == POSITIVE else exclude
        if points:
            points.pop()
        return render(image, dump_points(keep), dump_points(exclude), order, request)

    def clear(request: gr.Request):
        return render(None, "[]", "[]", [], request)

    def edited(image, positive, negative, order, request: gr.Request):
        # Manual JSON edits replace click history; stale entries cannot undo a
        # different point kind. Undo then removes the final nonempty list entry.
        return [[], *render(image, positive, negative, [], request)[3:]]

    input_image.select(
        add_point,
        inputs=[
            input_image,
            selection,
            control_type,
            point_mode,
            positive_points,
            negative_points,
            history,
        ],
        outputs=point_outputs,
        **quiet,
    )
    undo_point.click(
        undo,
        inputs=[input_image, positive_points, negative_points, history],
        outputs=point_outputs,
        **quiet,
    )
    clear_points.click(clear, outputs=point_outputs, **quiet)
    # Coordinates belong to one image: any new, replaced or removed image clears them.
    gr.on(
        triggers=[input_image.change],
        fn=clear,
        outputs=point_outputs,
        **quiet,
    )
    gr.on(
        triggers=[positive_points.input, negative_points.input],
        fn=edited,
        inputs=[input_image, positive_points, negative_points, history],
        outputs=[history, point_summary, point_preview],
        **quiet,
    )
    return model
