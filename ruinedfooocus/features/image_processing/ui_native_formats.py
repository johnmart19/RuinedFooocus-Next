"""Explicit native export choices and original-preserving Library import/export."""

from pathlib import Path
import uuid

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui as t
from ruinedfooocus.features.image_processing.native_image_formats import FORMAT_FIELDS, OPTIONS


def format_choices(kind):
    depths, spaces = OPTIONS[kind]
    return (
        gr.update(choices=list(depths), value=depths[0]),
        gr.update(choices=list(spaces), value=spaces[0]),
        gr.update(visible=kind == "avif"),
    )


def create_controls():
    controls = {}
    controls["output_format"] = gr.Dropdown(
        label=t("Image file format"),
        choices=[("PNG", "png"), ("OpenEXR", "exr"), ("Still AVIF", "avif")],
        value="png",
    )
    controls["output_bit_depth"] = gr.Dropdown(
        label=t("Image bit depth"), choices=list(OPTIONS["png"][0]), value="8-bit"
    )
    controls["output_color_space"] = gr.Dropdown(
        label=t("Input color space"), choices=list(OPTIONS["png"][1]), value="sRGB"
    )
    controls["output_crf"] = gr.Slider(
        label=t("AVIF CRF"), minimum=1, maximum=63, step=1, precision=0, value=18, visible=False
    )
    controls["output_format"].input(
        format_choices,
        inputs=controls["output_format"],
        outputs=[controls[key] for key in FORMAT_FIELDS[1:]],
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    return controls


def create_ui():
    from ruinedfooocus.app import state as shared

    with gr.Accordion(t("Image file output"), open=False):
        gr.Markdown(
            t(
                "PNG keeps the usual output. EXR preserves floating point pixels. Still AVIF supports RGB and grayscale; use PNG or EXR for alpha."
            )
        )
        controls = create_controls()
    for key, control in controls.items():
        shared.add_ctrl(key, control)
    return controls


def replay_values(metadata):
    from ruinedfooocus.features.image_processing.native_image_formats import format_settings

    values = format_settings({key: metadata[key] for key in FORMAT_FIELDS if key in metadata})
    depths, spaces = OPTIONS[values["output_format"]]
    return {
        "output_format": gr.update(value=values["output_format"]),
        "output_bit_depth": gr.update(choices=list(depths), value=values["output_bit_depth"]),
        "output_color_space": gr.update(choices=list(spaces), value=values["output_color_space"]),
        "output_crf": gr.update(
            value=values["output_crf"], visible=values["output_format"] == "avif"
        ),
    }


def export_selection(record, kind, depth, color, crf, cache):
    from ruinedfooocus.features.gallery.actions import source_path
    from ruinedfooocus.features.image_processing.native_image_formats import (
        convert_color,
        read_export_source,
        write_image,
    )

    try:
        source = source_path(record)
        imported = read_export_source(source)
        source_path(record)
        metadata = imported.metadata.get("parameters", "{}")
        import json

        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        metadata = metadata if isinstance(metadata, dict) else {}
        metadata["output_source_dtype"] = str(imported.pixels.dtype)
        options = dict(zip(FORMAT_FIELDS, (kind, depth, color, int(crf))))
        destination = Path(cache) / "native-image-exports" / f"export-{uuid.uuid4().hex}.{kind}"
        pixels = convert_color(imported.pixels, imported.color_space, color)
        return str(write_image(destination, pixels, options, metadata))
    except (OSError, ValueError, RuntimeError) as error:
        raise gr.Error(str(error)) from None


def import_selection(path, folder):
    from ruinedfooocus.features.image_processing.native_image_formats import import_image

    if not path:
        raise gr.Error("Choose a PNG, EXR or still AVIF file first.")
    try:
        imported, _ = import_image(path, folder)
    except (OSError, ValueError, RuntimeError) as error:
        raise gr.Error(str(error)) from None
    return t("Imported an original copy. Refresh images to find it in Library."), str(imported)


def create_library_ui(selected, manager):
    from ruinedfooocus.core.cache_contract import configured_cache_root

    cache = configured_cache_root(manager)
    with gr.Accordion(t("Import or export a still image"), open=False):
        gr.Markdown(
            t(
                "Imports retain the original file and precision. Browser previews are derived display copies. Exporting an 8-bit source at a higher depth does not add detail."
            )
        )
        controls = create_controls()
        export = gr.Button(t("Export selected image"), interactive=False)
        download = gr.File(label=t("Exported image"), interactive=False)
        upload = gr.File(
            label=t("Original still image"), file_types=[".png", ".exr", ".avif"], type="filepath"
        )
        import_button = gr.Button(t("Import original copy"))
        status = gr.Markdown()
        imported = gr.File(visible=False)
    export.click(
        lambda record, *values: export_selection(record, *values, cache),
        inputs=[selected, *controls.values()],
        outputs=download,
        concurrency_id="native-image-exports",
        concurrency_limit=1,
        api_visibility="undocumented",
    )
    import_button.click(
        lambda path: import_selection(path, manager.model_paths["temp_outputs_path"]),
        inputs=upload,
        outputs=[status, imported],
        concurrency_id="native-image-exports",
        concurrency_limit=1,
        api_visibility="undocumented",
    )
    return export
