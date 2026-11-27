"""Paint, layers and drag-crop on the current Editor image, applied as one undoable step.

Gradio's native ImageEditor owns the brush, eraser, layers and crop handles in the
browser. Nothing here runs a model: Apply flattens the canvas the user sees, saves
it beside other results and records it as a completed step of the owned Editor
session, so Undo, Redo and every later action treat it like any other edit.
"""

import gradio as gr

from ruinedfooocus.app.resource_lifecycle import ui_request_owner
from ruinedfooocus.core.translation_manager import translate_ui as t

OPERATION = "paint"
PALETTE = ["#000000", "#FFFFFF", "#E5484D", "#F5A524", "#30A46C", "#0090FF", "#8E4EC6"]


def flattened(value):
    """The visible canvas as RGBA, or None when nothing was loaded."""
    from PIL import Image

    composite = value.get("composite") if isinstance(value, dict) else None
    if composite is None and isinstance(value, dict):
        composite = value.get("background")
    return composite.convert("RGBA") if isinstance(composite, Image.Image) else None


def painted_layers(value):
    layers = value.get("layers") if isinstance(value, dict) else None
    return len(layers) if isinstance(layers, (list, tuple)) else 0


def save_result(image, source_name):
    """Write the flattened canvas to the outputs folder with plain processing metadata."""
    import json

    from PIL.PngImagePlugin import PngInfo

    from ruinedfooocus.app import state as shared
    from ruinedfooocus.core.util import generate_temp_filename

    path = generate_temp_filename(
        folder=shared.path_manager.model_paths["temp_outputs_path"], extension="png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    information = PngInfo()
    information.add_text(
        "parameters",
        json.dumps(
            dict(
                software="RuinedFooocus",
                kind="processing",
                controlnet={"type": OPERATION},
                width=image.width,
                height=image.height,
                image_utility_source=dict(
                    operation=OPERATION, source=source_name, policy="flattened browser canvas"
                ),
            )
        ),
    )
    image.save(path, pnginfo=information)
    return path


def execution(source_frame, image, layers):
    """Facts the Editor session needs to accept a result whose frame may differ."""
    result = dict(
        operation=OPERATION,
        layers=layers,
        alpha_policy="canvas alpha; erased areas become transparent",
    )
    size = (source_frame.width, source_frame.height)
    if image.size != size:
        result["geometry"] = dict(policy="canvas crop", source=list(size), result=list(image.size))
    if source_frame.dtype != "uint8":
        result["precision_conversion"] = "8-bit sRGB canvas"
    return result


class PaintCanvas:
    def __init__(self):
        with gr.Accordion(t("Paint and crop"), open=False, elem_id="editor-paint"):
            gr.Markdown(
                t(
                    "Load the current image, then paint with the brush, erase, add layers or drag the crop handles. Apply saves the result as a new step that Undo can take back."
                )
            )
            with gr.Row(elem_classes=["rf-actions", "rf-actions-compact"]):
                self.load = gr.Button(t("Load current image"), size="sm", interactive=False)
                self.apply = gr.Button(
                    t("Apply painting"), variant="primary", size="sm", interactive=False
                )
            self.canvas = gr.ImageEditor(
                type="pil",
                format="png",
                image_mode="RGBA",
                sources=(),
                layers=True,
                transforms=("crop",),
                brush=gr.Brush(colors=PALETTE, color_mode="defaults"),
                eraser=gr.Eraser(),
                show_label=False,
                interactive=True,
                buttons=["fullscreen"],
                elem_id="paint-canvas",
                height="var(--rf-viewer-height, min(65dvh, 100vw))",
            )

    def bind(self, workspace):
        store = workspace.selected_store

        def current(context, owner):
            if not isinstance(context, dict) or not context.get("session_id"):
                raise ValueError("Open an image in the Editor first.")
            snapshot = store().snapshot(context["session_id"], owner=owner)
            if snapshot.revision != context.get(
                "revision"
            ) or snapshot.current.artifact_id != context.get("source_id"):
                raise ValueError("The current image changed. Load it again before painting.")
            return snapshot

        def load(context, request: gr.Request = None):
            from PIL import Image, ImageOps

            owner = ui_request_owner(request)
            try:
                snapshot = current(context, owner)
                record = store().artifact(
                    snapshot.session_id,
                    snapshot.current.artifact_id,
                    revision=snapshot.revision,
                    owner=owner,
                )
                with Image.open(record.path) as image:
                    return ImageOps.exif_transpose(image).convert("RGBA"), gr.update(
                        interactive=True
                    )
            except (ValueError, OSError, KeyError) as error:
                raise gr.Error(str(error)) from None

        self.load.click(
            load,
            inputs=workspace.context,
            outputs=[self.canvas, self.apply],
            show_progress="minimal",
            api_visibility="private",
        )

        def sync_paint_buttons(context, value):
            has_context = isinstance(context, dict) and bool(context.get("session_id"))
            return (
                gr.update(interactive=has_context),
                gr.update(interactive=has_context and flattened(value) is not None),
            )

        gr.on(
            triggers=[workspace.context.change, self.canvas.change],
            fn=sync_paint_buttons,
            inputs=[workspace.context, self.canvas],
            outputs=[self.load, self.apply],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

        def apply(value, context, task_id, version, request: gr.Request = None):
            image = flattened(value)
            if image is None:
                raise gr.Error("Load the current image and paint or crop it first.")

            def commit():
                from pathlib import Path

                owner = ui_request_owner(request)
                snapshot = current(context, owner)
                record = store().artifact(
                    snapshot.session_id,
                    snapshot.current.artifact_id,
                    revision=snapshot.revision,
                    owner=owner,
                )
                layers = painted_layers(value)
                ticket = store().capture(
                    snapshot.session_id,
                    snapshot.revision,
                    OPERATION,
                    dict(layers=layers, width=image.width, height=image.height),
                    source_id=snapshot.current.artifact_id,
                    owner=owner,
                )
                path = save_result(image, Path(record.path).name)
                try:
                    completed = store().complete(
                        ticket,
                        path,
                        execution=execution(snapshot.current.frame, image, layers),
                        owner=owner,
                    )
                except BaseException:
                    path.unlink(missing_ok=True)
                    raise
                return workspace.snapshot_updates(completed, request=request)

            try:
                return workspace._source_action(task_id, version, request, commit, prepare=True)
            except (ValueError, OSError, KeyError) as error:
                raise gr.Error(str(error)) from None

        self.apply.click(
            apply,
            inputs=[
                self.canvas,
                workspace.context,
                workspace.run_controls["task"],
                workspace.version,
            ],
            outputs=workspace.source_outputs,
            js=workspace.version_js,
            show_progress="minimal",
            show_progress_on=workspace.notice,
            api_visibility="private",
        )
