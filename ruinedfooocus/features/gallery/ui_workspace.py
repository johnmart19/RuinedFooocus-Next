"""One artifact-backed editor over the existing Create controls and worker."""

from concurrent.futures import CancelledError
from contextlib import contextmanager
import logging
import os
from pathlib import Path
from uuid import uuid4

import gradio as gr

from ruinedfooocus.app.resource_lifecycle import latest_ui_requests, ui_request_owner
from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui as t
from ruinedfooocus.features.gallery import editor_session
from ruinedfooocus.features.gallery.editing_workflow import open_editor, working_source
from ruinedfooocus.features.gallery.ui_guided_results import GuidedResultActions


class EditorWorkspace:
    """The state holds opaque P3 references; text/model/source widgets stay shared."""

    SOURCE_REFERENCES = {
        "reference_images": [],
        "qwen_union_guide": None,
        "qwen_union_source": None,
        "qwen_union_mask": None,
        "reference_adapter_image": None,
        "model_patch_image": None,
        "model_patch_inpaint_image": None,
        "model_patch_mask": None,
        "control_inpaint_image": None,
        "control_inpaint_mask": None,
        "_editor_guide_handoff": None,
    }

    def __init__(self, *, store=None):
        self.store = store
        self.context = gr.State(None)
        self.source_input = gr.State("placeholder")
        self.preview_signature = gr.State(None)
        self.output_explicit = gr.State(False)
        self.version = gr.Number(1, visible=False, precision=0)
        self.destination = gr.Textbox("create", visible=False, interactive=False)
        self.latest_generated = gr.State(None)
        self.channel = f"editor-workspace-{self.context._id}"
        self.chat_open_events = []
        self.controls = {}
        self.run_controls = {}
        self.navigation = []
        self.version_js = "(...values) => { window.__rfEditorRequest=(window.__rfEditorRequest||1)+1; values[values.length-1]=window.__rfEditorRequest; return values; }"
        self.panel = gr.Accordion(t("Image workspace"), open=True, elem_id="editor-workspace")
        with self.panel:
            with gr.Row(elem_classes=["rf-actions", "rf-actions-compact"]):
                self.open_current = gr.Button(t("Open current image"), size="sm")
                self.open_latest = gr.Button(t("Latest generated"), size="sm")
                self.choose_gallery = gr.Button(t("Choose from Gallery"), size="sm")
                # History actions start their own group at the row end.
                self.undo = gr.Button(t("Undo"), size="sm", elem_classes=["rf-group-start"])
                self.redo = gr.Button(t("Redo"), size="sm")
                self.reopen = gr.Button(t("Reopen image history"), size="sm")
                self.use_selection = gr.Button(
                    t("Use result as selection mask"), size="sm", visible=False
                )
            self.guided_results = GuidedResultActions(self.use_selection)
            self.notice = gr.Markdown(
                t("Choose an image, an action and a model, then run it. The original is kept.")
            )
            with gr.Group(visible=False, elem_id="editor-gallery-picker") as self.picker:
                self.picker_records = gr.State([])
                self.picker_selected = gr.State(None)
                self.picker_version = gr.Number(0, precision=0, visible=False)
                self.picker_gallery = gr.Gallery(
                    label=t("Choose an original image"),
                    columns=4,
                    height="clamp(12rem, 32dvh, 24rem)",
                    preview=False,
                    # One click opens the image; an enlarged preview would hide the grid.
                    allow_preview=False,
                    object_fit="contain",
                    buttons=[],
                )
                with gr.Row(elem_classes=["rf-actions", "rf-actions-compact"]):
                    self.picker_previous = gr.Button(
                        t("Previous"), size="sm", min_width=72, scale=0
                    )
                    self.picker_page = gr.Number(
                        1, label=t("Page"), minimum=1, precision=0, min_width=72, scale=0
                    )
                    self.picker_total = gr.Markdown()
                    self.picker_next = gr.Button(t("Next"), size="sm", min_width=72, scale=0)
                with gr.Row(elem_classes=["rf-actions", "rf-actions-compact"]):
                    self.picker_open = gr.Button(
                        t("Open selected image"), size="sm", interactive=False, visible=False
                    )
                    self.picker_close = gr.Button(t("Close picker"), size="sm")
                self.picker_notice = gr.Markdown()
            self.action = gr.Radio(
                label=t("Image action"),
                elem_classes=["rf-chips"],
                choices=[
                    (t("Generate"), "generate"),
                    (t("Edit selected image"), "prompt-edit"),
                    (t("Inpaint selected image"), "inpaint"),
                    (t("Remove background"), "remove-background"),
                    (t("Select background"), "select-background"),
                    (t("Generate background"), "generate-background"),
                    (t("Compose foreground and background"), "compose"),
                    (t("Upscale selected image"), "upscale"),
                    (t("Depth map"), "depth-map"),
                    (t("Restore still image (SeedVR2)"), "restore-seedvr2"),
                    (t("Adjust image"), "adjust-image"),
                ],
                value="prompt-edit",
            )
            # One-click entry to image tools that otherwise sit in the tool list.
            self.quick_tool = gr.Radio(
                label=t("Image tools"),
                choices=[
                    (t("Face landmarks"), "Face landmarks"),
                    (t("Pose"), "Whole-body pose"),
                    # Depth map is an Image action above; no second shortcut.
                    # Multi-object selection; results become masks for the next edit.
                    (t("Detect objects"), "Detect objects"),
                    (t("Segment by points"), "Segment by points"),
                ],
                value=None,
                elem_classes=["rf-chips"],
            )
            self.prompt_processing = gr.Checkbox(
                label=t("Apply current styles and negative processing"),
                value=False,
            )
            self.denoise = gr.Slider(
                label=t("Edit strength"),
                minimum=0.01,
                maximum=1.0,
                step=0.01,
                value=0.3,
                visible=True,
            )
            with gr.Accordion(t("Action inputs"), open=True) as self.action_inputs:
                self.mask = gr.File(
                    label=t("Selection mask (grayscale)"), type="filepath", visible=False
                )
                self.mask_choice = gr.Radio(
                    [(t("Union mask"), "union"), (t("Instance mask"), "instance")],
                    value="union",
                    label=t("Mask selection"),
                    visible=False,
                )
                self.mask_index = gr.Number(
                    0, label=t("Instance index"), precision=0, minimum=0, visible=False
                )
                self.mask_count = gr.Number(
                    1, label=t("Instance count"), precision=0, minimum=1, maximum=64, visible=False
                )
                self.invert = gr.Checkbox(
                    label=t("Invert selection mask"), value=False, visible=False
                )
                self.background = gr.File(
                    label=t("Background image"), type="filepath", visible=False
                )
                self.framing = gr.Radio(
                    [(t("Preserve frame"), "preserve"), (t("Pad"), "pad"), (t("Crop"), "crop")],
                    label=t("Image framing"),
                    value="preserve",
                )
                self.preview = gr.Button(t("Check resources"), size="sm")

    @staticmethod
    def action_label(action, destination="editor"):
        if destination == "create":
            return t("Generate")
        return {
            "generate": t("Generate"),
            "prompt-edit": t("Edit image"),
            "inpaint": t("Inpaint image"),
            "remove-background": t("Remove background"),
            "select-background": t("Select background"),
            "generate-background": t("Generate background"),
            "compose": t("Compose images"),
            "upscale": t("Upscale image"),
            "depth-map": t("Create depth map"),
            "restore-seedvr2": t("Restore image"),
            "adjust-image": t("Apply adjustment"),
        }.get(action, t("Run image action"))

    def selected_store(self):
        return self.store or editor_session.default_store()

    def run_version_js(self, destination_index):
        return (
            """(...values) => {
            const destinations = ['create', 'editor', 'library', 'models', 'chat', 'settings'];
            const current = destinations.find(name => document.getElementById(`rf-nav-${name}-button`)?.getAttribute('aria-selected') === 'true');
            if (current) values["""
            + str(int(destination_index))
            + """] = current;
            window.__rfEditorRequest = (window.__rfEditorRequest || 1) + 1;
            values[values.length - 1] = window.__rfEditorRequest;
            return values;
        }"""
        )

    @property
    def reference_outputs(self):
        return [self.controls[name] for name in self.SOURCE_REFERENCES if name in self.controls]

    @classmethod
    def reference_outputs_from(cls, get_control):
        """Resolve the public registry while the app is declaring its Run event."""
        outputs = []
        for name in cls.SOURCE_REFERENCES:
            try:
                outputs.append(get_control(name))
            except (KeyError, ValueError):
                # Optional feature controls need not exist in every composition.
                continue
        return outputs

    def clear_references(self):
        return {
            self.controls[name]: [] if isinstance(value, list) else value
            for name, value in self.SOURCE_REFERENCES.items()
            if name in self.controls
        }

    @property
    def source_outputs(self):
        """Import/history only publish source widgets, never the full run/settings surface."""
        return list(
            dict.fromkeys(
                [
                    self.context,
                    self.source_input,
                    self.notice,
                    self.action,
                    self.mask,
                    *self.reference_outputs,
                    *(
                        self.controls[name]
                        for name in (
                            "image_source",
                            "input_image",
                            "aspect_ratios_selection",
                            "custom_width",
                            "custom_height",
                        )
                        if name in self.controls
                    ),
                    *(
                        self.run_controls[name]
                        for name in ("preview", "native_source", "inpaint_toggle", "inpaint")
                        if name in self.run_controls
                    ),
                    *self.navigation,
                ]
            )
        )

    @property
    def outputs(self):
        return list(
            dict.fromkeys(
                [
                    self.context,
                    self.source_input,
                    self.notice,
                    self.action,
                    self.mask,
                    *self.controls.values(),
                    *self.run_controls.values(),
                    self.latest_generated,
                    *self.navigation,
                ]
            )
        )

    @staticmethod
    def public_context(snapshot, *, additional=()):
        return dict(
            session_id=snapshot.session_id,
            revision=snapshot.revision,
            source_id=snapshot.current.artifact_id,
            additional_image_ids=list(additional),
        )

    def snapshot_updates(self, snapshot, *, additional=(), navigate=False, request=None):
        record = self.selected_store().artifact(
            snapshot.session_id,
            snapshot.current.artifact_id,
            revision=snapshot.revision,
            owner=ui_request_owner(request),
        )
        from ruinedfooocus.features.gallery.actions import source_record

        image, details = working_source(source_record(record.path))
        from ruinedfooocus.features.image_processing.native_image_formats import (
            display_image,
            needs_preview,
        )

        native_source = None
        display = record.path
        if needs_preview(record.path):
            from ruinedfooocus.core.cache_contract import configured_cache_root
            from ruinedfooocus.features.image_processing.native_image_outputs import (
                NativeImageOutput,
                result_presentation,
            )

            display = str(display_image(record.path, configured_cache_root()))
            native_source = result_presentation([NativeImageOutput(record.path, display)])[2][0]
        updates = {
            **self.clear_references(),
            self.context: self.public_context(snapshot, additional=additional),
            self.source_input: "preview",
            self.notice: localize(
                "Current image revision: {revision}",
                resolve_locale(request=request),
                revision=snapshot.revision,
            ),
            self.run_controls["preview"]: gr.update(value=display, visible=True),
            self.run_controls["native_source"]: native_source,
            self.controls["image_source"]: details,
            self.controls["input_image"]: image,
            self.run_controls["inpaint_toggle"]: False,
            self.run_controls["inpaint"]: gr.update(value=None),
            self.mask: None,
        }
        if details["conversion"] != "none":
            updates[self.notice] += " · " + localize(
                "A suitable working image is prepared automatically. The original is kept.",
                resolve_locale(request=request),
            )
        if navigate:
            updates.update(self.navigation)
        return updates

    def image_context(self, context, mode, reference=None, request=None):
        """Chat uses this same P3 session; a new image carries no source context."""
        owner = ui_request_owner(request)
        if mode in ("New image", "new", None):
            return None
        if mode in ("Uploaded image", "uploaded"):
            if reference is None:
                raise ValueError("Upload an image before selecting this source.")
            return self.public_context(self.adopt_image(reference, owner=owner))
        snapshot = (
            self.selected_store().snapshot(context["session_id"], owner=owner) if context else None
        )
        if (
            snapshot is None
            or snapshot.revision != context.get("revision")
            or snapshot.current.artifact_id != context.get("source_id")
        ):
            raise ValueError("Open the current owned image before selecting this source.")
        return dict(context)

    def persist_input(self, value):
        """Detach borrowed UI files from Gradio's reusable, timestamped cache."""
        from PIL import Image

        from ruinedfooocus.core.cache_contract import configured_cache_root

        if isinstance(value, (str, Path)):
            from ruinedfooocus.core.preparation import preparation_service
            from ruinedfooocus.features.image_processing.native_image_formats import MAX_BYTES
            from ruinedfooocus.features.models.model_identity import file_signature

            source = Path(value).resolve()
            with preparation_service.artifact_lease(source):
                signature = tuple(file_signature(source))
                if not source.is_file() or not 0 < source.stat().st_size <= MAX_BYTES:
                    raise ValueError("Choose an existing bounded still-image artifact.")
                target = configured_cache_root() / "editor-inputs" / uuid4().hex / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with source.open("rb") as incoming, target.open("xb") as outgoing:
                        copied = 0
                        while chunk := incoming.read(1024 * 1024):
                            copied += len(chunk)
                            if copied > MAX_BYTES:
                                raise ValueError("The editor input exceeds its bounded size.")
                            outgoing.write(chunk)
                    if tuple(file_signature(source)) != signature:
                        raise ValueError("The editor source changed during capture.")
                except BaseException:
                    target.unlink(missing_ok=True)
                    raise
            return target
        if not isinstance(value, Image.Image):
            raise ValueError("Choose a still image.")
        target = configured_cache_root() / "editor-inputs" / (uuid4().hex + ".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            value.save(target)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return target

    @contextmanager
    def _staged_input(self, value):
        """Retain a fresh input only after its session admission succeeds."""
        path = self.persist_input(value)
        try:
            yield path
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def adopt_image(self, value, *, native_source=None, owner):
        from ruinedfooocus.features.gallery.actions import source_record
        from ruinedfooocus.features.image_processing.native_image_formats import image_metadata

        def adopt(path):
            record = source_record(path)
            saved = image_metadata(path).get("parameters")
            return open_editor(record, saved, owner=owner, store=self.selected_store())

        if native_source is not None:
            from ruinedfooocus.features.gallery.actions import source_path
            from ruinedfooocus.features.image_processing.native_image_outputs import (
                capture_original_source,
            )

            bound = capture_original_source(
                dict(main_view=value, _native_main_source=native_source)
            )
            if isinstance(bound.get("main_view"), (str, Path)):
                raise ValueError("The precision preview changed. Select its original image again.")
            path = source_path(native_source)
            return adopt(path)
        with self._staged_input(value) as path:
            return adopt(path)

    def bind_chat_open(self, button, result):
        """Register at Chat construction; app wiring installs it after native tabs exist."""
        if self.controls:
            self._bind_chat_open(button, result)
        else:
            self.chat_open_events.append((button, result))

    def _bind_chat_open(self, button, result):
        def open_chat(value, task_id, version, request: gr.Request = None):
            return self._source_action(
                task_id, version, request, lambda: self.open_result(value, request), prepare=True
            )

        button.click(
            open_chat,
            inputs=[result, self.run_controls["task"], self.version],
            outputs=self.source_outputs,
            js=self.version_js,
            show_progress="minimal",
            show_progress_on=self.notice,
            api_visibility="private",
        )

    def result_context(self, result, request=None):
        """Validate a typed terminal without publishing or constructing another editor."""
        payload = (result or {}).get("editor") or (result or {}).get("snapshot")
        if not isinstance(payload, dict):
            raise ValueError("This result has no owned image to open.")
        snapshot = self.selected_store().snapshot(
            payload["session_id"], owner=ui_request_owner(request)
        )
        if snapshot.revision != payload.get("revision"):
            raise ValueError("This image result was replaced. Reopen the current history.")
        if (
            payload.get("current", {}).get("artifact_id", snapshot.current.artifact_id)
            != snapshot.current.artifact_id
        ):
            raise ValueError("Select the current owned image result.")
        return self.public_context(snapshot)

    def open_result(self, result, request=None):
        context = self.result_context(result, request)
        snapshot = self.selected_store().snapshot(
            context["session_id"], owner=ui_request_owner(request)
        )
        return self.snapshot_updates(snapshot, navigate=True, request=request)

    def _source_action(self, task_id, version, request, function, *, prepare=False):
        if task_id is not None:
            return {
                self.notice: t(
                    "An image job is still running. Finish it or use Stop, then open this image again."
                )
            }
        token = latest_ui_requests.begin(ui_request_owner(request), self.channel, int(version))
        try:
            if prepare:
                result = function()
                return latest_ui_requests.publish(token, lambda: result)
            return latest_ui_requests.publish(token, function)
        except CancelledError:
            return gr.skip()
        finally:
            latest_ui_requests.finish(token)

    def bind(
        self,
        controls,
        *,
        run_request,
        stop_request,
        run_controls,
        top_tabs=None,
        create_tabs=None,
        navigation=None,
        guide_checkpoint=None,
        guide_control_plan=None,
    ):
        self.controls, self.run_controls = controls, run_controls
        self.run_request, self.stop_request = run_request, stop_request
        self.navigation = navigation or {
            tab: gr.update(selected=value)
            for tab, value in ((top_tabs, "create"), (create_tabs, "generation"))
            if tab is not None
        }

        automatic_size = controls.get("resolution_auto") or gr.State(True)

        def mark_output_explicit():
            return True

        output_inputs = [
            controls[name].input
            for name in (
                "output_format",
                "output_bit_depth",
                "output_color_space",
                "output_crf",
                "_adjustment_hdr_precision",
            )
            if name in controls
        ]
        if output_inputs:
            gr.on(
                triggers=output_inputs,
                fn=mark_output_explicit,
                outputs=self.output_explicit,
                queue=False,
                show_progress="hidden",
                api_visibility="private",
            )

        def open_current(
            path, source, native_source, automatic, task_id, version, request: gr.Request = None
        ):
            selected = path or source
            if selected is None:
                raise gr.Error("Open a source image first.")

            def adopt():
                snapshot = self.adopt_image(
                    selected, native_source=native_source, owner=ui_request_owner(request)
                )
                updates = self.snapshot_updates(snapshot, request=request) | {
                    self.action: "prompt-edit"
                }
                if automatic:
                    updates.update(
                        {
                            controls["aspect_ratios_selection"]: "Custom...",
                            controls["custom_width"]: snapshot.current.frame.width,
                            controls["custom_height"]: snapshot.current.frame.height,
                        }
                    )
                return updates

            return self._source_action(task_id, version, request, adopt, prepare=True)

        self.open_current.click(
            open_current,
            inputs=[
                run_controls["preview"],
                controls["input_image"],
                run_controls["native_source"],
                automatic_size,
                run_controls["task"],
                self.version,
            ],
            outputs=self.source_outputs,
            js=self.version_js,
            show_progress="minimal",
            show_progress_on=self.notice,
            api_visibility="private",
        )

        def open_latest(record, current, task_id, version, request: gr.Request = None):
            from ruinedfooocus.features.image_processing.native_image_outputs import (
                latest_generated_result,
            )

            def adopt():
                chosen, metadata = record, (record or {}).get("metadata")
                if not chosen or latest_generated_result([chosen]) is None:
                    # Nothing finished in this session: an empty Editor opens the
                    # newest Gallery image; an open image is never replaced.
                    chosen = None if current else self._newest_gallery_record(request=request)
                    if chosen is None:
                        return {
                            self.notice: t(
                                "No completed generated image is available yet. Choose an image from Gallery."
                            )
                        }
                    from ruinedfooocus.features.gallery.actions import source_path
                    from ruinedfooocus.features.image_processing.native_image_formats import (
                        image_metadata,
                    )

                    metadata = image_metadata(source_path(chosen)).get("parameters", {})
                snapshot = open_editor(
                    chosen,
                    metadata,
                    owner=ui_request_owner(request),
                    store=self.selected_store(),
                )
                return self.snapshot_updates(snapshot, request=request) | {
                    self.action: "prompt-edit"
                }

            return self._source_action(task_id, version, request, adopt, prepare=True)

        self.open_latest.click(
            open_latest,
            inputs=[self.latest_generated, self.context, run_controls["task"], self.version],
            outputs=self.source_outputs,
            js=self.version_js,
            show_progress="minimal",
            show_progress_on=self.notice,
            api_visibility="private",
        )

        def history(context, task_id, version, request: gr.Request = None, *, operation):
            if not context:
                raise gr.Error("Open an image history first.")
            store, owner = self.selected_store(), ui_request_owner(request)

            def move():
                if operation == "reopen":
                    snapshot = store.reopen(context["session_id"], owner=owner)
                else:
                    snapshot = getattr(store, operation)(
                        context["session_id"], context["revision"], owner=owner
                    )
                return self.snapshot_updates(snapshot, request=request)

            return self._source_action(task_id, version, request, move)

        from functools import partial

        for button, operation in (
            (self.undo, "undo"),
            (self.redo, "redo"),
            (self.reopen, "reopen"),
        ):
            button.click(
                partial(history, operation=operation),
                inputs=[self.context, run_controls["task"], self.version],
                outputs=self.source_outputs,
                js=self.version_js,
                show_progress="minimal",
                show_progress_on=self.notice,
                api_visibility="private",
            )

        def use_result_as_selection(context, task_id, version, request: gr.Request = None):
            if not context:
                raise gr.Error("Open an image history first.")

            def attach():
                snapshot, mask = self.selected_store().use_result_as_mask(
                    context["session_id"],
                    context["revision"],
                    context["source_id"],
                    owner=ui_request_owner(request),
                )
                updates = self.snapshot_updates(snapshot, request=request)
                updates[self.context] = dict(
                    self.public_context(snapshot), mask_id=mask.artifact_id
                )
                updates[self.action] = "inpaint"
                updates[self.notice] = t(
                    "Selection mask attached to its original image. The source pixels and alpha were preserved."
                )
                return updates

            return self._source_action(task_id, version, request, attach)

        if guide_control_plan is None:
            self.use_selection.click(
                use_result_as_selection,
                inputs=[self.context, run_controls["task"], self.version],
                outputs=self.outputs,
                js=self.version_js,
                api_visibility="private",
            )
        else:
            self.guided_results.bind(
                self,
                checkpoint=guide_checkpoint or controls["base_model_name"],
                control_plan=guide_control_plan,
                use_selection=use_result_as_selection,
                create_tabs=create_tabs,
            )

        def invalidate(task_id, version, request: gr.Request = None):
            # Controls are the next draft. An admitted job keeps its captured
            # inputs and terminal ownership until completion or explicit Stop.
            if task_id is not None:
                return int(version)
            token = latest_ui_requests.begin(ui_request_owner(request), self.channel, int(version))
            latest_ui_requests.finish(token)
            return int(version)

        triggers = []
        for control in dict.fromkeys(
            [
                *controls.values(),
                self.context,
                self.action,
                self.quick_tool,
                self.mask,
                self.background,
                self.framing,
                self.invert,
                self.mask_choice,
                self.mask_index,
                self.mask_count,
            ]
        ):
            event = getattr(control, "input", None)
            if event is not None:
                triggers.append(event)
        triggers.append(self.context.change)
        gr.on(
            triggers=triggers,
            fn=invalidate,
            inputs=[run_controls["task"], self.version],
            outputs=self.version,
            js=self.version_js,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

        def preview_trace(stage, **state):
            if os.environ.get("RF_EDITOR_PREVIEW_DEBUG") == "1":
                logging.getLogger(__name__).warning("Editor preview %s %s", stage, state)

        def source_changed(task_id, version, request: gr.Request = None, *, source="preview"):
            invalidate(task_id, version, request)
            if task_id is not None:
                return gr.skip()
            latest_ui_requests.cancel(ui_request_owner(request), self.channel + "-preview")
            preview_trace("source-changed", origin=source, version=version)
            return {
                **self.clear_references(),
                self.context: None,
                self.source_input: source,
                self.preview_signature: None,
                self.mask: None,
                run_controls["inpaint"]: gr.update(value=None),
            }

        def input_source_changed(task_id, version, request: gr.Request = None):
            return source_changed(task_id, version, request, source="input")

        source_events = []
        for triggers, callback in (
            ([run_controls["preview"].upload, run_controls["gallery"].select], source_changed),
            ([controls["input_image"].upload], input_source_changed),
        ):
            source_events.append(
                gr.on(
                    triggers=triggers,
                    fn=callback,
                    inputs=[run_controls["task"], self.version],
                    outputs=[
                        self.context,
                        self.source_input,
                        self.preview_signature,
                        self.mask,
                        run_controls["inpaint"],
                        *self.reference_outputs,
                    ],
                    js=self.version_js,
                    queue=False,
                    show_progress="hidden",
                    api_visibility="private",
                )
            )

        def preview_source(
            path, native, origin, previous, task_id, version, request: gr.Request = None
        ):
            preview_trace(
                "start",
                origin=origin,
                active=task_id is not None,
                path_present=path is not None,
                version=version,
            )
            if task_id is not None or origin == "input":
                return gr.skip()
            token = latest_ui_requests.begin(
                ui_request_owner(request), self.channel + "-preview", int(version)
            )
            try:
                if path is None or origin == "placeholder":
                    return latest_ui_requests.publish(
                        token,
                        lambda: {
                            controls["input_image"]: None,
                            controls["image_source"]: None,
                            self.preview_signature: None,
                        },
                    )
                from ruinedfooocus.features.gallery.actions import source_path, source_record
                from ruinedfooocus.features.models.model_identity import file_signature

                if native is not None:
                    from ruinedfooocus.features.image_processing.native_image_outputs import (
                        capture_original_source,
                    )

                    bound = capture_original_source(
                        dict(main_view=path, _native_main_source=native)
                    )
                    if isinstance(bound.get("main_view"), (str, Path)):
                        raise ValueError("The displayed image no longer matches its original.")
                    path = source_path(native)
                signature = [str(Path(path).resolve()), *file_signature(path)]
                if signature == previous:
                    preview_trace("unchanged", version=version)
                    return gr.skip()
                image, details = working_source(source_record(path))
                if signature != [str(Path(path).resolve()), *file_signature(path)]:
                    raise ValueError("The source image changed while preparing its preview.")
                updates = latest_ui_requests.publish(
                    token,
                    lambda: {
                        controls["input_image"]: image,
                        controls["image_source"]: details,
                        self.preview_signature: signature,
                    },
                )
                preview_trace("published", version=version, width=image.width, height=image.height)
                return updates
            except CancelledError:
                preview_trace("superseded", version=version)
                return gr.skip()
            except (ValueError, OSError) as error:
                logging.getLogger(__name__).warning(
                    "Editor input preview could not be prepared: %s", error
                )
                try:
                    return latest_ui_requests.publish(
                        token,
                        lambda: {
                            controls["input_image"]: None,
                            controls["image_source"]: None,
                            self.preview_signature: None,
                        },
                    )
                except CancelledError:
                    return gr.skip()
            finally:
                latest_ui_requests.finish(token)

        gr.on(
            triggers=[
                run_controls["preview"].change,
                run_controls["native_source"].change,
                self.source_input.change,
            ],
            fn=preview_source,
            inputs=[
                run_controls["preview"],
                run_controls["native_source"],
                self.source_input,
                self.preview_signature,
                run_controls["task"],
                self.version,
            ],
            outputs=[controls["input_image"], controls["image_source"], self.preview_signature],
            js=self.version_js,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        for event in source_events:
            event.success(
                preview_source,
                inputs=[
                    run_controls["preview"],
                    run_controls["native_source"],
                    self.source_input,
                    self.preview_signature,
                    run_controls["task"],
                    self.version,
                ],
                outputs=[controls["input_image"], controls["image_source"], self.preview_signature],
                js=self.version_js,
                queue=False,
                show_progress="hidden",
                api_visibility="private",
            )

        def action_context(action, destination, tool, mask_choice):
            masking = action == "inpaint"
            background = action in ("select-background", "compose")
            framing = action in (
                "prompt-edit",
                "inpaint",
                "generate-background",
                "upscale",
                "compose",
            )
            return {
                run_controls["run"]: gr.update(
                    value=(
                        t(tool)
                        if destination == "editor"
                        and action == "generate"
                        and tool in {value for _, value in self.quick_tool.choices}
                        else self.action_label(action, destination)
                    )
                ),
                self.action_inputs: gr.update(visible=masking or background or framing, open=True),
                self.mask: gr.update(visible=masking),
                self.mask_choice: gr.update(visible=masking),
                self.mask_index: gr.update(visible=masking and mask_choice == "instance"),
                self.mask_count: gr.update(visible=masking and mask_choice == "instance"),
                self.invert: gr.update(visible=masking),
                self.background: gr.update(visible=background),
                self.framing: gr.update(visible=framing),
                self.denoise: gr.update(visible=action == "prompt-edit"),
            }

        gr.on(
            triggers=[
                self.action.change,
                self.destination.change,
                self.mask_choice.change,
                *([controls["cn_selection"].change] if "cn_selection" in controls else []),
            ],
            fn=action_context,
            inputs=[
                self.action,
                self.destination,
                controls.get("cn_selection") or gr.State(None),
                self.mask_choice,
            ],
            outputs=[
                run_controls["run"],
                self.action_inputs,
                self.mask,
                self.mask_choice,
                self.mask_index,
                self.mask_count,
                self.invert,
                self.background,
                self.framing,
                self.denoise,
            ],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

        self.action.input(
            lambda action: action == "inpaint",
            inputs=self.action,
            outputs=run_controls["inpaint_toggle"],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

        def adjustment_action(action, current_type):
            utility = {
                "depth-map": "Depth map",
                "restore-seedvr2": "Restore still image (SeedVR2)",
                "remove-background": "RemBG",
            }.get(action)
            if utility is not None:
                return [utility, gr.skip()] + (
                    [gr.update(selected="image-tools")] if create_tabs is not None else []
                )
            if action != "adjust-image":
                return [gr.skip() for _ in adjustment_outputs]
            from ruinedfooocus.features.image_processing.still_image_utilities import (
                ADJUSTMENT_LABEL,
                ADJUSTMENT_KINDS,
                adjustment_kind,
            )

            kind = adjustment_kind(ADJUSTMENT_LABEL, current_type)
            choices = list(controls["cn_type"].choices)
            known = {value for _, value in choices}
            choices += [
                (value.capitalize(), value.capitalize())
                for value in ADJUSTMENT_KINDS
                if value.capitalize() not in known
            ]
            return [ADJUSTMENT_LABEL, gr.update(value=kind.capitalize(), choices=choices)] + (
                [gr.update(selected="image-tools")] if create_tabs is not None else []
            )

        if controls.get("cn_selection") is not None and controls.get("cn_type") is not None:
            adjustment_outputs = [controls["cn_selection"], controls["cn_type"]]
            if create_tabs is not None:
                adjustment_outputs.append(create_tabs)
            self.action.change(
                adjustment_action,
                inputs=[self.action, controls["cn_type"]],
                outputs=adjustment_outputs,
                queue=False,
                api_visibility="private",
                show_progress="hidden",
                show_progress_on=[],
            )

        legacy_tool = controls.get("cn_selection")
        if legacy_tool is not None:
            installed = {
                choice[1] if isinstance(choice, (tuple, list)) else choice
                for choice in (legacy_tool.choices or [])
            }
            quick_outputs = [legacy_tool, self.action, run_controls["inpaint_toggle"]]
            if create_tabs is not None:
                quick_outputs.append(create_tabs)

            def select_quick_tool(choice):
                if choice not in installed:
                    raise gr.Error("Choose an available image tool from Image tools.")
                return [choice, "generate", False] + (
                    [gr.update(selected="image-tools")] if create_tabs is not None else []
                )

            self.quick_tool.input(
                # Native change listeners reveal controls without replacing saved values.
                select_quick_tool,
                inputs=self.quick_tool,
                outputs=quick_outputs,
                queue=False,
                show_progress="hidden",
                show_progress_on=[],
                api_visibility="private",
            )
            quick_values = [choice[1] for choice in self.quick_tool.choices]
            legacy_tool.change(
                lambda choice: choice if choice in quick_values else None,
                inputs=legacy_tool,
                outputs=self.quick_tool,
                queue=False,
                show_progress="hidden",
                api_visibility="private",
            )
            legacy_tool.input(
                lambda choice: ("adjust-image" if choice == "Adjust image" else "generate", False),
                inputs=legacy_tool,
                outputs=[self.action, run_controls["inpaint_toggle"]],
                queue=False,
                show_progress="hidden",
                api_visibility="private",
            )

        def adopt_mask(path, context, request: gr.Request = None):
            if not path:
                return context
            if not context:
                # Run owns source and mask admission together for a fresh upload.
                return None
            store, owner = self.selected_store(), ui_request_owner(request)
            store.artifact(
                context["session_id"],
                context["source_id"],
                revision=context["revision"],
                owner=owner,
            )
            with self._staged_input(path) as owned_path:
                mask = store.bind_mask(
                    context["session_id"],
                    owned_path,
                    context["source_id"],
                    owner=owner,
                )
            return dict(context, mask_id=mask.artifact_id)

        self.mask.upload(
            adopt_mask,
            inputs=[self.mask, self.context],
            outputs=self.context,
            api_visibility="private",
        )

        def preview_readiness(*values, request: gr.Request = None):
            return self.readiness(dict(zip(self.controls, values)), request=request)

        self.preview.click(
            preview_readiness,
            inputs=list(controls.values()),
            outputs=self.notice,
            show_progress="minimal",
            api_visibility="private",
        )
        for button, result in self.chat_open_events:
            self._bind_chat_open(button, result)

    def _newest_gallery_record(self, *, request=None):
        """Prepare the existing cold index, then read its newest owned candidate."""
        browser = getattr(self, "_picker_browser", None)
        if browser is None:
            return None
        ticket = latest_ui_requests.peek(ui_request_owner(request), self.channel)
        locale = resolve_locale(request=request)
        try:
            browser.update_images(
                locale=locale,
                only_if_uninitialized=True,
                cancelled=ticket.cancelled,
                publish=lambda commit: latest_ui_requests.publish(ticket, commit),
            )
            records = browser.load_images(
                1,
                include_paths=True,
                include_page=True,
                page_size=1,
                locale=locale,
                cancelled=ticket.cancelled,
            )[2]
        except (OSError, ValueError):
            return None
        return dict(records[0]) if records else None

    def bind_picker(self, browser):
        """A bounded in-Editor view over the existing index, with its own browse fence."""
        self._picker_browser = browser
        import asyncio
        from ruinedfooocus.app.preparation_admission import preparation_admission
        from ruinedfooocus.features.gallery.actions import select_source, source_path

        descriptor = gr.JSON({}, visible=False)
        channel = self.channel + "-picker"
        begin_js = "(...v) => { window.__rfEditorPicker=(window.__rfEditorPicker||0)+1; v[v.length-1]=window.__rfEditorPicker; return v; }"
        page_outputs = [
            self.picker_gallery,
            self.picker_records,
            self.picker_selected,
            self.picker_page,
            self.picker_total,
            self.picker_previous,
            self.picker_next,
            self.picker_open,
            self.picker_notice,
        ]

        def page_request(page, version, request: gr.Request = None, *, delta=0, first=False):
            ticket = latest_ui_requests.begin(ui_request_owner(request), channel, int(version))
            try:
                page = int(page or 1)
            except (TypeError, ValueError, OverflowError):
                page = 1
            return {
                self.picker: gr.update(visible=True),
                descriptor: {
                    "version": ticket.version,
                    "page": 1 if first else max(1, page + delta),
                },
            }

        from functools import partial

        for trigger, callback in (
            (self.choose_gallery.click, partial(page_request, first=True)),
            (self.picker_page.input, page_request),
            (self.picker_previous.click, partial(page_request, delta=-1)),
            (self.picker_next.click, partial(page_request, delta=1)),
        ):
            trigger(
                callback,
                inputs=[self.picker_page, self.picker_version],
                outputs=[self.picker, descriptor],
                js=begin_js,
                queue=False,
                show_progress="hidden",
                api_visibility="private",
            )

        async def prepare_picker(value, request: gr.Request = None, progress=gr.Progress()):
            ticket = None
            try:
                ticket = latest_ui_requests.get(
                    ui_request_owner(request), channel, value["version"], start=True
                )
                locale = resolve_locale(request=request)

                def work():
                    status = browser.update_images(
                        locale=locale,
                        only_if_uninitialized=True,
                        cancelled=ticket.cancelled,
                        progress=lambda count, parsed, reused: progress(
                            None,
                            desc=localize(
                                "Scanning Library: {count} files ({reused} unchanged)",
                                locale,
                                count=count,
                                reused=reused,
                            ),
                        ),
                        publish=lambda commit: latest_ui_requests.publish(ticket, commit),
                    )
                    images, text, records, page = browser.load_images(
                        value["page"],
                        include_paths=True,
                        include_page=True,
                        page_size=8,
                        locale=locale,
                        cancelled=ticket.cancelled,
                    )
                    return images, f"{text}\n\n{status}" if status else text, records, page

                future = preparation_admission.submit_work(
                    work,
                    cancelled=ticket.cancelled,
                )
                latest_ui_requests.attach(ticket, future)
                images, text, records, page = await asyncio.wrap_future(future)
                result = {
                    self.picker_gallery: gr.Gallery(
                        value=images, selected_index=None, preview=False, render=False
                    ),
                    self.picker_records: [
                        dict(record, _picker_version=ticket.version) for record in records
                    ],
                    self.picker_selected: None,
                    self.picker_page: gr.update(value=page["page"], maximum=page["pages"]),
                    self.picker_total: f"{page['page']} / {page['pages']}",
                    self.picker_previous: gr.update(interactive=page["page"] > 1),
                    self.picker_next: gr.update(interactive=page["page"] < page["pages"]),
                    self.picker_open: gr.update(interactive=False),
                    self.picker_notice: text,
                }
                return latest_ui_requests.publish(ticket, lambda: result)
            except (CancelledError, asyncio.CancelledError):
                return gr.skip()
            except (ValueError, OSError, KeyError, TypeError) as error:
                if ticket is None:
                    return gr.skip()
                message = str(error)
                try:
                    return latest_ui_requests.publish(
                        ticket,
                        lambda: {
                            self.picker_notice: message,
                            self.picker_open: gr.update(interactive=False),
                            self.picker_records: [],
                            self.picker_selected: None,
                        },
                    )
                except CancelledError:
                    return gr.skip()
            finally:
                if ticket is not None:
                    latest_ui_requests.finish(ticket)

        descriptor.change(
            prepare_picker,
            inputs=descriptor,
            outputs=page_outputs,
            concurrency_limit=2,
            trigger_mode="always_last",
            show_progress="minimal",
            show_progress_on=self.picker_notice,
            api_visibility="private",
        )

        def validate(record, request):
            if not isinstance(record, dict):
                raise gr.Error("Choose an image in this Gallery picker first.")
            try:
                ticket = latest_ui_requests.get(
                    ui_request_owner(request), channel, record.get("_picker_version")
                )
                source_path(record)
                return ticket
            except (CancelledError, ValueError, OSError):
                raise gr.Error("The image list changed. Choose the image again.") from None

        def pick(records, evt: gr.SelectData, request: gr.Request = None):
            try:
                record = select_source(records, evt.index, evt.value.get("caption"))
                ticket = validate(record, request)
                return latest_ui_requests.publish(
                    ticket,
                    lambda: {
                        self.picker_selected: record,
                        self.picker_notice: Path(record["path"]).name,
                        self.picker_open: gr.update(interactive=True),
                    },
                )
            except (CancelledError, ValueError, OSError, AttributeError, gr.Error) as error:
                return {
                    self.picker_selected: None,
                    self.picker_notice: str(error),
                    self.picker_open: gr.update(interactive=False),
                }

        picked = self.picker_gallery.select(
            pick,
            inputs=self.picker_records,
            outputs=[self.picker_selected, self.picker_notice, self.picker_open],
            show_progress="hidden",
            api_visibility="private",
        )

        def open_pick(record, task_id, version, request: gr.Request = None):
            def adopt():
                ticket = validate(record, request)
                from ruinedfooocus.features.image_processing.native_image_formats import (
                    image_metadata,
                )

                metadata = image_metadata(source_path(record)).get("parameters", {})

                def publish():
                    snapshot = open_editor(
                        record,
                        metadata,
                        owner=ui_request_owner(request),
                        store=self.selected_store(),
                    )
                    return self.snapshot_updates(snapshot, request=request) | {
                        self.action: "prompt-edit",
                        self.picker: gr.update(visible=False),
                    }

                updates = publish()
                return latest_ui_requests.publish(ticket, lambda: updates)

            return self._source_action(task_id, version, request, adopt, prepare=True)

        def open_after_pick(record, task_id, version, request: gr.Request = None):
            if record is None:
                return gr.skip()
            return open_pick(record, task_id, version, request)

        # Choosing a tile is the whole action; no second confirmation click.
        picked.success(
            open_after_pick,
            inputs=[self.picker_selected, self.run_controls["task"], self.version],
            outputs=[*self.source_outputs, self.picker],
            js=self.version_js,
            show_progress="minimal",
            show_progress_on=self.notice,
            api_visibility="private",
        )
        self.picker_open.click(
            open_pick,
            inputs=[self.picker_selected, self.run_controls["task"], self.version],
            outputs=[*self.source_outputs, self.picker],
            js=self.version_js,
            show_progress="minimal",
            show_progress_on=self.notice,
            api_visibility="private",
        )

        def close_picker(request: gr.Request = None):
            latest_ui_requests.cancel(ui_request_owner(request), channel)
            return gr.update(visible=False), None

        self.picker_close.click(
            close_picker,
            outputs=[self.picker, self.picker_selected],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    def bind_library(self, button, selected, metadata, prompt_version, *, validate_record):
        from ruinedfooocus.features.prompting.ui_draft_fence import capture_draft, publish_draft

        size_auto = self.controls.get("resolution_auto") or gr.State(True)
        inputs = [
            selected,
            metadata,
            prompt_version,
            self.controls["prompt"],
            self.controls["negative"],
            self.controls["base_model_name"],
            self.controls["aspect_ratios_selection"],
            self.controls["custom_width"],
            self.controls["custom_height"],
            size_auto,
            self.run_controls["task"],
            self.version,
        ]
        result = gr.JSON(None, visible="hidden")
        guard, capture = capture_draft(self.controls["prompt"], inputs, f"editor-open-{result._id}")
        begin = (
            "(...values) => { window.__rfEditorRequest=(window.__rfEditorRequest||1)+1; values[values.length-2]=window.__rfEditorRequest; return ("
            + capture
            + ")(...values); }"
        )

        def open_library(
            record,
            saved,
            version,
            prompt,
            negative,
            model,
            ratio,
            width,
            height,
            automatic,
            task_id,
            publication_version,
            draft,
            request: gr.Request = None,
        ):
            if task_id is not None:
                return {
                    self.notice: t(
                        "An image job is still running. Finish it or use Stop, then open this image again."
                    )
                }
            owner = ui_request_owner(request)
            ticket = latest_ui_requests.begin(owner, self.channel, int(publication_version))
            try:
                validate_record(record, request)
                from ruinedfooocus.features.gallery.image_metadata import read_metadata

                data, warning = read_metadata(saved) or {}, ""
                original = version == "Original prompt"
                selected_prompt = data.get("raw_prompt" if original else "prompt")
                selected_negative = data.get("raw_negative" if original else "negative")
                metadata_updates = [
                    selected_prompt if isinstance(selected_prompt, str) else gr.skip(),
                    selected_negative if isinstance(selected_negative, str) else gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                ]
                if data.get("base_model_name"):
                    from ruinedfooocus.app import state as shared
                    from ruinedfooocus.features.gallery.actions import resolve_resource
                    from ruinedfooocus.features.models.selection import checkpoint_choices

                    try:
                        restored = resolve_resource(
                            shared.models,
                            "checkpoints",
                            data["base_model_name"],
                            data.get("base_model_hash"),
                        )
                        metadata_updates[2] = gr.update(
                            choices=checkpoint_choices(
                                shared.models.get_names("checkpoints"),
                                restored,
                                lambda name: shared.models.get_models_by_path(
                                    "checkpoints", name, fetch=False
                                ),
                            ),
                            value=restored,
                        )
                    except ValueError as error:
                        warning = str(error)
                validate_record(record, request)

                def adopt():
                    snapshot = open_editor(record, saved, owner=owner, store=self.selected_store())
                    if automatic:
                        metadata_updates[3:] = [
                            "Custom...",
                            snapshot.current.frame.width,
                            snapshot.current.frame.height,
                        ]
                    updates = self.snapshot_updates(snapshot, navigate=True, request=request)
                    if warning:
                        updates[self.notice] += "\n\n" + warning
                    updates[self.action] = "prompt-edit"
                    updates[result] = dict(
                        guard=draft,
                        updates=metadata_updates,
                        stale=None,
                        message=t(
                            "The Create draft changed while this image loaded. Your newer values were kept."
                        ),
                    )
                    return updates

                updates = adopt()
                validate_record(record, request)
                return latest_ui_requests.publish(ticket, lambda: updates)
            except CancelledError:
                return gr.skip()
            finally:
                latest_ui_requests.finish(ticket)

        event = button.click(
            open_library,
            inputs=[*inputs, guard],
            outputs=[*self.source_outputs, result],
            js=begin,
            show_progress="minimal",
            show_progress_on=self.notice,
            api_visibility="private",
        )
        publish_draft(
            event,
            result,
            prompt=self.controls["prompt"],
            outputs=[
                self.controls[name]
                for name in (
                    "prompt",
                    "negative",
                    "base_model_name",
                    "aspect_ratios_selection",
                    "custom_width",
                    "custom_height",
                )
            ],
        )

    def readiness(self, values, *, request=None):
        from ruinedfooocus.app import async_worker as worker
        from ruinedfooocus.features.generation.image_preparation import image_readiness

        data = self.build_request(values, request=request)
        try:
            if "_editor_selected_background" in data:
                return t("Ready: the selected background uses existing owned pixels.")
            data = worker.capture_image_data(data)
            readiness = image_readiness(data)
            return "\n\n".join(
                [
                    readiness.state,
                    *(
                        row.role + ": " + row.name + " · " + row.status
                        for row in readiness.requirements
                    ),
                    *readiness.reasons,
                ]
            )
        finally:
            editor_session.cancel_request(
                data, owner=ui_request_owner(request), store=self.selected_store()
            )

    def run(self, values, version):
        """Fence the existing stream; only a successful current terminal adopts pixels."""
        owner = ui_request_owner()
        latest_ui_requests.cancel(owner, self.channel + "-preview")
        ticket = latest_ui_requests.begin(owner, self.channel, int(version))
        task_id = None
        try:
            data = latest_ui_requests.publish(ticket, lambda: self.build_request(values))
            adopted = None
            if not values.get("_editor_context") and data.get("_editor_context"):
                adopted = self.selected_store().snapshot(
                    data["_editor_context"]["session_id"], owner=owner
                )
            selection = data.pop("_editor_selected_background", None)
            if selection is not None:
                from ruinedfooocus.features.image_editing.image_workflows import selected_background

                path, execution = selected_background(selection, self.selected_store(), owner=owner)
                snapshot = latest_ui_requests.publish(
                    ticket,
                    lambda: self.selected_store().complete(
                        selection.ticket,
                        path,
                        execution=execution,
                        owner=owner,
                        check_cancelled=ticket.cancelled.is_set,
                    ),
                )
                yield self.snapshot_updates(
                    snapshot, additional=[selection.ticket.source.artifact_id]
                )
                return
            with self._request_stream(data) as stream:
                for updates in stream:
                    updates = dict(updates)
                    if task_id is None and values.get("main_view") is not None:
                        updates.pop(self.run_controls["preview"], None)
                    observed_task = updates.get(self.run_controls["task"])
                    if observed_task is not None:
                        task_id = observed_task
                    if adopted is not None and task_id is not None:
                        source_updates = self.snapshot_updates(adopted)
                        # An uploaded/painted mask and manual frame belong to this
                        # captured request. Source admission must not reset them.
                        for control in (
                            self.mask,
                            self.run_controls["inpaint"],
                            self.run_controls["inpaint_toggle"],
                        ):
                            source_updates.pop(control, None)
                        updates = source_updates | updates
                        adopted = None
                    if not latest_ui_requests.current(ticket):
                        if task_id is not None:
                            self.stop_request(task_id)
                        yield {
                            self.notice: t(
                                "The image or settings changed. The older result was not applied."
                            ),
                            self.run_controls["run"]: gr.update(interactive=True, visible=True),
                            self.run_controls["stop"]: gr.update(visible=False),
                            self.run_controls["progress"]: gr.update(visible=False),
                            self.run_controls["task"]: None,
                        }
                        return
                    records = updates.get(self.run_controls["artifacts"])
                    terminal = None
                    if self.run_controls["artifacts"] in updates and task_id is not None:
                        from ruinedfooocus.app import async_worker as worker

                        terminal = worker.editor_result(task_id)
                    if self.run_controls["artifacts"] in updates and not records:
                        updates = {
                            control: update
                            for control, update in updates.items()
                            if control
                            in [
                                self.run_controls[name]
                                for name in ("run", "stop", "progress", "task", "hint")
                            ]
                        }
                        if values.get("main_view") is not None:
                            updates[self.run_controls["preview"]] = gr.update(
                                value=values["main_view"], visible=True
                            )
                    if records or terminal is not None and terminal.get("status") == "completed":
                        if terminal is not None and terminal.get("status") != "completed":
                            rejected = {
                                self.notice: t(
                                    "The image operation did not complete. Your current image was kept."
                                ),
                                self.run_controls["run"]: gr.update(interactive=True, visible=True),
                                self.run_controls["stop"]: gr.update(visible=False),
                                self.run_controls["progress"]: gr.update(visible=False),
                                self.run_controls["task"]: None,
                            }
                            if values.get("main_view") is not None:
                                rejected[self.run_controls["preview"]] = gr.update(
                                    value=values["main_view"], visible=True
                                )
                            yield rejected
                            continue

                        def adopt_terminal():
                            if terminal:
                                self.result_context(terminal)
                            snapshot = (
                                self.selected_store().snapshot(
                                    terminal["snapshot"]["session_id"], owner=owner
                                )
                                if terminal
                                else self.adopt_image(
                                    records[0].get("display", records[0]["path"]),
                                    native_source=records[0]
                                    if records[0].get("precision")
                                    else None,
                                    owner=owner,
                                )
                            )
                            context = data.get("_editor_context") or {}
                            additional = context.get("additional_image_ids", ())
                            if data.get("_editor_action") == "generate-background":
                                additional = [context["source_id"]]
                            elif data.get("_editor_action") == "compose":
                                additional = []
                            return dict(updates) | self.snapshot_updates(
                                snapshot, additional=additional
                            )

                        updates = latest_ui_requests.publish(ticket, adopt_terminal)
                    yield latest_ui_requests.publish(ticket, lambda: updates)
        except (ValueError, OSError, PermissionError, CancelledError) as error:
            yield {self.notice: str(error)}
        finally:
            latest_ui_requests.finish(ticket)

    @contextmanager
    def _request_stream(self, data):
        stream = self.run_request(data)
        closing_exit = False
        try:
            yield stream
        except GeneratorExit:
            closing_exit = True
            raise
        finally:
            try:
                stream.close()
            except Exception:
                if not closing_exit:
                    raise
                logging.getLogger(__name__).warning(
                    "Editor stream cleanup failed while closing", exc_info=True
                )

    def _source_values(self, values, *, request=None):
        """Admit the visible/uploaded still on Run, retaining valid owned context."""
        if values.get("_editor_destination") == "create" or values.get("_editor_context"):
            return values
        uploaded = values.get("_editor_source_input") == "input"
        selected = values.get("input_image") if uploaded else values.get("main_view")
        native = None if uploaded else values.get("_native_main_source")
        if selected is None:
            selected = values.get("input_image")
            native = None
        if values.get("_editor_source_input") == "placeholder":
            selected = None
        if isinstance(selected, (str, Path)):
            from ruinedfooocus.core.paths import asset_path

            if Path(selected).resolve() == asset_path("init_image.png").resolve():
                selected = None
        if selected is None:
            if values.get("_editor_action", "generate") == "generate":
                return values
            raise ValueError("Choose or upload an image to run this action.")
        snapshot = self.adopt_image(selected, native_source=native, owner=ui_request_owner(request))
        result = dict(
            values, **self.SOURCE_REFERENCES, _editor_context=self.public_context(snapshot)
        )
        result["reference_images"] = []
        result.pop("_editor_guide_request", None)
        return result

    def build_request(self, values, *, request=None):
        """Capture one explicit P4 step; the accepted worker owns final P1/P2 binding."""
        from ruinedfooocus.features.image_editing import image_workflows as workflows
        from ruinedfooocus.features.image_editing.mask_handoffs import capture_mask_handoff

        if values.get("_editor_destination") == "create":
            guided = (
                values.get("_editor_guide_handoff") is not None
                or values.get("_editor_guide_request") is not None
            )
            # Project new-image intent without clearing the shared Editor widgets.
            values = dict(
                values,
                _editor_action="generate",
                _editor_context=None,
                editor_context=None,
                input_image=None,
                reference_images=[],
                inpaint_toggle=False,
                inpaint_view=None,
                outpaint_left=0,
                outpaint_top=0,
                outpaint_right=0,
                outpaint_bottom=0,
                qwen_union_guide=None,
                qwen_union_source=None,
                qwen_union_mask=None,
                reference_adapter_image=None,
                model_patch_image=None,
                model_patch_inpaint_image=None,
                model_patch_mask=None,
                control_inpaint_image=None,
                control_inpaint_mask=None,
                cn_selection="None",
                cn_type=None,
                _editor_guide_handoff=None,
            )
            values.pop("_editor_guide_request", None)
            if guided:
                # The map route is transient. Keep the Editor controls intact,
                # but do not carry its selected consumer into a fresh image.
                for name in (
                    "controlnet",
                    "cn_model",
                    "cn_model_signature",
                    "cn_depth_preprocess",
                    "control_recipe",
                    "control_union_type",
                    "control_guide_denoise",
                ):
                    values.pop(name, None)
            if values.get("automatic_upscale_target") == "Input image only":
                values["automatic_upscale_target"] = "Generated image"
        action = values.get("_editor_action", "generate")
        values = self._source_values(values, request=request)
        if action != "generate" and values.get("_editor_guide_handoff") is not None:
            values = dict(values)
            handoff = values.pop("_editor_guide_handoff")
            values.pop("_editor_guide_request", None)
            context = values.get("_editor_context")
            if self.guided_results._matches(handoff, context):
                values["_editor_context"] = dict(context, additional_image_ids=[])
        if action == "generate":
            result = dict(values)
            if values.get("_editor_destination") == "editor" and values.get("_editor_context"):
                result["editor_context"] = values["_editor_context"]
                if values.get("cn_selection") in {value for _, value in self.quick_tool.choices}:
                    from ruinedfooocus.features.gallery.actions import source_record

                    context = values["_editor_context"]
                    source = self.selected_store().artifact(
                        context["session_id"],
                        context["source_id"],
                        revision=context["revision"],
                        owner=ui_request_owner(request),
                    )
                    image, details = working_source(source_record(source.path))
                    result.update(
                        input_image=image,
                        main_view=image.copy(),
                        image_source=details,
                        _native_main_source=None,
                    )
            return self._capture_guided_request(result, request=request)
        context = values.get("_editor_context")
        if not context:
            raise ValueError("Choose or upload an image to run this action.")
        owner, store = ui_request_owner(request), self.selected_store()
        source = store.artifact(
            context["session_id"], context["source_id"], revision=context["revision"], owner=owner
        )
        if action in ("depth-map", "restore-seedvr2", "adjust-image"):
            from ruinedfooocus.features.gallery.actions import source_record
            from ruinedfooocus.features.gallery.editing_workflow import build_library_request
            from ruinedfooocus.features.gallery.ui import REFERENCE_RESETS
            from ruinedfooocus.features.image_editing import ui as tools
            from ruinedfooocus.features.image_processing.still_image_utilities import (
                ADJUSTMENT_KINDS,
            )

            if action == "adjust-image":
                action = tools.tool_kind(values.get("cn_selection"), values.get("cn_type"))
                if action not in ADJUSTMENT_KINDS:
                    raise ValueError("Choose the Adjust image tool and its adjustment group.")

            return build_library_request(
                values,
                source_record(source.path),
                prompt=values.get("prompt", ""),
                negative=values.get("negative", ""),
                checkpoint=values.get("base_model_name"),
                action=action,
                resets=REFERENCE_RESETS,
                features=dict(native_edit=False, inpaint=False, qwen_local=False),
                presets=tools.controlnet.cn_options,
                custom=tools.NEWCN,
                tool_resource=values.get(
                    "native_vision_model" if action == "depth-map" else "seedvr2_model"
                ),
                editor_context=context,
                editor_store=store,
                owner=owner,
            )
        if action not in workflows.ACTIONS:
            raise ValueError("Choose a supported image action.")
        parameters = {
            key: values[key]
            for key in workflows.ACTION_FIELDS[action]
            if key in values and values[key] is not None
        }
        parameters.update(
            image_resize_mode=values.get("_editor_framing", "preserve"), alpha_policy="preserve"
        )
        if action != "select-background":
            width, height = (
                values.get("custom_width", source.frame.width),
                values.get("custom_height", source.frame.height),
            )
            if action in ("remove-background", "compose") or (
                action in ("prompt-edit", "inpaint", "generate-background")
                and values.get("resolution_auto", True)
                and parameters["image_resize_mode"] == "preserve"
            ):
                width, height = source.frame.size
            if int(width) != width or int(height) != height:
                raise ValueError("Choose whole-number output dimensions.")
            parameters.update(target_width=int(width), target_height=int(height))
        family = None
        if action == "prompt-edit":
            from ruinedfooocus.features.image_editing.ui_guidance import family_for_checkpoint

            family = family_for_checkpoint(values.get("base_model_name"))
            parameters["denoise"] = values.get("_editor_denoise", 0.3)
        if action == "upscale":
            upscaler = values.get("automatic_upscale")
            parameters["upscaler"] = (
                "2× General (RealESRGAN)" if not upscaler or upscaler == "Off" else upscaler
            )
            if values.get("resolution_auto", True):
                # Auto size follows the chosen pixel upscaler; shared generation
                # dimensions are not an explicit Editor target.
                from ruinedfooocus.features.generation.automatic_upscale import scale_for

                factor = scale_for(parameters["upscaler"])
                parameters.update(
                    target_width=round(source.frame.width * factor),
                    target_height=round(source.frame.height * factor),
                )
        if action == "select-background":
            path = values.get("_editor_background")
            if not path:
                raise ValueError("Choose the background image for this step.")
            with self._staged_input(path) as owned_path:
                parameters["background_id"] = store.bind_image(
                    context["session_id"], owned_path, owner=owner
                ).artifact_id
        if action == "compose":
            additional = context.get("additional_image_ids") or []
            background = values.get("_editor_background")
            if background:
                with self._staged_input(background) as path:
                    parameters["background_id"] = store.bind_image(
                        context["session_id"], path, owner=owner
                    ).artifact_id
            elif additional:
                parameters["foreground_id"] = additional[0]
            else:
                raise ValueError("Choose a background image to compose with this foreground.")
            parameters.setdefault("composition_mode", "SRC_OVER")
        mask = None
        if action == "inpaint":
            mask_id = context.get("mask_id")
            if not mask_id and values.get("_editor_mask"):
                with self._staged_input(values["_editor_mask"]) as path:
                    mask_id = store.bind_mask(
                        context["session_id"], path, context["source_id"], owner=owner
                    ).artifact_id
            if not mask_id and values.get("inpaint_view") is not None:
                from ruinedfooocus.features.image_editing.inpaint_utils import source_and_mask

                _, selection = source_and_mask(
                    source.path, values["inpaint_view"], preserve_alpha=True
                )
                with self._staged_input(selection) as path:
                    mask_id = store.bind_mask(
                        context["session_id"],
                        path,
                        context["source_id"],
                        owner=owner,
                    ).artifact_id
            if not mask_id:
                raise ValueError(
                    "Choose an exact-frame grayscale selection mask before inpainting."
                )
            choice = values.get("_editor_mask_choice", "union")
            mask = capture_mask_handoff(
                store,
                context["session_id"],
                context["revision"],
                context["source_id"],
                mask_id,
                owner=owner,
                choice=choice,
                instance_count=int(values.get("_editor_mask_count", 1)),
                instance_index=int(values.get("_editor_mask_index", 0))
                if choice == "instance"
                else None,
                invert=values.get("_editor_mask_invert", False),
            )
        step = workflows.capture_step(
            store,
            context["session_id"],
            context["revision"],
            action,
            parameters,
            owner=owner,
            mask=mask,
            family=family,
        )
        if action == "select-background":
            return {"_editor_selected_background": step, editor_session.KEY: step.ticket}
        data = dict(values) | workflows.worker_request(step, store, owner=owner)
        data.update(
            preset_selection=None,
            _metadata_prompt=True,
            _apply_prompt_processing=bool(values.get("_editor_prompt_processing")),
            obp_assume_direct_control=False,
            obp_image_prompt=False,
            cn_selection="None",
            cn_type=None,
            _image_guidance_checkpoint=values.get("base_model_name"),
        )
        if not values.get("_editor_prompt_processing"):
            data.update(style_selection=[], auto_negative=False)
        return data

    def _capture_guided_request(self, data, *, request=None):
        """Freeze owned original and guide before worker/ticket/plan admission."""
        from ruinedfooocus.features.image_editing import guide_handoffs as guides

        handoff = data.get(guides.HANDOFF_KEY)
        if handoff is None:
            return data
        if not self.guided_results._matches(handoff, data.get("editor_context")):
            raise ValueError(
                "The original image or its editing guide changed. Adopt the guide again."
            )
        from ruinedfooocus.app import state as shared
        from ruinedfooocus.features.image_editing import controlnet
        from ruinedfooocus.features.image_editing.image_guidance import effective_preset_checkpoint
        from ruinedfooocus.features.image_editing.ui_guidance import family_for_checkpoint

        checkpoint = effective_preset_checkpoint(
            data.get("base_model_name"),
            data.get("preset_selection"),
            shared.settings.default_settings.get("preset_mode", 7),
        )
        if checkpoint is None:
            raise ValueError("Select a current compatible checkpoint before controlled editing.")
        values = dict(data, controlnet=controlnet.get_settings(data))
        return guides.capture_request(
            values,
            store=self.selected_store(),
            owner=ui_request_owner(request),
            manager=shared.path_manager,
            family=family_for_checkpoint(checkpoint),
        )
