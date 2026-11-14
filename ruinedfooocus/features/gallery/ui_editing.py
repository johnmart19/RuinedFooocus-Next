"""Direct Library edits with the same Create submission and owned result stream."""

import asyncio
from concurrent.futures import CancelledError
from contextlib import closing
from copy import deepcopy
import secrets
from uuid import uuid4

import gradio as gr

from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui as t
from ruinedfooocus.features.gallery.actions import resolve_resource, source_path
from ruinedfooocus.features.gallery.editing_workflow import build_library_request, working_source
from ruinedfooocus.features.gallery.image_metadata import read_metadata

OPERATIONS = (
    ("New image from prompt", "generate"),
    ("Edit selected image", "edit"),
    ("Inpaint selected image", "inpaint"),
    ("Upscale selected image", "upscale"),
    ("Remove background", "rembg"),
    ("Depth map", "depth-map"),
    ("Restore still image (SeedVR2)", "restore-seedvr2"),
)


def same_source(left, right):
    return (
        isinstance(left, dict)
        and isinstance(right, dict)
        and left.get("path") == right.get("path")
        and left.get("signature") == right.get("signature")
    )


class LibraryEditor:
    def __init__(self):
        from ruinedfooocus.app import state as shared
        from ruinedfooocus.features.image_editing.qwen_image21_local_edit import PAINT_COLOR
        from ruinedfooocus.features.models.selection import checkpoint_choices

        def current_value(name, default):
            try:
                return shared.get_ctrl(name).value
            except (ValueError, IndexError, KeyError, AttributeError):
                return shared.settings.default_settings.get(name, default)

        self.source = gr.State(None)
        self.task = gr.State(None)
        self.setup = gr.State({})
        self.setup_choice = gr.Textbox("", visible=False)
        self.terminal = gr.State(None)
        with gr.Accordion(t("Gallery draft"), open=True):
            self.notice = gr.Markdown(t("Select an image to start a Library draft."))
            self.preview = gr.Image(
                label=t("Selected original"),
                interactive=False,
                type="pil",
                image_mode="RGBA",
                buttons=["fullscreen"],
                height="clamp(12rem, 28dvh, 24rem)",
            )
            self.prompt = gr.Textbox(label=t("Prompt"), lines=3)
            self.negative = gr.Textbox(label=t("Negative prompt"), lines=2)
            self.prompt_processing = gr.Checkbox(
                label=t("Apply current styles and negative processing"),
                value=False,
                info=t(
                    "Use selected styles and automatic negative prompt. Random prompt tools stay off for this instruction."
                ),
            )
            current = current_value("base_model_name", "")
            self.model = gr.Dropdown(
                label=t("Image model"),
                choices=checkpoint_choices(shared.models.get_names("checkpoints"), current),
                value=current or None,
            )
            self.operation = gr.Dropdown(
                label=t("Action"),
                choices=[(t(label), value) for label, value in OPERATIONS],
                value="edit",
            )
            self.resource = gr.Dropdown(
                label=t("Tool model"), choices=[], value=None, visible=False
            )
            from ruinedfooocus.features.image_processing.native_seedvr2 import COLORS

            current_codec = current_value("seedvr2_vae", None)
            with gr.Group(visible=False) as self.seedvr2_controls:
                self.seedvr2_vae = gr.Dropdown(
                    label=t("SeedVR2 codec"),
                    choices=[current_codec] if current_codec else [],
                    value=current_codec,
                )
                self.seedvr2_scale = gr.Slider(
                    label=t("Restoration scale"),
                    minimum=1,
                    maximum=4,
                    step=0.25,
                    value=current_value("seedvr2_scale", 4),
                )
                self.seedvr2_color = gr.Dropdown(
                    label=t("Restoration color correction"),
                    choices=list(COLORS),
                    value=current_value("seedvr2_color", "none"),
                )
            self.denoise = gr.Slider(
                label=t("Denoise"),
                minimum=0.01,
                maximum=1,
                step=0.01,
                value=0.3,
            )
            with gr.Row():
                self.width = gr.Number(
                    label=t("Width"),
                    value=current_value("custom_width", 1024),
                    minimum=1,
                    maximum=8192,
                    precision=0,
                )
                self.height = gr.Number(
                    label=t("Height"),
                    value=current_value("custom_height", 1024),
                    minimum=1,
                    maximum=8192,
                    precision=0,
                )
            self.mask = gr.ImageEditor(
                label=t("Paint the area to change"),
                type="numpy",
                format="png",
                sources=[],
                layers=False,
                transforms=(),
                visible=False,
                brush=gr.Brush(colors=[PAINT_COLOR], color_mode="fixed"),
                height="clamp(14rem, 32dvh, 28rem)",
            )
            gr.Markdown(
                t(
                    "Run uses the current Create sampling, upscale, component and LoRA settings. "
                    "Uploaded references are required again. Any working image is prepared automatically; the original is kept."
                )
            )
            with gr.Row(elem_classes=["rf-actions", "rf-actions-compact"]):
                self.run = gr.Button(t("Run"), variant="primary", interactive=False)
                self.stop = gr.Button(t("Stop"), variant="stop", visible=False)
                self.clear_setup = gr.Button(t("Use current sampling settings"), size="sm")
                self.open_result = gr.Button(t("Open result in Editor"), interactive=False)
            self.progress = gr.HTML(visible=False)
            self.result = gr.Image(
                label=t("New result"), interactive=False, buttons=["fullscreen"], visible=False
            )
            self.downloads = gr.File(
                label=t("Original image files"),
                file_count="multiple",
                interactive=False,
                visible=False,
            )

    def bind(
        self,
        selected,
        metadata,
        prompt_version,
        *,
        validate_record,
        resets,
        run_request,
        stop_request,
        run_controls,
        prompt_button=None,
        setup_button=None,
        setup_values=None,
    ):
        from ruinedfooocus.app import state as shared
        from ruinedfooocus.app.preparation_admission import preparation_admission
        from ruinedfooocus.features.image_editing import ui as tools
        from ruinedfooocus.features.image_editing.inpaint_utils import editor_payload
        from ruinedfooocus.features.prompting.ui_draft_fence import (
            capture_draft,
            publish_draft,
        )

        if prompt_button is not None:
            prompt_result = gr.JSON(None, visible="hidden")
            prompt_inputs = [
                selected,
                metadata,
                prompt_version,
                self.prompt,
                self.negative,
                self.model,
                self.width,
                self.height,
            ]
            prompt_guard, prompt_begin = capture_draft(
                self.prompt, prompt_inputs, f"gallery-prompt-{prompt_result._id}"
            )

            def use_local_prompt(
                record,
                saved,
                version,
                current_prompt,
                current_negative,
                model,
                width,
                height,
                guard,
                request: gr.Request = None,
            ):
                validate_record(record, request)
                source_path(record)
                data = read_metadata(saved) or {}
                original = version == "Original prompt"
                prompt = data.get("raw_prompt" if original else "prompt")
                negative = data.get("raw_negative" if original else "negative", "")
                if not isinstance(prompt, str) or not isinstance(negative, str):
                    raise gr.Error("Choose an image with a saved text prompt.")
                return dict(
                    guard=guard,
                    updates=[prompt, negative],
                    stale=None,
                    message=t(
                        "The Gallery draft changed while this prompt loaded. Your newer values were kept."
                    ),
                )

            event = prompt_button.click(
                use_local_prompt,
                inputs=[*prompt_inputs, prompt_guard],
                outputs=prompt_result,
                js=prompt_begin,
                show_progress="hidden",
                api_visibility="private",
            )
            publish_draft(
                event, prompt_result, prompt=self.prompt, outputs=[self.prompt, self.negative]
            )

        if setup_button is not None and setup_values is not None:
            setup_result = gr.JSON(None, visible="hidden")
            setup_inputs = [
                selected,
                metadata,
                self.setup,
                self.setup_choice,
                self.prompt,
                self.negative,
                self.model,
                self.width,
                self.height,
                self.denoise,
            ]
            setup_guard, setup_begin = capture_draft(
                self.prompt, setup_inputs, f"gallery-setup-{setup_result._id}"
            )

            def use_local_setup(
                record,
                saved,
                previous,
                choice,
                prompt,
                negative,
                model,
                width,
                height,
                denoise,
                guard,
                request: gr.Request = None,
            ):
                validate_record(record, request)
                source_path(record)
                updates = setup_values(saved)
                values = {
                    name: updates[control]["value"]
                    for name, control in zip(shared.state["ctrls_name"], shared.state["ctrls_obj"])
                    if control in updates
                    and isinstance(updates[control], dict)
                    and "value" in updates[control]
                }
                validate_record(record, request)
                token = uuid4().hex
                snapshots = dict(previous or {})
                snapshots[token] = dict(source=deepcopy(record), values=values)
                while len(snapshots) > 8:
                    expired = next(key for key in snapshots if key not in (choice, token))
                    snapshots.pop(expired)
                return {
                    self.setup: snapshots,
                    setup_result: dict(
                        guard=guard,
                        updates=[
                            *(
                                values.get(name, gr.skip())
                                for name in (
                                    "prompt",
                                    "negative",
                                    "base_model_name",
                                    "custom_width",
                                    "custom_height",
                                    "cn_strength",
                                )
                            ),
                            token,
                            t("Saved setup loaded into this Gallery draft. Create was kept."),
                        ],
                        stale=None,
                        message=t(
                            "The Gallery draft changed while this setup loaded. Your newer values were kept."
                        ),
                    ),
                }

            event = setup_button.click(
                use_local_setup,
                inputs=[*setup_inputs, setup_guard],
                outputs=[self.setup, setup_result],
                js=setup_begin,
                show_progress="minimal",
                show_progress_on=self.notice,
                api_visibility="private",
            )
            publish_draft(
                event,
                setup_result,
                prompt=self.prompt,
                outputs=[
                    self.prompt,
                    self.negative,
                    self.model,
                    self.width,
                    self.height,
                    self.denoise,
                    self.setup_choice,
                    self.notice,
                ],
            )

        clear_result = gr.JSON(None, visible="hidden")
        clear_inputs = [self.prompt, self.setup_choice]
        clear_guard, clear_begin = capture_draft(
            self.prompt, clear_inputs, f"gallery-clear-setup-{clear_result._id}"
        )

        def clear_local_setup(prompt, choice, guard):
            return {
                self.setup: {},
                clear_result: dict(
                    guard=guard,
                    updates=[
                        gr.skip(),
                        "",
                        t(
                            "Current sampling settings selected. Your Gallery text and image were kept."
                        ),
                    ],
                    stale=None,
                    message=t("The Gallery draft changed. Select current sampling settings again."),
                ),
            }

        clear_event = self.clear_setup.click(
            clear_local_setup,
            inputs=[*clear_inputs, clear_guard],
            outputs=[self.setup, clear_result],
            js=clear_begin,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        publish_draft(
            clear_event,
            clear_result,
            prompt=self.prompt,
            outputs=[self.prompt, self.setup_choice, self.notice],
        )

        draft_result = gr.JSON(None, visible="hidden")
        detail_outputs = [
            self.source,
            self.preview,
            self.mask,
            self.notice,
            self.run,
            self.setup,
            draft_result,
        ]

        async def load_selection(
            record,
            saved,
            previous,
            version,
            current_model,
            action,
            current_prompt,
            current_negative,
            draft_model,
            draft_width,
            draft_height,
            guard,
            request: gr.Request = None,
        ):
            if record is None:
                return {
                    self.preview: gr.update(visible=False),
                    self.mask: gr.update(visible=False),
                    self.run: gr.update(interactive=False),
                    self.notice: t("Select an image to start a Library draft."),
                }
            try:
                ticket = validate_record(record, request)
                if same_source(previous, record):
                    return {
                        self.run: gr.update(interactive=True),
                        self.preview: gr.update(visible=True),
                        self.mask: gr.update(visible=action == "inpaint"),
                    }
                image, details = await asyncio.wrap_future(
                    preparation_admission.submit_work(
                        lambda: working_source(record),
                        cancelled=ticket.cancelled if ticket else None,
                    )
                )
                validate_record(record, request)
                data = read_metadata(saved) or {}
                original = version == "Original prompt"
                prompt = data.get("raw_prompt" if original else "prompt", "")
                negative = data.get("raw_negative" if original else "negative", "")
                model = current_model
                notice = details["filename"]
                if details["conversion"] != "none":
                    notice += " · " + localize(
                        "Any working image is prepared automatically; the original is kept.",
                        resolve_locale(request=request),
                    )
                if data.get("base_model_name"):
                    try:
                        model = resolve_resource(
                            shared.models,
                            "checkpoints",
                            data["base_model_name"],
                            data.get("base_model_hash"),
                        )
                    except ValueError as error:
                        notice += "\n\n" + str(error)
                from ruinedfooocus.features.generation.size_plan import replay_size

                try:
                    width, height = (
                        replay_size(data)
                        if data.get("width") and data.get("height")
                        else image.size
                    )
                except ValueError:
                    width, height = image.size
                from ruinedfooocus.features.models.selection import checkpoint_choices

                return {
                    self.source: deepcopy(record),
                    self.setup: {},
                    self.preview: gr.update(value=image, visible=True),
                    draft_result: dict(
                        guard=guard,
                        updates=[
                            prompt if isinstance(prompt, str) else "",
                            negative if isinstance(negative, str) else "",
                            gr.update(
                                choices=checkpoint_choices(
                                    shared.models.get_names("checkpoints"), model
                                ),
                                value=model,
                            ),
                            width,
                            height,
                            "",
                        ],
                        stale=None,
                        message=t(
                            "The Library draft changed while metadata loaded. Your text was kept."
                        ),
                    ),
                    self.mask: gr.update(
                        value=editor_payload(image, preserve_alpha=True),
                        visible=action == "inpaint",
                    ),
                    self.notice: notice,
                    self.run: gr.update(interactive=True),
                }
            except (CancelledError, asyncio.CancelledError):
                return gr.skip()
            except (ValueError, OSError) as error:
                return {
                    self.source: None,
                    self.preview: None,
                    self.mask: None,
                    self.notice: str(error),
                    self.run: gr.update(interactive=False),
                }

        selection_inputs = [
            selected,
            metadata,
            self.source,
            prompt_version,
            shared.get_ctrl("base_model_name"),
            self.operation,
            self.prompt,
            self.negative,
            self.model,
            self.width,
            self.height,
        ]
        guard, begin = capture_draft(
            self.prompt, selection_inputs, f"library-metadata-{draft_result._id}"
        )
        event = selected.change(
            load_selection,
            inputs=[*selection_inputs, guard],
            outputs=detail_outputs,
            js=begin,
            trigger_mode="always_last",
            concurrency_limit=2,
            show_progress="minimal",
            show_progress_on=self.preview,
            api_visibility="undocumented",
        )
        publish_draft(
            event,
            draft_result,
            prompt=self.prompt,
            outputs=[
                self.prompt,
                self.negative,
                self.model,
                self.width,
                self.height,
                self.setup_choice,
            ],
        )

        def select_prompt(saved, version):
            data = read_metadata(saved) or {}
            original = version == "Original prompt"
            prompt = data.get("raw_prompt" if original else "prompt")
            negative = data.get("raw_negative" if original else "negative", "")
            if not isinstance(prompt, str):
                raise gr.Error(
                    "This image has no original prompt. Choose the rendered prompt instead."
                )
            return prompt, negative

        prompt_version.input(
            select_prompt,
            inputs=[metadata, prompt_version],
            outputs=[self.prompt, self.negative],
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )
        self.operation.change(
            lambda action: gr.update(visible=action == "inpaint"),
            inputs=self.operation,
            outputs=self.mask,
            queue=False,
            show_progress="hidden",
            show_progress_on=[],
            api_visibility="undocumented",
        )
        self.operation.change(
            lambda action: [gr.update(visible=action != "restore-seedvr2") for _ in range(7)],
            inputs=self.operation,
            outputs=[
                self.model,
                self.prompt,
                self.negative,
                self.prompt_processing,
                self.denoise,
                self.width,
                self.height,
            ],
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )

        def resource_context(
            action, upscale, background, vision, restoration, request: gr.Request = None
        ):
            from ruinedfooocus.features.generation.automatic_upscale import choices
            from ruinedfooocus.features.image_processing.ui_native_background import (
                choices as background_choices,
            )
            from ruinedfooocus.features.image_processing.ui_native_vision import model_choices

            if action == "upscale":
                available = [(value, value) for value in choices() if value != "Off"]
                current = upscale if upscale != "Off" else "2× General (RealESRGAN)"
            elif action == "rembg":
                current = background or ""
                available = background_choices(shared.path_manager, current)
            elif action == "depth-map":
                available = model_choices(shared.path_manager, action, vision or "")
                current = vision or None
            elif action == "restore-seedvr2":
                from ruinedfooocus.features.image_processing.native_seedvr2 import (
                    model_choices as restoration_choices,
                )

                available = restoration_choices(shared.path_manager)
                current = restoration or None
            else:
                return gr.update(visible=False)
            if current is not None and current not in {value for _, value in available}:
                available.append(
                    (
                        localize(
                            "Unavailable: {name}", resolve_locale(request=request), name=current
                        ),
                        current,
                    )
                )
            return gr.update(choices=available, value=current, visible=True)

        try:
            restoration_current = shared.get_ctrl("seedvr2_model")
        except (ValueError, IndexError, KeyError):
            restoration_current = gr.State(None)
        self.operation.change(
            resource_context,
            inputs=[
                self.operation,
                shared.get_ctrl("automatic_upscale"),
                shared.get_ctrl("rembg_native_model"),
                shared.get_ctrl("native_vision_model"),
                restoration_current,
            ],
            outputs=self.resource,
            show_progress="hidden",
            api_visibility="undocumented",
        )
        self.operation.change(
            lambda action: gr.update(visible=action == "restore-seedvr2"),
            inputs=self.operation,
            outputs=self.seedvr2_controls,
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )

        def codec_context(action, current=None):
            if action != "restore-seedvr2":
                return gr.update()
            from ruinedfooocus.features.image_processing.ui_native_seedvr2 import selection_choices

            return gr.update(choices=selection_choices(shared.path_manager, current, codec=True))

        self.operation.change(
            codec_context,
            inputs=[self.operation, self.seedvr2_vae],
            outputs=self.seedvr2_vae,
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )

        def run(
            record,
            draft_source,
            prompt,
            negative,
            checkpoint,
            action,
            strength,
            width,
            height,
            tool_resource,
            mask,
            prompt_processing,
            request: gr.Request = None,
            *values,
        ):
            validate_record(record, request)
            if not same_source(record, draft_source):
                raise gr.Error(
                    "The selected image changed. Wait for its Library draft before running."
                )
            controls = dict(zip(shared.state["ctrls_name"], values))
            staged = values[len(shared.state["ctrls_name"]) + 3 :]
            if len(staged) >= 2 and staged[1]:
                chosen = staged[0].get(staged[1]) if isinstance(staged[0], dict) else None
                if (
                    not isinstance(chosen, dict)
                    or not same_source(chosen.get("source"), record)
                    or not isinstance(chosen.get("values"), dict)
                ):
                    raise gr.Error(
                        "The Gallery setup no longer matches this image. Reuse its setup or select Use current sampling settings."
                    )
                controls.update(deepcopy(chosen["values"]))
            data = None
            try:
                source_path(record)
                if action in ("upscale", "rembg", "depth-map", "restore-seedvr2"):
                    features = dict(native_edit=False, inpaint=False, qwen_local=False)
                else:
                    if not isinstance(checkpoint, str) or not checkpoint:
                        raise ValueError("Select an image checkpoint before running this action.")
                    features = tools.checkpoint_features(
                        checkpoint, controls.get("moss_image_edit_format")
                    )
                if action == "restore-seedvr2":
                    restoration_vae, restoration_scale, restoration_color = values[
                        len(shared.state["ctrls_name"]) : len(shared.state["ctrls_name"]) + 3
                    ]
                    controls.update(
                        seedvr2_vae=restoration_vae,
                        seedvr2_scale=restoration_scale,
                        seedvr2_color=restoration_color,
                    )
                from ruinedfooocus.app.resource_lifecycle import ui_request_owner
                from ruinedfooocus.features.gallery import editor_session
                from ruinedfooocus.features.gallery.editing_workflow import open_editor
                from ruinedfooocus.features.image_processing.native_image_formats import (
                    image_metadata,
                )

                owner = ui_request_owner(request)
                store = editor_session.default_store()
                snapshot = open_editor(
                    record,
                    image_metadata(source_path(record)).get("parameters"),
                    owner=owner,
                    store=store,
                )
                context = dict(
                    session_id=snapshot.session_id,
                    revision=snapshot.revision,
                    source_id=snapshot.current.artifact_id,
                )
                if action == "inpaint":
                    from ruinedfooocus.core.cache_contract import configured_cache_root
                    from ruinedfooocus.features.image_editing.inpaint_utils import source_and_mask

                    image, _ = working_source(record)
                    _, selection = source_and_mask(
                        image, mask, preserve_alpha=bool(features.get("qwen_local"))
                    )
                    mask_path = configured_cache_root() / "editor-inputs" / (uuid4().hex + ".png")
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        selection.save(mask_path)
                        bound = store.bind_mask(
                            snapshot.session_id,
                            mask_path,
                            snapshot.current.artifact_id,
                            owner=owner,
                        )
                    except BaseException:
                        mask_path.unlink(missing_ok=True)
                        raise
                    context["mask_id"] = bound.artifact_id
                controls.update(
                    aspect_ratios_selection="Custom...",
                    custom_width=width,
                    custom_height=height,
                    size_mode="target",
                )
                data = build_library_request(
                    controls,
                    record,
                    prompt=prompt,
                    negative=negative,
                    checkpoint=checkpoint,
                    action=action,
                    resets=resets,
                    features=features,
                    presets=tools.controlnet.cn_options,
                    custom=tools.NEWCN,
                    strength=strength,
                    mask=mask,
                    tool_resource=tool_resource,
                    prompt_processing=prompt_processing,
                    editor_context=context,
                    editor_store=store,
                    owner=owner,
                )
                if data.get("seed_random"):
                    data["seed"] = secrets.randbelow(2**32)
                validate_record(record, request)
            except (ValueError, OSError) as error:
                if data is not None:
                    editor_session.cancel_request(data, owner=owner, store=store)
                raise gr.Error(str(error)) from None
            yield {
                self.run: gr.update(interactive=False),
                self.stop: gr.update(visible=True),
                self.result: gr.update(value=None, visible=False),
                self.downloads: gr.update(value=[], visible=False),
                self.open_result: gr.update(interactive=False),
            }
            try:
                with closing(run_request(data)) as results:
                    for updates in results:
                        output = {}
                        for name, local in (
                            ("run", self.run),
                            ("stop", self.stop),
                            ("progress", self.progress),
                            ("task", self.task),
                        ):
                            if run_controls[name] in updates:
                                output[local] = updates[run_controls[name]]
                        if run_controls["preview"] in updates:
                            output[self.result] = updates[run_controls["preview"]] | {
                                "visible": True
                            }
                        if run_controls["artifacts"] in updates:
                            records = updates[run_controls["artifacts"]]
                            paths = [item["path"] for item in records]
                            output[self.downloads] = gr.update(value=paths, visible=bool(paths))
                            if records:
                                output[self.result] = gr.update(
                                    value=records[0].get("display", records[0]["path"]),
                                    visible=True,
                                )
                                completed = store.snapshot(snapshot.session_id, owner=owner)
                                if completed.revision > snapshot.revision:
                                    output[self.terminal] = {"snapshot": completed.to_dict()}
                                    output[self.open_result] = gr.update(interactive=True)
                        latest_source = run_controls.get("latest_source")
                        if latest_source is not None and latest_source in updates:
                            output[latest_source] = updates[latest_source]
                        yield output
            except Exception as error:
                yield {self.notice: str(error)}
            finally:
                editor_session.cancel_request(data, owner=owner, store=store)
            yield {
                self.run: gr.update(interactive=True, visible=True),
                self.stop: gr.update(visible=False),
                self.task: None,
            }

        self.run.click(
            run,
            inputs=[
                selected,
                self.source,
                self.prompt,
                self.negative,
                self.model,
                self.operation,
                self.denoise,
                self.width,
                self.height,
                self.resource,
                self.mask,
                self.prompt_processing,
                *shared.state["ctrls_obj"],
                self.seedvr2_vae,
                self.seedvr2_scale,
                self.seedvr2_color,
                self.setup,
                self.setup_choice,
            ],
            outputs=list(
                dict.fromkeys(
                    [
                        self.task,
                        self.terminal,
                        self.open_result,
                        self.run,
                        self.stop,
                        self.progress,
                        self.result,
                        self.downloads,
                        *(
                            [run_controls["latest_source"]]
                            if "latest_source" in run_controls
                            else []
                        ),
                    ]
                )
            ),
            show_progress="minimal",
            show_progress_on=self.progress,
            concurrency_id="compute:shared",
            concurrency_limit=1,
            api_visibility="undocumented",
        )
        self.stop.click(
            stop_request,
            inputs=self.task,
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )
