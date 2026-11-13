"""Compact owned-map actions over existing Editor controls and handoff services."""

from collections.abc import Mapping
from concurrent.futures import CancelledError

import gradio as gr

from ruinedfooocus.app.resource_lifecycle import ui_request_owner
from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui as t


CONTROL_FIELDS = frozenset(
    (
        "cn_selection",
        "cn_type",
        "cn_model",
        "cn_model_signature",
        "cn_depth_preprocess",
        "control_recipe",
        "control_union_type",
    )
)


class GuidedResultActions:
    """The result is a guide; only an explicit later Run performs the edit."""

    def __init__(self, button=None, *, denoise=0.35):
        self.button = button or gr.Button(t("Use map for editing"), size="sm", visible=False)
        self.handoff = gr.State(None)
        self.status = gr.Markdown("", visible=False)
        self.denoise = gr.Slider(
            label=t("Edit denoise"),
            minimum=0,
            maximum=1,
            step=0.01,
            value=denoise,
            visible=False,
            info=t("Lower values retain more of the original image. The map guides the edit."),
        )
        self.objects = gr.Dropdown(
            label=t("Mask selection"), choices=[], value=[], multiselect=True, visible=False
        )

    @property
    def controls(self):
        return (self.button, self.handoff, self.status, self.denoise, self.objects)

    @staticmethod
    def _current(workspace, context, owner):
        if not isinstance(context, dict):
            raise ValueError("Open a current owned image result first.")
        snapshot = workspace.selected_store().snapshot(context["session_id"], owner=owner)
        if snapshot.revision != context.get(
            "revision"
        ) or snapshot.current.artifact_id != context.get("source_id"):
            raise CancelledError("The current image result changed.")
        return snapshot

    @staticmethod
    def _derived(snapshot):
        result = snapshot.derived_results
        return (
            result
            if result
            and result.get("operation") == "segment"
            and result.get("revision") == snapshot.revision
            and result.get("source_id") == snapshot.current.artifact_id
            and result.get("source_frame_id") == snapshot.current.frame_id
            else None
        )

    @staticmethod
    def _selection(snapshot):
        derived = GuidedResultActions._derived(snapshot)
        if derived is not None:
            return bool(
                derived.get("output") == "mask"
                and derived.get("selection_eligible")
                and derived.get("result_ids")
                and snapshot.current.frame.dtype == "uint8"
                and snapshot.current.frame.color_space == "sRGB"
            )
        step = next(
            (
                step
                for step in reversed(snapshot.steps)
                if step["result_id"] == snapshot.current.artifact_id
            ),
            None,
        )
        return bool(
            step
            and step.get("operation") in ("image-mask", "face-landmarks")
            and step.get("execution", {}).get("image_utility_source", {}).get("mask_role")
            == "inpaint-selection"
        )

    @staticmethod
    def _matches(handoff, context):
        return bool(
            handoff is not None
            and isinstance(context, dict)
            and callable(getattr(handoff, "public_context", None))
            and all(value == context.get(name) for name, value in handoff.public_context().items())
        )

    @staticmethod
    def _guide_label(kind, locale):
        return (
            localize("Normal map · editing guide", locale)
            if kind == "normal"
            else localize("Depth map · editing guide", locale)
        )

    def bind(
        self,
        workspace,
        *,
        checkpoint,
        control_plan,
        use_selection,
        create_tabs=None,
    ):
        """control_plan validates compatibility before returning existing control fields.

        It receives (GuideSpec, effective checkpoint, selected control, signature),
        performs no download/model execution and returns only CONTROL_FIELDS.
        Request capture independently validates those fields and both owned files.
        """
        from ruinedfooocus.features.image_editing import guide_handoffs as guides
        from ruinedfooocus.features.gallery.ui_workspace import latest_ui_requests

        controls = workspace.controls
        control_model = controls.get("cn_model") or gr.State("")
        control_signature = controls.get("cn_model_signature") or gr.State(None)
        inputs = [workspace.context, self.handoff, workspace.action]
        self.outputs = list(
            dict.fromkeys(
                [
                    *workspace.outputs,
                    *self.controls,
                    workspace.mask_choice,
                    workspace.mask_index,
                    workspace.mask_count,
                    *([create_tabs] if create_tabs is not None else []),
                ]
            )
        )
        # Projection can restore these source widgets; explicit actions keep full outputs.
        projection_outputs = list(
            dict.fromkeys(
                [
                    *self.controls,
                    workspace.context,
                    workspace.notice,
                    workspace.mask,
                    controls["image_source"],
                    controls["input_image"],
                    *(
                        workspace.run_controls[name]
                        for name in ("preview", "native_source", "inpaint_toggle", "inpaint")
                    ),
                ]
            )
        )

        def plan(spec, name, selected, signature):
            values = control_plan(spec, name, selected, signature)
            if not isinstance(values, Mapping) or set(values) - CONTROL_FIELDS:
                raise ValueError("The guide control plan contains unsupported fields.")
            if set(values) - set(controls):
                raise ValueError("The editing controls required by this guide are unavailable.")
            return values

        def project(context, handoff, action, request: gr.Request = None):
            owner = ui_request_owner(request)
            locale = resolve_locale(request=request)
            hidden = {
                self.button: gr.update(visible=False),
                self.status: gr.update(visible=False),
                self.denoise: gr.update(visible=False),
                self.handoff: None,
                self.objects: gr.update(visible=False, choices=[], value=[]),
            }
            ticket = None

            def publish(updates):
                return latest_ui_requests.publish(ticket, lambda: updates)

            try:
                ticket = latest_ui_requests.peek(owner, workspace.channel)
                snapshot = self._current(workspace, context, owner)
                derived = self._derived(snapshot)
                attached = self._matches(handoff, context)
                if attached and action != "generate":
                    updates = workspace.snapshot_updates(snapshot, request=request)
                    self._current(workspace, context, owner)
                    return publish(updates | hidden)
                if attached:
                    spec = handoff.spec
                    message = t(
                        "The original image is restored. Its editing guide is attached separately."
                    )
                elif self._selection(snapshot):
                    self._current(workspace, context, owner)
                    return publish(
                        hidden
                        | {
                            self.button: gr.update(
                                value=t("Use result as selection mask"), visible=True
                            ),
                            self.status: gr.update(
                                value=t("Selection map · use it to inpaint the original image."),
                                visible=True,
                            ),
                            self.objects: gr.update(
                                choices=[
                                    (str(index + 1), str(index))
                                    for index in range(len(derived["result_ids"]))
                                ]
                                if derived
                                else [],
                                value=[str(index) for index in range(len(derived["result_ids"]))]
                                if derived
                                else [],
                                visible=bool(derived),
                            ),
                        }
                    )
                elif derived is not None:
                    self._current(workspace, context, owner)
                    return publish(
                        hidden
                        | {
                            self.status: gr.update(
                                value=t("No matching instances found."),
                                visible=not derived["result_ids"],
                            )
                        }
                    )
                else:
                    spec = guides.describe_result_guide(
                        workspace.selected_store(),
                        context["session_id"],
                        context["revision"],
                        context["source_id"],
                        owner=owner,
                    )
                    message = self._guide_label(spec.kind, locale)
                if not attached:
                    message = localize(
                        "{guide}\n\nSelect a compatible local ControlNet in More tools before using this map.",
                        locale,
                        guide=message,
                    )
                self._current(workspace, context, owner)
                return publish(
                    {
                        self.button: gr.update(
                            value=t("Use map for editing"), visible=not attached
                        ),
                        self.status: gr.update(value=message, visible=True),
                        self.denoise: gr.update(visible=attached),
                        self.handoff: handoff if attached else None,
                    }
                )
            except CancelledError:
                return {control: gr.skip() for control in self.controls}
            except (ValueError, OSError, PermissionError, KeyError):
                if ticket is None:
                    return {control: gr.skip() for control in self.controls}
                try:
                    return publish(hidden)
                except CancelledError:
                    return {control: gr.skip() for control in self.controls}

        self.projection = gr.on(
            triggers=[workspace.context.change, workspace.version.change, workspace.action.change],
            fn=project,
            inputs=inputs,
            outputs=projection_outputs,
            trigger_mode="always_last",
            queue=False,
            show_progress="hidden",
            show_progress_on=[],
            api_visibility="private",
        )

        def use_map(
            context,
            name,
            selected,
            signature,
            task_id,
            objects,
            version,
            request: gr.Request = None,
        ):
            owner = ui_request_owner(request)
            snapshot = self._current(workspace, context, owner)
            if self._selection(snapshot):
                derived = self._derived(snapshot)
                if derived is not None:
                    from ruinedfooocus.app.preparation_admission import preparation_admission

                    token, prepared, holder = None, None, []
                    committed = False
                    store = workspace.selected_store()
                    try:
                        if task_id is not None:
                            raise ValueError("Finish the current image job before selecting masks.")
                        if not isinstance(objects, list) or any(
                            not isinstance(index, str) or not index.isdecimal() for index in objects
                        ):
                            raise ValueError("Select one or more object masks.")
                        indices = [int(index) for index in objects]
                        token = latest_ui_requests.begin(owner, workspace.channel, int(version))

                        def check_cancelled():
                            if token.cancelled.is_set():
                                raise CancelledError("The mask selection changed.")

                        def prepare_selection():
                            result = store.prepare_derived_selection(
                                context["session_id"],
                                context["revision"],
                                indices,
                                owner=owner,
                                check_cancelled=check_cancelled,
                            )
                            holder.append(result)
                            return result

                        future = preparation_admission.submit_work(
                            prepare_selection, cancelled=token.cancelled
                        )

                        def discard_failed_preparation(done):
                            if holder and (
                                token.cancelled.is_set()
                                or done.cancelled()
                                or done.exception() is not None
                            ):
                                store.discard_derived_selection(holder[0])

                        future.add_done_callback(discard_failed_preparation)
                        latest_ui_requests.attach(token, future)
                        prepared = future.result()
                        # Source display decoding also stays outside the publication lock.
                        source_updates = workspace.snapshot_updates(snapshot, request=request)

                        def attach_selection():
                            current, mask = store.commit_derived_selection(
                                prepared, owner=owner, check_cancelled=check_cancelled
                            )
                            return source_updates | {
                                workspace.context: dict(
                                    workspace.public_context(current), mask_id=mask.artifact_id
                                ),
                                workspace.action: "inpaint",
                                workspace.mask_choice: "instance" if len(indices) == 1 else "union",
                                workspace.mask_index: indices[0] if len(indices) == 1 else 0,
                                workspace.mask_count: len(derived["result_ids"]),
                                workspace.notice: t(
                                    "Selection mask attached to its original image. The source pixels and alpha were preserved."
                                ),
                            }

                        updates = latest_ui_requests.publish(token, attach_selection)
                        committed = True
                    except CancelledError:
                        return {control: gr.skip() for control in self.outputs}
                    except (ValueError, OSError, PermissionError, KeyError, RuntimeError) as error:
                        raise gr.Error(str(error)) from None
                    finally:
                        staged = prepared or (holder[0] if holder else None)
                        if staged is not None and not committed:
                            store.discard_derived_selection(staged)
                        if token is not None:
                            latest_ui_requests.finish(token)
                else:
                    updates = use_selection(context, task_id, version, request)
                return updates | {
                    self.handoff: None,
                    self.status: gr.update(visible=False),
                    self.denoise: gr.update(visible=False),
                    self.objects: gr.update(visible=False, choices=[], value=[]),
                }

            # The first explicit file verification can hash a large ControlNet.
            # Keep it outside the shared latest-intent publication lock.
            try:
                spec = guides.describe_result_guide(
                    workspace.selected_store(),
                    context["session_id"],
                    context["revision"],
                    context["source_id"],
                    owner=owner,
                )
                values = plan(spec, name, selected, signature)
            except (ValueError, OSError, PermissionError, KeyError) as error:
                raise gr.Error(str(error)) from None

            def attach():
                current_spec = guides.describe_result_guide(
                    workspace.selected_store(),
                    context["session_id"],
                    context["revision"],
                    context["source_id"],
                    owner=owner,
                )
                if current_spec != spec:
                    raise CancelledError("The current editing guide changed.")
                self._current(workspace, context, owner)
                restored, handoff = guides.restore_result_guide(
                    workspace.selected_store(),
                    context["session_id"],
                    context["revision"],
                    context["source_id"],
                    owner=owner,
                )
                images = guides.handoff_images(workspace.selected_store(), handoff, owner=owner)
                updates = workspace.snapshot_updates(restored, request=request)
                updates.update(
                    {controls[field]: gr.update(value=value) for field, value in values.items()}
                )
                updates.update(
                    {
                        workspace.context: handoff.public_context(),
                        self.handoff: handoff,
                        self.button: gr.update(visible=False),
                        self.status: gr.update(
                            value=t(
                                "The original image is restored. Its editing guide is attached separately."
                            ),
                            visible=True,
                        ),
                        self.denoise: gr.update(visible=True),
                        workspace.action: "generate",
                        controls["input_image"]: images["input_image"],
                    }
                )
                if create_tabs is not None:
                    updates[create_tabs] = gr.update(selected="image-tools")
                return updates

            try:
                return workspace._source_action(task_id, version, request, attach)
            except (ValueError, OSError, PermissionError, KeyError) as error:
                raise gr.Error(str(error)) from None

        self.event = self.button.click(
            use_map,
            inputs=[
                workspace.context,
                checkpoint,
                control_model,
                control_signature,
                workspace.run_controls["task"],
                self.objects,
                workspace.version,
            ],
            outputs=self.outputs,
            js=workspace.version_js,
            show_progress="hidden",
            show_progress_on=self.status,
            api_visibility="private",
        )
        return self
