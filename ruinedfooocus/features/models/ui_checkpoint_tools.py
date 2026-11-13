from ruinedfooocus.core.translation_manager import translate_ui
from ruinedfooocus.core.paths import asset_path

"""Local checkpoint actions for the Main Model tab."""

import filecmp
from concurrent.futures import CancelledError
from pathlib import Path
import threading

import gradio as gr
import ruinedfooocus.app.state as shared
from safetensors import SafetensorError
from ruinedfooocus.core.model_formats import NATIVE_TORCH_SUFFIXES, is_safetensors
from ruinedfooocus.features.models.model_identity import file_signature, verified_native_source
from ruinedfooocus.features.models.ui_requests import stamp, version_js
import ruinedfooocus.features.models.checkpoint_tools as tools


def checkpoint_source(name):
    try:
        path, _ = _selected(name)
        return name, tuple(file_signature(path))
    except (OSError, ValueError):
        return name, None


class CheckpointToolSelection:
    """One browser's selection; file tools retain their separate serialized lane."""

    def __init__(self):
        self.lock = threading.RLock()
        self.version = -1
        self.source = None

    def __deepcopy__(self, memo):
        return type(self)()

    def capture(self, name, version, *, select=False):
        if type(version) is not int or not 0 <= version < 2**53:
            raise CancelledError("Invalid checkpoint selection version")
        source = checkpoint_source(name)
        with self.lock:
            if self.version < 0 or select and version > self.version:
                self.version, self.source = version, source
            if version != self.version or source != self.source:
                raise CancelledError("The checkpoint selection changed")
            return version, source

    def publish(self, ticket, build):
        with self.lock:
            if (
                ticket != (self.version, self.source)
                or checkpoint_source(ticket[1][0]) != ticket[1]
            ):
                raise CancelledError("The checkpoint selection or file changed")
            return build()


def selected_result(session, name, version, build, *, outputs):
    try:
        ticket = session.capture(name, version)
        result = build()
        return session.publish(ticket, lambda: result)
    except CancelledError:
        return gr.skip() if outputs == 1 else tuple(gr.skip() for _ in range(outputs))


def checkpoint_result(version, updates):
    return stamp({"updates": list(updates)}, "checkpoint-tools-selection", version)


def publish_checkpoint_result(event, result, outputs):
    """Fence transport-delayed results immediately before native client updates."""
    publish = """(result) => {
        const identity = result?.requestIdentity;
        const current = window.__rfModelUIRequests?.['checkpoint-tools-selection'] || 0;
        if (!identity || identity.version !== current) {
            return Array.from({length: %s}, () => ({__type__: 'update'}));
        }
        return result.updates;
    }""" % len(outputs)
    return event.success(
        fn=None,
        inputs=result,
        outputs=outputs,
        js=publish,
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
        api_visibility="undocumented",
    )


def _selected(name):
    if name not in shared.models.get_names("checkpoints"):
        raise ValueError("Select an installed checkpoint first.")
    filename = shared.models.get_file_from_name("checkpoints", name)
    if filename is None:
        raise ValueError("The selected checkpoint is no longer available.")
    path = Path(filename)
    info = shared.models.get_models_by_path("checkpoints", name, fetch=False)
    return path, info


def _preview(path):
    cached = shared.models.cache_paths["checkpoints"] / path.name
    for suffix in (".jpg", ".jpeg", ".png", ".gif"):
        for candidate in (
            cached.with_suffix(suffix),
            path.with_suffix(".preview" + suffix),
            path.with_suffix(suffix),
        ):
            if candidate.is_file() and not filecmp.cmp(
                candidate, str(asset_path("warning.jpeg")), shallow=False
            ):
                return candidate


def run_action(name, action, progress, on_verified=None, *, verify_source=True):
    try:
        path, info = _selected(name)
        if (
            action == "inspect"
            and verify_source
            and path.suffix.lower() in (*NATIVE_TORCH_SUFFIXES, ".gguf")
        ):
            result = tools.verify_checkpoint_source(
                path, shared.models.cache_paths["checkpoints"], info
            )
            shared.models.revision += 1
            if on_verified is not None:
                on_verified()
            return (
                f"Verified published source: {result['family']}\nSHA256: {result['sha256']}\n"
                "No tensors were loaded or checked for NaN/Inf. "
                "Workflow controls refresh automatically."
            )
        if action == "convert":
            result = tools.create_fp16_copy(
                path, shared.models.cache_paths["checkpoints"], info, _preview(path), progress
            )
            if result["path"]:
                destination = result["path"]
                new_name = str(Path(name).with_name(destination.name))
                names = shared.models.names["checkpoints"]
                if new_name not in names:
                    names.append(new_name)
                    names.sort(key=str.casefold)
                shared.models.revision += 1
                try:
                    cached = shared.models.cache_paths["checkpoints"] / destination.name
                    if not result["preview"]:
                        shared.models.copy_embedded_preview(destination, cached)
                except OSError as error:
                    result["message"] += f" Preview cache could not be updated: {error}"
            return result["message"]
        report = tools.inspect_checkpoint(path)
        checked = tools.check_weights(path, progress) if action == "check" else None
        message = tools.format_report(report, shared.models.get_model_base(info), checked)
        if action == "inspect" and verify_source and is_safetensors(path):
            from ruinedfooocus.features.models.model_identity import is_native_checkpoint_source
            from ruinedfooocus.features.models.model_registry import load_registry

            try:
                # Only an explicit click can hash a reviewed size candidate.
                # Unknown safetensors retain their ordinary header inspection.
                candidates = any(
                    is_native_checkpoint_source(row) and row["artifact"]["bytes"] == report["bytes"]
                    for row in load_registry().entries()
                )
                if candidates:
                    result = tools.verify_checkpoint_source(
                        path, shared.models.cache_paths["checkpoints"], info
                    )
                    shared.models.revision += 1
                    if on_verified is not None:
                        on_verified()
                    message += (
                        f"\nVerified published source: {result['family']}\nSHA256: {result['sha256']}\n"
                        "Workflow controls refresh automatically."
                    )
            except (OSError, ValueError) as error:
                message += (
                    f"\nSource verification failed: {error} "
                    "Model Defaults needs a verified source or complete checkpoint recipe. "
                    "Choose Own settings or inspect the correct unchanged publisher file."
                )
        return message
    except (OSError, ValueError, SafetensorError) as error:
        return f"Checkpoint tool: {error}"


def create_ui(base_model, visible=True, model_refresh=None):
    with gr.Column(visible=visible, elem_id="checkpoint-tools") as group:
        with gr.Row(equal_height=True, elem_classes=["rf-actions"]):
            inspect = gr.Button(translate_ui("Inspect"), min_width=100)
            check = gr.Button(
                translate_ui("Check weights"),
                min_width=100,
                interactive=is_safetensors(base_model.value or ""),
            )
            convert = gr.Button(translate_ui("Create FP16 copy"), min_width=140, interactive=False)
        report = gr.Textbox(label=translate_ui("Checkpoint report"), lines=7, interactive=False)
        selection = gr.State(CheckpointToolSelection())
        selection_version = gr.Number(value=0, precision=0, visible=False)
        revision_control = model_refresh if model_refresh is not None else gr.State(0)
        channel = "checkpoint-tools-selection"
        capture_version = """(...values) => {
            values[values.length - 1] = window.__rfModelUIRequests?.['checkpoint-tools-selection'] || 0;
            return values;
        }"""

        def inspect_selected(name, revision=0, progress=gr.Progress()):
            verified = []
            if Path(name).suffix.lower() in (*NATIVE_TORCH_SUFFIXES, ".gguf"):
                result = (
                    run_action(name, "inspect", progress, lambda: verified.append(True)),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                )
            else:
                result = selection_report(
                    name, verify_source=True, on_verified=lambda: verified.append(True)
                )
            if model_refresh is not None:
                result += (revision + 1 if verified else gr.skip(),)
            return result

        def check_selected(name, progress=gr.Progress()):
            return run_action(name, "check", progress)

        def convert_selected(name, progress=gr.Progress()):
            message = run_action(name, "convert", progress)
            # List a new copy at once; the original stays selected.
            # JSON staging bypasses Dropdown's usual label/value normalization.
            choices = [(value, value) for value in shared.models.get_names("checkpoints")]
            return message, gr.update(choices=choices)

        def inspect_current(name, revision, session, version, progress=gr.Progress()):
            updates = selected_result(
                session,
                name,
                version,
                lambda: inspect_selected(name, revision, progress),
                outputs=3 + (model_refresh is not None),
            )
            return checkpoint_result(version, updates)

        def check_current(name, session, version, progress=gr.Progress()):
            update = selected_result(
                session, name, version, lambda: check_selected(name, progress), outputs=1
            )
            return checkpoint_result(version, [update])

        def convert_current(name, session, version, progress=gr.Progress()):
            try:
                ticket = session.capture(name, version)
            except CancelledError:
                return checkpoint_result(version, [gr.skip(), gr.skip()])
            result = convert_selected(name, progress)
            try:
                updates = session.publish(ticket, lambda: result)
            except CancelledError:
                # A completed copy still belongs in the inventory; keep the new selection.
                updates = gr.skip(), result[1]
            return checkpoint_result(version, updates)

        for button, callback, outputs in (
            (
                inspect,
                inspect_current,
                [report, convert, check] + ([model_refresh] if model_refresh is not None else []),
            ),
            (check, check_current, report),
        ):
            result = gr.JSON(None, visible="hidden")
            event = button.click(
                callback,
                inputs=[base_model, revision_control, selection, selection_version]
                if button is inspect
                else [base_model, selection, selection_version],
                outputs=result,
                js=capture_version,
                concurrency_id="checkpoint_tools",
                concurrency_limit=1,
                show_progress_on=report,
                api_visibility="undocumented",
            )
            publish_checkpoint_result(
                event, result, outputs if isinstance(outputs, list) else [outputs]
            )
        conversion_result = gr.JSON(None, visible="hidden")
        conversion = convert.click(
            convert_current,
            inputs=[base_model, selection, selection_version],
            outputs=conversion_result,
            js=capture_version,
            concurrency_id="checkpoint_tools",
            concurrency_limit=1,
            show_progress_on=report,
            api_visibility="undocumented",
        )
        publish_checkpoint_result(conversion, conversion_result, [report, base_model])

        def selection_report(name, *, verify_source=False, on_verified=None):
            # Selection reads headers only; source verification needs an explicit click.
            useful = False
            if name:
                try:
                    path, info = _selected(name)
                    if path.suffix.lower() in (*NATIVE_TORCH_SUFFIXES, ".gguf"):
                        verified = verified_native_source(path, info)
                        return (
                            f"Verified published family: {verified['native_family']}. No tensor integrity scan."
                            if verified is not None
                            else "Click Inspect to verify a reviewed source identity. No tensors are opened on selection.",
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                        )
                    inspection = tools.inspect_checkpoint(path)
                    payload = sum(inspection["dtypes"].values())
                    useful = bool(payload and inspection["saving_bytes"] >= payload * 0.01)
                except (OSError, ValueError, SafetensorError):
                    pass
            return (
                run_action(
                    name, "inspect", lambda *_: None, on_verified, verify_source=verify_source
                )
                if name
                else "",
                gr.update(interactive=useful),
                gr.update(interactive=bool(name) and is_safetensors(name)),
            )

        def selection_current(name, session, version):
            updates = selected_result(
                session, name, version, lambda: selection_report(name), outputs=3
            )
            return checkpoint_result(version, updates)

        def begin_selection(name, session, version):
            try:
                session.capture(name, version, select=True)
                return version
            except CancelledError:
                return gr.skip()

        selection_result = gr.JSON(None, visible="hidden")
        selection_event = base_model.change(
            begin_selection,
            inputs=[base_model, selection, selection_version],
            outputs=selection_version,
            js=version_js(channel),
            queue=False,
            trigger_mode="multiple",
            show_progress="hidden",
            show_progress_on=[],
            api_visibility="undocumented",
        ).then(
            selection_current,
            inputs=[base_model, selection, selection_version],
            outputs=selection_result,
            js=capture_version,
            concurrency_id="checkpoint_selection",
            concurrency_limit=2,
            trigger_mode="always_last",
            show_progress="hidden",
            show_progress_on=[],
            api_visibility="undocumented",
        )
        publish_checkpoint_result(selection_event, selection_result, [report, convert, check])
    return group
