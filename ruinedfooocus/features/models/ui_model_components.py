from ruinedfooocus.core.translation_manager import translate_ui

"""Image checkpoint component inventory for Models."""

import json
import os
from pathlib import Path
from concurrent.futures import CancelledError
import threading

import gradio as gr

from ruinedfooocus.features.models.component_profiles import component_inventory, session_choices
from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.resource_defaults import RESOURCE_LABELS

_downloads = {}
_download_lock = threading.Lock()


def component_label(key):
    if key.startswith("vae_"):
        return "VAE"
    return RESOURCE_LABELS.get(key, key).split(" (")[0]


def selection_source(row, choices, values):
    if row.get("locked"):
        return "Local bundle"
    if row["status"] == "Embedded in checkpoint":
        return "Built into checkpoint"
    if row["key"] in choices:
        return "Saved for this checkpoint" if choices[row["key"]] else "Family default"
    return "Profile default" if values.get(row["key"]) else "Family default"


def selected(name):
    import ruinedfooocus.app.state as shared

    if name not in shared.models.get_names("checkpoints"):
        raise ValueError("Select an installed image checkpoint")
    path = shared.models.get_file_from_name("checkpoints", name)
    if path is None:
        raise ValueError("The selected checkpoint is no longer installed")
    from ruinedfooocus.features.models.local_diffusers import is_diffusers_index, bundle_metadata

    if is_diffusers_index(path):
        # Bundle presentation is local/data-only and never imports its advertised pipeline.
        return Path(path), bundle_metadata(path)["baseModel"]
    # Cached local metadata only: selection must not cause network access or model loading.
    cache = (shared.models.cache_paths["checkpoints"] / Path(path).name).with_suffix(".json")
    try:
        metadata = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    signature = metadata.get("_rf_file_signature")
    if signature is not None and signature != file_signature(path):
        metadata = {}
    return Path(path), metadata.get("baseModel", "Unknown")


def offline():
    from ruinedfooocus.app.arguments import args

    return args.offline or any(
        os.environ.get(name, "").upper() in ("1", "TRUE", "YES", "ON")
        for name in ("RF_OFFLINE", "HF_HUB_OFFLINE")
    )


def refresh(name, overrides=None, scope=None, message=""):
    import ruinedfooocus.app.state as shared

    try:
        path, family = selected(name)
        choices, identity = session_choices(
            path, shared.settings.default_settings, overrides, scope
        )
        state = component_inventory(
            path, family, shared.settings.default_settings, shared.path_manager, choices
        )
        rows = [
            [
                row.get("label") or component_label(row["key"]),
                "Built into checkpoint"
                if row["status"] == "Embedded in checkpoint"
                else row["filename"],
                "Ready"
                if row["status"] == "Embedded in checkpoint"
                else row["status"]
                + " · "
                + selection_source(row, choices, shared.settings.default_settings)
                + (
                    " · Incompatible"
                    if row.get("compatibility") == "incompatible"
                    else " · Unverified"
                    if row.get("compatibility") == "unknown"
                    else ""
                )
                + "\n"
                + (
                    f"{row['bytes'] / 1024**3:.2f} GiB (local)"
                    if row["bytes"] is not None
                    else f"{row['published_bytes'] / 1024**3:.2f} GiB (published)"
                    if row.get("published_bytes") is not None
                    else "Unknown size"
                ),
            ]
            for row in state["rows"]
        ]
        details = {
            row["key"]: {
                "file": "Built into checkpoint"
                if row["status"] == "Embedded in checkpoint"
                else row["filename"],
                "location": row["location"],
                "source": row["source"] or "Local / bundled",
                "selection": selection_source(row, choices, shared.settings.default_settings),
                "download_available": row["download_available"],
                "compatibility": row.get("compatibility", "unknown"),
                "compatibility_reason": row.get("compatibility_reason", ""),
                "codec_files": row.get("files", []),
                "native_contract": next(
                    (
                        contract
                        for contract in state.get("compatibility_contract", {}).get("roles", ())
                        if contract["key"] == row["key"]
                    ),
                    {},
                ),
            }
            for row in state["rows"]
        }
        ready = not any(row["status"] == "Missing" for row in state["rows"])
        incompatible = any(row.get("compatibility") == "incompatible" for row in state["rows"])
        summary = (
            "Incompatible components"
            if incompatible
            else "Components present"
            if ready
            else "Missing components"
        )
        status = f"**{state['architecture']}** · {summary}\n\n"
        if incompatible:
            status += "Choose compatible components before generating.\n\n"
        total = state.get("missing_payload_bytes")
        if not ready:
            status += (
                "Missing download size: "
                + (f"{total:,} published bytes" if total is not None else "Unknown total")
                + ".\n\n"
            )
        status += message
        if offline():
            status += "\n\nOffline mode: downloads disabled."
        return (
            rows,
            details,
            status,
            gr.update(
                interactive=not offline() and any(r["download_available"] for r in state["rows"])
            ),
            choices,
            identity,
            gr.update(
                choices=[
                    (row.get("label") or component_label(row["key"]), row["key"])
                    for row in state["rows"]
                ],
                value=state["rows"][0]["key"] if state["rows"] else None,
            ),
        )
    except (OSError, ValueError, TypeError) as error:
        return (
            [],
            {},
            str(error),
            gr.update(interactive=False),
            None,
            None,
            gr.update(choices=[], value=None),
        )


def refresh_selected_role(name, overrides, scope, selected_role):
    """Keep the editor on its current role when refreshing this checkpoint."""
    result = list(refresh(name, overrides, scope))
    roles = result[-1]
    if selected_role in {value for label, value in roles.get("choices", ())}:
        result[-1] = {**roles, "value": selected_role}
    return tuple(result)


def download_selected(name, overrides, scope, request: gr.Request, progress=gr.Progress()):
    import ruinedfooocus.app.state as shared

    key = request.session_hash
    job = {"cancelled": threading.Event(), "handle": None}
    with _download_lock:
        _downloads[key] = job
    try:
        if offline():
            raise ValueError("Offline mode: component downloads are disabled")
        path, family = selected(name)
        choices, identity = session_choices(
            path, shared.settings.default_settings, overrides, scope
        )
        state = component_inventory(
            path, family, shared.settings.default_settings, shared.path_manager, choices
        )
        files = []
        for row in state["rows"]:
            if not row["download_available"]:
                continue
            if job["cancelled"].is_set():
                raise CancelledError()

            def report(received, total, speed, filename=row["filename"]):
                amount = (
                    f"{received:,} / {total:,} bytes"
                    if total
                    else f"{received:,} bytes · total unknown"
                )
                progress(
                    received / total if total else None, desc=f"Downloading {filename} · {amount}"
                )

            progress(None, desc="Preparing " + row["filename"])
            handle = shared.path_manager.prepare_file(
                row["download_key"], progress=report, destination=row["location"]
            )
            with _download_lock:
                job["handle"] = handle
                if job["cancelled"].is_set():
                    handle.cancel()
            handle.result()
            files.append(row["filename"])
        return refresh(
            name,
            choices,
            identity,
            "Downloaded: " + ", ".join(files) if files else "No downloadable missing components.",
        )
    except CancelledError:
        return refresh(
            name, overrides, scope, "Download cancelled. Installed components remain usable."
        )
    except Exception as error:
        return refresh(name, overrides, scope, "Download failed: " + str(error))
    finally:
        with _download_lock:
            if _downloads.get(key) is job:
                _downloads.pop(key, None)


def cancel_download(request: gr.Request):
    with _download_lock:
        job = _downloads.get(request.session_hash)
        if job:
            job["cancelled"].set()
            if job["handle"] is not None:
                job["handle"].cancel()


def create_ui(base_model, model_refresh=None):
    """Create inside the desired model tab; selection changes refresh local status only."""
    import ruinedfooocus.app.state as shared

    overrides, scope = gr.State(None), gr.State(None)
    shared.add_ctrl("image_component_overrides", overrides)
    shared.add_ctrl("_image_component_scope", scope)
    with gr.Column(min_width=0, elem_id="model-components-panel") as panel:
        table = gr.Dataframe(
            headers=[
                "Component",
                "Effective file",
                "Status / file size",
            ],
            datatype=["str"] * 3,
            column_widths=["25%", "45%", "30%"],
            interactive=False,
            wrap=True,
            elem_id="model-component-inventory",
        )
        status = gr.Markdown(
            translate_ui("Select an image checkpoint and click Refresh components.")
        )
        with gr.Row(elem_classes=["rf-actions"]):
            refresh_button = gr.Button(translate_ui("Refresh components"))
            download_button = gr.Button(translate_ui("Download missing"), interactive=False)
            cancel = gr.Button(translate_ui("Cancel component download"), size="sm")
        with gr.Accordion(translate_ui("Component locations and sources"), open=False):
            details = gr.JSON(label=translate_ui("Selected components"), value={})
        from ruinedfooocus.features.models.ui_component_overrides import (
            create_ui as create_overrides_ui,
        )

        outputs = [table, details, status, download_button, overrides, scope]
        # The editor's role control is the final refresh output.
        role, _, _ = create_overrides_ui(base_model, overrides, scope, refresh, outputs)
        outputs.append(role)

        def refresh_after_choice(name, choices, identity, selected_role):
            result = refresh_selected_role(name, choices, identity, selected_role)
            # A Hub Use updates session choices. Refresh the visible inventory
            # without outputting those same state controls and creating a loop.
            return (*result[:4], result[-1])

        gr.on(
            triggers=[overrides.change, scope.change],
            fn=refresh_after_choice,
            inputs=[base_model, overrides, scope, role],
            outputs=[*outputs[:4], role],
            show_progress="hidden",
            api_visibility="undocumented",
            trigger_mode="always_last",
        )
        gr.on(
            triggers=[base_model.change]
            + ([model_refresh.change] if model_refresh is not None else []),
            fn=refresh,
            inputs=[base_model, overrides, scope],
            outputs=outputs,
            show_progress="hidden",
            api_visibility="undocumented",
        )
        refresh_button.click(
            refresh_selected_role,
            inputs=[base_model, overrides, scope, role],
            outputs=outputs,
            show_progress="hidden",
            api_visibility="undocumented",
        )
        download_button.click(
            download_selected,
            inputs=[base_model, overrides, scope],
            outputs=outputs,
            concurrency_id="image_component_downloads",
            concurrency_limit=1,
            show_progress="minimal",
            api_visibility="undocumented",
        )
        cancel.click(cancel_download, queue=False, api_visibility="undocumented")
    return panel
