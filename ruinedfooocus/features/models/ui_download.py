from ruinedfooocus.core.translation_manager import translate_ui

"""Explicit first-model download for a fresh clone or source archive."""

import gradio as gr

from ruinedfooocus.features.models.selection import checkpoint_choices
from ruinedfooocus.features.models.ui_model_components import offline
from ruinedfooocus.features.generation.defaults import STARTUP_DEFAULTS


def download_default(progress=gr.Progress()):
    if offline():
        raise gr.Error(
            "Offline mode: add a checkpoint locally or restart online to download the default model."
        )
    from ruinedfooocus.app import state as shared

    name = STARTUP_DEFAULTS["base_model"]

    def report(received, total, speed):
        progress(received / total if total else 0, desc=f"Downloading {name}")

    shared.path_manager.get_folder_file_path("checkpoints", name, progress=report)
    shared.models.update_all_models()
    return gr.update(
        choices=checkpoint_choices(
            shared.models.get_names("checkpoints"),
            name,
            lambda filename: shared.models.get_models_by_path("checkpoints", filename, fetch=False),
        ),
        value=name,
    )


def create_ui(base_model):
    gr.Markdown(
        translate_ui(
            "Add checkpoints to `models/checkpoints` or configure **Settings → Storage → Paths**, then refresh files. The default SDXL checkpoint is an optional download of several GB; no model weights are included in source releases."
        )
    )
    button = gr.Button(
        translate_ui("Download default image model"), interactive=not offline(), size="sm"
    )
    button.click(
        download_default,
        outputs=base_model,
        concurrency_id="checkpoint_download",
        concurrency_limit=1,
        api_visibility="undocumented",
    )
