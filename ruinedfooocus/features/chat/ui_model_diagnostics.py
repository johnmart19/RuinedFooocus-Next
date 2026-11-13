"""Explicit per-session Chat provenance in Settings, without model execution."""

from pathlib import Path
import json

import gradio as gr

from ruinedfooocus.app.state import path_manager, settings
from ruinedfooocus.core.translation_manager import translate_ui as t
from ruinedfooocus.features.chat.model_browser import catalogue_local_file, safe_model_evidence
from ruinedfooocus.features.chat.model_preparation import initialized_runtime_status
from ruinedfooocus.features.models.model_capabilities import configured_runtime


def _roots():
    result = {}
    for key in ("path_llm", "path_vision"):
        folders = path_manager.paths.get(key, [])
        if isinstance(folders, str):
            folders = [folders]
        result[key] = [path_manager.get_abspath(folder) for folder in folders]
    return result


def selected_evidence(quant):
    evidence = safe_model_evidence(
        quant,
        path_manager.DOWNLOADABLE_FILES,
        configured_runtime(settings.default_settings.get("llm_runtime", "llama.cpp")),
        roots=_roots(),
        projector=settings.default_settings.get("llama_mmproj") or "Auto",
    )
    snapshot = initialized_runtime_status()["runtime_snapshot"]
    if snapshot:
        for key, label in (
            ("model_identity", "Initialized model identity"),
            ("runtime_identity", "Initialized runtime identity"),
            ("requested", "Requested load configuration"),
            ("effective", "Effective load configuration"),
            ("placement", "Observed placement"),
            ("moe_cache", "MoE cache evidence"),
        ):
            evidence.append(
                [label, json.dumps(snapshot.get(key), sort_keys=True, ensure_ascii=False)]
            )
    return evidence


def refresh_sampling_recommendations(quant):
    from ruinedfooocus.features.chat.sampling_sources import prepare_sampling_source

    path = (
        Path(quant)
        if quant and Path(quant).is_absolute()
        else catalogue_local_file(quant, path_manager.DOWNLOADABLE_FILES, _roots())
    )
    if path is None or not path.is_file():
        return selected_evidence(quant) + [["Sampling refresh", "Choose an installed GGUF first."]]
    source = prepare_sampling_source(path, force=True)
    evidence = selected_evidence(quant)
    if source.get("last_lookup") == "unavailable":
        evidence.append(
            ["Sampling refresh", "Lookup unavailable; current cached evidence retained."]
        )
    return evidence


def create_model_diagnostics(selected_quant, runtime_status, sampling_status):
    """Use the browser's selected quant and initialized-state projections."""
    with gr.Accordion(t("Chat model diagnostics"), open=False) as group:
        gr.Markdown(t("Last Chat session load and sampling"))
        runtime = gr.Markdown("")
        sampling = gr.Markdown("")
        with gr.Row(elem_classes=["rf-actions"], equal_height=True):
            refresh = gr.Button(t("Refresh file presence"), size="sm", min_width=112)
            recommendations = gr.Button(
                t("Refresh sampling recommendations"), size="sm", min_width=112
            )
        details = gr.Dataframe(
            headers=["Field", "Evidence"],
            datatype=["str", "str"],
            column_widths=["36%", "64%"],
            value=[],
            interactive=False,
            wrap=True,
            elem_classes=["rf-technical-table"],
        )
    group.expand(
        lambda quant, active, sample: (active, sample, selected_evidence(quant)),
        inputs=[selected_quant, runtime_status, sampling_status],
        outputs=[runtime, sampling, details],
        queue=False,
        api_visibility="undocumented",
    )
    refresh.click(
        selected_evidence,
        inputs=selected_quant,
        outputs=details,
        queue=False,
        api_visibility="undocumented",
    )
    recommendations.click(
        refresh_sampling_recommendations,
        inputs=selected_quant,
        outputs=details,
        show_progress="hidden",
        concurrency_id="chat-source-metadata",
        api_visibility="undocumented",
    )
    return group
