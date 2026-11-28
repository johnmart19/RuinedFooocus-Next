"""Settings presentation that consumes runtime facts without initializing a backend."""

import gradio as gr

from ruinedfooocus.core.translation_manager import translate_ui as t


def create_image_runtime_summary(rows):
    """Keep initialized identity visible and requested policies in diagnostics."""
    identity_fields = {"Active image backend", "Selected device", "PyTorch build"}
    identity, details = [], []
    for row in rows:
        (identity if row[0] in identity_fields else details).append(row)
    summary = gr.Dataframe(
        label=t("Active image runtime"),
        headers=["Field", "Current state"],
        datatype=["str", "str"],
        column_widths=["36%", "64%"],
        value=identity,
        interactive=False,
        wrap=True,
        elem_classes=["rf-technical-table"],
    )
    with gr.Accordion(t("Runtime diagnostics"), open=False):
        diagnostics = gr.Dataframe(
            headers=["Field", "Current state"],
            datatype=["str", "str"],
            column_widths=["36%", "64%"],
            value=details,
            interactive=False,
            wrap=True,
            elem_classes=["rf-technical-table"],
        )
    return summary, diagnostics
