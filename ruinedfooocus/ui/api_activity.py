"""Private owner activity view; no request, model, error or credential data."""

import gradio as gr

from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui as t


def activity_view(locale):
    from ruinedfooocus.app import state as shared

    if (
        getattr(getattr(shared.server_app, "state", None), "rf_openai_api_requires_key", None)
        is None
    ):
        return localize("API activity is unavailable for this launch.", locale), []
    from ruinedfooocus.app.api_activity import activity

    snapshot = activity.snapshot()
    counts = snapshot["counts"]
    summary = localize(
        "Recent API jobs: {retained}; waiting {waiting}, running {running}, completed {completed}, failed {failed}, cancelled {cancelled}.",
        locale,
        retained=snapshot["retained"],
        **{
            state: counts.get(state, 0)
            for state in ("waiting", "running", "completed", "failed", "cancelled")
        },
    )
    rows = [
        [
            localize(row["kind"].title(), locale),
            localize(row["state"].title(), locale),
            row["wait_ms"],
            row["run_ms"],
            row["total_ms"],
        ]
        for row in snapshot["rows"]
    ]
    return summary, rows


def create_api_activity_controls(tab):
    with gr.Accordion(t("Recent API activity"), open=False):
        gr.Markdown(
            t(
                "Local memory only: up to 128 jobs for one hour. Request contents, filenames, model paths, keys and error messages are not collected. Times include preparation and worker execution, not client transfer or GPU-only time."
            )
        )
        summary = gr.Markdown()
        table = gr.Dataframe(
            headers=[t("Workload"), t("State"), t("Wait (ms)"), t("Run (ms)"), t("Total (ms)")],
            column_count=5,
            interactive=False,
            wrap=True,
            elem_id="api-activity-summary",
        )
        refresh = gr.Button(t("Refresh API activity"), size="sm")

    def show_activity(request: gr.Request):
        return activity_view(resolve_locale(request=request))

    def select_activity(request: gr.Request, event: gr.SelectData):
        return show_activity(request) if event.selected else (gr.skip(), gr.skip())

    options = dict(
        outputs=[summary, table], queue=False, api_visibility="private", show_progress="hidden"
    )
    refresh.click(show_activity, **options)
    tab.select(select_activity, **options)
