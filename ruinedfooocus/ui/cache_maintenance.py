"""Settings exposes targeted derived-cache operations through bounded preparation."""

import asyncio
import gradio as gr

from ruinedfooocus.app.preparation_admission import preparation_admission
from ruinedfooocus.core.cache_maintenance import cache_roots, clear_category, maintenance_categories
from ruinedfooocus.core.translation_manager import (
    localize,
    localized_output,
    resolve_locale,
    translate_ui as t,
)


def _size(value):
    return f"{value / 1024**2:.2f} MiB"


def cache_location_text(roots, locale=None):
    locations = [f"{localize('Model and Library cache', locale)}: `{roots[0]}`"]
    if len(roots) > 1:
        locations.append(f"{localize('Uploads and runtime cache', locale)}: `{roots[1]}`")
    return "\n\n".join(locations)


def cache_headers(locale=None):
    return [
        localize("Category", locale),
        localize("Files", locale),
        localize("Size", locale),
        localize("Retention", locale),
    ]


def create_cache_maintenance(manager):
    with gr.Accordion(t("Cache storage"), open=False):
        roots = cache_roots(manager)
        locations = gr.Markdown(cache_location_text(roots))
        localized_output(lambda locale: cache_location_text(roots, locale), [], [locations])
        gr.Markdown(
            t(
                "Refresh sizes to inspect cache storage. Inputs, model files and installed runtimes are retained. Counts are bounded snapshots."
            )
        )
        table = gr.Dataframe(
            headers=cache_headers(),
            datatype=["str", "number", "str", "str"],
            value=[],
            interactive=False,
            wrap=True,
        )
        localized_output(lambda locale: gr.update(headers=cache_headers(locale)), [], [table])
        category = gr.Dropdown(
            choices=[
                (t("Model hash records"), "model-hashes"),
                (t("Library index"), "library-index"),
                (t("Library thumbnails"), "library-thumbnails"),
                (t("Startup verification"), "startup-checks"),
            ],
            value="model-hashes",
            label=t("Cache to clear"),
        )
        with gr.Row():
            refresh = gr.Button(t("Refresh cache sizes"))
            clear = gr.Button(t("Clear selected cache"))
        status = gr.Markdown()

        async def sizes(request: gr.Request = None):
            locale = resolve_locale(request=request)
            rows = await asyncio.wrap_future(
                preparation_admission.submit_work(lambda: maintenance_categories(manager))
            )
            return [
                [
                    localize(row["label"], locale),
                    row["files"],
                    _size(row["bytes"]),
                    localize(row["reason"], locale),
                ]
                for row in rows
            ]

        async def clear_selected(value, request: gr.Request = None):
            locale = resolve_locale(request=request)
            result = await asyncio.wrap_future(
                preparation_admission.submit_work(lambda: clear_category(value, manager))
            )
            text = localize(
                "Cleared {count} derived cache entries; skipped {skipped} active or retained entries.",
                locale,
                count=result["removed"],
                skipped=result["skipped"],
            )
            if result["errors"]:
                text += "\n\n" + "\n".join(result["errors"])
            if value == "library-index":
                text += "\n\n" + localize(
                    "Refresh Library to rebuild its index. Original images are preserved.", locale
                )
            return text

        refresh.click(
            sizes,
            outputs=table,
            concurrency_id="cache_maintenance",
            concurrency_limit=1,
            trigger_mode="always_last",
            show_progress="minimal",
            show_progress_on=table,
            api_visibility="undocumented",
        )
        clear.click(
            clear_selected,
            inputs=category,
            outputs=status,
            concurrency_id="cache_maintenance",
            concurrency_limit=1,
            show_progress="minimal",
            show_progress_on=status,
            api_visibility="undocumented",
        )
