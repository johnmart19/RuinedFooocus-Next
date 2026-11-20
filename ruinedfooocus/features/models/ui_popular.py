"""Models page tab: what is popular now on Civitai and Hugging Face, ready to download."""

from concurrent.futures import CancelledError
from pathlib import Path

import gradio as gr

from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui as t
from ruinedfooocus.features.models import popular_models as feed
from ruinedfooocus.features.models.resource_cards import default_preview

ALL = "All"


def short_count(value):
    value = int(value or 0)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def size_text(value):
    if not isinstance(value, (int, float)) or value <= 0:
        return ""
    return f"{value / 1024**3:.1f} GB" if value >= 1024**3 else f"{value / 1024**2:.0f} MB"


def is_installed(entry, manager):
    if not entry.get("file"):
        return False
    try:
        path = feed.installed_path(entry, manager)
        return bool(path and path.is_file())
    except (OSError, KeyError):
        return False


def families(entries):
    return sorted({entry["base"] for entry in entries if entry.get("base")})


def card_data(entries, selected, manager, family=ALL, locale=None):
    """The shared card grid's value for one list."""
    from ruinedfooocus.integrations.media_paths import card_picture, card_picture_roots

    roots = card_picture_roots(manager)
    shown = [entry for entry in entries if family in (ALL, None, "") or entry["base"] == family]
    cache = manager.model_paths["cache_path"]
    cards = []
    for entry in shown:
        fallback = default_preview(entry["category"], cache)
        preview = feed.thumbnail_path(entry, cache)
        preview = str(preview) if preview is not None and preview.is_file() else None
        cards.append(
            dict(
                key=entry["key"],
                artworkKey="",
                label=entry["name"],
                # A chat repository is told apart by its publisher, an image model by its base.
                kind=entry["version"] if entry["source"] == "Hugging Face" else entry["base"],
                state=" · ".join(
                    filter(
                        None,
                        (
                            localize("Installed", locale) if is_installed(entry, manager) else "",
                            "↓ " + short_count(entry["downloads"]),
                            entry.get("published", ""),
                        ),
                    )
                ),
                preview=card_picture(preview or fallback, roots),
                fallback=card_picture(fallback, roots),
                preview_is_default=preview is None,
                caption=entry["base"],
            )
        )
    return dict(
        heading=localize("Popular models", locale),
        acceptsDrop=False,
        empty=localize("Nothing to show for this selection.", locale),
        selected=selected if any(card["key"] == selected for card in cards) else "",
        defaultPreviewLabel="",
        cards=cards,
    )


def details_text(entry, manager, locale=None):
    """What is known about the selected entry, as the publisher's listing states it."""
    if entry is None:
        return localize("Select a model to see its details.", locale)
    facts = [
        f"**{entry['name']}**"
        + (
            f" — {entry['version']}"
            if entry.get("version") and entry["source"] != "Hugging Face"
            else ""
        ),
        localize("Publisher: {name}", locale, name=entry["version"])
        if entry["source"] == "Hugging Face"
        else localize("Base model: {base}", locale, base=entry["base"]),
        localize(
            "{downloads} downloads · {likes} likes",
            locale,
            downloads=short_count(entry["downloads"]),
            likes=short_count(entry["likes"]),
        ),
    ]
    if entry.get("published"):
        facts.append(localize("Published: {date}", locale, date=entry["published"]))
    if entry.get("file"):
        facts.append(
            localize("File: {file}", locale, file=entry["file"])
            + (f" ({size_text(entry.get('bytes'))})" if size_text(entry.get("bytes")) else "")
        )
    else:
        facts.append(
            localize("The file is chosen when you download: a 4–5 bit GGUF that fits.", locale)
        )
    if is_installed(entry, manager):
        facts.append(localize("Already in your models folder.", locale))
    facts.append(f"[{entry['page'].split('?')[0]}]({entry['page']})")
    return "  \n".join(facts)


def create_ui(manager, models):
    """Build the tab body. Returns the handles the caller needs to refresh it on open."""
    source_default = feed.SOURCES[0]
    entries, selection = gr.State([]), gr.State("")
    with gr.Column(elem_id="popular-models", min_width=0):
        gr.Markdown(
            t(
                "What people download now, limited to models this app can run. "
                "Lists come from the public Civitai and Hugging Face listings and refresh by themselves."
            )
        )
        with gr.Group(elem_classes=["rf-section"]):
            with gr.Row(elem_classes=["rf-filters"]):
                source = gr.Radio(
                    list(feed.SOURCES), value=source_default, label=t("Source"), min_width=160
                )
                category = gr.Dropdown(
                    list(feed.CATEGORIES[source_default]),
                    value=feed.CATEGORIES[source_default][0],
                    label=t("Category"),
                )
                sort = gr.Dropdown(
                    list(feed.SORTS[source_default]),
                    value=next(iter(feed.SORTS[source_default])),
                    label=t("Sort by"),
                )
            with gr.Row(elem_classes=["rf-filters"]):
                period = gr.Dropdown(list(feed.PERIODS), value="This month", label=t("Period"))
                family = gr.Dropdown([ALL], value=ALL, label=t("Base model"))
                mature = gr.Checkbox(
                    label=t("Include models marked mature"),
                    info=t("Only pictures rated for everyone are shown either way."),
                    value=False,
                )
        with gr.Row(
            equal_height=True,
            elem_classes=["rf-actions", "rf-actions-compact"],
        ):
            note = gr.Markdown(elem_id="popular-models-note")
            refresh = gr.Button(t("Refresh now"), size="sm", scale=0, min_width=0)
        details = gr.Markdown(t("Select a model to see its details."))
        with gr.Row(elem_classes=["rf-actions"]):
            download = gr.Button(
                t("Download"), interactive=False, variant="primary", elem_id="popular-download"
            )
            cancel = gr.Button(t("Cancel this download"), visible=False, size="sm")
        progress = gr.Markdown(elem_id="popular-models-progress")
        assets = Path(__file__).resolve().parents[2] / "ui/assets"
        cards = gr.HTML(
            value=dict(
                heading=localize("Popular models"),
                acceptsDrop=False,
                empty=localize("Open this tab to load the list."),
                selected="",
                defaultPreviewLabel="",
                cards=[],
            ),
            html_template=(assets / "adapter_cards.html").read_text(encoding="utf-8"),
            css_template=(assets / "adapter_cards.css").read_text(encoding="utf-8"),
            js_on_load=(assets / "adapter_cards.js").read_text(encoding="utf-8"),
            elem_id="popular-cards",
        )

    def options(chosen):
        categories = list(feed.CATEGORIES[chosen])
        sorts = list(feed.SORTS[chosen])
        return (
            gr.update(choices=categories, value=categories[0]),
            gr.update(choices=sorts, value=sorts[0]),
            *[gr.update(visible=chosen == "Civitai")] * 3,
        )

    def note_text(result, shown, locale):
        if result["error"]:
            text = localize(result["error"], locale)
            if result["stale"] and result["fetched"]:
                text += " " + localize(
                    "Showing the list from {time}.", locale, time=result["fetched"]
                )
            return text
        return localize(
            "{count} supported models · updated {time}",
            locale,
            count=shown,
            time=result["fetched"] or "—",
        )

    def show(chosen, kind, order, span, with_mature, base, force, request):
        """Saved list first, then the fresh one, then again once artwork has arrived."""
        locale = resolve_locale(None, request)
        if kind not in feed.CATEGORIES.get(chosen, ()) or order not in feed.SORTS.get(chosen, {}):
            # The source changed and its dependent choices have not arrived yet.
            kind, order = feed.CATEGORIES[chosen][0], next(iter(feed.SORTS[chosen]))
        chosen_list = (chosen, kind, order, span if span in feed.PERIODS else "This month")
        cache = manager.model_paths["cache_path"]
        options = dict(cache_dir=cache, include_mature=bool(with_mature))

        def view(result, waiting=""):
            rows = result["entries"]
            bases = families(rows)
            shown_base = base if base in bases else ALL
            value = card_data(rows, "", manager, shown_base, locale)
            if waiting:
                value["empty"] = waiting
            return (
                rows,
                value,
                "",
                gr.update(choices=[ALL, *bases], value=shown_base),
                waiting or note_text(result, len(value["cards"]), locale),
                details_text(None, manager, locale),
                gr.update(interactive=False),
            )

        saved = feed.peek(*chosen_list, **options)
        if force or saved is None or saved["stale"]:
            waiting = localize("Loading the {source} list…", locale, source=chosen)
            yield view(saved or dict(entries=[], fetched=None, stale=False, error=""), waiting)
        result = feed.load(*chosen_list, force=force, **options)
        yield view(result)
        if feed.fetch_artwork(result["entries"], cache):
            # Pictures only: a model chosen while they arrived stays chosen.
            rows, value, _, shown, *_ = view(result)
            value.pop("selected")
            yield rows, value, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    def open_list(chosen, kind, order, span, with_mature, base, request: gr.Request = None):
        yield from show(chosen, kind, order, span, with_mature, base, False, request)

    def refresh_list(chosen, kind, order, span, with_mature, base, request: gr.Request = None):
        yield from show(chosen, kind, order, span, with_mature, base, True, request)

    inputs = [source, category, sort, period, mature, family]
    outputs = [entries, cards, selection, family, note, details, download]
    # Listing requests only read public pages and the cache, so several may run;
    # they must not wait behind one another or behind a download.
    event_options = dict(
        api_visibility="undocumented",
        show_progress="minimal",
        concurrency_limit=4,
        concurrency_id="popular-models",
    )
    listing = dict(inputs=inputs, outputs=outputs, **event_options)
    source.change(
        options,
        inputs=source,
        outputs=[category, sort, period, mature, family],
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    # A new source replaces the category and sort choices, which loads its list.
    gr.on(
        [category.change, sort.change, period.change, mature.change],
        open_list,
        trigger_mode="always_last",
        **listing,
    )
    refresh.click(refresh_list, **listing)

    def narrow(rows, base, request: gr.Request = None):
        locale = resolve_locale(None, request)
        value = card_data(rows, "", manager, base, locale)
        return value, "", details_text(None, manager, locale), gr.update(interactive=False)

    family.change(
        narrow,
        inputs=[entries, family],
        outputs=[cards, selection, details, download],
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def find(rows, key):
        return next((entry for entry in rows if entry["key"] == key), None)

    def select(rows, evt: gr.EventData, request: gr.Request = None):
        key = evt._data.get("key") if isinstance(evt._data, dict) else None
        entry = find(rows, key)
        locale = resolve_locale(None, request)
        return (
            entry["key"] if entry else "",
            details_text(entry, manager, locale),
            gr.update(interactive=entry is not None and not is_installed(entry, manager)),
        )

    cards.click(
        select,
        inputs=[entries],
        outputs=[selection, details, download],
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def run_download(rows, key, base, request: gr.Request = None):
        locale = resolve_locale(None, request)
        entry = find(rows, key)
        if entry is None:
            raise gr.Error(localize("Select a model first.", locale))
        state = {}
        busy = gr.update(interactive=False), gr.update(visible=True)
        idle = gr.update(visible=False)
        yield localize("Starting…", locale), *busy, gr.skip(), gr.skip(), gr.skip()
        subscription = None
        try:
            entry = feed.resolve_file(entry)
            destination, subscription = feed.start_download(
                entry,
                manager,
                progress=lambda received, total, speed: state.update(
                    received=received, total=total, speed=speed
                ),
            )
            while subscription is not None:
                try:
                    subscription.result(timeout=0.5)
                    break
                except TimeoutError:
                    received, total = state.get("received", 0), state.get("total", 0)
                    total = total or entry.get("bytes") or 0
                    yield (
                        localize(
                            "Downloading {file}: {done} of {total} ({speed}/s)",
                            locale,
                            file=entry["file"],
                            done=size_text(received) or "0 MB",
                            total=size_text(total) or "?",
                            speed=size_text(state.get("speed", 0)) or "0 MB",
                        ),
                        *busy,
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                    )
            subscription = None
        except CancelledError:
            yield (
                localize("Download cancelled.", locale),
                gr.update(interactive=True),
                idle,
                *([gr.skip()] * 3),
            )
            return
        except Exception as error:  # Shown to the reader; nothing is left half-written.
            subscription = None
            yield (
                localize(feed.explain_failure(error, entry), locale),
                gr.update(interactive=True),
                idle,
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )
            return
        finally:
            # Leaving the page or pressing Cancel stops the transfer.
            if subscription is not None:
                subscription.cancel()
        models.update_all_models()
        if destination.suffix.lower() == ".gguf":
            from ruinedfooocus.features.chat.llama_models import invalidate_model_catalogue

            invalidate_model_catalogue()
        # A Hugging Face entry learns its file here, so the card can say Installed.
        rows = [entry if row["key"] == entry["key"] else row for row in rows]
        yield (
            localize("Saved and verified: {path}", locale, path=str(destination)),
            gr.update(interactive=False),
            idle,
            rows,
            card_data(rows, key, manager, base, locale),
            details_text(entry, manager, locale),
        )

    running = download.click(
        run_download,
        inputs=[entries, selection, family],
        outputs=[progress, download, cancel, entries, cards, details],
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def cancelled(request: gr.Request = None):
        return (
            localize("Download cancelled.", resolve_locale(None, request)),
            gr.update(visible=False),
            gr.update(interactive=True),
        )

    cancel.click(
        cancelled,
        outputs=[progress, cancel, download],
        cancels=[running],
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    return dict(open=open_list, inputs=inputs, outputs=outputs, options=event_options)
