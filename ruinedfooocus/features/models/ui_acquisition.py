"""One Models acquisition view over the existing catalogue and preparation jobs."""

from pathlib import Path
from concurrent.futures import CancelledError
from collections import Counter
from collections.abc import Mapping
import re
from types import SimpleNamespace
from urllib.parse import quote, urlsplit

import gradio as gr

from ruinedfooocus.core.translation_manager import (
    localize,
    localized_output,
    resolve_locale,
    translate_ui as t,
)
from ruinedfooocus.features.models.acquisition import (
    AcquisitionSession,
    Catalogue,
    CATEGORIES,
    checkpoint_name,
    inspected_local,
    size_label,
)
from ruinedfooocus.features.models.model_previews import (
    SUFFIXES,
    is_warning,
    local_preview,
    valid_preview,
)
from ruinedfooocus.features.models.resource_cards import default_preview
from ruinedfooocus.features.models.selection import checkpoint_choices, model_label
from ruinedfooocus.features.models.acquisition_actions import (
    action_state,
    component_targets,
    use_component,
)
from ruinedfooocus.features.models.catalogue_previews import CataloguePreviews, preview_identity
from ruinedfooocus.core.preparation import preparation_service
from ruinedfooocus.features.image_editing.tool_activation import installed_tool
from ruinedfooocus.features.models.ui_requests import (
    current_identity,
    current_source,
    project,
    require_source,
    stamp,
    validate_source,
    validate_selection,
    validate_view,
    version_js,
)

PAGE_SIZE = 36


def quantization_label(value):
    if value in (None, "", "Unknown", "unknown", "unspecified"):
        return ""
    return "Original" if value == "unspecified_original" else value


def cached_preview(row, manager):
    candidates = []
    if row["local"]:
        local = local_preview(row["local"], ambiguous=row.get("_ambiguous_sidecar", False))
        if local:
            return local
    filename = manager.DOWNLOADABLE_FILES.get(row["key"], {}).get(
        "filename", Path(row["destination"]).name
    )
    cache = manager.model_paths.get("cache_path")
    if cache and not row.get("_ambiguous_preview_cache"):
        for category in (
            row["category"],
            "checkpoints" if row["category"] == "diffusion_models" else row["category"],
        ):
            base = Path(cache) / category / Path(filename).name
            candidates.extend(base.with_suffix(suffix) for suffix in SUFFIXES)
    return next(
        (
            str(path)
            for path in candidates
            if path.is_file() and valid_preview(path) and not is_warning(path)
        ),
        None,
    )


def preview_cache_key(row):
    category = row["category"]
    if category == "diffusion_models":
        category = "checkpoints"
    return category, Path(row["local"] or row["destination"]).stem.casefold()


def preview_sidecar_key(row):
    if row["local"]:
        return str(Path(row["local"]).with_suffix("")).casefold()
    return None


def listed_records(rows):
    return [row for row in rows if not row.get("_details_only")]


def card_label(row):
    label = row["label"]
    is_filename = bool(re.search(r"\.(?:safetensors|gguf|ckpt|pt|pth|bin)$", label, re.I))
    return model_label(label, None if is_filename else {"rf_display_name": label})


def card_data(rows, selected, manager, page=1, locale=None, previews=None):
    rows = listed_records(rows)
    image_only = bool(rows) and all(row["category_label"] == "Image models" for row in rows)
    pages = max(1, (len(rows) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(1, min(int(page or 1), pages))
    cards = []
    for row in rows[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]:
        preview = cached_preview(row, manager) or (previews.cached(row) if previews else None)
        fallback = default_preview(row["category_label"], manager.model_paths["cache_path"])
        cards.append(
            dict(
                key=row["key"],
                artworkKey=(preview_identity(row) or {}).get("key", ""),
                label=card_label(row),
                kind="" if image_only else localize(row["category_label"], locale),
                state=localize("Custom", locale)
                if row["key"].startswith("local:")
                else " · ".join(
                    filter(
                        None,
                        (
                            localize(row["status"], locale),
                            quantization_label(row["quantization"]),
                            row["task"] if row["task"] == "Bundle" else "",
                        ),
                    )
                ),
                preview=gr.data_classes.FileData(path=preview or fallback).model_dump(),
                fallback=gr.data_classes.FileData(path=fallback).model_dump(),
                preview_is_default=not preview or preview == fallback,
            )
        )
    return dict(
        heading=localize("Models and downloads", locale),
        acceptsDrop=False,
        empty=localize("No matching models.", locale),
        selected=selected or "",
        defaultPreviewLabel=localize("Default preview", locale),
        cards=cards,
        pagination=dict(
            page=page,
            pages=pages,
            previous=localize("Previous", locale),
            next=localize("Next", locale),
            label=localize("Page {page} of {pages}", locale, page=page, pages=pages),
        ),
    ), page


def detail_rows(row, locale=None):
    if row is None:
        return [], localize("Select a model or component.", locale), False, False
    profile, annotation = row["profile"], row["annotation"] or {}
    license_info = profile.get("license") or {}
    if not isinstance(license_info, Mapping):
        license_info = {"value": str(license_info)}
    estimates = (
        annotation.get("hardware", {}).get("planning_profiles")
        or profile.get("resource_estimates")
        or ()
    )
    notes = profile.get("hardware_notes") or ()
    memory = "; ".join(str(note) for note in notes[:1]) if isinstance(notes, (list, tuple)) else ""
    scenarios = (
        [
            f"{estimate.get('mode', 'unknown')} · context {estimate.get('context_tokens', 'unknown')} tokens · "
            f"{estimate.get('estimated_full_offload_vram_range_gib', 'unknown')} GiB estimated full-offload VRAM · "
            f"{estimate.get('recommended_system_ram_gib', 'unknown')} GiB system RAM"
            for estimate in estimates
            if isinstance(estimate, Mapping)
        ]
        if isinstance(estimates, (list, tuple))
        else []
    )
    if scenarios:
        memory = "Unmeasured planning envelope. " + "\n".join(scenarios)
    if not memory:
        memory = "No workflow memory estimate recorded."
    entries = [
        [localize("Model", locale), profile.get("title") or profile.get("label") or row["label"]],
        ["File", row["local"] or row["destination"]],
        ["Workflow", row["support"]],
        [
            "Format / quantization",
            " / ".join(filter(None, (row["format"], quantization_label(row["quantization"])))),
        ],
        ["Published file size", size_label(row["published_bytes"])],
        ["Local file size", size_label(row["local_bytes"]) if row["local"] else "Not installed"],
        [
            "Integrity",
            "Size differs; inspect before use."
            if not row["size_matches"]
            else "Presence is not a checksum or execution check.",
        ],
        ["Memory suitability", memory],
        [
            "License",
            license_info.get("declared_upstream") or license_info.get("value") or "Unknown",
        ],
        ["Source", row["source"]],
        [
            "License source",
            license_info.get("source")
            or ", ".join(license_info.get("source_urls", ()))
            or "Unknown",
        ],
        ["Declared dependencies", "; ".join(profile.get("dependencies", ())) or "None recorded"],
    ]
    identity, provenance = profile.get("identity"), profile.get("provenance")
    repository = (
        identity.get("checkpoint_repo_resolved") if isinstance(identity, Mapping) else None
    ) or (provenance.get("upstream") if isinstance(provenance, Mapping) else None)
    if repository:
        entries.append([localize("Recorded model source", locale), repository])
    runtime = annotation.get("runtime", {}).get("native_llama_cpp", {})
    if runtime.get("tested_source_build") and scenarios:
        entries.append(
            [
                localize("Reference runtime", locale),
                "Native llama.cpp " + runtime["tested_source_build"],
            ]
        )
    companion_ids = set(annotation.get("companions", {}).get("text", ())) | set(
        annotation.get("companions", {}).get("image", ())
    )
    entries.append(["Registered companions", ", ".join(sorted(companion_ids)) or "None recorded"])
    label = re.sub(r"([\\`*_{}\[\]()#+.!|<>])", r"\\\1", " ".join(card_label(row).splitlines()))
    status = f"**{label}**  \n{row['status']} · {row['support']}"
    if row["reason"]:
        status += "  \n" + row["reason"]
    for label, url in (
        ("Source", row["source"]),
        (
            "License",
            license_info.get("source") or next(iter(license_info.get("source_urls", ())), ""),
        ),
    ):
        parsed = urlsplit(url)
        if parsed.scheme == "https" and parsed.hostname and not parsed.username:
            status += " · [" + label + "](" + quote(url, safe=":/?&=%#@") + ")"
    if row.get("_details_only"):
        status += "\n\n" + localize("The selected file is outside the current list.", locale)
    return (
        [[localize(label, locale), value] for label, value in entries],
        status,
        bool(row["allowed"] and not row["local"]),
        row["can_activate"],
    )


def create_ui(manager, models, *, offline):
    """Caller supplies the already configured profile/runtime and navigation container."""
    catalogue = Catalogue(manager)
    session = gr.State(AcquisitionSession(), delete_callback=lambda job: job.cancel())
    selection, records, revision = gr.State(None), gr.State([]), gr.State(0)
    action_revision, activation_context = gr.State(0), gr.State(None)
    view_version = gr.Number(0, visible=False, precision=0)
    details_version = gr.Number(0, visible=False, precision=0)
    selection_version = gr.Number(0, visible=False, precision=0)
    page_version = gr.Number(0, visible=False, precision=0)
    roles_version = gr.Number(0, visible=False, precision=0)
    previews = CataloguePreviews(Path(manager.model_paths["cache_path"]) / "catalogue-previews")
    preview_revision = gr.State(0)
    with gr.Column(elem_id="model-acquisition", min_width=0):
        gr.Markdown(
            t(
                "Browse installed files and available downloads. Downloading never changes your active model."
            )
        )
        # Search, library and the three filters are one "find" card.
        with gr.Group(elem_classes=["rf-section"]):
            with gr.Row(elem_classes=["rf-filters"]):
                query = gr.Textbox(
                    label=t("Search models and components"),
                    placeholder=t("Name, filename or workflow"),
                    scale=3,
                )
                view = gr.Radio(
                    [(t("Discover"), "discover"), (t("On device"), "on-device")],
                    value="discover",
                    label=t("Model library"),
                    scale=2,
                    min_width=160,
                )
            with gr.Row(elem_classes=["rf-filters"]):
                category = gr.Dropdown(
                    ["All", *sorted({"Image tools", *CATEGORIES.values()})],
                    value="All",
                    label=t("Category"),
                )
                task = gr.Dropdown(
                    [
                        "All",
                        "Image",
                        "Chat",
                        "Component",
                        "Image tools",
                        "Local import",
                        "Bundle",
                    ],
                    value="All",
                    label=t("Workflow"),
                )
                quantization = gr.Dropdown(["All"], value="All", label=t("Quantization"))
        # Result count and catalogue refresh share one line above the cards.
        with gr.Row(
            elem_id="model-catalogue-actions",
            equal_height=True,
            elem_classes=["rf-actions", "rf-actions-compact"],
        ):
            count = gr.Markdown(elem_id="model-catalogue-count")
            refresh = gr.Button(t("Refresh local catalogue"), size="sm", scale=0, min_width=0)
        page = gr.State(1)
        assets = Path(__file__).resolve().parents[2] / "ui/assets"
        cards = gr.HTML(
            value=dict(
                heading=localize("Models and downloads"),
                acceptsDrop=False,
                empty=localize("Refresh to browse models."),
                selected="",
                defaultPreviewLabel=localize("Default preview"),
                cards=[],
            ),
            html_template=(assets / "adapter_cards.html").read_text(encoding="utf-8"),
            css_template=(assets / "adapter_cards.css").read_text(encoding="utf-8"),
            js_on_load=(assets / "adapter_cards.js").read_text(encoding="utf-8"),
            elem_id="acquisition-cards",
        )
        status = gr.Markdown(t("Select a model or component."))
        component_role = gr.Dropdown(label=t("Component role"), choices=[], visible=False)
        with gr.Accordion(t("Source, license, compatibility and resources"), open=False):
            details = gr.Dataframe(
                headers=[localize("Property"), localize("Recorded information")],
                datatype=["str", "str"],
                interactive=False,
                wrap=True,
            )
        with gr.Row(elem_classes=["rf-actions"]):
            download = gr.Button(
                t("Download selected files"), interactive=False, visible=False, variant="primary"
            )
            activate = gr.Button(
                t("Use installed checkpoint in Create"),
                interactive=False,
                visible=False,
                variant="primary",
            )
        with gr.Accordion(t("Download options"), open=False):
            companions = gr.Checkbox(label=t("Include registered companions"), value=True)
            mode = gr.Radio(
                [("Text companions", "text"), ("Image companions", "image")],
                value="text",
                label=t("Companion set"),
            )
            reserve = gr.Number(
                label=t("Keep free disk space (GiB)"), value=100, minimum=0, maximum=10000
            )
            unknown = gr.Checkbox(label=t("Allow downloads with unknown total size"), value=False)
        progress = gr.Markdown(t("No download running."))
        cancel = gr.Button(t("Cancel this download"), interactive=False, visible=False, size="sm")
        timer = gr.Timer(0.5, active=False)
        artwork_timer = gr.Timer(1, active=True)

    def refresh_rows(
        search, kind, view, workflow, quant, selected, current_page, request=None, ticket=None
    ):
        try:
            cancelled = ticket.cancelled if ticket is not None else None
            on_device = view == "on-device"
            all_rows = catalogue.records(cancelled=cancelled)
            preview_cache = Counter(preview_cache_key(row) for row in all_rows)
            sidecars = Counter(preview_sidecar_key(row) for row in all_rows if row["local"])
            rows = catalogue.records(
                query=search or "",
                category=kind,
                status="Installed" if on_device else "All",
                task=workflow,
                quantization=quant,
                include_local=on_device,
                cancelled=cancelled,
            )
            matched = len(rows)
            selected = current_source("acquisition-selection", request, selected)
            selected_row = next((row for row in all_rows if row["key"] == selected), None)
            if selected_row is not None and not any(row["key"] == selected for row in rows):
                rows.append({**selected_row, "_details_only": True})
            rows = [
                {
                    **row,
                    "_ambiguous_preview_cache": preview_cache[preview_cache_key(row)] > 1,
                    "_ambiguous_sidecar": sidecars[preview_sidecar_key(row)] > 1,
                }
                for row in rows
            ]
            all_quantizations = sorted(
                {
                    row["quantization"]
                    for row in catalogue.records(include_local=False, cancelled=cancelled)
                }
            )
            selected = selected if selected_row is not None else None
            data, current_page = card_data(
                rows, selected, manager, current_page, resolve_locale(request=request), previews
            )
            if ticket is not None:
                rows = [row | {"_view_version": ticket.version} for row in rows]
                if not rows:
                    rows = [{"key": "", "_details_only": True, "_view_version": ticket.version}]
                data = stamp(data, "acquisition-view", ticket.version)
                data["selectionIdentity"] = current_identity("acquisition-selection", request)
            visible = {card["key"] for card in data["cards"]}
            previews.request(
                [row for row in rows if row["key"] in visible and not cached_preview(row, manager)],
                offline=offline(),
            )
            return (
                rows,
                data,
                selected if ticket is None else gr.skip(),
                current_page,
                localize("{count} matching files", resolve_locale(request=request), count=matched),
                gr.update(
                    choices=[
                        ("All", "All"),
                        *(
                            (quantization_label(value) or "Unspecified", value)
                            for value in all_quantizations
                        ),
                    ],
                    value=quant if quant in all_quantizations else "All",
                ),
            )
        except (OSError, ValueError) as error:
            raise gr.Error(str(error)) from None

    filters = [query, category, view, task, quantization]
    refresh_inputs = filters + [selection, page, view_version]

    def refresh_local(
        search,
        kind,
        view,
        workflow,
        quant,
        selected,
        current_page,
        request: gr.Request = None,
        version=None,
    ):
        def build(ticket):
            catalogue.invalidate()
            return refresh_rows(
                search, kind, view, workflow, quant, selected, current_page, request, ticket
            )

        return project("acquisition-view", version, request, build, outputs=6)

    def filter_rows(
        search,
        kind,
        view,
        workflow,
        quant,
        selected,
        current_page,
        request: gr.Request = None,
        version=None,
    ):
        def build(ticket):
            result = list(
                refresh_rows(
                    search, kind, view, workflow, quant, selected, current_page, request, ticket
                )
            )
            result[-1] = (
                gr.skip()
            )  # Typing/filtering never reconfigures its own quantization input.
            return tuple(result)

        return project("acquisition-view", version, request, build, outputs=6)

    gr.on(
        triggers=[query.input, category.change, view.change, task.change, quantization.change],
        fn=filter_rows,
        inputs=refresh_inputs,
        outputs=[records, cards, selection, page, count, quantization],
        trigger_mode="multiple",
        queue=False,
        js=version_js("acquisition-view"),
        show_progress="minimal",
        show_progress_on=count,
        api_visibility="undocumented",
    )
    gr.on(
        triggers=[refresh.click, revision.change],
        fn=refresh_local,
        inputs=refresh_inputs,
        outputs=[records, cards, selection, page, count, quantization],
        show_progress="minimal",
        show_progress_on=count,
        trigger_mode="multiple",
        queue=False,
        js=version_js("acquisition-view"),
        api_visibility="undocumented",
    )

    def select_record(
        rows,
        companion_mode,
        include_companions,
        evt: gr.EventData,
        job=None,
        context=None,
        request: gr.Request = None,
        version=None,
    ):
        key = evt._data.get("key")
        if not any(row["key"] == key for row in listed_records(rows)):
            raise gr.Error("The list changed. Select the model again.")

        def build(ticket):
            if ticket is not None:
                return (key, *(gr.skip() for _ in range(4)))
            return key, *_selected_details(
                key, rows, companion_mode, include_companions, job, context
            )

        return project(
            "acquisition-selection",
            version,
            request,
            build,
            outputs=5,
            source=key,
            prepare=False,
            validate=lambda: validate_source(key, rows, request),
        )

    cards.click(
        select_record,
        inputs=[records, mode, companions, session, activation_context, selection_version],
        outputs=[selection, details, status, download, activate],
        queue=False,
        trigger_mode="multiple",
        js=version_js("acquisition-selection"),
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def targets_for(row, context):
        if not context or not context.get("checkpoint"):
            return []
        try:
            return component_targets(
                row,
                context["checkpoint"],
                context["family"],
                component_values(),
                manager,
                context.get("overrides"),
            )
        except (OSError, ValueError):
            return []

    def component_values():
        import ruinedfooocus.app.state as shared

        return shared.settings.default_settings

    def _selected_details(
        key,
        rows,
        companion_mode="text",
        include_companions=True,
        job=None,
        context=None,
        request: gr.Request = None,
        locale=None,
    ):
        locale = locale or resolve_locale(request=request)
        row = next((row for row in rows if row["key"] == key), None)
        if row is not None:
            row = inspected_local(row, models)
        table, text, _, can_activate = detail_rows(row, locale)
        missing_companions, transfer, targets, tool = False, None, [], None
        if row is not None:
            companions_state, missing = catalogue.companion_status(row, companion_mode)
            table.extend([["Companion", " · ".join(values)] for values in companions_state])
            table.append(["Missing registered companion payloads", size_label(missing)])
            if row["allowed"] and row["size_matches"] and include_companions:
                missing_companions = any(values[1] == "Available" for values in companions_state)
            if not row["local"] and row["allowed"]:
                try:
                    transfer = preparation_service.transfer_status(
                        manager.preparation_request(row["key"])
                    )
                    if transfer["state"] != "absent":
                        table.append(
                            [
                                "Transfer state",
                                transfer["state"] + " · " + size_label(transfer["received_bytes"]),
                            ]
                        )
                        text += "\n\n" + transfer.get("reason", "")
                except (ValueError, KeyError, OSError):
                    pass
            targets = targets_for(row, context)
            can_activate = can_activate or bool(targets)
            try:
                tool = installed_tool(row, manager)
            except (OSError, ValueError):
                tool = None
            can_activate = can_activate or tool is not None
            if (
                row["local"]
                and row["category"] in ("clip", "vae", "clip_vision")
                and not targets
                and tool is None
            ):
                text += "\n\n" + localize(
                    "No compatible external role on the selected checkpoint. Choose another checkpoint or inspect its component choices.",
                    locale,
                )
        actions = action_state(
            row,
            can_use=can_activate,
            companions_missing=missing_companions,
            transfer=transfer,
            job=job.snapshot() if job else None,
            offline=offline(),
        )
        labels = {
            "Download": localize("Download selected files", locale),
            "Download missing companions": localize("Download missing companions", locale),
            "Resume download": localize("Resume download", locale),
            "Retry download": localize("Retry download", locale),
        }
        return (
            table,
            text,
            gr.update(
                value=labels[actions["label"]],
                visible=actions["download"],
                interactive=actions["download_enabled"],
                variant="secondary" if actions["secondary"] else "primary",
            ),
            gr.update(
                value=localize("Use for selected checkpoint", locale)
                if targets
                else localize("Select installed file in Chat", locale)
                if row and row["task"] == "Chat"
                else localize("Use installed file in Image tools", locale)
                if tool is not None
                or row
                and row["category"] in ("upscalers", "background_removal", "geometry_estimation")
                else localize("Use installed checkpoint in Create", locale),
                visible=actions["use"],
                interactive=actions["use"],
            ),
        )

    def selected_details(
        key,
        rows,
        companion_mode="text",
        include_companions=True,
        job=None,
        context=None,
        request: gr.Request = None,
        version=None,
        locale=None,
    ):
        return project(
            "acquisition-details",
            version,
            request,
            lambda ticket: _selected_details(
                key, rows, companion_mode, include_companions, job, context, request, locale
            ),
            outputs=4,
            source=key,
            validate=lambda: validate_selection(key, rows, request),
        )

    gr.on(
        triggers=[
            selection.change,
            records.change,
            mode.change,
            companions.change,
            action_revision.change,
            activation_context.change,
            cards.click,
        ],
        fn=selected_details,
        inputs=[selection, records, mode, companions, session, activation_context, details_version],
        outputs=[details, status, download, activate],
        show_progress="minimal",
        show_progress_on=status,
        queue=False,
        trigger_mode="multiple",
        js=version_js("acquisition-details"),
        api_visibility="undocumented",
    )
    localized_output(
        lambda locale, key, rows, companion_mode, include, job, context: selected_details(
            key, rows, companion_mode, include, job, context, locale=locale
        ),
        [selection, records, mode, companions, session, activation_context],
        [details, status, download, activate],
    )

    def update_cancel_action(key, job):
        snapshot = job.snapshot()
        return gr.update(
            visible=snapshot["active"],
            interactive=snapshot["active"],
            variant="primary" if snapshot["selection_key"] == key else "secondary",
        )

    gr.on(
        triggers=[selection.change, action_revision.change],
        fn=update_cancel_action,
        inputs=[selection, session],
        outputs=cancel,
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def render_page(rows, chosen, current, request: gr.Request = None, version=None, locale=None):
        observer_identity = current_identity("acquisition-page", request)

        def validate():
            validate_view(rows, request)
            if (
                version is None
                and current_identity("acquisition-page", request) != observer_identity
            ):
                raise CancelledError("The catalogue page changed")

        def build(ticket):
            selected_key = current_source("acquisition-selection", request, chosen)
            payload = card_data(
                rows,
                selected_key,
                manager,
                current,
                locale or resolve_locale(request=request),
                previews,
            )[0]
            visible = {card["key"] for card in payload["cards"]}
            previews.request(
                [row for row in rows if row["key"] in visible and not cached_preview(row, manager)],
                offline=offline(),
            )
            if ticket is not None:
                payload = stamp(payload, "acquisition-view", rows[0]["_view_version"])
                payload["selectionIdentity"] = current_identity("acquisition-selection", request)
                payload["pageIdentity"] = (
                    {"channel": "acquisition-page", "version": ticket.version}
                    if version is not None
                    else observer_identity
                )
            return payload

        channel = "acquisition-page" if version is not None else "acquisition-view"
        result = project(
            channel,
            version,
            request,
            build,
            outputs=1,
            validate=validate,
        )
        return result[0] if isinstance(result, tuple) else result

    def render_requested_page(
        rows, chosen, event: gr.EventData, request: gr.Request = None, version=None
    ):
        target = event._data.get("page") if isinstance(event._data, dict) else None
        if type(target) is not int or target < 1:
            raise gr.Error(t("Choose a valid catalogue page."))
        identity = current_identity("acquisition-view", request)
        source_view = event._data.get("viewVersion")
        if identity is not None and (
            type(source_view) is not int or source_view != identity["version"]
        ):
            return gr.skip(), gr.skip()
        payload = render_page(rows, chosen, target, request, version)
        if not isinstance(payload, dict) or "pagination" not in payload:
            return gr.skip(), gr.skip()
        return payload, payload["pagination"]["page"]

    cards.paginate(
        render_requested_page,
        inputs=[records, selection, page_version],
        outputs=[cards, page],
        show_progress="hidden",
        queue=False,
        trigger_mode="multiple",
        js=version_js("acquisition-page"),
        api_visibility="undocumented",
    )
    localized_output(
        lambda locale, rows, chosen, current: (
            render_page(rows, chosen, current, locale=locale),
            gr.update(
                headers=[localize("Property", locale), localize("Recorded information", locale)]
            ),
        ),
        [records, selection, page],
        [cards, details],
    )

    def refresh_artwork(seen, rows, chosen, current, request: gr.Request = None):
        current_revision = previews.revision
        if seen == current_revision:
            return gr.skip(), gr.skip()
        visible_rows = listed_records(rows)
        current = max(1, int(current or 1))
        visible_rows = visible_rows[(current - 1) * PAGE_SIZE : current * PAGE_SIZE]
        previews.request(
            [row for row in visible_rows if not cached_preview(row, manager)], offline=offline()
        )
        patches = [
            dict(
                key=row["key"],
                artworkKey=(preview_identity(row) or {}).get("key", ""),
                preview=gr.data_classes.FileData(path=preview).model_dump(),
            )
            for row in visible_rows
            if (preview := previews.cached(row))
        ]
        # A delayed tick can update only matching artwork. It cannot restore an
        # older query, selection, card order, labels or pager in the browser.
        return current_revision, dict(previewPatch=patches)

    artwork_timer.tick(
        refresh_artwork,
        inputs=[preview_revision, records, selection, page],
        outputs=[preview_revision, cards],
        queue=False,
        trigger_mode="always_last",
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def role_targets(key, rows, context, request: gr.Request = None, version=None):
        from ruinedfooocus.features.models.ui_model_components import component_label

        def build(ticket):
            row = next((row for row in rows if row["key"] == key), None)
            targets = targets_for(row, context) if row else []
            return gr.update(
                choices=[(component_label(target["key"]), target["key"]) for target in targets],
                value=targets[0]["key"] if targets else None,
                visible=bool(targets),
                interactive=bool(targets),
            )

        result = project(
            "acquisition-roles",
            version,
            request,
            build,
            outputs=1,
            source=key,
            validate=lambda: validate_selection(key, rows, request),
        )
        return result[0] if isinstance(result, tuple) else result

    gr.on(
        triggers=[selection.change, records.change, activation_context.change],
        fn=role_targets,
        inputs=[selection, records, activation_context, roles_version],
        outputs=component_role,
        show_progress="hidden",
        api_visibility="undocumented",
        trigger_mode="multiple",
        queue=False,
        js=version_js("acquisition-roles"),
    )

    def bind_component_context(checkpoint_control, overrides_control, scope_control):
        from ruinedfooocus.features.models.ui_model_components import selected

        context_version = gr.Number(0, visible=False, precision=0)

        def capture_context(name, overrides, scope, request: gr.Request = None, version=None):
            def build(ticket):
                try:
                    path, family = selected(name)
                    return dict(
                        checkpoint=str(path), family=family, overrides=overrides, scope=scope
                    )
                except (OSError, ValueError):
                    return None

            result = project("acquisition-context", version, request, build, outputs=1, source=name)
            return result[0] if isinstance(result, tuple) else result

        activation_context.value = capture_context(
            checkpoint_control.value, overrides_control.value, scope_control.value
        )
        gr.on(
            triggers=[checkpoint_control.change, overrides_control.change, scope_control.change],
            fn=capture_context,
            inputs=[checkpoint_control, overrides_control, scope_control, context_version],
            outputs=activation_context,
            show_progress="hidden",
            api_visibility="undocumented",
            trigger_mode="multiple",
            queue=False,
            js=version_js("acquisition-context"),
        )

    def start_download(
        key,
        job,
        include,
        companion_mode,
        free_space,
        unbounded,
        generation=0,
        rows=None,
        request: gr.Request = None,
    ):
        try:
            # Selection authority belongs to the latest card acknowledgement,
            # not a possibly delayed Gradio State update.
            require_source(key, rows or [], request)
            if offline():
                raise ValueError("Offline mode: downloads are disabled.")
            plan = catalogue.capture(
                key,
                include_companions=include,
                mode=companion_mode,
                reserve_gib=free_space,
                allow_unknown=unbounded,
            )
            job.start(plan, offline=offline())
            return (
                job.message,
                gr.update(active=True),
                gr.update(interactive=True, visible=True),
                generation + 1,
            )
        except (ValueError, OSError, KeyError) as error:
            raise gr.Error(str(error)) from None

    download.click(
        start_download,
        inputs=[selection, session, companions, mode, reserve, unknown, action_revision, records],
        outputs=[progress, timer, cancel, action_revision],
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def poll(job, generation):
        from ruinedfooocus.features.models.model_identity import record_prepared_native_source

        was_running = job.active
        message, running, completed = job.poll()
        changed, source_error, chat_changed = False, False, False
        for request, path, signature in job.drain_verified():
            installed = Path(path).resolve()
            if installed.suffix.lower() == ".gguf" and any(
                installed.is_relative_to(Path(root).resolve())
                for kind in ("llm", "vision")
                for root in manager.get_model_folders(kind)
            ):
                chat_changed = True
            try:
                changed = (
                    bool(
                        record_prepared_native_source(
                            path, request, signature, manager.model_paths["cache_path"]
                        )
                    )
                    or changed
                )
            except (OSError, ValueError):
                source_error = True
        if chat_changed:
            from ruinedfooocus.features.chat.llama_models import invalidate_model_catalogue

            invalidate_model_catalogue()
        if changed:
            models.revision = getattr(models, "revision", 0) + 1
        if source_error:
            message += "\n\n" + localize(
                "Downloaded files are retained. Source details could not be cached; use Inspect before Model Defaults."
            )
        finished = was_running and not running
        if finished:
            models.update_all_models()
        return (
            message,
            gr.update(active=running),
            gr.update(interactive=running, visible=running),
            generation + int(finished),
        )

    timer.tick(
        poll,
        inputs=[session, revision],
        outputs=[progress, timer, cancel, revision],
        show_progress="hidden",
        api_visibility="undocumented",
        trigger_mode="always_last",
    )

    def cancel_download(job, generation):
        job.cancel()
        models.update_all_models()
        return (
            job.message,
            gr.update(active=False),
            gr.update(interactive=False, visible=False),
            generation + 1,
        )

    cancel.click(
        cancel_download,
        inputs=[session, revision],
        outputs=[progress, timer, cancel, revision],
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    return SimpleNamespace(
        selection=selection,
        records=records,
        activate=activate,
        status=status,
        component_role=component_role,
        bind_component_context=bind_component_context,
        use_component=use_component,
        preview_cache=previews,
        refresh=refresh,
        refresh_rows=refresh_local,
        refresh_inputs=refresh_inputs,
        refresh_outputs=[records, cards, selection, page, count, quantization],
        refresh_js=version_js("acquisition-view"),
        require_source=require_source,
        catalogue=catalogue,
        checkpoint_name=checkpoint_name,
        inspected_local=inspected_local,
        checkpoint_choices=checkpoint_choices,
    )
