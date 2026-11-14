from ruinedfooocus.core.translation_manager import (
    localize,
    localized_output,
    resolve_locale,
    translate_ui,
)

"""Checkpoint browsing in Models, sharing Create's quick-picker file identities."""

from pathlib import Path
import gradio as gr

from ruinedfooocus.core.util import get_checkpoint_thumbnail
from ruinedfooocus.features.models.selection import (
    checkpoint_records,
    model_label,
    selected_checkpoint,
)
from ruinedfooocus.features.models.ui_model_components import create_ui as create_components_ui
from ruinedfooocus.features.models.ui_checkpoint_tools import create_ui as create_checkpoint_ui


def create_ui(base_model, models, model_refresh=None, catalogue_button=None):
    from ruinedfooocus.features.models.ui_requests import (
        current_identity as request_identity,
        current_source,
        project,
        stamp,
        validate_view,
        version_js,
    )

    if catalogue_button is not None:
        catalogue_button.render()

    def describe(name, version=None):
        from ruinedfooocus.features.models.ui_requests import project

        def build(ticket):
            info = models.get_models_by_path("checkpoints", name, fetch=False)
            publisher = info.get("model", {})
            title = publisher.get("name", "") if isinstance(publisher, dict) else ""
            return "\n".join(
                filter(
                    None,
                    (
                        model_label(name, info),
                        f"File: {name}",
                        title,
                        f"Family: {info.get('baseModel', 'Unknown')}",
                    ),
                )
            )

        result = project("checkpoint-description", version, None, build, outputs=1, source=name)
        return result[0] if isinstance(result, tuple) else result

    def selected_identity(name, request: gr.Request = None, version=None):
        def build(ticket):
            info = models.get_models_by_path("checkpoints", name, fetch=False)
            return model_label(name, info)

        result = project("checkpoint-identity", version, request, build, outputs=1, source=name)
        return result[0] if isinstance(result, tuple) else result

    # The chosen model and the search that finds another share one row.
    with gr.Row(elem_id="model-browser-head", equal_height=True):
        current_identity = gr.Textbox(
            label=translate_ui("Selected model"),
            value=selected_identity(base_model.value),
            interactive=False,
            lines=1,
            elem_id="selected-checkpoint-identity",
            min_width=240,
        )
        query = gr.Textbox(
            label=translate_ui("Search checkpoints"),
            placeholder=translate_ui("Search model name or filename"),
            value="",
            elem_id="model-search",
            min_width=240,
        )
    query_version = gr.Number(1, visible=False, precision=0)
    cards_version = gr.Number(0, visible=False, precision=0)
    identity_version = gr.Number(0, visible=False, precision=0)
    description_version = gr.Number(0, visible=False, precision=0)

    def refresh(search, request: gr.Request = None, version=None):
        def build(ticket):
            def check_cancelled():
                if ticket is not None and ticket.cancelled.is_set():
                    from concurrent.futures import CancelledError

                    raise CancelledError()

            def metadata(name):
                check_cancelled()
                return models.get_models_by_path("checkpoints", name, fetch=False)

            records = checkpoint_records(models.get_names("checkpoints"), metadata, search)
            fallback = fallback_payload() if records else None
            version = ticket.version if ticket is not None else 1
            presented = []
            for row in records:
                check_cancelled()
                presented.append(
                    row | {"_view_version": version, "_card": card_payload(row, fallback)}
                )
            pictures = [(row["_card"]["preview"]["path"], row["caption"]) for row in presented]
            return gr.update(value=pictures), presented or [
                {"_request_only": True, "_view_version": version, "name": "", "caption": ""}
            ]

        return project("checkpoint-query", version, request, build, outputs=2)

    assets = Path(__file__).resolve().parents[2] / "ui/assets"

    def preview_payload(name):
        from ruinedfooocus.features.models.catalogue_previews import local_checkpoint_preview
        from ruinedfooocus.features.models.model_previews import is_warning
        from ruinedfooocus.features.models.resource_cards import default_preview

        preview = local_checkpoint_preview(name, models) or get_checkpoint_thumbnail(name)
        if not preview or is_warning(preview):
            preview = default_preview("Image model", models.cache_paths["checkpoints"].parent)
        return gr.data_classes.FileData(path=preview).model_dump()

    def fallback_payload():
        from ruinedfooocus.features.models.resource_cards import default_preview

        return gr.data_classes.FileData(
            path=default_preview("Image model", models.cache_paths["checkpoints"].parent)
        ).model_dump()

    def card_payload(row, fallback):
        preview = preview_payload(row["name"])
        return dict(
            key=row["caption"],
            label=row["label"],
            kind="",
            state="",
            preview=preview,
            fallback=fallback,
            preview_is_default=preview["path"] == fallback["path"],
        )

    def visible_cards(rows, current_model, locale=None):
        rows = [row for row in rows if not row.get("_request_only")]
        # Presentation belongs to these immutable, versioned query rows. Model
        # selection and locale changes reuse it; metadata refresh rebuilds it.
        cards, fallback = [], None
        for row in rows:
            card = row.get("_card")
            if card is None:  # Compatibility with callers supplying plain rows.
                fallback = fallback or fallback_payload()
                card = card_payload(row, fallback)
            cards.append(card)
        return dict(
            heading=localize("Image models", locale),
            acceptsDrop=False,
            empty=localize("No matching image models.", locale),
            selected=next((row["caption"] for row in rows if row["name"] == current_model), ""),
            defaultPreviewLabel=localize("Default preview", locale),
            cards=cards,
        )

    initial, records = refresh("")
    selection = gr.State(records)
    gallery = gr.Gallery(
        label=translate_ui("Image checkpoints"),
        show_label=False,
        height="auto",
        allow_preview=False,
        preview=False,
        columns=2,
        object_fit="contain",
        buttons=[],
        min_width=60,
        value=initial["value"],
        elem_id="model-gallery",
    )
    gallery.visible = False
    cards = gr.HTML(
        value=visible_cards(records, base_model.value),
        elem_id="checkpoint-cards",
        html_template=(assets / "adapter_cards.html").read_text(encoding="utf-8"),
        js_on_load=(assets / "adapter_cards.js").read_text(encoding="utf-8"),
        css_template=(assets / "adapter_cards.css").read_text(encoding="utf-8"),
    )

    def visible_cards_for(
        rows, current_model, request: gr.Request = None, version=None, locale=None
    ):
        from concurrent.futures import CancelledError

        def validate():
            validate_view(rows, request, "checkpoint-query", initial_version=1)
            if (
                version is None
                and current_source("checkpoint-cards", request, current_model) != current_model
            ):
                raise CancelledError("The selected checkpoint changed")

        try:
            validate()
        except CancelledError:
            return gr.skip()

        def build(ticket):
            payload = visible_cards(rows, current_model, locale or resolve_locale(request=request))
            if ticket is not None:
                payload = stamp(payload, "checkpoint-query", rows[0]["_view_version"])
                payload["pageIdentity"] = request_identity("checkpoint-cards", request)
            return payload

        # Language observers share the latest query ticket; normal card/model
        # projections use their own browser ordering counter.
        channel = "checkpoint-cards" if version is not None else "checkpoint-query"
        result = project(
            channel,
            version,
            request,
            build,
            outputs=1,
            source=current_model if version is not None else None,
            validate=validate,
        )
        return result[0] if isinstance(result, tuple) else result

    localized_output(
        lambda locale, rows, current: visible_cards_for(rows, current, locale=locale),
        [selection, base_model],
        [cards],
    )
    gr.on(
        triggers=[selection.change, base_model.change],
        fn=visible_cards_for,
        inputs=[selection, base_model, cards_version],
        outputs=cards,
        queue=False,
        trigger_mode="multiple",
        js=version_js("checkpoint-cards"),
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def choose_card(rows, current_model, evt: gr.EventData, request: gr.Request = None):
        from concurrent.futures import CancelledError
        from ruinedfooocus.features.models.ui_requests import validate_view

        try:
            validate_view(rows, request, "checkpoint-query", initial_version=1)
        except CancelledError:
            raise gr.Error("The model list changed. Select the file again.") from None
        record = next((row for row in rows if row["caption"] == evt._data.get("key")), None)
        if record is None or not models.get_file("checkpoints", record["name"]):
            raise gr.Error("The model changed. Refresh and select again.")
        if record["name"] == current_model:
            return gr.skip()
        return gr.update(value=record["name"])

    cards.click(
        choose_card,
        inputs=[selection, base_model],
        outputs=base_model,
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )

    gr.on(
        triggers=[query.change, query.submit]
        + ([model_refresh.change] if model_refresh is not None else []),
        fn=refresh,
        inputs=[query, query_version],
        outputs=[gallery, selection],
        api_visibility="undocumented",
        show_progress="hidden",
        queue=False,
        trigger_mode="multiple",
        js=version_js("checkpoint-query"),
    ).then(
        # Gradio 6.31's unqueued transport drops State.change notifications.
        # Read the completed query's rows and latest model explicitly, with a
        # fresh card counter so a model change during the query can recover.
        fn=visible_cards_for,
        inputs=[selection, base_model, cards_version],
        outputs=cards,
        queue=False,
        trigger_mode="multiple",
        js=version_js("checkpoint-cards"),
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def select(records, current_model, evt: gr.SelectData, request: gr.Request = None):
        from concurrent.futures import CancelledError

        try:
            validate_view(records, request, "checkpoint-query", initial_version=1)
        except CancelledError:
            raise gr.Error("The model list changed. Select the file again.") from None
        try:
            name = selected_checkpoint(records, evt.index, evt.value.get("caption"))
        except (ValueError, AttributeError) as error:
            raise gr.Error(str(error)) from None
        if not models.get_file("checkpoints", name):
            raise gr.Error(
                "This checkpoint is no longer installed. Refresh files and select again."
            )
        return gr.skip() if name == current_model else gr.update(value=name)

    gallery.select(
        select, inputs=[selection, base_model], outputs=base_model, api_visibility="undocumented"
    )
    with gr.Accordion(translate_ui("Selected model details"), open=False):
        current = gr.Textbox(
            label=translate_ui("Selected model"),
            show_label=False,
            value=describe(base_model.value),
            lines=3,
            interactive=False,
        )
    gr.on(
        triggers=[base_model.change]
        + ([model_refresh.change] if model_refresh is not None else []),
        fn=selected_identity,
        inputs=[base_model, identity_version],
        outputs=current_identity,
        api_visibility="undocumented",
        show_progress="hidden",
        queue=False,
        trigger_mode="multiple",
        js=version_js("checkpoint-identity"),
    )
    gr.on(
        triggers=[base_model.change]
        + ([model_refresh.change] if model_refresh is not None else []),
        fn=describe,
        inputs=[base_model, description_version],
        outputs=current,
        api_visibility="undocumented",
        show_progress="hidden",
        queue=False,
        trigger_mode="multiple",
        js=version_js("checkpoint-description"),
    )
    with gr.Accordion(translate_ui("Image model components"), open=False):
        create_components_ui(base_model, model_refresh=model_refresh)
    with gr.Accordion(translate_ui("Checkpoint tools"), open=False):
        create_checkpoint_ui(base_model, model_refresh=model_refresh)
    return query, gallery, selection, refresh
