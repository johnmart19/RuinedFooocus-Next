"""One card browser for LoRAs and prompt embeddings; native LoRA transport stays private."""

from pathlib import Path
import hashlib
from types import SimpleNamespace

import math


import gradio as gr


from ruinedfooocus.core.translation_manager import translate_ui as t
from ruinedfooocus.core.translation_manager import localize, localized_output, resolve_locale

from ruinedfooocus.features.models.embedding_browser import (
    embedding_trigger_map,
    embedding_trigger,
    embedding_spans,
    embedding_target,
    insert_embedding,
    scan_embeddings,
    selected_embedding,
)

from ruinedfooocus.features.models.resource_cards import (
    resource_card,
    resource_records,
)
from ruinedfooocus.features.models.model_identity import file_signature


def lora_values(gallery):
    return {item[1].split(" - ", 1)[1]: float(item[1].split(" - ", 1)[0]) for item in gallery or []}


def _remove_owned_prefix(text, receipt, preceding):
    """Undo our unchanged insertion, allowing newer owned prefixes before it."""
    if not receipt:
        return None
    offset = 0
    preceding = list(preceding)
    while not text.startswith(receipt, offset):
        prefix = next((p for p in preceding if text.startswith(p, offset)), None)
        if prefix is None:
            return None
        offset += len(prefix)
        preceding.remove(prefix)
    return text[:offset] + text[offset + len(receipt) :]


def active_cards(records, gallery, positive, negative, cache_dir=None):
    """Derive selection from current controls, including Library/default replay."""
    triggers = embedding_trigger_map(
        [card["embedding"] for card in records if card["kind"] == "Embedding"], cache_dir
    )
    weights = lora_values(gallery)
    result = []
    for card in records:
        if card["kind"] == "LoRA":
            state = f"Active | {weights[card['name']]:g}" if card["name"] in weights else ""
        else:
            entry = card["embedding"]
            state = " + ".join(
                target
                for target, prompt in (("Positive", positive), ("Negative", negative))
                if embedding_spans(prompt or "", entry, triggers)
            )
        if state:
            result.append((card, state))
    return result


def create_ui(
    prompt,
    negative_prompt,
    *,
    models,
    directories,
    cache_dir,
    defaults,
    thumbnail,
    keywords,
    weight_min=0,
    weight_max=2,
):
    """Build UI without loading weights, generation or network access."""

    def discover():
        embeddings, status = scan_embeddings(
            directories() if callable(directories) else directories
        )
        cards = resource_records(
            [
                (name, path)
                for name in models.get_names("loras")
                if (path := models.get_file("loras", name))
            ],
            "LoRA",
            cache_dir=cache_dir / "loras",
            fallback_cache_dir=cache_dir,
        )
        for card in cards:
            try:
                card["signature"] = tuple(file_signature(card["path"]))
            except OSError:
                card["signature"] = None
        for entry in embeddings:
            card = resource_card(
                entry.name,
                entry.signature[0],
                "Embedding",
                cache_dir=cache_dir / "embeddings",
                fallback_cache_dir=cache_dir,
            )
            card["embedding"] = entry
            cards.append(card)
        return cards, status

    records, status = discover()

    def caption(card):
        return f"[{card['kind']}] {card['caption']}"

    def tiles(cards):
        return [(card["preview"], caption(card)) for card in cards]

    initial_loras = [(thumbnail(name), f"{weight} - {name}") for name, weight in defaults]
    initial_active = active_cards(
        records,
        initial_loras,
        prompt.value or "",
        negative_prompt.value or "",
        cache_dir / "embeddings",
    )
    transport = gr.Gallery(value=initial_loras, visible=False, elem_id="lora-native-transport")
    catalog = gr.State(records)
    selected = gr.State(None)
    destinations = gr.State({})
    insertions = gr.State({})
    # Search and type filter narrow the same list, so they share one row.
    with gr.Row(elem_id="adapter-filters", equal_height=True):
        search = gr.Textbox(
            label=t("Search LoRAs & embeddings"),
            placeholder=t("Search name or metadata"),
            lines=1,
            max_lines=1,
            scale=3,
            min_width=220,
        )
        kind = gr.Radio(
            choices=[(t("All"), "All"), (t("LoRA"), "LoRA"), (t("Embedding"), "Embedding")],
            value="All",
            label=t("Filter"),
            scale=2,
            min_width=220,
        )
    available = gr.Gallery(
        value=tiles(records),
        label=t("Available LoRAs & embeddings"),
        elem_id="adapter-gallery",
        columns=2,
        rows=3,
        height="auto",
        object_fit="contain",
        allow_preview=False,
        preview=False,
        interactive=False,
        buttons=[],
    )
    active = gr.Gallery(
        value=[(card["preview"], f"{state} | {caption(card)}") for card, state in initial_active],
        label=t("Selected LoRAs & embeddings"),
        elem_id="adapter-active-gallery",
        columns=2,
        rows=2,
        height="auto",
        object_fit="contain",
        allow_preview=False,
        preview=False,
        interactive=False,
        buttons=[],
    )
    # Hidden galleries keep the select/replay contract; the HTML cards are the visible list.
    available.visible = False
    active.visible = False
    assets = Path(__file__).resolve().parents[2] / "ui" / "assets"

    def card_payload(
        records,
        gallery,
        positive,
        negative,
        query="",
        filter_kind="All",
        selected_only=False,
        locale=None,
        selected_key=None,
    ):
        def shown_state(state):
            # "Active | 0.8" or "Positive + Negative" -> the browser's language.
            if state.startswith("Active"):
                return localize("Active", locale) + state[len("Active") :]
            return " + ".join(localize(part, locale) for part in state.split(" + ") if part)

        states = {
            card["caption"]: state
            for card, state in active_cards(
                records, gallery, positive, negative, cache_dir / "embeddings"
            )
        }
        cards = []
        for card in records:
            if selected_only and card["caption"] not in states:
                continue
            if not selected_only and (filter_kind != "All" and filter_kind != card["kind"]):
                continue
            if (
                not selected_only
                and (query or "").casefold()
                not in (card["label"] + " " + card["name"] + " " + str(card["metadata"])).casefold()
            ):
                continue
            preview = card["preview"]
            if not isinstance(preview, str):
                preview = gr.processing_utils.save_pil_to_cache(
                    preview, cache_dir=str(cache_dir / "card-previews")
                )
            cards.append(
                dict(
                    key=card["caption"],
                    label=card["label"],
                    kind=localize(card["kind"], locale),
                    state=shown_state(states.get(card["caption"], "")),
                    preview=gr.data_classes.FileData(path=preview).model_dump(),
                    fallback=gr.data_classes.FileData(path=card["fallback"]).model_dump(),
                    preview_is_default=card["preview_is_default"],
                )
            )
        return dict(
            heading=localize("Selected LoRAs & embeddings", locale)
            if selected_only
            else localize("Available LoRAs & embeddings", locale),
            acceptsDrop=selected_only,
            acceptsDrag=not selected_only,
            defaultPreviewLabel=localize("Default preview", locale),
            selected=next(
                (
                    card["caption"]
                    for card in records
                    if (card["kind"], card["name"]) == tuple(selected_key or ())
                ),
                "",
            ),
            cards=cards,
            empty=localize("Select a card and choose Activate, or drag it here.", locale)
            if selected_only
            else localize("No matching resources.", locale),
        )

    def surface(value, identifier):
        return gr.HTML(
            value=value,
            elem_id=identifier,
            html_template=(assets / "adapter_cards.html").read_text(encoding="utf-8"),
            js_on_load=(assets / "adapter_cards.js").read_text(encoding="utf-8"),
            css_template=(assets / "adapter_cards.css").read_text(encoding="utf-8"),
        )

    available_cards = surface(
        card_payload(records, initial_loras, prompt.value or "", negative_prompt.value or ""),
        "adapter-cards",
    )
    selected_cards = surface(
        card_payload(
            records,
            initial_loras,
            prompt.value or "",
            negative_prompt.value or "",
            selected_only=True,
        ),
        "adapter-selected-cards",
    )

    detail = gr.Textbox(label=t("Selected resource"), interactive=False, lines=3)
    weight = gr.Slider(
        label=t("LoRA weight"),
        minimum=weight_min,
        maximum=weight_max,
        step=0.05,
        value=1,
        visible=False,
    )
    target = gr.Dropdown(
        [(t("Automatic"), "Auto"), (t("Positive"), "Positive"), (t("Negative"), "Negative")],
        value="Auto",
        label=t("Embedding target"),
        visible=False,
    )
    with gr.Row(elem_classes=["rf-actions"]):
        activate = gr.Button(t("Activate"), interactive=False)
        remove = gr.Button(t("Remove"), interactive=False)
        refresh = gr.Button(t("Refresh LoRAs & embeddings"))
    status_control = gr.Textbox(
        label=t("Resource discovery"), value=status, interactive=False, visible=False
    )
    trigger_words = gr.Textbox(
        value="",
        label=t("Active resource triggers"),
        interactive=False,
    )
    # Styles and request transport consume raw LoRA triggers, not the attributed
    # display text or embedding triggers (which already live in the prompts).
    keyword_transport = gr.Textbox(visible=False)

    def triggers_for(card, records):
        if card["kind"] == "LoRA":
            words = card.get("metadata", {}).get("trainedWords")
            if isinstance(words, list):
                text = ", ".join(
                    word.strip() for word in words if isinstance(word, str) and word.strip()
                )
                if text:
                    return text
            return keywords(card["name"]).strip()
        mapping = embedding_trigger_map(
            [row["embedding"] for row in records if row["kind"] == "Embedding"],
            cache_dir / "embeddings",
        )
        return embedding_trigger(card["embedding"], mapping)

    def render(records, query, kind, gallery, positive, negative):
        query = (query or "").casefold()
        filtered = [
            card
            for card in records
            if (kind == "All" or kind == card["kind"])
            and query
            in (card["name"] + " " + card["label"] + " " + str(card["metadata"])).casefold()
        ]
        selected_cards = active_cards(
            records, gallery, positive, negative, cache_dir / "embeddings"
        )
        return gr.update(value=tiles(filtered), selected_index=None), gr.update(
            value=[
                (card["preview"], f"{state} | {caption(card)}") for card, state in selected_cards
            ],
            selected_index=None,
        )

    render_inputs = [catalog, search, kind, transport, prompt, negative_prompt]
    for component in (search, kind, transport, prompt, negative_prompt, catalog):
        component.change(
            render,
            inputs=render_inputs,
            outputs=[available, active],
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )

    def checked(records, key, *, verify_file=True):
        card = next(
            (card for card in records if (card["kind"], card["name"]) == tuple(key or ())), None
        )
        if card is None:
            raise gr.Error("Select an installed resource card first.")
        if not verify_file:
            return card
        if card["kind"] == "LoRA" and "path" in card:
            try:
                unchanged = tuple(file_signature(card["path"])) == card.get("signature")
            except OSError:
                unchanged = False
            if not unchanged:
                raise gr.Error("The LoRA file changed or was removed. Refresh and select again.")
        if card["kind"] == "Embedding":
            try:
                selected_embedding([card["embedding"]], card["name"])
            except ValueError as error:
                raise gr.Error(str(error)) from error
        return card

    def select_card(records, gallery, evt: gr.SelectData, choices=None):
        event_caption = evt.value.get("caption", "") if isinstance(evt.value, dict) else ""
        # Match identity-bearing caption rather than a stale positional index.
        card = next(
            (
                card
                for card in records
                if event_caption == caption(card) or event_caption.endswith(" | " + caption(card))
            ),
            None,
        )
        if card is None or not evt.selected:
            raise gr.Error("The selected resource changed. Refresh and select its card again.")
        is_lora = card["kind"] == "LoRA"
        value = lora_values(gallery).get(card["name"], 1)
        return (
            (card["kind"], card["name"]),
            f"{card['kind']}: {card['label']}\n{card['name']}\nTriggers: {triggers_for(card, records) or 'No known triggers'}",
            gr.update(visible=is_lora, value=value),
            gr.update(
                visible=not is_lora,
                value=(choices or {}).get(card["name"], "Auto") if not is_lora else "Auto",
            ),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    selection_outputs = [
        selected,
        detail,
        weight,
        target,
        activate,
        remove,
    ]
    for gallery in (available, active):
        gallery.select(
            select_card,
            inputs=[catalog, transport, destinations],
            outputs=selection_outputs,
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )

    def change_lora(records, key, gallery, value, remove=False):
        card = checked(records, key, verify_file=not remove)
        if card["kind"] != "LoRA":
            raise gr.Error("LoRA weights apply only to LoRAs.")
        if not remove and (not math.isfinite(value) or not weight_min <= value <= weight_max):
            raise gr.Error("Choose a LoRA weight within the configured range.")
        values = lora_values(gallery)
        if remove:
            values.pop(card["name"], None)
        else:
            values[card["name"]] = value
        return gr.update(
            value=[(thumbnail(name), f"{weight} - {name}") for name, weight in values.items()]
        )

    def update_weight(records, key, gallery, value):
        card = checked(records, key)
        if card["kind"] != "LoRA":
            raise gr.Error("LoRA weights apply only to LoRAs.")
        if card["name"] not in lora_values(gallery):
            return gr.skip()
        return change_lora(records, key, gallery, value)

    def set_lora(records, key, gallery, value):
        return change_lora(records, key, gallery, value)

    def delete_lora(records, key, gallery, value):
        return change_lora(records, key, gallery, value, remove=True)

    weight.release(
        update_weight,
        inputs=[catalog, selected, transport, weight],
        outputs=transport,
        queue=False,
        api_visibility="undocumented",
    )

    def sync_weight(gallery, key):
        if not key or key[0] != "LoRA":
            return gr.update(visible=False)
        return gr.update(value=lora_values(gallery).get(key[1], 1), visible=True)

    transport.change(
        sync_weight,
        inputs=[transport, selected],
        outputs=weight,
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def refresh_files():
        records, status = discover()
        return records, status, None

    refresh.click(
        refresh_files, outputs=[catalog, status_control, selected], api_visibility="undocumented"
    )

    def active_triggers(records, gallery, positive, negative):
        return "\n".join(
            f"{card['label']} ({state}): {triggers_for(card, records) or 'No known triggers'}"
            for card, state in active_cards(
                records, gallery, positive, negative, cache_dir / "embeddings"
            )
        )

    trigger_words.value = active_triggers(
        records, initial_loras, prompt.value or "", negative_prompt.value or ""
    )

    def lora_keywords(records, gallery):
        active = lora_values(gallery)
        return ", ".join(
            text
            for card in records
            if card["kind"] == "LoRA" and card["name"] in active
            if (text := triggers_for(card, records))
        )

    keyword_transport.value = lora_keywords(records, initial_loras)
    gr.on(
        triggers=[transport.change, catalog.change],
        fn=lora_keywords,
        inputs=[catalog, transport],
        outputs=keyword_transport,
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )

    gr.on(
        triggers=[transport.change, prompt.change, negative_prompt.change, catalog.change],
        fn=active_triggers,
        inputs=[catalog, transport, prompt, negative_prompt],
        outputs=trigger_words,
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def render_cards(
        locale, records, query, filter_kind, gallery, positive, negative, selected_key
    ):
        return (
            card_payload(
                records,
                gallery,
                positive,
                negative,
                query,
                filter_kind,
                locale=locale,
                selected_key=selected_key,
            ),
            card_payload(
                records,
                gallery,
                positive,
                negative,
                selected_only=True,
                locale=locale,
                selected_key=selected_key,
            ),
        )

    def html_render(
        records,
        query,
        filter_kind,
        gallery,
        positive,
        negative,
        selected_key=None,
        request: gr.Request = None,
    ):
        return render_cards(
            resolve_locale(request=request),
            records,
            query,
            filter_kind,
            gallery,
            positive,
            negative,
            selected_key,
        )

    html_inputs = [*render_inputs, selected]
    localized_output(render_cards, html_inputs, [available_cards, selected_cards])

    for component in (search, kind, transport, prompt, negative_prompt, catalog, selected):
        component.change(
            html_render,
            inputs=html_inputs,
            outputs=[available_cards, selected_cards],
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )

    def remember_target(key, value, choices):
        choices = dict(choices or {})
        if key and key[0] == "Embedding":
            choices[key[1]] = value
        return choices

    target.input(
        remember_target,
        inputs=[selected, target, destinations],
        outputs=destinations,
        queue=False,
        api_visibility="undocumented",
    )

    def restore_target(key, choices):
        return gr.update(
            value=(choices or {}).get(key[1], "Auto") if key else "Auto",
            visible=bool(key and key[0] == "Embedding"),
        )

    selected.change(
        restore_target,
        inputs=[selected, destinations],
        outputs=target,
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def clear_selection(key):
        if key:
            return (gr.skip(),) * 4
        return (
            "",
            gr.update(visible=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    selected.change(
        clear_selection,
        inputs=selected,
        outputs=[detail, weight, activate, remove],
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def change_resource(records, key, gallery, positive, negative, choices, owned, action, value=1):
        card = checked(records, key, verify_file=action != "remove")
        owned = dict(owned or {})
        if card["kind"] == "LoRA":
            return (
                (delete_lora if action == "remove" else set_lora)(records, key, gallery, value),
                gr.skip(),
                gr.skip(),
                owned,
            )
        name = card["name"]
        if action == "remove":
            changed = []
            for destination, current in (("Positive", positive), ("Negative", negative)):
                receipt = owned.pop((name, destination), None)
                # Only undo an insertion from this session. Replayed or manually
                # written tokens remain under the author's control.
                current = current or ""
                updated = _remove_owned_prefix(
                    current,
                    receipt,
                    [prefix for (other, place), prefix in owned.items() if place == destination],
                )
                changed.append(updated if updated is not None else gr.skip())
            return gr.skip(), *changed, owned
        mapping = embedding_trigger_map(
            [row["embedding"] for row in records if row["kind"] == "Embedding"],
            cache_dir / "embeddings",
        )
        choice = (choices or {}).get(name, "Auto")
        destination = (
            choice
            if choice in ("Positive", "Negative")
            else embedding_target(card["embedding"], card.get("metadata"), default="Positive")
        )
        prompts = {"Positive": positive or "", "Negative": negative or ""}
        changes = {"Positive": gr.skip(), "Negative": gr.skip()}
        previous = "Negative" if destination == "Positive" else "Positive"
        if embedding_spans(prompts[previous], card["embedding"], mapping):
            updated = _remove_owned_prefix(
                prompts[previous],
                owned.get((name, previous)),
                [
                    prefix
                    for (other, place), prefix in owned.items()
                    if other != name and place == previous
                ],
            )
            # Never relocate authored/replayed references or a prefix the user
            # has edited. Moving an unchanged owned insertion is atomic.
            if updated is None or embedding_spans(updated, card["embedding"], mapping):
                return gr.skip(), gr.skip(), gr.skip(), owned
            changes[previous] = updated
            owned.pop((name, previous))
        current = prompts[destination]
        if not embedding_spans(current, card["embedding"], mapping):
            updated = insert_embedding(current, card["embedding"], mapping)
            owned[(name, destination)] = updated[: len(updated) - len(current)]
            changes[destination] = updated
        return (
            gr.skip(),
            changes["Positive"],
            changes["Negative"],
            owned,
        )

    def activate_selected(records, key, gallery, positive, negative, choices, owned, value):
        return change_resource(
            records, key, gallery, positive, negative, choices, owned, "activate", value
        )

    def remove_selected(records, key, gallery, positive, negative, choices, owned, value):
        return change_resource(
            records, key, gallery, positive, negative, choices, owned, "remove", value
        )

    for button, callback in ((activate, activate_selected), (remove, remove_selected)):
        button.click(
            callback,
            inputs=[
                catalog,
                selected,
                transport,
                prompt,
                negative_prompt,
                destinations,
                insertions,
                weight,
            ],
            outputs=[transport, prompt, negative_prompt, insertions],
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )

    def interact(records, gallery, positive, negative, choices, owned, evt: gr.EventData):
        data = evt._data
        card = next((row for row in records if row["caption"] == data.get("key")), None)
        if card is None:
            raise gr.Error("This resource changed. Refresh and select again.")
        event = gr.SelectData(None, dict(index=0, value={"caption": caption(card)}, selected=True))
        updates = list(select_card(records, gallery, event, choices))
        changed = [gr.skip(), gr.skip(), gr.skip(), owned]
        if data.get("action") == "activate":
            changed = change_resource(
                records,
                (card["kind"], card["name"]),
                gallery,
                positive,
                negative,
                choices,
                owned,
                "activate",
                lora_values(gallery).get(card["name"], 1),
            )
        return (*updates, *changed)

    for component in (available_cards, selected_cards):
        component.click(
            interact,
            inputs=[catalog, transport, prompt, negative_prompt, destinations, insertions],
            outputs=[*selection_outputs, transport, prompt, negative_prompt, insertions],
            queue=False,
            show_progress="hidden",
            # A double-click sends "select" then "activate"; the default "once"
            # mode drops the second event while the first is still pending.
            trigger_mode="multiple",
            api_visibility="undocumented",
        )

    return SimpleNamespace(
        transport=transport,
        search=search,
        available=available,
        active=active,
        catalog=catalog,
        kind=kind,
        keywords=keyword_transport,
        triggers=trigger_words,
        status=status_control,
        selected=selected,
        destinations=destinations,
        insertions=insertions,
        render=render,
        discover=discover,
        refresh_files=refresh_files,
    )
