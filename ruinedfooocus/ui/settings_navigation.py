"""Field-level Settings navigation without rebuilding or saving the form."""

import gradio as gr

from ruinedfooocus.core.paths import asset_path
from ruinedfooocus.core.translation_manager import (
    catalogue,
    localize,
    localized_output,
    resolve_locale,
)


def source_text(value):
    """Resolve native static-label metadata before passing it to custom HTML."""
    from gradio.i18n import I18nData

    if isinstance(value, I18nData):
        key = value.key.removeprefix("ruinedfooocus.")
        return catalogue().translations["en"].get(key, key)
    return str(value or "")


def navigation_records(root, sections, *, excluded=()):
    records = []

    def walk(node, route=(), section=""):
        if node in excluded:
            return
        label = source_text(getattr(node, "label", None))
        if isinstance(node, (gr.Tab, gr.Accordion)):
            if not node.elem_id:
                node.elem_id = f"rf-settings-section-{node._id}"
            route = (*route, ("tab" if isinstance(node, gr.Tab) else "accordion", node.elem_id))
            if isinstance(node, gr.Tab) and node.id is None:
                node.id = node.elem_id
            if isinstance(node, gr.Tab) and not section:
                section = label
        if isinstance(node, gr.Dataframe) and "rf-technical-table" not in (node.elem_classes or []):
            node.elem_classes = [*(node.elem_classes or []), "rf-technical-table"]
        if isinstance(node, gr.components.Component) and not isinstance(node, (gr.State, gr.HTML)):
            if isinstance(node, gr.Button):
                label = source_text(node.value)
            if label and node.visible is not False and route:
                if not node.elem_id:
                    node.elem_id = f"rf-settings-field-{node._id}"
                choices = getattr(node, "choices", ()) or ()
                records.append(
                    {
                        "id": node.elem_id,
                        "label": label,
                        "section": section,
                        "route": route,
                        "terms": " ".join(
                            [source_text(getattr(node, "info", None)), sections.get(section, "")]
                            + [
                                source_text(value[0] if isinstance(value, tuple) else value)
                                for value in choices
                            ]
                        ),
                    }
                )
        for child in getattr(node, "children", ()):
            walk(child, route, section)

    walk(root)
    return records


def bind_navigation(root, search, results, sections, *, excluded=()):
    records = navigation_records(root, sections, excluded=excluded)
    bindings = {}
    layouts = []

    def collect(node, tabs=None):
        if node in excluded:
            return
        if isinstance(node, gr.Tabs):
            tabs = node
        if isinstance(node, gr.Tab) and tabs is not None:
            bindings[node.elem_id] = (tabs, {"selected": node.id})
            if tabs not in layouts:
                layouts.append(tabs)
        elif isinstance(node, gr.Accordion):
            bindings[node.elem_id] = (node, {"open": True})
            layouts.append(node)
        for child in getattr(node, "children", ()):
            collect(child, tabs)

    collect(root)
    indexed = {record["id"]: record for record in records}

    def translated(locale):
        return {
            "locale": locale,
            "focus_target": {},
            "records": [
                dict(
                    record,
                    label=localize(record["label"], locale),
                    section=localize(record["section"], locale),
                )
                for record in records
            ],
            "no_matches": localize("No matching setting. Try a shorter term.", locale),
            "unavailable": localize(
                "This setting is unavailable in the current configuration.", locale
            ),
            "found": localize(
                "Choose a result to open and focus the setting. Your unsaved changes are kept.",
                locale,
            ),
        }

    results.props.update(translated("en"))
    localized_output(lambda locale: gr.update(**translated(locale)), [], [results])
    search.input(
        fn=lambda query: query or "",
        inputs=search,
        outputs=results,
        queue=False,
        trigger_mode="always_last",
        show_progress="hidden",
        api_visibility="undocumented",
    )

    def navigate_setting(event: gr.EventData, request: gr.Request = None):
        payload = event._data if isinstance(event._data, dict) else {}
        identity = payload.get("id")
        record = indexed.get(identity) if isinstance(identity, str) else None
        token = payload.get("token")
        locale = resolve_locale(payload.get("locale"), request)
        if record is None or type(token) is not int or not 0 < token < 2**31:
            raise gr.Error(
                localize("This setting is unavailable in the current configuration.", locale)
            )
        updates = {}
        for _, identity in record["route"]:
            target, properties = bindings[identity]
            updates[target] = gr.update(**properties)
        updates[results] = gr.update(
            **(translated(locale) | {"focus_target": {"id": record["id"], "token": token}})
        )
        return updates

    results.navigate(
        navigate_setting,
        inputs=[],
        outputs=[*layouts, results],
        queue=False,
        trigger_mode="always_last",
        show_progress="hidden",
        api_visibility="undocumented",
    )


def create_search_results():
    return gr.HTML(
        "",
        elem_id="settings-search-results",
        html_template='<p role="status"></p><ul></ul>',
        js_on_load=asset_path("settings_search.js").read_text(encoding="utf-8"),
    )


def profile_receipt(profile, values):
    from ruinedfooocus.app.arguments import args
    from ruinedfooocus.core.profile_store import selected_profile

    running = getattr(args, "_runtime_profile", args.settings)
    return {
        "running": running or "default",
        "profile": profile or "default",
        "next": selected_profile() or "default",
        "saved": list(values),
    }


def profile_labels(locale):
    return dict(
        running_label=localize("Running profile", locale),
        saved_label=localize("Save target", locale),
        next_label=localize("Next launch", locale),
        dirty_label=localize("Unsaved changes", locale),
        clean_label=localize("All form changes saved", locale),
        timing=localize(
            "Language, text size and theme apply at once; the rest on Save. Startup, folders and image runtime need a restart.",
            locale,
        ),
    )


def profile_update(receipt, request=None, *, locale=None):
    # Gradio replaces the custom props dictionary. Send the complete summary
    # while keeping its value (the latest browser draft) out of server updates.
    return gr.update(**receipt, **profile_labels(locale or resolve_locale(request=request)))


def create_profile_state(profile, values):

    receipt = profile_receipt(
        profile,
        [control.value for control in values],
    )
    receipt_state = gr.State(receipt)
    component = gr.HTML(
        receipt["saved"],
        elem_id="settings-profile-state",
        html_template='<dl></dl><p role="status"></p><p class="apply-timing"></p>',
        js_on_load=asset_path("settings_profile.js").read_text(encoding="utf-8"),
        **receipt,
        **profile_labels("en"),
    )
    localized_output(
        lambda locale, receipt: profile_update(receipt, locale=locale),
        [receipt_state],
        [component],
    )
    gr.on(
        triggers=[control.change for control in values],
        fn=None,
        inputs=values,
        outputs=component,
        js="(...values) => [{__type__: 'update', value: values}]",
        queue=False,
        trigger_mode="always_last",
        show_progress="hidden",
        api_visibility="undocumented",
    )
    return component, receipt_state
