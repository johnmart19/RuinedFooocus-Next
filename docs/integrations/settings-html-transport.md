# Settings HTML transport

[Compact model and profile controls](compact-model-profile-controls.md) records
pager ownership and content sizing for the Settings actions/search containers.

Settings uses Gradio HTML component values and property watchers for its search
and profile summary. The custom renderer's `element` is inside the component;
callbacks on other controls must not attach or call functions on an outer DOM
element to reach it.

`ui/settings_navigation.py` registers a small native input callback with the
search HTML as its output (`queue=False`, `trigger_mode="always_last"`). It only
returns the query; translated records and messages stay cached as HTML
properties. No model worker or form-field output is involved.
`settings_search.js` watches both, so a locale update preserves the current query.
Delegated result clicks send a registered field key through the native HTML
`navigate` event. A bounded callback updates only the recorded native
`Tabs.selected` and `Accordion.open` properties, then acknowledges the focus
target. Tabs without an explicit ID receive a stable Settings ID. Repeated
selection always opens the same chain; no header clicks or toggles are
synthesized, and no form control is an output. The renderer only waits for the
acknowledged field to mount and focuses it. A client token discards superseded
focus acknowledgements. Results are rendered as text nodes.

The profile HTML value contains the current form array. Its browser-only change
event receives the setting controls and updates that value through the same
native output path, returning an explicit one-element output list. The initial
baseline retains native control values, including a nullable Custom GGUF path;
displaying a blank textbox does not turn its stored `null` into an empty string.
Server Save
and Save as callbacks return complete property dictionaries containing the
submitted baseline, profile identities and browser-language labels. They never
return a new HTML value. A receipt from an earlier Save therefore leaves later
edits dirty. A separate Gradio State keeps each session's receipt for locale and
next-launch updates. These updates also send all owned properties because the
native renderer replaces its custom-property dictionary. Each browser keeps its
own current draft, receipt and server-held profile base; no draft is stored in
browser storage or reused by another session.

The watchers observe `Object.keys(props)`. At this pin, the HTML component also
recognizes quoted names in its script as custom events; spelling property names
as quoted watcher arguments can cause its update serializer to emit event
callables alongside the intended custom properties. The regression requires
actual native update packets to be JSON-safe before applying them as complete
property replacements.

The installed Gradio 6.31.0 HTML renderer and its custom-HTML guide provide
`props`, `watch` and stable-element event delegation. The integration follows
the [native HTML component guide](https://www.gradio.app/guides/custom-HTML-components)
(Context7 `/gradio-app/gradio`, accessed 2026-11-11). It uses the installed
renderer, without a package change or global DOM property bridge.

Focused CPU checks:

```sh
python tools/check_cpu.py test_settings_navigation.py test_settings_ui.py test_profile_transactions.py test_html_templates.py
```

The regression processes the actual registered search and
locale/Save/Save as/next-launch callbacks through real Gradio session states. It
also runs the exact browser draft callback. It checks the emitted output-list and
JSON/property-replacement boundary, native first/repeated navigation properties,
registered target validation, later edits, locale receipt preservation and
separate browser drafts.
Its minimal DOM fixture is not browser qualification. Real renderer acceptance
must additionally type a query with actual keyboard events, activate a result,
edit and save a private profile, and check a second browser session without model
execution. Programmatic textbox filling alone does not establish that an input
event was exercised.
