# Compact model and profile controls

Models Hub uses a content-sized Refresh action. Its pager is part of the current
model-card payload, with native HTML buttons and a short page status. A single
page hides the pager; unavailable Previous/Next directions remain disabled.
Multiple pages retain keyboard activation, visible focus and touch targets of
at least 44 pixels. Matching counts exclude a retained selection outside the filter.

The page is server-only Gradio State. One absolute `paginate` event carries the
target and view identity from the last successfully rendered payload and uses the existing
`acquisition-page` request counter. It returns cards and State in one guarded
projection. There is no browser Number/change hop that could create another
counter from an old delivered page value. Current server rows are validated;
the `acquisition-selection` channel is unchanged. The existing six refresh
outputs and preview-only artwork patches are retained.

Both view and page identities gate card/pager painting and actions. Locale
observers capture the page identity before background preparation and reject
their response when it changes, so an older observer cannot acquire a newer
page stamp. Rapid page actions, a newer query, retained selections and delayed
view/page/locale responses have separate guarded CPU/DOM cases.

Settings actions align at their content height and wrap within their container.
Name, Save, Save as profile and Use next launch keep their controls, labels,
help, values and existing save behavior. An empty Settings search-result
container collapses; populated results and status messages remain available.
These rules do not change the theme's font size or the Generate prompt layout.

Feature `compact-model-profile-controls` declares actual Gradio construction and
event/State tests, source-backed card DOM tests, request-race cases, and the
existing Settings navigation/draft tests. The owning worker has no exposed
browser surface. Root's final render of the exact source at 390×844 must separately check
pager/profile/search geometry, focus/touch use, horizontal overflow and actual
post-response ordering. CPU/DOM tests do not establish those rendered results.
