# Themes

Open **Settings → Appearance & language → Theme and updates**. Choose Default,
Soft or Monochrome to restyle the current browser immediately. **Save** stores the
choice in the current profile for later launches; **Save as profile** gives another
profile its own choice. Selecting a theme checks availability and reports problems.
An unsaved choice is a browser preview and is lost on reload.
Choosing Default restores the Default palette in that browser. The restricted
guest interface uses the saved profile theme without exposing owner Settings.

The built-in choices use system fonts, compact shared spacing and rounded
corners, and text sizes that follow the browser preference. Default and Soft use
darker shades from their native palettes for readable buttons, selected
controls and help text. Browser text size and browser zoom remain separate preferences.

## Add a custom theme

Expand **Add custom theme**, enter a public Hugging Face Gradio theme ID such as
`author/theme`, and click **Download theme**. Gradio semantic version expressions
are supported, for example `author/theme@>=1.0.0,<2.0.0`. The download adds the
theme to the selector and applies it in the current browser. Save to keep it.

This action downloads JSON theme tokens. It does not execute a Space's app,
uploaded Python or custom stylesheet code. Custom fonts and font stylesheets
are discarded, remote assets in tokens are rejected, and the app retains its
shared text sizing and spacing. A custom theme can still choose unsuitable
colors; inspect its light and dark appearance before relying on it.

Custom downloads are unavailable in offline mode. Previously cached themes and
the built-in choices work without theme downloads or font requests. Startup and
selector changes never fetch a theme. Older saved custom IDs remain visible;
download the same ID explicitly to make it available locally. If a saved theme
is missing or invalid, the app uses Default and shows an availability message in
Settings, also printed during startup. It retains the saved ID so other profile
changes can be saved without losing that preference. Select Default and Save to
clear it deliberately. New custom selections must be cached before saving.

An unsuccessful download retains the previous cache. The next selection or launch
uses the most recent successful download for that ID, including for an ID without
an explicit version expression.

## Native integration and validation

`ruinedfooocus/ui/themes.py` owns selection validation, the cache, acquisition and
offline resolution. Settings and `ui/app.py:launch_app` both use that resolver.
Gradio receives the resulting native theme object at `Blocks.launch(theme=...)`,
as required by the installed Gradio 6.31.0 API. Settings uses the same resolver to
compile theme CSS for the running browser. `ui/html.py` owns the live CSS update
and the class that enables the selected palette. Shared application CSS in
`ui/assets/ui.css`, `shell.css` and `palette.css` supplies typography and layout.

Acquisition uses an unauthenticated `huggingface_hub.HfApi(token=False)` public
Space metadata request and Gradio's `get_theme_assets` / `get_matching_version`
semantic-version selection. Discovery filters out non-schema files first.
The JSON request uses `hf_hub_url(..., repo_type="space", revision=info.sha)`,
binding it to the inspected revision. Metadata and streamed-content limits are
1 MiB, with bounded connection/read timeouts. The schema accepts known native
theme tokens and validates references with Gradio's native CSS compiler. It
does not use the native `Theme.from_hub()` startup path, which would acquire
implicitly and load unrestricted stylesheet/font fields.

The cache is under the writable data root at
`cache/themes/<sha256-of-theme-id>.json`. Each entry records the requested ID,
published revision and normalized schema. Acquisition publishes by atomic
replacement; normalization and size checks also run when reading the cache.
Concurrent download events use the `theme-download` queue group. Replacing a cache
entry alone does not alter open browsers; the selector's change event applies
the resolved CSS to that browser. Saving retains the selection for startup and
does not broadcast a theme change to other sessions.

Upstream references: [Gradio theming guide](https://github.com/gradio-app/gradio/blob/main/guides/11_other-tutorials/theming-guide.md)
and [Gradio 6 migration](https://github.com/gradio-app/gradio/blob/main/guides/11_other-tutorials/gradio-6-migration-guide.md).
The API was checked against installed Gradio 6.31.0 on 2026-11-09; the coordinator
used Context7 library `/gradio-app/gradio` for the launch-time theme contract.

Focused CPU validation:

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_ui_themes.py test_ui_runtime.py test_guest_localization.py test_html_templates.py test_settings_ui.py test_profile_transactions.py
```

The tests cover native schemas, mocked explicit downloads, revision/version
selection, malformed and oversized data, cache failure recovery, offline and
legacy choices, and real profile save/reload with the startup resolver binding.
They also exercise live CSS updates, Default fallback, guest startup head and the
restart watcher's response validation using synthetic fixtures and local ASGI requests.
Computed native color pairs for curated body text, labels, primary buttons,
secondary hover, selected controls and links meet 4.5:1 in light and dark mode.
This is source/CPU evidence. Actual browser layout, focus/disabled states,
mobile readability, text scaling, zoom and restart behavior require separate
browser acceptance; no real theme download or GPU execution is certified here.
