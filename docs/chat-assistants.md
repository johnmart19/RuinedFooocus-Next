# Chat assistants and local models

In **Chat**, choose an assistant, with **New**, **Edit**, **Delete**, and **Refresh** directly
below the selector. Assistant and model actions wrap into separate buttons on
narrow screens. **Use at startup** remains a secondary action.
Selecting an assistant starts its conversation explicitly. **New** opens a name
and system-prompt draft; **Save** writes it to the persistent catalogue.
Malformed character cards are skipped so other assistants remain available.
**Save as new** creates a separate entry. Duplicate display names are
allowed. Editing a bundled assistant or character card creates your own copy.

Saving or refreshing the catalogue preserves the current conversation, message
draft, model, and selected assistant. Select the saved assistant to use its new
instructions. **Cancel** discards the editor draft. An assistant needs neither an
avatar nor a greeting. An optional uploaded image is saved with the assistant;
without one, its display uses the General chat image. General chat has an empty
default system prompt and a protected identity: renaming it keeps the same assistant.
It cannot be deleted. **Delete** asks for confirmation, preserves open messages,
repairs the selected/startup choice, and leaves shared/default images intact.
Deleting a bundled entry hides it from the catalogue without deleting its resources.

**Use at startup** saves that choice in the active settings
profile. Each newly opened/reloaded page reads the current saved choice and fresh
catalogue; existing sessions keep their
own selection. Ordinary selection does not change another client's persona or
save a startup preference. Guest sessions cannot save assistants or preferences.
The assistant selector follows the browser language for the built-in General
chat name. Names you save remain literal text, including when the language changes.
Refreshing repairs a removed selection and its portrait/instructions to General
without clearing messages or the draft. For an available selection, it refreshes
the name/image while keeping that conversation's instructions until explicit selection.

Saved assistants use stable IDs independent of names, under
`assistants/<stable_id>/info.json` in the writable data folder. Legacy absolute
catalogue selections remain readable. Save only accepts catalogue IDs, never a
client-supplied filename. Updates preserve greeting/context fields and use an
atomic configuration write. A draft includes the SHA-256 of the opened JSON;
if it changed, reopen **Edit** before saving. A process lock serializes saves
from this app's clients. This is not a cross-process editing database; avoid
editing these files in another process during a save.

The collapsible **Model library** offers installed-model search, friendly names,
quantization, **Import GGUF**, and a separate **Download** action for missing
weights. Installed entries correspond to local files. Search or changing the view
clears a selection outside the results while retaining the loaded model and open
conversation. Download appears only for a missing selected payload; Cancel appears
only for a pending transfer. Opening the library revalidates files added outside
the app; ordinary search reuses its current inventory. Search and catalogue
refresh do not load or download models. **Load**, **Loading…**, and **Loaded**
describe the exact selected file and captured runtime configuration. Changing
quantization or runtime settings requires another load. Runtime acquisition,
verification, device probing, and initialization show progress with **Cancel
loading**; unknown totals remain indeterminate. Errors allow retry without
claiming an uninitialized candidate is loaded. Canceling a reload keeps the
previously loaded model only while its original initialized runtime remains
unchanged. Unknown custom weights
remain selectable; actual architecture, companion, and backend checks still run
when loading.

**Settings → Local Chat → Chat model diagnostics → Refresh sampling recommendations** inspects the
selected installed GGUF. Import and explicit Load also prepare this source cache.
Selection, search and file-presence refresh never hash weights or fetch metadata.
Source details show whether advice is verified, unverified, absent or retained
offline. Refreshing a selected candidate does not change the active model.

After Load, diagnostics show **Recommended settings**, **Your overrides** or **Runtime defaults** and
the active model's effective sampling values and their winning sources.
The same values are reported for each Chat turn. Explicit request values override
native server arguments, saved sampling overrides and publisher/variant advice;
zero and false values stay explicit. Turning publisher defaults off retains user
overrides. Fields without a recommendation remain native runtime defaults.

**Attach files** lists the supported extensions beside the compact picker. It
accepts up to four files of 256 KiB each; select a `.rag` manifest by itself.
Ready files retain their remove and export actions. Image references are separate
and appear only for an image action or enabled model vision; upload appears only
for the Uploaded image source.

**Image generation**, **Rewrite image prompts**, and **Update Create prompt** are
independent permissions and start Off. Image generation applies to one message
and resets after submission. **Check image readiness** previews resource needs
without running an image. The optional **Image prompt to preserve** is an explicit
existing visual prompt: with rewriting Off, that text stays exact. Leaving it
empty permits a new visual prompt for the current request. Chat prose and an
unrelated Create draft are not automatically reused as a supplied image prompt.
Create updates use the captured draft; a newer draft keeps the proposal pending
for **Apply result** or **Dismiss result**. **Undo** restores only an unchanged
accepted edit. Changing permissions, model, source, assistant, or conversation
invalidates pending authority. Other Create inputs remain untouched.

Renamed registered quantizations retain their exact verified profile. Another
quantization can inherit the same packaged sampling recipe only after its actual
payload is matched to the same reviewed immutable artifact-repository revision
and its GGUF architecture agrees. A family name, source URL, base-model hint or
quantizer's new generation configuration is insufficient. This sampling advice
does not enable unverified reasoning, vision companions or runtime support.

Browser Python execution has been retired. Chat code blocks remain readable
content; there is no runner control, asset-download route, or replacement server
executor. Existing caches and chat history are left untouched. Old extra profile
keys are inert and are not rewritten for this migration.

The default model is unchanged: no compact candidate has passed all semantic
gates for both engines. See the maintained
[qualification record](integrations/local-llm-qualification.md).

**Show reasoning** controls display only. It is hidden until reasoning controls
are supported or the current history actually contains reasoning; unloading a
model does not hide reasoning already present in that history. Hiding the display
does not disable reasoning computation. Unsupported computation/effort preferences
are hidden while their saved values are retained. After loading a verified model,
**Reasoning computation** offers only the modes established for its initialized
backend and embedded template. **Auto** leaves the template default; **Off** and
**On** send the model's boolean thinking flag where supported. Boolean thinking
does not imply graded effort. GPT-OSS instead offers its known Low/Medium/High
effort values and no Off switch. Low effort is not disabled computation.
Its inapplicable Auto/On computation selector is hidden; supported Effort and
the independent reasoning display choice remain available.

Unknown models, custom native template/reasoning arguments, and unsupported
runtime builds do not expose these controls. LFM3 controls remain unavailable
until its native reasoning-budget/template interaction is resolved. Each turn
captures its browser's requested mode/effort. The status reports the template
arguments actually sent, without claiming that hidden reasoning computation was
measured. An explicit unsupported request fails with an actionable reason;
unsupported saved preferences stay inactive.

Rewrite mode and effort are separate preferences. Existing profiles without the
new rewrite mode retain their legacy behavior: Low maps to Off for supported
boolean-thinking profiles, Low effort for GPT-OSS, and the existing LFM3 policy.
Explicit Auto adopts the initialized template default. Saving unrelated settings
must not silently replace an absent legacy rewrite mode.

Source owners are `features/chat/assistants.py` (catalogue/save), `ui.py`
(session controls), and `llama_models.py` (weights). Save uses
`core.config_io.save_json`; startup selection uses the existing paired
`SettingsManager.save_profile` transaction. Focused CPU checks:

```sh
python tools/check_cpu.py test_chatbots.py test_chat_resources.py test_chat_model_browser.py test_ui_workflows.py test_html_templates.py test_download_link_audit.py
python tools/check_cpu.py test_model_profiles.py test_llama_rewrite.py test_chat_reasoning.py test_chat_preparation.py test_chat_vision_modes.py test_chat_boundaries.py
```

These checks cover storage boundaries, stale/failed saves, selection retention,
runner absence, exact reasoning transport, persona/tool serialization, and direct callers. They do not certify rendered browser layout,
model quality, GPU behavior, or physical 3 GB/6 GB cards.
