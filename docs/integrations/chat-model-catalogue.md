# Local Chat catalogue and actions

`features/chat/llama_models.py` discovers GGUF variants recursively in the configured
Chat and vision folders, plus the saved explicit local file. Imported files keep
their absolute identities; duplicate basenames remain distinct. Projectors are
excluded using their name or bounded `general.architecture`/`general.type` scalar
metadata. `chat/gguf_metadata.py` caches that inspection by resolved file signature,
closes the GGUF mapping and reads no tensor values. Unknown metadata does not
establish runtime support; loading still applies the existing native checks.
One malformed or truncated local file cannot abort discovery of other choices.

Installed choices describe real files. Download candidates describe missing
configured catalogue destinations. A custom file with the same basename in another
folder does not establish that a packaged candidate is present. Presence is a path
observation, not checksum verification; preparation and loading retain their own
identity and runtime checks. No model discovery action downloads or hashes weights.
Capture and direct model resolution use the same canonical destination for a
relative catalogue name. Choosing an absolute custom path keeps that separate file.

`model_browser.model_action_state(value, downloads, session, roots=...)` returns
`download_visible`, `cancel_visible` and `load_available`. An absolute installed
selection uses its exact path. A downloadable basename uses its declared tuple and
configured category roots. Download is offered for a missing selected candidate
when no owned transfer is pending; Cancel is offered only for the session's pending
transfer. These projections never activate or unload a model. Eligibility checks
still decide whether a present file can load on the selected runtime.

`selection_choices` returns valid browsing choices or an empty selection rather
than injecting an unavailable artifact. `ChatModelSession.active` remains separate
from its browsing selection and transfer revision. UI search/source filtering can
clear browsing without replacing the already loaded model or its Chat history.
The UI owner wires these helpers to selection and transfer terminal events.

Each Chat app keeps at most three `ModelCatalogueCache` browsing views (combined,
Installed and Download). Warm search/filter events reuse that snapshot without
recursive discovery or GGUF header reads. Deliberately expanding **Model library**
revalidates files changed outside the app, preserves an available exact selection,
and reapplies search. Reindex remains available to internal callers; no manual
Refresh button occupies the main flow. Import, hide/delete, transfer
termination, verified Hub GGUF commits in Chat/vision roots and Hub selection
invalidate it; configured roots, hidden files,
saved explicit path and catalogue destinations also form its context. Concurrent
cold callers share one scan, and a writer invalidating an in-flight scan prevents
that older result from becoming the snapshot. Live `local_models`, preparation
and model resolution retain their filesystem and identity checks.

Search publication uses the Chat session lock, its selection revision and a
per-session search version. Refresh builds its catalogue outside the session lock,
then checks the same revision/version before selecting or publishing. A slow
superseded search or refresh cannot clear a newer query's
selection or replace a newly chosen candidate; active runtime and history remain
separate. Refresh preserves a still-available exact quantization and reapplies
the current filter; it never chooses a sibling file implicitly.

Focused checks include `test_chat_model_browser_work.py`,
`test_chat_model_browser_race.py`, `test_chat_catalogue.py` and
`test_chat_model_identity.py`.
They cover imported vision GGUFs, multiple local quantizations, custom files sharing
catalogue basenames, filtered selections, owned transfer controls, bounded cached
metadata and a tiny real GGUFWriter/Reader fixture. No real-model generation,
performance, GPU or rendered-browser qualification is implied. Sampling-source
resolution and effective-default attribution are a separate follow-up contract.
