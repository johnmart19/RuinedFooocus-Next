# Recovery after removed application storage

The writable data root is the application folder by default, or the existing
directory selected by `RF_DATA_DIR`. Missing default `settings`, `outputs` and
`cache` folders are recreated when their owning services next use them. Default
image/Chat model categories under `models` recover at startup, Recheck folders,
actual model refresh/scan or first-use preparation. The finite category inventory
is `core/model_folders.py:MODEL_DIRECTORIES`; no legacy alias folders are created.
Model roots overridden by custom profile paths are not created automatically.
Recovery
creates empty folders; it does not restore removed images or other original files.

Removing a profile's settings or path document causes the next configuration read
to reload that document from factory defaults. The current profile name is kept.
New dictionaries replace the old in-memory configuration, so an accepted request
retains the settings and paths it already captured. A configuration change
notification is published after settings recover.

Gallery callbacks open and close their own SQLite connection. If the default
index has disappeared, they initialize a new schema. Opening Gallery with an
uninitialized index completes its initial scan through the bounded preparation
pool before publishing the page. This also rebuilds an index cleared by Maintenance
before Gallery's first visit, without an extra Refresh action. A completed empty
index is distinct from an uninitialized one; ordinary paging never starts a scan.
Recreating default outputs invalidates
their old rows before the next page read and enables a new initial scan. Indexed
archive rows and their source files are retained.

Automatic recovery is limited to known default application folders. It does not
revive a missing custom source, archive, cache or model mount. Initial explicit custom
output/cache configuration keeps its existing behavior. Model preparation captures
its root in `PreparationRequest.model_root` and checks it again before creating
destination subdirectories, so a queued transfer cannot recreate a vanished custom
root. Default categories use the same guarded recovery at that boundary. The writable data root itself
must exist, and owned-folder recovery rejects symbolic links and Windows reparse
points. Permission failures, incompatible database schemas and other filesystem
errors remain errors rather than being treated as empty storage.

`tests/test_storage_recovery.py` exercises these lifetimes with private temporary
folders, including factory settings, captured dictionaries, database reopening,
custom roots, link rejection and permission failures. It never removes real user
settings, outputs, caches, models or runtime installations.

`tests/test_model_directory_recovery.py` additionally covers the real Models
inventory and refresh callback after live deletion, default/RAG first-use transfer,
custom-root loss before and after capture, reparse replacement, and unchanged
original bytes. CPU fixtures do not recover deleted weights or certify model/GPU behavior.
