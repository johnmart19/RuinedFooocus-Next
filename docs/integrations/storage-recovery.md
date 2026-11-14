# Recovery after removed application storage

The writable data root is the application folder by default, or the existing
directory selected by `RF_DATA_DIR`. Missing default `settings`, `outputs` and
`cache` folders are recreated when their owning services next use them. Recovery
creates empty folders; it does not restore removed images or other original files.

Removing a profile's settings or path document causes the next configuration read
to reload that document from factory defaults. The current profile name is kept.
New dictionaries replace the old in-memory configuration, so an accepted request
retains the settings and paths it already captured. A configuration change
notification is published after settings recover.

Gallery callbacks open and close their own SQLite connection. If the default
index has disappeared, they initialize a new schema; a refresh can then rebuild
its rows from the remaining original images. Recreating default outputs invalidates
their old rows before the next page read and enables a new initial scan. Indexed
archive rows and their source files are retained.

Automatic recovery is limited to known default application folders. It does not
revive a missing custom source, archive or cache mount. Initial explicit custom
folder configuration keeps its existing behavior. The writable data root itself
must exist, and owned-folder recovery rejects symbolic links and Windows reparse
points. Permission failures, incompatible database schemas and other filesystem
errors remain errors rather than being treated as empty storage.

`tests/test_storage_recovery.py` exercises these lifetimes with private temporary
folders, including factory settings, captured dictionaries, database reopening,
custom roots, link rejection and permission failures. It never removes real user
settings, outputs, caches, models or runtime installations.
