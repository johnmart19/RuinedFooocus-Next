# Confirmed folder maintenance

Settings → Maintenance → Folder maintenance offers **Recheck folders** and three
previewed actions. Opening the panel does not scan or delete files.

- **Clear rebuildable cache** resets eligible model-hash records, the Gallery
  index, derived thumbnails and startup checks. It retains other cache data,
  including legacy runtimes, RAG model payloads, active inputs and resumable work.
- **Clear cache and outputs** adds regular files and empty subfolders below the
  default application `outputs` folder. Custom cache/output roots are retained.
- **Factory reset settings** replaces the current profile's `settings.json` and
  `paths.json` together through the existing recoverable profile transaction.
  Other profiles, models, repositories and original files remain untouched.
  Restart the app to load its factory startup/folder defaults; current unsaved
  controls and loaded models are not rebuilt by this action.

Each action first shows the exact paths, file sizes and affected index-row counts,
plus retained paths. Output deletion and factory reset require the permanent
deletion checkbox and **Confirm reset**. A changed target snapshot requires a new
preview. Recheck uses the existing owned-folder recovery service; it does not
invent missing custom mounts or follow linked targets.

Execution requires idle compute, preparation and profile owners. Existing leases
protect active files, and new work waits outside the maintenance window. Cancel
is cooperative between deletions; a paired profile save finishes or rolls back as
one operation. Partial cancellation or errors report an incomplete result and the
actual reset count, rather than success. Only previewed cache files are eligible;
files appearing afterward are retained.

The private-fixture tests exercise confirmation, stale previews, busy owners,
profile rollback, links/custom roots, protected legacy payloads, cancellation,
permission failures and concurrent admission. They never reset real user folders.
