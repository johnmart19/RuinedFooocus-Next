# Responsive preparation and Library ownership

GPU/model state still belongs to `JobRegistry` and `FairAdmission` in
`app/resource_lifecycle.py`. All production heavy services share
`compute:shared`. A separate Gradio queue or available VRAM does not authorize
parallel native model use. The existing compute cancellation, authenticated
owner/guest boundaries and model leases remain authoritative.

## Preparation and UI requests

`PreparationAdmission` retains two workers and eight total admitted jobs.
`submit(prepare, ready, failed)` keeps the compute caller contract;
`submit_work(prepare, cancelled=event)` returns a future for CPU/I/O-only work.
Both use the same bounded pool and copy the caller's context. Queue-full/closed
submissions fail promptly. Close signals active cancellation events, cancels
queued futures and shuts down the executor. Work must check its cancellation
event at useful I/O boundaries; cancelling a running Python future cannot
interrupt synchronous code.

Artifact transfer admission is independent of that pool. In `PreparationService`,
the last subscriber to a waiting transfer cancels its executor future and returns
the destination and bounded job slot immediately. Cancelling one subscriber keeps
the other subscribers' transfer alive. Active transfers retain ownership until
they reach a cooperative cancellation boundary and unwind. Executor submission
failure rolls back the captured destination so a retry is possible.

`LatestUIRequests` binds each channel to an authenticated owner and a monotonic
ordering version. `begin(owner, channel, version)` invalidates its prior ticket;
out-of-order acknowledgement cannot replace a newer one. `get(..., start=True)`
claims a ticket once. `publish(ticket, callback)` checks current ownership and
runs publication under the same lock as invalidation. `attach(ticket, future)`
binds queued preparation to that ticket; superseding/cancelling it cancels a
waiting future and returns its slot before other active I/O completes. `peek`
permits read-only projections of a current selection. `cancel` signals work;
`finish` permits bounded retention. Successful authenticated logout calls
`cancel_owner(login_prefix)` from `PrivateResponses`, invalidating every tab and
channel for that login while leaving other login tokens alone. The fallback
request path derives the same authenticated prefix and appends its session hash.
Client JSON never supplies owner identity.

The Library captures a browser-tab ordering counter through
`gallery.ui.request_version_js`. Queue-free acknowledgements remain reachable
while preparation runs. A hidden request descriptor triggers an `always_last`
Gradio event in a two-slot Library lane, which awaits the bounded preparation
pool. Page and selection channels are separate. Progress is restricted to the
Library status or metadata control. Refresh preserves page/search; search starts
at page one. Cancel starts a fresh browse request against the previous index.

Page records carry `_view_version`; selected records also carry
`_selection_version`. Selection, prompt/setup reuse and every image-tool handoff
validate those versions before using the source. A prepared older page arriving
after a newer acknowledgement cannot activate a wrong source. Server publication
and SQLite commit are fenced. A response already returned to Gradio can still be
delayed in transport; the counter alone does not establish a latest-painted
frontend guarantee. Browser acceptance must inspect that visible timing separately.

Installed Gradio **6.31.0** `events.py` and `queueing.py` were inspected on
2026-11-11. Shared `concurrency_id` uses the lowest declared limit; cancellation
does not stop a running synchronous callback; `always_last` schedules a final
submission after the pending one. Context7 `/gradio-app/gradio` was queried only
for those semantics and cross-checked against installed source. The API source
is [Gradio events](https://github.com/gradio-app/gradio/blob/gradio%406.31.0/gradio/events.py).

## Model catalogue and local checkpoint projections

`features/models/ui_requests.py` owns model UI projection scheduling. Filters,
refresh, details/roles/context and local checkpoint descriptions use separate
per-owner channels and browser ordering counters. Queue-free Gradio callbacks
admit CPU/I/O reads to the existing two-worker/eight-slot preparation pool.
Replacing a request cancels its waiting future; active scans check cancellation
between catalogue entries and paths. A late success or error cannot publish over
a newer request. Model/device mutation still requires native compute admission.

Card selection acknowledges only the key and source identity, without metadata
inspection. Details are prepared separately with progress restricted to their
status control. Filters and refreshes retain the browser selection, including a
details-only row when its card is outside the filter; hidden page records carry
their view version, including an empty-list descriptor. Download and Hub Use
validate the current server-owned selection and view before resolving a file.
Login/session ownership comes from the same authenticated request prefix as
Library; JSON never supplies an owner. Layout construction and trusted direct
builder calls have no browser context and retain their pure projection interface.
Hub and Library tool handoffs update the actual `Tabs` owner of `image-tools`.
Gradio can create separate implicit groups for Generation/Models and Prompt
tools/Image tools, so the former cannot select a tab in the latter.

The local checkpoint browser seeds its server-created initial snapshot at version
one. Its first browser query advances beyond that seed, and a cancelled/newer
ticket cannot be revived by initial rows. Custom card payloads carry view/page
versions; `adapter_cards.js` rejects a returned older payload before painting and
does not reset a newer browser selection. Standard Gradio widget delivery after a
response has already left the server remains a separate browser acceptance row.
Source-action guards continue to reject obsolete rows even if a widget paints late.

`Catalogue.invalidate()` increments a cache generation under a short publication
lock. Scans run outside that lock; an older scan can return to its original caller
but cannot replace a newer shared snapshot. Filter-only callers reuse the current
snapshot. This prevents refresh invalidation from being lost during slow scans.

Installed native utility activation rebuilds its Dropdown through
`ui_native_vision.model_choices`, the same header-only owner used by the tool
picker. It retains every compatible choice and publisher/native display label
instead of appending a lone technical path to the process layout's stale defaults.
The exact resource value, image checkpoint and prompt remain independent inputs.

Focused CPU coverage adds `test_model_ui_requests.py` and
`test_tool_activation_choices.py` to the existing model, Library, lifecycle and API
callers. Barriers cover invalidated-cache publication, delayed model descriptions,
two sessions, stale success/error, queued-capacity return and wrong-source action
rejection. The real card JavaScript runs under Node for delayed-payload rejection;
the tool regression exercises an actual Gradio Dropdown callback and session config.
These checks do not certify actual full-app transport paint latency or native inference.

## Incremental Library index

`gallery/index_store.py` owns index schema v2. The configured cache root holds
`images.db`; an old data-root database remains untouched when roots differ.
Legacy rows remain readable and acquire signatures on first refresh. Unknown
newer schemas are rejected without mutation.

Each callback owns its SQLite connection. WAL readers see the previous complete
snapshot while a process-wide per-database writer scans. Canonical source identity
deduplicates root aliases/overlaps; the first root retains its relative filename.
Size, mtime and ctime nanoseconds identify changed files. Unchanged metadata is
reused, including when root ordering changes. Source changes during parsing,
missing roots, I/O/commit failure and cancellation roll back the transaction.
The final commit also checks the owning UI ticket under its publication lock.

A stable file rejected by a media decoder is omitted from the new index while
valid images remain available. The refresh reports the total rejected files and
up to 20 filename/error details (1 KiB each); rejected files are retried on the
next refresh. PNG metadata reads verify the file instead of silently indexing
failed reads as empty metadata. Unexpected parser errors and filesystem failures
still preserve the complete previous snapshot; decoder rejection never removes
an original output file. A source change or cancellation during rejection also
rolls back the whole refresh.

Paging counts and selects rows in one read transaction, stats/copies only the
selected page and caps it at 100 files. It does not stat the entire index or
delete missing entries during browsing. Refresh removes missing paths in the
same complete-snapshot transaction. Original image paths, dimensions, alpha and
generation metadata remain the authority for actions.

## Action inventory and remaining acceptance

Private API activity, installed Chat runtime inspection and native image file
transport have distinct ownership groups. Activity is a short redacted memory
snapshot, runtime inspection owns its bounded child and per-session cancellation,
and image import/export owns bounded codec work plus atomic no-clobber files in
its own Gradio lane. These actions do not acquire native model execution merely
because they appear beside model controls. Histogram analysis currently uses the
admitted serial worker even though it needs no generation model.

`test_action_contract_owner_boundaries.py` checks those specific source groups
before generic UI patterns, with isolated registration fixtures. Final manifests
must bind the actual source and contract hashes; a zero-unassigned source report
does not certify runtime expansion, browser reachability or delivery latency.

Generate source-bound external evidence without importing the app:

```powershell
venv/Scripts/python.exe tools/audit_action_contracts.py --strict --output <owned-evidence>/action-manifest.json
```

The receipt includes hashes of every declaring source, a full source span for
each registration, lexically resolved callback identity/hash, inputs/outputs,
queues, progress, shared option dictionaries and cancellation settings. Chained
events have distinct IDs even when their starting line is identical. Ordinary
executor submissions, session selections and model loads are excluded explicitly.
Schema v3 also records direct/decorated `gr.api` handlers and FastAPI
`add_api_route` registrations, retaining the handler rather than mistaking the
route path for a callback. API names and explicit HTTP methods remain source
expressions. Legacy synchronous compute RPC, local RPC reads and image capability
delivery have separate resource/cancellation contracts. Gradio callback cancellation
does not establish an immediate interrupt guarantee for a synchronous legacy RPC.
Custom HTML `navigate` and `paginate` registrations carry the same source and
callback receipts. Ordinary methods with those names remain excluded unless
the call supplies Gradio event options in a Gradio source module.
Localization registrations and enclosing loops remain visible as helper sites;
one helper can bind several runtime controls.

[The reviewed action groups](../../tools/action_contracts.json) are the maintained
owner/cancellation/persistence contract for every captured source site. `--strict`
fails if a new source has no matching group. It does not infer a safety guarantee
from a callback name. External callback factories remain marked for following the
declaring source. Counts refer to source registrations/helpers/routes, never the
number of reachable runtime actions. Source mapping does not prove actual
browser delivery, native resource admission or every conditional route's live
reachability. Current receipts stay outside the source distribution.

| Reachable action group                                                     | State/resource owner and scheduling                                                                    | Cancellation and persistence disposition                                                                                                                    |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Library refresh, browse/search/page, selection                             | Per-session versioned snapshots; bounded preparation; SQLite single writer                             | Cooperative per-file cancellation, atomic index publication, stale action rejection. CPU/Gradio caller races verified.                                      |
| Create/Chat generation, load/unload, helper compute, API compute           | Existing request registry + model leases + `compute:shared`                                            | Existing job-scoped Stop/owner cleanup remains in force; new preparation pool callers are regression-tested. Live multi-service lifecycle remains separate. |
| Artifact downloads/component preparation                                   | `PreparationService`: two transfer workers/eight jobs, destination single-flight, subscriber context   | Subscriber cancellation is independent; verified no-clobber publication under artifact lease; queued/active work protected from cleanup.                    |
| Model catalogue/filter/artwork/defaults, model-specific UI transitions     | Model feature owner (`model_handler`, `ui_acquisition`, `ui_adapters`, resolution/component callbacks) | Static declaration inventory available; complete callback-by-callback mutation/staleness and live journeys remain W02/W03/W13 acceptance.                   |
| Chat attachments/retrieval/indexing                                        | Request-owned attachment corpus; pinned embedding preparation and model/source identity                | Existing request cleanup/leases; pool/Chat callers regression-tested. Browser cancel/retry, export and two-session journeys remain separate.                |
| Profile/paths, runtime install/update, package acquisition, source updater | Their existing settings/config transaction, package plan and runtime installer owners                  | No global queue policy was changed. Runtime/persistence action audit and simultaneous live operation remain with their owners.                              |
| Maintenance derived caches                                                 | Allowlisted cache owner + inactive artifact write lease; bounded preparation lane                      | Targeted operations only; inputs, pending jobs and runtimes protected. Separate Maintenance fixtures/UI acceptance required.                                |

Focused checks use `tools/check_cpu.py` with `test_gallery_index.py`,
`test_gallery_scheduling.py`, `test_ui_request_fencing.py` and the existing
gallery, lifecycle, preparation, path and Chat callers. They exercise real
SQLite transactions and installed Gradio callback construction with barriers,
without starting an app, model, download or GPU. Warm fixture scans parse zero
unchanged metadata. Real catalogue/gallery latency, network-visible ordering,
simultaneous downloads and Chat→image→Chat hardware acceptance are not certified
by these CPU checks.

## Controlled service and browser acceptance

[The inert fixture](../../tools/concurrency_fixture.py) launches a private
localhost Gradio service using the production `PreparationService`,
`PreparationAdmission`, `LatestUIRequests`, `FairAdmission` and SQLite index
functions. Its A/B files are synthetic bytes with `.inert` extensions; Use B
only reads a checksum under admission. Its Library holds 128 tiny synthetic PNGs.
It imports no application state/native runtime, installs nothing, and acquires
no model. A new exclusive output directory owns every fixture file and receipt.
The service is limited to 900 seconds and all controlled I/O holds have deadlines.

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_concurrency_service.py
venv/Scripts/python.exe tools/concurrency_fixture.py --port 7906 --seconds 600
```

Use an unused port and the built-in browser. The launcher prints its owned path,
PID, URL and source receipt; it retains a final receipt after closing. The fixture
never restarts or closes the ordinary app. The generated receipts contain full
source hashes, Gradio/Python versions and per-operation timings. Source identity
is captured once at fixture construction and retained through shutdown, so a
later checkout edit or rebase cannot relabel an already running service.
Draft Textboxes explicitly set `interactive=True`; no worker/navigation event
outputs either draft. Export the current source/service receipt with the Settings
button (or queue-free `export_receipt` API) before ending a check. To close the
owned fixture gracefully, create a file named `stop` in its printed output root;
the normal close path writes `final-receipt.json`. An external hard termination
can skip Python cleanup, so a missing final receipt must never be claimed as saved.

The real-service test uses two independent Gradio Client sessions plus real
queue-free HTTP requests. While A's HTTP body and one SQLite scan are delayed,
the other session can Use B, filter the previous complete index and navigate.
It checks subscriber-specific cancellation, no partial final artifact, verified
retry, cancelled scan rollback, metadata failure/retry, newer-request publication,
two-session isolation and a warm scan with zero reparsed files. It records warm
filter p50/p95 through both Client queue transport and the direct queue-free
`/gradio_api/run` route. These routes have different transport overheads; neither
measurement certifies browser acknowledgement or paint.

For browser acceptance, complete the following within a controlled hold deadline:

1. In Settings, start **Download A**. In Create, edit the draft and **Use installed
   B**. Visit Chat and edit its draft, then return and verify both drafts persist.
2. In Library, leave **Hold refresh** enabled and start **Refresh Library**. Filter
   and browse the previous committed snapshot, navigate all four tabs and verify
   only the refresh status shows loading. **Cancel refresh** must preserve its
   previous revision; the older callback must not repaint its result.
3. Cancel A, then release its HTTP body. No final A file may appear. Reset A,
   start it again and release it; completion must publish the expected checksum.
4. Start another held refresh, then submit a newer refresh with Hold disabled.
   Only the newer version may become visible. Exercise **Fail metadata read**,
   confirm the previous revision remains, then retry successfully.

Record actual visible acknowledgement against 200 ms, warm result against one
second and visible cancellation against one second. Browser screenshots and
network-visible delivery belong to the coordinator's acceptance receipt. The
Create generation binding separately restricts Gradio's loading indicator to
its progress control; returning preview/source/button values does not put all
those output controls into a loading overlay. Model catalogue/selection races,
actual authenticated app journeys and real Chat→image→Chat remain distinct rows.
