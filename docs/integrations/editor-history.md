# Owned editor source and history

`features/gallery/editor_session.py` owns schema 1 of a private artifact
manifest. It joins existing Create, Library, API and optional Chat image services.
Existing native loading, sampling, composition, upscaling, preparation and shared
compute admission retain execution ownership. The accepted native source pins
remain in [`app/bootstrap.py`](../../ruinedfooocus/app/bootstrap.py).

## Source and request contract

`EditorSessionStore.start(trusted_path, provenance=..., owner=...)` adopts an
existing still image. `SessionSnapshot` keeps immutable `original`, separate
`current`, `revision`, `head_id`, inherited `source_generation` and bounded
completed `steps`. Source replacement starts another session. The history store
retains no full-size pixels. `open_editor(record, metadata, ...)` is the Library
adapter; `snapshot` reads and `reopen` recovers the same owned manifest.

`ArtifactRef` contains opaque `artifact_id`, kind, SHA-256, existing file signature
and `ImageFrame`: oriented width/height, mode, dtype, color space and alpha policy.
Internal refs include the validated path; `to_dict()` omits it. Server owner
tuples come from `resource_lifecycle.current_owner`, never job JSON; manifests
store their hash. `artifact(session_id, artifact_id, revision=..., owner=...)`
resolves a source inside that owned session. Handles cannot select arbitrary paths.

`SessionSnapshot.to_dict(public=True)` also omits explicit path/path-list fields
and path-bearing signature entries nested in inherited provenance, plans and
steps. It preserves hashes, resource names, opaque IDs and literal prompt text.
Internal `source_generation`/`steps` remain exact for replay.

[Editor prompt conditioning](editor-prompts.md) describes literal instructions,
native edit versus ordinary image-to-image, and bounded source prompt context.

## UI import and Gallery drafts

Editor **Open current image** adopts the displayed still; **Latest generated**
uses a separate request-local accepted final-image record. Terminal result
presentation assigns an output role from saved producer semantics. Selected
batch index, original path and file signature identify the final image; technical
maps, masks, helpers and previews do not replace this shortcut. All files remain
in the result/Gallery surfaces, and maps retain their explicit guide actions.
Without such a record, an Editor with no open image takes the first entry of the
Gallery index (newest dated output); an open image is kept.
A thumbnail, matching dimensions or filesystem sort order never selects an
original. Precision originals and alpha remain unchanged.

**Choose from Gallery** opens an eight-image page inside Editor through the
existing SQLite index and thumbnail service. A separate picker request fence
guards page, selection and import; a newer page, closed picker, changed or missing
original rejects an old choice. Paging reads bounded rows without scanning
outputs again. A fresh index is populated on first picker use through the existing
incremental scan and preparation admission; its commit uses the picker request
fence. Concurrent first opens share the index writer lock, and an initialized
snapshot (including an empty one) is reused without another folder scan. Source
import/history callbacks declare only affected source
controls and scope progress to their notice. Gallery opening prepares pixels
once after session admission; metadata updates use the existing draft fence.

Browsing and import preparation do not stop an accepted job. While the Editor's
task owns its terminal, source/history adoption explicitly defers and asks the
user to retry after completion or explicit **Stop**. Draft input changes apply
to the next request; the current job keeps captured inputs, history and terminal
ownership. A newer source adoption after the job completes still supersedes old
unpublished UI work. Stop and existing shared compute admission retain their
own cancellation/serialization contracts.

Gallery has a separate prompt/setup/edit draft. Prompt/setup reuse changes local
controls through the same final-paint draft fence as metadata selection. Trusted
setup snapshots stay in bounded server State; Run resolves only the token that
the browser actually accepted, so a late rejected setup cannot change sampling
behind newer text. Each snapshot binds the source path/signature; a missing or
other-source painted token fails with an explicit request to reuse setup again.
**Use current sampling settings** explicitly clears that saved setup, revokes
older reuse work through the same fence, and retains the local text and source.
Common edit, inpaint, upscale and background actions choose that local
operation. Its Run captures an owned source session and a frame-bound mask before
submission through the existing worker. Stream/task/result/download outputs stay
local, including Stop's exact task ID; only a completed final-image identity may
update the shared Latest shortcut. Browsing another source stages the next local
draft without cancelling the captured job or clearing its result. **Open result
in Editor** opens the same completed history. **Open in Editor**, grid/composition
handoffs and **Continue in Create** remain explicit navigation choices.

`bind_image` adopts an additional existing background/overlay without moving the
head. `bind_mask(session_id, path, image_id, inverted=..., owner=...)` accepts only
an 8-bit single-channel selection mask with the exact oriented source dimensions. It
records `source_image_id`, `source_frame_id`, purpose `selection-mask`, channel
`L` and explicit inversion. RGBA cutouts cannot act as masks. Generic operations
bind masks to current; `image-composite` binds one to the first additional overlay.
Same-sized masks from another image still fail image/frame identity checks.
Precision image originals remain supported; a precision mask needs an explicit
caller conversion before adoption, so later L8 processing cannot silently quantize it.

`capture(session_id, revision, operation, parameters, source_id=..., mask_id=...,
additional_image_ids=..., effective_plan=..., owner=...)` returns a frozen
`OperationTicket`. Parameters and optional P1 plan metadata are bounded JSON.
Tickets contain immutable artifact refs. Only current can be the primary source.
A newer capture invalidates an older pending ticket even at the same revision.

Shared helpers are `capture_request`, `validate_request`, `complete_request`,
`cancel_request` and `fail_request`. Public `editor_context` accepts only
`session_id`, `revision`, `source_id`, optional `mask_id` and
`additional_image_ids`. Private `_editor_operation` is a captured Python ticket,
never a deserialized public ticket. Editor jobs return exactly one image. P1 is
consumed through its captured plan's `metadata()` protocol. P2 preparation is
called by existing services only after source admission.

When Library supplies a draft ticket before final P1 capture,
`bind_effective_plan(data, plan, owner=..., store=...)` atomically binds the final
metadata and replaces that ticket. It preserves source/mask/parameters, invalidates
the old operation ID, and refuses a ticket that has started. Identical metadata
retains the existing ticket. Workflow adapters must also update their step's ticket.

## Caller and terminal boundaries

`editing_workflow.build_library_request` accepts optional `editor_context`,
`editor_store` and trusted `owner`. It resolves current before reading pixels,
requires the Library record to match, and freezes the normal operation request.
Callers without context retain their existing source-local behavior.
Scalar intent uses the existing Library replay field contract, including native
guidance fields, `cn_strength` denoise and explicit image resize mode. Owned
inpainting requires an adopted `mask_id` before pixel preparation and loads that
mask's authoritative frame/inversion; an unrelated supplied brush is not reused.

`image_composition.capture` preserves the ticket and workflow marker. With a
ticket, `bind_editor_inputs` reads its current/additional artifacts through the
existing `working_source` conversion, replacing unbound uploads and validating
again after loading. Ordered composition keeps native operations and its
uniform contain/pad geometry. Precision originals remain untouched; the existing
8-bit sRGB working conversion stays explicit.

API/worker integration must obtain actual input pixels from the ticket, validate
before preparation and compute, and call `validate_request(..., start=True)` at
most once. Inline uploads cannot claim an unrelated owned source. Integration
owns these boundaries in `app/api_runtime.py` and `app/async_worker.py`; this
service alone adds neither an HTTP route nor editor UI. P6 owns the controls.

`complete(ticket, result_path, execution=..., owner=..., check_cancelled=...)`
requires a durable result and revalidates owner/ticket/revision/source signatures.
`describe_frame(trusted_path)` exposes the actual result frame to the terminal
adapter without converting its pixels or inferring a geometry policy.
It never writes a source. Changed dimensions require `execution.geometry` with
`policy`, `source: [width, height]` and `result: [width, height]`. Losing alpha
requires `alpha_policy`; changing precision requires `precision_conversion`.
This records deliberate conversion rather than silently stretching or applying
alpha to an incompatible frame.

Each completed step has `operation_id`, `operation`, `source_id`, `mask_id`,
`additional_image_ids`, scalar `parameters`, optional `effective_plan`,
`parent_id`, `result_id`, `execution` and `completed_at`. Requested/rendered
prompts, negatives, seed/denoise, LoRAs, model/component IDs and size ownership
remain in inherited `source_generation`. They do not become utility execution.
Pure composition continues to clear prompt/model/sampling worker fields.

New utility PNGs retain a bounded `source_generation` recipe from the current
owned source's latest generation ancestor, separately from their actual
processing metadata. Replay restores its inert prompt/model settings through
normal installed-model resolution, uses the utility output's current dimensions,
and disables inherited automatic upscaling. It never follows an `image_source`
filename or re-executes the tool chain. Existing outputs without that recipe are
unchanged. An RGB result may discard a source alpha channel only after validating
that every source alpha value is fully opaque; the execution records this policy.

`cancel`/`fail` clear only the matching pending ticket and add no step. Source
replacement, expiry, later capture, undo/redo and duplicates reject publication.
Completion uses the existing registry cancellation lock for an active owned job.
API terminal publication must also hold its terminal/cancel lock and supply
`check_cancelled`, making publication and history one terminal decision. A
preflight no-op can skip capture/completion and retain the revision.

## Persistence, navigation and recovery

Atomic JSON writes use `core.config_io.save_json`, existing preparation artifact
leases and a process lock. Default private storage is configured cache's
`editor-sessions` leaf. It is not a public media root and needs no Gradio path
allowlist. Artifacts stay in existing trusted input/output/export locations and
their approved media routes. Leases serialize cooperating callers inside one
RF process; sharing a writable history root across RF processes is unsupported.

Defaults are 32 completed steps, 64 sessions, 136 refs/session, seven-day expiry,
128 KiB per metadata snapshot and 2 MiB per manifest. Overflow drops old refs,
never source/result files. Original/current stay pinned. Old branches or undo
predecessors can expire at the bounded boundary. Unused bound sources can fill
the ref limit, requiring a new session. Session capacity rejects new history
until expired private manifests can be removed; active histories are not evicted.

`undo`/`redo` move the head and increment revision without regeneration. Editing
an old head branches and clears redo, preserving output artifacts. `reopen`
validates refs and discards pending work rather than resuming an interrupted job.
Schema 1 has no older history format to migrate. Unknown schemas fail without
rewriting files; reopen the original as a new session for recovery. Missing,
changed or expired artifacts fail admission instead of substituting another image.

## Focused checks

```sh
python tools/check_cpu.py test_editor_session.py test_library_editing_workflow.py test_image_composition.py test_image_composition_integration.py test_api_image_edits.py test_upscale_transparency.py test_resource_lifecycle.py
```

UI identity/caller checks additionally use `test_editor_gallery_imports.py`,
`test_editor_workspace_events.py`, `test_editor_workspace_actions.py`,
`test_library_editing_ui.py`, `test_gallery_index.py`, `test_gallery_scheduling.py`
and the native-output transport callers. `test_editor_gallery_measurements.py`
records one cold and two warm model-free route samples and callback/output counts
when `RF_EDITOR_GALLERY_RECEIPT` is set. It measures Python/Gradio fixture work;
queue wait, transfer, browser presentation and actual model execution are outside
that receipt. A measured reduction in preparation/output count alone does not
certify improved production latency.

Local PIL/PNG fixtures, native PNG16 inspection, real caller DTOs and the existing
lifecycle registry cover nondestructive lineage, reopen/metadata/odd RGBA/
precision, undo/redo/branch, two owners, source replacement, mask frames, stale
publication, failure/cancel, bounded refs, atomic failure and terminal races.
CPU checks do not establish rendered UI, model quality, performance, GPU stability
or complete HTTP journeys. Keep exact source/fixture/closure receipts with the
receiving task's evidence record.
