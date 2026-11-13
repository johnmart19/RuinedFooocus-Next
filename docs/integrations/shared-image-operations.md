# Shared image requests, editor publication and Chat tools

`app/async_worker.py` and `app/api_runtime.py` connect the existing image
services. Create, Library and authenticated image APIs freeze operation intent,
components, manual/replay sampling and geometry before preparation. Chat supplies
one validated instruction to those same owners; it does not add another image
pipeline, downloader, executor or history store.

## Capture and first use

The component snapshot and `SizePlan` remain the owners of their facts. A private
draft effective plan freezes active required roles before
`image_preparation.capture_image_assets`. Native adapters can canonicalize their
own control settings there, including Layered and MiniMax routes. The final
effective plan is captured after this normalization and engine capture, using the
same frozen component tuple. Only that final plan crosses preparation and enters
editor metadata. An incoming typed plan is verified before normalization; a
caller cannot change its negative, denoise, LoRAs, size or components and reuse it.

An explicit missing primary selection can use a trusted catalogue binding.
Existing preparation transfers and validates that primary before final native
family/variant capture and required companions. A missing arbitrary custom path
stays blocked. Settings, performance presets and resolution values are frozen
while the request waits. Navigation and image-tool Preview acquire no files.
Python packages and runtime installation remain outside image jobs.

Submission snapshots also retain the configured model roots, trusted catalogue,
installed-name/hash inventory and metadata-cache roots. Chat retains that passive
snapshot before its initial queue wait; a later validated action uses the same
owners. The native loader receives the final plan's checkpoint path and captured
LoRA sources. Inline adapters introduced only by later dynamic prompt expansion
must match that captured adapter intent before loading.

Library replay uses the existing metadata normalizer after default qualification
and before component, geometry and final-plan capture. Recorded sampling values
therefore agree with execution. API and server-bound tool prompts remain literal.

## Owned editor results

`editor_context` contains only `session_id`, `revision`, `source_id`, optional
`mask_id`, and bounded additional image IDs. The server chooses the owner and
resolves current pixels from those artifacts. An accompanying source upload
cannot replace the handle's image. An editor inpaint operation requires an owned
mask; the typed mask adapter owns exact frame, grayscale selection and inversion.

Before preparation and compute, the worker validates the editor ticket and final
plan. After writing a successful artifact it stages scalar publication evidence.
The existing terminal boundary checks cancellation and commits the history step
under the existing registry lock. Failed, canceled and stale work does not append
a completed step. Pure processing retains its own execution metadata separately
from inherited source generation context.

`async_worker.editor_result(task_id)` returns an owner-protected bounded result:

```json
{ "status": "completed", "snapshot": { "session_id": "…", "revision": 2 } }
```

Other states are `cancelled`, `failed` and `stale`, with no completed snapshot.
The existing terminal callback reads this status once. UI publication also checks
its current source/configuration fence. Snapshots expose opaque artifact IDs and
redacted provenance; manifests retain exact internal identities.

## Chat preparation and continuation

The app captures `ImageToolContext` with explicit Text, Preview or Run intent.
The existing Force image checkbox maps to Run at worker admission only when no
new explicit intent is supplied. Enabling optional tools alone does not upgrade
a text turn. Advertisement uses the initialized model/template capability and
current selected image resources. Metadata is not a visual observation.

Malformed, truncated, parallel, unknown and incompatible calls are rejected
before image capture/preparation. A validated `ImageToolContinuation` leaves the
outer Chat compute scope. Existing `PreparationAdmission` prepares its files with
local cancellation and requeues the same owner/job. Image execution uses the
normal image handoff and restores the captured Chat model. There is at most one
tool action and mutation per turn. Pure rendering precedes the atomic history
commit, tool-result event and existing terminal result.

`api_runtime.prepare_chat_image_tool` and `execute_chat_image_tool` are the
internal service boundary. Preview reports passive readiness; Run executes once.
`inspect_image` needs an initialized compatible vision model and reads actual
selected pixels. No automatic critique/regenerate loop is added.

## Authenticated HTTP routes

Ordinary `/v1/chat/completions` tools remain client-managed transport. RF does
not execute a tool merely because that endpoint returns its call.
`POST /v1/rf/image-tools` is the explicit server-executed route. It requires an
initialized compatible local Chat model/template and accepts a strict call,
app intent, optional editor context, response format and streaming flag:

```json
{
  "call": {
    "id": "call_1",
    "type": "function",
    "function": { "name": "generate_image", "arguments": "{\"prompt\":\"Two blue cups\"}" }
  },
  "intent": "preview"
}
```

Owner, settings, filesystem paths, graphs and installer options are not public
authority fields. The API captures current image defaults. JSON and SSE return
one typed terminal with the original call ID; internal image paths become the
existing expiring `/v1/images/…` URLs or requested base64 data. Preview includes
readiness and missing requirements. Responses are private and not cached.

The API dispatches only validated, named image-tool Preview requests and the
literal `propose_create_prompt` copy through the existing bounded
`PreparationAdmission` pool (two workers, eight jobs). They retain the API Job
capacity, registry task ID, authenticated owner, cancellation, permission/runtime
checks and private terminal channel, without waiting for the serial compute
worker. Preview captures local readiness without preparing files or invoking
inference, and the prompt-copy handler returns the validated supplied prompt;
neither enters native service transitions or cache cleanup. Actual Chat completion,
model-backed prompt enhancement and image-tool Run retain their existing serial
compute admission and leases. This API-only separation does not qualify browser
responsiveness.

`POST /v1/rf/editor/sessions` adopts the existing bounded inline PNG/JPEG/WebP
transport as an immutable original. `GET /v1/rf/editor/sessions/{id}` reads its
current public snapshot; `POST …/{id}/undo` and `POST …/{id}/redo` accept the
current revision. The existing bearer key defines one API owner across restarts;
an unkeyed local API owner lasts for that app instance. UI handles belong to their
separate authenticated browser owner. Foreign handles fail with 403; stale
revisions fail with 409. Image generation and upscale requests can pass
`editor_context` instead of an inline image and return the committed editor
snapshot alongside the existing image transport.

## Verification

Focused CPU callers:

```powershell
python tools/check_cpu.py test_image_plan_callers.py test_chat_image_service.py test_chat_image_tool_continuation.py test_api_image_edits.py test_api_live_contracts.py
python tools/check_cpu.py test_pending_image_callers.py test_api_native_inpaint.py test_utility_sampling_capture.py
python tools/check_cpu.py test_api_preview_dispatch.py test_chat_image_service.py test_chat_image_tool_continuation.py test_api_activity.py test_api_live_contracts.py
```

Fixtures exercise real request capture, strict model-output DTOs, preparation
continuations, editor artifacts, cancellation and HTTP serialization with inert
compute. Native Gradio event checks also cover the production Chat/workspace
join, source and draft fencing, cancellation, history and deliberate Open/Edit.
Passive API dispatch checks hold an inert compute job with an Event barrier and
complete Preview and a literal prompt copy before releasing it, while a second
compute job stays queued. They also cover queued/running cancellation, error and
capacity release, runtime/request isolation and revoked prompt-copy permission.
Independent source reviews and declared prerequisite closures passed. Counts
from focused runs and closure replays overlap and must not be added together.

The bounded 2026-11-12 GPU check invoked BiRefNet twice and Janku editing once.
The first cutout reached CUDA compute but failed before publication; the retry
saved an 832×1216 RGBA cutout after the utility metadata repair. The app mask-output
binding repair then allowed validated history continuation. The edit saved a
1024×1024 RGB result using explicitly selected model-auto sizing and padding
(701×1024 content), separately from ordinary preserve-geometry CPU checks.
Effective sampling used the labelled Illustrious family fallback, rather than
a complete checkpoint recommendation. The subject remained recognizable; the
requested blue accent was not clearly evident, so full prompt adherence is unproven.

Upscale was not invoked before the power deadline. Browser rendering, mobile
layout and actual 200%/400% zoom remain blocked by browser access. Preview transport
failures came from the owned private-profile fixture, without a product permission
change. These observations do not qualify general image quality, performance or
GPU stability. Detailed source, model, artifact and cleanup identities remain in
the external acceptance records; all owned application processes were stopped.
The next rendered check is one fresh built-in-browser session against the regular
online launcher, followed by desktop/mobile and actual zoom checks if access works.
