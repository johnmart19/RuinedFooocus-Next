# First-use image files

`features/generation/image_preparation.py` captures selected files and uses the
existing `PreparationService`, `PreparationAdmission`, signature caches and
artifact leases. It adds no downloader, executor, native loader or package
installer. Navigation, model choices, readiness and operation previews acquire
no model files. Preparation runs only for an explicitly submitted image action,
before the serial compute lane. Runtime device selection remains unchanged.

## Captured requirements

An installed primary uses P1's `EffectiveImagePlan.required_components`: only
active, required, external roles become preparation requests. Embedded roles,
inactive alternatives and already captured approximate codec pairs are omitted.
The existing companion inventory remains the fallback for older internal
callers. Both paths retain explicit custom files and the selected native variant;
they never choose a smaller encoding or turn on a Chat/vision/RAG helper.
Pending capture also respects existing local logo, automation and search routes;
API, tool-call and replay prompts retain their literal image-request semantics.
After a missing GGUF encoder is acquired, the existing projector owner captures
its implicit local sidecar or absence. P1's trusted preparation finalizer binds
only that new packet to the same plan; all other intent and existing packet rows
remain unchanged, and component/recipe/size references are retained.
Selected active trusted LoRAs join the same required-role preparation. Their full
captured URL, destination, size and digest tuple must still match before staging;
the trusted finalizer binds verified file signatures. Inactive LoRAs acquire no
files, and missing custom files need local installation.
Prompt batches share transfers for the same selected file while retaining each
original prompt's captured LoRA names and weights, including an empty selection.

`image_readiness(captured_data)` returns frozen `ImageReadiness` and
`ImageRequirement` records. States are `ready`, `preparable` and `blocked`;
requirements report installed/missing/blocked files, source, expected bytes,
integrity evidence and recovery reasons. This is a passive stat/free-space check:
it does not hash weights, fetch metadata or initialize a tensor/device runtime.
SHA256 supplied, size-only and unknown evidence stay distinct. Download bytes
are not measured RAM/VRAM or model/device qualification.
An appeared file still awaiting its captured verification remains `preparable`;
concurrent local removal is reported as a typed blocked requirement.

Capture can freeze `reserve_gib` and `allow_unknown_size`. Defaults preserve the
existing first-use policy: no extra disk reserve and unknown sizes permitted
with explicit unknown status. A caller with a stricter policy passes it at
capture; later settings changes cannot relax that queued request. Preparation
checks the full set and cumulative destination/cache space before subscribing.
Offline state, invalid sources, stale local files, conflicting source tuples and
disallowed unknown sizes block the set before any transfer starts. A removed
captured local file requires a new selection and never turns into a download.

`PathManager.preparation_request` first binds registry annotations to the unchanged
catalogue tuple and checks size/digest conflicts. A distinct registry publisher
pin is used only with positive byte size and SHA-256, HTTPS on Hugging Face, the
same owner/repository and complete repository-relative file path, and an immutable
40-character revision. Malformed or mismatched pins fail before submission.
Absent or unchanged pins retain the existing request contract. This keeps
historical catalogue identities intact while production first-use requests use
the reviewed immutable source; no private acquisition driver is required.

At submission, each active required registered plan component freezes that actual
`PreparationRequest`: URL, destination, expected bytes and SHA-256. Its canonical
catalogue identity and display/source URL remain separate. Passive component
inventory does not bind preparation requests. Before preparation, the current
canonical identity and complete producer request must both match the frozen
values, and the broker receives the captured request object. A reviewed immutable
transport pin therefore does not conflict with a catalogue URL that uses `main`;
later source, destination, size or digest changes still require resubmission.
Plan metadata serializes the captured destination as a string for PNG/API exports.

An explicitly selected local encoder or codec instead freezes a local-only
request: empty URL, captured local destination and no publisher byte size or
checksum. The file's captured signature remains mandatory. A catalogue URL may
remain as provenance for a same-named resource; it does not authorize a transfer
or impose that publisher artifact's integrity on the user's compatible local
choice. Local changes or removal require resubmission, and a missing local file
cannot become a download or silently accept a file that appeared after capture.
Installed local choices work offline without changing the selected encoder/codec.

## Missing primary: two stages

`capture_pending_image_selection(data, paths, ...)` recognizes only an explicitly
selected trusted existing catalogue artifact/profile with the image-checkpoint
role and a connected family/loader contract. The selection may be its exact key,
an unambiguous registered filename or its configured destination. It never
infers a family from a filename or acquires an arbitrary missing custom path.
Installed uncurated files retain normal native header admission.

The frozen `PendingImageSelection` carries source tuple, artifact/profile IDs,
expected native class/family/variant and the primary `PreparationRequest`.
Public request intent and images are copied. This is a pending source selection;
it does not claim that the absent bytes have been recognized as executable.

The worker/API integration follows this order under its existing preparation job:

1. Capture pending selection before installed-primary lookup, defaults or P1.
2. Call `prepare_pending_image_selection` outside compute. Source contracts are
   checked before and after transfer. Existing service SHA receipts seed the
   current source cache without a second full-file hash.
3. Validate the actual installed native header/class/family. Contradictory
   variant declarations fail; an undeclared variant needs current verified
   exact-source provenance. Native source predicates handle safetensors/SFT
   without starting a runtime. The same native predicate owner accepts logical
   GGUF headers through its explicit `include_gguf=True` option; Comfy/Torch imports
   are blocked in the cold caller fixture. Initialized native metadata readers
   retain their existing path. No detector or reader is duplicated.
4. Recapture the installed request through the existing defaults, component,
   recipe, size and P1 owners. Preserve manual request values, the prepared
   primary marker, source warnings and captured preparation policy.
5. Prepare only that final plan's active required companions, then enter compute.

The returned absolute primary path and signature are retained by final asset
capture. The existing native component read scope leases it while loading;
Cascade keeps its established stage read owner. Calling companion preparation
directly on an unresolved pending selection fails before acquisition. UI action
exposure belongs to the image workspace integration; selection itself is passive.

## Native tools

Native detection/SAM/DA3, SeedVR2 and depth/pose capture use the same
`selected_image_artifact` boundary. A missing file must resolve to one exact
compatible registered tuple inside its configured category roots. Explicit
source access gates block acquisition. Arbitrary URLs, traversal, ambiguous
basenames, wrong roles and unregistered missing custom files are refused.

Existing files are validated at capture. Missing selected files defer header and
task-specific source checks until preparation, under file read leases and before
compute. Depth utility requests then freeze their acquired signatures/provenance.
SAM text still requires the selected checkpoint's paired language encoder; it
does not download a separate generation encoder. Marigold's strict Edit2509,
LoRA, VAE and conditioning proof remains local/source verified. Roles without a
reviewed acquisition binding retain the explicit local-install recovery path.

## Transfer ownership and recovery

The process-local service deduplicates canonical destinations, supports separate
subscribers and owns bounded transport, staging, digest verification and atomic
no-clobber publication. Repeated roles in one request subscribe once. A canceled
subscriber cannot admit its request or cancel another subscriber; canceling the
last subscriber prevents late publication. A completed file needed by a surviving
subscriber may remain installed. Interrupted, failed or canceled work can be
retried explicitly with the same frozen source; no unbounded retry loop is added.

Successful SHA receipts are reused under matching file identity. Optional source
cache write failure retains the verified file and exposes an Inspect recovery
warning. Hash/source/header failure blocks compute and preserves prior files;
existing user files are never silently replaced. Python/native runtime packages
remain separate from model-file preparation and are never changed inside a job.

Accepted native source pins are ComfyUI
`989d93d0d968e704c589ce0146bb18c77f6b50cc` and ComfyUI-GGUF
`da5f435cd9a9e54ad27eed417ed72b1f2543ef41`, as recorded in bootstrap. Relevant
source owners are native predicates/metadata readers, P1's effective plan, the
packaged catalogue/registry and the existing preparation service. No external
library documentation or mutable publisher metadata lookup is needed here.

```sh
python tools/check_cpu.py test_first_use_image_assets.py test_image_preparation.py test_preparation_service.py test_preparation_queue.py test_preparation_verification_receipt.py test_native_utility_admission.py test_native_vision.py test_native_seedvr2.py test_depth_utilities.py
```

Fixtures cover exact tuple/role trust, missing-primary stages and intent, native
class/variant disagreement, P1 required/optional closure, tool finalization,
settings/root drift, offline/disk/unknown-size gates, subscriber dedup/cancel,
last-cancel publication, interrupted explicit retry, receipt reuse and failed
source caching. They use small private headers and local fixture transports;
the existing service suite also exercises owned loopback HTTP interruption.
These checks do not download real weights, install packages, probe GPUs or
qualify real-model execution, output quality, speed or device stability.
