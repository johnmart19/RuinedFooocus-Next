# Application map

Start with `launch.py` (direct launch) or `entry_with_update.py` (dependency/update
bootstrap). The Windows and Linux launchers select Python and call the latter.
Application code lives under `ruinedfooocus`; downloaded backends live under
`repositories` and are not application source.

| Feature                                      | Source under `ruinedfooocus/`                           | UI entry                                    | Backend                                      |
| -------------------------------------------- | ------------------------------------------------------- | ------------------------------------------- | -------------------------------------------- |
| Generation, batches, automatic settings      | `features/generation`, `features/models/performance.py` | Create → Generation                         | ComfyUI sampling and model loading           |
| Image editing, masks, ControlNet, geometry   | `features/image_editing`                                | Editor / Editor → More tools                | Native ComfyUI edit/ControlNet nodes         |
| Upscale, background removal, captioning      | `features/image_processing`                             | Editor / image upload                       | ComfyUI upscale nodes; ONNX / caption models |
| Models, LoRAs, components, checkpoint copies | `features/models`                                       | Models                                      | ComfyUI; local safetensors inspection        |
| Prompt transforms, styles, rewrite presets   | `features/prompting`                                    | Create → Prompt tools / Generation → Styles | Shared local chat; ComfyUI conditioning      |
| Random prompts and datasets                  | `features/prompting/one_button`                         | Prompt tools                                | One Button Prompt                            |
| Local chat, assistants, image tools, vision  | `features/chat`                                         | Chat                                        | Shared llama runtime and image worker        |
| Gallery, search, metadata replay             | `features/gallery`                                      | Gallery / Create → Details                  | SQLite, local images                         |
| Local commands and search routing            | `features/automation`                                   | Prompt commands                             | Existing feature services                    |
| HTTP/API access                              | `integrations`, `app/api*.py`                           | Startup options                             | FastAPI, shared serial worker                |
| Runtime selection and installation           | `packaging`, `backends`                                 | Launchers / Settings                        | CUDA, ROCm, DirectML, CPU, llama, ONNX       |

`app` composes services and owns the serial job worker. `core` owns paths,
configuration, process handling and common utilities. `ui` owns the shell, shared
controls and assets. Feature-specific UI stays beside its feature. `backends/comfy`
contains adaptations to the pinned ComfyUI revision in `app/bootstrap.py`; do not
copy native model/sampler implementations into feature folders.

`features/models/native_snapshot.py` reads bundled immutable native family,
recipe and sampler facts; `tools/update_native_snapshot.py` explicitly rebuilds
them from accepted sources. `features/models/effective_image_plan.py` freezes
operation, components, recipe, size, projector and per-prompt LoRA intent once
for API/worker callers. Native loaders and the pipeline retain execution ownership.

`features/generation/image_preparation.py` captures required asset requests and
passive readiness. It stages an eligible trusted primary before final plan and
companion capture, reusing `core/preparation.py` for transfers, leases and
cancellation. The plan owner's trusted finalizer binds newly verified projector/LoRA packets
without replacing the captured component, recipe or size intent.

`features/gallery/editor_session.py` owns immutable original/current artifact
references and bounded editor history. `features/image_editing/mask_handoffs.py`
owns selection-mask purpose, frame and alpha handoff;
`features/image_editing/image_workflows.py` binds finite scalar workflow steps to
existing native feature services. `app` retains worker/API execution and terminal
publication ownership.

`features/gallery/ui_workspace.py` owns the shared Create/Editor workspace
binding over those editor services. `ui/app.py` constructs one workspace and
passes it to `features/chat/ui.py`. Native callbacks retain source, result and
draft fencing before updating the same context and display. Chat exposes explicit
Text/Preview/Run intent and New/Previous/Upload sources; completed Chat images
and Gallery images reach Editor through deliberate Open/Edit. Ordinary generation
setup and prompt replay prepare Create; source-bound image setup prepares Editor.
Models owns the full existing manager, while Create retains the same quick model picker.

`features/chat/image_tool_requests.py` owns bounded tool requests and the frozen
Text/Preview/Run context. `features/chat/llama_pipeline.py` yields a typed image
continuation; `app/async_worker.py` leaves outer Chat compute before preparation,
then resumes the same job through existing image services and terminal publication.
`app/api_runtime.py` and `integrations/openai_api.py` share these capture and result
contracts. They retain worker/API ownership rather than introducing another executor.

`features/chat/ui_moe_cache.py` owns the saved MoE preference controls and passive
form/saved/initialized status. It reuses `core/profile_store.py` for fresh-page
restoration and `backends/llama/moe_cache.py` for capability/precedence policy;
refreshing the status does not construct a Chat runtime.

The native image pin is `989d93d0d968e704c589ce0146bb18c77f6b50cc`.
The [image and optional Chat boundary](integrations/image-chat-scope.md) owns
product scope; native registrations do not automatically become RF features.
`backends/comfy/comfy_compat.py` translates finalized launch/profile choices,
reuses native allocator and DynamicVRAM setup, and preserves native loading,
attention, precision and offload decisions. `cpu_startup.py` supplies the bounded
probe-free CPU import needed before worker threads; temporary discovery changes
are restored before independent services start. CPU import evidence is separate
from model output and hardware acceptance.

`features/generation/image_pipeline.py` owns checkpoint/LoRA caches and request
recovery. Its optional `component_overrides` mapping is request-local: absent keys
retain legacy profile choices, empty/null keys select the family recipe, and
bundled encoder/VAE slots cannot be replaced. Shared pure resolution lives in
`backends/comfy/component_selection.py`, also consumed by component inventory and
downloads. Cache identity includes explicit choices and external file signatures.
The generation worker/preload callers and metadata replay must carry the same
snapshot when wiring per-checkpoint UI preferences. See the
[native caller contract](integrations/comfy-platforms.md#external-component-caller-contract)
for APIs, upstream delta and validation limits.

`features/image_editing/lora_hooks.py` owns optional scheduled LoRA capture and
native hook scope. Its request clones and conditioning cache are restored after
success, cancellation or failure; static LoRA caches remain owned by the pipeline.
See [native LoRA hooks](integrations/native-lora-hooks.md).

`core/settings.py` stores independent profiles and accepts an optional change callback;
it does not import app state or parse launcher arguments. Generation startup defaults
live in `features/generation/defaults.py`; the app composes general preferences.
`core/profile_store.read_profile_data` recovers paired saves and reads a JSON object
without feature validation. The updater uses it only for its two strictly typed
startup booleans, before application dependencies are ready. Application settings
continue through `read_profile` and complete known-field validation. Backend
selection uses the same pure saved-profile path selector, then full validation
after application arguments are initialized; it does not read the updater's
boolean-only policy as a complete profile.

## Resources and user data

`core/paths.py` owns root resolution. Resource discovery does not depend on the
working directory. `resources` and `ui/assets` are bundled inputs. Writable data
defaults to the clone root; `RF_DATA_DIR` selects another data directory for an
extracted application. Existing absolute model paths remain valid. Backend source
and Python environments are installed separately from writable user data.
Legacy `../models/...` path values retain their former `modules/` base semantics.

| Bundled source                       | User override                                                              |
| ------------------------------------ | -------------------------------------------------------------------------- |
| `resources/prompt_presets/*.txt`     | `prompt_presets/`; legacy `llamas/` remains readable                       |
| `resources/assistants/`              | `assistants/`; legacy `chatbots/` remains readable                         |
| `resources/defaults/`                | `settings/` and named settings profiles                                    |
| One Button bundled datasets/presets  | Existing `random_prompt/userfiles/`                                        |
| Model download catalogue             | `settings/model_settings.json` for recommendations                         |
| `resources/generation_presets/*.png` | Configured preset folder; legacy clone `presets/` remains readable         |
| `resources/wildcards/`               | Configured wildcard folders; legacy `wildcards_official/` remains readable |

New user folders override legacy folders, which override bundled entries with the
same relative identity. Discovery never renames, copies or overwrites custom
presets/assistants. Rewrite presets are instructions, assistants are personas,
and GGUF files are model weights; all LLM features share the selected chat runtime.

Keep weights, generated images, caches and environments out of source commits.
Existing `models`, `outputs`, custom wildcards, generation presets and model caches
must not be moved by a source upgrade. Imported settings and PNG metadata retain
their public keys; folder names are internal implementation details.

## Job and compute ownership

`app/resource_lifecycle.py` owns immutable job identities, bounded private result
channels, cancellation events and fair resource admission. The real UI worker and
API callbacks use these contracts; an event's Gradio queue does not grant compute
ownership. Main Stop carries its browser's explicit generation task ID. Polling
or canceling another session's ID is refused, and a missing request context cannot
stop an arbitrary job. Authenticated logout cancels only jobs from that server login.
API disconnect/timeout ends that job's lease; an active operation retains its slot
until it unwinds. Streamed chunks have bounded backpressure and a reserved terminal
position, preserving successful output. Completion releases capacity once.

`RuntimeState` keeps previews, interruption and last-image state in the owning job
or browser view. Both worker and browser access retain at most 256 owner views
under one lock; eviction drops only the view reference, preserving a live job's
own state. Settings, registry and service caches remain process-wide.
Effective sampling recipes, component choices and optional-engine policy are
captured before submission waits. Remove background and input-only/standalone
Upscale capture their utility assets without resolving a checkpoint sampling
recipe or components. `_sampling_captured` distinguishes a frozen sampling recipe
(`True`), an intentionally captured utility (`False`) and an unmarked legacy
caller. Retargeting a captured utility to sampling requires explicit submission
capture again; manual-only model families retain strict recipe checks. Control
classification uses the same imported metadata precedence at capture and worker
execution, while API and tool-call JSON prompts remain literal. Component capture
follows that same boundary: JSON prompt content cannot replace an API/tool request's
selected checkpoint or explicit component choices; UI metadata replay remains supported.
Chat model/projector, context and optional-image
preparation use bounded `preparation_admission` continuations outside compute.
Canceling a queued preparation returns capacity even when its callback never ran.
The existing preparation service still owns file transfer and publication.

Native image submission also captures ControlNet, standalone/input-only/automatic
upscalers, external text-encoder/VAE roles and the selected depth preprocessor in
`features/generation/image_preparation.py`. Missing catalogue files use that same
preparation lane before compute; installed files keep a local-only fast path.
Ordered roots, destinations, effective control options, Automatic component
choices and offline intent are captured together. Native loaders hold read leases
for the exact prepared file and reject changed or unplanned dependencies instead
of starting a transfer under compute. Offline, cancellation and transfer failures
terminate preparation without admitting an incomplete request.

Depth Auto reads only the already initialized native `cpu_state` and cached
`total_vram` capacity: CPU selects Small; other initialized devices with at least
8192 MiB select Base. An absent/invalid capacity snapshot requires an explicit
Small/Base choice. Supplied depth maps do not prepare DA3. Header-plan support is
limited to known safetensors/image GGUF architectures; bundled and pixel-space
roles need no external file. Missing custom overrides and late native/header
disagreements fail explicitly, without substituting another file or downloading
from the native loader. Pickle companion discovery is not inferred from weights.
The existing serial compute and shared atomic publication policy remain in force.

Source histogram analysis is owned by `features/image_processing/image_histogram.py`
and `ui_image_histogram.py`. It freezes pixels before serial admission and returns
typed histogram data through its own UI action and `/v1/images/histogram`, with no
generation-model dependency. See [native histograms](integrations/native-image-histogram.md).

`features/image_processing/still_image_utilities.py` dispatches still operations
to their feature helpers. Model-free filters live in `image_filters.py` and
`ui_image_filters.py`; the serial utility pipeline, preparation, API, replay and
metadata writer retain the shared output contract. See
[native filters](integrations/native-image-filters.md).

`image_shaders.py` and `ui_image_shaders.py` add bounded source curves and named
shader recipes to that dispatcher. Native `CurveEditor`/`CurveInput` own LUT
interpolation; native `GLSLShader` or the finite CPU consumer applies those LUTs.
`image_effect_cpu.py` reuses native inversion/blending and bounded CPU sampling
or recipe arithmetic. CPU policy is authoritative; absent/failed acceleration
falls back with a terminal warning. Passive UI/API capture and CPU execution are
GL-free. The existing worker owns artifacts and replay. See
[curves/effects](integrations/native-image-curves-effects.md) for the unqualified
platform/GL boundary and traced CPU evidence limits.

Standalone Lotus/MoGe depth, MediaPipe facial overlays, SDPose maps and Marigold
albedo are owned by `features/image_processing/depth_utilities.py` and its
operation-specific native adapters. A typed request freezes RGBA source bytes,
dimensions, scalar options and exact component signatures. The existing image
preparation service, serial worker, API and output writer carry these utilities
without a generation checkpoint or prompt/sampling recipe. Native patchers own
device placement; the utility cache owns their lifetimes. Library replay restores
the named operation and scalar values, clearing source uploads and component
selections. See [still geometry utilities](integrations/native-still-geometry-utilities.md).

All production heavy stages reserve `compute:shared`. They remain serialized,
including independent devices, until physical isolation and memory policy are
qualified. The admission primitive supports independent fake resource lanes and
FIFO contention tests; those tests do not enable parallel native inference.
Preparation, ordinary Library/configuration work and cached status do not acquire
this compute resource. Resource-specific Settings verification is an explicit job.

| Caller family                                           | Current coordination                                                                             |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Chat load/reply/vision/unload and API Chat/review       | Chat service cache, captured artifacts, same compute owner                                       |
| Create/edit/inpaint/ControlNet/image batches            | Native image service, captured recipe/components, native cancellation only for its active owner  |
| Upscale/caption                                         | Native allocation service transitions, shared compute admission                                  |
| Background removal                                      | ONNX prediction requires admitted ownership, active sessions cannot be closed by a transition    |
| Still image grids / masked and Porter–Duff composition  | Immutable ordered pixel capture; pure pinned native bodies on CPU; same serialized image lane    |
| Chat image tools/review                                 | Inline reentrant stages; no second queued job behind itself                                      |
| Enhancement/rewrite/Hyperprompt                         | Captured helper preparation before compute, reentrant Chat stage and owned temporary model       |
| RAG/document indexing                                   | Prepared URL inputs, Chat compute stage; complete index is published only after successful build |
| Downloads/model scans/import/inspection/conversion      | Existing file/publication leases; no new parallel compute permission                             |
| Library/search/replay, random prompts, wildcards/styles | Existing CPU/database/UI contracts; tokenizer use remains inside generation ownership            |
| Settings/profile/localization, backend install/build    | Existing atomic save/install locks; captured requests retain their selected policy               |
| Provider discovery/qualification, API/guest             | Explicit admitted operations, authenticated private channels, bounded queues                     |

Service transitions release only the inactive owning cache. Native Comfy model
and allocator cleanup occurs when crossing that service boundary, rather than
after every task. Cached ONNX calls carry an active-use fence. Unmanaged GPU
prediction/cleanup is refused instead of evicting another service's allocations.
Runtime failures discard the failed cache; later submissions can load afresh.
Worker failure terminates stranded ready jobs and later work can restart the worker.

For Chat → image → review, conversation text and captured settings survive while
the owned Chat model/KV/projector are released before image execution. A successful
or ordinary unsupported request restores Chat; cancellation or a runtime failure
defers reload until a later Chat request. Review helpers follow the same rule.
Selected and inline local-model styles and One Button helper aliases capture the
text helper's model/settings before queueing, without expanding prompts. Standalone
rewrites use the same preparation lane. An uncaptured nested helper refuses before
loading rather than acquiring missing weights while holding compute admission.
The final Chat reference is rebound to its captured image request without rereading
engine/device policy or acquiring artifacts during that handoff. This is a source
contract; physical 3 GB/6 GB residency and allocation peaks remain unverified.

Native device masks are resolved before initialization by `comfy_compat.py` with
backend-specific CUDA/HIP precedence. Inherited/requested/effective values are
retained, conflicts are explained and initialized runtime changes require restart.
Physical counting uses a bounded isolated child environment, never a temporary
mutation of the live process masks. Independent llama/SDCPP/ONNX children undo
only a proven application startup override in their own environment copy; inherited
user masks remain authoritative and later unrecognized changes are refused.

Focused CPU evidence lives in `test_resource_lifecycle.py`, worker/Chat/API caller
tests, `test_onnx_runtime.py`, `test_comfy_runtime.py` and the SDCPP caller tests.
Barriers and fake devices cover isolated results/Stop/logout, fairness, failed
load, preparation/stream capacity, active-session cleanup and compound recovery.
They do not certify GPU overlap, real-model stability or optional-engine receipts.

## File preparation ownership

`core/preparation.py` owns transfer, verification and final file publication.
Its process-local `preparation_service` defaults to two I/O workers and eight
outstanding destination jobs. It never imports a compute pipeline, admits GPU
work or unloads models. `core/path.py` and `core/hub_download.py` route existing
synchronous download entry points through it. Chat/API model and projector,
context, optional-image runtime and selected prompt-helper callers prepare before
compute admission as described above. Other native pipeline first-use consumers
retain their existing synchronous loading contracts.

`PreparationRequest(url, destination, expected_size=None, sha256=None)` captures
an absolute canonical destination and immutable transfer identity. Equal requests
for that destination share one job, including catalog aliases; an incompatible
request raises `PreparationConflict`. Capacity raises `PreparationBusy` rather
than growing an unbounded queue. `submit(request, progress)` returns a subscriber
with `request`, `result(timeout)`, `done()` and `cancel()`. `PathManager.prepare_file`
captures the catalog/root and returns that handle; `preparation_request` exposes
the snapshot for callers composing preparation requests. `download_file` and
`download_hub_file` block until preparation finishes for compatibility.

Progress callbacks receive `(received_bytes, total_bytes, bytes_per_second)` on
the I/O worker in a separate context captured from each subscriber at submission,
including Gradio request/event context, and must return promptly. Callback failure cannot fail other
subscribers. Cancellation detaches only that subscriber, suppresses subsequent
callbacks and makes its result raise `concurrent.futures.CancelledError`. The
last cancellation requests cooperative transport cancellation; network reads
may finish at their next timeout/progress boundary. A cancelling destination
rejects new subscriptions until its worker exits, after which it can be retried.
Completion returns the file path only; activation, UI selections, settings and
request/session fencing belong to callers.

The service checks nonempty files, HTTP response length when supplied, and the
request's optional expected size/SHA256. These checks do not certify a model's
format or semantics when no trusted digest was provided. Existing final files
are verified and reused; verification failure never replaces user data.
New SDK staging lives in the configured cache root's `preparation/` directory
under a destination/URL
hash, outside the model directories; this application cache must not be registered
as a model root. A service may receive an explicit external `staging_root` for
isolated callers/tests. Ordinary HTTPS Hub files with a pinned commit, exact size,
and trusted SHA256 can use an application-owned partial file and receipt. The
transport checks the commit, size, and strong ETag before resuming, validates
partial responses, and starts from zero when the server returns the whole file.
The installed `huggingface_hub.hf_hub_download(local_dir=..., tqdm_class=...)`
API continues to handle other Hub sources and Xet. Cross-request resume for those
transfers depends on the installed SDK. Rejected or already-published complete
artifacts are removed.

The [cache contract](integrations/cache-contract.md) classifies legacy data-root
staging and protected inputs/runtimes; no existing resume state is moved.
The [concurrency guide](integrations/concurrency.md) describes the shared bounded
preparation pool, incremental Library index and per-session UI request fences.
Raw SDK filenames never enter the destination's model directory before
verification. Final publication follows verification and cannot overwrite
a competing final file: Windows uses atomic `os.rename`, POSIX uses `os.link`
then unlinks staging. If external staging is on another volume, preparation copies
the verified artifact into an exclusive random `.part` at the destination,
flushes and verifies that copy, then publishes it on the destination filesystem.
Copy failure/cancellation removes the destination `.part`; scanners cannot select
it as a model. No vendor scanner patches are needed.

`artifact_lease(destination, write=False, cancelled=None)` protects a captured
artifact while a cooperative loader uses it. Import/delete/replace operations
take `write=True`; preparation holds that exclusive lease through transfer and
publication. Different destinations can proceed independently. Acquire leases
outside compute admission and never wait for preparation of a destination while
holding its lease. All cooperating consumers must use the same service instance.
These leases do not coordinate other processes or existing callers that have
not adopted them; atomic publication still preserves a noncooperating writer's
final file. Runtime loading, installer transactions and registry publication
retain their own ownership until their integration phases.

Focused fake-transport checks are `test_preparation_service.py`,
`test_hub_download.py`, `test_resource_paths.py`, `test_model_folder_cleanup.py`
and `test_validation_asset_download.py`, run through `tools/check_cpu.py`.
They cover shared subscribers, independent targets, scoped cancellation,
transfer/disk/hash failures, captured settings paths, leases, SDK resume staging
and competing publication, cross-volume copy failure, per-subscriber Gradio
context, and staged GGUF/safetensors invisibility to RF Chat/Models/embedding and
native ComfyUI discovery. They do not establish real-network, Chat responsiveness
or model/runtime acceptance.

## Validation and distribution

Developer formatting tools are pinned separately from application dependencies:

```sh
python -m pip install -r tools/format/requirements.txt
npm ci --prefix tools/format
python tools/format_code.py --check
```

Dependency updates must preserve the selected Torch bundle and platform markers.
Current Gradio requires Pillow below 13, MCP below 2 and Pydantic at most 2.12.5;
use compatible versions until its integration supports newer APIs.

Omit `--check` to format. Only tracked files are considered; vendor sources,
tokenizers, prompt text, datasets and user files are excluded. Formatting verifies
that executable Python ASTs remain identical. Ruff also checks syntax and undefined
names. Markdown, JSON, CSS, HTML and standalone JavaScript use Prettier.

Use this checkout's Python environment and pinned `repositories/ComfyUI`.
`python tools/check_cpu.py test_image_input_transport.py` runs a focused guarded CPU check with private data; real generation,
visual quality and GPU stability require separate evidence. Register feature
commits through `tools/feature_commits.json`: each commit adds its own
`tools/features/<id>.json` descriptor with its unique subject, actual prerequisites
and focused CPU tests. Separate descriptors avoid conflicts between unrelated
feature registrations. Validate their declared prerequisites with
`tools/validate_feature_cherry_picks.py --feature <id> --tests`.
Checks cover local imports and constant bundled resource paths. CPU tests use
isolated writable data and exclude video modules before discovery. This verifies
service prerequisites; the `ui` feature supplies the complete application shell.
Panel registrations (`ui-image-tools`, `ui-prompt-tools`, `ui-models`, `ui-chat`,
`ui-library`, `ui-settings`) precede shell composition. `launchers` activates the
application, while source archives and legacy retirement have separate commits.
See [legacy source retirement](legacy-migration.md) for replacement owners.

Portable releases must build separately for Windows/Linux and CUDA/ROCm. Bundle
an appropriate Python/runtime rather than copying a development venv. Drivers
remain system prerequisites. Preserve writable data on upgrade, keep model
downloads explicit, and bypass Git update behavior for archive builds. A source
or CPU smoke check does not certify a portable GPU distribution.

Build a source ZIP from a committed revision with:

```sh
python tools/build_source_archive.py --output dist/RuinedFooocus-source.zip
```

`--ref` selects another source revision. The ZIP excludes user data, environments
and weights, and records the source revision and backend pins in `distribution.json`.
It uses the existing launchers, with explicit model downloads in Models. Git updates
are disabled without a root `.git`; source updates use a safe fast-forward and
preserve local edits and ignored files. Reinstall requests and dependency, gallery,
llama and ONNX caches follow `RF_DATA_DIR`. Existing source files must remain
writable for online backend installation; a prebuilt portable runtime is separate.
Pinned native backend updates preserve dirty checkouts and report that the update
was skipped. Clean checkouts use Git's safe conflict checks; concurrent edits are
never discarded with a forced checkout.

Task controls are saved with **Create → Save current defaults**; Settings contains persistent app preferences and advanced defaults. Prompt rewrite previews before applying. Model preloading is optional; browsing alone does not require weights to be loaded.

The maintained [documentation index](README.md) links user workflows, Settings, prompt data, localization and [development checks](development.md). Catalogues live in `resources/translations`; native Gradio owns static frontend localization. Settings paired-file recovery and startup profile selection live in `core/profile_store.py`; native compute validation remains a lightweight shared schema with backend application in `backends/comfy/image_compute.py`.
