# Model library presentation and component choices

The **Models** destination hosts the checkpoint browser, component choices,
checkpoint tools, combined LoRA/embedding browser and model acquisition view.
Create retains its quick model picker; both routes use the same selected model
control. The existing widgets and session state move together rather than
creating a second model selection. These panels use `features/models/ui_*` and the local
`ui/assets/adapter_cards.*` renderer. Publisher names are presentation; filenames,
registered tuples and file signatures remain action identities. Source and local
files stay available in details. Filtering does not change the active model.

## Component selection

The selected checkpoint's inventory supplies the visible roles. Installed
alternatives use `component_candidates.installed_candidates` with the inventory's
effective companion rows, including the mixed encoder reader constraint. A
filename, precision label or shared latent channel count does not prove
compatibility. Unknown files appear only through the explicit advanced option;
incompatible files cannot become alternatives. An existing missing, incompatible
or unverified choice stays visible until the user explicitly changes it.

The editor reports the effective file. The inventory distinguishes a local
bundle, embedded checkpoint components, checkpoint-specific saved choices,
profile defaults and family defaults. Embedded/bundled roles stay fixed. An
explicit Automatic reset selects the family recipe rather than inheriting an
old profile-wide override. Component details include the native reader,
tokenizer/layout contract where the resolver provides it. Saving remains scoped
to the exact checkpoint identity and takes effect on its next load.

No selection, filter or component refresh loads model tensors or downloads model
weights. Inspection and the existing useful FP16-copy tools remain separate
explicit actions; INT8 files do not acquire an invented conversion route.

## Cards and refresh

The custom renderer uses text nodes for labels, keeps card identities out of
visible names. Grid tracks use a 10rem minimum and share the remaining container
width, allowing two cards in a 384px panel at the default text size. Narrower
containers collapse to one column without horizontal overflow; the same rule
applies to the wider model catalogue and LoRA/embedding surfaces. Artwork updates reuse existing card/image
nodes and retain nested scroll and focus. LoRAs and embeddings retain their
combined available/active views, triggers, weights, automatic prompt target and
explicit Activate/Remove controls alongside drag/double-click shortcuts.
Available cards search names and cached metadata, matching the native gallery.
The focused resource remains marked while prompts or active resources update;
this marker does not activate it. Refreshing files clears the previous resource
details, weight control and actions until another card is selected. Per-embedding
target preferences and active LoRA weights remain in their existing session controls.
Action rows use the shared `rf-actions` container, with the catalogue refresh
using `rf-actions-compact`; search and catalogue filters use `rf-filters`.

Focused checks:

```sh
python tools/check_cpu.py test_component_overrides.py test_model_resource_ui.py test_adapter_browser.py test_resource_cards.py
```

CPU tests cover the resolver's UI callers, retained exact choices, visible native
roles, effective selection precedence and explicit adapter actions. Rendered
layout, live downloads, artwork arrival and real-model execution require their
separate browser/runtime receipts; source/CPU checks do not certify them.

## Acquisition actions and artwork

Callable installed pixel upscalers, native BiRefNet and DA3 expose **Use installed
file in Image tools**. The shared [Library/tool handoff](library-editing.md)
validates the actual file and updates its preset/type/resource together. The
action preserves prompt/source choices and waits for Generate. Unsupported or
ambiguous local files do not advertise Use.

`acquisition_actions` resolves installed component activation against the selected
checkpoint's current external roles and exact installed paths. Use revalidates
the checkpoint scope, local signature and native companion compatibility, then
returns session overrides. The app binding opens the owning Models panel without model
loading, downloading again, saving a profile or changing the prompt. Override
changes refresh the inventory without feeding its own state controls back into
the refresh event.

Chat exposes `select_catalogue_file` through its existing local catalogue and
`select_source` path. It resolves an exact installed path, preserves its family
and quantization, clears a conflicting model search and updates only candidate
controls. The root binding opens Chat without loading, changing the active
runtime, replacing conversation history or changing the message draft.

Successful SHA-verified preparation exposes an immutable local completion
identity captured under the artifact write lease. The acquisition session drains
that proof once into the [prepared source cache](prepared-native-source.md).
Current native defaults and background metadata reads reuse it without a second
model hash. Size-only, failed, cancelled and changed-file results never create
that source proof; optional metadata-cache errors do not undo a completed
download. This source identity remains separate from runtime/device qualification.

`AcquisitionSession` captures a selection identity independently of browsing.
Its typed running/failed/cancelled/complete state drives the primary action.
Companion acquisition stays secondary for an already installed usable file.
The read-only [transfer state API](transfer-state.md) provides truthful partial
and conditional HTTP Resume state. A different installed file remains usable
while a transfer runs, and completion never activates a model automatically.

`catalogue_previews` reuses local/cached images first. A missing registered image
can use two bounded Hugging Face reads: pinned revision metadata containing the
exact artifact SHA, then one declared raster image. There is no repository search,
guessed/social thumbnail, remote video or weight fetch. The process owns two
workers, at most four pending jobs and a ten-minute failure backoff. Responses,
dimensions, redirects and request time are bounded. URLs remain HTTPS within the
publisher's service/CDN domains. Artwork and source provenance live in
`cache/catalogue-previews`, keyed by the pinned source tuple; ordinary cache
maintenance does not remove this artwork.

Installed-model discovery, recommendation refresh and registered artwork share
the process-owned metadata cache in `model_sources`. Exact repository/revision
reads deduplicate in flight and reuse successful manifests; an unrevisioned
manifest also supplies its returned immutable revision. At most two HF reads run
at once, with eight metadata reservations and 128 cached responses/16 MiB. Mutable
manifests expire after an hour, pinned manifests after a day, access/not-found
misses after ten minutes and transient failures after a minute. A restart permits
one fresh attempt; positive raster files remain on disk. No inline retries sleep
or loop. HTTP 429 applies a shared HF/CDN cooldown using seconds/date Retry-After
(one second to one day; absent/invalid headers use one minute). Explicit Refresh
respects this policy and retains previously usable artwork/settings on failure.

A verified prepared artifact's pinned source precedes cached HF identity and the
checkpoint header. Known sources never trigger search. Discovery for an unknown
image source keeps the model-name tokens and admits only image-task/diffusers
candidates before fetching at most five manifests. A format or publisher token
does not qualify an LLM repository. The local SHA must match the chosen manifest;
wrong hashes or revisions never supply another model's artwork. The presentation
scan makes one HF attempt after existing publisher art is absent or fails,
including a Civitai version with unusable images. Offline paths start no requests;
switching the catalogue offline cancels queued work and prevents a subsequent
raster read after metadata already in flight completes.

Exact `<filename.ext>.preview.*` sidecars take priority over nonambiguous legacy
sidecars and valid cached publisher images. A local display hit also suppresses
background artwork lookup and its model hashing. Checkpoint, acquisition and
LoRA/embedding cards always receive a bundled category illustration when artwork
is missing, invalid, unavailable or rate-limited; a browser decode failure uses
the same fallback. A visible localized **Default preview** badge identifies this
conceptual artwork, including a decode fallback, and clears when real artwork
arrives. The illustrations contain no baked-in text. The sixteen category assets
and their provenance are documented in
[`ui/assets/model-previews/README.md`](../../ruinedfooocus/ui/assets/model-previews/README.md).
Their disposable copies live separately in `cache/default-previews` and carry no
publisher provenance. They require no model load, local generation, download or
network request. Animated publisher art is rejected without replacing a valid
cached static image.

The timer sends preview-only patches with both catalogue and source identities.
An older response cannot restore an old query, page, selection, title or card
order. The renderer updates only a still-visible card with the same source key.
The checkpoint browser also checks exact local preview sidecars before the existing background
metadata worker has copied them into its cache. Shared-stem variants require
exact sidecars to avoid borrowing another quantization's image.

```sh
python tools/check_cpu.py test_hf_preview_reliability.py test_catalogue_previews.py test_resource_cards.py test_model_acquisition_ui.py test_embedding_metadata.py test_model_recommendations.py test_prepared_native_source.py
```

The CPU suite uses fake bounded responses and inert model files. No live
publisher request, model download, image generation or GPU execution is implied.
