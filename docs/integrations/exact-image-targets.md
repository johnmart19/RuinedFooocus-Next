# Exact image targets and native pixel upscaling

[Finite image workflows](image-workflows.md) use this same exact-target capture
for their explicit upscale step. The step freezes the selected native model,
source artifact/alpha and both target dimensions, then appends lineage only
after terminal success. A preset does not store an uploaded source or model path.

Create's Generation size editor is owned by `ui/app.py` and
`features/models/resolutions.py`. Local recommendation precedence is an explicit
user choice, current signature-bound publisher metadata, applicable family/variant
recipe, then an identified generic fallback. `ModelRecommendations.family_details`
and `family_recipe` keep publisher dimensions and smaller local working recipes
separate. A recommendation is not a loader maximum or a memory guarantee.
Qwen Image 2.1's publisher 2048 default is distinct from the local 1024 working
recipe; Turbo behavior comes from verified variant identity, independently of
GGUF or another quantization.

## Capture and geometry

`features/generation/size_plan.py` owns frozen `SizePlan` version 1. UI worker
capture and API submission freeze the requested target, native factor, nominal
integer content, native canvas, restoration rectangle and policy before waiting
for preparation or compute. Subsequent browser edits cannot mutate that object.
The loaded native VAE supplies spatial compression; native pixel-space family
patch alignment is applied separately. HiDream O1 uses its native 32-pixel
alignment and 64–4096 side contract. Other model recommendations are not used as
hard limits. Application API working-canvas/pixel limits remain separate.

Create's **Effective stages** shows the current generation dimensions when
automatic upscaling is Off. Selecting an upscaler shows the content, native
canvas and target stages; named sizes and custom dimensions use the same helper.

For 1920 × 1080 at 2×, content is 960 × 540. An alignment-16 native canvas is
960 × 544, with two padding rows on each side. Decode restores exactly the
960 × 540 content before native upscaling produces 1920 × 1080. At 4×, content
is 480 × 270 and the alignment-16 canvas is 480 × 272. Original content is
neither stretched nor silently cropped.

For targets not divisible by the factor, integer content rounds each side up.
After native enlargement, one uniform fit brings the complete image inside the
exact target; any remaining border is neutral RGB or transparent alpha. The
`uniform_fit_pad` policy is explicit in metadata. Divisible targets use
`padding_only` and have no final resize. Source framing's deliberate Crop option
remains available in its own image-edit workflow. Default source-preserving
edits, masks, denoise, outpaint and native diffusion SD4x retain their established
geometry instead of silently acquiring a new target plan.

Input-only upscaling preserves oriented source size × native factor while model
size following is enabled. A manually selected target uses the same uniform fit
and padding contract, preserving alpha and leaving uploaded pixels untouched.
PNG metadata records the oriented source dimensions, mode and pixel SHA-256.
Editor and direct-tool requests explicitly capture their upscale source; the shared
UI builder preserves that image through worker admission, even when Create's
separate upscale upload is empty or stale. Ordinary Create input-only upscaling
continues to use its dedicated upload.

## Native integration and caching

The native source remains ComfyUI
`989d93d0d968e704c589ce0146bb18c77f6b50cc`, with ComfyUI-GGUF
`da5f435cd9a9e54ad27eed417ed72b1f2543ef41`. The adapter calls pinned
`UpscaleModelLoader.execute` and `ImageUpscaleWithModel.execute`; the latter
separates alpha from RGB and restores uniformly resized alpha after tiled model
execution. RF retains its existing interrupt checks and shared compute ownership.
No model loader, sampler, independent compute queue or new engine is introduced.

`automatic_upscale.registered_upscalers()` reads exact catalogue-bound native
scale facts for all 40 registered pixel models: nine 2×, twenty-seven 4×, two 3×
and two 8×. Filenames do not establish scale. The selected factor and loaded
descriptor must agree; RGB channel compatibility is checked before processing.
Latent and diffusion upscalers stay in their separate native workflows.

Unknown local pixel models can first run through Image tools' native loader.
Successful native inspection stores scale and channel facts under the configured
`cache_path/upscaler-metadata`, using the versioned cache contract and full file
identity. Browsing reads these small records, never weight tensors. Replacing a
file invalidates its inspection record. Missing or stale metadata requires native
inspection before that model can own an exact target. Removing this derived cache
does not delete weights or user size choices.

Optional stable-diffusion.cpp routes retain aligned ordinary canvases. Auto routes
requests requiring native padding restoration or a pixel-upscale stage to native
ComfyUI; an explicit incompatible engine choice explains the unsupported request.

## Persistence, API and migration

Named sizes use schema-1 `image_sizes.json`/`video_sizes.json` in the active
profile folder, scoped by family. Writes are atomic; bundled default resources
are never changed. Legacy `resolutions.json` entries remain available, and old
numeric selections are retained as deliberate manual choices. Existing
`resolution_auto` migration remains authoritative. Only explicit user input or
**Use model recommended size** changes ownership of a manual selection.
Malformed or unsupported custom-size resources leave model defaults usable and
report the problem in Size details. A save cannot overwrite that damaged file.

New generation PNGs record `size_plan`, `size_mode`, effective content dimensions
as `generation_width`/`generation_height`, and actual file dimensions as
`width`/`height`. JSON import, Saved settings and Library setup replay restore
the final target before recapturing the plan. Old PNGs without `size_plan`
continue to use recorded generation dimensions; they do not acquire inverse
target math twice. Editing a dimension or choosing a shortcut opts into a new
exact target. Library replay runs generation; input-only source pixels must be
selected explicitly rather than inferred from a private filesystem path.

Image API requests accept `automatic_upscale` and an optional required
`automatic_upscale_factor` (2 or 4). `size` always names the final target; factor
mismatches fail before queueing. The standalone upscale endpoint accepts optional
`size`; omission preserves native source × factor behavior. Its original PNG
setup/profile text remains intact, while explicit target provenance uses the
separate `ruinedfooocus_upscale` PNG text record.

## Verification and limits

Use the checkout interpreter and CPU guard:

```sh
python tools/check_cpu.py test_size_plan.py test_exact_upscale_target.py test_resolution_settings.py test_image_upscale_metadata.py test_api_image_edits.py test_upscale_transparency.py test_sdcpp_caller.py test_utility_sampling_capture.py
```

Patterned tensor/alpha fixtures check removable padding, complete edges, exact
2×/4× output, odd dimensions, source immutability, signature-bound inspection,
profile/family separation and legacy/new PNG/API/Library replay. Other direct
callers cover generation defaults, source editing and specialized SD4x behavior.
The PNG metadata fixture executes the current indexed worker output loop with
plain images and typed native output plans, preserving per-instance order and
native sampling provenance alongside generation and final dimensions.
These are CPU/source checks. Browser rendering, real 2×/4× models, visible framing,
image quality, device performance and GPU stability require root's separately
serialized acceptance and exact source/model/artifact receipts.

Size visibility, recommendation and effective-stage projections use hidden
progress with empty native progress targets. They retain their actual outputs
and programmatic change listeners. Auto/manual ownership keeps its existing
user-input and replay rules. `test_resolution_ui_events.py` checks the real native registrations,
Auto/manual transitions and repeated size projections without launching the app.
Run it with `test_resolution_transitions.py`, `test_resolution_settings.py` and
`test_layout_progress_targets.py` through the CPU check runner.
