# Native still-image schedules and output groups

The unconditional picker filters current bounded Ideogram headers, excluding the
primary and known conditional artifacts. Refresh rechecks the candidates.
Repeated capture revalidates the immutable recipe and reapplies public typed
parameters, preserving a prepared override independently of its baseline.

Create uses the current checkpoint's **Model Defaults** for Flux.2/Klein,
Chroma Radiance and Ideogram 4. **Native sampling**, beside the image controls,
exposes matching schedule parameters and Ideogram's unconditional checkpoint.
Empty parameters use the reviewed recipe. Explicit ordinary schedules retain
their values while a required native model guider stays active.

The panel follows the effective checkpoint, including **Saved settings (PNG)**
model replay, without replacing the model selection or prompt draft. Compatible
typed schedule values and a still-valid unconditional selection survive model
and metadata refreshes; parameters absent from the new schedule are cleared.

| Recipe                 | Native execution contract                                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Flux.2 dev             | Canvas-dependent Flux2Scheduler, FluxGuidance 4, BasicGuider, sampler CFG 1                                                    |
| Klein base 4B          | Flux2Scheduler and CFGGuider, default CFG 5                                                                                    |
| Klein distilled 4B     | Flux2Scheduler and CFGGuider, default 4 steps and CFG 1                                                                        |
| Klein distilled 9B KV  | Flux2Scheduler, CFGGuider and request-local FluxKVCache with native reference timestep method                                  |
| Chroma Radiance        | ModelSamplingAuraFlow shift 1, BetaSamplingScheduler alpha 0.4/beta 0.4, CFGGuider                                             |
| Ideogram 4 conditional | Ideogram4Scheduler mean 0.5/std 1.75, separate conditional/unconditional UNETs, DualModelGuider with a text-free negative pass |

The Scheduler menu and `/v1/capabilities` expose actual native node IDs. They are
validated against the captured model recipe, without changing
`KSampler.SCHEDULERS` or constructing KSampler with an invented enum value.

## Capture and execution

`generation/native_sampling.py` supplies the frozen `NativeSampling` contract.
ModelRecommendations validates native schedules separately; PerformanceSettings
carries the plan into the request. Asset capture resolves it again before compute.
Native sampling capture stays inside image preparation's existing generation
boundary. Standalone background removal and upscaling do not resolve the selected
generation checkpoint, native schedule parameters or unconditional companion,
including requests marked `_sampling_captured=False`. Ordinary generation still
resolves and captures its current native sampling plan before entering compute.
The source or current native configuration plus explicit behavioral variant
chooses the recipe. Filenames and storage precision do not identify training
variants. Passive controls perform no network lookup, payload hash or weight load.

Ideogram's current source identifies the unconditional companion by registry
artifact ID. URL, expected digest and byte count use the existing preparation
service. Explicit alternatives are configured relative model names; absolute
paths, traversal, the primary file and other native families are refused.
Compatible alternatives do not acquire publisher-artifact qualification.

The secondary file read lease covers loading and sampling; its cache uses the
current signature. Loaded models must have matching Ideogram input channels and
layer count. Each inference dtype must be native-supported; compatible mixed
BF16/FP32 is allowed. ComfyUI owns loading, guider lifecycle, memory and sampling
cancellation. Native KV patches clone the model for the request.

Explicit steps, CFG, sampler and ordinary scheduler values remain intact.
Flux.2 dev's BasicGuider contract requires sampler CFG 1; an incompatible value
fails clearly. Its distilled guidance is a separate conditioning value. Chroma's
explicit supported sampling shift is preserved; its automatic native AuraFlow
shift is 1.

The API accepts `native_negative_model` and four optional parameters:
`native_schedule_alpha`/`native_schedule_beta` (finite 0.01..50),
`native_schedule_mu` (-10..10), `native_schedule_std` (0.1..5). Zero beta parameters
are excluded because the distribution is undefined there. Partial denoise uses
the native schedule's tail; more than 10,000 internal steps is refused before
allocation. Metadata records the actual schedule, parameters, guider, source
basis and secondary inference dtypes. Effective guidance records the applied
shift and sigmas. Normal PNG output is unchanged.

## Native output groups

`image_output_batches.py` separates repetitions from layer/instance outputs.
`OutputPlan` declares per-generation minimum/maximum counts. Every capture mints
a private request token. The native helper returns `ImageOutputBatch` with ordered
images and copied per-image metadata, validated against that captured plan.

Only successfully completed worker groups produce `ImageBatchArtifacts` carrying
the current plan and completed repetition count. The API checks this proof and
the declared counts separately from `n`. Successful segmentation can return zero
instances. Cancellation, partial runs, ordinary empty generations, saved claims
and previous requests' proofs cannot authorize an empty result. Create reports
"No matching instances found." for a proven completed empty group.

`native_output_group` records order/count/generation index; `native_image_output`
records operation-specific details. They are display metadata. Capture removes
previous display fields and private plans before constructing a fresh contract.
Tokens and completion objects are never serialized into image metadata or
accepted as API input.

## Source and checks

ComfyUI pin: `fca60fe70e5b84c5c6b058263653d15cb3401e44`. The adapters call
`nodes_flux.py`, `nodes_custom_sampler.py`, `nodes_ideogram4.py` and
`nodes_model_advanced.py`. Reviewed controls come from installed
`comfyui_workflow_templates_json==0.1.129` variant workflows; resource metadata
retains their exact hashes.

Feature `native-specialized-image-sampling` declares guarded CPU checks covering
prepared source/default callers, native schedules/guiders/patches, secondary
capture/leases/configuration/dtypes, Gradio/API transport and the actual worker's
completed-empty versus failed-empty behavior. Source receipts are under
`cache/plan-execution-20261111`. Independent replay, rendered browser checks and
real-model/device/quality qualification are separate.

[Qwen Layered](native-qwen-layered.md) adds a separate complete source route for
ordered RGBA layers. [Cascade source conditioning](native-cascade-source.md)
and [SDXL base/refiner handoff](native-sdxl-refiner.md) have their own feature
contracts and acceptance limits; this sampling packet does not qualify them.
