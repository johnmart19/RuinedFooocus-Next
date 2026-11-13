# Family defaults and source precedence

`features/models/resource_defaults.py` reads `resources/defaults/index.json`
and one small `image/` family recipe. `model_recommendations.family_recipe`
is the optional UI-facing reader; unknown or damaged optional data returns no
recommendation. `performance_defaults()` merges the same resources for
`PerformanceSettings`. These readers do not fetch metadata, import tensor or
application runtimes, hash model files, or read qualification history.

The index has schema 1, canonical JSON digests, exact aliases and behavioral
variant references. Each recipe separately records sampling, recommended sizes,
the fresh default size, component roles, applicability and source provenance.
An alternate quantization can inherit a compatible architecture/variant recipe.
It never inherits a claim about an exact artifact's visual quality or hardware
qualification. Explicit checkpoint overrides remain last in the sampling merge;
current saved dimensions remain a user decision.

Missing optional checkpoint recommendations do not block a currently recognized,
connected ordinary native image route. Model Defaults uses existing family
settings or fresh app defaults and labels that source as a fallback, without
claiming a checkpoint recommendation. Saved overrides and compatible source
recipes retain precedence. Ambiguous behavioral variants, specialized native
schedules/conditioning and unavailable loaders retain their explicit requirements;
switching models never reuses the previous family's automatic sampling values.
The same fallback label is shown in defaults details and captured with sampling
settings as `sampling_defaults_source`; Own settings clears the automatic label.

The retained 26 image presets are projected from indexed image resources. The
compatibility snapshot and its index share canonical digest
`5dde2b88f603d0cc76a070fbaddac7f0ed9c3894e8eb83a33ee44c09c9d7beb9`.
The eight video presets and dedicated resolution data are retired. Existing saved
video-only rows are ignored and preserved when a profile is written; shared rows
with valid image consumers retain their image settings. The current index has
92 image recipes. See [product scope](image-chat-scope.md).

## Effective image capture

`features/models/effective_image_plan.py` joins the existing submission captures
into a frozen `EffectiveImagePlan` at `_effective_image_plan`. Call
`capture_effective_image_plan(data, models, values, paths, family=..., controls=...,
performance=..., resolutions=...)` after component, sampling and `SizePlan`
capture and before sampling-asset preparation. It returns a new request dictionary
and preserves the actual component tuple and native size-plan object. The worker
and API use the same entry point; downstream history and Chat use
`get_effective_image_plan(data)` and its detached JSON `metadata()`.

`component_profiles.snapshot_request` reads the checkpoint inventory once and
freezes its selected roles at `_captured_image_components`. The effective plan
exports `required_components` as active required rows with selected concrete paths,
catalogue source keys, byte expectations and installed-file signatures. Optional
and inactive rows do not become required downloads. Built-in components retain
their ownership, explicit overrides stay selected and approximate codec pairs
retain distinct encoder/decoder identities. Native loading and preparation still
own compatibility validation, leases, acquisition and execution.

The plan records the selected native class/configuration, explicit behavioral variant,
header storage types, operation, recipe provenance and automatic/manual/replay/source
size ownership. It retains `SizePlan` geometry without recomputing source framing,
letterboxing, mask alignment or alpha. Pure image operations capture their existing
composition/source geometry and operation assets with empty diffusion sampling.
Prompts, negatives, denoise, LoRAs, masks and pixel ownership stay with the existing
request/editor/asset owners. A settings/geometry intent digest and file signatures
reject stale plan reuse, including a replaced diffusion file behind an unchanged
local bundle index; pixel buffers are never hashed to create this digest.
Client metadata dictionaries cannot stand in for service-captured plan objects.

The plan also retains the existing typed GGUF encoder/projector/reader capture,
including a deliberately captured absent projector. Signature checks cover that
paired dependency; a later sidecar cannot retarget the queued selection. The plan
does not discover or resolve projectors. After preparation of a previously missing
active encoder, `finalize_prepared_image_plan(original_data, prepared_data)` can
bind its newly resolved existing-owner packet only with the original missing CLIP
request and a matching typed preparation signature. It preserves the component,
recipe and size objects and rejects every other intent change. Replay ownership
is included in the intent digest, independently of identical pixel dimensions.

Installed LoRAs retain their existing resolver's typed source captures in the
same plan. The intent binds resolved paths, weights and file signatures before
queuing; execution consumes those captures rather than resolving a later model
root. Original prompt batches retain an immutable exact adapter selection for
each scalar child through `plan.loras_for_prompt(original_prompt)`, sharing the
same file captures. An unknown child or changed selection requires a fresh
capture. Replaced sources or a changed adapter packet require a fresh capture. This
does not add a LoRA loader or change native adapter application. A missing active
catalogue adapter exposes its captured trusted request through the existing
required-component preparation bridge. The finalizer binds only a matching typed
preparation receipt; a missing custom adapter still requires an installed source.

`generation/request_metadata.capture_sampling_replay` runs the existing Library
metadata normalizer once before component, sampling and size capture. The legacy
prompt importer delegates to it. API and tool prompts stay literal at their
submission boundaries, and recorded replay ownership cannot be added to a frozen
manual plan. The UI's single-document `[metadata_json]` transport is normalized
before control/default capture; ordinary prompt lists retain their batch shape.

Native facts come from the accepted local
[capability snapshot](native-image-snapshot.md), including pin, parser,
source-manifest and snapshot identities. Reading that snapshot never scans native
source or polls a publisher. An unambiguous connected generation example can fill
missing ordinary-family sampling or canvas advice only after current native
configuration admission. Its label says **Native workflow example** and explicitly
states that publisher advice is unavailable. Unresolved controls, source-image
examples, conflicting branches, unmatched behavioral variants and specialized schedules do
not supply this fallback. Qwen-edit, Flow/Flux and native inpaint retain their
existing stricter recipe and component contracts.

Automatic sampling application now reads local recommendation/source caches.
The existing UI refresh action, model-selection refresh and background publisher
metadata discovery retain their network ownership. Manual and saved checkpoint
overrides keep precedence while metadata is pending. Compatible renamed/custom
files and alternate storage types do not require an artifact SHA entry to use
their family contract; checksums remain integrity/provenance evidence.

Focused checks:

```sh
python tools/check_cpu.py test_effective_image_plan.py test_source_image_defaults.py test_component_profiles.py test_component_family_contracts.py test_resolution_settings.py test_component_capture_boundaries.py test_compatible_family_defaults.py test_image_sampling_replay.py
```

These are source/CPU/local-fixture contracts. Actual model behavior, generated
pixels, runtime fit and GPU stability require separate acceptance.

## Qwen Image 2.1

The [publisher model card](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/README.md)
shows a 2048 square default/example and seven recommended 2K ratios. Its
Diffusers example uses 40 steps. The separate
[ComfyUI base workflow](https://github.com/Comfy-Org/workflow_templates/blob/611c497cca4087967ecd85fd62bc12fa7130e885/templates/image_qwen_image_2_1_t2i.json)
uses 1024 square, 25 steps, CFG 1 and Euler/Simple. The
[Turbo workflow](https://github.com/Comfy-Org/workflow_templates/blob/611c497cca4087967ecd85fd62bc12fa7130e885/templates/image_qwen_image_2_1_turbo_t2i.json)
uses the explicit eight-step sigma list. Source metadata was inspected on
2026-11-11; revision and fetch time travel with each recipe.

The recipe therefore exposes publisher geometry with `default_size=[2048,2048]`,
the smaller native recipe under `working_recipe`, and native sampling provenance
under `sampling_provenance`. The 1024 recipe is not labelled as the publisher
default. This data contract does not silently resize an existing image or replace
a deliberate current size. Fresh 2048 selection and real output acceptance belong
to the integrated resolution/UI workflow; metadata and CPU tests alone do not
qualify that execution or its memory requirements.

`ModelRecommendations.family_details(checkpoint, family=...)` resolves a current
selection. Qwen accepts either a verified exact publisher source or a current
native Qwen architecture plus one unambiguous base/Turbo declaration. Native
classification must already be initialized. It never selects Turbo from a local
filename or dtype. Contradictory, absent or unsupported declarations retain Own
settings/explicit override recovery. FP16/BF16/FP8 headers and logical GGUF shapes
can use the same compatible recipe; the exact model digest remains a separate
integrity/qualification fact. The dedicated Qwen RGBA VAE and Qwen3-VL encoder
roles stay explicit; a same-channel-count VAE is not automatically compatible.

## Defaults consumers

| Resource                                     | Consumer and scope                                                                                                                      |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `defaults/image/*.json`, `index.json`        | Model recommendations, `PerformanceSettings`, resolution recommendations; saved performance choices remain under the user settings root |
| `performance.default` compatibility snapshot | Older direct resolution readers; no new source metadata is written there                                                                |
| `resolutions.default`                        | `ResolutionSettings` initial menu/legacy profiles; custom dimensions are persisted separately by the resolution feature                 |
| `powerup.default`                            | `features/image_editing/controlnet.py` named image-tool presets                                                                         |
| `styles.default`                             | `features/prompting/{sdxl_styles,style_catalog}.py` prompt style defaults                                                               |

Explicit refresh/acquisition retains the existing bounded publisher metadata
adapters. This bundled reader does not load weights or acquire components when
metadata refreshes. A configured model-cache root is used for new implicit
`ModelRecommendations` cache locations; explicit callers retain their selected
cache directory, and user overrides remain in `settings/model_settings.json`.
No original models, metadata, cache, settings or outputs are migrated or deleted.

## Extending and checking

Add an exact family/variant recipe with a stable ID and its source URL/revision,
fetch/inspection time and evidence kind. Link it from the index, record its
canonical `metadata_digest`, and add aliases only for verified family mappings.
Use `local_family_recipe` for inherited app recipes whose exact publisher
recommendation is unavailable. A local recipe is not publisher evidence.
Keep `legacy` values for existing preset names so saved selections survive.

Recipes also accept their stable ID. New native contracts link labels through
`index.native_recipes[native_class]`; applicability records the native
configuration, architecture, explicit variant aliases and whether a documented
configuration baseline can omit a variant. `compatible_native_recipe(path,
metadata)` joins current exact source proof or an already initialized native
header to those contracts. Hashes identify exact artifacts; compatible storage
quantizations can use the same recognized family recipe without inheriting
artifact/device qualification. A local filename never declares a variant.
Specialized schedulers, second models and codec layouts remain explicit
requirements, with `auto_recipe_supported=False` until their caller honors them.
See [modern native image contracts](native-modern-image-catalogue.md) and
[verified preparation metadata](prepared-native-source.md).
The [extended catalogue contracts](native-extended-image-catalogue.md) retain
empty sampling maps and explicit Auto refusal when only a provisional working
canvas is known. Their provenance does not turn that canvas into a publisher
dimension recommendation.

```python
from ruinedfooocus.features.models.model_recommendations import family_recipe

recipe = family_recipe("QwenImage21", variant="base")
assert recipe["default_size"] == [2048, 2048]
assert recipe["working_recipe"]["default_size"] == [1024, 1024]
```

```sh
python tools/check_cpu.py test_model_default_resources.py test_compatible_family_defaults.py test_model_recommendations.py test_native_variant_defaults.py
```

The tests cover exact legacy projection, aliases, media separation, corrupt or
missing resources, no benchmark/tensor imports in pure lookup, alternate storage,
ambiguous and conflicting variants, and saved override precedence. UI rendering,
actual generated pixels and hardware behavior require their own acceptance.

The output-size dropdown uses stable pixel-dimension values for model
recommendations; changing a family label cannot invalidate a queued selection.
Legacy recommendation labels migrate at this field boundary. Named saved sizes
and Custom retain their identities. Automatic sizing selects the new model's
recommendation or an explicitly labeled generic fallback when no recipe is
available; manual and Library replay sizes keep ownership. The typed dropdown
admits bounded stale dimensions only for reconciliation, while native request
planning still enforces model-specific canvas limits.
The field uses Gradio's native Dropdown configuration schema so its translated
Output target label and inherited control properties survive dynamic updates.
