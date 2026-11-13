# Native SDXL base/refiner handoff

Select an ordinary SDXL base as the image model. The contextual **SDXL base →
refiner** panel selects an installed native SDXL refiner and an explicit switch
step, from one through total steps minus one. Refiner CFG is optional: blank
keeps the base CFG and zero is an explicit override. A split refiner can use an
explicit installed CLIP-G encoder; a bundled refiner keeps its embedded encoder.
Final decoding uses the base model's selected SDXL VAE. Switching to a different
model family disables the refiner selection without changing CFG or split values.

Capture validates the switch against the effective total from `custom_steps`, the
same control the worker uses to create runtime `steps`. Saved **Own settings**,
**Model Defaults** and an explicit API `steps` override keep their existing
precedence. For example, saved 30 steps with switch 24 is admitted; requesting
20 steps with switch 24 is refused without increasing the total or moving the
switch. Already-normalized direct callers may supply only `steps`.

`generation/sdxl_refiner.py` captures a frozen `RefinerPlan` plus both checkpoint
sources and the actual separate encoder when required. Existing primary component
capture also freezes the base's encoders and VAE. Preparation remains outside the
serialized compute worker; source read leases remain held through native loading
and sampling. Refiner cache identity includes its checkpoint and encoder signatures
and is cleared when the primary model changes. Cancelled loading does not publish
a cache entry. Native ComfyUI memory management continues to own model residency.

`image_pipeline.process` dispatches to `sample_handoff` after source/mask encoding
and native base conditioning. It reuses the requested seed, denoise, sampler,
scheduler, mask and callback. Both runtime models must be native ordinary
four-channel SDXL base/refiner models with matching prediction types, latent
formats and complete native sigma schedules. A different source, model, split or
schedule fails before sampling. Safetensors checkpoints are required by this
bounded adapter; external CLIP-G uses the existing native state-dictionary/GGUF
reader. Unknown encoder headers remain unqualified, and runtime features must
have native 1280-wide CLIP-G tokens and pooled output.

The sampler calls reuse `comfy.sample.sample` with the native
`KSamplerAdvanced` contract. The base stops at the split with
`force_full_denoise=False`, preserving residual noise. The refiner receives that
same latent, the same native sigma schedule and the remaining step interval,
with empty added noise and full final denoise. The adapter never decodes or
re-encodes the intermediate latent. LCM uses the existing native `SamplerLCM`
controls and native custom sampler with the same schedule split. Callback steps
are offset into the original total. Zero denoise returns the original latent
without loading or sampling the refiner and records that skipped handoff.

The refiner receives the same processed prompt text through native
`CLIPTextEncodeSDXLRefiner`, with actual image geometry and native aesthetic
defaults (positive 6, negative 2.5). Clip Skip, embeddings and explicit prompt
switches keep their existing contracts; switches are bound against the complete
shared schedule. Base LoRAs and base-only control conditioning retain their
existing primary owner. The refiner is a separate selected checkpoint, without
implicit LoRA transfer.

UI, API and Library setup replay use `sdxl_refiner_model`,
`sdxl_refiner_switch_step`, `sdxl_refiner_cfg` and `sdxl_refiner_clip`. Runtime
metadata `sdxl_refiner_source` records source signatures, schedule, seed, denoise,
split, effective CFG and the noisy-latent/no-added-noise handoff. A new request
clears old runtime proof and recaptures the selected files. No new Auto recipe
or source qualification flag is enabled. Explicit stable-diffusion.cpp requests
refuse staging, while automatic backend routing retains the native route.

Pinned source, inspected 2026-11-11:
[ComfyUI fca60fe native advanced sampler](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/nodes.py#L1635)
and [SDXL refiner conditioning](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy_extras/nodes_clip_sdxl.py#L7).
GGUF encoder adaptation remains pinned to
`da5f435cd9a9e54ad27eed417ed72b1f2543ef41`.

Feature `native-sdxl-refiner-handoff` declares the focused guarded CPU checks.
Tests execute pinned native sampler/schedule/conditioning bodies and actual
production callers using weightless fixtures, including partial denoise, noisy
split preservation, zero refiner noise, explicit CFG zero, scheduled prompts,
LCM controls, captured source replacement, cache reuse, cancellation and replay.
Real base/refiner weights, GPU stability, memory fit, image quality, rendered UI
acceptance and root independent feature replay remain separate evidence gates.
