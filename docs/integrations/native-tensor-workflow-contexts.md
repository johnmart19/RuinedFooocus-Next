# Saved native workflow tensor contexts

Create → **Advanced latent and conditioning files** also handles locally exported
Layered, Cascade and refiner tensors, numeric masks, reference tensor lists and
multiple safe conditioning entries. **Preserve native workflow tensor context**
opts ordinary generation into this format. Native image/reference/layered/staged
requests select it automatically when tensor I/O is enabled. The checkbox alone
does not enable I/O, capture files or copy tensors.

This extends the [ordinary native file consumer](native-tensor-generation-io.md).
Bare external files keep that consumer's deliberately narrower model/codec rules.
Complex files require their original locally saved SHA-256 context receipt. A
channel count alone cannot identify a layer axis, a Cascade C/B pair, a noisy
refiner split, or the text/vision encoders which produced reference conditioning.

## Native data, source identity and replay

The workflow adapter uses native `SaveLatent`/`LoadLatent` and
`SaveConditioning`/`ConditioningLoader`. Native `SaveConditioning` writes one entry,
so each original entry is saved separately, then its unchanged safetensors payload
is joined under bounded entry prefixes. Import reconstructs each native file and
calls the native loader for that entry. No pickle, executable object serialization,
latent resizing, axis squeezing or float conversion is added. Native `LoadLatent`
still creates float32 working samples; the original uploaded bytes and dtype remain
available in the retained file and metadata.

The safe extension preserves native numeric/bool/string options, tensor options
such as `mask` and `concat_latent_image`, and lists such as `reference_latents`.
It retains every entry and interval instead of flattening scheduled conditioning.
There are at most 64 entries, 256 tensors and 16 references per option, with the
existing 32-MiB file, 128-KiB header and 64-MiB combined-input limits. Primary
conditioning remains floating batch-one rank-three text data. Mask/attention
options can additionally retain native bool/int32/int64 dtypes. Input headers bound
allocation; output bounds precede native copies. Nonfinite tensors are refused
before sampling or publication.
Malformed structure types, unknown/duplicate tensor names and invalid replay
roles are controlled input errors before receipt lookup or native file reading.

Each export has a server-owned context receipt keyed by the complete file SHA-256
beneath the configured cache. Inputs are accepted only when that receipt matches
their complete saved context. No route lists, edits or imports receipt paths.
Cache maintenance can remove receipts; a file without its original receipt cannot
claim a locally verified complex workflow. This is a local provenance boundary,
not a portable trust signature for files from another installation.

Context binds checkpoint identities, resolved codec/text-encoder components,
static LoRAs, native model configuration, sampling controls and original source
pixels/masks/reference order. Source pixels and masks are checked again before
use. File stamps use decimal-string nanosecond values; absolute source directories
are omitted. A different source, codec, encoder, stage companion, layer count or
native frame fails explicitly. Inputs hold the existing read leases, and
cancellation or partial output never publishes an incomplete tensor group.

Workflow conditioning replaces its chosen final numeric slot after native
source/mask/reference attachment. It cannot append the saved reference twice.
The counterpart remains the current request's native conditioning. Existing model
and source preparation still runs; a file does not supply missing companions or
fabricate prompts. Library **Reuse full setup** continues to require re-uploading
the exact original input bytes and reattaching original source images/masks.

## Native layouts and stages

Qwen Layered files preserve `[1,16,layers+1,H/8,W/8]`, including the reconstruction
slot. Export occurs before the native layer cut. Import returns the same native
layout; the existing `OutputPlan` and native layer cut produce the ordered RGBA
outputs. Tensor downloads attach to the completed `ImageBatchArtifacts` without
replacing its output-plan proof or turning files into gallery images. The RGBA
codec, layer count, original source, complete frame and aggregate layer budget
remain required.

Cascade prior files contain the sampled `[1,16,Hc,Wc]` C prior and the exact paired
`[1,4,Hb,Wb]` B target grid. B's output is four times its own grid; it is never
derived by applying an SD scale to C. The existing captured Stage B model and
encoder perform the second stage. Original source/output frame, compression and
size-plan intent are retained. A bare C tensor cannot restore an unknown B grid.

Standalone SDXLRefiner remains four-channel rank four at spatial scale eight.
The still Hunyuan refiner remains `[1,32,1,H/16,W/16]` with its dedicated codec and
original concat-source conditioning. A fully sampled staged SDXL latent is a
final latent and can start another explicitly denoised staged request. It is never
silently treated as an intermediate noisy split.

**Export noisy SDXL refiner handoff** is a separate opt-in control/API boolean
`native_export_handoff`. It requires an explicit captured SDXL base/refiner pair.
The file is identified as `kind: "sdxl-handoff"` in API downloads and records the
split step, complete sigmas, original denoise/seed, both CFG values, sampler/noise
options, source signatures, prompts hash and optional native noise mask. Import
through `native_latent_input` resumes only the remaining refiner steps, with empty
added noise and the same full schedule. A changed prompt, seed, schedule, source
or companion fails. One split is resumed per request. Its recorded denoise is
part of that explicit resume operation; it does not become an arbitrary img2img
starting point.

The public generation fields, image responses, opaque expiring download URLs and
Library exact-byte replay route remain the existing ones. Workflow files add
`native_tensor_workflow: true`; uploads are still binary UI data/base64 JSON,
never paths or a generic tensor manager. `native_export_handoff` defaults false,
as do the existing latent and conditioning export controls.

## Object state and composition boundary

Native `hooks`, `gligen` and `control` conditioning values contain executable
patch/model objects. These are refused by the file serializer; JSON metadata is
not a substitute for their live native state. CLIP-only hook embeddings are numeric
and may retain all native intervals when the captured hook source/settings context
matches. A live model-hook group cannot be claimed as serialized conditioning.
The [numeric patch combinations](native-tensor-patch-contexts.md) describe fresh
hypernetwork callbacks, latent-only GLIGEN and exact CLIP-only interval provenance.
Latent-only reuse still requires the original request to rebuild its native model
patches. IP-Adapter objects and dual-model sampler state require separate explicit
reconstruction contracts.

Latent composition has three narrow helper contracts in `native_tensor_io`:
`capture_latent_companion(raw,data,manager,models)`,
`load_latent_item(owner,data,item)` and `assert_composable(primary,secondary)`.
The gate admits locally receipted native-still rank-four layouts at codec scale
eight or sixteen, with exact model/codec/text/static-LoRA context, shape and
original dtype. Packed 64-channel rank-four tensors retain all channels. Native
composition divides its offset arguments by eight: an output-pixel coordinate
must align to the actual codec scale and becomes `(pixels / codec_scale) * 8` for
that native argument. Layered rank-five axes and Cascade paired grids require
their own operation; they cannot be silently reinterpreted as this canvas.

The standalone composition consumer uses `capture_composition_inputs(first,
second,manager)` to obtain a private `TensorPlan` with two owned inputs. It reads
only local receipts and uploaded bytes, without selecting, inspecting or loading
a checkpoint. `load_composition_latent(data,item)` runs native `LoadLatent` on the
owned file. `save_composed_latent(data,primary,latent,composition=...)` accepts the
actual `LatentComposite`/`LatentCompositeMasked` result and uses native `SaveLatent`,
retaining the original model/codec/text provenance with both input hashes and the
operation metadata. Native working samples and the resulting file are explicitly
float32; uploaded float16/bfloat16 bytes remain unchanged. The exporter refuses
casts, reshapes, executable metadata and paths from another job. These are the
transport contracts for the complete
[saved latent composition UI/API consumer](native-latent-composition.md); they
do not create a file-manager endpoint or run generation.

Source inspected 2026-11-11: pinned ComfyUI
`fca60fe70e5b84c5c6b058263653d15cb3401e44`, `nodes.py`,
`comfy_extras/nodes_cond.py`, `nodes_qwen.py`, `nodes_latent.py`,
`nodes_stable_cascade.py`, `nodes_hunyuan.py` and the existing RF Layered/Cascade/
SDXL-refiner callers. No dependency changed.

`python tools/check_cpu.py test_native_tensor_context.py test_native_tensor_io.py
test_layered_generation.py test_native_cascade.py test_sdxl_handoff.py
test_image_output_batches.py` exercises real native serializer/cut/sampler bodies
on tiny weightless CPU fixtures and production capture, final conditioning,
handoff, result-plan, UI/API and replay callers. These checks do not qualify
real-model behavior, rendered-browser interaction, visual quality or GPU execution.
