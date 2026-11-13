# Native latent and conditioning generation files

Create → **Advanced latent and conditioning files** supplies optional native
`.latent` and conditioning `.safetensors` inputs to ordinary native SD1, SD2,
SDXL, SD3 and Flux generation. Each input is an explicit upload. The latent
replaces the empty latent and determines the complete image size; its denoise
defaults to 0.5. A positive or negative conditioning upload replaces only that
prompt's encoded conditioning. Its counterpart is encoded normally. Full denoise
can discard the imported latent in flow models.

**Export sampled native latent** and **Export native positive and negative
conditioning** are off by default. Selected exports appear beside the original
images in **Original output files**. Images remain the gallery's image records.
Exports capture the final sampled latent before decoding and ordinary conditioning
used for sampling. They precede any optional pixel upscaling or image-file color
conversion. Off performs no tensor capture, file creation, import or output clone.
These choices and uploads are request controls, not saved generation defaults.

[Saved workflow contexts](native-tensor-workflow-contexts.md) extend this route to
locally exported layers, stages, refiners, masks, reference tensors and multiple
safe conditioning entries, with original source and companion identity checks.

The caller is `image_pipeline.pipeline.process`; capture belongs to
`image_preparation.capture_image_assets`, and transport to the existing worker,
Create result callback and image-generation API. The adapter in
`features/generation/native_tensor_io.py` executes the pinned native
`LoadLatent.load`, `SaveLatent.save`, `ConditioningLoader.execute` and
`SaveConditioning.execute` bodies. Request-local path bindings leave native global
input, embedding and output directories unchanged. Native loaders use safe tensor
readers; pickle files are refused before native loading.

## API

`POST /v1/images/generations` accepts the existing generation fields plus:

```json
{
  "model": "image/installed-model",
  "prompt": "continue this image",
  "native_latent_input": "<base64 native .latent bytes>",
  "native_latent_denoise": 0.5,
  "native_positive_input": "<optional base64 native conditioning bytes>",
  "native_negative_input": "<optional base64 native conditioning bytes>",
  "native_tensor_family": "SDXL",
  "native_export_latent": true,
  "native_export_conditioning": true,
  "size": "auto",
  "n": 1,
  "response_format": "url"
}
```

Omit unused input fields. `native_tensor_family` can be `auto`, `SD15`, `SD20`,
`SDXL`, `SD3`, `Flux` or `FluxSchnell`. An external native file without saved RF
context requires its original family explicitly. RF exports carry that context.
An explicit API size must match the latent's full native size. One through four
image repetitions are allowed. Inputs accept bytes through binary UI uploads or
base64 through JSON; they never accept local paths or another request's file ID.

The usual response `data` contains images. An additional `native_tensors` array
contains `kind` (`latent`, `positive`, `negative`), `family`, `sha256`, `seed`,
`generation_index`, and either `url` or `b64_json` matching `response_format`.
URLs use the existing opaque, expiring image-asset transport and retain the actual
`.latent`/`.safetensors` download filename. There is no separate tensor listing,
file-management or arbitrary-path API.

## Context, precision and bounds

Files do not supply the diffusion model, VAE, text encoders, LoRAs or sampling
settings. Use the original model and companions. Capture inspects the native
checkpoint class; compute validates the loaded class, native input channels,
2D codec, latent channels and spatial scale eight. RF context records model hash,
codec type and fingerprints of selected component and LoRA file identities. A
changed companion selection or known model mismatch requires a fresh compatible
input. File fingerprints intentionally bind local installed companions; moving
or replacing those files can invalidate an old export's context. For external
files without these receipts, family declaration is necessary but cannot prove
which checkpoint or companion originally produced them.

Each file is at most 32 MiB and its JSON header at most 128 KiB; the three inputs
together are at most 64 MiB. Only float16, bfloat16 and float32 are admitted. Input
headers are checked before tensor allocation: exact names, dtype, rank, batch one,
shape, byte count, nonoverlapping contiguous offsets, metadata and no trailing
data. Latents have rank four and at most 16,777,216 decoded pixels. Conditioning
has rank three, a family-matching encoder width and at most 16,384 tokens, with
the family's pooled output when required. Only bounded ordinary numeric/bool
conditioning options are admitted. Nonfinite tensors are refused. Export bounds
run before the native CPU clone, with at most three files per repetition.

Uploaded bytes are retained verbatim in a random request-owned directory beneath
the configured cache. Headers and output metadata retain original precision and
SHA-256 identity. Native `LoadLatent` deliberately creates float32 working samples;
legacy unmarked SD1/SD2 files also receive its native `1 / 0.18215` conversion.
Marked modern latents do not receive that legacy conversion. Exported tensors
retain their native dtype, including bfloat16. RF adds only bounded context
metadata to the native file; tensor payload bytes are unchanged.

Input read leases cover identity verification and native reads. Replacement or
cancellation fails the request. Cancellation is checked around each import/export;
partial export groups are never attached to a result. Private cache files follow
the existing cache lifecycle; the bounds above are per request, not a global disk
quota or an automatic deletion promise. Raw uploads, private paths and tensors
are not embedded in PNG metadata or settings.

Library **Reuse full setup** restores tensor family, denoise and export choices.
A tensor-dependent image requires re-uploading its exact original input bytes in
Create first; the replay checks SHA-256 and preserves those uploads. It never
silently replaces an absent latent or conditioning input with a fresh one. API
replay likewise supplies the original bytes along with the saved generation setup.

## Deliberately bounded first consumer

Bare external files use this ordinary still contract. Locally exported complex
files use the separate saved-workflow provenance and native-layout contract linked
above. Executable GLIGEN, model-hook and ControlNet objects cannot be serialized
as numeric conditioning. SD.cpp, video and non-generation utilities remain outside
this native still consumer. Inputs do not replace installed companions or skip
the app's existing model/component preparation.

Source inspected 2026-11-11: ComfyUI
`fca60fe70e5b84c5c6b058263653d15cb3401e44`, `nodes.py`,
`comfy_extras/nodes_cond.py`, `comfy_api/latest/_io.py`,
`comfy/supported_models.py` and `comfy/sd.py`. No dependency pin changed.

Run `python tools/check_cpu.py test_native_tensor_io.py
test_native_image_output_transport.py test_metadata_guidance_replay.py
test_gallery_actions.py test_api_image_edits.py test_api_live_contracts.py`.
These execute pinned native serializer bodies on tiny CPU tensors, production
capture, sampler-handoff/export-before-decode statements, worker packets, actual
Create/Library callbacks, real Gradio component registration and public API
requests/downloads. They do not qualify real model execution, rendered browser
behavior, visual quality, accelerator memory or GPU stability.
