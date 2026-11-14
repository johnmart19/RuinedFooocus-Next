# Native SeedVR2 still-image restoration

Image tools → **Restore still image (SeedVR2)** restores one source image with an
installed SeedVR2 diffusion model and its matching SeedVR2 codec. With no local
match, the [3B model and official codec defaults](native-utility-defaults.md)
download on Run. The files are
selected independently of the generation checkpoint. The model has the explicit
`image_utility_model` role; it is not an ESRGAN upscaler. The adapter does not load
a text encoder or use a generation prompt, LoRA, reference adapter, or inpainting.

Explicit API actions may select the compatible registered SeedVR2 model and
codec keys instead of installed paths. [First-use preparation](image-preparation.md)
captures only those two selected roles, checks source/size/disk/offline policy and
validates both acquired native headers before compute. Custom missing paths stay
local-install requirements. File preparation never changes runtime packages.

The pinned official still recipe supplies Lanczos resizing, SeedVR2 preprocessing,
tiled VAE encode, SeedVR2 conditioning, one Euler/simple step at CFG 1 and denoise
1, tiled VAE decode, then SeedVR2 postprocessing. Encode/decode use the recipe's
512-pixel tiles, 128-pixel overlap, 4096 temporal size and 8 temporal overlap.
The adapter requires exactly one encoded and decoded frame; it never invokes
temporal chunking, video loading or video output. ComfyUI owns native loading,
sampling, codec memory and color correction. RF owns capture, leases, cancellation,
bounded dimensions, result metadata and image-file transport.

Scale is 1–4, default 4. The native resize rounds each scaled side independently.
RF pads odd reference dimensions to even dimensions, then native preprocessing
pads the right and bottom to multiples of 16. Native postprocessing restores alpha
and aligns with that even reference. RF removes only artificial reference padding,
so no original edge is cropped. An 11×7 source at scale 1 returns the complete
11×7 image. Both the
source and padded target are limited to 16,777,216 pixels. Color correction is
`none` by default, matching the still recipe; `lab`, `wavelet` and `adain` use the
native postprocessor. The RGBA source is an oriented, immutable 8-bit working
copy; precision image export retains the native floating result after restoration.

API JSON and multipart image edits accept:

```json
{
  "controlnet": { "type": "restore-seedvr2" },
  "input_image": "<base64 still image>",
  "seedvr2_model": "seedvr2_3b_int8_convrot.safetensors",
  "seedvr2_vae": "seedvr2_ema_vae_fp16.safetensors",
  "seedvr2_scale": 4,
  "seedvr2_color": "none",
  "seed": 0,
  "n": 1,
  "size": "auto"
}
```

File names may be root-relative or listed absolute paths beneath configured
diffusion-model/checkpoint and VAE roots. Header detection must identify SeedVR2;
the codec uses the pinned native SeedVR2 discriminator. Runtime loading also checks
the native model class, 16-channel codec and one-frame layout. Missing installed
files fail locally. No automatic model acquisition is introduced. Size, if explicit,
must match the requested native scale result. Masks, canvas borders and additional
references are rejected. PNG16/EXR/still AVIF use the existing precision output
contract; AVIF's alpha restriction still applies.

Queued requests retain both file signatures and preparation records. Both read
leases cover native load and all processing stages. Replacement requires a new
capture. Cancellation is checked before loading and between resize/preprocess,
encode, conditioning/sampling, decode and publication; the existing admitted image
lane also owns native interruption. Cached models are keyed by both file identities,
and both patchers are released when replaced or when leaving the utility.
Source/resized/native-reference/grid-padded/output geometry, alpha policy, model/codec basenames and
fixed native sampling are recorded without leaking local directories.

Library **Reuse full setup** recognizes this typed utility metadata independently
of generation prompts and checkpoints. It validates the selected original, resolves
both installed components by their saved SHA-256 identities and current native
headers, and opens Image tools with that original and the recorded scale, color
method and seed. Ambiguous or missing components require a deliberate manual
selection; a basename alone is never enough. Component SHA-256 receipts are cached
by current file identity and checked again when their identity is recorded.
Library → **Edit in Library** also offers **Restore still image (SeedVR2)** with
its own model, codec, scale and color controls and the same owned request stream.
These are fresh operations on the selected original file, not access to a prior
request's discarded upload. Precision sources use the existing explicit 8-bit
working conversion. Models can activate an installed SeedVR2 file directly in
Image tools; its codec remains an explicit compatible selection.

Source inspected 2026-11-11: ComfyUI
`fca60fe70e5b84c5c6b058263653d15cb3401e44`, `comfy_extras/nodes_seedvr.py`,
`comfy_extras/nodes_post_processing.py`, `nodes.py`, `comfy/sd.py`; installed
`comfyui_workflow_templates_json==0.1.129` recipe
`utility_seedvr2_3b_int8_upscale_image.json`, SHA-256
`7075ec75f71f30eeb381510c2a549e3cf7419b906a48e89bd3570a115a608c8c`.
Indexed source artifacts remain the existing WaveC SeedVR2 3B/7B and codec entries.

Run `python tools/check_cpu.py test_native_seedvr2.py test_native_vision.py
test_api_image_edits.py test_api_live_contracts.py`. These execute real pinned
preprocess/conditioning/postprocess node bodies on tiny CPU stand-ins, actual
request capture and public JSON/multipart callers, cancellation, stale-file refusal
and Gradio component registration. They do not qualify real-model semantics,
visual quality, runtime memory, GPU stability or rendered browser behavior.
