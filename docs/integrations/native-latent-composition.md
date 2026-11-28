# Saved native latent composition

Create → Image tools → **Compose saved latents** composes two `.latent` uploads
from this installation into one real latent file. It uses no generation
checkpoint, prompt, LoRA, codec loading, text encoding or sampling. Results use
the existing original-file download surface; the displayed image gallery stays
in place. API image generation returns `data: []` and one `native_tensors` entry,
with an owned download URL or `b64_json`.

Both inputs must carry this installation's original saved workflow receipt and
have identical original dtype, full shape, family, model, codec, text/static-LoRA
and source context. Compatible rank-four `B=1` still layouts at codec scale 8 or
16 are admitted, including a native packed 64-channel rank-four layout. A layered
five-dimensional latent and Cascade's paired stage grids need different native
operations; no layer/frame axis is squeezed or repurposed. External or modified
files without the original local receipt fail before native loading. Each input
is at most 32 MiB and the pair at most 64 MiB. Public inputs are binary uploads
or base64 bytes, never server paths or tokens.

**Replacement with edge feathering** executes the pinned `LatentComposite`
body. **Masked replacement** executes `LatentCompositeMasked`, always with
`resize_source=False`. Coordinates and feathering are output-image pixels and
must be multiples of the original codec scale. The adapter passes
`(pixel_offset // codec_scale) * 8` to the native node, whose offsets are divided
by eight. The destination's complete canvas and axes stay fixed; explicit offsets
clip the overlay to its visible region. Feathering must fit that visible region.
An optional still grayscale mask must match the full saved image canvas and may
contain at most 16 megapixels. White copies the overlay; the native masked body
maps the mask to the latent grid with its own bilinear interpolation.

`native_tensor_io.capture_composition_inputs` copies the two original payloads
into one private request folder and verifies their saved contexts and receipts.
`load_composition_latent` uses the native loader's explicit CPU float32 working
samples. The original F16/BF16 payloads are retained; the output is explicitly
float32, without a silent recast. The native formulas clone the destination,
retain its supported mask data and leave both input tensors unchanged.
`save_composed_latent` uses native `SaveLatent`, records the original context,
input hashes, original/working dtype, operation, coordinates, feathering, mask hash
and exact parameters, and creates a new owner-private source receipt.

Re-upload the same two original files and mask and retain the recorded parameters
to repeat the operation. Repeated requests have separate private folders and
produce the same numeric result and saved-context identity. A saved composed
output can become a later input when its original dtype and context match the
other input. Prior composition history alone does not make otherwise identical
contexts incompatible. No Library image or thumbnail is fabricated for a tensor.

The API operation is:

```json
{
  "controlnet": { "type": "latent-compose" },
  "latent_composition_primary": "BASE64_SAVED_LATENT",
  "latent_composition_secondary": "BASE64_SAVED_LATENT",
  "latent_composition_mode": "replace",
  "latent_composition_x": 0,
  "latent_composition_y": 0,
  "latent_composition_feather": 0,
  "n": 1
}
```

Use `masked` and optional `latent_composition_mask` with the API's normal inline
image transport for a selection mask. Generation tensor fields remain separate.
Only mode and numeric values can be saved as startup defaults. Uploads, private
plans, output paths and source receipts are never saved as settings. Inactive
mode controls retain their form values and do not enter the current operation.

Execution stays on the existing owned image lane. Cancellation is checked around
native reads, composition and serialization. A successful tensor-only completion
must be the typed result for the captured request, contain no fabricated image,
and identify exactly one unchanged native export in that request's directory.
Unfinished, foreign or modified outputs fail completion validation.

Source: ComfyUI `fca60fe70e5b84c5c6b058263653d15cb3401e44`,
[`LatentComposite`](https://github.com/comfyanonymous/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/nodes.py)
and
[`LatentCompositeMasked`](https://github.com/comfyanonymous/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy_extras/nodes_mask.py).
The adapter compiles only the pinned pure method bodies with request-local
namespaces and signatures. It does not import the complete native node module
to discover the operation. File transport reuses
[saved tensor contexts](native-tensor-workflow-contexts.md).

Focused CPU checks use `tools/check_cpu.py` with `test_latent_composition*.py`,
`test_native_latent_composition_io.py`, `test_native_tensor_context.py` and
`test_native_tensor_io.py`. These exercise actual pinned native math and
serializers, mounted installed Gradio controls, worker/API callers, owned media,
receipt refusal, codec geometry and repeatability. They do not establish rendered
browser acceptance, real-model quality, GPU behavior or platform performance.
