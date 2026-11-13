# Native WeVisOne detection and matting

Create → **Image tools → WeVisOne detection** draws named objects on the source.
**WeVisOne matting** creates a cutout, grayscale matte or green matte overlay.
Enter one label per line for detection, or describe the subject for matting.
Both operations use one source image and an explicitly selected WeVisOne encoder;
they require no diffusion checkpoint, generation prompt, LoRA or sampler.

The backend prerequisite is ComfyUI
`989d93d0d968e704c589ce0146bb18c77f6b50cc` (0.45). Install a compatible
WeVisOne-VL 2B or 4B safe-tensor file below the configured **text encoders** roots.
The passive native detector must identify `WEVISONE_2B` or `WEVISONE_4B`, including
the mask decoder. An ordinary Qwen3-VL encoder does not satisfy this contract.
The installed model's **Use in Image tools** action selects the exact encoder
and matting operation together. The model picker does not choose a file by default.

## Native caller and ownership

`features/image_processing/wevisone.py` uses native `comfy.sd.load_clip` with
`CLIPType.STABLE_DIFFUSION` and the default device policy, matching the public
native CLIPLoader recipe. It verifies the loaded `WeVisOneTEModel` and
`WeVisOneTokenizer`. Native `WeVisOneDetect.execute` then owns prompt construction,
tokenization, greedy generation, answer parsing, image preprocessing and mask
decoding. Detection uses its `detection` task; matting uses `segmentation` with
`individual_masks=False`. Other native tasks are outside this finite adapter.

Capture freezes the oriented RGBA8 pixels, complete dimensions, query/options,
model signature, positive native type and native source signatures. The model
artifact has role `wevisone-clip` and category `clip` (physical `text_encoders`).
The existing preparation service holds its read lease throughout native execution;
changed files or native sources fail before publication. The existing image lane
owns compute admission, cancellation and the native patcher cache/unload lifecycle.
The native backend's CPU/device policy remains authoritative.

Queries contain at most 4096 characters and 64 nonempty lines. Answer length is
an integer from 1 through 32768 native tokens, default 4096. Sources are single
eight-bit still images with at most 16,777,216 pixels; animated and higher-precision
uploads require explicit conversion before this operation. The canvas is never
resized by the adapter. Native preprocessing retains its own documented resizing.
Inline API uploads use the shared native precision reader before Pillow conversion;
encoded PNG16 and AVIF10 are rejected before image compute or encoder loading.

Native segmentation creates temporary per-box masks even for a union result.
A request-local CLIP wrapper uses native `parse_shapes` immediately after native
decode to bound the answer to 131072 characters, 64 boxes and 67108864 temporary
mask elements **before** that allocation. Invalid/nonfinite/out-of-frame boxes
and invalid native masks fail explicitly. Narrow the query or crop the source
when its instance masks exceed the budget. No native algorithm or global parser
is replaced.

Detection and overlays preserve source alpha. Cutouts use the existing native
segmentation renderer to multiply original alpha by the soft native union mask.
Grayscale output contains the native scalar mask. Overlays use native
`ImageCompositeMasked` with green at 35% mask strength. PNG retains the ordinary
8-bit representation; EXR follows the existing floating output contract. An empty
native answer yields no invented boxes and a zero matte.

## API and replay

`POST /v1/images/generations` accepts:

```json
{
  "controlnet": { "type": "wevisone-matte" },
  "input_image": "BASE64_SOURCE_IMAGE",
  "wevisone_model": "wevisone_vl_2b_int8_convrot.safetensors",
  "wevisone_query": "person",
  "wevisone_output": "cutout",
  "wevisone_max_length": 4096,
  "n": 1,
  "size": "auto"
}
```

Use `wevisone-detect` for annotated detection. Matting output is `cutout`, `mask`
or `overlay`. An explicit size must match the oriented source. Generation tensor
inputs/exports, masks, references, borders and upscaling do not compose with this
request. `/images/edits` also accepts the existing single multipart image upload;
encode `controlnet` and `wevisone_max_length` as JSON form fields.

The ordinary response `data` contains exactly one generated image. Its bounded
`wevisone` object additionally supplies native boxes in source pixel coordinates,
points, raw answer, operation/options, source dimensions and public model identity.
Download URLs use the existing owned capability transport. PNG metadata and Library
carry the same provenance without private plans, absolute model directories or
fictitious generation settings. **Reuse full setup** restores the operation and
options, clears generation tensor controls, and requires the original source and
encoder to be selected again. Startup settings save scalar intent only.

## Evidence boundary

The public native example uses CLIPLoader `stable_diffusion` / `default` and
WeVisOneDetect. The compact official candidate is
`Comfy-Org/WeVisOne-VL@135355350443103e34089dd39264fe0651691929`, file
`text_encoders/wevisone_vl_2b_int8_convrot.safetensors`, 2,890,564,784 bytes,
SHA256 `88f3b1be21ff0828f20f0a711a14e00c55f401bceafad3c8943b7ce2c577a03e`.
See [WeVisOne acquisition](native-wevisone-acquisition.md) for the reviewed 2B
source and Models Hub download route. Acquisition/catalogue publication and
exact-file qualification are separate.

Guarded CPU fixtures execute the actual pinned node, parser, drawer and compositor
bodies with tiny inert CLIP responses. They exercise capture, alpha/geometry,
allocation refusal, cancellation, file leases/cache, installed activation, real
Gradio callbacks, actual API/worker/Library callers and native TE header identities.
These do not qualify real-model answers, rendered browser output, performance,
GPU/device operation or visual quality.

Source: [native WeVisOne node](https://github.com/Comfy-Org/ComfyUI/blob/989d93d0d968e704c589ce0146bb18c77f6b50cc/comfy_extras/nodes_wevisone.py),
[native model/tokenizer](https://github.com/Comfy-Org/ComfyUI/blob/989d93d0d968e704c589ce0146bb18c77f6b50cc/comfy/text_encoders/wevisone.py),
[public native example](https://github.com/user-attachments/files/34577573/WeVisOne-VL_example.json).
