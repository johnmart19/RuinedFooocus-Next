# Native special image workflows

Special image checkpoints need their pinned native conditioning or stage caller;
their shared runtime class does not establish ordinary text-to-image behavior.
The native source is ComfyUI `fca60fe70e5b84c5c6b058263653d15cb3401e44`.

## Flux inpainting

[Owned result-mask handoffs](image-workflows.md#selection-masks-and-inpainting)
feed this same conditioner with an exact source/frame-bound selection mask.
They retain explicit inversion, denoise, negatives and LoRAs, and do not change
the native mask growth, concat conditioning or original latent/noise-mask rules.

The native `FluxInpaint` configuration uses the `Flux` runtime with 96 input
channels. RF identifies the initialized native configuration, rather than a
filename. In Image tools, enable **Inpainting**, put the source in **Main** and
paint a mask. A missing source or painted mask is an error; this variant does
not silently run an ordinary text-to-image or image-to-image request.

The existing source/mask framing, denoise and mask expansion controls apply.
RF calls native `GrowMask` for pixel expansion, then
`nodes.InpaintModelConditioning.encode`. That node supplies masked-source
`concat_latent_image` and `concat_mask` to positive and negative conditioning,
and returns the original source latent plus the sampling mask. Prompt scheduling
and Flux guidance remain in the existing sampler caller. Ordinary Flux does
not gain this mask capability, and bundled SDXL ControlNet weights remain
restricted to SDXL. The existing SDXL inpaint caller is preserved.

Component selection reuses the Flux encoder/VAE roles, captured before enqueue.
The adapter does no acquisition or model loading and adds no GPU parallelism.
CPU tests execute the pinned native mask/conditioning methods with a weightless
encoder and exercise the real UI and pipeline callers. Actual model execution,
image quality and device stability require separate source/model-bound evidence.

## Canvas extension

With an SDXL checkpoint or native Flux Inpaint, upload the source to **Main**
and open **Image tools → Inpainting → Extend canvas**. Enter the border size
in pixels on each side and describe the new scene in the prompt. With all sides
at zero, inpainting uses the painted mask. With any positive border, the
extension uses its generated mask and sizes the canvas from the source plus
the four borders. Each side accepts up to 1024 added pixels and the resulting
canvas can be up to 4096 pixels wide or tall.

RF calls the pinned ComfyUI `ImagePadForOutpaint.expand_image` to construct the
RGB image and border mask, including the optional edge blend. It aligns the
pair for VAE encoding, uses the existing SDXL or Flux Inpaint conditioning
path, then removes the alignment padding and composites the unmasked source
pixels back in. The output PNG records the geometry and effective edge blend.
Automatic upscaling, if selected, runs on the completed image.

## Qwen Image 2.1 localized edits and background removal

With Qwen Image 2.1 selected, open the source in **Main**, enable **Image tools →
Localized edit**, and paint a red stroke where the change should happen. Keep the
stroke inside the intended region and name it in the prompt, for example: “Make
the jacket in the red area blue.”

- **Paint guide** sends the marked image as `<image1>`. Qwen can change other parts
  of the image as well, so it can adjust details around the painted area.
- **Mask** sends the same marked reference and composites the result over the
  original inside the brush stroke. Source RGBA pixels outside the stroke are
  kept exactly at source size. If you choose Pad or Crop, this applies to the
  resulting framed source. Optional output upscaling runs afterward.

Both modes can use additional references as `<image2>` onward. Preserve framing
keeps Main's original dimensions, subject to the native reference size limit
(16 megapixels and 4096 pixels per side); Pad and Crop use the selected size.
The reference is aligned to a 32-pixel grid for sampling, and that padding is
removed after decoding. The native `TextEncodeQwenImage21` node uses the first
reference to size its target latent. It has no separate mask input or inpaint
denoise slider. The red paint technique follows
[ComfyUI's Qwen Image 2.1 edit guide](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1#prompting-for-image-edit);
Mask adds the application-side composite.

PNG metadata records the applied Paint guide or Mask mode. **Library → Reuse
setup** restores that choice, with source references and the mask cleared; add
the source and paint again before enabling Localized edit. Earlier files that
recorded the mode only inside their image geometry are supported as well.

For Qwen's own transparent background removal, turn off **Localized edit**,
select **Native image edit**, and upload the source to **Image tools**. Use the
instruction `Remove the background, and output a PNG image` and save the output
as PNG. This calls the same Qwen edit workflow and returns native RGBA. The
standalone **Remove background** tool uses a separate model.

## Masked editing and canvas extension through the API

Both operations use the native image engine. SDXL and Flux Inpaint support both;
Qwen Image 2.1 supports localized edits with a mask. Use an `image/` ID from
`/v1/models` and describe the edit or
extension in `prompt`. The existing unmasked edit operation still selects
native edit or image-to-image based on the checkpoint.

- `POST /v1/images/edits` accepts one multipart `image` or `image[]` file and an
  optional `mask` file. For canvas extension, send a `borders` form field
  containing JSON instead of a mask file.
- `POST /v1/images/generations` and `POST /tools/image_generation` accept JSON.
  Supply `input_image` as base64, an image data URL, or a current image URL from
  this API. Use `mask` for an inline base64 PNG or a PNG data URL, or supply a
  `borders` object such as `{"left": 128, "bottom": 96, "feather": 16}`.

The PNG mask needs alpha and must match the source dimensions after EXIF
orientation. Transparent mask pixels are editable, opaque pixels leave that
area outside the mask, and partial transparency gives intermediate coverage.
With SDXL or Flux Inpaint, `inpaint_mask_grow` adds 0–128 pixels to the mask and
defaults to 20. `inpaint_denoise` is greater than 0 and at most 1, with a default
of 1. These controls use the same numeric fields in JSON or multipart form data.
For Qwen, `qwen_local_edit_mode` accepts `"mask"` (the API default) or `"paint"`
with a PNG mask. Set this field explicitly to also pass `reference_images`. Qwen
does not accept the SDXL/Flux denoise and mask expansion fields.

For masked edits, omitted `size` or `"auto"` preserves the oriented source size.
An explicit `size` gives the exact output dimensions with pad framing by default.
Set `image_resize_mode` to `"crop"` for crop framing. Masked output can be
64–2048 pixels per side. Odd dimensions use hidden alignment padding that the
native worker removes after decoding.

Each canvas border is a whole number from 0–1024 pixels. At least one side must
be positive. Optional `feather` accepts 0–128 pixels and defaults to 32, subject
to the native limit for small sources. Canvas extension always returns the
source size plus its borders, up to 4096 pixels per side. If supplied, `size`
must agree with that result. The native compositor preserves source pixels
outside the blend region.

Choose one operation per request. Masks and canvas borders are mutually
exclusive with other image tools and reference adapters. Multipart uploads allow
up to 21,000,000 bytes per image and mask. The decoded source and mask are each
limited to 16,777,216 pixels. The API closes multipart uploads before queueing
and passes the decoded source and mask as in-memory job data. The output
metadata records render settings and geometry.

`test_api_native_inpaint.py` exercises the public JSON and multipart routes,
mask polarity, EXIF sizing, geometry and denoise capture. The pinned native
geometry cases also run in `test_image_inpaint_geometry.py` with a weightless
encoder.

## Qwen Image 2.1 reference editing

Select a Qwen Image 2.1 checkpoint with its Qwen3-VL 8B encoder and 64-channel,
16× RGBA VAE. In **Image tools → Native image edit**, upload the primary image
and add any extra images in order. Refer to them with `<image1>`, `<image2>`,
and so on in the prompt. Qwen documents up to ten references, and the app
accepts up to sixteen through its pinned native node. The primary supplies
the target geometry. Text-to-image uses no references. To generate a
transparent PNG, ask for an RGBA image with a transparent background.

For **Model Defaults**, use **Inspect** once to verify a registered base or
Turbo checkpoint, including the listed Q4_K_M GGUF. The base uses the
[ComfyUI base workflow](https://comfy.org/hub/workflows/qwen-image-2-1-image-edit-5fbe136ba6f9/)
settings of 25 steps, CFG 1, Euler, and the simple scheduler. The pinned native
model supplies its sampling shift, and the edit keeps its source framing.
The full Turbo checkpoint uses the [ComfyUI Turbo graph](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_turbo_t2i.json)
with eight Euler steps, CFG 1 and its fixed sigma schedule. Choose **Own settings**
to change the sampling recipe. **Image guidance → Qwen sampling schedule** offers
Auto, Standard and Turbo 8-step. Auto follows Model Defaults. Standard uses the
selected scheduler. Turbo 8-step selects the published fixed schedule and requires
8 steps, CFG 1, Euler and Simple, including with a Turbo LoRA under Own settings.
Library restores this control when reusing a saved Turbo image's settings.
The LoRA browser can apply the Turbo adapter or compatible edit adapters.
Choose the 1024 × 1024 ComfyUI default
or a larger ratio from the [Qwen model card](https://huggingface.co/Qwen/Qwen-Image-2.1).

`POST /v1/images/generations` and `/tools/image_generation` accept the primary
as `input_image` and the extra images in `reference_images`. The multipart
`/v1/images/edits` endpoint accepts one `image` file and a JSON
`reference_images` form field. Supplying a primary image selects native edit
by default for Qwen Image 2.1. Explicit `controlnet.type="img2img"` still
selects denoise-based image-to-image with only the primary image.

RF preserves alpha through request framing. The pinned `TextEncodeQwenImage21`
node composites the reference over white for its vision tower and passes the
RGBA pixels to the VAE. It attaches ordered reference latents and image slots
to both positive and negative conditioning. The request reuses encoded VAE
references across scheduled prompts. [Pinned Qwen conditioning node](https://github.com/comfyanonymous/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy_extras/nodes_qwen.py#L111)

The worker aligns the primary to the native 32-pixel canvas. It samples the
node's zero latent on the first-reference grid, decodes through the matching
RGBA VAE, and removes hidden alignment padding. Preserve framing fits the
source to the working pixel budget. Pad or Crop framing gives the requested
output dimensions. Each reference is limited to 16 megapixels before framing
and 4096 pixels per side on the working canvas. Older Qwen Image Edit and
Qwen Image Layered use separate encoder and latent contracts.

Guarded CPU fixtures exercise the pinned node with 0–16 references, native
alpha treatment, scheduled prompts and exact frame restoration. Windows ROCm
checks on an RX 7900 XTX exercised base and Turbo INT8, and Q4_K_M GGUF. A mask
preserves pixels outside its painted region, so leave enough space around a new
object for its full silhouette and shadow. Large reference sets can run even
when the generated image omits or repeats an object.

## Qwen Image 2.1 prompt enhancement

With a Qwen Image 2.1 checkpoint selected, open **Prompt tools → Qwen Image 2.1
prompt enhancement** and preview a rewrite. Text-to-image uses PE-T2I. Native
image edit uses PE-I2I with the primary image and up to nine additional
references in upload order. The preview shows the rewritten prompt and Qwen's
suggested ratio or reference image. Apply the rewrite explicitly, and use
**Size and upscale** to select the desired framing and dimensions.

The first preview can download the 9.47 GB Comfy-Org INT8 ConvRot rewriter
and the matching Qwen system prompt. The worker verifies new downloads,
uses the pinned ComfyUI Generate Text node with reasoning enabled, and
releases native models when the rewrite finishes. The default 4096-token limit
matches the ComfyUI templates. Raise it in Prompt tools if the model stops
before producing the final rewrite.
[PE-T2I model card](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I)
[PE-I2I model card](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I)
[ComfyUI model files](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/aacf31a6d48e707f352ecba714cee0680da3479b/text_encoders)

## Qwen Image 2.1 Fun ControlNet Union

Select a Qwen Image 2.1 checkpoint and choose **Create → Image tools → Qwen
Image 2.1 Union**. Select the Union model patch and guide type. Upload a prepared
Canny, Depth, HED, Lineart, MLSD, Pose, or Scribble map. Grayscale also accepts a
photo and converts it to luminance. Strength and the start and end fractions
control how much of sampling uses the patch.

For inpainting, add a source and a same-size mask under **Union inpainting**.
White marks the region to redraw and black provides the context to keep. Choose
**Inpaint only** for the source and mask, or keep a guide type selected to apply
both conditions. The patch resizes the inputs to the selected generation canvas.

The default patch is the Comfy-Org INT8 ConvRot safetensors file in
`models/model_patches`. You can also select the published BF16 patch or another
compatible local safetensors file in that folder. The path is configurable in
**Settings → Profiles & storage → Paths**. Published patches can be prepared at
job submission. Offline jobs require the selected patch to be present locally.
Both published patches use the Qwen Research License. [Comfy-Org files and model card](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) · [Alibaba-PAI Union model card](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union)

For JSON image requests, set `controlnet.type` to `qwen-union` and choose a
`mode` from `canny`, `depth`, `grayscale`, `hed`, `lineart`, `mlsd`, `pose`,
`scribble`, or `inpaint`. The optional `model` is a relative patch filename.
Use `qwen_union_guide` or `input_image` for a structural guide. Use
`qwen_union_source` and `qwen_union_mask` together for inpainting.

Output PNG metadata records the selected patch filename and local file signature,
guide type, strength, and sampling interval. **Library → Use setup** restores
those controls. Supply the guide, source, and mask again from their separate
files. Reselecting a patch binds its current file identity.

The worker captures a typed safetensors asset before compute and holds its
file lease through `ModelPatchLoader`. The pinned `ZImageFunControlnet` node
dispatches to `QwenImage21FunControlPatch` and applies it to the request's
sampling model. The native patch builds 129 channels from the control latent,
keep mask, and masked-source latent, then injects into every second base block.
[Pinned native model patch code](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy_extras/nodes_model_patch.py#L630)

## JoyAI Image Edit

Choose the JoyAI Image Edit checkpoint with the dedicated JoyImage Qwen3-VL 8B
encoder and Wan 2.1 VAE. In **Image tools → Native image edit**, upload one
source and describe the change. The pinned JoyImage node makes the visual and
VAE reference conditioning for both prompts. Sampling starts from a separate
empty latent at the framed output size.

To match the official INT8 ConvRot workflow's sampling, choose **Own settings**
with 40 steps, CFG 4, Euler, and the normal scheduler. In Image guidance, set
CFG norm strength to 1 and enable pre-CFG. The official graph uses one
reference; this route also takes one. Explicit Image to image uses the
separate denoise-based path.
[Official JoyAI Image Edit workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_joyai_image_edit.json)

A bounded Windows RX 7900 XTX run returned a 384 × 512 RGB PNG with the
original dimensions and request metadata. The six-step result changed the
wall layout and placed the oval off-center. The separate 40-step request hit
a 180-second client timeout at 20 steps. Output quality and performance at the
published settings remain open.

## Standalone SDXL refiner

Select an SDXL refiner checkpoint as the image model. It uses its native
`SDXLRefiner` runtime, a single CLIP-G encoder and the SDXL VAE; bundled components
retain their existing protection and external choices are captured before enqueue.
Choose **Own settings** with the exact checkpoint's sampling recipe, or supply
a complete published/saved per-checkpoint recipe for **Model Defaults**.

`CLIPTextEncodeSDXLRefiner.execute` supplies the actual request width and height,
with the native positive aesthetic default 6 and negative default 2.5. The text
cache includes that geometry, so an unchanged prompt at a different size cannot
reuse another request's dimensions. Scheduled positive prompts use the same
geometry; CLIP layer selection, embeddings and LoRA warmup keep their existing
callers. Native four-channel latents, sampling and SDXL VAE decoding are reused.
The native refiner label overrides stale SDXL family metadata for request routing
without rewriting the publisher's cached title or provenance.

This is a standalone checkpoint workflow. Explicit two-checkpoint generation
uses the separate [SDXL base/refiner handoff](native-sdxl-refiner.md) panel and
retains a noisy latent across the native schedule split.
CPU fixtures exercise the native text node/loader branch and actual request/cache
callers without model weights. Real model execution and image quality remain
separate acceptance gates.

## Instruct-pix2pix image editing

Native `SD15_instructpix2pix` and `SDXL_instructpix2pix` checkpoints have their
own runtime/component routes. The SD1 variant uses CLIP-L and the SD VAE;
the SDXL variant uses CLIP-L/G and the SDXL VAE. Select **Native image edit**,
upload the source in **Image tools**, and describe the edit in the prompt.
The API chooses this native edit mode when an input image is supplied for one
of these checkpoint families and no tool is named.

These checkpoints require the source-image edit operation. The UI excludes
ordinary image-to-image and SDXL ControlNet choices; a previous image-to-image
selection changes to native edit when switching to an instruct checkpoint.
Missing sources, other sampling modes and masked inpainting are refused.
Standalone utilities remain separate operations.

Checkpoint changes update the available custom tool types and their selected
default together. Dependent Gradio events can still carry the previous model's
type while that update reaches the browser. The image tool type dropdown maps
only a known, newly unsupported tool type to the current supported default
before native dropdown preprocessing. Supported selections remain intact;
unknown values still fail native validation and preset saving still checks the
selected checkpoint's capabilities. This prevents stale `Img2img`/`Edit` errors
when switching ordinary, instruct and SD4x models without enabling custom values.
`test_image_tool_transitions.py` exercises actual Gradio session updates and queued
preprocessing in both directions; model execution is not part of that CPU check.

RF uses the existing source framing policy, then calls native
`InstructPixToPixConditioning.execute`. It attaches the encoded source to both
positive and negative conditioning, including scheduled prompts, and returns the
native zero sampler latent. It does not substitute ordinary image-to-image
initialization or copy conditioning math. SDXL EDM/CosXL source scaling remains
inside the native runtime. The existing sampler CFG is retained; this change
adds no separate image-CFG control or recommendation.

Use **Own settings** or a complete exact-checkpoint published/saved recipe.
Native family detection overrides stale generic SD/SDXL routing labels without
rewriting publisher metadata. CPU checks cover native node behavior, UI/API
selection, source/mask refusal, framing and actual sampler handoff. Real model
execution, quality and device stability remain unverified by those fixtures.

UnCLIP, X4 upscaler, Cascade, Hunyuan image refiner and PiD have distinct native
source routes with their own conditioning, codecs and stages. See
[native model coverage](native-model-coverage.md) and the
[finite implementation/acceptance delta](image-chat-scope.md) for the current
consumers and exact-artifact limits; a generic alias does not establish them.

At the 0.45 pin, [MiniMax H3 Image](native-minimax-image.md) connects its distinct
PDD generation/reference-edit graph. [WeVisOne](native-wevisone.md) is a separate
standalone detection/matting tool with an image-utility encoder. Their source,
component and output contracts do not inherit neighboring family qualification.
