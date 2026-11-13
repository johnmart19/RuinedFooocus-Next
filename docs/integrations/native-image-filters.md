# Native still-image filters

**Create → Image tools** offers **Blur image**, **Sharpen image**, **Adjust
brightness** and **Adjust contrast** as explicit operations. They are off until
selected. Add one source, set the contextual controls, then press Generate.
The existing serial image worker returns one source-sized RGBA result. A
generation checkpoint, prompt, LoRA, sampling recipe or model download is not
required. The operation ignores hidden generation preferences at the shared
capture boundary.

Blur and sharpen expose radius 1–31 and sigma 0.1–10. Sharpen also exposes
strength 0–5; zero preserves the original RGB values. Native reflect padding
requires the radius to be smaller than both source dimensions; capture rejects
an invalid radius instead of resizing. Brightness and contrast accept a factor
0–2; one preserves the original RGB values. Contrast is centered on native 0.5.

## Native contracts

Reviewed source is ComfyUI `fca60fe70e5b84c5c6b058263653d15cb3401e44`,
2026-11-11. `image_filters.py` calls native
`comfy_extras.nodes_post_processing.Blur.execute` (`ImageBlur`) and
`Sharpen.execute` (`ImageSharpen`), including native kernels, device selection
and intermediate-device return. It does not reproduce their filtering math.
Brightness and contrast call the pure tensor `_process` methods on
`comfy_extras.nodes_dataset.AdjustBrightnessNode` and `AdjustContrastNode`.
They do not invoke dataset traversal, file processing, augmentation, training or
video routines. The finite ledger overrides the image domain for these exact
two node IDs only; it retains the other dataset/training exclusions.

`still_image_utilities.py` dispatches captured still operations to their feature
owners. Existing depth/pose/albedo helpers keep their own kinds, models and
native contracts. The shared preparation/API/UI/worker hooks and existing serial
utility pipeline use the dispatcher; entering a filter clears an earlier native
geometry model cache. No filter model/cache is created.

Source capture reuses the histogram's bounded, EXIF-oriented immutable RGBA8
snapshot and dimension-aware pixel SHA-256. It accepts at most 16,777,216 pixels;
direct/API high-bit, float and CMYK sources are rejected. Existing Gradio source
uploads decode to RGBA8 before capture. Native processing receives RGB float32.
API admission checks native encoded-frame precision before Pillow can reduce a
color PNG16 or AVIF10 to RGB8; filter and curve requests share this check.
The original alpha bytes are restored, so filtering never blurs or sharpens
coverage. No ICC conversion or transfer-function linearization is performed.
PNG8 output rounds/clips native RGB, preserving original dimensions and alpha.
The existing precision export can retain the native float32 RGB and source alpha
in supported PNG16/EXR formats. Native still AVIF cannot preserve RGBA and is
rejected before utility compute; use PNG or EXR. Container/codec behavior retains that
exporter's existing qualification limits.

Cancellation is checked before and after every native call and before output.
A native convolution is atomic between checks. Non-finite values or unexpected
image dimensions fail without publishing output provenance. The worker writes
utility-only metadata: operation/node/pin, source digest and layout, dimensions,
decoded mode/profile/alpha policy and active parameters. It records no local
paths or fictitious generation-model/sampling recipe.

## API and replay

The existing JSON `/v1/images/generations` and multipart `/v1/images/edits`
endpoints accept `controlnet.type` values `image-blur`, `image-sharpen`,
`adjust-brightness` and `adjust-contrast`. Generation-model lookup is bypassed.
Supply one `input_image` or uploaded `image`/`image[]`, and optional top-level
`filter_radius`, `filter_sigma`, `filter_strength` or `filter_factor` as relevant.
Only one output is accepted; an explicit size must equal the oriented source.
Extra references, masks and canvas borders are rejected. Existing output format,
response format, authorization, bounded uploads and disconnect cancellation
remain in force.

```json
{
  "input_image": "data:image/png;base64,...",
  "controlnet": { "type": "adjust-contrast" },
  "filter_factor": 1.2,
  "response_format": "b64_json"
}
```

Library Apply/Reuse restores the named filter and active scalar values, clears
source uploads and model-component selectors, and disables inpainting, recipes,
reference adapters and automatic upscaling. It asks for a new source rather than
silently treating a filtered result as the original. The source SHA identifies
pixels but cannot reconstruct them. Existing generation setup is not resolved.

## Verification

`python tools/check_cpu.py test_image_filters.py test_depth_utilities.py
test_depth_utilities_ui.py test_image_histogram.py test_api_image_edits.py
test_utility_sampling_capture.py test_native_sampling_preparation_boundary.py`
checks actual pinned filter bodies with CPU device stand-ins, strict parameters,
source/alpha freezing, cancellation, native output errors, prior model cleanup,
JSON/multipart/API capture, real Gradio mount/events and the actual serial worker
PNG/metadata writer. The filter feature descriptor records the wider affected
caller cohort and independent prerequisite replay.

These are source/CPU contracts. They do not certify rendered browser layout,
production device execution, visual preference or performance. No GL module is
imported by this feature. Curves and named shader effects are separate operations.
