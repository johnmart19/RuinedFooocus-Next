# Image adjustment integration

Image tools → **Adjust image** provides Transform, Mask and Color groups with
operation-specific controls. The Image workspace has the same **Adjust image**
action and uses the current owned artifact. Selecting that action opens the
existing More tools tab and its selected
adjustment group without changing source images, prompts or operation settings.
The existing Run/Stop flow serializes these operations on the image worker.
No checkpoint, text encoder, sampler,
model download or learned component is needed.

The native contracts live in [geometry](native-image-geometry.md),
[masks](native-image-mask-tools.md) and [color](native-image-color-tools.md).
`ui_image_adjustments.py` registers scalar controls and original-file inputs;
`still_image_utilities.py` dispatches the three immutable requests. The UI alias
and selected native kind are cross-checked at capture. Inactive uploads stay in
the browser draft and are omitted from that operation's captured inputs.

The image API uses `controlnet.type` values `image-transform`, `image-mask` and
`image-color`. Their prefixed scalar fields and `mask_image`, `mask_reference`
and `color_reference` inline images belong only to that group. One source and
one output are required. Explicit `size` must equal the captured native output,
including crop, rotation and padding. Generation references, borders, recipes
and post-upscaling are excluded. PNG16/EXR sources use the shared native reader;
Editor handles read the original owned artifact directly without an 8-bit preview
or an inline-image re-encoding round trip.
Adjustment capture decodes that original once and derives its source details
from those native samples, without preparing an unused sRGB8 working copy.
Precision exports remain explicit and frozen with the request.

When Editor's file-output controls still have their untouched defaults, adjustment
requests choose a precision-preserving output automatically: PNG16 for native
uint16 sRGB samples, or floating-point EXR for float/linear/HDR samples. Scalar
masks use their existing sRGB output contract; SDR-to-HDR selects EXR/HDR. This
choice belongs to the request and does not change the shared output controls.
Non-default options and choices made in the output controls remain authoritative.
Requests without Editor's output-intent marker retain the existing strict API
contract, including rejection of an incompatible precision export.

The additional operations stay in these existing surfaces. **Transform → Crop to
selection** uses an exact-frame mask and crops to its nonzero bounding box without
resampling. **Color** also offers text overlay, RGB morphology and SDR-to-HDR.
Text requires an 8-bit sRGB source and adds glyph opacity; morphology preserves
source alpha. HDR conversion requires an explicit floating point EXR/HDR choice
in its contextual output selector, which mirrors the existing file settings and
does not change them merely by browsing operations.

The existing MoGe panel offers perspective depth or surface normals, plus
360-degree panorama depth. Panorama normals are not supported. The existing Face
panel offers native region masks with all regions or explicit custom regions;
its registered first-use bundle uses the ordinary preparation flow. Proven face
mask results can use the same source-preserving selection handoff below.

Mask results remain image artifacts until **Use result as selection mask** is
chosen. This validates the current owner/revision, proven mask operation, exact
source frame and grayscale channels. It attaches a lossless RGB8-to-L8 selection
and restores the retained source in one manifest transaction. The original RGB,
alpha, metadata and mask-result history remain intact; Redo reopens the mask.
Inpainting automatically prepares a request-local sRGB8 copy of the retained
source. Proven precision mask results normalize their scalar coverage to an L8
selection; they do not pass through a color transfer curve or source alpha.
Original precision images and mask artifacts stay unchanged, and native
adjustments/export continue to use original samples rather than display previews.

Blend uses the native RGB clamp to 0–1. Transfer requires encoded sRGB in range,
and its Histogram choice is labeled **Histogram (256-bin)**. Quantize explicitly
uses the native RGB8 palette conversion. These are explicit effect policies;
they do not imply unrestricted HDR grading or model quality qualification.

Replay restores scalar intent and export format, resets all uploaded images and
requires a fresh source. It does not replay generation models or overwrite
prompt drafts. Cooperative cancellation and stale publication use the existing
owned worker/editor boundaries. Geometry, alpha and precision changes record
their explicit execution policies before history publication.

Focused checks (project interpreter):

```sh
python tools/check_cpu.py test_image_adjustment_integration.py test_editor_workspace_actions.py test_editor_workspace_events.py
```

Source/CPU checks exercise real pinned deterministic processors and Gradio
events. Browser delivery is a separate root-owned check. No GPU inference,
performance or learned-model quality claim is made by these checks.
