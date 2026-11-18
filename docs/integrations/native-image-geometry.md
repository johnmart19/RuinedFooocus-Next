# Native still-image geometry

The `image-transform` still utility implements **Transform image** as one
operation family: resize, crop, crop to selection mask, pad, rotate and flip. It requires one source and
no generation model, prompt, sampling recipe or download. Its caller uses the
existing serialized image utility worker, output writer and cancellation policy.
The adapter is in `features/image_processing/image_geometry_tools.py`.

`transform_operation` chooses the contextual controls. Resize defaults to zero
width and height, preserving the original frame. One nonzero dimension derives
the other with the original proportions. Two nonzero dimensions default to
**contain**, producing the largest proportion-preserving image inside that box.
Stretch and centered crop require the corresponding explicit `transform_fit`.
Crop uses `transform_x`, `transform_y`, width and height; a zero crop dimension
uses the remaining source extent. The full crop must stay inside the source.
Rotation uses native quarter turns, and flip selects horizontal or vertical.

`transform_operation="crop-to-mask"` also accepts the existing `mask_image`
input. The selection must be an opaque grayscale image in exactly the oriented
source frame. Nonzero mask samples select pixels; the smallest rectangle
containing every selected pixel determines the output dimensions. An empty mask,
colored or translucent mask, invalid sample range or different frame is rejected
before processing. No resize, threshold, inversion, square framing or padding is
implicit. The whole source rectangle is cropped, including its original alpha
and pixels outside a nonrectangular selection. The mask does not replace alpha.

Padding accepts explicit left, top, right and bottom pixel counts. Its default
transparent style retains source pixels and alpha and adds zero-alpha borders.
Color style uses the selected opaque RGB border, while mirror style reflects
all native channels, including alpha. Optional native feathering blends the
source edge into the selected fill; it is an explicit change to edge pixels.
Source alpha and color channels undergo the same native resize geometry.

## Native computation and immutable requests

The reviewed source is ComfyUI
`989d93d0d968e704c589ce0146bb18c77f6b50cc`. `geometry_native.py` selects the
unchanged pure bodies of native `ResizeImageMaskNode.execute`,
`BoundingBox.execute`, `ImageCropV2.execute`, `PadImageNode.execute`, `ImageRotate.execute` and
`ImageFlip.execute`, together with their native geometry and color helpers.
The bridge follows the existing composition adapter pattern. It avoids importing
the native server and model loaders. The native crop's graph
preview transport is suppressed because RF owns artifact publication; its pixel
body is unchanged. No separate resize, reflection, cropping or feathering math
is implemented in RF.

Selection cropping uses the captured mask's nonzero bounds to construct the native
`PrimitiveBoundingBox`, then runs `ImageCropV2` on all source channels. The pinned
`ImageCropToMask` is deliberately excluded from this precision-preserving route:
its documented purpose is fixed-frame 3D input preparation, and it discards alpha,
clamps/composites RGB, pads out-of-bounds bounds and forces a byte-based Lanczos
resize. Those semantics do not implement an exact 2D selection crop.

Capture freezes oriented source bytes, dtype/channels, color space, source
digest, validated scalars, final output dimensions and canonical native source
signatures. Normalized export format, depth, color space and quality also belong
to that immutable request; worker capture, processing, metadata and replay reuse
the frozen choice. Processing creates float32 tensors in inference mode. Later
source/control changes cannot alter that request. A native source change before
or during processing invalidates it. Cancellation is checked before and after
the native operation; a single tensor operation finishes between checks.
Inputs and outputs are limited to 16,777,216 pixels and 16384 pixels per side.
Output bounds are checked before allocating the transformed image.
Selection cropping also freezes mask bytes, dtype, digest and the exact crop
rectangle. Its metadata records mask identity and bounds; replay clears the mask
and requires a fresh exact-frame selection.

Compute selection happens in the serialized worker through the shared
`utility_device.run_utility` policy. Auto uses the configured native Comfy compute
device; an explicitly applied CPU mode stays on CPU without accelerator probes.
Geometry tensors move to that device before native Torch operations. Source/mask
capture, bounding-box discovery and final image encoding remain host operations.
A device failure can restart the entire operation on CPU from the frozen source;
the native pixel algorithms and float32 precision are unchanged. Validation and
the final host transfer run inside the attempt so deferred device failures also
reach that fallback boundary. Cancellation is checked before any retry and before
metadata publication. The recorded `image_utility_source.compute` describes the
actual device, backend and fallback reason. Replay does not force an old device.

The adapter accepts byte PIL images and internal NumPy/native `ImportedImage`
pixels with uint8, uint16, float16 or float32 samples. It does not accept arbitrary
paths. Native precision originals must reach capture without an earlier byte
conversion: supplied working-copy provenance is rejected explicitly. Integer
samples normalize to float32 without an 8-bit intermediate. RGB, RGBA and one
grayscale channel are supported; ordinary byte PIL inputs use oriented RGBA.
There is no ICC conversion or implicit transfer/color-space conversion.

**Straighten** rotates by any angle from −180° to 180° (positive is clockwise) with
bicubic sampling and crops to the largest upright rectangle inside the rotated
image, so the result has no empty corners and gains no alpha. It is application
arithmetic over `torch.grid_sample`, not a native node.

Nearest-exact, bilinear, area and bicubic use native interpolation.
**AMD FSR 1** (`amd-fsr1`) enlarges with AMD FidelityFX Super Resolution 1: the
edge-adaptive EASU upsampler in passes of at most 2× per axis, then RCAS
sharpening at 0.2 stops, with bicubic alpha. `amd_fsr1.py` is a tensor port of the
MIT-licensed `ffx_fsr1.h` from the AMD FSR SDK 2.4.0 and runs on any device,
including CPU; it loads no AMD binary. It accepts 8-bit or 16-bit sRGB sources
and refuses a target smaller than the source in either dimension before queueing.
With **center-crop** it enlarges to cover the target and trims the excess. The
SDK's temporal upscalers (FSR 2, 3 and 4), frame generation and denoisers need
motion vectors, depth and a swap chain, so they do not apply to a still image. The pinned
native Lanczos helper converts through Pillow uint8 and is intentionally omitted
from this precision-preserving surface. The adapter does not substitute a new
Lanczos implementation.

Precision inputs require PNG16 or EXR output. Output color space must match the
captured source; linear/HDR inputs require EXR, and out-of-range floating samples
require EXR. Marked Editor requests derive this color space from the captured
source while retaining the selected supported container/precision and the global
format draft. Direct API requests still reject a mismatched color space.
Scene-linear Rec.2020 EXR keeps its samples and native Rec.2020 chromaticities;
the HDR transfer function is not applied again. The existing encoder owns transfer tags and quantization. Native
AVIF cannot export RGBA. Original uploads remain unchanged. Utility metadata
records operation/scalars, source digest, dtype, dimensions, color/alpha policy
and native revision; it contains no local paths or invented generation recipe.
Replay restores scalars and asks for a fresh source.

## Caller contract and verification

The adapter exports `KINDS`, `LABELS`, `DEFAULTS`, `FIELDS`, `capture_assets`,
`worker_data`, `process`, `output_metadata` and `replay_values`, matching the
existing still utility owners. `capture_assets` accepts the normal `input_image`
and returns an immutable request with no preparation assets. Its worker data
retains one output with the actual transformed dimensions and strips sampling
and model intent. The bounded source-generation recipe is retained separately
for derived-image lineage; source handles, paths and uploaded pixels are omitted
from that recipe. UI/API admission and rendered user reachability are separate
caller-owned integration checks.

```sh
python tools/check_cpu.py test_image_geometry_tools.py test_image_crop_to_mask.py test_geometry_mask_devices.py test_image_geometry.py test_image_filters.py test_image_composition.py
```

The new tests execute actual pinned native pure pixel bodies, rather than only
mocking output. They cover asymmetric RGBA crop/rotation/flip, proportion-preserving
and explicit changed-aspect resize, all padding styles, uint16 and float precision,
source immutability, strict bounds, color/format rejection, cancellation, source
changes, worker metadata and replay. These are source/CPU contracts; rendered
browser behavior, real codec execution, image quality, throughput and GPU
stability require separate evidence.
The selection-crop checks additionally cover nonrectangular and one-pixel edge
selections, unchanged transparent alpha, uint16 and signed/HDR float samples,
immutable mask capture, strict frame/range rejection and late cancellation.
