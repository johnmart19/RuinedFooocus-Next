# Native mask tools

The **Mask** group of **Adjust image** edits a selection on the main image's
oriented frame. Upload a grayscale mask for grow/shrink, feather, combination,
threshold or inversion. Channel and color extraction use the main image itself.
White selects, black protects, and intermediate grayscale values remain soft.
There is no automatic mask crop or resize. Combine masks requires two masks with
the same dimensions as the main image.

Grow accepts negative values to shrink and has a tapered-corner option. Feather
attenuates the four image-frame edges independently; it is not a general blur of
the selection boundary. Combination exposes native multiply/add/subtract and
binary and/or/xor. Binary operations round the masks as defined upstream.
Threshold selects values strictly greater than the chosen threshold. Input
inversion applies to the primary mask before the selected operation.

Results are exported as opaque grayscale RGB images. **Use result as inpaint
mask** can reuse an 8-bit PNG result against its matching 8-bit sRGB source in
the Editor. The canonical Editor handoff currently rejects precision PNG16/EXR
masks or sources; retain the original and make an explicit working copy when an
8-bit Editor selection is needed. PNG16/EXR mask exports retain their precision
for other consumers. Color-key extraction uses native rounded RGB8 equality;
it ignores source alpha. Extracting the alpha channel is a separate operation.

## Source and request contract

`ruinedfooocus/features/image_processing/image_mask_tools.py` owns `image-mask`
and frozen `MaskRequest` records. It reuses geometry's `capture_source` and
`FrozenPixels`; uint16 and floating samples are not routed through an 8-bit
display image. Captured source/mask/reference pixels, typed scalar parameters,
native file signatures and export settings are immutable. Worker transport
keeps only the request, lifecycle tickets, output settings and required progress
fields. Later upload, scalar or export-setting changes do not alter a queued
request. Replay restores scalar/output choices and clears all image inputs.
The bounded, detached source-generation recipe survives worker transport so the
shared output writer can attach it to the derived result; source images and
source-path authority are excluded from that provenance.

`mask_native.py` reads the pinned `comfy_extras/nodes_mask.py` bodies, verifies
source signatures before and after work, and compiles only the selected pure
methods into a request-local namespace. Algorithms are native `GrowMask`,
`FeatherMask`, `MaskComposite`, `ImageToMask`, `ImageColorToMask`,
`ThresholdMask`, `InvertMask` and `MaskToImage`. Grow checks cancellation between
native one-pixel passes. The bridge imports no native server or model loaders.
Cancellation or invalid output does not publish metadata.

Worker execution uses the shared `utility_device.run_utility` policy. Auto moves
Torch selection batches to the configured native Comfy compute device; explicit
CPU mode remains authoritative without accelerator probes. Channel/color
extraction, feathering, combination, threshold and inversion follow that selected
tensor device. Native `GrowMask` calls NumPy/SciPy and requires host tensors, so
its capability is explicitly reported as CPU-only without probing or pretending
to use a GPU kernel. Final encoding also remains on the host.

Each eligible failed device attempt can restart on CPU from immutable captured
pixels; the complete native operation, validation and final host transfer belong
to that attempt. Cancellation is checked before retry and publication. Metadata's
`image_utility_source.compute` records actual device/backend, host-only capability
or genuine fallback reason. Source capture and replay do not select a device.

Every input mask must be grayscale, finite, in [0, 1], and in the main image's
frame. RGB masks must contain equal R/G/B samples; alpha is ignored rather than
silently replacing the grayscale selection. Precision inputs require PNG16 or
EXR, including a precision reference in a combination. Masks have scalar sRGB
export semantics without ICC conversion. Native still AVIF is excluded for
reusable masks. Metadata records input hashes/dtypes, frame, native operation,
selection semantics, output precision and explicit input inversion.

## Pinned API and verification

The source contract is ComfyUI
`989d93d0d968e704c589ce0146bb18c77f6b50cc`, inspected locally on 2026-11-12,
[native mask nodes](https://github.com/comfyanonymous/ComfyUI/blob/989d93d0d968e704c589ce0146bb18c77f6b50cc/comfy_extras/nodes_mask.py).
The adapter uses installed NumPy, Torch and SciPy; it adds no dependencies.

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_image_mask_tools.py test_geometry_mask_devices.py
```

The focused tests execute the actual pinned methods on small CPU images and
tensors, including soft growth/shrink, tapered corners, all six combinations,
frame-bound feathering, channel precision, RGB color keys, strict threshold,
alpha independence, immutable capture/export/replay, stale-source rejection,
invalid input/output and early/final cancellation. Browser behavior, GPU/model
execution, visual quality and large-image latency require separate evidence.
