# Native image detection and segmentation

The native utility backend recognizes `RT_DETR_v4` for detection and `SAM3` /
`SAM31` for segmentation from the pinned ComfyUI classifier. It accepts
local safetensors in configured checkpoint/diffusion-model roots; no generation
checkpoint, VAE, sampling recipe or prompt helper is required. Point/box modes
need no text encoder. Text mode uses the selected SAM checkpoint's own paired
language encoder, not the generation encoder. File names
do not determine architecture. Unknown or wrong-family files fail before loading.

Explicit API actions can also select one compatible registered catalogue key or
filename. [First-use preparation](image-preparation.md) prepares that missing
file through the existing service and validates its native header before compute.
Unregistered custom paths need local installation. Choices, navigation and
previews never acquire a file; SAM text never acquires a separate text encoder.

API image requests use `controlnet.type: detect` or `segment`, `input_image` and
`native_vision_model` (a configured-root relative name or listed absolute path).
Detection accepts `vision_threshold` (0–1), `detect_class` (`all` or a native COCO
class), and `detect_max_count` (1–1000). Point segmentation accepts
`segment_positive_points` JSON, such as `[{"x":10,"y":20}]`, and
`segment_negative_points`. Coordinates address the oriented source pixels;
each list is limited to 256 points within the source bounds. Refinement is 0–5;
`segment_output` is `cutout` or `mask`. Each request processes one source image,
limited to 16 megapixels. Explicit generation recipes/reference adapters and
inpainting are rejected for these utilities. Saved point prompts never leak into
an API request for a different image.

The backend copies the oriented RGBA source and validates point JSON before
queueing. Captured local-file signatures and read leases cover loading and
inference; replacement requires a new capture. Native model caching follows
file identity and operation, releases replaced patchers, and clears on leaving
the utility pipeline. The existing image compute lane owns cancellation and
native memory management. Native `RTDETR_detect` → `DrawBBoxes` produces the
annotated image while preserving source alpha. Native
`SAM3_Detect` produces a union mask or ordered per-instance masks; cutouts multiply
them by source alpha. Output dimensions remain unchanged. Text categories,
source-image boxes and separate results are documented in
[native SAM prompts and instances](native-sam-segmentation.md). Video tracking
remains outside these still-image routes.

Outputs use the normal PNG/Library path with utility parameters and native model
provenance rather than fictitious generation sampling metadata. Scalar defaults
can be saved; image uploads and point lists are request-owned. The class picker
reads the pinned COCO literal without importing tensor/device code.

Native contract source:
`repositories/ComfyUI` at `fca60fe70e5b84c5c6b058263653d15cb3401e44`,
`comfy/model_detection.py`, `comfy/supported_models.py`,
`comfy_extras/nodes_rtdetr.py` and `comfy_extras/nodes_sam3.py`.
The `native-vision-utilities` feature descriptor declares focused CPU checks,
including real pinned detector/class-matcher and node bodies with tiny CPU
network stand-ins, plus actual worker PNG and public API callers. Such tests
establish source contracts; real model output, visual quality, device fit and
GPU stability require separate acceptance evidence.
