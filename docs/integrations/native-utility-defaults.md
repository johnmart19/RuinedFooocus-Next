# Ready-use image utility components

Editor actions and quick tools prefer a compatible installed component. With no
local match, the selector uses an existing registered first-use source:

| Tool                | Default source                                            |
| ------------------- | --------------------------------------------------------- |
| Depth map           | Depth Anything 3 Small                                    |
| Restore still image | SeedVR2 3B INT8 with the official SeedVR2 EMA FP16 codec  |
| Face landmarks      | Complete MediaPipe FP32 detector/landmarker bundle        |
| Whole-body pose     | SDPose FP16 whole-body checkpoint with its bundled SD VAE |
| Detect objects      | RT-DETR v4 HGNet FP16                                     |
| Segment by points   | SAM 3.1 multiplex FP16; add a point, box or text category |

Opening a tool or refreshing its choices only inspects local headers and the
existing catalogue. Missing components are downloaded through the existing
preparation service only when the tool runs. Offline mode identifies the missing
component for that action; other Editor actions remain available. Explicit API
component selections keep their normal validation and are never silently
replaced. A shared UI selector switches incompatible known defaults when changing
tools; an unknown missing saved path remains visible for correction.

The defaults are source-compatible choices, not measured device-fit claims.
They retain the existing shared image lane, native memory management and supported
CPU fallback. No model files, new sources or runtime dependencies are bundled.
Advanced Lotus/MoGe/Marigold controls retain their existing component requirements;
these local workflows are not presented as automatically acquired defaults.

`image_processing/utility_defaults.py` owns only selection preferences. Download
URLs, immutable source identities and acquisition checks remain owned by the
model catalogue, registry and image preparation service. CPU regressions in
`test_utility_defaults.py` cover real capture and Gradio picker callbacks without
downloading or executing model weights.
