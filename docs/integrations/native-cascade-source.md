# Native Cascade source conditioning

Stage B is exposed as a [stage companion](native-stage-companions.md), with
Stage C selected as the primary image model.

Choose a bundled Stage C checkpoint and its bundled Stage B companion. In
**Image tools**, choose **Native image edit** or **Image to image** and add one
source. The **Stable Cascade · Stage B** panel exposes compression (default 42)
and source denoise (default 0.75). Zero denoise retains the encoded Stage C
prior; Stage B still samples its own configured schedule. Masks and other image
tools are refused. Clear the image tool for the existing text-to-image route.

`generation/cascade.py` extends the existing stage owner. The production
`image_pipeline.process` caller converts the EXIF-oriented source to RGB. With
**Preserve proportions** and no captured exact target, it passes the original
source dimensions to pinned `StableCascade_StageC_VAEEncode.execute`. An exact
output target, including the API's explicit `size`, uniformly fits and pads the
complete source to the captured working content size, then pads to the native
eight-pixel canvas. It keeps the immutable size plan: after Stage B decoding,
the existing reference-frame restoration removes only that alignment padding,
then the native pixel-upscale stage and `SizePlan.finish` produce the exact
requested target. Aspect-fit padding remains part of the generated canvas;
cropping is refused at submission. Source alpha remains native RGB input.
The native node performs its EffNet
resampling and returns both stage grids. Stage C sampling uses the requested
denoise. Stage B receives the sampled prior through
`StableCascade_StageB_Conditioning`, retains the encoder's paired quarter-size
grid, and uses the existing bundled Stage A codec for final decode. Native grid
rounding determines output dimensions for the source-size route. Metadata
records original source dimensions, the framed native input, the current source
frame, both stage shapes, compression, denoise and the Stage B source signature.
The frame is request-local and removed once; model-cache reuse does not reuse
another source's geometry.

Both checkpoints keep their existing capture/read-lease and cache owners.
Stage B uses the request seed, its explicit CFG/sampler/scheduler and its own
prompt schedule. Cancellation is checked before loading, sampling and decoding,
with native sampler cancellation unchanged. API fields `cascade_compression`
and `cascade_source_denoise`, queued UI controls and Library setup replay share
the same contract. A new request clears stale runtime conditioning metadata.
Full Stage C now has a native-configuration Model Defaults recipe: stage `c`,
conditioning width 2048, text mapper `[2048,1280]` and output projection
`[16,2048,1,1]`. It uses C20/CFG4/`euler_ancestral`/`simple`. Current conflicting
variant metadata, lite and unknown layouts do not select this recipe. A full
Stage B companion can supply B10/CFG1.1/`euler_ancestral`/`simple` when native
configuration identifies its `[320,640,1280,1280]` hidden layout and its bundled
CLIP-G projection and Stage A codebook match. Existing verified publisher-source
defaults remain available. Explicit or saved sampling settings keep precedence;
Stage B remains a companion and cannot become the primary generation model.

The retained explicit C01/C02 runtime receipt, SHA-256
`2667e903d2e61fc4b2aff2dafd5a78bd626f12e5acfc7ff15a9b842fbf44d296`,
records both native C/B samplers and Stage A decode on source `a7cf09a6` with
native pin `fca60fe7`. These successful source-image and SR/vision cases establish
the full configuration recipe; automatic caller execution after this source
change remains a separate acceptance step. CPU checks exercise actual typed API
preparation, the UI callback, stage-B capture, explicit overrides and the native
header matcher without reading model payloads. They do not establish lite,
alternate-device, performance or stability qualification. Layered qualification
is separate and remains unchanged.

The separate [Stage C guidance contract](native-cascade-stage-guidance.md) adds
explicit native super-resolution control and ordered vision mixing while
retaining this default source-image route.

Source pin: [ComfyUI fca60fe](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy_extras/nodes_stable_cascade.py),
inspected 2026-11-11. GGUF pin remains
`da5f435cd9a9e54ad27eed417ed72b1f2543ef41`; this bundled Cascade route does not
introduce a GGUF codec. Focused guarded CPU checks are declared by feature
`native-cascade-source-conditioning`: native node bodies, actual production
source/latent/final-size callers, public API square/odd/upscaled targets,
crop admission refusal, preserved settings/metadata, secondary signature leases,
cancellation and contextual controls. These fixtures do not qualify actual
weights, device memory, output quality or performance. No inference or download
is required for the source checks.
