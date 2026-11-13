# Native Cascade Stage C guidance

In Create, select a bundled Stage C primary and its bundled Stage B companion.
The contextual **Cascade Stage C guidance** panel has two independent opt-ins:

- **Native Cascade super-resolution** uses a local native super-resolution
  adapter with the full 2048-wide Stage C model. Add the source in **Native
  image edit** or **Image to image**. Choose strength, sampling interval and
  Stage C denoise explicitly. Ordinary Cascade compression/source denoise apply
  to the existing image-to-image path when super-resolution is disabled.
- **Cascade vision mixing** uses a local CLIP ViT-L/14 vision model with
  768-wide projected embeddings. Upload one through eight images in order and
  enter one strength per image, from -10 through 10. Blank means 1 for each;
  zero omits that image's contribution. Native full-image versus native center
  crop controls the vision encoder's preprocessing. Vision references can guide
  text-to-image, ordinary source-image generation or super-resolution. Prompt
  text and scheduled prompt intervals are retained.

The selectors inspect bounded safetensors headers, including native aliases,
under the configured `controlnet` and `clip_vision` roots. The vision picker also
offers native Torch state-dictionary containers as **Compatibility unknown**.
Browsing never deserializes them; the pinned weights-only vision reader and the
768-wide projected-output check validate them during Run. The super-resolution
control adapter still requires its safe-header contract. Names and extensions
alone do not establish the role. The full Cascade super-resolution adapter has
16 input channels, eight native 2048-wide projections and the large Cascade
backbone; an ordinary ControlNet or a lite Stage C model cannot substitute for
that pairing. External weights remain local selections. This feature performs
no automatic acquisition and enables no Auto qualification flags.

Vision admission includes the native loader's layer 0/22/23 dispatch markers
and rejects higher-layer H/G dispatch. A partial L-shaped header cannot pass
admission while missing the key the native loader immediately requires.

`generation/cascade_guidance.py` owns the frozen request contract. It captures
the primary, adapter and vision-file signatures before compute, copies ordered
vision images into immutable RGB bytes, and records normalized strengths. The
existing preparation service remains outside serialized compute admission.
`stage_read_leases` surrounds the actual production `image_pipeline.process`
call so lazy native reads of all three sources remain protected through source
encoding, conditioning, sampling and decoding. Nested helper reads reuse that
scope. Cancellation releases every scope, and changed sources fail rather than
reusing stale files. Cache keys contain exact file signatures and primary model
changes clear the guidance cache. Native control application copies the adapter
for the request, preserving cached text conditioning and the hint-free original.

Super-resolution delegates to native
`StableCascade_SuperResolutionControlnet.execute`. The native EffNet hint feeds
the pinned large Cascade T2I adapter through `ControlNetApplyAdvanced`; both
conditioning branches keep the native control semantics. The returned empty C
and B grids sample through existing owners, and B's fourfold Stage A decoder
makes the operation natively twofold. Sources fit and pad to a 32-aligned canvas.
The output frame doubles the input frame, strips only alignment padding once,
and preserves the full source. With Preserve proportions and no exact target,
output dimensions are twice the source. Source alpha becomes native RGB.

For an explicit final target, `SizePlan` composes the native 2× factor with the
optional registered pixel upscaler. Its content describes the low-resolution
source canvas; native B output is 2× that content before any pixel upscale.
The same immutable plan and final `SizePlan.finish` retain exact square, odd
and upscaled targets. Crop requests fail admission. The recorded plan includes
the total factor, while `automatic_upscale` continues to identify only the
optional pixel model. Replay validates both factors instead of dropping size
intent. The default text-to-image and ordinary Cascade source paths retain
their existing grids and geometry contracts.

Vision mixing calls native `CLIPVisionEncode` and appends each active weighted
reference using `unCLIPConditioning` with zero augmentation. The pinned native
Stage C runtime builds `clip_img` from those ordered 768-wide embeddings. Vision
conditioning applies to the positive Stage C branch; negative conditioning and
Stage B's separately encoded prompts retain their native contracts. No prompt
replacement, embedding arithmetic or device-management implementation is added
to the adapter.

UI/API fields share the `cascade_super_resolution`, `cascade_control_*`,
`cascade_vision_enabled`, `cascade_vision_model`, `cascade_vision_images`,
`cascade_vision_strengths` and `cascade_vision_crop` contract. The API accepts
ordered inline images and numeric strength arrays. Metadata records component
signatures, input/output frames, native scale, image pixel identities and active
strengths. Library setup restores controls and clears the gallery; upload the
vision references again in the same order. Image bytes are never embedded in
generation metadata. Standalone utilities discard these generation controls
and do not resolve a Cascade checkpoint.

Source pin, inspected 2026-11-11:
[ComfyUI fca60fe Cascade nodes](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy_extras/nodes_stable_cascade.py),
[native Cascade adapter reader](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy/controlnet.py)
and [native image conditioning](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/nodes.py).
GGUF remains pinned to `da5f435cd9a9e54ad27eed417ed72b1f2543ef41`; these new
guidance component selectors use the bounded native safetensors contract.

Feature `native-cascade-stage-guidance` declares guarded CPU/native-body checks:
actual adapter reader selection, native super-resolution grids, native control
and vision nodes, native Stage C embedding assembly, production dispatch,
immutable image order, explicit zero, wrong components, source replacement,
cache reuse, whole-process read leases/cancellation, public API, exact size and
Library replay. These checks do not establish neural output, real weight
compatibility beyond the inspected contract, visual quality, GPU memory,
performance/stability or rendered browser acceptance. Those require separate
source/artifact-bound execution evidence.
