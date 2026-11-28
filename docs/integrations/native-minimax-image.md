# MiniMax H3 Image

Create uses the native MiniMax H3 Image text-to-image graph when its image
checkpoint is selected. In Image tools, **Native image edit** accepts a primary
image and up to three additional ordered references. Describe them as
`<Picture 1>`, `<Picture 2>`, and so on. The optional 9B prompt generator in the
official example is separate from the required 32B conditioning encoder.

Select Model Defaults for the source-identified six-step PDD recipe, or choose
Own settings with the checkpoint's PDD step count, CFG 1, Euler, Simple and CLIP
skip 1. Clear the negative prompt. Keep automatic upscaling Off. The first
adapter executes this released graph; masks, image-to-image denoise, tensor
overrides, staged models, GLIGEN, hypernetworks and LoRA/other patch recipes
are rejected explicitly. This bounds the connected application graph and does
not claim those operations are impossible in future native workflows.

The Image tools selector accepts the complete checkpoint capability map during
startup and model changes. MiniMax exposes native editing while its unsupported
image-to-image denoise preset and custom tool type stay hidden.

## Components and public source

The adapter requires ComfyUI
[`989d93d0d968e704c589ce0146bb18c77f6b50cc`](https://github.com/Comfy-Org/ComfyUI/tree/989d93d0d968e704c589ce0146bb18c77f6b50cc),
the exact 0.45 adoption feature, and its existing runtime bundle. It adds no
installer, package, driver or model download at import time.

The WeVisOne catalogue prerequisite supplies metadata for the combined registry
fingerprint. MiniMax runtime assets remain its image checkpoint, 32B conditioning
encoder and image VAE; selecting this workflow acquires or loads no WeVisOne weights.

The default image component roles are `clip_minimax_h3_image` (the official
50-layer Qwen3-VL 32B INT8 ConvRot conditioning encoder) and
`vae_minimax_h3_image` (the dedicated MiniMax H3 Image VAE). The existing
MiniMax video defaults stay separate. The 9B prompt helper, recovered video
encoders, PH3 image codec and temporal H3 video codec do not fill these roles.
Positive native text predicates and the native image-codec topology are checked
before weight loading; the loaded model, encoder and codec are checked again.

The official image checkpoint and VAE are registered from
[`Comfy-Org/MiniMax-H3-Image@0bb149f51e3156ed4b90dd45d5620346bfe1df8f`](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/0bb149f51e3156ed4b90dd45d5620346bfe1df8f).
The catalogue records publisher SHA-256 and size; no local full-weight or
hardware qualification is implied. The required INT8 encoder already has the
same payload registered from the official MiniMax H3 repository.

| Component                 | Published bytes | SHA-256                                                            |
| ------------------------- | --------------: | ------------------------------------------------------------------ |
| Image DiT INT8 ConvRot    |  20,979,314,802 | `83eb140f1ff95c068f8a6886e6d3f3d0d4ab2e07bb2fe6d2ef992a51146ee622` |
| Qwen3-VL 32B INT8 ConvRot |  27,141,342,152 | `bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923` |
| Image VAE FP16            |   1,168,505,312 | `d80e9c2e5cfc9844869961ff5a58c9a9e52182df6bfb30b26018dd7f348721af` |

These three payloads total **49,289,162,266 bytes (45.90 GiB)** before runtime
objects, dequantized weights, activations and workspaces. Keeping the largest
packed component resident alone is about **25.28 GiB**. Native partial loading
and offloading can change device residency substantially; these figures are
storage/residency estimates, not minimum VRAM or a promise that a device fits.
The 2048-square native reference recipe also incurs text/vision and DiT
activation memory. Actual peak system/device memory requires a real-model run.
No large artifact acquisition or generation was performed for this feature.

Native text loading also recognizes the official 32B NVFP4 AWQ conversion
(15,687,142,551 bytes); its logical type is not a claim of AMD acceleration.
At this pin `supports_nvfp4_compute` returns false for CPU and non-NVIDIA
devices, and requires NVIDIA compute capability 10 or newer for that fast path.
The native mixed-precision code retains dequantization paths. Their exact
artifact/device memory, correctness and speed remain unqualified here. INT8 is
the image default. A separately supplied unquantized 50-layer encoder must pass
the same native identity checks; this audit did not establish a public BF16
encoder tuple. The smaller 9B model is not a conditioning alternative.

## Native calls, framing and ownership

`features/generation/minimax_image.py` captures immutable RGBA reference bytes,
checkpoint identity and the checkpoint's original PDD heads/sigma metadata.
Existing component inventory/preparation captures the model, encoder and codec;
native reads hold the corresponding file leases. Changes after submission or
before lazy reload fail instead of silently changing a job. Cancellation is
checked before conditioning, between stages and at every sampler callback.
The existing serialized image job owns execution.

The adapter calls `MiniMaxH3ImageConditioning.execute`, then native
`ConditioningZeroOut`, `comfy.sample.prepare_noise`, `comfy.sample.sample` and
the existing native image VAE decoder. The model already owns
`MiniMaxH3ImageModelSampling`; the adapter applies neither a generic flow shift
nor a replacement sigma curve. The latent is exactly
`[1, 24, 1, canvas_height/16, canvas_width/16]`; no audio or video graph runs.

The requested output uses the existing `SizePlan`, padded to the native
32-pixel canvas grid. Decoding removes only this artificial border and returns
the exact requested size, including odd targets. Reference conditioning uses
the pinned native, uncropped short-edge/grid resize policy: the text encoder
sees a 1024-pixel short edge and the VAE a 1024–2048 short edge. Metadata records
original dimensions, RGBA hashes, native reference sizes, padded canvas and
final output separately. Transparent reference pixels are composited over
neutral RGB before native encoding; the image VAE produces opaque RGB.

There are at most four references, a combined 16-megapixel upload budget, a
combined 16-megapixel native VAE-reference budget, aspect ratios within 1:8–8:1,
and a 16-megapixel output canvas with sides from 64 through 4096 pixels. These
are bounded application inputs, not measured hardware capacity.
Each repetition returns one `ImageOutputBatch` under the captured `OutputPlan`.
Existing API/worker completion checks reject partial or cancelled groups.

## API and replay

Use the existing `POST /images/generations` fields: image model ID, prompt,
size, seed and sampling settings. An `input_image` selects native editing;
`reference_images` supplies the additional ordered pictures. All input images
are owned decoded uploads, with alpha retained until the explicit RGB policy.

Saved image metadata includes `native_minimax_image`, the ordinary prompt,
checkpoint, seed, sampling settings, exact target plan and component choices.
Library **Use full setup** restores those settings and clears stale uploads.
For an edit, reattach the exact original ordered references; their RGBA hashes
are checked. A saved result is not substituted for its original references.
API callers may pass that saved `native_minimax_image` object through the typed
`minimax_image_context` field alongside the original uploads. Display-only
metadata is ignored by the replay schema. Model/encoder/codec local identities
must match; nanosecond timestamps use decimal strings to survive browser JSON.
Provenance never supplies a path, executable object or tensor.

## Source and CPU evidence

The official T2I and edit examples are the native graph references:
[T2I](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_t2i.json)
and [edit](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_edit.json).
The inspected `comfyui_workflow_templates_json==0.1.131` copies have SHA-256
`73f7d9c137e48de2183d9f9e19cdd0ea9009d9433fd3a48f10c43534eecf255b`
and `5b4330e9c8bc69fe0c8ccc7ba8e433484a419cf23f5cd099626ecaa15b05325f`.
Their optional prompt generator is not necessary to execute the image graph.

Run `tools/check_cpu.py test_minimax_image.py` with the project interpreter and
exact pinned native source. The suite executes actual native conditioning,
reference resize/order, zero-negative and PDD/simple-schedule bodies on CPU
tensors. It also exercises actual API capture, the production pipeline entry,
component admission, cancellation, metadata replay and exact odd targets.
Weight computation is isolated at native model/sampler boundaries. Source/CPU
evidence does not certify large-artifact loading, GPU stability, speed, generated
content or visual quality. Root coordinates those separate acceptance steps.
