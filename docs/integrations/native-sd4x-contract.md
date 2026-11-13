# Native SD4x diffusion upscaling

`features/image_editing/sd4x_conditioning.py` and the generation pipeline reuse
the pinned ComfyUI `SD_4XUpscale_Conditioning.execute` node. For a recognized
`SD_X4Upscaler` checkpoint, **Create → Image tools → Diffusion 4× upscale**
requires a source image. Ordinary image-to-image and masked inpainting are
unavailable. Remove background and standalone pixel upscaling remain separate
utilities. This is a source/CPU integration; real-model acceptance is separate.

The output-size note follows the uploaded source, including EXIF orientation.
Generic resolution and framing controls become read-only for this operation;
their values remain available when switching to another checkpoint. API size
`auto` or omission follows the source. An explicit incompatible size, padding or
cropping is rejected. Existing API output bounds still apply. Imported metadata
cannot silently retarget a different source size. No family-wide Model Defaults
recipe is invented; use a complete checkpoint recipe or explicit Own settings.
Generation's primary **Output size** uses this native base and the existing
post-upscale calculation, so an additional 2×/4× enlargement correctly shows
8×/16× the original source dimensions. The Image tools note names the native base
before post-upscaling and hides for unrelated image utilities; input-only pixel
upscaling keeps its own source-size calculation.

At ComfyUI `fca60fe70e5b84c5c6b058263653d15cb3401e44`, the node takes an RGB source,
positive/negative conditions, scale ratio and noise augmentation. The helper fixes
the native scale to four. The node normalizes source pixels to `[-1, 1]`, adds
`concat_image` and `noise_augmentation` to copied condition metadata, and creates
a zero four-channel latent at the source resolution. This is the native SD4x
latent geometry: the decoded result has four times the source dimensions.
Native `SD_X4Upscaler.extra_conds` owns source resizing, seeded augmentation,
`c_concat` and noise-level `y`; do not duplicate those operations in RF.

Asset capture rejects missing/incorrect source mode before dependency preparation,
and the process checks it again. The native SD20 CLIP-H encoder is distinct from
CLIP-L. The native `SD_X4Upscaler.latent_format` selects `SD_X4`, whose scale
factor is `0.08333`. RF does not normalize the sampled latent again. A runtime
decode-compression check rejects an ordinary eightfold VAE override; native VAE
loading identifies the dedicated three-level fourfold codec. Component choices
and captured identities retain the existing request-local lifecycle.

The native model generates RGB. For transparency, RF keeps source alpha separate,
composites conditioning RGB over neutral gray (128), resizes the alpha uniformly
by four using bilinear interpolation, and reattaches it to generated RGB before
optional post-upscaling. This is preserved source alpha, not native RGBA inference.
Opaque RGBA and EXIF orientation are retained; the original upload and its metadata
are not changed. RGB sources keep RGB output.

Official public metadata is pinned to
`stabilityai/stable-diffusion-x4-upscaler@572c99286543a273bfd17fac263db5a77be12c4c`.
The preferred native full checkpoint `x4-upscaler-ema.ckpt` includes both
components (3,531,514,018 bytes; SHA256
`57c63b7ed94406ba07e93f38e82622ff61952bb3fc0dec684a4058fa880cddd0`).
Separate optional text/VAE entries use the same publisher revision and exact LFS
hashes; app filenames `sd4x_clip_h.safetensors` and `sd4x_vae.safetensors` avoid
collisions between the publisher's generic filenames. Metadata-only acquisition
identity does not certify loading. Standalone Diffusers UNet conversion is not
qualified by this route.

For the official `.ckpt`, choose **Models → Checkpoint tools → Inspect** explicitly.
This hashes through the existing signature-bound cache and matches the packaged
publisher checksum and size, then records the verified native family and bundled
encoder/VAE declaration. Successful inspection refreshes workflow controls, component
inventory and source-derived sizing without changing the selected checkpoint or uploads.
The metadata remains tied to this resolved file/signature; a conflicting adjacent
publisher sidecar cannot replace its verified family. A changed file invalidates
the proof. Inventory and request capture show both components built in and acquire
no duplicate split files. Unknown hashes receive no verified family. Passive
selection never unpickles or hashes Torch weights. **Check weights** and **Create
FP16 copy** stay disabled for this container: source verification is not tensor
integrity inspection. General unknown Torch-container inspection remains a
separate format-parity gap, not something established by known publisher hashes.

Primary contracts (reviewed 2026-11-09):

- [Native conditioning node](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy_extras/nodes_sdupscale.py)
- [Native model and augmentation](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy/model_base.py)
- [Native supported model](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy/supported_models.py)
- [Native latent format](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy/latent_formats.py)
- [Official component configs and artifacts](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/tree/572c99286543a273bfd17fac263db5a77be12c4c)

Focused checks use the project interpreter:
`python tools/check_cpu.py test_sd4x_conditioning.py test_api_image_edits.py test_instruct_pix2pix.py test_powerup_family_controls.py test_image_preparation.py`.
The tests execute the pinned node and actual RF process with CPU tensors/fake
models, verify odd dimensions, source/size guards, RGB/alpha/EXIF retention,
native scale ownership, exact component metadata and strict sampling defaults.
They do not load a real decoder or certify UI rendering, quality, performance or
GPU stability. Bounded real-model acceptance and the declared prerequisite
replay remain independent publication gates.

`cache/plan-execution-20261109/sd4x-real-source-cpu-acceptance-v3.json` records the
actual downloaded full file through the Inspect backend, verified metadata,
bundled inventory and zero-dependency capture/preparation. The exact pinned native
weights-only loader with meta storage accepts 1,207 tensors and matches the native
SD4x configuration after RF's CPU initializer, with strict CUDA/XPU/MPS guards.
It does not read tensor values or instantiate the model. An earlier direct Comfy
import bypassed that initializer and attempted HIP discovery; the explicit GPU
guard stopped it before initialization. Its separate narrow AST-loader proof is
retained. None of these checks establishes model inference or GPU qualification.
