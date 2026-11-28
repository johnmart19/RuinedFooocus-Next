# Named native image patches and control recipes

In **Create → Image tools**, a compatible checkpoint exposes **Native model
patch**. Choose the recipe, an installed patch and that recipe's own images.
Generate executes the captured request. These controls use the native image
engine and acquire no models automatically.

| Recipe                   | Checkpoint and local assets                                                                            | Inputs and native application                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Anima LLLite             | Anima and named-key v2 LLLite weights in `model_patches`                                               | Own guide, optional mask for a four-channel patch, strength and sampling interval; `AnimaLLLiteApply.apply_patch`              |
| Qwen Image DiffSynth     | Ordinary Qwen Image and its block-wise DiffSynth patch in `model_patches`                              | Own guide and optional mask when the patch has extra mask channels; `QwenImageDiffsynthControlnet.diffsynth_controlnet`        |
| Z-Image Fun              | Z-Image and its Fun patch in `model_patches`                                                           | Guide, or dedicated inpaint source plus white-edit mask, optionally both; `ZImageFunControlnet.diffsynth_controlnet`           |
| Flux USO style reference | Ordinary Flux/Flux Schnell, USO projector in `model_patches`, matching SigLIP encoder in `clip_vision` | Own style reference; `CLIPVisionEncode` and `USOStyleReference.apply_patch`. Native USO has no strength or interval parameter. |

For a ControlNet tool, **Native ControlNet recipe** offers family-specific choices:

- **Union mode** selects Auto or a named mode on an ordinary SDXL/Flux Union
  ControlNet. Select local weights above, then add the ordinary tool's guide.
  `SetUnionControlNetType.execute` copies the ControlNet before setting its mode;
  `ControlNetApplyAdvanced` receives that copy. A mode outside the selected
  weights' embedding table is rejected.
- **Qwen AliMama inpaint / outpaint** requires ordinary Qwen Image, a compatible
  68-input ControlNet, and its 16-channel, spatial-factor-eight RGB VAE. Select
  **ControlNet / T2I adapter**, its local file and this recipe. Upload the dedicated
  source and white-edit/black-preserve mask, or choose **Extend canvas** and its
  separate edge controls. AliMama uses `ControlNetInpaintingAliMamaApply.execute`;
  these weights belong in `controlnet`, rather than `model_patches`.

Qwen Image 2.1 **Fun ControlNet Union** remains the distinct existing
[Image 2.1 workflow](native-special-image-workflows.md#qwen-image-21-fun-controlnet-union).
It does not use ordinary ControlNet's Union-mode embedding table.

## Request, geometry and lifecycle

The source owners are `features/image_editing/model_patches.py`,
`native_control_recipes.py` and `ui_model_patches.py`; production capture lives in
`features/generation/image_preparation.py`, with execution in `image_pipeline.py`.
Queued images are oriented copies. Compatible safe tensor headers establish the
native reader branch and expected shapes. Native Torch files are explicit
candidates labelled for validation on Run; the loaded native patch class must
still match the selected recipe and checkpoint. File identity is checked again
while acquiring prepared-asset read leases.

Each native application receives a request-local model clone. Only validated
weight/vision objects are cached; reference tensors, native image caches and
source frames belong to the request. Read leases cover patch loading and the
sampling call. Cancellation is checked before loading, after native application
and during the existing worker/sampler cancellation path. Failed or cancelled
application cannot publish a new weight-cache entry. Existing controls use their
prepared ControlNet reader and exact captured file identity.

Fun and AliMama source denoise defaults to 1 and accepts values greater than zero
through 1, independently of patch/control strength. Their dedicated source is
encoded with native `VAEEncode`. A painted mask is also attached through
`SetLatentNoiseMask`. Source and mask share one preserve/pad/crop frame; a
preserved source is not forced into an unrelated generation target plan.

AliMama canvas extension freezes its exact source-plus-border dimensions before
queueing, without importing native runtime/device code. Compute then calls native
`ImagePadForOutpaint` once. The full extended canvas is encoded without a latent
noise mask, matching that native recipe. Its alignment padding is removed once
and original pixels are blended back outside the generated border/feather. A
conflicting explicit API size or crop/pad mode is rejected. Automatic upscale is
disabled for this dedicated AliMama source recipe.

These patch/control operations produce native RGB images. Uploaded RGBA guides
and sources supply their RGB channels; they do not establish alpha-generation
support. Library originals, including alpha/precision data, remain unchanged.

## Defaults, Library and API

Saving Create defaults stores recipe, local filenames and numeric settings. It
never saves uploads. Plain Library Generate/Edit/Inpaint/utility actions clear
named patch/control recipes and their uploads while preserving the selected
Library source, denoise, geometry, components and LoRAs. **Use full setup**
explicitly restores saved recipe settings and clears uploads, so the user must
add the recipe's images again. Metadata contains settings and source/mask hashes,
never uploaded images or private capture objects.

The image request schema exposes flat `model_patch_*`, `control_recipe`,
`control_union_type`, `control_inpaint_*` and `control_outpaint_*` fields. The
per-model capabilities response lists `model_patch_recipes` and `control_recipes`.
API recipes are explicit and never inherit another session's uploaded images.
Mixed main-mask, ordered-reference, adapter and named-patch recipes are rejected.
For example, a Z-Image Fun source request uses:

```json
{
  "model": "image/local-zimage.safetensors",
  "prompt": "Repair the marked area",
  "model_patch_recipe": "zimage-fun",
  "model_patch_model": "local-fun.safetensors",
  "model_patch_inpaint_image": "<inline source image>",
  "model_patch_mask": "<inline white-edit mask>",
  "model_patch_denoise": 0.8,
  "image_resize_mode": "preserve"
}
```

AliMama uses `control_recipe=alimama-inpaint`,
`controlnet={"type":"control","model":"local-alimama.safetensors"}`,
`control_inpaint_image` and `control_inpaint_mask`. For extension, set
`control_inpaint_mode=outpaint` and positive `control_outpaint_left/top/right/bottom`
values instead of a mask. Unlike the ordinary masked-edit API's alpha mask,
these dedicated patch/control mask images use white for pixels to regenerate.

## Source and evidence boundary

The adapter follows ComfyUI pin `fca60fe70e5b84c5c6b058263653d15cb3401e44`:
[model-patch nodes](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy_extras/nodes_model_patch.py),
[control recipes](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy_extras/nodes_controlnet.py),
and [Union modes](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy/cldm/control_types.py),
inspected locally on 2026-11-11. No Context7 gap or dependency change was needed.
The existing Windows/Linux native image runtime owns device and package support;
this packet adds no hardware qualification or automatic asset acquisition.

Focused guarded checks:

```sh
python tools/check_cpu.py test_native_model_patches.py test_native_control_recipes.py test_native_model_patches_ui.py test_native_patch_integration.py test_control_model_selection.py test_library_editing_workflow.py test_gallery_actions.py test_generation_defaults.py test_image_request_isolation.py test_image_metadata_boundary.py test_api_image_edits.py
```

Checks exercise pinned native method bodies with inert dependencies, actual
Gradio callbacks, public ASGI admission, lease exclusion, cancellation, file/type
guards, source/mask geometry and Library defaults/replay. They do not run model
inference. Rendered browser behavior, representative real artifact comparisons,
visual quality, performance and GPU/driver stability remain unverified.
