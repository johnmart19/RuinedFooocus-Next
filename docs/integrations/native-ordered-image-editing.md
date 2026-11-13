# Ordered native image edits and SD inpainting

Native schedules and dual-model guiders use the
[specialized sampling contract](native-specialized-sampling.md), which also
defines the shared output-group boundary for layers and instances.

The named edit recipes adapt pinned ComfyUI nodes on the existing image worker.
They share the selected checkpoint's native encoder/VAE and keep reference order
inside one request. No sampler, loader or device manager is reimplemented.
The source pin is `fca60fe70e5b84c5c6b058263653d15cb3401e44`, inspected 2026-11-11.

In **Create → Image tools → Native image edit → Edit recipe**, select the matching
Edit checkpoint and recipe. Put the primary image first and add extra references
in upload order. The Image tools binding is coordinated with the Library/shared
editing packet; `ui_ordered_references.py` owns these controls.

| Recipe                    | References including primary | Native conditioning                                            | Matching components                                                       |
| ------------------------- | ---------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Qwen Image Edit           | 1                            | `TextEncodeQwenImageEdit`                                      | Qwen2.5-VL, ordinary Qwen RGB VAE                                         |
| Qwen Edit Plus 2509       | 1–3                          | `TextEncodeQwenImageEditPlus`, native default reference method | Same Qwen components                                                      |
| Qwen Edit Plus 2511       | 1–3                          | EditPlus plus `index_timestep_zero` reference method           | Same Qwen components                                                      |
| LongCat Image Edit        | 1                            | Native image-aware text encode plus `index` reference method   | LongCat Qwen2.5 encoder, Flux VAE, distinct LongCat **Edit** checkpoint   |
| Flux.2 / Klein references | 1–16                         | `ReferenceMultiLatent` with a separate zero target             | Family-specific Mistral/Qwen3 encoder and 128-channel, 16-fold Flux.2 VAE |

Automatic Qwen/LongCat editing requires a current edit-variant identity from
verified source metadata or an explicit current native-header variant. The
shared class name and filename are insufficient: base and edit weights can use
the same native class. Unknown/base variants receive an actionable error;
an explicit compatible recipe permits unregistered local edit weights without
claiming that artifact has been tested. None of these source decisions certifies
real-model output, quality or device execution.

The edit nodes own resizing, image tokens and VAE reference encoding. RF owns
private ordered image copies and caches VAE results only inside that request.
The cache seals recipe, VAE identity, immutable image versions/order and native
preprocessing shape/dtype/device. Positive, negative and scheduled prompts reuse
those exact reference latents. Scheduled text keeps its own start/end intervals;
negative conditioning is encoded once. Source image pixels and cached ordinary
text conditioning are not overwritten.

Flux.2 native reference edits use `Flux2Scheduler` with the actual canvas size.
Klein is identified by the native Flux2 configuration's 20/24 single blocks and
retains ordinary sampler CFG; plain Flux.2 retains its embedded guidance behavior.
The saved metadata includes the actual scheduler and sigmas. Full text-to-image
Model Defaults transport, KV-specific recipes, Ideogram dual models and other
specialized schedules remain G2 work. Their catalogue Auto flags stay gated;
this packet does not qualify a Model Defaults path through helper tests.

## SD1/SD2 nine-channel inpainting

With a native nine-channel SD1 or SD2 checkpoint, use **Image tools → Inpainting**
and paint a source mask. Ordinary four-channel SD1/SD2 models do not gain that
contract. The passive capability picker reads bounded native header/configuration
metadata; the initialized model is checked again. Nine-channel weights require
the source/mask operation and their native four-channel, eightfold VAE.

RF calls the existing native `GrowMask` and `InpaintModelConditioning` adapters.
Both conditioning branches receive masked source latents and the mask; sampling
starts with the native original-source latent and its noise mask. Native
`BaseModel.concat_cond` forms the five additional input channels. Native SDXL,
Flux Fill and Qwen2.1 paths keep their separate existing contracts. SD1/SD2 uses
an RGB VAE; this adds no RGBA codec promise.

## API, replay and lifecycle

Image requests accept `native_edit_recipe`: `auto`, `qwen-edit`,
`qwen-edit-2509`, `qwen-edit-2511`, `longcat-edit`, or `flux2-reference`.
Explicit recipes require an unmasked primary `input_image` and the native edit
operation. Extra images use the existing `reference_images` array. The typed
API validates recipe-specific counts; family, source variant and codec checks
are repeated at native use. SD1/SD2 mask requests use the existing image/mask or
canvas-border API, including native mask expansion and denoise controls.

`native_edit_recipe` records the actual applied recipe. The grouped
`native_reference_conditioning` records ordered reference indices, dimensions,
pixel hashes, native node/method and source-identity basis. The inpaint group
records its native model configuration and mask polarity. No uploaded pixels,
local source paths or references are embedded. Library full-setup replay restores
the recipe while clearing uploads/masks. Plain Library editing/tool actions reset
the recipe to Automatic through the shared editing owner.

Inactive UI recipe fields are omitted from unrelated queued work, preserving the
panel choice. The existing prepared checkpoint/component snapshots, read leases,
single image compute lane, cancellation and retry own model resources. All new
reference tensors/caches live only for the active request. No acquisition,
package installation or extra GPU lane is added.

## Focused evidence

The feature descriptor `native-ordered-image-editing` declares the native-node,
pipeline caller, API, geometry, guidance and direct editing fixtures. Run it with
`venv/Scripts/python.exe tools/check_cpu.py` and the descriptor's test filenames.
The receipt `cache/plan-execution-20261111/W07-RG1-native-editing.json` retains exact
file/source/template hashes and command. CPU fixtures exercise the actual pinned
node bodies, ordered inputs, cache invariants, source-variant refusal, native
sigmas/CFG, nine-channel concatenation, API capture and existing Qwen2.1/HiDream
callers. Real weights, rendered full-app behavior, quality, performance and GPU
stability require separate source/artifact-bound acceptance.
