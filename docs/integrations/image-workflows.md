# Owned image masks and finite workflows

`features/image_editing/mask_handoffs.py` and `image_workflows.py` join the existing
native image services to the artifact-backed editor store. They capture and
translate one explicit operation at a time. The shared image worker retains
preparation, resource leases, serialized compute, cancellation and terminal
publication. These backend contracts add no execution loop, graph engine or
parallel compute lane. UI wiring belongs to the Create/Library workspace.

## Selection masks and inpainting

`capture_mask_handoff(store, session_id, revision, source_id, mask_id, owner=...,
choice=...)` accepts an owned P3 image and a grayscale selection-mask artifact.
The result freezes source/mask IDs, both frame IDs, revision, union versus one
instance, the zero-based instance index/count and effective inversion. A union
has no instance index; instance results retain native ordering. No implicit union
of unrelated images is constructed.

The mask must refer to this exact source artifact and oriented source frame.
Matching dimensions alone are insufficient. A cutout is an image with alpha and
cannot bind as a selection mask. White selects the area to edit, black preserves
it and gray gives fractional coverage. Stored mask inversion and a requested
inversion combine exactly once. Selection coverage remains independent of source
alpha; a native cutout has already multiplied its mask by source alpha, so its
alpha cannot reconstruct the original selection mask.

`inpaint_inputs` revalidates the owner/revision/artifact signatures before and
after reading. It rejects changed files, wrong geometry, cutouts, and empty masks
before asset preparation. It translates grayscale coverage into the existing
editor layer's alpha, without resizing or applying the public PNG API's inverse
alpha convention. `capture_step(..., "inpaint", ..., mask=handoff)` records that
typed mask in the P3 operation ticket.

The existing [SAM detector and renderer](native-sam-segmentation.md) still own
`SAM3_Detect.execute`, text encoding, union/instance output and cutout alpha.
RT-DETR detection remains independent. `inpaint_utils.prepare_inpaint` applies
one identical geometry plan to source and selection. The existing native SD/Flux
conditioners still own `GrowMask` and `InpaintModelConditioning.encode`, including
original latents, concat conditioning and noise masks. P4 introduces no sampler,
segmentation, inpaint or mask-growth algorithm.

## One captured step

`capture_step` accepts the server-owned editor store, session/revision, one of
the seven finite actions below, and only that action's settings. It captures the
current source, optional mask/additional image refs, exact target, framing, alpha
policy, literal prompt/negative, LoRAs, denoise and selected resources. Settings
are detached bounded finite JSON. Later edits to a profile or an input mapping
cannot change this ticket.

| Action                | Existing operation and input binding                                                                                                            |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `remove-background`   | Existing native/ONNX background service; current image, selected file and explicit force flag                                                   |
| `select-background`   | Owned additional image becomes current without compute; original foreground remains a history artifact                                          |
| `generate-background` | Existing generation request; literal prompts/negative/LoRAs/current resources, without source-image conditioning                                |
| `compose`             | Current background followed by the owned foreground; existing native `ImageCompositeMasked` or Porter–Duff service                              |
| `prompt-edit`         | Existing `image_components.edit_control` selects native edit or img2img from captured family/checkpoint/MOSS format; denoise intent is retained |
| `inpaint`             | Exact owned result-mask handoff through existing native inpaint conditioning                                                                    |
| `upscale`             | Existing native pixel upscaler, selected descriptor and both explicit exact target dimensions                                                   |

`worker_request(step, store, owner=...)` returns the existing worker fields plus
`_editor_operation` and `_image_workflow_step`. It starts no preparation or model
work. Callers use `capture_image_data`, then
`bind_effective_plan(data, captured_plan, store=..., owner=...)`, then the existing
preparation and worker lifecycle. Binding the final P1 plan supersedes the draft
ticket while retaining source/mask/additional refs and operation settings. The
old ticket cannot execute or publish. A changed native edit family requires a
fresh edit capture. P2 readiness and preparation resolve active resources for
each explicit step; preset selection and navigation acquire nothing.

Background selection uses `selected_background` to return the owned artifact and
its explicit no-compute geometry receipt. It does not enqueue an image job.
Worker completion is owned by the integrating service: after actual terminal
success, call P3 completion once, using the final bound ticket. A typed pre-save
`finish_worker_output` hook validates the captured output frame and preserves
source alpha for RGB-only prompt edits/inpaint. Existing native RGBA output is
authoritative, including Qwen's localized alpha blend; the hook never overwrites
its alpha. Float-array RGB values retain their precision. Deliberate `native` or
`flatten-black` alpha handling is explicit.
Composition and background selection/generation require an already owned
flattened artifact when flattening is desired; they reject a flatten policy
that authoritative source rebinding would otherwise discard. Generated
backgrounds use the existing exact-target plan, including native canvas padding
and restoration for targets that are not alignment multiples.
`execution_metadata` records actual geometry/alpha policy beside actual worker
metadata and rejects a different target. Pure processing removes inherited
diffusion fields rather than claiming sampling occurred.

Model workflows automatically prepare a request-local 8-bit sRGB working image
from the owned original, using the same native precision/color conversion as
Library editing. Embedded ICC profiles use Pillow's CMS with alpha kept separate
from the color transform. No export or manual preparation step is required.
The original artifact, metadata, precision, dimensions and history identity stay
unchanged. The request and completed history record the working conversion;
native adjustments and precision exports continue to read original samples.
Owned selection masks keep their exact source/frame identities and soft coverage;
proven precision mask results normalize to an L8 selection without a color curve.
An edit size change requires explicit `pad` or `crop`; default preservation
cannot silently resize, crop or stretch. Composition retains native contain/pad
geometry and source alpha. [Composition](native-image-composition.md) and
[exact upscale targets](exact-image-targets.md) remain the algorithm owners.

Direct Compose also accepts one owned `background_id` while the current image
remains the foreground and owns the output canvas. The background fits uniformly
with transparent padding; it is never cropped or stretched. Native capture
revalidates and rebuilds both inputs from their owned IDs, preserving role order.
The existing current-background plus `foreground_id` route remains supported;
the two role selectors are mutually exclusive. Neither route invents an
intermediate history step.

## Finite presets and lineage

`preset_values` permits only the named preset's scalar intent. Background
replacement stores select/generate intent, native source-over and framing.
Portrait enhancement stores framing and denoise. Presets exclude uploads,
masks, base64, model paths, arbitrary operations, graph edges and repetition.
Models, files, prompts, LoRAs and targets are selected and captured per step.
Relevant preset scalars supply each step's defaults; explicit step parameters
take precedence. The captured preset and the effective step settings both remain
in that operation's lineage.

`begin_workflow` creates a cursor bound to the current session revision. The
background sequence is remove background → select or generate background →
compose → prompt edit → upscale. Portrait enhancement is prompt edit → upscale.
The caller explicitly submits each next step; no step starts its successor.
Composition binds the completed background as current and the exact retained
foreground as its additional image. Every completed operation has its own P3
source/target/parameters/effective plan and result ID. Inherited generation
provenance remains separate from operation execution.

`advance_workflow` requires the exact operation ID in a newly completed P3
snapshot. Failed, canceled, replaced and stale tickets cannot advance it.
`skip_background_removal` only skips a current already-transparent source; it
keeps the revision and creates no fictitious completed operation. An opaque
source cannot use this shortcut. Undo/redo or another completed edit changes the
revision, invalidating an old cursor and mask capture. Owners remain isolated.

## Native source and focused checks

Native contracts remain on ComfyUI
`989d93d0d968e704c589ce0146bb18c77f6b50cc` and ComfyUI-GGUF
`da5f435cd9a9e54ad27eed417ed72b1f2543ef41`. No backend pin, installed package,
model file or device policy changes. The implementation uses source-local native
services and requires no external API documentation.

```sh
python tools/check_cpu.py test_image_workflow_handoffs.py test_sam_prompt_utilities.py test_mask_input_contracts.py test_image_inpaint_geometry.py test_api_native_inpaint.py test_image_composition_integration.py test_exact_upscale_target.py test_native_sd_inpaint.py test_upscale_transparency.py
```

Fixtures exercise the actual artifact store, native SAM rendering, native pure
composition and geometry adapters with tiny CPU images. Generation/upscale chain
results are labeled local fixtures; those checks establish transport and lineage,
not learned-model execution. Real-model masks, visual quality, rendered UI,
performance and GPU stability require separate evidence. GPU work remains paused.
