# Native SAM text, boxes and individual results

The [owned mask handoff](image-workflows.md#selection-masks-and-inpainting) binds
grayscale results to an exact editor image/revision for native inpainting.
Union versus instance and inversion are explicit. Cutouts remain alpha images;
their multiplied alpha is not silently reused as a selection mask.

Editor → More tools → Segment by points retains its existing click-to-keep and
click-to-exclude controls. It also accepts optional **Text categories**,
**Source-image boxes**, and **Separate result for each object**. Choose an
installed SAM 3 or 3.1 model. A single source image and a single native detector
run produce the results; generation batch size is not the number of objects.

Editor keeps the source photo current when segmentation finishes, including an
individual run with no matches. Masks and cutouts remain available in the result
gallery. For grayscale results, **Mask selection** lists the objects in native
result order, numbered from one. Select one or more, then **Use result as
selection mask** to prepare Inpaint on the same photo. Multiple selections use
the greatest grayscale value at each pixel, retaining soft coverage; source
alpha remains separate. A cutout's multiplied alpha is never used as a mask.
The initial editing handoff requires exact-frame 8-bit sRGB source/masks.

The existing Editor session owns a bounded `derived_results` receipt and its
zero through 64 artifact references. Terminal publication advances the revision
without replacing the source head or original generation provenance. Output
plan/completion proof, owner, pending ticket, file signatures and hashes fence
publication. Image decoding and hashing happen before the worker's terminal
registry lock; terminal publication uses frozen refs and short identity checks.
The explicit selection action prepares its union in the existing bounded
preparation pool, outside shared UI/store locks, then revalidates and atomically
binds a normal
selection-mask artifact; stale, changed, foreign, empty or precision selections
cannot enter inpainting. Cutout/precision exports remain available without being
silently converted. The receipt is private session state, not a public PNG ticket.

Text uses the pinned native SAM tokenizer syntax, for example `person:3, dog:2`.
The native parser, encoder and detector own category interpretation, detection
ordering, thresholding and refinement. Text is literal: generation commands,
wildcards, prompt enhancement, sampling settings and imported PNG prompts do not
interpret it. The selected checkpoint must contain its own paired SAM language
backbone; the loader does not substitute a generation CLIP or download another
encoder. Point-only and box-only requests retain the diffusion-only native loader.

Boxes use oriented source pixels:

```json
[{ "x": 10, "y": 20, "width": 100, "height": 80 }]
```

Coordinates and positive extents must stay inside the source. Each box has exactly
those four finite numeric fields. The interface accepts at most 64 boxes, 256
positive and 256 negative points, 16 text categories, 1,024 text characters and
32 KiB of box JSON. A request needs a positive point, a box or a text category.
The source is copied and EXIF-oriented before coordinate validation, with the
existing 16-megapixel source limit.

**Maximum instances** is between 1 and 64. Prompt counts are checked before
queueing, including the point mask when text and points are combined. The group
also has a 64-megapixel mask limit and a 256 MiB returned-image limit. Float RGBA
output therefore admits fewer instances than an eight-bit grayscale mask at the
same source size. Capture records the effective limit; output validation checks
it again before conversion. No oversized output is silently truncated.

The native detector returns its own ordered masks and detected text boxes. Union
mode keeps one mask/cutout, including a black mask or transparent cutout when text
finds nothing. Individual mode can finish with zero text matches. Point and box
masks keep their native minimum count. Per-instance files retain native order,
zero-based instance index, total result count and model basename/size. Detected
boxes/scores remain in source coordinates. Cutouts multiply native masks by the
source alpha. PNG16, float EXR and still AVIF continue through the existing native
image-file transport; no eight-bit preview replaces a precision result.

The API uses the existing utility route:

```json
{
  "controlnet": { "type": "segment" },
  "input_image": "<inline image>",
  "native_vision_model": "<installed SAM safetensors>",
  "segment_text": "person:3, dog:2",
  "segment_boxes": "[]",
  "segment_positive_points": "[]",
  "segment_negative_points": "[]",
  "segment_individual_masks": true,
  "segment_max_instances": 8,
  "segment_output": "cutout",
  "n": 1
}
```

`n` remains one source run. The private captured `OutputPlan` and returned
`ImageOutputBatch` describe a separate result count. A completed empty native
batch has a completion receipt; a bare empty list, cancelled run, swallowed error
or stale request receipt cannot masquerade as a successful no-match result.
Text, boxes and points are explicit request inputs, not inherited API defaults.
Scalar instance preferences can be saved. Model paths in output metadata are
basenames rather than local folder names.

The existing image compute lane owns admission and cancellation. A prepared-file
read lease covers both the detector and its paired text encoder. Model caching
uses the captured file signature and utility kind. A text request upgrades a
point-only cache through the native checkpoint loader; replacing/leaving the
utility releases both detector and encoder patchers. A missing paired encoder
does not leave a partially initialized cache. Cancellation is checked around
inference and while converting each result.

The pinned six intrinsic utility classes receive a current header/signature-bound
role from `Models.get_models_by_path`. Cached publisher labels and cached copies
of the role cannot relabel a changed file. Create's ordinary checkpoint cards and
API image-model discovery exclude that positive utility role. Unknown local
models remain visible. The complete file inventory and configured model roots
remain intact. Typed catalogue utility rows appear under **Image tools**, even
when stored in `diffusion_models`. Installed RT-DETR/SAM Use actions validate the
current header and atomically select the matching named tool plus exact file;
they do not change the generation checkpoint or run a model. Other utility
payloads stay acquire-only until their actual tool adapters exist.

Utility admission also works in a cold process before Torch or Comfy modules are
imported. It reuses the bounded safetensors header reader and the pinned native
detector function bodies with native `BASE.matches`; only the six utility
configurations can produce a role. Unknown headers remain unclassified. The
header and predicate caches bind both the model identity and the native source
file signatures, so changing either invalidates a previous positive result.
Browsing and installed-tool activation read neither tensor payloads nor model
hashes and do not initialize the tensor runtime.

Native source is ComfyUI `fca60fe70e5b84c5c6b058263653d15cb3401e44`:
`comfy_extras/nodes_sam3.py`, `comfy/text_encoders/sam3_clip.py`,
`comfy/supported_models.py`, `comfy/sd.py` and `nodes.CLIPTextEncode`.
CPU checks execute those native parser/node bodies with tiny model stand-ins,
the actual worker PNG path, public API schemas and Gradio callbacks. A separate
local receipt, `cache/plan-execution-20261111/U1-real-utility-cold-admission.json`,
binds actual installed RT-DETR/SAM3.1 headers to pinned native shape classification,
stale-SDXL correction, picker/capture exclusion and exact tool activation in a
process with Torch and Comfy imports blocked. Header reads
are guarded against tensor payload access and leave model/cache bytes unchanged.
These are source/CPU contracts. Rendered full-app behavior, real model masks,
quality, device memory fit and GPU stability require root's separate acceptance.

```sh
python tools/check_cpu.py test_sam_editor_results.py test_sam_prompt_utilities.py test_editor_session.py test_guided_result_ui.py test_native_vision.py test_native_image_output_transport.py
```
