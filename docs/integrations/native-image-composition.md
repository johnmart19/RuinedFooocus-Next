# Still image grids and composition

[Finite image workflows](image-workflows.md) capture the completed background and
owned foreground as separate editor artifacts before calling this same native
composition service. Each successful composition appends its own source/target
lineage; the sequence adds no composition algorithm or parallel compute lane.

The [owned editor history](editor-history.md) adapter resolves captured
source/overlay/mask artifacts before this operation and retains its ticket through
the utility worker packet. Pure composition retains its separate execution metadata.

Create → Image tools provides **Make image grid** and **Compose images**. Add
images to **Ordered input images**; select an item and use Move selected earlier
or later to change its position. Grid reads every item in order. Composition
requires exactly two: base, then overlay. Generate performs one operation. It
does not use the selected generation checkpoint, prompts, LoRAs, model recipes,
size recommendations, sampling or automatic upscale. No model files are needed.

Library's **Start image grid** and **Start image composition** copy the validated
selected original into the first input slot and open Create. Add more images
there before Generate. Full setup restores scalar composition and file-format
choices, and clears uploads and mask. Saved pixel identities cannot reconstruct
source images; originals remain unchanged. Save current defaults stores only
scalar controls, never input images or masks.

## Framing and alpha

Uploads are oriented, copied to 8-bit RGBA working pixels and treated as sRGB.
Precision Library originals first use the existing explicit sRGB working-copy
conversion. Composition is not a floating-point image importer; float output
retains the precision of the native image math over these byte inputs. Original
files are retained. Geometry uses uniform contain, LANCZOS resampling and centered
padding; it never stretches or implicitly crops an input.

- Grid cells and spacing follow native ImageGrid bounds: 1–20 columns,
  32–2048 pixels per cell dimension and 0–50 pixels spacing. Source alpha is
  flattened onto black; grid padding, unused cells and output are opaque RGB.
- The base determines the composition canvas. Left/top start inside the base.
  Zero rectangle width/height uses the remaining canvas in that dimension.
  An explicit rectangle must fit wholly inside the base. The overlay fits that
  rectangle with transparent padding. A mask must match the oriented overlay;
  it is fitted using exactly the overlay's scale and offset. White selects,
  black preserves; grayscale gives fractional selection.
- **Masked replace** uses native `ImageCompositeMasked`: every RGBA channel is
  interpolated by the mask (`source × mask + destination × (1 − mask)`). It is
  a channel replacement, including alpha.
- Porter–Duff modes come from the pinned native enum. The overlay is placed on
  a full transparent canvas; its alpha is multiplied by the selection mask.
  Native `PorterDuffImageComposite` receives RGB and inverted alpha masks and
  performs its own premultiplied-alpha formulas. **SRC_OVER** is the default.
  Mode behavior outside the overlay follows that native full-canvas formula.

PNG supports byte output or native 16-bit export; EXR supports native float
export. RGBA composition rejects still AVIF because the native encoder has no
alpha support. RGB grids can use still AVIF when both dimensions are at least
64 pixels. File-format validation and atomic new-file publication reuse
`native_image_formats`; display previews remain separate from originals.

## Ownership, limits and native contract

`image_processing.image_composition` owns both operation names, scalar settings,
request validation and immutable captures. UI uses an interactive PIL Gallery
restricted to images. The capture rejects paths, animation, empty lists, more
than 32 images, sources over 16,777,216 pixels, combined sources and mask over
33,554,432 pixels, or output over 16,777,216 pixels. Bounds are checked before
copying the next input. The public API also limits each inline string to
28,000,000 characters and the combined strings to 56,000,000 characters before
decoding. No arbitrary filesystem paths or remote URL fetches are accepted.

Copied RGBA/L bytes, ordered pixel SHA-256 identities, scalar parameters, output
size and native source signatures belong to one queued request. Later uploads,
reordering or UI edits cannot mutate it. The worker retains only those inputs,
file transport and request lifecycle fields. It has no image-result cache.
The existing serialized image resource lane owns processing and cancellation;
this feature enables no parallel compute lane. Cancellation is checked between
images, before/after native operations and before metadata publication. A
single native tensor operation finishes before the next cancellation check.

`composition_native` selects unchanged pure bodies from ComfyUI pin
`fca60fe70e5b84c5c6b058263653d15cb3401e44`:

| Native source                       | Reused operation                                                  |
| ----------------------------------- | ----------------------------------------------------------------- |
| `comfy_extras/nodes_dataset.py`     | `ImageGridNode._group_process`, PIL/tensor conversion             |
| `comfy_extras/nodes_mask.py`        | `ImageCompositeMasked.execute`, `composite` with multiplier 1     |
| `comfy_extras/nodes_compositing.py` | enum, `porter_duff_composite`, `PorterDuffImageComposite.execute` |
| `node_helpers.py`, `comfy/utils.py` | alpha channel matching, batch repeat                              |

The bounded AST bridge avoids importing the dataset/training module or model
runtime. Capture and enum discovery do not initialize Torch or Comfy. Processing
imports Torch with an explicit CPU device and inference mode. Only parsed native
source is cached, keyed by canonical source signatures; a fresh per-request
namespace owns callbacks and images. Changed native files invalidate a queued
capture rather than mixing operation versions. Native source is trusted pinned
installation code, not an uploaded program.

Worker metadata records operation, scalar recipe, file format, output size,
ordered working-pixel identities, alpha policy and actual fit rectangles. It
does not record private paths, uploaded pixels or fictitious generation fields.
Library replay restores scalars while requiring fresh uploads. Ordinary Library
actions clear composition inputs. Latent/tensor composition is a separate
context-sensitive feature and is not exposed by these image operations.

## API and checks

POST `/v1/images/generations` with `controlnet.type` equal to `image-grid` or
`image-composite`, ordered `composition_images` inline base64 strings, and
`n: 1`. The control object accepts only `type`. Optional `composition_mask`
is an inline grayscale selection image. Grid fields are `grid_columns`,
`grid_cell_width`, `grid_cell_height`, `grid_padding`; composition fields are
`composition_mode`, `composition_x`, `composition_y`, `composition_width`,
`composition_height`. Optional `size` must equal the actual output canvas or
`auto`. File output uses the existing output-format fields. The endpoint bypasses
generation model resolution; mixed edit/reference/recipe/upscale inputs are
rejected. For example:

```json
{
  "controlnet": { "type": "image-composite" },
  "composition_images": ["BASE_INLINE_BASE64", "OVERLAY_INLINE_BASE64"],
  "composition_mode": "SRC_OVER",
  "composition_x": 16,
  "composition_y": 16,
  "n": 1,
  "response_format": "b64_json"
}
```

Focused guarded checks:

```sh
python tools/check_cpu.py test_image_composition.py test_image_composition_integration.py test_image_composition_ui.py
```

They execute actual pinned native pure bodies, ordered full-frame/alpha math,
capture immutability, limits, cancellation/source changes, real PNG worker
publication, typed ASGI dispatch, Library replay and installed Gradio events.
No model, GPU, real browser render or hardware throughput is qualified by them.
