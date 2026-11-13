# Bernini-R still editing

Create → Image tools → **Native image edit** accepts the source image, then up to
eight ordered references. Select the signed Bernini-R high-noise image checkpoint;
the **Bernini-R image edit · two stages** panel selects its low-noise companion,
switch step, optional Lightx2v strengths and reference long edge. Existing image
framing, positive/negative prompts and component overrides remain request-owned.
The adapter requires a source without a mask. It exposes no temporal controls.

Standard Model Defaults use 40 steps, CFG 5, `res_multistep` and `simple`, switching
at 20. Optional Lightx2v uses the published model-only LoRA at strengths 3 and 1.5;
choose six steps, CFG 1 and switch three for that example. Enabling Lightx2v does
not overwrite explicit sampling fields. A blank switch chooses the example value,
bounded by the selected total steps. Saved startup fields persist through the
existing generation-default controls. API image edits accept the same six
`bernini_*` fields; existing Chat image edits use the same preparation/worker path.

`features/generation/bernini_image.py` adapts pinned ComfyUI
`989d93d0d968e704c589ce0146bb18c77f6b50cc` and
`blueprints/Image Edit (Bernini-R).json` (SHA256
`da28b3c871ac9f5e09c30ca1ed182bcf3ce29cd028cf2383bd43b99013c42ed2`).
The native `BerniniConditioning.execute` receives exactly one source frame and a
single batch. It produces `[1,16,1,H/8,W/8]` zeros and shares ordered
`context_latents` with positive and negative conditioning: source first, then
reference slots. References preserve their proportions under the native long-edge
cap. Transparent inputs are composited over neutral RGB; output is opaque RGB.
The positive prompt receives the blueprint's image-editing instruction prefix;
the negative prompt remains explicit. Native CLIP layer selection and embedding
trigger expansion reuse the existing consumers.

The official high/low FP8 pair is tied to immutable publisher URLs, full payload
SHA256, payload bytes and bounded header SHA256 in
`resources/bernini_image_components.json`. Matching header and size only identify
a candidate during browsing; shared preparation must verify the signed payloads
before native loading. Generic Wan models remain retired. The low model cannot be
selected as an image primary. The component catalogue uses image-specific IDs,
while preserving the existing physical encoder/codec/LoRA filenames so matching
installed payloads can be reused. Explicit local encoder/codec replacements need
the native UMT5 XXL / RGB Wan2.1 contracts; missing custom replacements fail before
compute rather than becoming download URLs.

Required high/low models, encoder and codec use the existing preparation service,
offline policy, disk reserve, cancellation and read leases. The two UNET payloads
total 31,149,666,432 bytes; the standard complete recipe totals 38,139,379,607 bytes.
These are download/storage sizes, not peak VRAM claims. No asset is acquired by
picker rendering or source capture. An explicit Run uses normal preparation.
The encoder and codec are offloaded after conditioning. The low UNET is loaded
after native sampling releases high-stage GPU residency. `SplitSigmas` preserves
the shared boundary; only high adds seeded noise, low receives zero noise and its
native seed zero. Ordinary captured image LoRAs reuse the existing loader for both
stages. Lightx2v remains model-only. Existing worker progress/cancellation and
output publication own the request; no second compute queue is created.
Primary replacement shares `image_model_lifecycle` with the ordinary pipeline:
old primary/patcher/companion and auxiliary caches are released, followed by
collection and native cleanup, before the first replacement payload is loaded.

Output metadata records source/reference pixel hashes and order, signed stage
identities, framing and stage settings. Replay requires the original source and
ordered references. Switching models clears the private Bernini execution plan;
saved fields remain available for the next deliberate Bernini selection.

Focused check: `python tools/check_cpu.py test_bernini_image.py`. It exercises the
actual capture/preparation caller, signed installed-file tamper rejection, pinned
native conditioner on CPU tensors, stage topology, zero-noise continuation and
cancel-before-low cleanup. Source metadata was inspected on 2026-11-12 from the
[official Bernini publisher](https://huggingface.co/Comfy-Org/Bernini-R) and pinned
blueprint. No weights, GPU probes or generations were performed for this change.
Real-model execution, GPU memory fit, stability, performance and visual quality
remain unqualified. Browser acceptance is tracked separately by the parent task.
