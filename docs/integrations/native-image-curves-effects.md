# Curves and bounded native GLSL effects

**Create → Image tools → RGB curves** edits master RGB, red, green and blue
point tables. Use 2–32 normalized input/output pairs per channel. Input values
start at 0, end at 1 and increase by at least 0.000001; output values can rise or
fall within 0–1. Choose linear or monotone cubic interpolation. Master RGB
runs first, then the separate channels. Add one source and press Generate.

**GLSL image effect** offers three bundled recipes: **Invert RGB**, **BT.709
grayscale** and **Posterize**. Amount 0 preserves source RGB; amount 1 applies
the full effect. Posterize accepts 2–32 levels per channel. Both operations
remain off until selected and use the source dimensions and alpha. The source
limit is 4,194,304 pixels with no side longer than 4096 pixels. No generation
checkpoint, prompt, sampling recipe or model download is required.

The effect programs are RF-authored, bounded recipes with native-GL and finite
CPU consumers. `--cpu` and the initialized backend's CPU mode are authoritative:
these operations run on CPU without importing ANGLE, EGL/GLES or PyOpenGL. An
accelerated request with missing GL packages goes directly to CPU; GL
initialization/rendering failures also fall back to CPU. Both automatic fallback
cases warn in the terminal. There is no general CPU shader interpreter. Neither
UI nor API accepts shader code, shader paths, arbitrary uniforms, extra images,
alternate output slots or custom shader output sizes.

## Native ownership and actual consumer

Reviewed source is ComfyUI `fca60fe70e5b84c5c6b058263653d15cb3401e44`,
2026-11-11. `features/image_processing/image_shaders.py` owns immutable capture
and delegates native computation through the shared serial image utility lane.

- `comfy_extras.nodes_curve.CurveEditor.execute` parses the four validated point
  sets through native `CurveInput.from_raw`. The adapter does not implement
  interpolation. `CurveEditor` alone does not modify pixels.
- For native GL, `comfy_extras.nodes_glsl.GLSLShader.execute` is the pixel consumer. It
  calls each native curve object's `to_lut(256)`, casts the LUTs to float32 and
  passes them to its native rendering routine. The bundled curves program samples
  these four textures in master/red/green/blue order. Its coordinate mapping
  samples the LUT's texel centers, including the endpoints.
- The named recipes use exactly one native image and bounded amount/level
  uniforms. Each program declares only output location zero and one pass. The
  native node returns four IMAGE tensors; RF validates that contract and publishes
  output zero only. This accounts for the smaller source-size limit.
- A local subclass inherits the native execute body and overrides only native
  graph-preview output construction with an empty dictionary. RF owns preview
  and artifact writing. This prevents native graph UI code from saving duplicate
  temporary source/output images; no native renderer or interpolation is replaced.

`image_effect_cpu.py` supplies the bounded CPU consumers. Curves reuse native
`CurveInput.to_lut(256)` and sample the same float32 LUTs with PyTorch
`grid_sample`, linear filtering, border clamping and `align_corners=True`. This
matches the GLSL program's `(value * 255 + 0.5) / 256` texel-center mapping,
including endpoints. Master/channel order and final clamping are the same. Native
interpolation is not copied. Invert reuses native `ImageInvert`; amount blending
uses native `ImageBlend` in normal mode. BT.709 grayscale uses the installed
Kornia RGB operator with explicit 0.2126/0.7152/0.0722 weights. Posterize uses the
finite recipe's `floor(color * (levels - 1) + 0.5) / (levels - 1)` tensor math;
native palette quantization and bit-dropping posterizers have different semantics.
Both consumers retain float32 values until the shared output writer.

Bundled program bytes are normalized to LF and verified against reviewed digests
at capture. This binds the finite CPU math to the exact static GLSL recipes. A
modified program is rejected rather than silently interpreted differently by CPU.
The captured program and digest remain immutable after admission.

The native GLSL module preloads `comfy_angle` EGL/GLES and PyOpenGL on import.
RF imports it only during accelerated processing, after CPU policy, passive
package-presence and cancellation checks. Panel mounting, source changes, metadata
replay, API capture, resource scans, CPU execution and known-missing GL packages
do not import it. There is no automatic install or device probe. If present
packages cannot initialize or render, the terminal reports CPU fallback and the
finite CPU consumer completes the request. Invalid native GL output also falls
back; cancellation remains terminal and never becomes a retry. The native GL
implementation owns its context, cleanup and rendering. The configured
image/PyTorch backend alone does not identify its physical adapter.

**Actual EGL/GLES import, shader compilation/rendering, OS/driver/device selection,
CPU/GL pixel equivalence, rendered visual output and GL performance remain unqualified in the current source/CPU
phase.** They require the later explicitly authorized platform/GL acceptance
pass. No GL dependencies were installed, initialized or probed for this packet.
Exposing the explicit source route does not establish hardware support or failure.

## Source, cancellation and outputs

Capture uses the bounded histogram RGBA8 snapshot: EXIF-oriented pixels, original
decoded mode, profile/alpha presence, dimensions and a dimension-aware SHA-256.
Direct/API high-bit, float and CMYK sources are rejected. Gradio source uploads
are already decoded to RGBA8. RGB remains encoded; no ICC conversion, assumed
sRGB conversion or transfer-function linearization occurs. BT.709 grayscale
uses encoded RGB weights and is not a linear-light photometric measurement.

Control points, active scalar options, the bundled UTF-8 program and its SHA-256
are frozen before admission. Later source/table/recipe edits cannot change a
queued operation. Native input is one RGB float32 tensor. Output shape/count and
finite pixels are validated, and original alpha bytes are restored. Existing
PNG8 or precision PNG16/EXR writing handles the result without resizing. These
RGBA utility outputs reject native still AVIF at capture, before processing,
because that native encoder cannot preserve alpha. No alpha is silently dropped.
Source alpha participates in source identity even though color operations ignore
it. Existing precision export's codec/color qualification limits still apply.

Cancellation is checked before/after native curve/LUT calls and each bounded CPU
operator or sampler, around GL import/rendering when requested, before fallback
and before publishing output. Native GL and individual tensor calls are atomic
between checks. Cancellation publishes no output provenance or completed artifact;
a failed GL call may complete through CPU. Utility-only metadata identifies the
actual CPU/native-GL backend, operators, fallback status, source/program digests,
active parameters, native pin, output slot, LUT size, alpha/color policy and
dimensions. It contains no local paths or fictitious generation/sampling recipe.

## Typed API and replay

JSON `/v1/images/generations` and multipart `/v1/images/edits` accept
`controlnet.type` of `image-curves` or `glsl-effect`, bypassing generation-model
resolution. Provide one `input_image` or `image`/`image[]` upload. `n` must be one;
an explicit size must equal the oriented source. Masks, borders and extra
references are rejected. Existing bearer authorization, upload limits, output
format/response controls and disconnect cancellation remain in force.

```json
{
  "input_image": "data:image/png;base64,...",
  "controlnet": { "type": "image-curves" },
  "curve_master": [
    [0, 0],
    [0.5, 0.65],
    [1, 1]
  ],
  "curve_interpolation": "monotone_cubic",
  "response_format": "b64_json"
}
```

Optional `curve_master`, `curve_red`, `curve_green` and `curve_blue` default to
identity point sets. `curve_interpolation` is `linear` or `monotone_cubic`.
Named effects accept `shader_effect` (`invert`, `bt709-grayscale`, `posterize`),
`shader_amount` (0–1) and, for posterize, `shader_levels` (integer 2–32). Multipart
point tables and numeric fields are JSON text; interpolation/effect names are
plain text. OpenAPI carries bounded request types. Arbitrary shader fields are
rejected.

Library Apply/Reuse restores point tables/interpolation or the named effect and
active parameters. It clears source uploads and utility component selectors,
disables inpainting/reference adapters/recipes/automatic upscaling, and does not
resolve a generation checkpoint. A source digest cannot recreate absent pixels;
provide a source again before running.

## Source/CPU verification

The feature descriptor declares the exact affected suites and prerequisite
closure. `test_image_shaders.py` checks actual CPU pixels for all named recipes,
identity/partial amounts, alpha, native interpolation and LUT endpoints/interior
texel sampling, plus the actual serial worker writing CPU results and provenance.
It executes pinned native `ImageInvert`/`ImageBlend`/`CurveInput`/`CurveEditor`
bodies and the installed CPU Kornia operator. CPU-authoritative policy, absent GL,
failing GL, invalid GL output and cancellation are checked without native GL
module initialization.

Separate native-call-body tests execute `GLSLShader.execute` and its source
output/pass parsers. The GL render boundary is a
tracer returning deliberately fixed, non-semantic image values. It proves that
native 256-entry LUTs, one-image batches, source dimensions and named uniforms
reach the real node body; **it does not evaluate shaders or prove their rendered
pixel semantics**. No `nodes_glsl.py` top-level code or GL dependency executes.

Other checks cover passive import guards, real Gradio point-table preprocessing
and host mounting, immutable programs/points/pixels, bounds, cancellation and
native errors, typed JSON/multipart/API capture, actual serial worker PNG/metadata
writing, precision output and Library replay. Platform/GL and rendered browser
acceptance are separate required evidence.
