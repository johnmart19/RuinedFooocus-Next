# Native still-image histograms

**Create → Image tools → Histogram analysis** analyzes the current source only
when **Analyze histogram** is pressed. Its closed-by-default panel shows RGB
composite, BT.709 luminance and separate red, green and blue series, plus the
complete numeric result. Changing or clearing the source clears the chart and
cancels outstanding analysis. Cancel analysis drops the result; it cannot
interrupt the native node inside its single bounded CPU calculation.

No checkpoint, prompt, sampler, model download or generation defaults are used.
The analysis returns data beside the source; it does not replace the source or
write a generated image, PNG recipe, Library item or temporary chart file.
Replay means submitting the source again. A result contains no reconstructable
source pixels or local file path.

## Native source and color contract

The adapter calls `comfy_extras.nodes_curve.ImageHistogram.execute` at pinned
ComfyUI `fca60fe70e5b84c5c6b058263653d15cb3401e44`, reviewed 2026-11-11.
This is a CPU tensor/NumPy node; no model/device selection is necessary.
`features/image_processing/image_histogram.py` owns immutable capture, output
validation and cancellation. `ui_image_histogram.py` mounts beside the source.
The existing serial external-job lane carries both UI and API requests.

Capture EXIF-orients one image of at most 16,777,216 pixels and freezes RGBA8
bytes, oriented dimensions and a SHA-256 identity over four-byte little-endian
width, four-byte little-endian height, then those bytes. Native input is one
RGB float32 image divided by 255 on CPU. The original alpha bytes remain in the
snapshot and its identity; all pixels count equally, including RGB stored under
zero alpha. Alpha is not a histogram channel or a weight.

RGB, RGBA, grayscale, binary and palette inputs are converted to encoded RGB8.
The result identifies the decoded input mode, RGB8 working mode and whether an
ICC profile or alpha was present. No profile conversion, assumed sRGB tagging
or transfer-function linearization occurs. Unsupported high-bit, float and CMYK
images are rejected by direct/API capture. Gradio's existing source upload decodes
to RGBA8 before the panel receives it; the reported UI input mode describes that
decoded source, rather than the original file's precision/profile history.
For inline API files, the shared native reader inspects the encoded frame's bit
depth before Pillow conversion. This includes color PNG16 and AVIF10, which
Pillow can expose as RGB8; they are rejected before queueing or creating files.

Each output has 256 bins. Native quantization clips `value * 255` to 0–255 and
truncates to uint8. The RGB composite is the per-bin integer floor of the three
channel counts' average, so its sum can be smaller than the pixel count. Native
luminance uses weights 0.2126, 0.7152 and 0.0722 on the encoded float RGB values
before quantization. It is not a linear-light photometric measurement. Each
individual channel and luminance sums to the source pixel count.

## Typed API

`POST /v1/images/histogram` uses the existing optional bearer authorization and
serial queue/cancellation rules. JSON accepts exactly `input_image`, as a base64
image, image data URL or unexpired image URL previously returned by this API.
No arbitrary URL is fetched. Existing inline-image byte/pixel limits apply;
generation fields are rejected. A disconnected client cancels the job and a
cancel check before and after the native call prevents publishing partial data.

```json
{ "input_image": "data:image/png;base64,..." }
```

The response has `object: "image.histogram"`, `native_node`, `native_revision`, `source`,
`quantization`, `rgb_composite`, `luminance_weights` and `bins`. The last object
contains exactly `rgb`, `luminance`, `red`, `green` and `blue`, each a typed
256-element integer array. `source` records oriented width/height, input and
working modes, alpha/profile presence and handling, and the frozen pixel digest.
The OpenAPI response schema describes this complete data contract.

## Verification and limits

`python tools/check_cpu.py test_image_histogram.py test_depth_utilities_ui.py
test_api_image_edits.py` checks the actual pinned native histogram body on CPU,
transparent pixels, BT.709 bins, integer composite rounding, source freezing,
EXIF/mode boundaries, cancellation, malformed native output, authenticated API
closure and real Gradio event/host mounting. Native weights and GL are unused.
This source/CPU evidence does not certify rendered browser behavior or other
curve/shader operations. Curves require an actual LUT consumer and are a separate
feature.
