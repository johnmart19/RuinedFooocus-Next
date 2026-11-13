# Native precision still-image files

Create → **Image file output** selects PNG, OpenEXR or still AVIF. Ordinary
8-bit PNG keeps its existing saving behavior. Explicit precision output retains
the decoded floating point pixels through native generation, native pixel upscale,
Depth Anything 3, detection and segmentation; quantization happens at the encoder.
Other pipelines that already return an 8-bit image can export that image, but a
higher bit depth does not restore detail that was already discarded.

| Format     | Native depths                     | Input color space                            | Alpha               |
| ---------- | --------------------------------- | -------------------------------------------- | ------------------- |
| PNG        | 8-bit, 16-bit                     | sRGB                                         | Preserved           |
| OpenEXR    | 16-bit float, 32-bit float        | sRGB, HDR/HLG, linear Rec.709, linear ACEScg | Preserved           |
| Still AVIF | Auto, 8-bit YUV420, 10-bit YUV420 | sRGB, HDR/HLG, HDR PQ                        | Rejected explicitly |

EXR stores scene-linear values: the pinned encoder applies the inverse sRGB or
HLG transfer function when requested. Linear inputs pass through. Its native
chromaticities metadata identifies Rec.709, Rec.2020 or ACEScg primaries. EXR32
retains negative and above-one finite samples. EXR16 rounds to half precision.
AVIF uses the native SVT-AV1 encoder, CRF 1–63, minimum 64 pixels per side, and
exactly one still frame. The native encoder owns transfer/color tags and YUV420
subsampling. This format is lossy even when its source has more precision.

These are adapters around pinned ComfyUI `SaveImageAdvanced`'s `_encode_image`,
metadata injection and `_save_avif`; RF does not implement replacement image
codecs. The source pin is `fca60fe70e5b84c5c6b058263653d15cb3401e44`, inspected
2026-11-11. Existing PyAV supplies the codecs; no OpenEXR Python package or
automatic dependency/model download is added. Missing native codec dependencies
remain an execution error, separately from source support.

## Library and request transport

Library → **Import or export a still image** copies a PNG, EXR or still AVIF into
the configured outputs without changing the uploaded original. The importer
validates format signatures, one-frame decoding, dimensions and metadata bounds;
files are limited to 512 MiB and 16,777,216 pixels. The byte limit includes room
for a full 256 MiB RGBA float32 image plus codec/header overhead, and applies to
both native outputs and external imports. EXR header metadata is bounded
to 2 MiB. The original bytes and native precision remain the indexed source.
Single-frame RGBA8 AVIF imports also retain their original bytes and alpha.
The reader recognizes the native Color/Alpha component pair, validates one
8-bit frame per component on the same bounded canvas, then uses Pillow's AVIF
codec to reconstruct RGBA. Unrelated streams and animations remain rejected.
Higher-bit auxiliary-alpha AVIF is refused before Pillow can reduce precision;
use PNG or EXR for that input. RGB10 AVIF and native PNG/EXR precision imports
retain their existing paths. Native AVIF export still rejects RGBA.
PNG16, EXR and AVIF display through derived PNG previews under the configured
cache. Preview identity includes original path, size, modification and creation
times. Library actions retain the original file identity; image tools receive
the derived display conversion for EXR/AVIF. Preview generation does not replace
the original file. Original/Exported file downloads retain their actual formats.
Create exposes **Original image files** separately from its display gallery;
precision previews have no competing download action. Original/display mappings
belong to the originating job and browser session. Selecting a result retains
the original file identity. A subsequent image tool reads that original at full
resolution, then makes its explicit 8-bit working conversion; it never uses the
cached, possibly smaller display PNG as its source. This does not claim floating
point input support for image tools that currently accept 8-bit working images.

Publication reuses the preparation service's no-clobber file contract: atomic
Windows rename, or same-filesystem POSIX hard-link publication. Existing files
are refused. POSIX filesystems without hard-link support cannot provide this
publication contract and return an error instead of replacing a destination.

The selected image can be exported to a new cache file. Library converts its
known source color space to the selected encoder input space with the pinned
native color-space node. EXR to PNG/AVIF uses native display tone and gamut
mapping; choosing linear EXR preserves linear samples. Scene-linear Rec.2020
uses the native primaries matrix before this conversion because the save node
has no corresponding linear input option. Ordinary JPEG/WebP sources may also
be exported, retaining their actual 8-bit source depth.

JSON image requests accept `output_format`, `output_bit_depth`,
`output_color_space` and `output_crf`. The same fields are accepted in multipart
edits. Defaults are `png`, `8-bit`, `sRGB`, CRF 18. Format/depth/space combinations
are validated before queueing. API URLs and `b64_json` return the original
precision file; EXR uses `image/x-exr` and AVIF uses `image/avif`. Saved metadata
records effective choices, source dtype and alpha and survives Library setup
replay. Decoder output, optional upscaling, export and preview remain serialized
with the existing image job; no additional model or GPU worker is created.

Histogram, filter and curve API inputs have an explicit RGB8 working contract.
Their opt-in admission check reuses this reader's native codec/frame precision
metadata on the bounded inline bytes, before Pillow conversion. Color PNG16,
AVIF10 and floating point files are rejected with an 8-bit-source error before
queue admission, output writes or temporary-file creation. The inspection uses
CPU PyAV without an external probe or a hardware decoder. Other API decode
routes retain their existing original/working-copy policies; precision output
and Library original-file import/export are unchanged.
Inline inspection covers the primary stream and optional auxiliary alpha,
preserving ordinary RGBA8 AVIF input. Missing header precision falls back to
each component's first frame; unknown precision is rejected. The original-file
importer's separate one-frame admission stays strict; its only multi-stream
case is the validated 8-bit AVIF Color/Alpha pair described above.

Local `.latent` and saved conditioning are separate tensor contracts. They are
not image imports, and no pickle/unrestricted local tensor importer is added by
this packet. Their [dedicated generation UI/API consumer](native-tensor-generation-io.md)
and [saved workflow contexts](native-tensor-workflow-contexts.md) validate native
layouts and original source identities separately from image-file transport.

## Verification and limits

Run with the checkout interpreter:

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_native_image_formats.py test_native_vision.py test_gallery_actions.py test_gallery_scan.py test_gallery_scheduling.py test_gallery_index.py test_api_live_contracts.py test_api_image_edits.py test_image_upscale_metadata.py test_exact_upscale_target.py
```

The real pinned codec routines run on tiny CPU fixtures. Tests cover PNG16
sub-byte RGB/alpha values; EXR32 negative/HDR/alpha values; native EXR half rounding
and sRGB conversion; native still AVIF and embedded metadata; original-preserving
imports; precision decoder output; derived previews; actual Gradio format events;
API MIME/base64; and direct Library/geometry callers. PyAV 19.0.2 exposes half-float
EXR frames as `gbrapf16le`, whose generic conversion can quantize values. The
import adapter reads those native half planes and row strides directly rather
than converting them through an integer scaler.

This CPU evidence does not qualify rendered browser behavior, model semantics,
wide-gamut/HDR visual accuracy, device stability or any particular model output.
Those remain distinct source/artifact-bound acceptance checks.
