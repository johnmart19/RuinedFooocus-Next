# Native layout and attention guidance

Create exposes **Native layout and attention** for ordinary SD1, SD2 and SDXL
checkpoints. Both selectors start **Off**. Expanding the section or choosing
**Refresh guidance files** lists local files; startup and ordinary generation do
not scan these folders. Configure **GLIGEN model folders** and **Hypernetwork
folders** under Settings → Profiles & storage → Paths, save and restart to change
roots. Defaults are `models/gligen` and `models/hypernetworks`. This operation
does not download models or activate them from the Models browser.

Select the checkpoint and compatible local guidance weights deliberately.
Family labels only expose the controls: the loaded native UNet, text encoder
and codec must pass the checks below. There is no automatic model choice or
real-model quality/device qualification for this route.

## GLIGEN text boxes

Choose **GLIGEN model**, then enter **Layout boxes (JSON pixels)**, for example
on a 512 × 512 target:

```json
[
  { "text": "a red cup", "x": 32, "y": 64, "width": 160, "height": 240 },
  { "text": "a blue bowl", "x": 272, "y": 224, "width": 176, "height": 160 }
]
```

The list retains its order and literal text. Coordinates and sizes use target
pixels, start at the top left, must be multiples of eight and must remain inside
the eight-aligned target. Use one through 30 boxes, at most 1,024 characters per
box and 8,192 total text characters. The overall positive/negative prompts stay
separate. Box text does not use RF prompt switching or style expansion.

GLIGEN currently uses a fresh text-to-image canvas. Clear image tools, inpainting,
reference adapters, automatic upscaling and the staged SDXL refiner first.
Scheduled CLIP hooks use the native interval adapter for each box's pooled text.
Each interval preserves matching input ranges, box order and geometry. Constant
CLIP and model-only hooks keep their existing native scope. The
[numeric tensor patch contract](native-tensor-patch-contexts.md) permits locally
receipted GLIGEN latent-only reuse, while refusing executable position-model
conditioning files.

`gligen_layout.py` captures immutable ordered boxes and the selected checkpoint
and guidance file identities. It requires complete native position-network and
fuser shapes, matching UNet SpatialTransformer order/query/context widths, and
the matching unprojected CLIP pooled width. The pinned GLIGEN position normalizer
requires pooled/context widths to agree because it pads the object list using
the context width. The codec must expose four latent channels, eight-times
encode/decode compression and RGB output. Incompatible combinations fail before
sampling; displaying an SDXL selector does not qualify arbitrary SDXL GLIGEN files.

The production path calls `comfy.sd.load_gligen`, then the pinned
`nodes.GLIGENTextBoxApply.append` once per box. Native code converts pixels to
latent units with division by eight and normalizes against the actual target
latent canvas. RF preserves every conditioning entry and interval, leaves the
cached text conditioning untouched and lets the native sampler own positive and
empty-negative GLIGEN model handling.

## Hypernetworks

Choose **Hypernetwork model** and signed **Hypernetwork strength** between -10
and 10. The source must be the native nested Torch state dictionary; flat
safetensors and GGUF are not hypernetwork substitutes. Ordinary SD generation
and its image-to-image/control/inpaint paths can use the transform when their
native attention dimensions match. In an SDXL handoff it affects the primary
model; the refiner retains its own attention. Locally receipted numeric tensor
imports/exports retain the original source and signed strength and rebuild the
native callback for the current request.

`native_hypernetwork.py` validates the pinned activation, paired key/value
networks, layer normalization, dropout flags, floating tensor shapes and dense
parameter bounds before constructing the native patch. Every supplied dimension
must match a loaded UNet attention input, and each network must return that same
dimension. Unsupported activation never becomes a successful unmodified result.
The implementation calls the pinned `load_hypernetwork_patch`, clones the
request sampling model and attaches its native callback to both attention patch
slots. The callback leaves queries unchanged and applies the native signed
key/value transform. Cached weights are shared; strength and model patch lists
belong to the request. Native dropout behavior is retained.

## Capture, lifetime and replay

Only relative filenames inside configured roots are accepted. Files and expanded
Torch ZIP payloads are limited to 2 GiB, with at most 4,096 archive members and
16 MiB of Torch pickle metadata; the ZIP directory is bounded before allocation.
Finite dense-parameter budgets also bound constructed native modules.
Browsing does not deserialize weights. Compute
uses the pinned native safe loader (`weights_only=True` for Torch containers),
with no permissive pickle fallback. Native safetensors headers are bounded and
validated separately. Unknown local files are candidates for native validation,
not proof of model compatibility.

Capture freezes canonical path, size, modification and creation timestamps.
The prepared-asset service verifies them and holds read leases during native
loading and through sampling. Metadata contains the relative name, file size
and decimal-string timestamps, ordered boxes, effective dimensions and strength;
it contains no absolute path, native patch object or source tensor. These are
local change-detection identities, not publisher hashes. A changed source must
be selected again (choose Off, then reselect). Checkpoint/component changes clear
the caches. Turning a feature Off unloads its old cache on the next image request.
Hypernetwork modules follow native model movement and offload after sampling,
including cancellation; GLIGEN uses the native CoreModelPatcher/memory manager.

**Save current defaults** is an explicit opt-in to persist these controls.
Library **Apply metadata** / **Reuse setup**, PNG prompt import and the model
section of a PNG preset preserve source proofs and ordered literal boxes.
Applying metadata without guidance clears previous guidance. A new Library edit
starts with both selectors Off and never inherits a hidden Create patch.
Standalone image utilities drop all generation guidance fields and private plans.

Both features require native image execution. Auto selects native when either
is requested; an explicit optional stable-diffusion.cpp engine is refused.
They share the existing serialized image job and cancellation owner, with no
independent compute lane or global patch mutation. When Off, they perform no
asset lookup, header read, native load, tensor capture or model clone.

## API and source evidence

`ImageRequest` accepts `gligen_model`, `gligen_layout` (the JSON list encoded as
a string), `hypernetwork_model` and `hypernetwork_strength`. Exact replay also
supplies `gligen_model_signature` / `hypernetwork_model_signature` from output
metadata: `[relative_name, byte_size, mtime_ns_string, ctime_ns_string]`.
Fresh requests may omit these proofs and capture the current selected local file.
Omitting a model selector inherits saved controls for the same checkpoint;
an explicit JSON null turns that selector Off and clears its source proof.
The existing prompt, size, sampler, negative prompt and repetition fields retain
their meaning. `/v1/capabilities` describes the field names, bounds and folder
settings without loading or scanning guidance models.

Source inspected 2026-11-11 at ComfyUI
`fca60fe70e5b84c5c6b058263653d15cb3401e44`:

- `nodes.py`: `GLIGENLoader`, `GLIGENTextBoxApply`.
- `comfy/sd.py`: `load_gligen`; `comfy/gligen.py`: `load_gligen`, `Gligen.set_position`.
- `comfy_extras/nodes_hypernetwork.py`: `load_hypernetwork_patch`, `HypernetworkLoader`.
- `comfy/model_patcher.py`: attention patch setters and native model movement.
- `comfy/utils.py`: native safetensors reader and weights-only Torch load.

Focused guarded CPU checks:

```sh
python tools/check_cpu.py test_native_auxiliary_guidance.py test_native_auxiliary_callers.py test_native_auxiliary_ui.py
python tools/validate_feature_cherry_picks.py --feature native-layout-attention-guidance --tests --output cache/native-layout-attention-replay.json
```

Fixtures execute the pinned native box conversion/position normalization,
hypernetwork loader and attention transform, and the full production process
through a sampler stub. API admission, real Gradio events, source changes,
metadata replay and cancellation leases are covered. These checks do not load
production weights, invoke inference, or qualify rendered UI, visual quality,
GPU behavior, performance or arbitrary third-party guidance artifacts.
