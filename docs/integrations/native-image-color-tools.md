# Native color finishing

The **Color** group of **Adjust image** provides native image blending,
reference color transfer, palette quantization, text, morphology and SDR-to-HDR.
These actions need no model
or prompt. Blend requires a reference with the same oriented dimensions; resize
explicitly first. Color transfer uses reference statistics and accepts a
different reference size. Quantize uses only the main image.

Blend exposes normal, multiply, screen, overlay, soft light and difference with
an amount control. These are the pinned node's exact semantics: difference is
signed subtraction followed by clamping, and every blend clamps RGB to [0, 1].
Source alpha is retained; reference alpha is not a selection mask.

Color transfer exposes Reinhard Lab, MKL Lab and histogram methods with strength
from zero through ten. Zero strength is identity. The Lab algorithms use float
color math; histogram matching uses native 256-bin channel lookup tables. Native
transfer clamps RGB to [0, 1]. Transfer requires encoded sRGB samples in [0, 1];
convert color and tone-map HDR explicitly first. The adapter performs no hidden
ICC conversion or tone mapping.

Quantize creates a palette with one through 256 RGB colors and native none,
Floyd–Steinberg or Bayer 2/4/8/16 dithering. This is palette quantization, separate
from the existing per-channel GLSL Posterize effect. Native quantization
explicitly uses uint8 RGB; source alpha stays at its captured precision.
Quantize requires encoded sRGB RGB values in [0, 1].

Text overlay uses the pinned default Pillow font, with text color, top/bottom
position, left/center/right alignment, relative font size and optional black
outline. It uses native word wrapping and shrink-to-fit behavior; literal `\n`
and `\t` escapes are expanded by the native node. There is no font-file chooser
or font download. The bundled default font has limited character coverage.
Text overlay requires an explicit **8-bit sRGB source** and refuses uint16,
floating and HDR originals. Make a deliberate working copy when needed.
Its native RGBA source-over operation unions glyph alpha with source alpha, so
text can appear over transparent regions. Alpha outside glyphs is unchanged;
the native compositor clears hidden RGB where the resulting alpha is zero.
The original source is unchanged. Text is bounded to 2048 characters and 64
lines, with an additional font-work limit before rendering.

Morphology exposes native erode, dilate, open, close, gradient, bottom hat and
top hat on RGB. It preserves source alpha exactly, including floating alpha,
and retains native uint16/floating input precision through float processing.
Kernel sizes from 3 through 999 include even sizes as supported upstream. A
64-million spatial-work budget (pixels × kernel area × native passes) rejects
oversized work before allocating the native kernel. Reduce image or kernel size
when a request exceeds the budget. This is image-color morphology; use the Mask
group for selection growth and shrinkage.
Out-of-range floating RGB requires EXR so the export cannot silently clip it.

SDR-to-HDR expands encoded sRGB using the native Lottes algorithm into Rec.2020
HLG. The target luminance is 0–10000 nits; zero retains the native calibrated
rendering and the default is 650. The source must contain finite SDR RGB values
in [0, 1]. Source uint16/floating precision and alpha are retained; source alpha
is not transformed. Choose **EXR** and the **HDR** output color space. The native
EXR writer receives HLG samples and stores scene-linear samples. This action
requires a float export and does not pass its result through an 8-bit display
working copy.

## Precision, source and request contract

`ruinedfooocus/features/image_processing/image_color_tools.py` owns `image-color`
and frozen `ColorRequest` records. Geometry's `capture_source` and `FrozenPixels`
retain uint16/floating samples and native input color-space metadata. Blend
accepts floating HDR originals and retains their float samples until the native
algorithm's documented RGB clamp. Original source/reference files and captured
alpha are never overwritten. PNG16 or EXR is required when the operation retains
precision samples; explicit palette-only RGB quantization can return PNG8.
Output color space must match the source, except explicit SDR-to-HDR expansion.
Linear/HDR blending requires EXR;
RGBA cannot use native still AVIF.

Captured source/reference pixels, typed scalar parameters, native signatures
and export choices are immutable. Worker transport retains only the request,
lifecycle tickets, output settings and required progress fields. Replay restores
scalar/output choices and clears both image inputs. Metadata includes source and
reference hashes/frame, source dtype/color space, native operation, alpha policy
and the operation's clamp/quantization precision policy.
The bounded, detached source-generation recipe survives worker transport so the
shared output writer can attach it to the derived result; source images and
source-path authority are excluded from that provenance.

`color_native.py` compiles the pinned `Blend`, `ColorTransfer` and `Quantize`
class bodies and the native `image_alpha_fix` helper into a fresh namespace.
It also extracts the exact `TextOverlay`, `Morphology` and `ImageSDRToHDR`
bodies plus the HDR helper functions and constants from their owning sources.
Schema/server imports are omitted. Worker jobs resolve the applied native
ComfyUI device policy: Auto uses its selected accelerator where supported,
explicit CPU remains authoritative, and unsupported accelerator operations
recover once on CPU with the actual reason in a warning and output metadata.
Selection happens during execution, never while browsing or capturing input.
Native Pillow text rendering and palette quantization remain host operations;
metadata records that capability instead of claiming GPU execution or a failed
accelerator. Progress checks cancellation, and implicit native blend resizing is
rejected. Algorithms
remain the native bodies. Transfer receives RGB only, and the adapter reattaches
the unchanged source alpha. Morphology also receives RGB and restores source alpha;
text receives RGBA for the native glyph alpha union. Native source signatures
are checked before and after
processing; cancellation or invalid output does not publish metadata.
Each accelerator attempt owns fresh tensors, and host transfer is inside the
recovery boundary so asynchronous device failures can trigger the same CPU
recovery. Pixel work and intermediates remain float32, including source alpha;
generation-only fp16-intermediate settings do not lower image-tool precision.
The nested `compute` metadata records requested
policy, actual device/backend, intermediate device/dtype and any recovery or
host-only reason.

## Pinned API and verification

The contract is ComfyUI `989d93d0d968e704c589ce0146bb18c77f6b50cc`,
inspected locally on 2026-11-12:
[native post-processing nodes](https://github.com/comfyanonymous/ComfyUI/blob/989d93d0d968e704c589ce0146bb18c77f6b50cc/comfy_extras/nodes_post_processing.py)
and [alpha helper](https://github.com/comfyanonymous/ComfyUI/blob/989d93d0d968e704c589ce0146bb18c77f6b50cc/node_helpers.py).
The additional native contracts are
[text overlay](https://github.com/comfyanonymous/ComfyUI/blob/989d93d0d968e704c589ce0146bb18c77f6b50cc/comfy_extras/nodes_text_overlay.py),
[morphology](https://github.com/comfyanonymous/ComfyUI/blob/989d93d0d968e704c589ce0146bb18c77f6b50cc/comfy_extras/nodes_morphology.py)
and [SDR-to-HDR](https://github.com/comfyanonymous/ComfyUI/blob/989d93d0d968e704c589ce0146bb18c77f6b50cc/comfy_extras/nodes_images.py).
Existing Torch, NumPy, Pillow and Kornia supply the native dependencies; nothing
is installed or downloaded by these tools.

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_image_color_tools.py
venv/Scripts/python.exe tools/check_cpu.py test_image_color_finishing.py
venv/Scripts/python.exe tools/check_cpu.py test_image_color_device_policy.py
```

Focused tests run actual native CPU bodies on small images/tensors. They cover
all six blend modes, source/reference alpha differences, all three transfer
methods and different reference dimensions, all six dither modes and palette
counts, float alpha/uint16 precision, HDR clamp policy, immutable queued input,
frozen export/replay, stale native sources and early/final cancellation. They do
not establish browser rendering, GPU/model execution, visual quality or
large-image performance.

The finishing tests execute actual native CPU text rendering, all seven Kornia
morphology modes and native Lottes SDR-to-HDR with calibrated/explicit luminance.
They cover transparent and nonopaque source-over alpha, unchanged source pixels,
text escape handling and immutable content, explicit text precision refusal,
even morphology kernels, extended HDR float samples, native work admission,
SDR ramps, uint16/float alpha, frozen HDR EXR exports, replay, stale sources and
cancellation. These tests use only small images and do not qualify display HDR,
the new UI controls, codec output quality or large-image performance.

Device-policy tests run real native CPU callbacks, check that capture does not
select hardware, inspect injected accelerator descriptors without allocating on
them, simulate an unsupported accelerator before any device initialization and
verify the real CPU recovery, and distinguish host-only capability from failure.
They do not establish actual accelerator execution; that is a separate final
acceptance gate.
