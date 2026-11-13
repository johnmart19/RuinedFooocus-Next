# Pinned native model coverage

Native detection and model loading belong to ComfyUI, pinned at
`989d93d0d968e704c589ce0146bb18c77f6b50cc` (0.45). RF keeps component selection,
preparation, request ownership and operation-specific callers. A native config
class, quantization, checkpoint variant and RF image family are different things.
The active [image/Chat scope](image-chat-scope.md) contains 62 image/utility
configuration rows, 28 logical encoder types and 26 encoder-model types. Full
upstream registrations are source facts, not an RF feature or acceptance target.

## Current source disposition

The former full-native programme is stopped. The [product contract](image-chat-scope.md)
links current scope and preserves the old programme by Git reference. The
source-specific receipts below remain reference material; they do not resume
old model runs, GPU qualification or all-native coverage work.

The [0.45 adoption](comfy045-adoption.md) retains the historical fca inventories
and source/hardware receipts. [WeVisOne](native-wevisone.md) now has an explicit
detection/matting UI/API consumer and reader-proven Hub activation; it is not a
diffusion encoder. [MiniMax H3 Image](native-minimax-image.md) has its own PDD
generation/reference-edit graph, conditioning encoder and image VAE. Source/CPU
integration does not qualify their exact weights, device memory or visual results.

RGB curves now connect native `CurveEditor` LUTs to a bounded `GLSLShader`
consumer; three named static effects share that route. These are source/CPU
integration rows with explicit later GL/platform acceptance, not shader-render
or device qualification. See [curves/effects](native-image-curves-effects.md).

Histogram analysis and explicit blur, sharpen, brightness and contrast have
source/CPU UI/API callers. Their operation rows do not add native nodes or
configurations. The two pure `AdjustBrightness`/`AdjustContrast` bodies are
explicit image-domain exceptions within the dataset module; other training rows
retain their exclusions. See [histograms](native-image-histogram.md) and
[still filters](native-image-filters.md) for their bounded contracts.

The still geometry utility packet adds standalone Lotus depth, MoGe perspective
depth, MediaPipe facial landmark overlays, SDPose full-image whole-body maps and
Marigold V2 albedo. These are distinct named UI/API operations with frozen source
pixels/components and no generation checkpoint/prompt recipe dependency. See
[still geometry utilities](native-still-geometry-utilities.md). The finite
operation inventory records five additional groups without changing the pinned
109 configuration or 707 node denominators. Source/CPU contracts remain separate
from real-weight/device, rendered browser and visual acceptance. DA3 retains its
existing standalone route; MoGe mesh/panorama and pose/video execution are separate.

### Historical pin-bound source receipts

At historical source `f7ecf262` with the fca pin, **60 of 60 reviewed applicable
native configurations have RF source consumers**. That 109-row ledger
also has 49 excluded session-domain configurations (35 video/audio-video,
six audio and eight depth/3D/multiview). Its original four missing consumers
were LotusD, RT_DETR_v4, SAM3 and SAM31; all four now have captured native callers.
The retained baseline is
`cache/plan-execution-20261109/native-config-dispositions-109-v1.json`.
This count describes source integration, not model execution or device support.

RT_DETR_v4 detection and SAM3/SAM31 point segmentation use tracked native
backend/API/UI consumers with shared image admission, copied source/point capture,
file leases and generation-independent controls. See [native vision utilities](native-vision-utilities.md).
Bounded RT-DETR/SAM31 native/app output comparisons passed; native interpreter
teardown logged ON_DETACH warnings. SAM3 model access is gated. These results
do not certify other variants, SAM text/video modes, quality or broad stability.

BiRefNet (`1915c589`), ControlNet/T2I model selection (`d5e44d73`) and native
StyleAdapter/Redux/PhotoMaker (`3a5f14d0`, context follow-up `4c8e20b0`) add
auxiliary consumers outside that 109-config ledger, without inflating its count.
Their source/CPU checks and bounded UI evidence remain distinct from actual
weight execution, provider qualification, image quality and GPU stability.
All five final cohorts through Lotus passed their declared CPU prerequisite
closures; those checks do not load production weights.
The handoff separates passing source/transport/geometry evidence from N04's
failed bottom-band visual result, N06 semantic failure, remaining Chat useful-
answer/reasoning-Off, zoom and provider/hardware gates. Per-snapshot archive and
publication receipts are maintained in the originating checkpoint.
This source count does not certify full plan completion.

[GLIGEN layouts and hypernetworks](native-layout-attention.md) add native
conditioning/attention consumers with captured local sources, contextual controls,
API and replay. The finite operation audit records these two routes separately
from checkpoint configurations. Their CPU/source evidence does not qualify
production weights, visual results or device execution.

The following native detection labels reuse already supported RF runtime classes
and component roles. Cached display metadata remains intact; no weights or
sampling behavior are changed by normalization.

| Native detection label                    | Native runtime class | RF component contract                                 |
| ----------------------------------------- | -------------------- | ----------------------------------------------------- |
| SSD1B, KOALA_700M, KOALA_1B, Segmind_Vega | SDXL                 | CLIP-L + CLIP-G + SDXL VAE; bundled slots stay locked |
| CosmosT2IPredict2                         | CosmosPredict2       | Existing old-T5 + Wan VAE roles                       |

The following additional ordinary image routes use the pinned native loaders
and conditioning. Their sampling recipe must be supplied for the exact
checkpoint; RF does not substitute a generic family recommendation.

| Native runtime   | Text encoder / native CLIP type                                 | Native latent and VAE                                                                           |
| ---------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Chroma           | T5-XXL / CHROMA                                                 | EmptySD3LatentImage,16 channels at1/8; Flux VAE                                                 |
| ChromaRadiance   | T5-XXL / CHROMA                                                 | EmptyChromaRadianceLatentImage,3 RGB channels at full resolution; native pixel-space conversion |
| LongCatImage     | Qwen2.5-VL7B / LONGCAT_IMAGE                                    | EmptySD3LatentImage,16 channels at1/8; Flux VAE                                                 |
| ZImagePixelSpace | Qwen3-4B / LUMINA2                                              | EmptyChromaRadianceLatentImage,3 RGB channels at full resolution; native pixel-space conversion |
| Kandinsky5Image  | Qwen2.5-VL7B + CLIP-L / KANDINSKY5_IMAGE                        | Native Flux16-channel latent and VAE; image-specific template and pooled CLIP-L output          |
| JoyImage         | Dedicated JoyImage Qwen3-VL8B / JOYIMAGE                        | Native Wan2.1 latent/VAE; one image frame                                                       |
| Omnigen2         | Dedicated Qwen2.5-VL3B / OMNIGEN2                               | EmptySD3LatentImage,16 channels at1/8; existing Flux VAE                                        |
| Lens             | GPT-OSS20B + embedded tokenizer / LENS                          | Native Flux2 128-channel,16x latent and existing exact Flux2 VAE                                |
| MingImage        | Ming thinker/connector + embedded tokenizer                     | Native16-channel single-frame latent; dedicated Wan2.1-layout RGBA codec at8x                   |
| HunyuanImage21   | Qwen2.5-VL7B + ByT5 glyph / HUNYUAN_IMAGE                       | Native EmptyHunyuanImageLatent64-channel/32x and dedicated planar codec                         |
| QwenImage21      | Dedicated Qwen3-VL8B / QWEN_IMAGE                               | Native paired text conditioning and64-channel/16x RGBA codec                                    |
| SenseNovaU15     | Native tokenizer sentinel; language weights remain in diffusion | Native full-resolution pixel-space conversion; no external encoder/VAE                          |

`ZImagePixelSpace` has a source-bound [Zeta-Chroma tuple](native-zeta-pixel-source.md)
and the existing Qwen3-4B/pixel-codec component UI. Auto sampling is disabled until
a recipe is qualified. Official L2P uses a different convolutional decoder; the
native converter's ordinary-ZImage fallback is explicitly refused for that source.

Compatible VAE roles offer explicit [full paired approximate codecs](native-approximate-codecs.md),
including replacement of a bundled SDXL codec. Automatic keeps the normal VAE.
Both local sources, native normalization and codec geometry are captured separately
from decoder-only sampling previews; source/CPU proof does not qualify image quality.

`comfy/supported_models.py` owns these runtime/latent configurations;
`comfy/sd.py` selects the family-specific native tokenizer and encoder.
LongCat's native tokenizer supplies its text-to-image template and its encoder
metadata passes through native `CLIPTextEncode` unchanged. This route does not
enable LongCat's separate reference-image edit workflow. Pixel-space routes
construct the native `pixel_space_vae` conversion, never download an unrelated
VAE, and component inventory explicitly says no VAE is required. Original
components, bundled-slot protection and frozen request choices keep the same
preparation/download leases as existing families.
Chroma and LongCat inherit native Flux classes but retain sampler CFG and
negative conditioning: LongCat removes embedded guidance, while Chroma's
separate native distilled-guidance input defaults to zero. RF does not move
their explicit CFG into that input or force sampler CFG to one.

Until a qualified family recipe is recorded, Model Defaults requires all four
validated checkpoint sampling fields: steps, CFG, sampler and scheduler. Published
version settings and saved per-checkpoint overrides keep their existing
precedence. Otherwise choose **Own settings** using the checkpoint publisher's
recipe; explicit settings are preserved. Startup and workflow controls remain
usable and display the unavailable-recipe status; generation still refuses an
incomplete automatic recipe. An API request may instead provide all four fields
(`steps`, `cfg`, `sampler`, `scheduler`); it becomes an explicit recipe and keeps
the usual validation. Partial API overrides do not qualify as complete defaults.
Authoritative native labels for these routes replace stale family labels
for routing while leaving cached publisher provenance unchanged.
CPU contract tests are not hardware or
quality qualification, and these families still need real-model acceptance.

### Lens and Ming Image ordinary generation

The ordinary Create route now admits both native configurations. Lens reuses
native `GPT_OSS_20B` text dispatch, multi-layer stacked features and aligned
attention masks, plus `EmptyFlux2LatentImage`. Its GPT-OSS NVFP4 encoder is an
image companion, not a Chat model. The pinned Lens diffusion config excludes
FP16 because of NaNs; RF retains its native BF16/FP32 eligibility and sampling.
Ming reuses native `MING_IMAGE` dispatch with its embedded tokenizer and quantization
metadata. Its encoder's `direct_context` reaches native model conditioning;
native empty-latent fix-up adds one frame and native decoding returns RGBA.
The dedicated Ming codec requires four-channel image geometry and cannot be
replaced by the RGB Qwen/Wan VAE merely because both have sixteen latents.

The existing immutable acquisition receipts supply five exact source tuples:
Lens BF16, GPT-OSS20B NVFP4, Ming Design INT8 ConvRot, Ming thinker/connector
INT8 ConvRot, and Ming RGBA VAE. Lens shares the existing Flux2 VAE whose published
SHA matches. Registry additions bind the exact download tuple, payload size/hash,
component dependencies and native family; local integrity and runtime acceptance
remain separate. No new weight transfer is performed by this integration.

The immutable official workflows at
[`611c497c`](https://github.com/Comfy-Org/workflow_templates/tree/611c497cca4087967ecd85fd62bc12fa7130e885/templates)
provide the recipes: [Lens base](https://github.com/Comfy-Org/workflow_templates/blob/611c497cca4087967ecd85fd62bc12fa7130e885/templates/image_lens_t2i.json)
uses20 steps,CFG5 and Euler/Simple; [Lens Turbo](https://github.com/Comfy-Org/workflow_templates/blob/611c497cca4087967ecd85fd62bc12fa7130e885/templates/image_lens_turbo_t2i.json)
uses4 steps and CFG1 with the same sampler/scheduler. This exact template's
four-step setting takes precedence over the documentation's approximate eight-step
description. Both recommend native CFGNorm strength1. Lens exposes **Use model CFG
normalization**, enabled on a fresh Lens selection; entering a strength disables
Auto. Unchecked plus blank/None means Off, while explicit numeric values retain
priority. Older profiles with a saved blank/None strength remain Off. The boolean
intent travels with the existing guidance capture, preset and metadata fields.
[Ming Design](https://github.com/Comfy-Org/workflow_templates/blob/611c497cca4087967ecd85fd62bc12fa7130e885/templates/image_ming_image_01_design_t2i.json)
uses12 steps,CFG1,Euler/Simple and native `ModelSamplingFlux` with max_shift1.15,
base_shift0.5 and the actual target dimensions. That native patch runs after
framing has established the effective geometry. Explicit sampling shift and Own
settings retain priority. Recipe-specific private guidance is cleared on every
performance application so another checkpoint cannot inherit it.

Automatic recipes require a current explicit base/turbo/design declaration or
an explicitly verified packaged source identity. Header declarations, signed
source metadata and publisher/user sampling overrides retain their existing
precedence. Passive selection never hashes a model or infers a training variant
from its basename. Unknown variants need complete checkpoint settings or Own
settings. Optional Ming prompt enhancement and reference-image editing are
separate operations; ordinary generation preserves the typed prompt.

`test_native_lens_ming.py` covers actual pinned text dispatch, Lens features/masks,
Ming direct context, native single-frame RGBA decoding, native dynamic sampling,
dedicated roles, source-bound variants and override isolation. Focused CPU checks
also exercise component, registry, image-guidance, sampling and loader callers.
These contracts do not qualify real-model execution, image quality, memory fit,
device speed or GPU stability.

Kandinsky's official image workflow uses both text encoders; its native runtime
requires the CLIP-L pooled output in addition to Qwen context. Its automatic
50-step, CFG3.5, Euler/Simple compatible baseline applies to the native Lite image
configuration (`model_dim=2560`, `visual_embed_dim=64`), including compatible
quantizations and renamed imports. It does not apply to native video configurations.
Published checkpoint recipes and saved overrides still win.
The native image model supplies shift3 without a new sampling patch.
See the [immutable official workflow](https://github.com/Comfy-Org/workflow_templates/blob/611c497cca4087967ecd85fd62bc12fa7130e885/templates/image_kandinsky5_t2i.json).

JoyImage uses a separate `clip_joyimage_8b` role for the exact
[official encoder](https://huggingface.co/Comfy-Org/JoyAI-Image-Edit/tree/8010e7b4c2ac3f762d33775ac5d6a011eea4166c),
with published bytes and LFS SHA recorded in the registry. Existing Boogu and
Ideogram component choices stay independent. The native encoder slices its
34-token prefix and attention mask together; the native sampler inserts the
single latent frame and the normal image VAE node returns one image. This
ordinary generation route does not enable Joy's separate reference-image edit
workflow, and the official edit recipe is not a general text-to-image default.
Missing companions use the existing atomic preparation, progress/cancel and
offline paths. Installed files require no transfer; captured paths remain stable
when roots or settings change.

OmniGen2 uses the dedicated `clip_omnigen2_3b` role and native
`comfy/text_encoders/omnigen2.py` tokenizer/encoder, including its image prompt
template. The exact [official encoder](https://huggingface.co/Comfy-Org/OmniGen-2/tree/babe908cd4ed58efc7359e9e35b426cd73071949)
is 7,509,337,224 bytes with LFS SHA-256
`ba05dd266ad6a6aa90f7b2936e4e775d801fb233540585b43933647f8bc4fbc3`;
the registry records publisher metadata, not a local payload audit. Native
`QWEN25_3B` detection accepts this encoder role and excludes the Qwen Image 7B
role. Its published `ae.safetensors` is byte-identical to the existing Flux VAE
(SHA-256 `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38`),
so this route adds no duplicate VAE. Captured companion choices and paths keep
the existing preparation leases, offline behavior and missing-file handling.
`CLIPTextEncode` metadata remains intact; native `Omnigen2.extra_conds` forwards
cross-attention and derives `num_tokens` from the attention mask, retaining the
mask when padding is present.

`image_sampling.sample_omnigen2` reuses pinned `DualCFGGuider` in regular mode,
`BasicScheduler`, and native sampler objects. As in the
[official T2I workflow](https://github.com/Comfy-Org/workflow_templates/blob/611c497cca4087967ecd85fd62bc12fa7130e885/templates/image_omnigen2_t2i.json),
the middle and negative inputs share the negative text conditioning. The user's
CFG remains native `cfg1`; native `cfg2` stays 2. Keeping the native guider also
preserves CFGNorm and custom CFG hooks; substituting ordinary CFG is not equivalent
with those hooks. Noise, seed, denoise, mask, callbacks, cancellation and native
errors retain their request ownership. Results return to the native intermediate
device/dtype. Explicit LCM noise options still use the native LCM sampler.

OmniGen2's 20-step, CFG 5, Euler/Simple compatible baseline applies to the exact
normal native configuration, independently of payload checksum, precision or source
URL. Boogu's distinct native configuration cannot inherit it. Published recipe
fields and saved overrides retain precedence, and Own settings preserves manual CFG.
This is an ordinary T2I route; native instruction editing and multiple references
require their separate conditioning workflow and are not enabled here. CPU checks
use fake weights and bounded tensors; actual model execution, rendered UI, visual
quality, performance and GPU stability remain unqualified.

### HunyuanImage 2.1 memory planning

The large HunyuanImage 2.1 BF16 base and refiner downloads were removed at the
user's request as unsuitable for their hardware. They are not queued and must not
restart automatically. Bounded header/ownership evidence remains; the smaller
HunyuanDiT 1.2 model and shared encoders were preserved. The removed files'
recorded headers contain BF16 tensors:
the base file is 34.85 GB (32.46 GiB), and the optional refiner is 30.08 GB
(28.01 GiB). Either model's weights alone exceed 24 GiB VRAM, before text
encoders, VAE and working memory. Native RAM offload can reduce residency but adds
host/device transfers; practical speed depends on hardware, resolution and steps
and has not been measured here.

The same [pinned publisher](https://huggingface.co/Comfy-Org/Hunyuan-Image-2.1/tree/78cfe74376b50a2f52995c4800f792f99ab575b2/split_files/diffusion_models)
offers FP8 base/refiner files of 17.43/15.04 GB (16.23/14.01 GiB). Existing
native float8 loading and RF's quantized diffusion GGUF adapter provide smaller
storage paths, with casting/dequantization overhead. These alternatives have not
been acquired or qualified for this workload and require separate hardware and
variant qualification before acquisition/use;
smaller weights do not guarantee the complete job fits 24 GiB. Family integration, exact variant validation and
actual speed/quality acceptance remain separate gates.

### HunyuanDiT and HunyuanDiT1

The ordinary image route admits both pinned native configurations. Both construct
native `model_base.HunyuanDiT`; ComfyUI retains their distinct beta schedules,
size/style flags, SDXL latent normalization and four-channel/eightfold geometry.
The ordinary text node supplies the same prompt to native Chinese RoBERTa and
MT5-XL tokenizers. Its scheduled conditioning retains both attention masks and
`conditioning_mt5xl`; native sampling derives `image_meta_size` from the actual
latent width/height. RF adds no HyDiT sampler or conditioning encoder.

The primary acquisition is the official native all-in-one
[`hunyuan_dit_1.2.safetensors`](https://huggingface.co/comfyanonymous/hunyuan_dit_comfyui/blob/ffdcfdc13da97c04769ead9ea1ea45c9e852d503/hunyuan_dit_1.2.safetensors),
8,240,228,270 bytes, SHA-256
`4fb84f84079cda457d171b3c6b15d1be95b5a3e5d9825703951a99ddf92d1787`.
Its BERT, MT5-XL/tokenizer and VAE remain on native checkpoint loading. Component
inventory uses the detected native config's `text_encoder_key_prefix` and
`vae_key_prefix`, including this file's `text_encoders.` and `vae.` keys, without
reading neural payloads. Embedded roles reject conflicting request overrides and
produce no split download requests. Before backend initialization the inventory
continues using its existing Torch-free metadata checks.

Compatible local split encoders use dedicated `clip_hydit_bert` and
`clip_hydit_mt5` choices plus the existing SDXL VAE slot. No split acquisition is
advertised without a qualified publisher tuple. The pinned native dual loader
omits the embedded MT5 tokenizer. `backends/comfy/hunyuan_dit.py` adapts that
constructor boundary using native safe-file loading, old-quant preprocessing and
native `CLIP`; it forwards the bounded CPU U8 tokenizer to the native tokenizer.
The caller holds captured read leases, and the native patcher reload factory
reacquires leases on the same frozen identities, supports `disable_dynamic` and
refuses changed files or cancellation. It retains no raw weight state in reload
arguments. The actual constructor validates native encoder dimensions and the
tokenizer; unknown local formats are not a compatibility promise.

Auto's 30-step, CFG 6, Euler/`sgm_uniform` compatible baseline applies to native
`HunyuanDiT` configurations, including quantizations and renamed/offline imports;
the structurally distinct `HunyuanDiT1` still needs a checkpoint recipe or Own settings.
Source verification remains available through **Models > Model > Inspect**.
Explicit Inspect verifies a safetensors file
only when its size matches a reviewed packaged checkpoint candidate, writes the
signature-bound source identity and refreshes the model controls. Passive
selection reads headers without hashing; unknown sizes retain header inspection.
A candidate with a mismatched checksum keeps its header report and explains how
to use Own settings or inspect the correct publisher file. Inspect verifies source
identity; **Check weights** remains the separate tensor-integrity operation.
The recipe comes from the publisher's native example workflow
([ComfyUI examples revision](https://github.com/comfyanonymous/ComfyUI_examples/tree/f9431bb000ce792094ff345446e22cac1ea6cef3/hunyuan_dit)).
Saved sampling overrides and manual CFG retain their normal precedence;
there is no family-wide automatic schedule or shift patch.

`test_native_hydit.py` covers the native constructor, offline tokenizer, reload,
quantization preprocessing and variant contracts. `test_hydit_route.py` covers
the actual metadata, capture/preparation, loader, text/process sampling and Auto
callers with owned tiny fixtures. The real official header/tokenizer receipt is
`cache/plan-execution-20261109/native-hydit-official-header-tokenizer.json`; the
official workflow metadata receipt is
`cache/plan-execution-20261109/native-hydit12-official-workflow-parameters.json`.
`native-hydit-unseeded-inspect-auto-cpu.json` records the real official file through
the actual explicit Inspect callback from an empty private cache into offline
Auto, without manually seeded identity metadata. It also verifies that passive
selection performs no source verification or hash.
These source/CPU checks do not qualify actual model construction, rendered UI,
generation, quality, performance or GPU stability.

`comfy/model_detection.py` and `comfy/supported_models.py` remain the classifier;
`features/models/model_architecture.py` adapts their result without loading
weights. `image_components.py` normalizes labels and resolves installed/custom
or explicitly acquired components. `component_profiles.py` freezes checkpoint
choices before enqueue. `features/generation/image_pipeline.py` still controls
which native runtime classes and conditioning workflows RF can execute.

### Local component compatibility at selection and submission

The checkpoint component inventory now reports local metadata compatibility
separately from installation status. The editor filters alternatives for the
selected role and the other active encoders. An incompatible existing choice
stays visible for correction; submission and image-asset capture reject it before
weight loading or preparation. Capture inspects the current file signature again,
so replacing an encoder/VAE at the same path after queue capture cannot reuse an
old compatibility result. Embedded components retain priority over profile-wide
legacy settings and reject explicit replacements. Saved checkpoint choices,
Library replay and intentional unverified advanced choices retain their behavior.
Saving an alternative and clicking **Refresh components** keep the editor on
the selected role when it still exists, including VAE choices; a removed role
falls back to the first role available for the current checkpoint.

Safetensors text alternatives reuse initialized native `detect_te_model` with
bounded header shapes, plus CLIP-L/H/G projection dimensions and HyDiT's MT5
tokenizer presence check. VAE roles use planar RGB geometry (four or sixteen
latents, eightfold scale), the distinct SD4x fourfold codec, or the full Wan2.1
RGB/sixteen-latent resampling contract. All current ordinary image VAE roles have
an explicit layout contract. Flux2 uses native `sd.py`'s planar 32-channel core,
128-channel BN statistics and 2x2 packing, producing 128 effective latents at
16x scale. Mage-Flow uses `student.dconv_encoder` and `pipeline` CoD/DConv geometry,
also 128 latents at 16x but a distinct codec. Same latent count does not qualify
one of these as the other. Diffusers planar geometry follows the native
conversion marker and sampler keys, without loading or reproducing its converter.

GGUF logical shapes can reject known architecture conflicts but do not qualify
an alternative: native key conversion, embedded tokenizer data and optional
vision companions are not reproduced by this picker. Those candidates remain
**Unverified** and require the existing advanced choice. HyDiT's native split
loader does not accept GGUF and is rejected. The ordinary mixed GGUF text-loader
route also rejects a `scaled_fp8` companion, matching pinned
`DualCLIPLoaderGGUF.load_data`. This contract does not certify other GGUF
quantization/device combinations, text-only Qwen3-VL Dynamic VRAM routes,
malformed/unknown headers, tokenizer training semantics or payload integrity.

Focused validation: `venv/Scripts/python.exe tools/check_cpu.py
test_component_compatibility_capture.py test_component_candidates.py
test_component_profiles.py test_image_components.py test_component_overrides.py
test_image_preparation.py` (66 CPU checks). The new suite executes the pinned
native text classifier with header-only fixtures. The installed Flux2 and Mage
headers were inspected without tensor payloads. No new GPU, generation, quality
or performance qualification is claimed.

The local model picker lists installed/custom checkpoints; it is not a native
architecture compatibility certificate. The model registry annotates exact
download tuples and is not an exhaustive native model allowlist. Missing
catalogue entries are not evidence of upstream removal. No deprecated model is
removed without explicit pinned-source and caller evidence.

Other native image families and special refiner, cascade, instruct, UnCLIP and
masked-conditioning workflows require their actual encoder, latent, VAE and
caller contracts before RF enables them. Audio, video and 3D output are outside
the product scope. Detection remains a retained still-image utility. Listing a
model in native source does not add an RF route. Optional custom nodes beyond
the pin are not automatically admitted; no bulk acquisition is implied.
See [native special image workflows](native-special-image-workflows.md) for
operation-specific conditioning and [native model formats](native-model-formats.md)
for safe-container inspection and pinned-loader limits.

Focused CPU checks cover normalization against current native labels, actual
component inventories and unchanged bundled/custom/preparation callers. They do
not certify new model execution, image quality or GPU stability. Preserve the
source/model-bound N01/N02 acceptance receipts separately from this source audit.
The additional route checks run with
`python tools/check_cpu.py test_native_model_coverage.py test_model_recommendations.py test_model_defaults_ui.py test_api_image_edits.py`.
Kandinsky/Joy caller checks add `test_native_ordinary_pair.py` and
`test_model_registry.py`; these tests use small local fixtures, never real weights.
OmniGen2 checks add `test_omnigen2_native.py` for actual application process/encode
dispatch, native attention/latent/guidance contracts, captured companions and
native variant automatic defaults.

## Compatible native variant baselines

For the three configurations above, Auto uses the pinned native detector and
class matcher on safetensors headers or GGUF logical shapes. Inspection caches
are bound to the selected path and current file stat signature; replacing a file
invalidates them. No model weights, full-file checksum or network request is needed
to qualify a baseline. Checksums continue to identify and verify acquired sources.
This bounded qualification path covers safetensors (including native aliases) and
GGUF logical headers. Native Torch containers still need an explicit checkpoint
recipe or Own settings; this change does not add a safe header inspector for them.
The settings explanation labels fallback values **Compatible native variant baseline**,
separately from checkpoint metadata and saved user overrides. A missing URL or
training label never prevents an otherwise compatible baseline.

Explicit behavioral variant declarations such as `modelspec.variant=distilled`
can require a complete checkpoint recipe; ordinary model capabilities and filenames
are not variant evidence. Generic `pretrain` alone is not a demonstrated sampling
contradiction. Own settings preserves CFG zero, denoise and manual geometry. Existing
recommended sizes and custom resolution selections are unchanged; wider source-size
normalization remains separate work. No dtype-based resolution is inferred.

Focused checks: `python tools/check_cpu.py test_native_variant_defaults.py
test_model_recommendations.py test_hydit_route.py test_omnigen2_native.py
test_model_defaults_ui.py test_native_ordinary_pair.py`. These prove header/config,
caller precedence and cache behavior with small fixtures, including actual Q8 GGUF
metadata. They do not certify the quality or device performance of every quantization.

## Current variant and quantization defaults

Sampling defaults qualify behavior separately from storage precision. The pinned
detector classifies FLUX 3 flow-map weights from both safetensors and GGUF logical
headers; RF additionally checks the image, modulation and reference-time tensor
geometry before applying the existing Klein recipe. Renamed and unindexed absolute
imports need no display-cache entry or payload hash. Base, malformed and unknown
FLUX 3 variants retain explicit saved settings or Own settings. These decisions
and the three native baselines above use signature-cached header inspection, not
dtype, file size or local filenames.

| Existing family                       | Automatic disposition                                                                                                                                                           |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HunyuanDiT, Kandinsky5Image, OmniGen2 | Existing exact native-configuration baseline, with behavioral declaration guards                                                                                                |
| FLUX 3                                | Geometry-qualified native flow-map baseline; base/unknown requires explicit settings                                                                                            |
| HiDream I1                            | Current explicit full/dev/fast declaration selects its distinct sampler and sampling shift; ambiguous/conflicting variants require a complete checkpoint recipe or Own settings |
| HiDream O1                            | Current full/base/dev declaration selects the recipe and the same native noise, clipping and seam defaults; ambiguous/conflicting variants require explicit settings            |
| Qwen Image, Mage Flow                 | Current base/edit/turbo source declarations select the corresponding recipe; missing/conflicting variant evidence requires complete checkpoint settings or Own settings         |
| Other existing ordinary families      | Existing family/native preset and publisher/user precedence remain unchanged; local names do not select a different turbo recipe                                                |

Behavioral evidence consists of bounded explicit header declarations, matching
signature-bound source metadata, or bounded variant tokens in a current exact
publisher version/model title. Publisher titles are source evidence and never
read from the selected local filename. Conflicting structured declarations do
not fall back to source-title guesses. GGUF preserves the same four behavioral
declaration keys as safetensors. Source caches carrying a file signature are
invalidated for every family when the selected file changes. O1's native patch
helper reuses the active recommendation store, preserving private profiles and
the same variant decision as automatic sampling.

Offline or unavailable recommendation refreshes can reuse public version-example
consensus already present in the model metadata cache. This requires a current
local file signature and its separately computed checksum matching a file in that
exact version, including the selected precision variant. Picker/render paths do
not hash weights or fetch metadata. Explicit creator fields take precedence over
inferred examples, cached network recipes retain their precedence, and saved user
overrides remain authoritative. The UI labels cached example consensus as inferred;
missing fields continue to use the family baseline. A bare checkpoint with no
embedded/source lineage metadata cannot establish a fine-tuning family from its
filename or tensor architecture alone.

When background discovery changes the selected checkpoint's family or public
identity, the catalogue timer also signals the existing metadata-refresh controls.
This refreshes automatic sampling, component and size choices without reselecting
the checkpoint. Unchanged ticks and artwork-only updates do not signal these
controls; prompt, adapter selections, manual sampling and size overrides remain
owned by their existing controls.

Successful exact-version recommendation refreshes also cache explicit
`generationConfig.width/height`: each must be an even integer from 256 through 4096. No size is inferred from gallery examples, dtype or model-card prose.
`ModelRecommendations.cached_resolution` is read-only and returns size plus
provenance; a successful refresh with no published size clears the old size,
while a failed or offline refresh retains it. A stale signed source cannot
contribute resolution or sampling recommendations. The resolution consumer keeps
explicit user dimensions ahead of suggestions and reports the chosen source.

Focused CPU checks add `test_quantized_model_contracts.py`, including actual tiny
Q8 GGUF fixtures, renamed imports, conflicting declarations, malformed flow-map
geometry, exact publisher size caching, offline retention and stale signatures.
Existing HiDream sampler, guidance and API tests use explicit variant fixtures
and preserve publisher/user overrides and request-owned sampling shifts. This
qualifies metadata and caller contracts only; it does not certify real-model
execution, visual quality, memory fit, device eligibility or GPU performance.

## Remaining ordinary image routes

The remaining ordinary image adapters use the same captured component choices,
native loaders and serialized worker. SenseNova's split loader constructs the
native sentinel while holding the unchanged checkpoint's read lease; delegate
reloads recheck that identity and cancellation. The official HF eight-shard
SenseNova index now has a [named local bundle reader](native-sensenova-bundles.md).
It freezes and leases all nine source files, retains native per-file tensor storage,
and feeds the native state-dictionary loader without creating merged weights.
The catalogue still has no exact acquisition tuple for this bundle; full-model,
default-recipe and hardware qualification remain separate.

Qwen Image2.1 always uses `TextEncodeQwenImage21`, including scheduled prompts,
with native `keep_vision` and `prevent_empty_text` flags. Its zero latent retains
the requested nonsquare 16-pixel grid for text-to-image. Reference edits use
the native first-reference target grid and the shared source framing policy.
Text conditions are request-local and explicit CLIP-layer choices apply to a
clone. See the [reference edit contract](native-special-image-workflows.md#qwen-image-21-reference-editing).
Its dedicated RGBA codec and Hunyuan's
64-channel codec have different spatial layouts; equal latent counts do not
make them interchangeable. Existing old-Qwen/Wan/Ming codecs are rejected.

The exact Qwen Image2.1 base template at workflow revision
`611c497cca4087967ecd85fd62bc12fa7130e885` (blob
`d6c27aa5889f1bf54a86ba2ad425a8665800db75`) supplies25steps,CFG1,
Euler/Simple. Current explicit base declarations or the verified exact official
INT8 source tuple qualify it. Ambiguous variants require explicit settings.
The checkpoint annotation uses the native image diffusion checkpoint role;
its current source verification therefore carries the base recipe across INT8
storage without requiring an embedded variant tag. Changed source signatures
invalidate that proof.
The Turbo template uses a manual sigma list; a generic8-step schedule is not
claimed equivalent. Hunyuan and SenseNova require complete saved checkpoint
settings or Own settings until a matching recipe is qualified. Native Hunyuan
conditioning retains its native guidance default separately from sampler CFG.

Five immutable acquisition rows add Qwen2.1 diffusion/encoder/VAE and Hunyuan
ByT5/VAE companions. The removed large Hunyuan BF16 diffusion/refiner payloads
remain excluded. Existing scaled-FP8 Qwen2.5 is the native-compatible Hunyuan
companion, without claiming byte equality to its FP16 alternative. These are
source/CPU contracts only: no model generation, device fit, quality or speed
qualification is implied. Focused checks are declared in
`tools/features/native-ordinary-remaining-routes.json` and include the preserved
native SenseNova and Qwen paired-node regression suites.

## UnCLIP image guidance

UnCLIP L/H now uses the ordinary native loader with bundled vision output enabled.
Select the checkpoint, choose Image tools → Edit and provide a source image. RF
uses native `CLIPVisionEncode` and `unCLIPConditioning` on positive conditioning,
including scheduled intervals; negative text remains unchanged. Native strength1,
noise augmentation0 and center crop apply. The native config requires768-wide L
or1024-wide H embeddings and mismatches are rejected before sampling. The bundled
vision tower survives an external text/VAE fallback and follows the checkpoint's
cache lifetime. No arbitrary standalone vision tower is substituted. A split
model lacking bundled vision can run native text-only zero-ADM generation, but
image guidance requires the bundled tower and reports that requirement. Complete
checkpoint sampling settings or Own settings are required; no sampler recipe is
guessed from L/H names. Focused native/caller tests are declared by
`tools/features/native-unclip-guidance.json`; real-model/UI/device qualification
remains separate.

## Native special image workflows

Create's contextual workflow controls now call the pinned native image nodes.
Cascade uses bundled C and B checkpoints: C creates the compression42 prior,
B receives that prior in positive conditioning, samples its separate fourfold
latent and decodes through B's bundled StageA codec. The exact official
[workflow example](https://github.com/comfyanonymous/ComfyUI_examples/blob/f9431bb000ce792094ff345446e22cac1ea6cef3/stable_cascade/stable_cascade__text_to_image.png)
(Git blob `67c3698790b5846455582267b6be6b8018446d31`) supplies C20steps/CFG4
and B10steps/CFG1.1, both Euler ancestral/Simple. Current verified official
source identities qualify Auto; explicit B settings win. Unknown sources need
complete settings. Each stage uses the request seed; B has its own prompt
schedule and native CLIP layer selection. Source requests use native
`StableCascade_StageC_VAEEncode` and retain its paired Stage B grid through
decoding. Compression and source denoise are explicit contextual controls;
source-size requests retain native geometry while exact targets uniformly fit
and pad the full source, then restore alignment and the captured final target.
Cropping is refused. Explicit native twofold control and ordered vision mixing
have a separate [Stage C guidance route](native-cascade-stage-guidance.md). See the
[Cascade source contract](native-cascade-source.md).

Hunyuan Image2.1 refiner requires Edit with a source image and its dedicated
32-channel, single-frame16x codec. The native header contract has3584text
dimensions and no ByT5 input, so the split loader uses native HUNYUAN_IMAGE
Qwen2.5-only encoding; it rejects a different text contract. Native
`HunyuanRefinerLatent` supplies source concat and noise augmentation (Auto0.10,
explicit0 preserved), and the native latent format folds/unfolds codec channels.
The large removed diffusion/refiner payloads are not restored or downloaded.

PiD requires Edit, a known-compatible local input VAE and an explicit native
latent-format choice. Native detector channel counts reject a mismatched input
codec before payload loading. Runtime also checks channels and the native
upscale factor against the original oriented source and requested output size.
`PiDConditioning` supplies both branches' LQ latent and degradation (Auto0);
output uses the native pixel-space conversion. The input codec is held under
its captured read lease during encoding. Unknown/GGUF input-codec contracts
remain unqualified. PiD and the Hunyuan refiner require complete checkpoint
sampling settings or Own settings until their particular recipe is qualified.

All secondary sources are captured before the existing serialized compute
worker. StageB's signature-bound cache is cleared on primary model changes;
leases, native cancellation and model reload factories retain existing owners.
Missing selections remain visible in the UI and fail submission. Replay and API
transport use the same typed fields; metadata records stage/codec parameters.
Focused source/CPU and contextual UI checks are declared by
`tools/features/native-special-image-workflows.json`. These establish caller
contracts, not actual neural output, memory fit, quality or device performance.

## Native BiRefNet background removal

Image tools → Rembg accepts an optional installed BiRefNet safetensors model
from the configured `background_removal` roots. The default remains the existing
Hardware ONNX selection. Explicit native selection uses the image runtime's
device and native `comfy.bg_removal_model.load` / `RemoveBackground`; it never
changes ONNX settings or adds model downloads. Current native header markers
gate discovery and capture; unknown containers remain unqualified.

The source signature is captured before compute and leased through native load
and mask inference. Model replacement invalidates the request/cache. Both UI
and API run this native choice under the shared image service with native
interrupt handling, not the independent ONNX service. Switching to native closes
the cached ONNX session; switching back unloads the native patcher through native
memory management. Native masks keep original oriented dimensions and straight
RGB, multiply existing alpha, and skip transparent inputs unless reprocessing
is explicit. Saved missing files stay visible and fail submission. Scalar choice
and source provenance round-trip in metadata. Focused source/CPU tests are
declared in `tools/features/native-background-removal.json`; real model output,
device fit and speed remain unqualified.

## Optional pinned-source inventory audits

The finite ledger produced by `tools/audit_native_model_coverage.py` keeps separate
denominators for registrations, artifact roles and recipes. The retained
2026-11-11 baseline at ComfyUI `fca60fe` and GGUF `da5f435` contains:

| Inventory                                | Rows and interpretation                                                                                                    |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Native node declarations                 | 707 across 147 explicitly loaded modules: 65 core V1, 82 bundled V1, 545 bundled V3 and 15 GGUF                            |
| Model configuration classes / CLIP types | 109 configurations and 39 enum entries; distinct from the 48 RF runtime configurations and 48 component configurations     |
| Folder roles                             | 27 native buckets plus 5 GGUF aliases; a folder suffix set only identifies candidates                                      |
| Pinned native blueprints                 | 116 files, 131 nested graphs, 1,946 leaf instances and 209 leaf node IDs; 131 graph-call wrappers are counted separately   |
| Blueprint model annotations              | 346 occurrences, each retaining recipe, graph, node instance, role, filename and source URL                                |
| Installed official workflow recipes      | 664 graph files, separately versioned from the native pin; 16 index/schema JSON assets are excluded from this recipe count |

The published 0.45 model inventory records 110 configurations, 39 logical
`CLIPType` values and 41 detected `TEModel` values. Its separately reviewed
template target has 672 recipes / 688 JSON assets; the installed baseline above
remains source-bound. See the adoption contract and finite delta rather than
transferring old node/recipe or hardware counts to the new pin automatically.

Historical `finite_inventory` output contains each row's former packet assignment and disposition
and separate source, UI/API, real-runtime and visual-quality evidence. A planned
operation review or an existing import is unfinished acceptance, never a support
claim. Packet assignments group format readers (F), exports (E), references (R),
patches (P), generation (G), utilities (U) and exact-artifact acceptance (A). Scoped
audio/video/3D/training and hosted-partner recipes remain explicit exclusions;
unregistered frontend/custom node IDs remain a review queue. No missing applicable
implementation is relabeled unsupported. Native Qwen 2.1 Union retains its own
source row. The [named patch/control routes](native-model-patches.md) connect
ordinary Qwen DiffSynth, Z-Image Fun, Anima LLLite, Flux USO, ordinary ControlNet
Union modes and Qwen AliMama68 inpaint/outpaint. CPU/native-body, Gradio and API
checks establish those source contracts; exact-artifact generation, rendered UI,
visual quality and device qualification remain separate acceptance work.

Blueprint bytes are checked against their pinned Git blobs. Official template data
is read without importing its package, with installed distribution versions,
per-recipe SHA-256 and core manifest comparison retained. The observed installation
uses meta package 0.11.103, core 0.3.394 and JSON 0.1.129. All 664 recipe hashes match
the installed core manifest; 13 localized/index assets differ and are recorded as
manifest mismatches. This does not promote the templates to executable RF routes.
Use `--templates-root` to select an explicit installed JSON bundle. If the template
data is unavailable, its separate inventory is blocked rather than silently zero
coverage. `--baseline` also compares added, removed and changed recipe rows.

The ledger's `W02_contract` and `native_roles` provide reader-specific component
requirements, provenance, unknown-file visibility and the versioned cache-envelope
boundary. Recipe annotations do not authorize downloads or establish artifact
identity. Preserve bundled components, role-compatible user overrides and exact
qualification receipts independently of family/variant recommendations.

LotusD has a photo-to-depth consumer in the existing Depth Image tools controls.
It requires an explicit local LotusD safetensors checkpoint and compatible SD VAE;
bounded native metadata rejects unrelated checkpoints/codecs before loading.
The [official workflow](https://github.com/Comfy-Org/workflow_templates/blob/611c497cca4087967ecd85fd62bc12fa7130e885/templates/image_lotus_depth_v1_1.json)
(blob `bdbeef4c9b841baee55aafec8266515f022375f8`) supplies LotusConditioning,
DisableNoise, BasicGuider, normal one-step sigmas with first sigma 999, native Euler
sampling, VAE decoding and ImageInvert. No text encoder is loaded. Input dimensions
must be divisible by eight; output keeps their dimensions and white-near convention.
Captured file signatures/read leases cover both components through inference;
replacement invalidates the native cache. Existing DA3 and supplied-map behavior
remain available. `test_lotus_depth.py` checks the native contract and caller using
CPU fixtures; real model output, quality and device memory remain unqualified.

Native image-prompt adapters now have an explicit Image tools consumer:
StyleAdapter for ordinary SD1 (1024-wide ViT-L input, 768-wide conditioning),
Redux for ordinary Flux/Schnell (1152-wide SigLIP384 input, 4096-wide conditioning),
and experimental PhotoMaker for dual-CLIP SDXL. Captured local files and a
separate reference upload feed the existing native nodes; negative conditioning
and cached text remain unchanged. PhotoMaker uses one native 77-token prompt
segment and the explicit `photomaker` trigger. Other vision layouts, custom-node
IPAdapter and neural quality/device qualification are not implied. See the
[reference adapter contract](native-reference-adapters.md); focused CPU cases
are `test_reference_adapters.py` and `test_reference_adapters_ui.py`.

Hunyuan discovery preserves the native image-refiner discriminator: a1×1×1
patch with no vector conditioning defers to the pinned classifier, including
32-channel fixture and128-channel official refiner headers. Positive video
discriminators retain the existing video exclusion. This prevents the early
metadata filter from hiding a native image refiner before its actual route is
considered; `test_native_refiner_discovery.py` executes the pinned detector and
class matcher without loading models or running video.

Native SD20 shares `BaseModel` with SD15 but has a different text encoder.
Runtime selection checks the exact native config class and uses the existing
SD2 CLIP-H role; the SD2 inventory alias follows that same role. CLIP-L is
rejected for this slot, while bundled text retains precedence. A display name,
context width or ordinary `BaseModel` identity alone does not select the route.
Focused checks are `test_sd2_components.py` and `test_component_candidates.py`.

Run `python tools/audit_native_model_coverage.py --output cache/native-coverage.json`
after changing either pin. The maintainer tool checks both checkout revisions
against bootstrap and verifies the audited files against their pinned Git blobs.
It derives model configurations, inherited factories, CLIP types, loader
branches, component discovery buckets, RF component routes and exact acquisition
tuples. It also lists the declared node IDs from Comfy's core, its explicit
built-in module list (V1 mappings and V3 extensions), and the pinned GGUF package.
The node rows cite source and distinguish scoped RF class references, graph
primitives, still-image scope exclusions and items needing caller review. Hosted
API nodes, other custom nodes and workflow templates are outside this denominator.
The audit imports no application or tensor runtime and runs no model.

Registry `native_domain` follows native module/category metadata; the finite
inventory separately applies the reviewed image operation boundary. The shared
`HunyuanRefinerLatent` and `MiniMaxH3ImageConditioning` nodes keep their still-image
consumers in scope, alongside the bounded SeedVR2/LTXV shared primitives. Temporal
frame operations, video-only latent upscaling and MoGe mesh conversion remain
excluded even when their native category says `model` or `image`. These exceptions
do not admit generic video execution. Pure image transform/color/batch and text
helpers from the dataset module retain their declared category; training dataset
and temporal-video operations stay excluded.

Recipe rows retain surfaced interface output types, excluded node IDs and shared
image node IDs. Nested preprocessing outputs do not change a recipe's exported
media type. Explicit root exports take precedence; direct graph-wrapper outputs
are used only when root exports are absent. A mixed MoGe image/mesh blueprint can describe its image outputs while its
mesh branch stays explicitly excluded; a mesh-only recipe remains 3D. Gemini
captioning/prompt blueprints remain hosted-partner recipes even without an `api_`
filename. Declarations and graph helpers are source inventory, not a count of
standalone user tools or proof that every branch has an RF consumer.

Use `operation_families` for the source acceptance queue. Qwen Image 2.1 text
conditioning and reference editing have separate source rows. SAM3 exposes
point/text/box prompts and combined or ordered individual masks through its
[standalone UI/API consumer](native-sam-segmentation.md). Cascade text to image and native source-image paired grids,
standalone SDXL refinement, [SDXL base/refiner handoff](native-sdxl-refiner.md)
and generic ControlNet each have their own source routes. Cascade
super-resolution ControlNet/vision mixing has its own implemented source row;
Union and dedicated patch modes retain separate operation rows. Still-image
[SeedVR2](native-seedvr2-restoration.md) has a standalone source, matching-codec,
UI/API/Library consumer. [Qwen Layered](native-qwen-layered.md) has a named native source route
with a distinct RGBA codec, bounded layer groups and ordered output transport;
the interrupted L1 comparison does not establish app/native equivalence,
device/visual acceptance or Auto. See the finite delta for exact case dispositions.
The pinned GGUF VAE node accepts only the MiniMax H3
Q8_CR video codec. Still-image component selection and request capture reject
`.gguf` VAE files in any casing, including saved choices whose file is missing.
Use a native state-dictionary VAE that matches the image checkpoint's latent
layout. The GGUF LoRA row separately checks whether the app's LoRA dispatch calls
its adapter and that adapter calls the pinned importer. Source status does not
certify UI/API transport, artifact compatibility, rendered output or device
behavior.

Use `--baseline <previous-report.json>` to list added/removed config classes,
changed contracts, possible renames, node IDs and operation disposition changes;
`--fail-on-drift` makes these changes a review gate. A legacy schema 1 report can
still be compared for model drift. The result flags the missing node baseline
until a new snapshot is recorded. Missing routes and new native tasks require
caller review. The report never enables models, deletes resources or labels an
absence as deprecation. Output paths must be new to preserve prior reports.
