# Native image model file formats

Still utility safe tensors have operation-specific readers. Standalone Lotus
uses a detected LotusD model plus SD VAE; SDPose additionally requires a heatmap
head and bundled SD VAE. MoGe and MediaPipe use their own geometry/face bundle
loaders. Marigold V2 albedo uses separate Qwen Image Edit, albedo LoRA, albedo VAE
and saved conditioning roles. A matching suffix or latent width is insufficient
to identify albedo versus depth/normals companions; the three task components
require their published albedo source identities. See
[still geometry utilities](native-still-geometry-utilities.md) for exact roles,
native output types, safe capture and qualification limits.

RF uses the pinned ComfyUI file loaders for native image checkpoints and standalone
diffusion weights. The current pin is
[`989d93d0d968e704c589ce0146bb18c77f6b50cc`](https://github.com/comfyanonymous/ComfyUI/tree/989d93d0d968e704c589ce0146bb18c77f6b50cc),
inspected from the local pinned source on 2026-11-12. See
[native model coverage](native-model-coverage.md) for architecture and operation
coverage; a readable container does not establish either.

[Zeta pixel-space sources](native-zeta-pixel-source.md) use the native MLP decoder.
Official L2P's convolutional decoder is a distinct unsupported architecture and is
refused before it can be loaded as ordinary ZImage.

`.safetensors` and `.sft` name the same safe container in the
[native file reader](https://github.com/comfyanonymous/ComfyUI/blob/989d93d0d968e704c589ce0146bb18c77f6b50cc/comfy/utils.py#L192).
RF accepts both, case-insensitively, for local discovery, bounded header metadata,
component selection and preparation, checkpoint inspection, LoRA names and native
encoder dispatch. The shared suffix predicate lives in `core/model_formats.py`.
Header caches remain tied to file identity; metadata, tensor dtypes and original
bytes are preserved. Explicit FP32 conversion continues to create a separate
`.fp16.safetensors` file through the existing conversion action.

`.ckpt`, `.pth`, `.bin`, `.pt`, `.pt2` and `.pkl` use the existing native
`torch.load(weights_only=True)` path. RF discovers these candidates but does not
read their weights merely to populate a picker or infer external components.
Tiny CPU fixtures through the pinned reader prove `torch.save` state dictionaries
under all six suffixes, with CPU mapping and both native mmap modes. Generic
pickle streams and unallowlisted Python objects remain refused. A `.pt2` name
does not make an arbitrary exported program a checkpoint, and a `.pkl` name does
not authorize arbitrary Python deserialization. Unknown component compatibility
continues to require an explicit advanced choice and native runtime validation.

Native file-based loading retains ComfyUI quantization metadata, precision and
device decisions, and patcher reload factories. GGUF uses the separately pinned
`molbal_comfyui_gguf` adapter and its architecture/quantization checks. Neither
suffix admission nor a native quantization registry entry certifies all devices,
quantized artifacts or operations. The optional stable-diffusion.cpp engine has
its own narrower qualification contract in [its guide](sdcpp.md).

| Native artifact path                                                                   | RF path and limits                                                                                                                                                                                                    |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Safe single-file checkpoint or diffusion state dictionary (`.safetensors`, `.sft`)     | Header inspection and native file loaders; architecture and operation checks remain separate.                                                                                                                         |
| Torch state-dictionary container (`.ckpt`, `.pth`, `.bin`, `.pt`, `.pt2`, `.pkl`)      | Native weights-only file loader; browsing leaves architecture/components unknown when no safe header exists.                                                                                                          |
| Image GGUF                                                                             | Separate pinned adapter and logical-header inspection; unsupported architecture/quantization metadata is refused.                                                                                                     |
| Diffusers single-file tensor layout                                                    | Native conversion/detection and file loaders, without copied conversion maps.                                                                                                                                         |
| Diffusers directory bundle (`model_index.json`, local `unet`/`vae`/text encoder files) | Models Refresh discovers local bundles beneath configured Diffusers roots. Frozen local companions feed native per-file loaders; the bundle's components stay fixed.                                                  |
| Explicit local YAML recipe                                                             | Data-only prediction/CLIP-layer overrides on native-detected ordinary SD1/SD2/SDXL/refiner checkpoints. Request-owned Clip Skip is default; opt-in recipe layers, file identities and replay are captured separately. |

See the [YAML recipe contract](native-yaml-recipe-contract.md) for bounded parsing,
native type guards and explicit API/UI ownership. YAML never selects classes or
downloads companions. The native directory loader chooses specific single
files; it does not assemble shard indexes. This bundle route follows that exact
layout contract and does not claim all sharded layouts or arbitrary Python
Diffusers pipeline compatibility.

The official [SenseNova U1.5 eight-shard bundle](native-sensenova-bundles.md) has
a separate named local reader for `model.safetensors.index.json`. It does not
expand generic index/JSON admission or the native Diffusers directory contract.

## Image API and component roles

Named image patch files use the [native model-patch contract](native-model-patches.md).
Safe tensor headers select Anima LLLite, ordinary Qwen DiffSynth, Z-Image Fun or
Flux USO by native keys and shapes. Native Torch containers remain explicit
candidates requiring type validation during Run; a suffix alone proves no role.
USO also validates its separate SigLIP vision encoder. Ordinary Union mode and
AliMama68 use the ControlNet reader and `controlnet` folder, independently of the
`model_patches` recipes and Qwen Image 2.1 Union.

`GET /v1/models` lists image candidates from the same local checkpoint and
diffusion-model inventory as the image picker. Single-file IDs keep the
`image/<relative filename>` form, including subfolders and the file's original
case. The API accepts native Torch suffixes and `.sft` in addition to the
existing checkpoint formats. Selection still validates the architecture and
required components at preparation and loading time.

Pixel-upscaler selectors offer safetensors and native Torch state dictionaries.
The pinned pixel upscaler passes those weights to Spandrel. The shared LoRA
picker and `/v1/capabilities` offer native Torch and safetensors candidates plus
GGUF files whose bounded metadata matches the native LoRA adapter contract below.
[Pinned ComfyUI pixel upscaler](https://github.com/comfyanonymous/ComfyUI/blob/989d93d0d968e704c589ce0146bb18c77f6b50cc/comfy_extras/nodes_upscale_model.py#L36)

## Explicit GGUF image LoRA

For the native image engine, place the adapter under a configured LoRA folder
and use **Create → LoRAs & embeddings → Refresh files** to select its card, pass
its name from `/v1/capabilities` in the image API's `loras` field, or use a prompt
tag such as `<lora:styles/detail.GGUF:0.6>`. Include its
relative subfolder and exact filename. The first matching configured folder
supplies the file, and a GGUF name never falls back to a Torch adapter with
the same stem. A tag can use a finite strength from −100 to 100, including
negative strength. A zero strength skips the adapter.

RF uses the pinned GGUF adapter loader and its MODEL/CLIP alias mapping. The
container needs `general.type=adapter`, `adapter.type=lora` and paired 2-D
`.lora_a`/`.lora_b` factors. The pin supports F32, F16, BF16 and Q8_0 factors.
Each pair needs one distinct native target with matching logical dimensions.
Native target slices are accepted when the slice has matching dimensions.
Target transforms and ambiguous or partial matches fail before the patch set
is installed. The standard ComfyUI patcher uses the same signed strength for
MODEL and CLIP. Mixed GGUF and Torch adapters use the existing patch order.
[Pinned GGUF LoRA importer](https://github.com/molbal/ComfyUI-GGUF/blob/da5f435cd9a9e54ad27eed417ed72b1f2543ef41/nodes.py#L705)
[Pinned GGUF factor loader](https://github.com/molbal/ComfyUI-GGUF/blob/da5f435cd9a9e54ad27eed417ed72b1f2543ef41/lora.py#L230)

The four types are the native LoRA reader's
[`_SUPPORTED_FACTOR_TYPES`](https://github.com/molbal/ComfyUI-GGUF/blob/da5f435cd9a9e54ad27eed417ed72b1f2543ef41/lora.py#L11).
Its [`_tensor_to_float` guard](https://github.com/molbal/ComfyUI-GGUF/blob/da5f435cd9a9e54ad27eed417ed72b1f2543ef41/lora.py#L168)
rejects every other type before factor shape or bytes are read, including Q4,
Q5, K and IQ encodings. General GGUF checkpoint dequantization does not widen
this adapter reader. The finite enum rejection check and real tiny GGUF fixtures
for all four admitted types run in `test_gguf_lora_factor_contract.py` and
`test_gguf_lora_admission.py`; no model weight download is needed.

`features/models/gguf_adapters.py` reads detached storage descriptors through
`read_gguf_header(include_storage=True)`. It refuses other GGUF roles, unpaired
or non-2-D factors, unsupported factor types and conflicting shape overrides.
Discovery skips invalid/partial files, preserves configured-root priority, and
never maps weights. The default logical header output and its digests remain
unchanged. API submission and prompt capture repeat typed admission against the
current source signature. An unknown architecture label does not exclude a
custom adapter; its compatibility remains unknown until native target matching.

The worker captures the resolved file signature after prompt tags are resolved
and checks it before and after the read. The native patch cache includes the
file signature and strength. The guarded `test_gguf_lora_dispatch.py` fixture
uses real pinned parsing, alias mapping and CPU MODEL/CLIP patch calculation
with small local tensors. `test_gguf_lora_admission.py` additionally compares all
four factor types with the actual pinned reader; `test_api_model_discovery.py`
checks the shared inventory, real Gradio card construction, public capabilities
and stale-file API admission. Ordinary safetensors LoRAs applied to a GGUF
checkpoint remain a separate existing path. Model, rendered browser and hardware
qualification is separate.

## Local Diffusers bundles

Place an existing local bundle beneath the configured `path_diffusers` roots
(default `models/diffusers`), then use **Models → Refresh**. The existing image
picker and cards show its directory title; selection identity is the absolute
`model_index.json` path. The model index is bounded JSON data: its pipeline/library
names never become imports, downloads or executable code.

`features/models/local_diffusers.py` captures the native file preference for
`unet`, `vae`, `text_encoder` and optional `text_encoder_2`. Missing declared
companions produce a clear error. The existing component panel lists those
local files and disables replacement/download actions for the fixed bundle.
Selecting individual checkpoint files continues to support normal component choices.

`GET /v1/models` includes bundles whose native file layout passes capture.
The public ID has the form `image/diffusers/<local ID>/<bundle name>/model_index.json`.
A bundle installed directly at a configured root uses just `model_index.json`
after the local ID. The ID is specific to the local bundle location, stays bound
across configured-root reordering and changes when the bundle moves. Use the ID
returned by the API. `image/default` also resolves the bundle selected by the
active image defaults or profile. The API passes the private picker selection
to normal request capture. The picker keeps incomplete bundles visible so their
component errors can be inspected.

Submission freezes source and resolved paths plus every selected file signature.
Existing image preparation carries local-only artifact records; native loading
holds their read leases, checks cancellation and verifies file identity. A changed
companion or symlink target refuses the stale request. A new higher-priority file
changes the next bundle identity without retargeting a queued request. The native
model cache includes that identity and all component signatures.

The current native diffusion, CLIP and VAE file APIs own tensor loading,
architecture/encoder validation, metadata, reload factories and device policy.
RF checks that the selected encoder count matches the native recipe; it does not
substitute catalogue components or silently drop additional encoders. Source and
CPU fixtures establish routing only: no arbitrary bundle, real checkpoint or GPU
quality qualification is inferred. This route uses the native image engine.

Native quantization dispatch is retained for FP8 E4M3/E5M2, NVFP4, conditional
MXFP8, tensorwise INT8, ConvRot W4A4, asymmetric W4A8 and W6A8 registrations in
the pinned `comfy/quant_ops.py`. These are loader/layout registrations, not a
promise of accelerated kernels on every device. The existing file-based caller
preserves metadata and native `convert_old_quants`; it does not substitute a
storage dtype or bypass native backend eligibility. GGUF quantization remains
the separate adapter contract above.

## Text-encoder reader dispatch

Encoder discovery and loading preserve the selected file's original path and case.
`backends/comfy/text_encoder_loader.py` detects `.gguf` in any casing, including
`.GGUF` and `.gGuF`. A wholly native state-dictionary encoder set uses
`comfy.sd.load_clip`; this also covers the supported Torch containers without
accidentally installing GGUF custom operations. A mixed set uses the pinned
`gguf_clip_loader` for GGUF members and native `load_torch_file(..., safe_load=True)`
for the others, retaining the selected order, native progress callbacks and
`DualCLIPLoaderGGUF.load_patcher` for patcher/device operations. Component capture,
file-signature checks, artifact read leases, cancellation and cache identity remain
owned by the existing image preparation/loading path.

This bounded adapter compensates for the pin's case-sensitive
`CLIPLoaderGGUF.load_data` suffix branch; it does not rename files, convert weights
or alter native encoder detection. Scaled-FP8 state dictionaries cannot share the
legacy GGUF custom operations. An explicit Qwen3-VL text-only 4B/8B marker selects
the pinned single-CLIP Dynamic VRAM loader, including its native TensorWiseINT8
eligibility, higher-precision fallback and reload factory. Dynamic VRAM must be
available; RF does not retry a refused dynamic load through the legacy reader.
The native loader still validates Q8_CR tensor metadata and rejects image
embeddings. Text-only encoders cannot join a mixed encoder set. Other supported
Qwen3-VL layouts retain the existing native parsing and patching.

The focused CPU check exercises `.PT`, `.SFT`, `.GGUF` and `.gGuF` through the
production loading branch, mixed ordering, safe-reader flags, native refusals and
parse failures:

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_text_encoder_formats.py test_qwen3vl_text_only_image_loader.py test_gguf_metadata.py test_model_cache.py
```

These checks establish reader dispatch and caller behavior. Real encoder/model
execution, rendered quality and device memory remain separate acceptance gates.

## Header detection and logical shapes

`features/models/model_architecture.py` creates meta tensors from bounded shape
headers. It calls native `model_config_from_unet` with the container metadata,
then native `convert_diffusers_mmdit` and `model_config_from_diffusers_unet` in
the file loader's fallback order. Conversion receives a separate mapping because
the native converter consumes its input. RF neither copies conversion tables nor
constructs models or materializes weight tensors to classify a header.

`features/models/gguf_metadata.py` supplies the same header representation to the
registry and component inventory. It reads typed `comfy.gguf.orig_shape.*`
metadata and the declared logical dimensions of packed ConvRot Q4/Q1 tensors;
it does not unpack or inspect tensor payloads. Malformed or conflicting dimensions
are refused, and the reader's owned mmap closes on success and failure. Native
`config` and `is_512` detection metadata is retained. Full architecture,
quantization, kernel and device validation remains with the pinned native loader
and GGUF adapter. A logical-shape match is not an execution qualification.

Hunyuan image, image-refiner and video checkpoints share a token-refiner key.
Passive video inspection now requires positive video evidence from native config
discriminators, allowing ambiguous headers to reach the existing native meta
classifier. An authoritative Image21/refiner classification can correct a matching
local file's old `Hunyuan Video` cache label; publisher title, provenance and other
metadata are retained. Refiner classification does not enable its special workflow.

The exact previous-to-current native pin comparison (`1332fab` to `fca60fe`)
contains no changes to `supported_models.py`, `model_detection.py`, `sd.py`,
`utils.py` or `folder_paths.py`. No model or format retirement was proven, and
none was removed. The deprecated label on the still-registered Diffusers
directory node is not evidence that native single-file Diffusers conversion or
any model family has been removed. Separately, the pinned GGUF adapter
[`da5f435`](https://github.com/molbal/ComfyUI-GGUF/blob/da5f435cd9a9e54ad27eed417ed72b1f2543ef41/loader.py#L346)
explicitly refuses retired Q4_PT (`int4_compact_gemm` / `int4_pytorch`) metadata
and recommends reconverting as Q8_CR. RF preserves that specific refusal and does
not delete existing files or retire the associated model architecture.

Focused CPU checks use small local files and mocked native loaders:

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_model_file_formats.py test_model_cache.py test_component_profiles.py test_image_preparation.py test_model_recommendations.py test_image_components.py
```

Header changes have their own focused command:

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_model_header_detection.py test_gguf_metadata.py test_distinct_image_architectures.py test_image_components.py test_model_file_formats.py
```

Native container discovery and weights-only refusal checks:

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_native_model_containers.py test_model_file_formats.py test_component_candidates.py test_model_cache.py
```

Local bundle capture, picker/component callbacks and native loader handoff:

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_local_diffusers.py test_component_overrides.py test_model_cache.py test_component_profiles.py test_image_preparation.py
```

The shared picker and public API model inventory has its own focused check:

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_api_model_discovery.py test_api_live_contracts.py test_model_file_formats.py test_local_model_refresh.py
```

These include real pinned native Diffusers SDXL/SD3 and Hunyuan classification on
meta tensors in a guarded CPU subprocess, a tiny local GGUF round trip, malformed
metadata and cache-migration fixtures. They do not certify GPU execution or image
quality. The original source-only findings are preserved in the owned cache
receipt `cache/plan-execution-20261109/native-format-retirement-audit.json`;
the detector and logical-shape gaps identified there are addressed by this change.
