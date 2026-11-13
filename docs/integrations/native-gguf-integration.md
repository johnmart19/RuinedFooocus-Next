# Native image GGUF integration

The image backend uses [Molbal ComfyUI-GGUF](https://github.com/molbal/ComfyUI-GGUF),
not a second app-owned quantization engine. The audited application base is
`3fe5b7e34aaa0a76d03c29c620821c6be21b5e3e`; bootstrap pins ComfyUI
`989d93d0d968e704c589ce0146bb18c77f6b50cc` and Molbal
`da5f435cd9a9e54ad27eed417ed72b1f2543ef41`. These are source/CPU conclusions.
They do not qualify every GGUF type, architecture, hardware configuration or image.

## Actual consumers

| Surface                   | Application seam                                                                                                                                                                                                                                                                                             | Native implementation and limit                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Bootstrap                 | [`git_repos` and repository path setup](../../ruinedfooocus/app/bootstrap.py)                                                                                                                                                                                                                                | Pins/clones both repositories. `pip/modules.txt` pins GGUF 0.19.0, SentencePiece and protobuf; this audit installs nothing. Direct imports execute the vendor package/nodes registration; RF does not need to launch a Comfy graph server.                                                                                                                                                                               |
| Diffusion                 | [`pipeline_utils.load_diffusion_model`](../../ruinedfooocus/features/generation/pipeline_utils.py), [`gguf_loader.load_diffusion_model`](../../ruinedfooocus/backends/comfy/gguf_loader.py)                                                                                                                  | Case-insensitive GGUF dispatch to `UnetLoaderGGUF.load_unet`, native `gguf_sd_loader`, custom operations and `GGUFModelPatcher`. Comfy `folder_paths.map_legacy` maps `unet` to the registered `diffusion_models` root.                                                                                                                                                                                                  |
| Dtype, storage, mmap      | Same diffusion adapter                                                                                                                                                                                                                                                                                       | Uses the legacy native node with native default dequant/patch dtype and `patch_on_device=True`. Vendor `loader.py`, `ops.py`, `dequant.py`, `quant_ops.py` own mapped storage, packed tensors, device handling and dequantization. RF does not reimplement bit unpacking or the memory manager. Diffusion dynamic-VRAM nodes remain separate; the exact text-only encoder route below selects its native dynamic loader. |
| MiniMax U16G              | `gguf_loader.compact_minimax_operations`                                                                                                                                                                                                                                                                     | The exact named local variant chooses native Q8 operations plus native `GGMLOps.Linear` for its ordinary packed layers. This is a narrow operations dispatch, not a new codec. Other filenames use the ordinary native node; equivalent performance for renamed variants is unqualified.                                                                                                                                 |
| Mixed text encoders       | [`text_encoder_loader.load_mixed_clip`](../../ruinedfooocus/backends/comfy/text_encoder_loader.py), [`image_pipeline.load_base_model`](../../ruinedfooocus/features/generation/image_pipeline.py)                                                                                                            | Keeps encoder order; GGUF uses native `gguf_clip_loader`, ordinary containers use native `load_torch_file(safe_load=True)`, then native `DualCLIPLoaderGGUF.load_patcher`. The common patcher handles the supplied state-dictionary list. RF adds suffix casing dispatch and captured-source binding.                                                                                                                    |
| Mixed-format refusals     | Same text adapter; [`component_candidates`](../../ruinedfooocus/features/models/component_candidates.py)                                                                                                                                                                                                     | Preserves the native mixed scaled-FP8 refusal. Explicit Qwen3-VL text-only 4B/8B metadata selects the native single-CLIP Dynamic VRAM route; mixed sets and image inputs remain refused. A catalogue label cannot override either constraint.                                                                                                                                                                            |
| Implicit vision projector | [`gguf_projectors`](../../ruinedfooocus/features/models/gguf_projectors.py), [`image_preparation`](../../ruinedfooocus/features/generation/image_preparation.py)                                                                                                                                             | Captures the pinned resolver's adjacent mmproj selection for Qwen2-VL, Qwen3.5 and non-text-only Qwen3-VL. The native conversion/merge still runs. See the corrected source-identity contract below.                                                                                                                                                                                                                     |
| Still-image VAE           | [`model_formats.require_image_vae_format`](../../ruinedfooocus/core/model_formats.py), [`component_profiles.component_contract`](../../ruinedfooocus/features/models/component_profiles.py)                                                                                                                  | GGUF VAE is refused before weights are read. The pinned `VAELoaderGGUF` accepts only `minimax_h3_vae` with `int8_convrot`, a video codec. Native safetensors/Torch VAEs and explicitly supported paired approximate codecs remain separate routes.                                                                                                                                                                       |
| Actual GGUF LoRA          | [`gguf_adapters`](../../ruinedfooocus/features/models/gguf_adapters.py), [`gguf_lora`](../../ruinedfooocus/backends/comfy/gguf_lora.py), `image_pipeline.load_loras`                                                                                                                                         | Bounded typed adapter discovery and native `lora.load_gguf_lora`/key remapping. Pinned factors are paired 2-D F32/F16/BF16/Q8_0; RF verifies current source, target shapes, finite alpha and complete native application. Ordinary LoRAs applied to a packed GGUF checkpoint are a different native patch path.                                                                                                          |
| Discovery / Inbox         | [`model_handler`](../../ruinedfooocus/features/models/model_handler.py), [`model_inbox`](../../ruinedfooocus/features/models/model_inbox.py), [`gguf_metadata`](../../ruinedfooocus/features/models/gguf_metadata.py)                                                                                        | `.gguf` is a case-insensitive candidate. Bounded logical headers classify roles; typed LoRAs have their own scanner. Unknown files remain explicit choices where allowed. Header admission never creates mmap/tensor views; this intentionally differs from the heavy vendor loader.                                                                                                                                     |
| Registry / Auto / size    | [`model_registry`](../../ruinedfooocus/features/models/model_registry.py), [`native_recipes`](../../ruinedfooocus/features/models/native_recipes.py), [`resolutions`](../../ruinedfooocus/features/models/resolutions.py)                                                                                    | Exact source/download identities and current native family/variant evidence control annotations and recipes. Quantization labels describe artifacts; they do not prove architecture, sampler defaults, memory fit or dimensions. Recommended sizes come from current recipe/publisher records, not the GGUF suffix.                                                                                                      |
| UI / API / Library        | [`ui_model_components`](../../ruinedfooocus/features/models/ui_model_components.py), [`component_profiles.snapshot_request`](../../ruinedfooocus/features/models/component_profiles.py), [`api_runtime`](../../ruinedfooocus/app/api_runtime.py), [`gallery.ui`](../../ruinedfooocus/features/gallery/ui.py) | Main model components expose Auto and explicit overrides, validated at capture and load. API and normal worker submission capture the same selected files. Library replays public checkpoint/override metadata into a fresh capture; private live file objects are not serialized. No browser session was exercised by this audit.                                                                                       |
| Chat GGUF                 | [`backends/llama`](../../ruinedfooocus/backends/llama/)                                                                                                                                                                                                                                                      | Native llama.cpp is intentionally separate from image Molbal. Image GGUF metadata support does not route a language model through a diffusion loader.                                                                                                                                                                                                                                                                    |

## Explicit Qwen3-VL text-only image encoder

The image component picker can select a local text-only Qwen3-VL 4B/8B GGUF
through its explicit advanced override. It remains **Compatibility unknown**
until the native loader validates the artifact. Bounded typed metadata selects
this route only when `general.architecture=qwen3vl`,
`comfy.gguf.qwen3vl.text_only=true` and the declared variant is `4b` or `8b`.
Architecture names alone never route a Chat model or sort the Inbox.

`text_encoder_loader.load_mixed_clip` passes that one selected file and the
current CLIP type to the pinned `_load_dynamic_gguf_clip` with
`enable_q8_native=True`. Native Dynamic VRAM availability, Q8_CR tensor metadata,
kernel/device eligibility, higher-precision fallback, vision pruning and reload
factory remain intact. A small reader-path string adapter preserves original
filesystem spelling while handling the pin's lowercase suffix predicate; no file
is renamed and no native globals are patched. This loader rejects a mixed
encoder set or image embedding inputs. It does not qualify vision editing with
a text-only encoder, and a disabled Dynamic VRAM route is not silently retried.

`test_qwen3vl_text_only_image_loader.py` executes the pinned loader and pruning
functions with tiny CPU dependency fixtures. It verifies the native eligibility
gate, fallback, captured reload arguments and image-input refusal. Together with
`test_text_encoder_formats.py` it proves dispatch/casing and metadata-only
admission, without real-model, GPU, performance or visual qualification.

## Captured static LoRA sources

`features/models/lora_sources.py` captures card and original inline-prompt LoRAs
with the existing shared parser and merge rules. Explicit prompt weights still
override the same card once. Installed sources keep their resolved file and stat
signature through the queue; changed roots cannot select a different file.
Public selections remain relative names inside configured LoRA folders.

The optional `image_pipeline.load_loras(..., captured_loras=...)` input verifies
that the runtime selection and files still match before cache reuse. It passes
captured paths to the existing native Torch/GGUF readers and LoRA application
methods. Adding, replacing or removing LoRAs uses a fresh capture and still starts
from the original base model. Dynamic prompt expansion cannot add an uncaptured
adapter silently. No weight hashing or inventory scan is added to generation.
Original prompt batches retain immutable name/weight rows for each scalar prompt;
the existing worker split selects that exact row from shared captured file refs.
A different prompt or another child's weights cannot replace the captured row.

A missing active catalogue LoRA retains its exact existing preparation request,
including destination, URL, byte count and optional digest. Only the matching
typed preparation receipt can bind its new signature. Newly prepared GGUF files
use the existing bounded adapter inspection; arbitrary missing custom files need
an installed selection. These are source/CPU contracts, not model/GPU validation.

```sh
python tools/check_cpu.py test_lora_source_capture.py test_lora_names.py test_gguf_lora_dispatch.py test_model_cache.py
```

## Corrected implicit-projector identity

Previously, replacing an adjacent mmproj left the main encoder/checkpoint cache
valid. Molbal then had an implicit source absent from RF's queued request and
component identity. A tiny actual production-cache regression reproduced one
native/CLIP load after replacement where two were required.

The projector adapter reads only bounded scalar metadata and executes the pinned
`gguf_mmproj_loader` resolver prefix, stopping before its first weight read. It
preserves native filename matching, enumeration order and ambiguity rules. The
capture freezes encoder, selected projector and resolver source signatures,
including an explicit absent-projector result. Qwen3-VL text-only encoders do not
capture a projector. Existing header reader output/digests are unchanged.

Submission captures installed sources; preparation verifies them and resolves a
newly acquired encoder only after acquisition completes. Queued source changes
fail before cache reuse. Fresh/direct requests re-evaluate native selection, so
replacement, addition or removal invalidates the corresponding cache. The
selected projector joins runtime component signatures, including native tensor
context identity. The normal component reader already leases the encoder; the
mixed loader leases each unique projector once and verifies both before/after
native work. It does not nest another encoder lease.

Per-call copies of native function globals restrict directory selection to that
captured choice or absence. Native module globals, heavy conversion, patching,
dtype policy and reload-state construction stay native. The imported reader must
come from the captured vendor source. A newly added file cannot silently enter a
queued request. Native missing-projector behavior is preserved; this does not
qualify an edit workflow that needs vision weights but lacks them.

Focused CPU checks use tiny owned files, the actual pinned resolver and projector
function bodies, and actual RF capture/cache/reader callers; heavy weight loading
is mocked. They establish source identity and dispatch, not real-model output,
GPU support, quality, performance or stability. Existing source-bound GGUF header,
LoRA, component and API proofs remain bound to their original sources; this audit
does not relabel those results or rerun their full suites.

```sh
python tools/check_cpu.py test_gguf_projector_dependencies.py test_text_encoder_formats.py test_image_preparation.py test_model_cache.py
```

The finite audit found one Required integration defect, corrected above. No
second app dequantizer or dead production image GGUF loader was demonstrated.
Unused vendor nodes represent capabilities outside these selected routes; their
presence does not imply RF UI/API support. In particular, CUDA-specific packed
formats and the dynamic loader are not newly qualified by these CPU checks.
