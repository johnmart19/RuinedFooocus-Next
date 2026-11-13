# Model folders

Standalone Image tools additionally use `geometry_estimation` for native MoGe
and Lotus depth safe tensors; Lotus also needs a compatible SD VAE. The
`detection` folder holds a complete native MediaPipe face landmarker bundle for
still facial overlays. RT-DETR/SAM retain their existing checkpoint/diffusion
utility roots. Marigold albedo uses task-specific saved conditioning in
`embeddings`, distinct from textual inversion. See
[still geometry utilities](integrations/native-still-geometry-utilities.md).

Native image [patches](integrations/native-model-patches.md) include Anima LLLite,
Qwen DiffSynth, Z-Image Fun and Flux USO, alongside Qwen Image 2.1 Union.

Local GLIGEN weights use `models/gligen`; native nested Torch hypernetworks use
`models/hypernetworks`. Their ordered custom roots are configured under Settings
→ Profiles & storage → Paths. Discovery is explicit from **Native layout and
attention** in Create; see its [compatibility and replay guide](integrations/native-layout-attention.md).

The image loader uses the pinned ComfyUI backend. The default layout follows
its canonical `text_encoders`, `diffusion_models` and `embeddings` names while
continuing to find files in the old locations. Files already in model folders
stay where they are. Use the canonical folders below or drop files into Inbox.
The small `put_..._here` files keep ready-to-use folders in source distributions;
they contain instructions, not model weights.

| Folder under `models/` | Purpose and application support                                                                                                                                                        |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `checkpoints`          | Complete supported image checkpoints; existing diffusion-only models here still work.                                                                                                  |
| `diffusion_models`     | Supported image diffusion-only weights, including image GGUF models. They appear in the same Models browser as checkpoints and need matching external text encoders/VAE.               |
| `diffusers`            | Local image bundles with `model_index.json` and [supported component files](integrations/native-model-formats.md#local-diffusers-bundles). Use **Models → Refresh** to discover them.  |
| `text_encoders`        | Canonical location for image-model text encoders. The profile key and download catalog identity remain `path_clip`.                                                                    |
| `vae`                  | Matching external image VAEs. Pixel-space image models do not require one.                                                                                                             |
| `clip_vision`          | Native still-image StyleAdapter/Redux vision towers and other explicitly compatible reference workflows.                                                                               |
| `style_models`         | Native StyleAdapter and Redux weights, validated separately from their vision towers.                                                                                                  |
| `photomaker`           | Native PhotoMaker SDXL weights with its explicit trigger/token contract.                                                                                                               |
| `loras`                | Adapters selected in Create, including [typed GGUF LoRAs](integrations/native-model-formats.md#explicit-gguf-image-lora), applied through native image-model and text-encoder loaders. |
| `embeddings`           | Textual-inversion embeddings used through `embedding:filename` with a compatible text encoder.                                                                                         |
| `controlnet`           | Models used by the existing canny, depth, recolour and sketch guidance workflows.                                                                                                      |
| `model_patches`        | Compatible safetensors patches for [Qwen Image 2.1 Fun ControlNet Union](integrations/native-special-image-workflows.md#qwen-image-21-fun-controlnet-union).                           |
| `geometry_estimation`  | Depth Anything 3 small/base weights for automatic depth preprocessing.                                                                                                                 |
| `background_removal`   | Compatible BiRefNet `.safetensors` or `.sft` weights for [Create → Image tools → RemBG](integrations/gpu-background-removal.md).                                                       |
| `upscale_models`       | Image upscalers used by existing image upscaling controls.                                                                                                                             |
| `vae_approx`           | Native approximate preview decoders and explicit full encoder/decoder pairs; see the paired codec guide.                                                                               |
| `llm`                  | Local Chat and optional prompt-rewrite GGUF weights, handled by the separate llama.cpp runtime.                                                                                        |
| `inbox`                | Automatic sorting for finished, identifiable model files. See [Inbox sorting](#inbox-sorting).                                                                                         |

The official [SenseNova U1.5 shard bundle](integrations/native-sensenova-bundles.md)
can be installed as one subfolder of `checkpoints` or `diffusion_models`.
Models Refresh offers its validated index and keeps its eight shards together.

## Models and downloads

**Settings → Models / downloads** is the shared acquisition view. **Create →
Models → Browse model catalogue** opens it without changing your prompt or selected
checkpoint. **Discover** lists registered catalogue artifacts, including installed
ones. **On device** lists installed files and custom local imports. Search and
the category, workflow and quantization filters use the current filesystem snapshot.
The selected file and its details stay open when a view or filter hides its card.
In **Create → Models**, the selected publisher name and version also stay above
checkpoint search. Card names omit technical identity suffixes; exact filenames
remain in details. Installed image diffusion files, including registered Qwen
Image 2.1 safetensors and GGUF variants, use their native family identity for
activation. The Image models category excludes declared video-only entries.
**Refresh local catalogue** rescans the configured roots without loading weights
or moving model files. Local sidecars and cached artwork appear immediately.
Bounded background requests can then add published artwork for an exact pinned
catalogue source; offline mode keeps the local/cache view and placeholders.

Cards use existing local sidecars and cached previews. When two variants share a
stem, an exact sidecar such as `model.gguf.preview.png` identifies the variant's
image. Select a card for its exact file, source, license, published/local size,
registered companions and recorded memory estimates for each available context.
Missing evidence stays unknown. Available weights are separate from
a callable workflow; video-only resources and bundle manifests cannot be downloaded
from this view. A same-name file with a different published size remains visible
as a conflict and cannot be replaced or activated by the catalogue.

The primary action follows the selected file: **Download selected files** for a
missing artifact, **Resume download** for a matching saved HTTP partial, or **Use**
for an installed compatible resource. Cancel belongs to the running download;
another installed model remains usable while that download continues. Missing
companions have a separate secondary action when the checkpoint is already
installed. Each download captures the current source tuples and destinations.
The optional companion set uses registered dependencies; additional workflow
dependencies without registered identities remain described in the details.
Download options include a free-space reserve (100 GiB by default) and an explicit
choice for unknown-size downloads. Published file size is not a model's RAM/VRAM
requirement. No device-fit claim is inferred from quantization or file size.

Preparation uses the same bounded service as image and Chat requests. Progress
belongs to the captured job even while you browse another card. **Cancel this
download** cancels this session's subscriptions, retaining installed files and
other sessions' subscribers. A failed file cancels only this session's remaining
subscriptions. Completion, partial failure and cancellation refresh local presence;
none changes the active model. Explicit **Use installed checkpoint in Create**
revalidates the local file and selects only the checkpoint control. Prompts and
other deliberate generation choices follow the existing model-selection contract.
Filtering keeps the selected file's details and action available even when its
card no longer matches. Use still checks the current selection and local file.
Selecting a custom image card inspects its native header through the existing
model index. Recognized supported image families can then be selected in Create;
unknown or changed files remain unqualified. Card selection publishes its details
and action availability together, without relying on a hidden-state event.

An installed text encoder or VAE can offer **Use for selected checkpoint** when
its native reader, tokenizer/layout and current companion set match an external
role on that checkpoint. This changes the session choice without downloading
the file again or saving a profile. The component inventory refreshes to show
the effective file; **Save this component choice** remains the explicit
persistent action. Bundled roles stay fixed, and unverified files remain
available through the advanced component editor.

Installed Chat weights offer **Select installed file in Chat**. This selects the
exact local family/file and opens Chat; **Load** remains explicit. The current
loaded runtime and conversation draft stay intact. Image-tool resources without
an atomic caller handoff remain visible with their compatibility details and do
not advertise an unsupported Use action.
Supported installed image tools open **Create → Image tools** with the exact
resource selected. This selects that tab's own group, independently of the
Generation/Models group. Library Edit, Inpaint and Remove background use the
same destination; Upscale opens Generation.

A failed or cancelled transfer can be retried. An ordinary HTTP partial offers
Resume only after its saved source, digest and size receipt matches; the server
must still permit resuming. SDK partial cache is shown as unverified until that
transport checks it. Size conflicts remain errors rather than replacement
downloads. Artwork arriving in the background updates matching images only,
preserving the selected file, search, page, focus and scroll.

After a checksum-verified download, current native image source details can be
cached directly from the preparation service's completion identity. This avoids
rehashing the same large file when its source-specific defaults are selected.
Changed files invalidate that proof. A cache write failure retains the installed
file and explains the explicit Inspect route. Source identity does not qualify
device fit or image quality. See [prepared source metadata](integrations/prepared-native-source.md).

Catalogue transfers remain disabled offline; installed files, evidence and local
selection are still available. The guest interface does not construct owner Settings
or these acquisition endpoints. Disk reserve and companion choices are session
options, not hidden changes to the user's saved profile.

### Actual folder consumers

The catalogue uses `PathManager.get_model_folders`; it introduces no new folder
authority. Existing canonical/legacy root ordering, custom roots and collisions
remain governed by `core/model_folders.py` and the local model index.

| Root/category                                             | Actual consumer                                                                                                                          |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `checkpoints`, `diffusion_models` (`unet` alias)          | `features/models/model_handler.py` local checkpoint index; `backends/comfy/model_paths.py` native registration; frozen image preparation |
| `text_encoders` (`clip` alias), `vae`                     | Contextual component inventory and captured image preparation; native model loaders                                                      |
| `loras`, `embeddings` (`embedding` alias)                 | Existing adapter cards and native image/text-encoder paths                                                                               |
| `controlnet` (`t2i_adapter` alias), `geometry_estimation` | Captured image guidance/depth assets and native ControlNet/depth consumers                                                               |
| `model_patches`                                           | Captured Qwen Image 2.1 Union assets loaded through the pinned ComfyUI model-patch registry                                              |
| `background_removal`                                      | Native BiRefNet selection, preparation and foreground masks                                                                              |
| `upscale_models`                                          | Shared image upscaling controls, captured preparation and native `UpscaleModelLoader`                                                    |
| `vae_approx`                                              | Sampling previews; explicit full pairs use `features/models/approximate_codecs.py` and the native VAE loader                             |
| `llm`, `vision`                                           | Chat local model/projector discovery and captured Chat preparation; source qualification remains separate                                |
| `clip_vision`, `style_models`, `photomaker`               | Captured native still-image reference adapters; each reader validates its own dimensions and compatible base family                      |
| `latent_upscale_models`                                   | Existing video pipelines; still-image native latent-upscaler contracts remain a separate review queue                                    |
| `faceswap`                                                | Retained legacy catalogue/root; a restoration weight's presence does not establish a callable restoration workflow                       |
| `diffusers`, `diffusers_cache`                            | Local bundle discovery and captured component files, plus the separate converted bundle cache                                            |
| `inbox`                                                   | The running app scans this root and uses the first configured destination for each identified model category.                            |

`PathManager.preparation_request` now joins an exact registered source tuple with
published size/checksum evidence, so legacy preparation and the catalogue submit
identical requests for the same destination. Conflicting explicit size/hash or a
changed download tuple is rejected; registry metadata never silently overrides it.
Bundled annotations use a [versioned resource index](integrations/model-registry-schema.md)
over family, artifact-role and lazy qualification files. Download tuples keep their
existing catalogue locations; source support and exact-artifact qualification stay
separate.
The [native core catalogue](integrations/native-core-image-catalogue.md) adds
immutable image-model tuples with explicit native family/variant, bundled or
external components and applicable defaults. Published payload size and header
classification remain separate from local installation and hardware qualification.
The shared service retains atomic publication, resume, integrity checks and file
leases. No models, aliases or profile roots were moved by this acquisition change.

Image tools lists installed ControlNet and T2I adapter files from the ordered
`controlnet` roots, including the existing `t2i_adapter` alias. Safetensors headers
are matched to the pinned native loader branches; torch containers stay explicitly
unverified until selected and loaded with the native safe reader. Filenames and
quantization labels never qualify a control family. Automatic retains only the
stock SDXL presets. Explicit choices freeze their resolved path and signature
before queueing; root changes, replacement files and incompatible families are
rejected instead of silently substituted. The control cache includes the primary
checkpoint, LoRA state and patcher identity because native difference controls can
incorporate the current model weights. Sampling still uses native
`comfy.controlnet.load_controlnet` and `ControlNetApplyAdvanced`; no new device or
download service is introduced. The source/CPU checks are
`test_control_model_selection.py`, `test_control_model_ui.py` and their declared
image preparation/API/cache callers. Real control-model execution remains a
separate acceptance gate.

## Inbox sorting

Drop finished files into `models/inbox`, or use the Inbox Folder configured
under Settings. Sorting starts with the app and normally notices a new file
within two scans, about ten seconds before any file or network processing.
The sorter uses exact catalogue checksums and recognized model headers.
With online model metadata enabled, it can also ask Civitai about an
unidentified file by SHA-256. Local identification works offline.

Recognized files and their matching previews, captions and metadata move
together. The sorter uses the first configured folder for the category and
keeps the filename. It keeps subfolders for categories that scan them. Refresh
an open model picker to see newly sorted files. An
occupied destination, incomplete file or uncertain category leaves the file
in the inbox. Place split weights and complete Diffusers bundles in their
documented folders. Use a model's own destination when you already know it.

## Profiles and priority

**Settings → Profiles & storage → Paths** exposes Checkpoint Folders, Diffusion
Model Folders, Text Encoder Folders, LoRA Folders, Embedding Folders and the other
existing storage controls. Save and restart to apply path changes to model
discovery and the initialized backend. A blank Diffusion Model Folders list
disables those additional browser roots.

`paths.json` retains existing settings keys and custom values. Newly created
profiles use `path_clip: "../models/text_encoders/"`,
`path_diffusion_models: ["../models/diffusion_models/"]` and
`path_embeddings: ["../models/embeddings/"]`. Profiles already pointing at
`../models/clip/` continue to prefer that folder. Either standard local location
also searches its counterpart. The same alias rule applies to standard
`diffusion_models`/`unet`, `embeddings`/`embedding`, and `controlnet`/`t2i_adapter`
folders. Explicit custom roots remain authoritative and do not gain an unexpected
local fallback. Folder discovery itself never creates alias folders or downloads.
New downloads targeting one of these standard local aliases use its canonical
counterpart. Existing files in either folder retain their configured lookup
priority. Startup prepares canonical destinations rather than recreating missing
local alias directories. Custom destinations are unchanged.
Text Encoder Folders accepts one root per line; saved single-string roots remain
readable and retain their order when saved as a list. Existing scalar cache and
download fallbacks use the first root while file lookup searches all of them.

The image browser searches configured checkpoint roots before diffusion-model
roots, in their configured order. Duplicate relative filenames appear once and
resolve to the first matching root; use distinct names when keeping variants.
Component inventory, catalog lookups and native folder registration use the same
ordered roots. Existing native backend defaults remain registered after profile
roots, including their native supported-extension rules.

The native capability ledger separately records all 27 pinned native folder roles
and five GGUF aliases, with native reader and RF root category. A registered bucket
does not imply an RF acquisition/UI consumer. GLIGEN, hypernetworks, classifiers,
detection and advanced codec/patch roles retain named implementation or consumer
review work; audio, video, training and arbitrary custom-node roots retain scoped
exclusions. See [native coverage](integrations/native-model-coverage.md#updating-the-pinned-coverage-inventory).

The default data root is the checkout; `RF_DATA_DIR` selects another writable root.
Legacy relative values still resolve from the former `modules/` base, so
`../models/text_encoders/` resolves to `<data root>/models/text_encoders/`.
See [Settings](settings.md) for profile persistence and restart behavior.

## Consolidating an existing installation

The source distribution keeps placeholders only for canonical folders. Existing
`clip`, `unet`, `embedding` and `t2i_adapter` folders still work and need not be
renamed for compatibility. To consolidate them explicitly, move their contents
to `text_encoders`, `diffusion_models`, `embeddings` and `controlnet`, respectively.
Preserve relative subdirectories and companion files. Check every destination
first and stop on a collision; never overwrite a model with the same name.
Record sizes and hashes before moving and verify them afterward. Remove the old
directory only when empty. Update standard local profile paths to their canonical
counterparts if desired; retain custom paths and their order.

Checkpoint subdirectories can contain model weights, download metadata, range
manifests or user organization. Their names alone do not establish that they are
obsolete. Keep unknown files, partial downloads and cache metadata until their
ownership and purpose are established. A diffusion-only weight can remain under
`checkpoints`, since both model categories share the image browser.

## Support boundary

File placement cannot make an unsupported architecture compatible. Use **Models →
Model → Image model components** to inspect the selected image model's expected
encoders/VAE and explicitly download missing catalog components. A diffusion-only
file is detected by the native loader and reuses its detected model while loading
missing external components; it is not treated as a complete checkpoint merely
because it appears in that browser.

The model browser selects installed files. Search, selection, details and Refresh
use local files and cached metadata; they do not acquire a default checkpoint.
Clicking the already selected model leaves it selected without repeating its
selection callbacks. Place additional supported weights in a configured root
and use **Refresh All Files** to rescan.

The component table shows only the selected image model's detected encoder and
VAE roles. **Built into checkpoint** means Automatic uses the native bundled
component; an external profile override is ignored for that bundled slot and is
not counted as a used file. An explicit compatible [full approximate codec](integrations/native-approximate-codecs.md)
can replace a bundled VAE and accounts for both local encoder and decoder files.
Existing saved overrides remain available to models that need
external components. **Component choices for this checkpoint** edits one of the
displayed roles. Automatic resets only that checkpoint/role to its recipe.
Legacy profile-wide choices remain inherited by untouched roles. Saved choices
are tied to the checkpoint's canonical path, size and timestamps, so replacing a
file starts a fresh scope; two different checkpoints retain independent choices.
Choices apply at the next model load, without an application restart. For
inspectable safetensors and image GGUF checkpoints, a queued image request, API
request and saved image setup capture their own choices;
subsequent edits cannot change an already queued request.
Legacy pickle checkpoints keep their existing profile-wide fallback behavior;
their bundles cannot be inspected safely without loading weights, so the
contextual editor and complete component snapshots are unavailable for them.
Installed alternatives use cached safetensors headers and the initialized native
text-encoder classifier. Known incompatible architectures are excluded from new
alternatives. Unverified alternatives appear only with the explicit advanced option;
header compatibility does not certify output quality. No model weights are
loaded when browsing. Built-in text encoders stay locked to the native bundle;
VAE roles can offer the compatible full approximate pair explicitly.
Qwen Image/Anima and Wan2.1 VAE alternatives require the native RGB, 16-latent,
eightfold spatial layout, including its temporal resampling signatures. A planar
16-channel VAE is not interchangeable. Wan2.2 and Qwen Image2.1 have distinct
layouts and are excluded from that role; other unrecognized formats remain
unverified and require the advanced option.
Identified [Qwen Layered](integrations/native-qwen-layered.md) sources instead
select their own `vae_qwen_image_layered` RGBA role (16 latent channels, 8× scale).
The native loader verifies four output channels before sampling.
An existing missing, incompatible or unverified choice remains displayed with
its status until you explicitly replace it or save Automatic. Saving commits
the profile before publishing new active settings; a failed save preserves the
previous settings and selection.
Missing catalogue companions have a separate **Download missing** action, disabled
offline. Its byte progress uses the shared preparation service; an unknown total
stays indeterminate. **Cancel component download** cancels this session's
subscription, preserving other sessions' transfers and installed files. Transfers
have bounded retries and report completion, cancellation or failure. Presence
alone does not certify architecture compatibility or integrity.

**Checkpoint tools** applies to the current image model. Inspect reads its local
safetensors header and checks whether FP16 conversion is useful; **Check weights** and **Create FP16 copy** are explicit disk
jobs. FP16 conversion preserves the source and non-FP32 tensors, refuses invalid
or overflowing values and never overwrites an existing result. The action is
disabled when the header predicts less than one percent saving, including wholly
BF16/FP8 files. These disk jobs currently run to completion once started. Selecting
another checkpoint updates its header details while a disk job runs. A completed
job for the previous selection cannot replace the current report or action
availability. A completed FP16 copy is retained; refresh files to list it if the
selection changed before its result arrived. INT8
export is unavailable: the pinned native loader requires per-layer `comfy_quant`
metadata and scales, so casting tensors to integers cannot produce a supported
export. Loading an existing compatible quantized model is a separate capability.

## LoRAs and embeddings

Search the combined resource cards by name or metadata, or filter by type. Select
a card to see its identity and known triggers, then use **Activate** or double-click
or drag it into Selected. **Remove** deactivates the selected LoRA or undoes an
unchanged embedding insertion made by the cards in this session. Manually written,
replayed or subsequently edited embedding references remain editable in the prompt;
the cards never relocate them. Repeated activation is idempotent. Multiple active
resources retain their own weights and prompt destinations.

LoRA weight appears only for a selected LoRA. Displaying its trigger words does
not insert them into the prompt; **Append LoRA trigger words to the prompt** remains
an explicit option. Known triggers come from local metadata or the existing text
sidecar cache. An absent LoRA trigger is shown as unknown rather than inferred
from its filename. Active resources show attributed triggers and their targets.

Each selected embedding has its own **Automatic / Positive / Negative** destination.
Automatic recognizes established negative embedding identities or a description
recommending only the negative prompt; otherwise it uses Positive. A local JSON
sidecar can supply `prompt_target: "Positive"` or `"Negative"`. An explicit per-item
choice overrides Automatic for future activations without moving existing words.
Embedding triggers use verified metadata or an unambiguous installed file stem;
`embedding:filename` remains supported. Manual weighted references such as
`(embedding:detail:0.8)` retain their native encoder behavior. Refresh rescans local
resources without loading weights or downloading files. Configure roots in
**Settings → Profiles & storage → Paths**.

The pinned backend also defines `configs`, `gligen`, `hypernetworks`,
`detection` and other native node categories. A folder alone
does not add an application workflow. Existing `clip_vision`, `latent_upscale_models`,
`latent_upscalers`, `faceswap` and `vision` folders are retained; their
presence is not a new support claim. Local `style_models` and `photomaker` files
have explicit image-prompt adapter consumers. RemBG uses native BiRefNet by
default when no background model selection is saved and retains Hardware/ONNX
as an explicit choice.
No video model support is added or qualified by this layout.

The cached **Flux.3 Klein** image family uses the existing FLUX 3 encoder/VAE
slots. Its automatic distilled recipe is selected from a matching local header,
not filename words; Base or ambiguous variants require explicit sampling choices.
See [FLUX 3 defaults and source limits](integrations/flux3-native-defaults.md).
