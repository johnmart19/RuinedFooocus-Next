# ComfyUI 0.45 adoption contract

The embedded backend is pinned to the official
[v0.45.0 release](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.45.0), commit
`989d93d0d968e704c589ce0146bb18c77f6b50cc`. The
[delta from the preceding pin](https://github.com/Comfy-Org/ComfyUI/compare/fca60fe70e5b84c5c6b058263653d15cb3401e44...989d93d0d968e704c589ce0146bb18c77f6b50cc)
contains ten commits and 35 changed files. This contract was reviewed on
2026-11-11 against exact source snapshots; it does not cover later upstream tips.

Bootstrap's `git_repos` owns the exact pin. The model registry's finite native
coverage index, source URLs and supported-model/CLIP inventories travel in the
same adoption feature. A native registration does not create an application
workflow, a reviewed artifact tuple or hardware qualification.

The catalogue retains the existing GLIGEN and hypernetwork consumer facts from
the [native layout and attention workflows](native-layout-attention.md). Their
folder roles and CPU caller evidence remain distinct from real-model acceptance;
the adoption feature depends on that source owner so its catalogue hashes refer
to the same reviewed role data in an isolated prerequisite replay.

## Packages and startup ownership

The release's [requirements](https://github.com/Comfy-Org/ComfyUI/blob/989d93d0d968e704c589ce0146bb18c77f6b50cc/requirements.txt)
change only `comfyui-workflow-templates`, from `0.11.103` to `0.11.105`.
`comfy-kitchen==0.2.41` and `comfy-aimdo==0.6.0` already belong to the preceding
contract. Torch, torchvision, the Transformers/tokenizer constraints and the
upstream Python floor of 3.10 are unchanged by this delta.

RF installs the template package through the exact native requirements base in
`packaging.package_plan.bootstrap_plan`. There is no duplicate RF overlay pin.
Normal startup updates the vendor to the selected revision, checks the native
requirements even when preserving an installed Torch bundle, and resolves the
base plus RF overlay as one frozen package plan. A template mismatch therefore
requests that package reconciliation while keeping the qualified Torch recipe.
This adoption introduces no installer, system HIP or blanket runtime upgrade.
Explicit skip-install/offline startup continues to require the caller to supply
the expected vendor and package environment.

The meta-package's exact pure-Python/data closure changes four packages:
`comfyui-workflow-templates` `0.11.105`, `comfyui-workflow-templates-core`
`0.3.396`, `comfyui-workflow-templates-json` `0.1.131` and
`comfyui-workflow-templates-media-assets-02` `0.1.30`. Their published wheels total
82,395,011 bytes. The other five media-package versions already match the reviewed
baseline. The prepared hash-locked data-only upgrade plan preserves those packages
and all native runtimes; applying it waits until active package readers finish.

RF's `read_official_templates` and `recipe_inventory` tools really consume the
manifest/JSON packages. The coverage index therefore keeps the audited `.103`
baseline and reviewed `.105` target identities separate, including package
versions, publisher wheel hashes, manifest hashes and recipe-inventory hashes.
The reviewed target contains 688 JSON assets and 672 workflow recipes, including
156 still-image/utility recipes, versus 680/664/154 in the baseline. Target data
was verified and read from isolated wheel contents without installing/importing
the package. All recipe JSON matches its publisher manifest; both versions have
13 explicitly recorded mismatches limited to non-recipe index JSON. These are
not silently promoted to verified recipe data or corrected in the environment.

Keep active native work bound to its loaded source until it finishes. The source
adoption can be reviewed independently of a live checkout; updating a shared
vendor while a GPU worker owns it is a separate coordinated operation.

## Existing application routes

The reviewed public constructors and loaders retain their contracts, including
`load_clip`, `load_checkpoint_guess_config`, `load_diffusion_model`,
`load_diffusion_model_state_dict`, `load_lora_for_models`, and CLIP/VAE entrypoints.
The API audit at RF `441dfa5021e66d396a7d02dabd460262bde0cf0e` found no missing
direct exports across 151 RF source/test import occurrences in the changed modules.
Dynamic attributes and model behavior need their own runtime checks.

| Native change                             | App disposition                                                                                                                                  |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| CombinedLinear prefetch/direct use        | Reuse the native fix; fresh direct casts discard stale combined residency and reload member weights. The existing FP8 split path remains intact. |
| Greedy repetition/presence penalties      | Reuse the native penalty policy before the greedy/sample branch; numeric CPU tests cover mask/history behavior.                                  |
| Qwen generation dtype and multimodal RoPE | Reuse native device dtype policy and next-position indexing. Full model answers and MTP performance remain unqualified by these checks.          |
| Fibo2/Qwen image preprocessing            | Keep native nodes and shared resize helpers; no application resize copy is introduced.                                                           |
| RT-DETR box drawing                       | Existing batch-one image utility retains its native drawing contract; a separate CPU comparison covers empty and multiple-class boxes.           |
| FLUX conditioning/LoRA aliases            | Existing route retains its public interfaces; newly added multiview behavior has no new RF control or execution claim.                           |
| 3D cameras, mesh baking and UltraTex      | Outside the requested still-image/Chat execution scope.                                                                                          |
| Partner Breeze TTS and Seedream 5.5       | Remote paid execution has no newly enabled RF route.                                                                                             |

The exact bytes of native model/memory management, model patchers, prefetch,
pinned-memory and CLI policies remain unchanged. All six SeedVR2 node/model/VAE
source files match the preceding pin; the adaptive VAE slicing listed in the
cumulative release notes was already present. Existing SeedVR2 framing, matching
codec, alpha, cancellation and owned-resource contracts remain required.

## New native models and honest limits

| Native family or role               | Finite app disposition and concrete missing contract                                                                                                                                                                                                             |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MiniMax-H3-image (`MiniMaxH3Image`) | Native image-family inventory entry. Its dedicated conditioning, `[B,24,1,H/16,W/16]` single-stream latent, PDD sigma grid and image VAE need an explicit RF generation consumer. No ordinary image, default recipe or real-model support follows from this pin. |
| WeVisOne-VL                         | Distinct native detection/matting utility using a CLIP generator and decoder hidden states for masks, boxes and points. It requires a connected utility lifecycle and output consumer; it is not a generic Chat model or diffusion text encoder.                 |
| Qwen3-VL-2B                         | Distinct native text/vision component recognition. Keep its companion/model-type identity separate; no new Chat default or standalone diffusion conditioning route is implied.                                                                                   |

The finite registry records these source-visible additions, their unavailable
execution/artifact dispositions and the actual source pin. It does not invent
download identities, working defaults or qualification receipts. Existing PH3
video consumers are outside this still-image audit; native inheritance changes
do not qualify video execution.

The denominators remain distinct: 110 native model configurations (60 image),
32 folder roles, 39 logical `CLIPType` values, and 41 detected `TEModel` values.
The separate `native_dispositions(kind="te_models")` source inventory adds
`WEVISONE_2B`, `WEVISONE_4B` and `QWEN3VL_2B`. These are detected encoder model
types, not three new diffusion configurations or logical CLIP types. Existing
artifact tuples and model defaults retain their original review identities.

Cold WeVisOne/Qwen2B role classification executes only the pinned native
`TEModel`/`detect_te_model` header predicates against bounded shapes and caches
them by source signature. It imports neither Torch nor Comfy. WeVisOne encoders
remain in the existing `text_encoders` root and cannot fill ordinary or unknown
generation CLIP slots. No new extension or model folder is introduced.

MiniMaxH3Image's passive model identity remains visible. The separately registered
[MiniMax H3 Image adapter](native-minimax-image.md) now admits its captured
PDD graph with the required 32B encoder and image codec. Direct loading without
that plan still fails before weights. The adoption pin alone does not add this
graph or qualify either MiniMax-H3-image or WeVisOne hardware execution.

## Device and validation boundaries

CPU uses the existing isolated native startup and eager numeric operations.
CUDA/ROCm/XPU device selection, allocator and attention contracts are unchanged
by this delta. AIMDO remains a separate eligible NVIDIA/ROCm controller, with
legacy native loading on controller failure; XPU image support does not require
AIMDO. RF's existing Python/wheel admission and Torch API-floor checks remain.
Package presence never establishes a device's execution support.

Run the release contract and native CPU regressions with the pinned interpreter:

```text
python tools/check_cpu.py test_comfy045_release.py test_comfy_upstream_contract.py
```

The release tests execute the two upstream CombinedLinear transition cases on
tiny real quantized CPU tensors, native greedy penalty/dtype helpers and the
actual package-base consumer with resolution/install side effects replaced.
CPU imports forbid accelerator discovery/initialization. Finite catalogue tests
must also match the exact supported-model/CLIP denominators, source provenance
and new-family admission limits.

For prerequisite isolation, run the feature's declared replay:

```text
python tools/validate_feature_cherry_picks.py --feature native-comfy-045-adoption --tests --output tmp/comfy045-adoption-replay.json
```

That replay exports the exact pin from available Git objects into private
validation storage. It does not require changing an active vendor checkout or
installing packages. Source/CPU, UI, real-model output, GPU performance and
stability receipts remain separate and bound to their tested revisions. Earlier
`fca60fe` acceptance does not become `989d93d` hardware evidence.
