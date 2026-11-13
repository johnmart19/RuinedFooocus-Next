# Reusing ComfyUI platform support

Native contract reviewed on 2026-11-11. Embedded ComfyUI is pinned to official
[v0.45.0, `989d93d0d968e704c589ce0146bb18c77f6b50cc`](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.45.0).
The [release adoption contract](comfy045-adoption.md) records its exact delta,
package ownership and finite new-model dispositions. Bootstrap reconciles the
vendor checkout on normal startup; an active runtime must finish before its
source changes. Source/CPU checks do not qualify model output, GPU performance
or stability, and earlier pin-bound hardware receipts retain their original scope.

## Earlier reviewed native update

The preceding 2026-11-09 review used
`fca60fe70e5b84c5c6b058263653d15cb3401e44`, then the upstream `master` tip.

The [six-commit delta](https://github.com/Comfy-Org/ComfyUI/compare/1332fabfcc507aa7c548cd5152071ed494c207e8...fca60fe70e5b84c5c6b058263653d15cb3401e44)
contains two changes to the embedded compute path:

- `comfy.ops.CombinedLinear.prepare` declines FP8/unsupported combined execution,
  preserving resident member weights and their native split forward path. This
  addresses Qwen FP8 text-generation crashes without an RF implementation copy.
- `comfy.pinned_memory.pin_memory` returns for zero-size aligned allocations,
  avoiding empty host registration for `UNetSelfAttentionMultiply`.

`main.py`, CLI arguments, requirements, `sd.py`, model/memory management,
model patchers, allocator policy and attention dispatch are byte-identical across
this delta. The remaining changes concern the asset database, partner nodes and
video frame retention; video execution was not tested. Existing RF warm text
encoding, MOSS prefill, request cancellation, previews, patch cloning and failed
load recovery remain in place. The optional Vulkan adapter is separate work.

## Runtime reuse versus environment installation

Use native ComfyUI for device/dtype selection, attention, loading/offload,
sampling, model patching, allocation and cleanup where its contract fits. RF
embeds core rather than launching standalone `main.py`, so a small startup
adapter remains necessary. `ruinedfooocus/backends/comfy/comfy_compat.py` owns
that boundary; packaging policy lives in `ruinedfooocus/packaging/`.

| Contract             | Reuse and app responsibility                                                                                                                                                                                                     |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Device and precision | Use native `get_torch_device`, `unet_dtype`, text/VAE dtype and offload decisions. Do not infer accelerated arithmetic from a checkpoint's storage dtype.                                                                        |
| Memory and lifecycle | Use native `load_models_gpu` / `free_memory` and the pinned allocator setup. RF owns request cancellation, service ownership and preventing one job from unloading another service.                                              |
| Startup              | Translate RF's CLI/profile choices before importing runtime modules; reproduce only required standalone initialization and DynamicVRAM setup. Keep requested and effective state distinct.                                       |
| CPU                  | Native CPU execution is reusable. RF's `cpu_startup.py` suppresses early accelerator discovery during the serialized model-manager import, before service threads, and restores Torch APIs/module entries on success or failure. |
| Hardware masks       | Preserve native behavior with process-scoped, backend-specific identity. Do not let image selection rewrite unrelated Chat/ONNX device visibility.                                                                               |

The [pinned model manager](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy/model_management.py)
already owns much of this logic. RF's allocator adapter invokes the pinned
`cuda_malloc.py`; adding a second allocator implementation would duplicate it.
Compare [standalone startup](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/main.py)
at each pin update. Preserve app-owned failure/recovery contracts when removing
a workaround; similarity of function names is not sufficient evidence.

Native `args.cpu` is applied after XPU/MPS and optional NPU/MLU discovery, so
setting it alone does not meet RF's probe-free startup contract. The CPU import
boundary temporarily answers availability/count queries with false/zero and
refuses `torch_npu`/`torch_mlu` imports. It leaves CPU tensor execution intact.
The guarded imports include native model management, ops and attention: ops and
comfy-kitchen attention each perform additional unconditional availability
queries. These modules are then cached before normal feature imports.
Comfy-kitchen requires Torch Dynamo; initialize its callable-identity registry
before temporary wrappers so it retains original Torch functions. Preserve
decorator metadata, including MPS `__wrapped__`. Explicit CPU device setup also
leaves inherited masks unchanged. GPU startup does not use this import boundary.

Native numeric device selection applies only to its installed/requested CUDA or
HIP backend. Physical enumeration uses a bounded child process; it never clears
the live parent's masks. Inherited conflicting masks and changes after native
initialization fail before mutation. The status records inherited, requested and
effective values. Independent service children restore only masks introduced by
RF image startup and preserve the user's inherited restrictions.

The HIP boundary accounts for `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES` and
`GPU_DEVICE_ORDINAL`, following the [AMD isolation variable documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html#gpu-isolation-variables)
(reviewed 2026-11-09). Unrelated Vulkan, Intel and Ascend selections remain intact.
CPU regression fixtures cover these identities; they do not qualify multi-GPU
hardware or driver behavior.

## External component caller contract

Native architecture detection retains distinct names. In particular, Qwen Image
2.1 is not an alias for Qwen Image: its Qwen3-VL encoder and 64-channel RGBA VAE
use the distinct `clip_qwen_image21` and `vae_qwen_image21` roles. Component
inventory and preparation retain those roles instead of substituting the older
Qwen Image/Edit or Anima companions. `test_distinct_image_architectures.py`
covers the separate architecture and inventory contract.

`features/generation/image_pipeline.py` accepts
`load_base_model(..., component_overrides=None)`. The mapping contains resource
slot keys (`clip_l`, `clip_g`, `vae_sdxl`, etc.) and local filenames or null.
`get_clip_and_vae(unet, component_overrides)` resolves the initialized native
architecture, returns its `clip_keys`/`vae_key` and effective filenames, and
never mutates global Settings. The pure `backends/comfy/component_selection.py`
contract is shared by image component inventory and explicit downloads.

- A missing key retains its legacy profile-wide choice for migration.
- A present empty/null key means Automatic: use the family recipe, even when a
  legacy profile-wide override exists.
- Bundled CLIP/VAE slots retain native precedence. Explicit nonempty replacement
  choices are rejected with an actionable reset message. No patcher splicing or
  partial replacement of a bundled multi-encoder CLIP is supported.
- A choice for a slot outside the actual architecture is rejected. File presence
  is checked by the loader; format/shape compatibility remains native validation.
- Cache identity includes a copied explicit mapping, effective resource values,
  checkpoint signature and signatures for external files. Recursive loading of a
  partially bundled checkpoint preserves the mapping and bundled objects.

The generation worker and startup preload carry the component mapping through
this boundary. `models/component_profiles.py` captures checkpoint-scoped choices;
`models/effective_image_plan.py` binds their selected roles and file identities
before preparation. UI preferences and metadata replay use the same request
contract. `test_component_profiles.py`, `test_component_capture_boundaries.py`
and `test_effective_image_plan.py` cover those callers; they do not qualify real
model output or GPU behavior.

Focused checks use `venv/Scripts/python.exe tools/check_cpu.py` with
`test_comfy_upstream_contract.py`, `test_comfy_runtime.py`,
`test_native_allocator.py`, `test_cpu_startup.py`, `test_model_cache.py`,
`test_image_native_helpers.py` and `test_image_components.py`. The upstream
contract suite executes the checked-out changed functions, and separately imports
the complete native CPU model manager in a fresh process with accelerator spies.
It does not load a checkpoint or initialize a GPU.
The same isolated child executes upstream's own FP8 split-forward regression
for both float8 formats on CPU tensors. Those numerical checks also use CPU
isolation, because kitchen lazily queries hardware for custom-op dispatch even
with CPU inputs. They do not claim probe-free execution of every later native
custom operation outside the startup boundary.

## Published platform recipes are versioned inputs

| Platform       | Upstream route                                                                         | What RF must still qualify                                                                                                                                                 |
| -------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NVIDIA current | CUDA Torch, native attention/memory; current manual recipe cu130                       | Driver, architecture of every selected GPU, OS/Python ABI and complete wheel closure                                                                                       |
| NVIDIA legacy  | Separate cu126/Python 3.12 portable                                                    | Legacy architecture/model limits and mixed-generation selection; RF's cu124/Torch 2.6 exception is below core's documented Torch 2.7 minimum and needs explicit acceptance |
| AMD Windows    | ROCm Torch with target device extras; no external HIP SDK required for wheel execution | Core's 10.0/Torch 2.13 recipe differs from RF's 10.1/2.14 and Standalone's 7.14/2.12; never combine companion packages across them                                         |
| AMD Linux/WSL  | ROCm native runtime                                                                    | Released OS/GFX/driver/runtime contract; WSL guest validation remains separate                                                                                             |
| Intel GPU      | Native `torch.xpu` Arc/Core Ultra route                                                | Actual generation/driver and wheel support; Intel vendor ID alone is insufficient. Do not add IPEX without a demonstrated need                                             |
| NPU            | Core `torch_npu` refers to Huawei Ascend                                               | It is not generic AMD XDNA or Intel NPU support; those require separate engines and compatible models                                                                      |
| CPU            | Native Torch CPU                                                                       | Explicit user selection or last compatible fallback; preserve other independent GPU services                                                                               |
| torch-directml | Legacy optional core branch                                                            | Current core discourages it. RF's old Torch 2.4.1 package is an isolated compatibility exception, not permission to downgrade modern environments                          |

References: [manual installation](https://docs.comfy.org/installation/manual_install),
[portable environments](https://docs.comfy.org/installation/comfyui_portable_windows)
and [system requirements](https://docs.comfy.org/installation/system_requirements).
Native XPU execution does not imply DynamicVRAM eligibility. The current
controller has separate NVIDIA/AMD and Torch requirements.

## Why not call another installer inside RF?

The old `Comfy-Org/desktop` is archived. Current
[Comfy-Desktop `c8890334`](https://github.com/Comfy-Org/Comfy-Desktop/tree/c8890334e952d1774c2bff2e9303602a7a813cfb)
uses [Standalone environments `fe00ec4c`](https://github.com/Comfy-Org/ComfyUI-Standalone-Environments/tree/fe00ec4cd3fa42d1566096da457d3d399b141a40),
including bundled Python and vendor-specific dependencies. Its
[Torch stack manifest](https://github.com/Comfy-Org/ComfyUI-Standalone-Environments/blob/fe00ec4cd3fa42d1566096da457d3d399b141a40/torch-index-stacks.json)
is useful comparative input, not an ABI guarantee for RF. Desktop compatibility
handling annotates some driver/architecture mismatches rather than rejecting
installation; RF still needs its stricter admission rules.

[comfy-cli `ad12ce37`](https://github.com/Comfy-Org/comfy-cli/tree/ad12ce37120c23412ae64be6202c41bbdf7b232c)
still selects `torch-directml` on AMD Windows in `command/install.py`. Its
`resolve_cuda_wheel` uses driver version without RF's all-selected-GPU and wheel
intersection. Blindly invoking it would regress the intended policy.

Maintain **one RF package-plan interface** consumed by bootstrap, Settings and
future portable packaging. It takes pinned publisher recipes and verified
artifact metadata as inputs. It also covers Chat and ONNX, which ComfyUI's
installer does not own. Reuse upstream implementations where possible; do not
maintain parallel selectors inside each UI or copy whole installers.

## Planned installer base and RF overlay

Match the pinned ComfyUI installation contract as closely as possible, with
RuinedFooocus-only additions kept separately identifiable:

1. **ComfyUI base:** the selected core revision's requirements plus one reviewed,
   versioned publisher Torch recipe for the target. Preserve upstream inputs and
   their hashes; resolve a qualified lock per platform/Python/device target.
   Prefer the core/manual recipe and record any vendor or Standalone deviation.
2. **RF overlay:** bootstrap tooling and RF application/API dependencies. Audit
   `pip/modules.txt` into shared constraints, actual additions and explicit
   compatibility exceptions. Shared packages resolve once. An RF pin must satisfy
   the base or produce a visible conflict; it cannot silently replace Torch or
   another base dependency. Exclude upstream frontend/Manager packages only after
   proving they have no embedded startup/import consumer.
3. **Optional runtimes:** retain isolated Chat/ONNX providers where their native
   packages or Python bindings conflict. Separate requirement files do not by
   themselves require duplicating the entire main Python environment.

The proposed pure interface is
`plan_packages(target, capability_decision, pinned_base, rf_overlay, installed_manifest, mode) -> PackagePlan`.
The exact symbol may follow existing repository conventions; its contract is
shared by bootstrap, Settings and future portable packaging. The result records
interpreter/ABI/OS, selected devices, core/base/recipe identities, requirements
digests, exact artifacts/indexes/hashes, combined transitive closure, environment,
exceptions/conflict chains and restart requirements. Planning initializes no
compute and does not install packages.

Resolve both layers before mutation. The executor consumes the frozen plan and
digest, stages the result under an environment lock, verifies that the overlay
preserved base constraints, then publishes atomically or reports restart/rollback.
Do not resolve a different set of packages between the UI preview and execution.
Offline mode requires matching cached identities. A base's documented API floor
can be stricter than an unbounded dependency line: qualify or refuse legacy
Torch combinations against the actual core, not just pip's metadata.

Worker 01 owns capability/admission data; 02 owns package planning/execution and
installer/bootstrap wiring; 03 proposes ComfyUI pin/base updates and proves native
caller compatibility; 09 audits dependency provenance/notices and distribution;
07 verifies all consumers show the same plan. Copying upstream code/data requires
source/revision/license/notices, distinct from simply consuming a package index.

## Verification required for the reuse change

### Bounded app acceptance on 2026-11-09

At RF `820d8dc`, the private owner UI loaded the exact Qwen3-1.7B Q8 artifact
(`061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`)
through managed native llama.cpp `b12110` on Vulkan0, AMD Radeon RX 7900 XTX.
While the separately selected LFM2.5 Q8 candidate downloaded through the real
preparation service, active Qwen replied in the UI at observed 6% progress.
Production **Cancel download** showed cancellation and retained active B. The
candidate final file was absent; owned server/native children exited, with exact
PID/creation checks and unchanged source/vendor pins. The native log recorded
29/29 offload, context 2048 and one slot. Image device selection was CPU with preload off;
no image/video or extra HTTP Chat request ran. This single-session scheduling
proof is separate from model quality, cross-session sharing, xllamacpp and
physical 3 GB/6 GB qualification. The user deliberately deferred changing default.

The same environment's base gap was repaired additively from all maintained
Comfy/RF/bootstrap/API requirements, constrained to its 152 existing versions.
The reviewed 18-wheel closure (622,610,723 bytes, including static workflow media)
was hash/size/tag verified and installed under the environment lock without
replacing those versions. All 83 applicable root requirements and postinstall
`pip check` passed; narrow CPU helper imports passed. No graphics/model runtime
was imported by repair validation. This was a reviewed additive repair, not a
claim that an additions-only report was a full new environment PackagePlan.
The result does not exempt unused base packages or certify GPU kernels.

Raw logs remain in local cache. Receipts are
`cache/plan-execution-20261109/chat-download-concurrency-final.json` and
`cache/plan-execution-20261109/missing-comfy-base-20261109/repair-20261109T192601Z-5d5249b7/result.json`;
the latter's SHA-256 is
`e07cda7cd6fdf07a7f7c832cff144192acc698d5fc369197fbdf21115bea507d`.

### Remaining reuse checks

- Focused CPU tests for pre-import setup, CLI/profile precedence, CPU probe
  suppression, GPU identity and recipe selection. `_read_installed_torch` can
  perform GPU matrix work; classify it as active validation, not inventory.
- Pin/caller checks for native API changes, allocator/attention behavior,
  DynamicVRAM eligibility, failed load/cancel/unload and exact prompt geometry.
- Fixture matrix for current/legacy CUDA, ROCm, XPU, CPU, multiple vendors and
  incompatible Python/transitive wheels. Installer/UI decisions must agree.
- Combined-layer conflicts: Gradio/Pillow, MCP/Pydantic, old DirectML with
  NumPy/SciPy/OpenCV, old Torch with Transformers/Kornia, and matching
  Torch/vision/audio/ROCm extras. Include free-threaded ABI, glibc, architecture,
  wrong index, offline identity, parallel install locks and failed-stage rollback.
  No missing base requirements, unexplained override or `--no-deps` escape hatch.
- Real device checks only for affected qualified routes. No CPU/mock evidence
  certifies hardware speed or stability, and no source NPU branch certifies an
  unavailable NPU. See the [AMD controls](amd-controls.md) before tuning flags.
