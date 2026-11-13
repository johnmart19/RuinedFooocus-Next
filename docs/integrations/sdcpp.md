# Optional stable-diffusion.cpp image backend

Source contract researched **2026-11-09**. RF owns the isolated adapter in
`ruinedfooocus/backends/sdcpp/`; upstream owns model loading, encoding, sampling,
VAE execution and backend memory. There is no copied ComfyUI sampler or device
manager. The adapter, nonvideo pipeline registry routing and application caller
are implemented. Shared worker admission/metadata hooks, Settings controls and
actual device/model acceptance remain separate integration work. No
installed runtime or real model execution is certified by these checks.

## Pin, artifacts and licenses

The source pin is
[`fc350be2ceeeb859bcc201ac435591ed7d46821d`](https://github.com/leejet/stable-diffusion.cpp/tree/fc350be2ceeeb859bcc201ac435591ed7d46821d),
release
[`master-1068-fc350be`](https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-1068-fc350be),
published 2026-11-08. The publisher's
[release API](https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/tags/master-1068-fc350be)
supplied these frozen compressed artifact identities. They are acquisition
inputs, not binary inspection or ABI/device qualification evidence.

| Platform / backend         | Filename                                                     |    Bytes | SHA256                                                             |
| -------------------------- | ------------------------------------------------------------ | -------: | ------------------------------------------------------------------ |
| Windows x86-64 CPU         | `sd-master-fc350be-bin-win-cpu-x64.zip`                      | 18225866 | `b3da78a4e222caf8b447144b1ec4f51102b5da902a45ab2c288968729962b1b9` |
| Windows x86-64 Vulkan      | `sd-master-fc350be-bin-win-vulkan-x64.zip`                   | 29995988 | `b229d9a21858ba91975d061ccfe37fa690254a423bcf77b39e72a1d8435a8b4a` |
| Ubuntu 26.04 x86-64 CPU    | `sd-master-fc350be-bin-Linux-Ubuntu-26.04-x86_64.zip`        | 29022149 | `11283f1358e0ac7f17a27f473fd51a53de338f9849b708f701516f5c64837ff0` |
| Ubuntu 26.04 x86-64 Vulkan | `sd-master-fc350be-bin-Linux-Ubuntu-26.04-x86_64-vulkan.zip` | 41894957 | `bd46e4dae1e145b91321fd7452d7e8e0765c34a95a336c392dd180f455ba8f17` |

Linux's published Ubuntu 26.04 build is not a generic glibc or WSL compatibility
promise. Parent admission must verify OS/loader, CPU instruction support and the
full runtime payload before activation. WSL needs guest-specific Vulkan/loader
acceptance. No ARM, macOS, CUDA, HIP, Metal, SYCL or OpenCL artifact is admitted
by this initial adapter. Python is an orchestrator; the executable is isolated
from the app's Torch/Python ABI and does not require replacing Torch.

Upstream uses the
[MIT license](https://github.com/leejet/stable-diffusion.cpp/blob/fc350be2ceeeb859bcc201ac435591ed7d46821d/LICENSE).
The pinned
[publisher build workflow](https://github.com/leejet/stable-diffusion.cpp/blob/fc350be2ceeeb859bcc201ac435591ed7d46821d/.github/workflows/build.yml)
packages upstream and GGML notices as `stable-diffusion.cpp.txt` and `ggml.txt`.
Preparation preserves and requires both; distributing a bundle must retain its
notices and dependency licenses. Model licenses remain independent. One primary
SDXL Base artifact is
[`sd_xl_base_1.0.safetensors`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors),
SHA256 `31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b`,
under the publisher's CreativeML Open RAIL++-M license. It is a reference identity,
not a required download or a universal qualification for SDXL derivatives.

## Capability and request contract

This is a deliberately bounded source-supported matrix. Actual acceptance is a
separate receipt bound to runtime payload, selected device, exact model SHA256
and operation. No receipts ship with the adapter.

| Dimension              | Initial contract                                                                                                      |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Architecture           | Standard SDXL 1.0 base, classified by the existing native caller                                                      |
| Variant                | Known standard base; unknown, distilled/Turbo, refiner and inpainting variants refused                                |
| Format                 | Full monolithic safetensors with bundled diffusion, CLIP-L, CLIP-G and VAE                                            |
| Storage / quantization | F16/F32 tensors only; no quantization, GGUF, BF16 or FP8                                                              |
| Operations             | Single-image txt2img or img2img                                                                                       |
| Backend                | Separate CPU artifact or an exact enumerated `VulkanN` device                                                         |
| Sampling               | Native Euler; RF `normal` maps to native `discrete`; native CPU RNG explicitly recorded                               |
| Geometry               | Exact multiples of 64, 64 through 8192 per dimension; this input bound is not a memory/performance guarantee          |
| Prompt                 | Fully expanded plain literal positive/negative text passed without rewriting                                          |
| img2img                | Prepared RGB image matching the output canvas; applied orientation; `0 < denoise <= 1` maps unchanged to `--strength` |

Masks, LoRAs, embeddings, ControlNet, external components, refiner, explicit CLIP
skip, batching, hires/upscale, IP adapters, video, previews and all unrecognized
conditioning/processing features are refused. The caller must enumerate other
requested features in `extra_features`. Parent routing must not omit a requested
feature to obtain admission. Prompt weighting, brackets, angle-bracket directives,
alternation, `BREAK`, `AND` and embedding references are refused because native
prompt semantics differ. Styles/wildcards must already be explicitly expanded
or be reported as unsupported. Matching sampler names/seeds does not imply
ComfyUI-identical pixels or denoise schedules.

Each prompt is bounded to 8192 characters to keep native command-line submission
bounded. Longer text is refused without truncation.

The app's `clip_skip` values are positive: `1` keeps the model's normal text
encoder layer and higher values request a layer override. The caller maps only
app `1` (or an absent choice) to the CLI's `-1` default sentinel. Higher choices
remain unsupported by this adapter; negative, fractional and nonnumeric app
choices cannot acquire optional admission. This conversion leaves the app's PNG
metadata and Library replay value unchanged. The nested engine details retain the
CLI value separately; ComfyUI's negative internal layer indices are not app
metadata values.

`ModelSpec` is immutable. `inspect_model(path, architecture=..., variant=...)`
uses RF's bounded safetensors-header reader and file signatures without loading
weights, hashing the full file or importing Torch. Architecture classification
comes from the existing caller. A shape classifier cannot prove whether an SDXL
derivative is distilled; the variant defaults to `unknown`. The caller must
supply known provenance and preserve an unknown classification. Header prefix
presence establishes bundled layout only; native loading remains final format/
shape validation. `with_model_hash` attaches an already verified acquisition or
explicit acceptance hash; it does not verify an arbitrary asserted hash.

`assess(request, candidate)` returns refusal reasons without I/O. Normal admission
requires the exact model hash and operation in the candidate's acceptance
receipt. `choose_runtime(..., native_gpu_compatible=True)` retains a usable
compatible native GPU first;
otherwise it considers qualified compatible Vulkan before CPU. A usable compatible
Vulkan candidate awaiting acceptance stays outside Auto. The app falls back to
the native CPU path with a terminal warning when it has no qualified GPU route.
Explicit CPU selects only CPU.
Native failure, crash, missing output and device-verification failure never
trigger an automatic CPU retry.

## Preparation, activation and execution

`Artifact.preparation_request(destination)` produces a frozen
`core.preparation.PreparationRequest` containing publisher URL, size and SHA256.
Only an explicit authorized acquisition action calls `prepare_runtime` with
`check_cancelled`. Parent policy applies offline/download and platform admission
before that call. It reuses `preparation_service` single-flight transfer and read
leases, plus `packaging.install_lock.installation_lock` for cross-process
publication. Existing immutable runtimes use `open_runtime` without downloads.
No status getter, import, capability evaluation or resolver acquires files.

Extraction accepts the publisher's flat ZIP layout, refuses traversal, duplicate
names, symlinks and oversized archives, and verifies the complete staged payload
and notices before an atomic rename. The manifest records the source pin,
archive hash and every extracted file's hash. `open_runtime` verifies that
manifest at explicit activation and captures file identities. Invalid caches
remain intact for diagnosis; live bundles are never overwritten. Competing
publication returns a busy error after a bounded lock attempt. Cancellation
detaches only the caller's transfer subscription and removes its private stage.

`verify_device` is an **explicit active operation**, never passive inventory.
Vulkan calls native `sd-cli --list-devices` and matches the exact tab-separated
backend name. CPU uses its CPU-only bundle's `--help`, avoiding Vulkan device
enumeration. The result captures native description and inherited Vulkan
visibility/driver environment. Verification clears previous acceptance receipts;
the parent can restore one only after matching its complete device/payload/model
identity. A native `VulkanN` index/description is not a PCI/LUID identity, and
duplicate physical cards need parent-side mapping. Enumeration certifies neither
model execution nor stability.

`generate(request, candidate, compute_lease=..., check_cancelled=...)` requires
the parent's shared compute admission context. Parent resource ownership must
account for Chat, ONNX and native images before admission; a separate subprocess
or the adapter's serialization lock is not proof of concurrent VRAM safety.
File read leases precede compute admission. All optional jobs are serialized;
cooperative cancellation remains active while waiting for file/adapter ownership.
The native process inherits service-independent visibility masks without mutating
the app environment. Dynamic GGML backend lookup is scoped to the verified bundle.
A changed Vulkan visibility/driver environment requires verification again.

The CLI uses explicit compute and parameter placement, disables native auto-fit,
and disables optional flash-attention/convolution selection probes. It leaves
thread count and checkpoint storage unchanged. It runs in a private directory,
passes arguments as a list without a shell, and reads only its prepared full
checkpoint and private image. Unrelated LoRA/embedding directories are not exposed.
Native arguments come from the pinned
[CLI argument implementation](https://github.com/leejet/stable-diffusion.cpp/blob/fc350be2ceeeb859bcc201ac435591ed7d46821d/examples/common/common.cpp),
[backend contract](https://github.com/leejet/stable-diffusion.cpp/blob/fc350be2ceeeb859bcc201ac435591ed7d46821d/docs/backend.md),
[SD examples](https://github.com/leejet/stable-diffusion.cpp/blob/fc350be2ceeeb859bcc201ac435591ed7d46821d/docs/sd.md)
and [image preprocessing contract](https://github.com/leejet/stable-diffusion.cpp/blob/fc350be2ceeeb859bcc201ac435591ed7d46821d/docs/image_preprocessing.md).

`GenerationResult` owns a copied decoded image, RF request/engine metadata and a
bounded diagnostic tail. Native PNG metadata is preserved alongside original
prompts, requested/native scheduler, seed, exact canvas, denoise, input/model
identities and runtime/device identity. The parent retains its normal final
Library/metadata writer. There is no intermediate progress/preview contract in
this version. CPU RNG selects upstream behavior; no RF sampler is duplicated.

Timeout, cancellation, load failure and invalid output reap only the owned child
through `core.child_process.start_process`, bounded terminate/kill/wait and
Windows Job cleanup. Private inputs/outputs are removed and compute/file leases
released. An invalid output canvas is an error; no resizing conceals it. The
adapter does not unload or alter another engine's resident model.

## Validation and remaining acceptance

### Application caller and Settings actions

`features/generation/pipelines.py` invokes the optional selector only from its
ordinary nonvideo image branch. Chat, automation commands, standalone image tools
and video routing retain their existing owners. The caller is
`features/generation/sdcpp_pipeline.py`; it exposes the generic worker's
`load_base_model`, `load_loras` and `process` shape with `pipeline_type` containing
`sdxl` and `sdcpp`. The former is compatibility with the existing image writer,
not a claim that ComfyUI-specific sampler or CLIP semantics apply. It validates
the queued model identity, refuses LoRAs/replacement components, passes only the
bounded backend request and returns copied NumPy image arrays to that writer.

Preferences owned by Settings are `image_engine` (`auto`, `native`,
`sdcpp-vulkan`, `sdcpp-cpu`) and `sdcpp_vulkan_device` (an exact `VulkanN`).
`native` explicitly retains ComfyUI. Global CLI CPU, saved global CPU, explicit
`image_backend: cpu` and legacy `image_device: CPU` prevent Vulkan admission.
Automatic native CPU fallback does not become an explicit CPU preference.
`auto` retains an initialized compatible native GPU first, seeks a qualified
Vulkan runtime next, and retains ComfyUI CPU only when no compatible optional GPU
route exists. Unsupported requested features stay on native routing in Auto;
explicit optional choices report the refusal. A usable compatible Vulkan route
awaiting qualification reports the missing acceptance instead of selecting CPU.

`capture_image_request(data, models, values, paths=None, launch_args=...,
native_gpu_compatible=..., resolution_settings=...)` runs at submission. It copies
uploads and nested user data, captures effective control settings, engine/device
preferences and the current checkpoint file identity, and freezes named canvas
dimensions. The parent supplies its trusted native GPU decision when available;
the default only reads the already initialized ComfyUI manager state and calls
no device query. API job objects retain identity. API-supplied prepared/hook
objects are removed. The snapshot retains original feature intent while the
worker separately expands prompts and resolves performance settings/random seeds.
Final values and captured conditioning are both checked; metadata processing
cannot erase an inpainting or replacement-component request to obtain admission.

Only `_rf_local_sha256` accompanied by the exact current `_rf_file_signature`
is accepted from the cached registry. A remote release's first hash, a renamed
filename, or the broad SDXL family label cannot prove a standard base variant.
The verified publisher Base hash above is known provenance; other standard base
models acquire that identity through an explicit acceptance record. Unknown
derivatives remain native-only in Auto. An explicit optional choice without a
verified local model hash requests the normal identity refresh. No model weight
hashing is added to browsing or the compute queue.

`needs_preparation(data)` identifies queued optional work.
`prepare_image_request(data, check_cancelled=...)` belongs in
`app.preparation_admission`, before the serial compute queue. It revalidates the
captured file, inspects the bounded header, checks an existing explicit activation
and captures init-image geometry/identity. It installs nothing and probes no
device. Missing installed-runtime activation requests an explicit Settings Verify
action. Init images must already be RGB, correctly oriented and match the exact
selected canvas; Auto retains native when those conditions fail. Input/model
replacement after preparation is refused. Images are copied to a private file
for the backend, and both caller and backend temporary files are cleaned up.

`backends/sdcpp/activation.py` supplies Settings-facing functions:

- `status(backend)` returns passive artifact/platform/cache/session facts, without
  hashing a bundle or initializing compute.
- `install(backend, check_cancelled=..., offline=..., skip_dependency_install=...,
progress=...)` is an explicit acquisition action. An offline action can verify
  and reuse an existing immutable bundle; it cannot acquire a missing bundle.
- `verify(backend, device, check_cancelled=..., identity=DeviceIdentity(...))`
  explicitly opens/verifies the bundle and runs the bounded native device check.
  It never acquires files or grants workload acceptance.
- `record_acceptance(activation, model, operation, evidence=...)` records the
  parent's separately performed actual acceptance. Installing/enumerating never
  invokes it. A string evidence reference is a record, not an automatic proof
  that the referenced validation occurred.
- `verify_model(path, architecture=..., check_cancelled=..., variant="unknown")`
  explicitly verifies a selected local model's header and SHA256 using the
  existing file-signature hash cache. It works offline in the preparation
  executor and populates signature-bound `model-identities.json`, which capture
  can read when the generic registry has no current local hash. It does not grant
  base provenance or execution qualification. Hashing uses the existing cache
  contract; cancellation is checked before and after that synchronous hash.

All runtime and receipt paths are below `data_path("cache", "sdcpp")`; portable
data redirection already applies. There is no independent runtime-root preference.
The atomic, installer-locked `acceptance.json` store matches source revision,
archive hash, full payload-manifest digest, backend name/description, device
environment, physical adapter identity, driver version/package identity, guest or
native location, model hash/architecture/variant/storage/quantization and operation.
`DeviceIdentity(adapter_id, driver_version, driver_identity, location, stable)`
must contain complete stable evidence for automatic qualification. Unknown or
ambiguous physical/driver evidence remains pending manual qualification.

A persisted acceptance record does not restore a Vulkan ordinal after restart.
`verify` establishes the current process's physical/driver binding; `current`
checks that binding's runtime file and visibility identities. Explicit
reverification clears candidate acceptance; `qualified` may then restore only a
matching receipt. Driver/device changes require the parent's new inventory and
explicit verification; the adapter does not run periodic probes in status reads.

The shared worker uses the following hooks:

1. After component submission capture, call `capture_image_request` with the
   shared resolution settings, launch arguments and native GPU eligibility.
2. Before compute admission, schedule `prepare_image_request` in
   `preparation_admission` whenever `needs_preparation` is true, including local
   ready runtimes that still require header/receipt validation.
3. At execution, provide `_sdcpp_execution = ExecutionHooks(check_cancelled,
compute_lease_factory)`. The factory receives `(request, runtime_candidate)`
   and returns 06's shared resource ownership context. The callback must retain
   UI/API cancellation. No production `nullcontext` bypass is supplied.
4. Merge the returned `_sdcpp_metadata` into the final PNG metadata with the
   effective `image_engine`, preserving engine identity on Library replay.

`async_worker.image_stage` supplies step 3 from the actual owning job and shared
compute admission. UI/API preparation runs before compute, API policy capture is
synchronous at submission, and final PNG metadata includes the selected engine.
Chat tools may rebind their final reference/controls with `retarget_image_request`
while retaining captured checkpoint, engine and device policy. This performs no
runtime acquisition or device probe. Direct callers without trusted ownership
still receive the pending-admission error before starting a native process.
Application wiring and CPU tests do not create an operator/model acceptance receipt.

### CPU checks and actual acceptance

Focused CPU checks:

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_sdcpp.py test_sdcpp_caller.py test_preparation_service.py
```

These tests use tiny headers, local fixture archives and inert subprocess mocks.
They cover source matrix/refusals, complete feature gating, literal CLI mapping,
native-first/Vulkan-before-CPU routing, exact RGB geometry/strength, immutable
file identities, device-enumeration separation, receipt invalidation, cancellation,
timeout, failed output, child cleanup, artifact integrity and atomic publication.
Caller checks additionally cover queued snapshot isolation, cached identity
binding, conservative variant provenance, legacy/global CPU precedence,
native-GPU-first routing, qualified Vulkan selection, pending driver/acceptance
refusal, generic worker compatibility, mandatory compute hooks, receipt matching,
fresh-process verification and unchanged video routing. The direct image/Chat
boundary combination exposed an existing Python import fixture issue: a
`sys.modules` replacement alone does not replace an already-bound parent package
attribute. The PowerUp family and CustomVision fixtures now replace both before
importing dependent modules, preserving their original behavior assertions.
They do not execute a native binary, probe a GPU, install/download a runtime,
download/read a real model or generate an image.

The parent schedules bounded actual acceptance separately. The internal
`allow_unqualified=True` flag bypasses only workload receipts for that explicit
validation action; pin, payload, source capability and device checks still apply.
Record exact source/runtime/dependency/model/input/output hashes, physical adapter
mapping, operation, geometry/strength, native placement evidence, cancellation,
cleanup and output inspection before granting a normal receipt. Record speed,
resolution and checkpoint results for each validated Vulkan workload. CPU
acceptance remains separate from Vulkan acceptance.

All native video engines/dependencies and their routing remain outside this
adapter. The upstream release's video/model list creates no RF video capability
or replacement dependency plan; no video was executed during this work.
