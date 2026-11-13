# ONNX provider integration

Audit scope and source identities are in the [integration index](README.md).
Platform contracts below distinguish implemented adapters and CPU fixtures from
hardware/model qualification. No new AMD route is qualified by this change.

The 2026-11-13 installer metadata review selects isolated CUDA 13 ORT 1.32.0
and the DirectML wheel `onnxruntime-windowsml` 1.30.1.202610302006, both with
published CPython 3.11–3.14 Windows x64 wheels; CUDA also has Linux x64 wheels.
The [official CUDA provider contract](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
and [published wheel metadata](https://pypi.org/pypi/onnxruntime-gpu/json) remain
separate from executed-model qualification. CUDA 12 retains its feed's newest
1.29.0 build. OpenVINO retains the latest 1.24.1 ORT wrapper and the
[documented matching 2025.4.1 runtime](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#requirements),
rather than replacing that ABI partner with the newest standalone OpenVINO.
Upstream recommends its newer standalone plugin; that is a different registration
and device-qualification contract, rather than a version-only replacement.
The Windows ML catalogue remains on the published 2.3.0
Python projection and its exact older ORT dependency; the separate DirectML
wheel update does not replace that catalogue environment or qualify AMDGPU.

## Current app boundary

`ruinedfooocus/backends/onnx/` owns optional environment installation
(`onnx_installer.py`), worker sessions (`onnx_worker.py`), Windows ML acquisition
and device records (`windows_ml.py`), client/recovery (`onnx_runtime.py`) and
model preprocessing contracts (`model_contracts.py`). The worker currently uses
the project's base interpreter. A separate package directory does not change
that interpreter's Python ABI. The audited Windows venv is Python 3.12.10.

The main environment retains CPU ONNX Runtime. Optional runtimes are isolated;
they must not overwrite its `onnxruntime` namespace or another service's DLLs.
New native Linux acquisition names the MIGraphX plugin recipe described below;
it refuses incomplete target/artifact evidence before creating cache paths.
Existing verified `onnxruntime-migraphx` 1.27.1 folders remain addressable, without
bumping or relabeling that integrated runtime. Provider enumeration and an
installation canary do not establish model qualification.

`model_contracts.py` retains rembg's native model class, download contract,
normalization and prediction methods. Its strict construction adapter is gated
to audited rembg 2.0.87 and classes inheriting native `BaseSession.__init__`.
That constructor initializes `model_name` and `inner_session` but does not forward
ORT's `enable_fallback`; a small subclass supplies `enable_fallback=False` and
calls `disable_fallback()`, without changing shared rembg/ORT modules. Native
factory model selection, OMP thread settings, explicit providers/options and
the existing Bria 1024-square symbolic overrides are retained. Other versions
keep native CPU construction; accelerated construction requires a fresh audit.

Initialization status explicitly distinguishes unverified execution. Where
available, native ORT `get_provider_graph_assignment_info()` records optimized
graph assignment with `session.record_ep_graph_assignment_info=1`, rejecting
zero accelerator assignment before inference. The first inference profile
then counts executed provider node events and CPU partitions;
it does not claim every graph node or device timing was observed. Missing
accelerator events fail visibly and invoke application-managed CPU recovery.
Normal CPU graph partitions are permitted, separately from disabled Python
constructor/run retry. Recovery states initialization versus inference failure,
keeps the reason and preserves model-specific Python options. Cancellation is
propagated rather than converted to CPU recovery. Session reuse is invalidated
when saved background-provider requests or runtime installation records change.

The isolated worker reports initialization phase starts and monotonic elapsed
seconds through its existing JSON channel: runtime loading, provider/device
selection, native model-session creation, identity/contract checks and readiness
publication. The client displays these through the existing runtime status;
progress is not readiness or executed-node evidence. Progress messages do not
reset the fixed deadline. Initialization still allows 180 seconds; an inference
response still allows 120 seconds. Timeout status names which wait expired and,
for initialization, the last reported phase and when it began. Payloads are
bounded and queued progress is capped; cancellation remains checked while
waiting and closes the owned worker. Older workers emitting only readiness
remain accepted. No timeout setting, default increase or automatic retry is
introduced by these diagnostics.

Two bounded original-Bria attempts on the RX 7900 XTX verified native CPU cold
and warm inference, but the named Windows ML MIGraphX worker did not publish
readiness within 180 seconds. The diagnostic attempt at source
`f711a0125824158cf0c10d12712eb56a180d2c9e` reported its last phase as
**creating native model session**, begun at 0.312 seconds. It never reached
model/contract checking or readiness, so no GPU inference request or executed
assignment was observed. The native construction phase may include graph
optimization/compilation; its timeout does not distinguish slow work from a hang
and does not prove a device fault or universal model incompatibility.

That attempt's CPU cold/warm times were 19.851/11.455 seconds, including native
rembg 2.0.87 preprocessing, inference, postprocessing and PNG writing. Its first
CPU inference recorded 4939 executed CPU nodes. Both 928 × 1664 RGBA results were
byte-identical to each other and to the earlier attempt; the unforced transparent
input skip also passed. GPU initialization then exhausted the unchanged
180-second limit; the harness refused application CPU recovery without running
it. The owned worker and supervisor/Windows Job closed after 212.516 seconds.
Original model/input, provider library and frozen source identities remained
unchanged. These are one-image observations, not a performance benchmark.

| Identity                                     | Recorded value                                                     |
| -------------------------------------------- | ------------------------------------------------------------------ |
| Runtime                                      | Python 3.12.10/cp312; main CPU ORT 1.31.1; rembg 2.0.87            |
| Original Bria SHA256                         | `5b486f08200f513f460da46dd701db5fbb47d79b4be4b708a19444bcd4e79958` |
| Original input SHA256                        | `140b25f444a9d8d8ef0317fb53c686a1a6997532e8bf7c7d6ca14bf5dec67c1d` |
| Main ORT DLL SHA256                          | `e45f61e50d8650885452be538a00bb14a715e46a1bc3691000ce947e89ed0306` |
| Named MIGraphX provider DLL SHA256           | `cb7496b3ab8f9daf2948b6030db5ec795596072ffcb25d3c55a28e4c0df7f8ab` |
| Frozen source/runtime/package receipt SHA256 | `ccca8fcb06cc994c6a73424b7a04b1e9f1e44af7f770c66ec42bd2e78145e5dc` |

Local catalogue certification and device discovery succeeded, but GPU readiness
never supplied a completed runtime/device/driver qualification contract. CPU/GPU
alpha, visual and speed gates remain unverified. No qualification record was
written; vendor Auto remains ineligible for this unqualified contract. No derived
graph, timeout increase or provider substitution was used. The first attempt and
diagnostic attempt are retained under
`cache/plan-execution-20261109/bria-final-original-01` and
`bria-final-diagnostic-02`; private receipts are excluded from source archives.
Further attempts use a bounded coordinated GPU window and retain failed evidence.

The first qualification workload is the unchanged rembg Bria graph and native
preprocessing. Other ONNX tools require their own model contracts. ComfyUI
diffusion and GGUF Chat remain separate engines: an EP does not accept arbitrary
checkpoint files. [ORT GenAI](https://onnxruntime.ai/docs/genai/howto/install.html)
has separate model, tokenizer, cache and runtime requirements.

## Platform and package contracts

| Route                            | Published contract                                                                                                       | RF decision / acceptance still needed                                                                                        |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| Linux AMD, ROCm 10.1             | ORT 1.29.0, MIGraphX 2.18, `onnxruntime-ep-migraphx==1.0.0+rocm10.1.0`; pip `migraphx-libs` and explicit EP registration | Implemented recipe/registration gates; artifact closure and real-device/model qualification still required                   |
| Windows ML AMDGPU                | Windows 11 24H2/build 26100+, RDNA3+, driver 25.10.13.09+, Windows ML 2.5.96+                                            | RX 7900 XTX matches the GPU class; local catalogue eligibility, compatible bridge, assignment and Bria remain unqualified    |
| Windows ML MIGraphX              | Microsoft is retiring this EP and excludes GenAI today                                                                   | Preserve existing explicit choices/recovery while qualifying a replacement; this retirement does not apply to Linux MIGraphX |
| Direct HIP-EP                    | A separate Windows/Linux plugin with release-specific SDK/GFX and Python-binding constraints                             | Experimental opt-in candidate, not an automatic Radeon fallback                                                              |
| WSL AMD                          | ROCm 10.1 WSL support is technical preview                                                                               | Guest-specific eligibility and workload evidence; native Linux and Windows results cannot certify it                         |
| CUDA / OpenVINO / CPU / ONNX DML | Independent release trains and model/device contracts                                                                    | Preserve their isolation and regression coverage; AMD package availability must not alter their selection                    |

AMD's [Linux guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/onnxruntime.html)
validates Python **3.12** and lists gfx950, gfx942, gfx90a, gfx1200/1201,
gfx1100/1101/1102 and gfx1150/1151/1152/1153. Do not expand this list to every
TheRock target. The [artifact index](https://stable.repo.amd.com/rocm/onnxruntime/whl-next/onnxruntime-ep-migraphx/)
also lists cp311/cp313/cp314 manylinux_2_28 x86_64 wheels; listed wheels are not
the same as validated interpreters. The pinned guide source is
[`a0f9d0c`](https://github.com/AMD-Ecosystem/ai-ecosystem/blob/a0f9d0c028bf5220f9a640581c0f2995b8e87d6d/docs/inference/onnxruntime.rst).
Pip MIGraphX libraries are a dependency requirement; an OS-installed SDK alone
does not satisfy it. Gfx115x describes GPU architecture, not NPU qualification.

The [Windows ML provider requirements](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)
and [release ledger](https://github.com/microsoft/WindowsML/wiki/Windows-ML-Execution-Provider-Releases)
are separate from AMD's Linux wheel guide. The observed AMD package is MSIX
1.8.65.0, dated 2026-10-28. Windows ML officially supports
[Python deployment](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/distributing-your-app):
Python 3.10–3.13, x64/ARM64, an unpackaged interpreter and framework-dependent
deployment with the matching Windows App SDK Runtime, MachineLearning projection,
bootstrap package and `onnxruntime-windowsml`. Resolve that supported closure
first. The audited app projection is 2.3.0; its metadata pins
`onnxruntime-windowsml==1.25.2.202605110140` and `winrt-runtime~=3.2.1.0`.
Its published release was still 2.3.0 on the
[publisher's PyPI page](https://pypi.org/project/wasdk-Microsoft.Windows.AI.MachineLearning/)
when checked 2026-11-09. Acquisition now uses pip's dependency resolver, rather
than `--no-deps`. The existing bridge does not establish a compatible AMDGPU
2.5.96+ native runtime/API closure: explicit AMDGPU acquisition and old bridge
records are rejected with that blocker. Projection and native SDK versions are
separate evidence; a package version is not proof of installed native APIs.
No native helper was built or installed. The audited Python surface already
exposes catalogue discovery/readiness APIs; a helper is not justified merely
to bypass their certification or dependency constraints. A future supported
closure still needs matching App SDK Runtime/bootstrap, ORT, certification,
provider device and original Bria acceptance.

[ROCm 10.1 release notes](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html)
document limits on WSL tooling. The legacy
[`ROCMExecutionProvider`](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html)
was removed from ORT 1.23; it is not the new MIGraphX plugin.

## Plugin registration and device identity

The [AMDGPU plugin source](https://github.com/onnxruntime/onnxruntime-ep-amdgpu/tree/58afd43bc4c2ddc7582712e1d75cbfd4c3bb03fc)
contains distinct AMDGPU umbrella, DirectML and MIGraphX implementations.
Its source-build requirements do not prove a ready-to-install Python wheel.
The observed public PyPI packages
[`onnxruntime-ep-amdgpu`](https://pypi.org/project/onnxruntime-ep-amdgpu/) and
[`onnxruntime-ep-migraphx`](https://pypi.org/project/onnxruntime-ep-migraphx/)
only listed placeholder 0.0.0 distributions. Acquisition must identify a real
native payload from the intended publisher, not just a matching package name.

Use the [ORT plugin API](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/usage.html):
register an absolute library path in the worker, enumerate its EP devices, and
attach the selected device through `SessionOptions.add_provider_for_devices`.
Devices passed to one call must belong to the same EP. Registration is different
from `register_custom_ops_library`; installed provider names are not device IDs.
Retain DLL-directory handles for the session lifetime. Keep vendor paths and
environment overrides process-local, including on mixed-vendor hosts.

Planned runtime records include package/source/hash, ORT API, Python/OS/ABI,
plugin name/path, native library/data closure, provider device identity,
options and model contract. Use a common worker record where practical; keep
acquisition adapters distinct for vendor wheels and Windows ML's catalogue.
Do not invent `HIPExecutionProvider`: direct HIP uses the caller's registration
alias (typically `hipgpu`), whereas the umbrella uses `AMDGPUExecutionProvider`.
HIP's upstream OGA configuration selects `{"AMDGPU": {"profile": "hip"}}` with
a compatible umbrella package. The EP name alone does not identify that path;
do not assume every Windows ML AMDGPU package exposes the same profile.

## HIP-EP evaluation gate

Audited [main `30869b0`](https://github.com/ROCm/hip-ep/tree/30869b0dab8171d801afcde0c42716182bd8e95d)
and [v0.5.2](https://github.com/ROCm/hip-ep/releases/tag/v0.5.2), whose tag is
`ffeed9deb4c13262fff5c1ac9d8350d33f6b8ef5`, are different dependency closures.
The release bundles ORT 1.30.0; main pins 1.31.0. The runtime negotiates ORT API
24+, but that is not a blanket ABI/model compatibility claim.

The [Windows SDK resolver](https://github.com/ROCm/hip-ep/blob/30869b0dab8171d801afcde0c42716182bd8e95d/cmake/deps.cmake)
automatically supplies gfx115x/gfx117x families, not gfx1100. Source accommodation
for gfx1100 does not supply matching SDK, kernels or hipBLASLt data. The published
Linux payload targets gfx1151. A 7900 XTX build needs a verified matching SDK
closure before any proposed UI install option.

The [Python package guide](https://github.com/ROCm/hip-ep/blob/30869b0dab8171d801afcde0c42716182bd8e95d/site/docs/get-started/python-package.md)
uses Python 3.14 for supplied ORT/OGA bindings. The plugin's generic `py3-none`
tag does not make those cp314 bindings usable in RF's cp312 worker. Evaluate a
supported cp312 closure or an explicitly managed alternate interpreter; do not
silently change the app Python. Plain ORT Bria does not inherently require OGA
or the umbrella. The [operator reference](https://github.com/ROCm/hip-ep/blob/30869b0dab8171d801afcde0c42716182bd8e95d/docs/supported-operations.md)
requires dtype/shape/attribute validation, not just matching operator names.

## Acceptance and recovery

1. CPU fixtures verify platform/ABI/target eligibility, artifact identity,
   registration failures, stale records, offline behavior and explicit CPU.
2. Optional acquisition is explicit, cancellable and atomic. Browsing Settings
   must not install a provider, compile a graph or probe GPU execution.
3. Session creation and inference are separate phases. Disable ORT's hidden
   run retry; report any application-managed CPU recovery and its reason.
   [Python session API](https://onnxruntime.ai/docs/api/python/api_summary.html).
4. Real qualification records device/node assignment, CPU partitions, original
   Bria hash, native rembg preprocessing, alpha/dimensions, numerical/visual
   comparison, cold compilation and warm end-to-end cost. Enumeration/canaries
   cannot substitute for this evidence. Preserve a useful explicit failure.
5. Cache reuse requires model, preprocessing, shape, dtype, provider/runtime,
   device, driver and options compatibility. Unqualified Windows ML/HIP paths
   remain excluded from Auto. No derived graph or TDR changes to obtain a pass.

Keep registration, execution, quality, speed and repeated-process stability
results separate. Unavailable vendor hardware remains blocked, not failed by
association with an AMD test. The bounded original-Bria attempts above did not
establish GPU model qualification; other devices/providers remain unverified.

## Implemented recipe and CPU evidence boundaries

`provider_recipes.py` produces exact `RuntimeCandidate` decisions through the
shared pure capability policy. Native Linux requires the published GFX allowlist,
an identified AMD adapter/driver, native guest policy, validated Python 3.12,
matching wheel tags and complete artifact metadata. It pins ORT 1.29.0,
plugin 1.0.0+rocm10.1.0 and pip `migraphx`/`migraphx-libs` 2.18.0+rocm10.1.0.
Every supplied closure artifact needs a hash; pip receives exact wheel URLs with
SHA-256 fragments. A package-manager SDK or a placeholder wheel cannot satisfy
this route. The explicit install action produces its own evidence with
`acquisition.prepare_migraphx`: current interpreter/tags, pip dry-run dependency
report, exact hash-checked wheel downloads and native-library inspection. It
installs those local wheels in staging. No user-supplied wheel metadata is needed.
Passive Linux KFD topology supplies GFX and the DRM device's physical PCI slot;
it does not initialize HIP or fabricate an EP device index. See the
[kernel topology source](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdkfd/kfd_topology.c).
The active worker then discovers EP devices, binds a singleton or explicitly
selected provider identity and runs the installation canary. Ambiguous physical
or provider devices fail with a selection diagnostic. Driver acceptance remains
unknown until that explicit verification succeeds. Missing sysfs/artifact/API
evidence returns its precise blocker. WSL remains a separate technical-preview gate.
HIP evaluation reports the cp314 binding and gfx1100 SDK/kernel blockers and
cannot become installable or automatic from a generic `py3-none` plugin tag.

`plugins.py` uses the publisher module's `get_ep_names()`/`get_library_paths()`;
it validates matching aliases, absolute native library paths inside the isolated
environment, recorded library hashes and AMD EP devices before attaching one
device with `add_provider_for_devices`. A process-local worker owns registration.
Windows ML validates every saved record before ranking, skips stale records,
tries later candidates after registration/device failures and retains DLL
directory handles during use. Explicit discovery can list all valid installed
providers. Device IDs hash EP name plus provider-specific hardware identity;
Torch, DirectML and plugin indices are not interchangeable.

Native acquisition stages a new environment, resolves declared requirements,
checks the worker's exact ORT version and publishes only after its installation
canary succeeds. A replaced folder is retained as `.previous-<identity>` for
rollback. Failed/cancelled staging is disposable. Acquisition APIs accept an
optional `check_cancelled` callback; its owned process/job is terminated without
publishing success. Offline/install-skip refuses before work. Windows ML publishes
its provider manifest atomically after successful certified acquisition. Wiring
an interactive cancel control to that callback belongs to the UI caller.

The complete native transaction holds `installation_lock(final_runtime_path)`,
including preparation, verification and replacement. Windows ML holds one lock
for its catalogue/manifest directory across acquisition and read/merge/publish,
including qualification publication. Installed state is rechecked inside the
lock. Unique temporary manifest files avoid fixed-name collisions; two vendors
cannot overwrite each other's provider rows. Locks coordinate cooperating
processes across profiles, rather than relying on a Gradio queue.

### Original Bria qualification record

Choosing the Windows ML engine with vendor `Auto` is automatic vendor selection.
Acquisition records and a generic `qualification="qualified"` flag cannot enable
it. Named vendors remain explicit bounded qualification routes; discovery lists
all valid acquired records, including unqualified ones.

`qualification.current_contract` binds the original rembg 2.0.87 Bria graph
(`5b486f08200f513f460da46dd701db5fbb47d79b4be4b708a19444bcd4e79958`),
native preprocessing, 1024-square float32 input, provider DLL/catalogue identity,
ORT version and hashes of its native capi binaries, interpreter/ABI, exact EP
device identity/type, Windows build/current display-driver inventory and options.
Existing model resolution never downloads or substitutes a graph. Missing or
changed evidence fails closed. The worker binds the recorded EP device and
rechecks the contract against the observed device before model initialization.
Display-driver metadata cannot certify an NPU; NPU vendor Auto remains blocked,
while explicit discovery and qualification remain available.

For the coordinator's real-model check, an explicit `IsolatedMaskSession` exposes
the current contract as `ready_info["qualification_contract"]` and the latest
successful executed-node evidence as `last_execution["assignment"]`. Failed or
CPU-recovered requests clear `last_execution`. Measure cold/warm and CPU
**end-to-end** costs externally, and evaluate the actual native postprocessed
outputs. Call `windows_ml.record_qualification(name, contract, evidence)` only
with those completed results. Its evidence schema requires `qualified`,
`real_model`, `original_bria`, `numerical_passed`, `visual_passed`,
`stability_passed` and `hidden_fallback_disabled` to be true, first-inference
`assignment` with executed nodes on the requested provider, and positive finite
`cold_end_to_end_ms`, `warm_end_to_end_ms`, `cpu_end_to_end_ms`; warm acceleration
must beat CPU. The API records schema 1, the full contract and its fingerprint
atomically after rechecking current identities. Subsequent Auto selection checks
that complete contract and the actual provider device again. A canary, slower
route or stale model/runtime/device/driver/options record cannot qualify Auto.
Compiled-model reuse remains disabled independently of this selection record.

Existing CUDA/OpenVINO/legacy MIGraphX Auto behavior is retained; its status is
initialization/execution evidence, not a new quality or performance claim. New
Linux plugin Auto remains hard-disabled until a matching Linux model/device/driver
qualification producer is shipped; a generic `qualified` marker cannot enable it.
Windows ML vendor Auto requires the exact original-Bria acceptance record above.
No compiled-model cache is enabled. `execution.cache_identity` requires the full
model/preprocessing/shape/dtype/provider/runtime/device/driver/options contract;
unknown fields cannot certify reuse.

Focused CPU command:

The retained explicit DirectML installer accepts an owned cancellation checker.
Its package subprocess is bounded to 300 seconds and its isolated canary to 60
seconds. Cancellation is checked again before publishing the verification marker;
that marker certifies the canary only, not the real background-removal model.
`test_legacy_onnx_install_cancellation.py` verifies this boundary without installing
packages or running a device workload.

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_onnx_execution.py test_onnx_provider_recipes.py test_onnx_runtime.py test_onnx_installer.py test_onnx_install_transactions.py test_windows_ml.py test_windows_ml_qualification.py test_mask_input_contracts.py test_rembg_runtime.py test_image_rembg_options.py
```

Fixtures cover strict init/run retry suppression, real CPU canary profiling,
no/partial assignment, stale-first records, registration/device failure,
ABI/GFX/vendor/native-payload/hash gates, offline/CPU scope, acquisition rollback,
cancellation and request/cache identity. These do not certify Radeon, NVIDIA,
Intel, NPU, native Linux or WSL model execution. Real Bria's original hash,
input/alpha/dimensions, cold/warm end-to-end cost, numerical/visual comparison
and repeated-process stability remain coordinator-owned acceptance.

## Catalogue browsing and explicit refresh

Hardware settings read the bounded cached catalogue for the current OS,
architecture and Python branch. Only locally certified, bridge-compatible
providers become candidates; an unavailable saved choice remains visible.
Browsing reads acquisition metadata without hashing provider payloads, importing
rembg, probing devices or acquiring packages. Runtime admission still verifies
the provider DLL and exact workload contract.

An explicit catalogue refresh runs the existing isolated projection's
`find_all_providers()` metadata API with a 45-second limit and cancellation.
It does not call `ensure_ready_async()` or install anything. CPU authority starts
no catalogue/device process. A failed refresh preserves the previous snapshot.
`test_windows_ml_catalog_ui.py` covers these boundaries; local catalogue presence
does not establish actual model acceleration.
