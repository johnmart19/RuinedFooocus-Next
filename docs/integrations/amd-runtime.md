# AMD runtime boundaries and optional tools

This is a source audit, not a performance report. See the
[audit identities](README.md) and [ONNX-specific contracts](onnx-providers.md).
Prefer native Torch/ComfyUI paths. A useful upstream project does not become an
application dependency merely because it is in the ROCm organization.

## Discover the installed SDK

TheRock separates runtime/core/device/development packages. Resolve libraries
from the selected interpreter's installed distribution instead of assuming
`C:\Program Files\AMD\ROCm\7.2` matches the Torch wheel. Its
[`rocm_sdk.find_libraries(*shortnames)`](https://github.com/ROCm/TheRock/blob/main/docs/packaging/python_packaging.md)
returns installed absolute paths. Record executable/interpreter, package
versions, device extras and effective paths; don't globally overwrite `HIP_PATH`
or `PATH` for unrelated CUDA/Intel/CPU workers.

SDK `version`/`targets`/path queries are distinct from active tests. First access
to development paths or `init` may expand/link files; `rocm-sdk test` performs
installation checks that can run GPU work. Package discovery must not implicitly
run it. [TheRock release instructions](https://github.com/ROCm/TheRock/blob/main/RELEASES.md).

Use the [support definitions](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md):
buildable, installable, sanity-tested and release-ready are separate states.
TheRock release readiness is also distinct from AMD's released-platform matrix.
Check the exact framework, Python ABI, platform and GFX artifact closure. Do not
turn TheRock's broad target list into an ONNX or Windows ML support list.

### Current ownership and passive evidence

The 2026-11-09 follow-up inspected RF `3174e80` and installed Python 3.12.10
metadata, without importing Torch/ROCm, executing a diagnostic or querying a GPU.
The active environment contains Torch `2.14.0+rocm10.1.0`, torchvision
`0.29.0a0+rocm10.1.0`, torchaudio `2.11.0.3+rocm10.1.0`, and `rocm`,
`rocm-sdk-core`, `rocm-sdk-libraries`, `rocm-sdk-device-gfx1100` at `10.1.0`.
These identities establish package ownership, not loaded DLLs or working kernels.

- **Image Torch:** its installed `torch/_rocm_init.py` calls the native
  `rocm_sdk.initialize_process` with a `10.1.*` check and absolute-path library
  preloading. RF retains that initializer. `gpu_installer.installed_torch_constraints`
  protects the wheel-owned ROCm dependencies when resolving an overlay, and the
  combined `PackagePlan` checks the selected interpreter and full closure before
  mutation. RF does not select a system `HIP_PATH` directory ahead of this route.
- **Optional native Chat ROCm:** the pinned llama.cpp build uses its separate
  ROCm `10.0.0` SDK environment. On Windows, `windows_rocm_environment` queries
  `find_libraries('amdhip64', 'hipblas', 'hipblaslt')` in an interpreter, then
  prepends those directories to a child environment. A 10.1 image wheel does not
  satisfy its exact 10.0 check. Linux uses that pinned SDK's known package layout.
  The published `.active.json` SDK pointer selects staged replacements for both
  serving and the installed-runtime diagnostic; invalid/outside-cache pointers
  retain the base SDK, including malformed JSON shapes and non-string paths.
  An inherited `HIP_PATH` is not rewritten globally.
- **Missing SDK evidence:** Windows discovery with `install=False` cannot acquire
  the optional SDK. Missing metadata stays unavailable. Target/GFX detection is
  a different action: `detect_rocm_targets` can query hardware and, on an allowed
  acquisition path, prepare its isolated helper. It is not passive inventory.

The existing native SDK APIs satisfy the discovery contract; no new SDK resolver,
devel wheel or diagnostics dependency is adopted. In the inspected SDK, the
three library names above belong to host packages. Target-specific discovery can
instead invoke `offload-arch`; do not generalize this metadata-only use into a
promise that every `rocm_sdk` query is passive. `initialize_process` itself loads
libraries and remains the owning framework's runtime action.

Installed-source SHA256 identities for this audit:

```text
torch/_rocm_init.py       79e195dd7081a8c0a6bab1955ab0af7a42979e996ad4a8971c133fa25867f069
rocm_sdk/__init__.py      ed87e36cc153b7811133fb934ff1f018114c394f32a56aa040663ef66c147c3d
rocm_sdk/_dist_info.py    1a275d0b2f2b4e3b69aa08ae8cee9f14d231eedd08e9cc8f3403e64f821f7876
```

## Optional standalone development SDK

Wheel-only inference does not require a separately installed HIP SDK. For
building native code, AMD publishes a full Windows ROCm 10.1.0
[gfx110X-all SDK](https://stable.repo.amd.com/rocm/core/tarball/therock-dist-windows-gfx110X-all-10.1.0.tar.gz).
The [stable target mapping](https://github.com/ROCm/TheRock/blob/b324869fb3116353ac7eec5e6926aa3d452ac987/cmake/therock_amdgpu_targets.cmake)
includes RX 7900 XTX/gfx1100. Use a released matching artifact before deciding
to build all of ROCm from source.

Follow [versioned installation guidance](https://rocm.docs.amd.com/en/docs-10.1.0/install/rocm.html):
inspect archive paths, retain component licenses, stage into an empty owned
directory, then publish a new versioned SDK directory. Preserve previous SDKs.
The archive may use a leading `./` layout; verify entries before stripping a
component. Record download SHA256 separately from publisher verification; an
S3 ETag is not a SHA256 checksum.

Use an explicit compiler path and child-process environment to select that SDK.
Ordinary runtime setup must not modify global `HIP_PATH` or redirect the app
away from its wheel libraries.
An explicit user request to change the Windows development default is a separate
setup action: back up the existing value, verify the SDK, change only `HIP_PATH`,
preserve version-specific aliases and unrelated PATH entries, then notify new
sessions. Already-running processes retain their inherited environment. This
does not change or qualify the application's wheel-owned inference runtime.
Initialize the existing MSVC developer environment when compiling on Windows.
A minimal HIP compilation with `--offload-arch=gfx1100 --cuda-device-only` checks
the toolchain without launching a kernel. HIP output can be an offload bundle:
list/unbundle it with `clang-offload-bundler` before checking its AMDGPU target
using `llvm-readobj`. This does not qualify driver/runtime execution.

Check the exact [OS/driver matrix](https://rocm.docs.amd.com/en/docs-10.1.0/compatibility/compatibility-matrix.html)
separately. A newer OS build is not automatically a tested branch. Before any
later GPU acceptance, inspect effective loaded runtime/DLL identities: a process
PATH does not override every Windows DLL search rule. Do not replace driver or
System32 files. Source builds remain a fallback requiring the
[matching build prerequisites](https://github.com/ROCm/TheRock/blob/b324869fb3116353ac7eec5e6926aa3d452ac987/docs/development/windows_support.md).

## Reuse native attention and memory handling

See [ComfyUI platform reuse](comfy-platforms.md) for the exact runtime/installer
ownership split and [precision and environment controls](amd-controls.md) for
the applicability of AMD's architecture-specific guidance.

Pinned ComfyUI `comfy/model_management.py` already has SDPA/AOTriton eligibility,
an active canary/fallback and a gfx11 VAE head-dimension guard. Its
`comfy/ldm/modules/attention.py` routes optional attention backends. Compare the
exact new pin and affected callers before deleting an RF workaround or adding
another global patch. Canaries remain active compute and must respect CPU mode.

The native update to `fca60fe70e5b84c5c6b058263653d15cb3401e44`
leaves startup, SDPA/AOTriton canary/fallback, the gfx11 VAE guard, native MIOpen
policy and optional attention dispatch unchanged. RF retains the separate gfx11
wide-head encoder safeguard and Windows ROCm Qwen workspace protection: neither
of the two new native compute fixes replaces those contracts. The FP8 combined
linear fix preserves a native split path; it does not enable native FP8 matrix
arithmetic on gfx1100. No AITER, FlashAttention or Triton replacement was added.

RF's CPU import boundary now suppresses the model manager's early optional
accelerator discovery before applying native CPU state, with exception-safe
restoration before service threads start. This is independent of the existing
CPU preference provenance and image-only versus global CPU policy. Complete
native CPU import and caller fixtures pass without accelerator probes; no AMD
operator, real-model, timeout, quality or stability acceptance follows from them.

**ComfyUI's `--use-ck-attention` means comfy-kitchen INT8 attention, not AMD
Composable Kernel.** `--enable-triton-backend` also controls comfy-kitchen.
Neither flag certifies AITER, FlashAttention's CK path or a hardware speedup.

| Project                                                                                                                                                                              | Decision and integration boundary                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [AMD FlashAttention](https://github.com/ROCm/flash-attention#amd-rocm-support) / [AITER](https://github.com/ROCm/aiter/blob/main/setup.py)                                           | Evaluate only an opt-in, pinned environment for a demonstrated native gap. Verify OS/GFX/dtype/head dimensions and exact Torch/Triton closure. Installation can replace Triton; no unconditional requirements entry or global attention patch.             |
| [rocWMMA](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocwmma) / [Composable Kernel](https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel) | Defer direct integration. Kernel-development libraries belong behind upstream consumers until an operation-specific need is measured. A package alone adds no Python acceleration.                                                                         |
| [MIGraphX](https://github.com/ROCm/AMDMIGraphX)                                                                                                                                      | Graph compiler used through a qualified inference integration; not a replacement for arbitrary dynamic ComfyUI execution.                                                                                                                                  |
| [Quark](https://quark.docs.amd.com/release-0.12/release_notes.html)                                                                                                                  | Defer app integration; reserve for separately requested export/quantization work. Version 0.12/packaging-only 0.12.post1 tests Torch 2.10/2.11 and Python 3.11–3.13, a different closure from RF's Torch 2.14/ROCm 10.1. No weight conversion is included. |

These decisions also protect NVIDIA and Intel installations from AMD-only
imports, DLL search paths and package substitutions. Keep the common capability
schema vendor-neutral; select an adapter only after platform and workload checks.

FlashAttention's audited installation baseline is Linux; Windows compilation
requires separate qualification. AITER's Windows fallback does not establish
full operator coverage or equivalence to its Linux acceleration paths.

## Optional diagnostics, not startup dependencies

[ROCm CLI](https://github.com/ROCm/rocm-cli/blob/main/README.md) (`rocm`) is a
separate managed local-AI launcher, not TheRock's `rocm-sdk`. It has its own
runtimes/engines; its documented default SDK is 7.14 unless 10.x is selected.
Do not silently adopt it as RF's installer. An explicit developer support action
could collect `rocm examine --json --framework skip`, with timeout and redaction.
That suppresses framework probes, not all hardware queries. Never run its
install/fix/adopt/update/serve commands from passive diagnostics.
**Decision: defer application integration.** Its own managed environment is not
needed to inspect RF's wheel metadata or resolve its runtime libraries.

[AMD SMI](https://rocm.docs.amd.com/projects/amdsmi/en/latest/index.html) is
monitoring/control infrastructure; prefer its API to scraping CLI text when a
supported installation is already present. Its
[WSL backend](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-wsl-mode.html)
is experimental and opt-in, with missing metrics; neither blanket WSL support
nor blanket WSL impossibility is accurate.
**Decision: defer a dependency or telemetry action** until a specific support
need and supported installation justify it. No SMI query is added at startup.

[ROCprofiler](https://rocmdocs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
is an optional active profiler. RDNA3/4 counter collection may require a changed
performance state. This is not a passive health check or permission to tune power.
**Decision: defer profiling integration.** A future measurement needs its own
bounded workload, output/retention policy, cancellation and power-state authority.

### Existing Settings diagnostic

**Settings → Local Chat → Installed chat runtimes → Refresh** is an explicit
device query, not passive package discovery. It runs each installed native
binary's `--version` and `--list-devices`, and queries xllamacpp device information
in a child interpreter. Each query, including Windows SDK metadata for this
action, has a 15-second timeout and 16 KiB combined stdout/stderr cap. No model is loaded or acquired by
this action. A Linux staged-SDK path mismatch is fixed so the query uses the same
active SDK as native serving. Malformed SDK metadata reports an unavailable build
instead of aborting the complete refresh.

**Cancel runtime refresh** removes this browser's queued Gradio event through the
native cancel route and signals its running child through the session event. An
actual inert `/queue/join` → `/cancel` test verifies that another session's queued
refresh remains. The complete
refresh, including discovery and SDK metadata, runs inside one owned child with a
45-second deadline and 64 KiB output limit. At most eight cached builds and 1,024
directory entries are considered; output identifies each build/backend separately
from the embedded wheel's recorded backend. Nested owned processes are terminated
and reaped on cancellation, deadline or overflow. Failed/cancelled refreshes retain
the previous completed display. The bounded I/O lane grants no model/compute slot.

Explicit global CPU authority uses package/cache metadata only and never runs a
native binary, imports xllamacpp or queries devices. An image-only CPU preference
does not widen into global CPU policy. Otherwise Refresh remains an explicit
device query which may initialize a driver. It uses local files with offline Hub
flags and `install=False`/`acquire=False`; no package, bridge or model acquisition
is hidden inside the action. The runtime-choice timer reads application state,
not this probe. Real hardware responsiveness and cancellation acceptance remain
separate from the fake-query CPU tests.

`test_runtime_info_refresh.py` and `test_sdk_discovery.py` verify session isolation,
duplicate-action fencing, private UI events, whole-operation limits, CPU policy,
build/backend identity, bounded SDK metadata and retained serving SDK paths.

Classify every diagnostic action:

- **Metadata:** package versions, wheel tags, configured and resolved paths.
- **Device queries / active work:** adapter enumeration, compute canary, model
  run, profiling. State which it is; bound time and preserve concurrent services.
- **Mutation:** install, SDK extraction, environment adoption, driver/power/TDR
  changes. Never hide these behind refresh or automatic startup.

Developer evidence records exact command/executable, Python, source and package
identities, driver/GFX, relevant environment overrides and whether any compute
ran. Redact machine identifiers and credentials. Absence of a diagnostic finding
does not establish model correctness or driver stability. No optional tools were
installed or executed during this audit.

Focused CPU evidence for the SDK path fix:

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_sdk_discovery.py test_installer_llama.py test_comfy_runtime.py test_gpu_support_matrix.py
```

Fake SDK directories, missing/conflicting pointers, stale system paths, native
API child commands, malformed results, timeouts and CPU/other-vendor routes are
covered. All diagnostic/native execution and acquisition calls are mocked. No
SDK extraction, `init`, `test`, device query, model run or GPU work was performed.
Upstream TheRock packaging, rocm-cli README, AMD SMI WSL, ROCprofiler and Quark
references were rechecked on 2026-11-09; their published support is not RF hardware
qualification.
