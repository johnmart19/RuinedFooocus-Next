# Native llama.cpp SDK acquisition

Native llama.cpp uses a separate ROCm 10.0 environment matching its published
binary. Image PyTorch keeps its own ROCm 10.1 environment. Do not mix their
libraries in the application process.

`backends/llama/llama_installer.py` reuses the bootstrap's hash-pinned AMD selector
source build. Its direct local wheel requirement carries the requested library
and device extras, so wheel-only dependency resolution can select the complete
SDK closure. Offline/install-skip mode rejects acquisition before any build.
Package installation still uses the resolved artifact hashes and an isolated
environment; failed replacements preserve the existing active SDK pointer.

Run `python tools/check_cpu.py test_installer_llama.py` for offline, staged
replacement and exact direct-wheel extras regressions. These checks do not
certify GPU execution. The Ubuntu WSL acceptance run on 2026-11-10 acquired the
SDK through Chat's normal Load action and answered with native ROCm on a
Radeon RX 7900 XTX. Vulkan GPU availability remains a separate driver capability.

ROCm 10.0 also requires ROCDXG inside WSL. The newer image SDK includes its own
bridge; the native SDK must not borrow image runtime libraries. On WSL only,
native acquisition extracts the single ROCDXG 1.2.2 library from AMD's official
release into `cache/llama.cpp/rocdxg-1.2.2`. Both archive and library SHA-256 are
pinned. Extraction uses `dpkg-deb --fsys-tarfile`; no package scripts run and no
system installation or administrator access is needed. A matching hash-verified
system ROCDXG library can also be reused without downloading another copy.
The wheel SDK's library directories stay first in the child search path, so
reusing the system bridge cannot replace the SDK with an older system HIP stack.
Offline diagnostics use only verified local libraries. Windows and native Linux
skip this bridge entirely.

The requirement and release are documented by [AMD's WSL guide](https://rocm.docs.amd.com/projects/rocm-cli/en/latest/install/wsl.html)
and the [ROCDXG release](https://github.com/ROCm/librocdxg/releases/tag/v1.2.2),
checked 2026-11-10. The library's source moved to ROCm/rocm-systems; this pinned
binary is used specifically with the native ROCm 10.0 runtime.

## Updating an official prebuilt

In Hardware, select the native llama.cpp CPU or Vulkan backend on Windows or
Linux, then click **Check for llama.cpp update**. The check looks at recent
[official llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)
for the highest numbered build with a matching OS, architecture, and backend.
The result shows the tag, prerelease status, and archive size. Offline and
skip-install modes disable release checks and downloads.

**Install llama.cpp update** downloads the selected archive, checks the published
byte count and SHA-256, and verifies the staged executable before selecting it.
Vulkan also checks for a usable device. The selection applies to the next
native Chat process using that explicitly configured backend. An active
Chat process continues with its current executable until it reloads.

**Use pinned llama.cpp build** clears the selection for that backend. Cached
versions remain available, including the app pin. Auto, custom `LLAMA_SERVER`
binaries, xllamacpp, CUDA, ROCm, and SYCL continue through their existing
selectors. Run `python tools/check_cpu.py test_llama_update.py` for the
offline release, archive, selection, and rollback contracts.

## Installed runtime inspection

Local Chat's **Installed chat runtimes → Refresh** inspects existing builds only.
It preserves each build/backend identity and the embedded wheel's recorded backend;
an installed package is not proof of a usable GPU or a qualified model. The action
has a whole-operation deadline/output cap and a per-browser **Cancel runtime refresh**
control. Global CPU authority shows passive package/cache metadata without device
queries. The [diagnostic contract](amd-runtime.md#existing-settings-diagnostic)
defines these limits and separates CPU fixtures from real provider execution.

## Reviewed managed prebuilt pin and acquisition

The managed native source pin is **b12366**, source commit
`31953bcf74ba489a8bbb0c936c5fcf24fcf9485c`. This is an upstream prerelease,
published 2026-11-13. The versioned
`backends/llama/releases/b12366.json` freezes all 38 official archive names, byte
counts, and SHA-256 digests reviewed on that date. The old b12327 manifest remains
available for verified older cache identities. The release source contains
`--moe-cache-cpu-share`; an individual executable still needs capability
discovery, and host experts plus an initialized cache remain independent
prerequisites. Source support and archive integrity do not establish device
execution, artifact quality, or performance.

The official Windows CPU-only archive passed bounded `--version` and `--help`
checks without loading a model or enumerating devices. It reports
`1.0.0-dev (build 12366)` and retains RF's server, projector and MoE-cache CLI
options. The source review preserves exact architecture/task gates separately;
neither help output nor passive Vulkan payload verification qualifies GPU or
real-model execution for this build.

Saved CUDA 13.0, 13.2, and 13.4 choices migrate to CUDA 13.5 for this pin. Driver
and architecture admission uses the effective 13.5 requirement before acquisition.
Version-specific package lookup retains the old 13.4 asset names for older
builds. Windows CUDA 12.4 and Linux CUDA 12.8 remain separate choices for eligible
legacy GPUs; CUDA 13 excludes pre-Turing devices. CUDA runtime companions are
verified independently and staged with their matching package. Vulkan remains
an independent backend. SYCL retains its existing driver/oneAPI requirements;
ROCm 10.0 keeps its separate ABI-matched SDK and the hash-pinned WSL ROCDXG bridge.
Custom `LLAMA_SERVER` binaries retain their own identity and capability checks.

Acquisition exposes resolving, transfer, verification, staging, and device-probe
phases. Transfer events include received bytes and the published total when
known; each CUDA companion is a separate package. SDK resolution and acquisition
do not invent a byte total. Cancellation is polled through transfer chunks,
manifest hashing, extraction, owned native children, and immediately before
publication. Offline and skip-install modes acquire nothing.

Archives must match their published size and SHA-256 before safe, bounded
extraction. A complete per-file manifest and unique server executable are staged
before the new cache directory is renamed into place. Updates additionally
verify the staged executable and Vulkan device before atomically selecting it.
The independent `backends/llama/runtime_probe.py` provider owns the bounded,
cancellable device check for both acquisition and Chat; the updater does not
depend on the Chat server implementation. Its caller adds the exact runtime
identity to progress events.
Known missing-loader diagnostics retain their dependency-recovery classification;
device crashes still require an explicit backend recovery decision. Bounded child
output is available only to trusted classification code; public failure text
does not include child stderr or command arguments.
Cancelled or failed attempts preserve existing verified cache directories and
explicit update selections. An explicit verified selection also survives an app
pin change; **Use pinned llama.cpp build** clears that selection for the next
load. Active processes keep their current executable and SDK. Replacing an
existing SDK prepares a separate environment and changes its active pointer only
after validation.

The optional backend callback contract is `progress_callback(event)` and
`check_cancelled()`; cancellation raises `concurrent.futures.CancelledError`.
Events carry a phase, runtime tag/backend/package, safe message, and known
received/total/rate fields. The Chat load owner supplies request identity and
rejects stale publication. These backend callbacks also preserve existing
callers that omit them.

Run `python tools/check_cpu.py test_installer_llama.py test_llama_update.py
test_llama_acquisition.py test_wsl_llama_bridge.py test_llama_runtime_probe.py`
with the checkout interpreter.
These use inert archives and mocked native work. They cover versioned package
families, legacy aliases, integrity, offline reuse, cancellation, retry, and
selection retention; they do not download or qualify a real runtime.

Primary sources, checked 2026-11-13:
[b12366 release](https://github.com/ggml-org/llama.cpp/releases/tag/b12366),
[release asset metadata](https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b12366),
[CPU-share parser](https://github.com/ggml-org/llama.cpp/blob/31953bcf74ba489a8bbb0c936c5fcf24fcf9485c/common/arg.cpp),
[CUDA architecture selection](https://github.com/ggml-org/llama.cpp/blob/31953bcf74ba489a8bbb0c936c5fcf24fcf9485c/ggml/src/ggml-cuda/CMakeLists.txt).
