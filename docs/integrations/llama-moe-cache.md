# Native MoE cache CPU share

**MoE cache CPU share** is enabled by default in RF, with a configurable share
of **0.5**. This midpoint is RF policy. Upstream does not recommend one share for
every machine, model or cache size, and no performance improvement is claimed.
The value is a fraction from 0 to 1, not a boolean or a VRAM allocation.

The native `--moe-cache-cpu-share F` option computes that fraction of cache misses
on the CPU from host-resident experts **after the MoE GPU cache is full**. Cache
hits remain on the device. Before the cache fills, misses are uploaded normally.
This requires an admitted GPU runtime, host MoE placement such as native measured
fit, `--cpu-moe` or `--n-cpu-moe`, and an already configured nonzero
`--moe-cache-mib`. RF does not invent a cache size, move experts or start a tuning
workload when this setting is changed. Dense models and ordinary CPU-only loads
do not gain a CPU-share workload from enabling the preference.
Captured zero GPU layers, the effective native device environment and CLI
offload overrides participate in this gate. Tensor overrides are checked per
assignment: an unrelated `output=CPU` rule does not place MoE experts on the CPU.
For an unchanged model whose bounded GGUF descriptors establish MoE weights,
the supported preference can be passed with an already configured cache before
native fitting chooses expert placement. That neither forces host experts nor
proves cache execution; initialized full-GPU placement keeps it dormant.

Turning the RF toggle off omits RF's option. Explicit `llama_server_args` values
and the native `LLAMA_ARG_MOE_CACHE_CPU_SHARE` environment override take priority
and are reported with their own source. Zero is a valid explicit share and means
no CPU miss share. Invalid, nonfinite or out-of-range values fail before runtime
resolution and process launch. Settings are captured for a load; changing them
requires Chat reload. The loaded status distinguishes requested preference,
selected value/source, parser support, an applied native option and observed
execution. Passing a supported option is not evidence that a full-cache miss was
computed on the CPU. Initialized full-GPU placement marks the preference dormant.

## Runtime support and migration

The legacy **b12110** source contains `--moe-cache-mib` but lacks
`--moe-cache-cpu-share`. Its requested default therefore remains **inactive**.
The reviewed **b12327** source contains the option; the installer owns its separate
pin/platform migration. Source support is not binary or hardware qualification.
The separately reviewed b12366 source retains this option. Its official Windows
Vulkan package has an independently verified archive, launcher and parser-DLL
identity registered alongside b12327; passive inspection found both option
literals in that exact DLL. No DLL was loaded or GPU probed for that check.
The native server checks its actual executable for both parser-option and
environment-option literals, with a file-signature cache and a bounded passive
read. No help/version subprocess, device discovery or model load is used for
this capability check. Missing or unestablished parser support stays inactive;
the [official b12327 Windows Vulkan archive](https://github.com/ggml-org/llama.cpp/releases/download/b12327/llama-b12327-bin-win-vulkan-x64.zip) uses a small launcher and places the
parser in `llama-common.dll`. This exact archive has a separate passive fallback:
the same-directory `rf-runtime.json` must match its registered publisher archive
identity and the exact registered launcher/DLL SHA-256 values, and both actual
files must match. Only those two binaries are inspected. An arbitrary neighboring
DLL or a fabricated local receipt cannot supply parser support. Other modular
assets remain unknown until their file identities are reviewed. Executable, DLL
and manifest signatures invalidate the cache; reads are capped at 128 MiB per
binary and 64 KiB for the receipt. No helper process is started.
When support is missing, RF omits this option before launch. Only an explicit unsupported native override
warns once per runtime identity; an ordinary dormant default does not warn. An
unsupported explicit occurrence is retained in saved arguments and omitted
from the effective launch, so it cannot cause an unknown-flag startup failure.
Other native arguments retain their order and values.

New and existing profiles without these keys acquire
`llama_moe_cache_cpu_share_enabled: true` and
`llama_moe_cache_cpu_share: 0.5` through existing defaults migration. Explicit
saved false/zero values and unrelated preferences survive. There is no
established equivalent for the current xllamacpp binding; its native preference
remains saved and inactive rather than being mapped to an unrelated control.

Implementation owners are
[`moe_cache.py`](../../ruinedfooocus/backends/llama/moe_cache.py),
[`llama_server.py`](../../ruinedfooocus/backends/llama/llama_server.py), the pure
settings schema/defaults, and Chat's existing load identity/status. UI navigation
can read `moe_cache_cpu_share_capability(settings)` without an executable; actual
initialized status is `pipeline.moe_cache_cpu_share`. The public state omits
private native argument tokens.

`pipeline.runtime_snapshot.moe_cache` adds `applied` and `observed` to the saved
request/support fields. Startup can establish that a flag was passed, but leaves
full-cache CPU miss execution unobserved. `active` is false until such execution
has evidence; no startup-only claim or hidden tuning workload supplies it.
The [load evidence contract](native-llama-load.md) describes placement provenance.

The Settings controls are owned by
[`ui_moe_cache.py`](../../ruinedfooocus/features/chat/ui_moe_cache.py), immediately
after Chat GPU layers. A native checkbox preserves the requested enabled state;
an advanced native number field stores the 0–1 share. The status-only callback
shows form versus saved values, native argument/environment precedence and the
existing service pipeline's initialized public state. Navigation, form edits and
**Refresh MoE status** perform no runtime construction, binary scan, model load
or device probe. Saving updates the profile; Chat reload applies the preference.
An unsupported preference remains editable and saved, with its inactive reason.
A native page-load callback restores only these two fields from the saved
profile when a fresh browser opens the existing app. Tab selection, status
refresh and saving a profile update status without replacing current form drafts.

## Official source evidence

Checked 2026-11-11 against official upstream source:

The exact b12327 parser/source review was refreshed 2026-11-12 at
[`81f21f4b`](https://github.com/ggml-org/llama.cpp/blob/81f21f4b27fc8252f11404a006744045aac5ea6e/common/arg.cpp).

- [Option implementation e95426f1](https://github.com/ggml-org/llama.cpp/commit/e95426f16982054c379ca7bdc8186996dd22f0fe)
  introduced the numeric parser option on 2026-11-07.
- [Pinned b12110 parser](https://github.com/ggml-org/llama.cpp/blob/b12110/common/arg.cpp)
  lacks the CPU-share option.
- [Current parser at 2cbe0ec4](https://github.com/ggml-org/llama.cpp/blob/2cbe0ec429b849a88664e0f62388fc92a5abd699/common/arg.cpp)
  accepts values from 0 to 1; native default is 0.
- [MoE cache guide at 2cbe0ec4](https://github.com/ggml-org/llama.cpp/blob/2cbe0ec429b849a88664e0f62388fc92a5abd699/docs/moe-cache.md)
  explains full-cache miss behavior and machine-specific tuning.
- [Latest stable v0.9.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.9.0)
  was published 2026-11-04, before the option's introduction. This source evidence
  does not establish device or performance qualification.

```sh
python tools/check_cpu.py test_llama_moe_cache_share.py test_llama_server.py test_llama_runtime_policy.py test_settings_store.py test_reasoning_settings.py test_chat_vision_modes.py test_moe_cache_settings_ui.py
```

These tests use passive executable fixtures, fake process launch, captured load
identities and isolated profiles. No actual executable, model, GPU probe or tuner
is run. Real hardware performance and current pin activation remain unverified.
