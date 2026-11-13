# Native Chat load identity and placement

Native Chat retains llama.cpp's measured `auto` GPU fitting for RF's layer value
`-1`. Values at or below `-2` request native `all`; nonnegative values are manual
layer allocations. Native command-line overrides retain their order and take
precedence over RF defaults. Explicit CPU authority is appended after those
overrides and disables device, layer, KV, operation and projector offload.

RF always captures its context choice. A positive context fixes that native
budget; `0` requests the full model context and disables native context reduction.
There is no new adaptive context mode or silent profile migration. Native fitting
accounts for model, context/KV, compute and available device memory; RF does not
divide file size by VRAM or force every layer onto the device. The existing shared
compute admission and service handoff reclaim inactive resident service weights
before Chat initialization. Direct pipeline initialization requires that owning
admission. Active user Chat and other owners are never reclaimed by a parallel
load or a navigation/status callback.

`llama_server.Server` accepts optional `load_id`, `progress_callback(event)` and
`check_cancelled()` arguments. The checker raises `concurrent.futures.CancelledError`.
`pipeline.load_base_model` exposes the same contract with callback name
`runtime_progress`. Callbacks receive only bounded public fields:

```text
{load_id, phase, runtime?, transfer?, message?, result?}
```

Phases are `resolving_runtime`, `downloading_runtime`, `verifying_runtime`,
`staging_runtime`, `probing_device`, `loading_model`, `ready`, `cancelled` and
`failed`. Runtime acquisition reuses the installer's verified atomic publication.
Device probes use the existing bounded owned-child runner. Health polling checks
cancellation before every wait and before Ready; cancellation closes only that
load's child, session and process-tree handle. A superseded load cannot publish
Ready or overwrite its newer owner's terminal state. Offline/install-skip
restrictions still reject acquisition. Failures and retries use new request IDs.

The initialized `runtime_snapshot` separates model/projector file signatures,
verified artifact identity when already available, executable SHA-256 and install
manifest identity, requested memory controls, effective context, observed
placement and MoE state. Custom `LLAMA_SERVER` paths retain their executable
identity but cannot inherit managed source qualification from a nearby manifest.
Reasoning/tool bindings use the initialized runtime build, rather than assuming
that the configured pin was the executable actually loaded.

`runtime_status()` returns copied load/initialized snapshots without probing.
The pure `model_preparation.runtime_selection_key` compares captured load-affecting
settings and model/projector file signatures; callers resolving an Auto vision
projector pass its path explicitly. Replacing the same projector path invalidates
the key. `initialized_runtime_status()` reads the resident service after terminal
cleanup so a cancelled replacement cannot leave an old Loaded badge behind.

Placement evidence comes only from the owned child's startup log, with a 2 MiB
read cap, 16 KiB line cap and 256-row limits. Reviewed managed sources enable
native JSONL logging. Native fit rows are labelled estimates and cannot establish
initialized placement. Actual layer/model/KV/compute buffer fields are separate;
all layers alone never prove full GPU residency. Host expert tensor evidence can
disprove it. Missing, malformed, truncated or insufficient evidence stays unknown.

For the reviewed b12110/b12327/b12366 LFM3 and GPT-OSS input-layer rule, bounded GGUF
descriptors can explain the mandatory host input tensor. Full offload requires
all offloadable layers plus device model buffers and either zero host model bytes
or host bytes matching that input tensor within native log rounding/alignment.
Full placement also requires the native model-loaded log fence after its buffer
rows; partial startup logging cannot establish it.
Host bytes exceeding that complete input allowance establish mixed placement;
other architectures or tensor overrides retain unknown attribution. Native host
input/metadata and host compute buffers do not mean every weight runs on the CPU,
and full offload does not promise zero host RAM. The snapshot names explicit
allocation versus native automatic fit; neither load duration nor backend labels
prove placement, throughput or stability.

Primary native contracts, checked 2026-11-12, are
[b12110 measured fitting](https://github.com/ggml-org/llama.cpp/blob/b12110/common/fit.cpp),
[b12110 JSONL schema](https://github.com/ggml-org/llama.cpp/blob/b12110/common/log.cpp),
and [b12327 input/buffer placement](https://github.com/ggml-org/llama.cpp/blob/81f21f4b27fc8252f11404a006744045aac5ea6e/src/llama-model.cpp).
The [exact native tensor override log](https://github.com/ggml-org/llama.cpp/blob/81f21f4b27fc8252f11404a006744045aac5ea6e/src/llama-model-loader.cpp#L1246)
provides positive host-expert evidence when debug logging is explicitly enabled.
The [exact source audit](native-llama-source-audit.md),
[SDK acquisition contract](native-llama-sdk.md) and
[MoE cache contract](llama-moe-cache.md) retain their separate scope.

```sh
python tools/check_cpu.py test_llama_runtime_events.py test_llama_runtime_policy.py test_llama_server.py test_llama_moe_cache_share.py test_chat_runtime_load.py test_chat_preparation.py test_chat_vision_modes.py test_chat_boundaries.py test_model_profiles.py
```

These fixtures mock all native execution. Browser phases, real model placement,
generation quality, latency and GPU stability require their own source-bound
evidence.
