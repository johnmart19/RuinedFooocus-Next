# AMD precision, isolation and tuning controls

Research dated 2026-11-09 against ROCm 10.1.0 documentation; precision audit at
RF `3174e80`, with device-mask source follow-up at `6b15d9c` and installer-child
integration at `ad085f2`. ComfyUI remains pinned to
`989d93d0d968e704c589ce0146bb18c77f6b50cc`; the
[0.45 adoption contract](comfy045-adoption.md) records the unchanged native
device/memory policies and separate qualification limits.
These are applicability rules, not settings newly enabled by RF. Leave native
defaults unless an observed problem and a bounded comparison justify an override.

## Match guidance to the actual hardware

The [RDNA3.5 optimization guide](https://rocm.docs.amd.com/en/docs-10.1.0/reference/system-optimization/rdna3-5.html)
addresses Ryzen APUs gfx1150/1151/1152 and Linux unified-memory/TTM/GTT setup.
Its BIOS carveout and kernel settings are not discrete RX 7900 XTX/gfx1100
Windows tuning. 128 GB system RAM does not become equivalent dedicated VRAM.

The [RDNA2 guide](https://rocm.docs.amd.com/en/docs-10.1.0/reference/system-optimization/rdna2.html)
describes Radeon PRO V620 Linux SR-IOV virtualization. It does not establish
RX 7900 XTX passthrough, partitioning or Windows/WSL optimizations. Apply neither
guide's system changes indiscriminately across AMD devices.

## Floating point and atomics

Separate four contracts: **weight storage**, **kernel input/arithmetic**,
**accumulation**, and **output precision**. An INT8/FP8 checkpoint can use unpacking
or emulation; its file format alone proves neither native matrix support nor
better quality/performance.

AMD's [precision table](https://rocm.docs.amd.com/en/docs-10.1.0/reference/precision-support.html)
lists RDNA3 matrix support for FP16/BF16, not TF32 or FP8/FP6/FP4. CU/type support
is a different table, and some library support can include emulation. Composable
Kernel's cross-architecture list is not a promise for every gfx1100 operation.
Preserve ComfyUI's device-aware dtype/accumulation/attention dispatch. Its
`--supports-fp8-compute` assumes support; do not silently enable it for this GPU.

The [gfx11 atomics table](https://rocm.docs.amd.com/en/docs-10.1.0/reference/gpu-atomics-operation.html)
requires memory location, coherence and operation scope. Coarse device-DRAM,
device-scope FP32 add/min/max is native; FP64 and packed half2/BF16x2 add use
CAS emulation. Some fine-grained/uncached floating paths can produce incorrect
results when the native operation is unsupported. Do not globally force host
coherence or unsafe floating atomics based on a broad “supported” label.

RF currently needs no custom atomic kernel. HIP programming/atomics become
implementation concerns only for an identified native operation; retain
PyTorch/ComfyUI/ORT allocation and synchronization contracts. The
[HIP overview](https://rocm.docs.amd.com/en/docs-10.1.0/reference/hip-programming.html)
does not make HIP a drop-in CUDA replacement or justify adding `hip-python`.

## Environment and device selection

| Control                | Contract                                                                                  |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| `ROCR_VISIBLE_DEVICES` | AMD's Linux recommendation; indices/UUIDs in that runtime                                 |
| `HIP_VISIBLE_DEVICES`  | AMD's Windows recommendation; HIP indices                                                 |
| `CUDA_VISIBLE_DEVICES` | Also affects HIP; cannot safely serve as a universal independent AMD/NVIDIA worker choice |
| `GPU_DEVICE_ORDINAL`   | Another ROCclr OpenCL/HIP visibility layer, not DXGI, Vulkan or ONNX identity             |

Use [environment-variable guidance](https://rocm.docs.amd.com/en/docs-10.1.0/reference/environment-variables/index.html)
and [isolation semantics](https://rocm.docs.amd.com/en/docs-10.1.0/reference/system-optimization/gpu-isolation.html).
Visibility masks are not security isolation. Compose an environment for the
owning worker before runtime initialization; do not change machine-wide values
or mutate shared process state under active jobs. Preserve explicit inherited
values and report conflicts with requested settings. WSL uses guest identities.

`comfy_compat.configure_device_environment` now applies numeric image selectors
only to the installed/requested native CUDA or HIP backend: it writes
`CUDA_VISIBLE_DEVICES` for CUDA or `HIP_VISIBLE_DEVICES` for HIP. An explicit
oneAPI selector writes `ONEAPI_DEVICE_SELECTOR`. It does not copy the same index
list into unrelated Ascend, Vulkan or other backend masks. Explicit image CPU
returns before device probes or mask writes.

Before mutation, startup records inherited, requested and effective values in
`args._device_environment_status` and the compute-status backend report. A
conflicting inherited value for a requested key fails without changing live
masks; HIP additionally checks inherited `ROCR_VISIBLE_DEVICES` and
`GPU_DEVICE_ORDINAL` against its request. These are the implemented conflict
checks, not a universal translation between every runtime's device namespaces.
Applying numeric/oneAPI selection after native initialization requires a restart.
The effective-mask report describes the startup environment, not proof of the
initialized adapter, actual workload placement or acceleration.

Default-device enumeration uses a bounded child process, clearing only that
child's relevant native masks; the live parent's masks remain intact. Ambiguous
multi-NVIDIA ordering requires `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Independent
Chat/ONNX installer children compose their default environment through
`child_device_environment`: in a copied
environment it restores only masks proven to have been introduced by RF image
startup to their inherited values, or removes them if originally absent. Other
inherited restrictions remain; an explicit caller-supplied child environment
stays caller-owned. A later change to a recorded startup mask refuses
child launch and requires explicit restart/verification instead of silently
discarding that change. This establishes scoped startup/child behavior, not
independent concurrent execution of arbitrary mixed-vendor workloads.

Keep logging/code dumps/launch serialization opt-in. `HIP_LAUNCH_BLOCKING` and
`AMD_SERIALIZE_KERNEL`/`AMD_SERIALIZE_COPY` are diagnostic controls, not default
speedups or proven driver fixes. The documented `GPU_MAX_HW_QUEUES` default is
4; larger values can give unpredictable performance. Do not rewrite memory-pool,
coherence or VM flags merely because they appear in an optimization list.

Current RF startup largely mirrors native ComfyUI, including early experimental
AOTriton setup and explicit-device masks. RF also sets ROCm's
`OCL_SET_SVM_SIZE=4194304` for AIMDO virtual address space; this does not allocate
that amount of physical VRAM. Audit these existing adaptations against the pin
before extending them. Keep the gfx11 wide-head SDPA guard (head dimension over 256) and WSL allocator
distinction until native replacement meets their actual contracts.

At native pin `989d93d0d968e704c589ce0146bb18c77f6b50cc`, those startup and
precision policies are unchanged. Explicit image CPU startup skips device
enumeration and mask writes, and its bounded model-manager import suppresses
XPU/MPS/NPU/MLU discovery before native CPU state is applied. Temporary Torch
availability wrappers and blocked extension entries are restored before service
threads start, including on import failure. This does not disable an independent
Chat/ONNX GPU service or qualify GPU execution. See the
[native caller checks](comfy-platforms.md#external-component-caller-contract).

## Compute-unit masks

[CU masks](https://rocm.docs.amd.com/en/docs-10.1.0/reference/environment-variables/setting-cus.html)
are queue controls, not GPU memory isolation or exclusive resource reservation.
`HSA_CU_MASK` uses `GPU_list:CU_list` sets (for example `0:0-15`); its GPU indices
follow `ROCR_VISIBLE_DEVICES` reordering. `ROC_GLOBAL_CU_MASK` affects HIP/OpenCL
queues. Invalid masks may truncate parsing; WGP mode imposes paired-CU rules.

Leave masks unset. If a developer needs one, require validated syntax/topology,
OS/runtime support and effective-mask evidence, with reset/restart semantics.
Do not advertise generic Windows/WSL support from the reference alone or use
masks to bypass service admission. They do not constrain graphics applications.

RF neither exposes a CU-mask setting nor validates inherited CU masks. Repeated
IDs, a zero mask, malformed sets or an unpaired WGP selection must therefore not
be described as RF-validated configurations. Native parsing can stop at an error
and leave later sets unapplied. A future control needs topology-aware validation
and reset semantics before it can be offered as a supported action.

## Acceptance before changing defaults

Existing fixtures cover Windows/Linux/WSL capability identity, explicit CPU,
mixed-vendor policy, adapter duplication, native index/reordering failures,
the named inherited-mask conflicts, initialized-runtime refusal and independent
installer-child environment preservation. Source-bound evidence is retained at
`6b15d9c` (`test_comfy_runtime.py`, `test_cpu_startup.py`) and `ad085f2`
(`test_installer_child_environment.py` and installer callers); this documentation
repair does not rerun them. These checks do not establish physical workload
placement, conflict attribution across every mask layer, CU/WGP mask validation
or concurrent GPU qualification. CPU-mode validation must not
initialize HIP. Native capability, active operator execution,
model correctness, numerical/visual quality, memory, performance and stability
remain separate evidence. Inspect the precise
[release notes](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html)
for a matching issue; a W7900 ComfyUI performance issue or a resolved Radeon
training crash is not evidence that RX 7900 XTX inference is fixed.
