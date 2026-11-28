# Native image utility devices

`features/image_processing/utility_device.py` owns worker-time placement for the
request-local Transform, Mask, Color, Face and MoGe operations. Capture, metadata browsing
and UI construction do not call its resolver or initialize a device. Normal
startup keeps Auto image processing, Auto Chat and Auto background removal; a
private CPU test profile does not select or rewrite the normal profile.

`run_utility(feature, operation, check_cancelled=..., cpu_only_reason=...,
native_model_placement=False)` calls
the operation with an immutable `UtilityPolicy`. Applied CLI device choices take
priority over environment and saved settings; an explicit CPU request wins for
that scope. Automatic native startup CPU recovery retains its Auto origin. The
resolver otherwise uses ComfyUI's `get_torch_device()` and
`intermediate_device()` at the actual worker job. It does not enumerate devices,
choose a separate GPU or change startup flags. There is no Torch Vulkan or
Windows ML implementation in this helper; Chat and ONNX retain their own native
runtime policies.

Pixel work uses float32, including its native intermediate dtype. Generation's
optional fp16 intermediates do not lower original pixel, HDR or alpha precision.
The callback creates fresh tensors from the captured source on `policy.device`,
uses the native algorithm and validates and transfers the result to the host
inside that attempt. This preserves the existing float32 and uint16 image
contracts and catches asynchronous device failures during output transfer.

An unsupported native device operation or accelerator runtime/memory failure
logs its actual reason through `core/device_fallback.py` and retries once with
fresh CPU tensors. CPU recovery never persists a setting or changes shared
global backend state. Native-model callbacks clear only their failed owned model
cache before rebuilding. Cancellation is checked before selection, after the attempt
and before recovery. Source validation, ordinary filesystem errors and failed
CPU operations propagate rather than silently retrying. Callers publish their
pixels and `image_utility_source.compute` receipt only after successful work.

Host-only native capabilities pass an explicit reason and do not probe a GPU or
claim an accelerator failure. Examples are the pinned GrowMask NumPy/SciPy
implementation and Pillow-based text or palette operations. Supported Torch and
Kornia operations use the selected native device. The receipt records requested
policy, actual device/backend, intermediate device/dtype, fallback and its
reason, and any host-only capability reason. `native-torch` names the selected
Torch device without claiming CUDA rather than ROCm from the shared `cuda`
device type.

Auto jobs also log a terminal explanation for an intentional host-only capability
or native model CPU policy. Their metadata retains `cpu_only_reason` with
`fallback=false`; a failed accelerator attempt instead retains its actual
`fallback_reason`. Explicit CPU requests remain silent. Browsing and capture do
not resolve or report devices.

MoGe and Face pass `native_model_placement=True`. This preserves native
`text_encoder_device()` placement, including LOW/NO VRAM capacity and forced FP32
precision decisions. Explicit CPU and a supported failure retry remain authoritative
for fresh native wrapper/patcher load devices; native offload devices, inference
dtypes and loading APIs remain intact. The effective device is part of the model
cache identity, and native inference, rendering, validation and host synchronization
run inside the callback.

Before any model CPU attempt, the wrapper checks native
`text_encoder_offload_device()`. An accelerator offload target, such as explicit
GPU-only mode, rejects CPU recovery before construction or a fallback warning.
The error directs the user to disable GPU-only/use Auto or launch with `--cpu`;
it cannot publish successful CPU recovery. Normal Auto with CPU offloading can
rebuild and retry once. No global device override or alternate constructor is used.

Focused policy check: `python tools/check_cpu.py test_utility_device_policy.py`.
It uses CPU tensors and fake native device decisions, rejects GPU probes,
verifies float32 intermediates, explicit CPU and CLI precedence, host-only
capabilities, selective recovery and cancellation. The paired geometry/mask and
color callers have their own native CPU checks. These tests do not qualify real
GPU operator coverage, image quality, latency or stability; actual device
acceptance remains separate.
