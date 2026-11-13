# Native LoRA hook schedules

In **Create → Image tools → Scheduled LoRA**, choose manual, interpolated or
strength-list keyframes, a local source, and its signed model/CLIP strengths. **Off** is the
default. It performs no LoRA inventory scan, file preparation, native hook loading
or additional encoding. Enabling the panel lists local state dictionaries; safe
native loading and actual target mapping happen on Run.

This adds scheduled sources after the existing card and prompt LoRAs. It does
not change their add, prompt override, replacement or removal behavior. Selecting
the same file in both places deliberately adds the scheduled effect to the
static effect. The prompt remains literal. Model-only hooks work with native
still-image diffusion families when the LoRA maps to the primary model's weights.
CLIP hooks use the reviewed external text encoders, including modern ordinary
and reference callers. PhotoMaker's direct native identity encoder runs through
the native interval adapter. Families without an external CLIP retain model-only
hooks. GLIGEN box text uses the same native interval adapter. See
[CLIP conditioning](native-hook-clip-conditioning.md) for the exact caller contracts.

The primary source can be LoRA weights or a checkpoint as LoRA. An additional
named JSON list can bring the total to eight sources, each with its own strengths
and keyframes. **Copy selected LoRAs into scheduled sources** copies card names
and weights without changing the cards or prompt. See the
[finite source/node contract](native-hook-sources.md) for fields and native calls.

Manual keyframes are a bounded JSON list, for example:

```json
[
  { "start": 0, "strength": 1 },
  { "start": 0.5, "strength": 0 }
]
```

The first manual start is 0. Starts must increase, with at most 100 entries;
multipliers and base strengths accept -20 through 20. Interpolated mode uses the
native linear, ease-in, ease-out or ease-in-out generator with 2–100 keyframes,
0–10 endpoint multipliers and distinct start/end percentages. Percentages refer
to the native full denoising schedule, rather than a fraction of the remaining
steps. Each generated keyframe holds its multiplier until the next start. The
native model scheduler guarantees one step for each manual keyframe; interpolated
keyframes guarantee one step only for the first entry. Denoise, sampling steps and
the native sigma schedule determine which keyframes execute.

With **Apply keyframes to CLIP encoding** off, CLIP uses its constant signed base
strength. With it on, the native encoder creates conditioning for each keyframe
interval. Before a delayed first interpolated start, the model uses its first
multiplier while CLIP uses its base strength. The last multiplier continues to
the end. Prompt intervals, masks and existing ControlNet metadata are retained.

## Capture, native calls and cleanup

`features/image_editing/lora_hooks.py` owns pure bounded settings, immutable
`HookPlan` capture and request scope; `ui_lora_hooks.py` mounts the controls.
`generation/image_preparation.py` captures the exact checkpoint and local LoRA
file identity before enqueueing. Only configured local native state dictionaries
are admitted. Native Torch containers use `safe_load=True`; GGUF adapter files
need a separate scheduled reader and are not admitted here.

At pinned ComfyUI `fca60fe70e5b84c5c6b058263653d15cb3401e44`, execution calls
`CreateHookLora` or `CreateHookLoraModelOnly`, `CreateHookKeyframe` or
`CreateHookKeyframesInterpolated`, `SetHookKeyframes`, optional `SetClipHooks`,
and `SetModelHooksOnCond`. Native `HookGroup`/`HookKeyframeGroup` own keyframe and
sigma behavior. This route does not interpolate static LoRA weights itself.
The native sampler supplies its own non-dynamic delegate when conditioning uses
hooks. Exact native registered patch counts must be nonzero for each requested
nonzero target; an unmodified model is an error.

The request holds its prepared LoRA read lease through encoding, sampling and
decode. It temporarily exposes a cloned primary UNet and optional native CLIP
clone through `pipeline.xl_base_patched`, isolates prompt-conditioning caches,
tracks descendant patchers through native clone callbacks, and cleans hooks in
`finally` before restoring the original model and cache. Stop, encode failure and
retry use the same cleanup path. A changed file or checkpoint is rejected.
Hooks target the primary checkpoint, matching the existing static LoRA boundary;
they do not automatically patch a Cascade Stage B or separate negative model.

The pinned GGUF patcher implements static packed-weight patches but has no packed
hook update adapter. Actual matched packed GGML/custom ConvRot weights without a
native setter/converter are rejected before encoding or sampling. Floating-point
GGUF targets are not excluded by filename. Packed scheduled weights remain an
unsupported by that pinned native patcher, rather than a hardware incompatibility.
Static GGUF LoRAs remain available. Combined sources and float-list/model-as-LoRA
variants use the same request boundary; specialized CLIP composition remains
separately qualified.

## API, saved settings and Library

`ImageRequest` accepts the `lora_hook_*` fields. `/capabilities` exposes
family-level model/CLIP availability; file-specific native mapping still decides
compatibility. The same pure validation serves UI saves, API calls and metadata
replay. Defaults and output metadata store names, strengths and keyframes;
private hook objects and captured plans are never serialized. Diagnostic
`native_lora_hooks` records source signatures, actual patch counts, target and
effective native keyframes. An explicitly supplied API `lora_hook_mode="off"`
overrides the captured Create defaults.

Plain Library prompt/edit/inpaint actions reset the schedule to Off while
preserving selected source pixels, alpha, denoise, size and ordinary LoRAs.
**Reuse full setup** explicitly restores saved schedule fields. Subsequent Run
captures the current local file identities again. Settings saves and full setup
replay never attach native hooks themselves.

## Validation scope

Guarded CPU tests execute the pinned native node/keyframe and scheduled-CLIP
method bodies, actual registered Gradio callbacks, public ASGI requests, Library
callers and request cleanup/cancel/retry behavior. Source signatures and the
declared prerequisite replay receipt bind that evidence to the tested commit.
No model inference, rendered-browser behavior, image quality or device/GPU
qualification is established by these CPU fixtures. Native source inspected
2026-11-11; no dependency installation or downloads are needed for the tests.
