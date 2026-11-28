# Combined native hook sources

**Scheduled LoRA** accepts a primary local source plus **Additional scheduled
sources JSON**, up to eight sources in total. Leave the primary filename empty
to use only the list. Every source uses a configured relative filename, a
nonzero model or CLIP strength and bounded keyframes. The additional list inherits
the controls above unless it explicitly overrides a named setting:

```json
[
  {
    "type": "lora",
    "name": "detail.safetensors",
    "model_strength": 0.75,
    "clip_strength": 0,
    "mode": "keyframes",
    "keyframes": [
      { "start": 0, "strength": 1 },
      { "start": 0.6, "strength": 0 }
    ]
  },
  {
    "type": "checkpoint",
    "name": "alternative.safetensors",
    "model_strength": 0.25,
    "clip_strength": 0,
    "mode": "floats",
    "strengths": [0, 1, 0],
    "start": 0,
    "end": 1
  }
]
```

Other optional per-source keys are `interpolation`, `strength_start`,
`strength_end` and `count`, with the same bounds as the interpolated controls.
Unknown keys, source paths outside the configured folders, absent files, invalid
keyframes and more than eight total sources are rejected before compute.
The strength list accepts 1–100 finite numbers from -20 to 20. Native
`CreateHookKeyframesFromFloats` spaces them evenly between start/end; a one-value
list uses the first start only. Its first keyframe guarantees one native step,
and later entries guarantee zero. This is a discrete native schedule.

**Copy selected LoRAs into scheduled sources** copies the existing card order,
filenames and weights to the list, clears the primary scheduled filename and
enables manual keyframes if needed. It leaves the cards and prompt unchanged.
The copied schedule is additional to those static LoRAs. Edit the copied JSON if
different strengths or per-source schedules are desired.

## Native node mapping

The source is `features/image_editing/lora_hooks.py` plus the pure bounded
`lora_hook_sources.py`. These are calls into pinned ComfyUI
`fca60fe70e5b84c5c6b058263653d15cb3401e44`, rather than an application weight
interpolator or graph editor.

| Native primitive                  | Actual application route                                                                                                                       |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `CreateHookLora`                  | Local LoRA source with a nonzero CLIP target, `_build_source`                                                                                  |
| `CreateHookLoraModelOnly`         | LoRA source with CLIP strength 0; inherited native raw-weight cache is request-local                                                           |
| `CreateHookModelAsLora`           | Checkpoint source with model/CLIP targets; native `model_as_lora` weight differences                                                           |
| `CreateHookModelAsLoraModelOnly`  | Checkpoint source with CLIP strength 0; same native loader and difference contract                                                             |
| `CreateHookKeyframe`              | Ordered manual start/multiplier pairs, including negative multipliers                                                                          |
| `CreateHookKeyframesInterpolated` | Native four-mode interpolation and exact generated endpoints                                                                                   |
| `CreateHookKeyframesFromFloats`   | Bounded strength list; one value represents the native scalar case                                                                             |
| `SetHookKeyframes`                | Assigns each source's own keyframe group before combination                                                                                    |
| `CombineHooks2`                   | Two nonempty source groups                                                                                                                     |
| `CombineHooks4`                   | Three or four groups; unused native optional inputs remain `None`                                                                              |
| `CombineHooks8`                   | Five through eight groups                                                                                                                      |
| `SetClipHooks`                    | Only sources with nonzero CLIP strength; native scheduled or constant CLIP mode                                                                |
| `SetModelHooksOnCond.attach_hook` | Only the nonzero model group, appended to final positive/negative conditions; this callable helper is absent from the native registration list |

All listed primitives have an application caller. No separate control is needed
for empty combine inputs, an absent keyframe group or automatic model attachment
from CLIP: the application refuses empty/inert source lists and attaches the
requested model group explicitly. Existing conditions, prompt intervals,
ControlNets and masks survive attachment. CLIP-only sources avoid a diffusion
clone or model-hook conditioning; model-only schedules add no CLIP intervals.

The primary checkpoint remains the target. Cascade Stage B and a separate
negative model do not inherit these patches. Each nonzero target of every
source must map native weights; one matching source cannot hide an inert second
source. Checkpoint-as-LoRA keys must also match the actual target tensor shapes.
Missing requested checkpoint CLIP/model weights produce an admission error.

## Sources and ownership

Checkpoint sources use native `load_checkpoint_guess_config` with its bundled
components, followed by `get_patch_weights_from_model`. The native helper omits
`model_sampling` weights and retains source tensors without copying or changing
the original file. The native model-as-LoRA formula uses the source-minus-baseline
weight difference. External component overrides of the generation target do not
invent missing components for this separate source.

The app captures every exact local source and holds read leases through the full
request scope. Duplicate canonical paths share a lease, preventing a nested
reader from waiting behind a queued writer. Each duplicate source also shares a
request-local native loader's raw weights; hook groups and keyframes remain
independent. Native clone callbacks, hook cleanup and original owner/cache
restoration cover cancellation, failed mapping and retry. Off captures no source
files, imports no native hook runtime and allocates no source weights.

Preparation preflights every captured local file before starting any transfer.
A removed captured identity is refused even when its catalogue entry has a URL;
it cannot become a replacement download. Assets that were already missing at
capture retain their existing explicit first-use acquisition behavior.

GGUF LoRA/checkpoint source files are not accepted by the native state-dictionary
hook readers. Packed GGUF target updates are unsupported by the pinned GGUF
patcher, which implements static patch updates without a corresponding packed
hook update method. Static GGUF LoRAs and native floating-point targets remain
available. Reviewed external CLIP families, direct PhotoMaker encoding and GLIGEN
box text use the [CLIP conditioning adapter](native-hook-clip-conditioning.md).
Off and constant CLIP retain their original single-call paths.

UI defaults, public API fields, output metadata and explicit full-setup replay
carry the named source list and strength generator. Plain Library actions reset
the schedule to Off. Captured plans and executable hooks are not exported.
Guarded CPU tests cover native bodies, real Gallery/control preprocessing,
public API capture, file leases and restoration. Real artifacts, rendered browser
behavior and device/image quality remain unqualified here.
