# Native hook CLIP conditioning

**Scheduled LoRA** supports the external CLIP targets of the current native still
image families. The existing named source, strength, keyframe and **Apply keyframes
to CLIP encoding** controls serve both ordinary and reference encoders. API fields,
metadata and explicit full-setup replay use the same `lora_hook_*` values. **Off**
and constant CLIP retain the original single-encode path. File-specific native
weight mapping remains mandatory; a family capability does not certify an arbitrary
LoRA or checkpoint source.

Reference file labels resolve translated text before adding a filename, so
PhotoMaker compatibility and unavailable-file labels stay readable with native
Gradio translation metadata enabled. Refresh uses the requesting browser's locale.

The following callers were inspected at ComfyUI
`fca60fe70e5b84c5c6b058263653d15cb3401e44` on 2026-11-11:

| Conditioning path                                                                                         | Native contract and application handling                                                                                                                                                       |
| --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ordinary still text, including SD3, Flux, Anima, Hunyuan, Lumina, PixArt and modern multimodal CLIP types | `CLIPTextEncode.encode` returns `CLIP.encode_from_tokens_scheduled` unchanged; `pipeline.textencode` retains the entire list                                                                   |
| SDXL refiner and Cascade C                                                                                | Native scheduled CLIP output and geometry/aesthetics metadata are preserved; hooks target the primary checkpoint                                                                               |
| Qwen edit/Plus and Qwen Image 2.1                                                                         | Native `TextEncodeQwenImageEdit`, `TextEncodeQwenImageEditPlus` and `TextEncodeQwenImage21` retain every scheduled entry; ordered-reference and paired T2I helpers preserve those lists        |
| MOSS, Mage, Fibo2, Joy and Boogu reference nodes                                                          | Native scheduled text calls retain every entry while adding reference latents, attention bias or text-image intervals; these metadata additions do not select the first entry                  |
| StyleAdapter and Redux                                                                                    | Native reference application preserves incoming scheduled metadata while modifying the conditioning                                                                                            |
| PhotoMaker                                                                                                | Native `PhotoMakerEncode.execute` directly encodes CLIP. The application invokes that complete identity node once per native hook interval and retains both pooled output and prompt intervals |
| HiDream O1 and SenseNova U1.5                                                                             | Their application component contracts have no external CLIP. Nonzero CLIP hooks are refused; model-only hooks remain available                                                                 |
| GLIGEN box text                                                                                           | Each box uses the direct native pooled encoder for matching CLIP intervals; pooled-width validation, input ranges, order and geometry are preserved                                            |

`CLIP_FAMILIES` lists the reviewed current component families explicitly. A new
catalogue family requires source review before it gains this capability. Existing
aliases resolve to their canonical family. The primary-only boundary also applies
to checkpoint-as-LoRA and model-only hooks: a Cascade Stage B or a separately loaded
refiner/negative model does not inherit the primary model's hooks.

## Direct native encoders

`features/image_editing/clip_hook_conditioning.py:direct_conditioning` obtains
intervals and the selected keyframes from native
`HookGroup.get_hooks_for_clip_schedule`. Each interval receives a non-dynamic
request CLIP clone, the same native hook references and original weight tensors,
and fresh native constant keyframe groups at that interval's selected multipliers.
Before a delayed first start the selected CLIP multiplier is the native default 1.
No custom interpolation or application-side weight blending is performed.

Native `Hook.clone` shares its keyframe group. The adapter replaces that group on
the interval clone before selecting a strength, preserving the model's independent
keyframe state. It executes the original native conditioning node, records
`clip_start_percent`/`clip_end_percent`, and cleans native hooks in `finally` after
each encode. The outer request scope also tracks these descendants and restores
the original owner/cache after completion, error or cancellation. Source read
leases remain held by that scope throughout encoding, sampling and decode.

For conditioning transforms, incoming CLIP and prompt intervals are intersected
before calling the native node. Nonoverlapping entries are omitted, and existing
masks, controls and other metadata retain their object identity. This bounds
composition to compatible intervals instead of multiplying unrelated entries.
PhotoMaker caches a prompt's full interval list within the request and adds its
existing prompt-switch metadata afterward. Negative text conditioning remains on
the ordinary native path.

## Finite hook work order and limits

All 13 assigned primitives have production callers: the four LoRA/checkpoint
constructors, three keyframe generators, `SetHookKeyframes`, `SetClipHooks`,
`SetModelHooksOnCond.attach_hook`, and `CombineHooks2/4/8`. Their exact mapping is
maintained in [combined hook sources](native-hook-sources.md). The model attachment
helper is callable in the pin but absent from its registered node list. Optional
empty combine inputs and an absent keyframe group need no separate UI: the named
application controls reject empty sources and attach requested targets explicitly.

Native state-dictionary readers do not accept GGUF source containers. The pinned
packed GGUF target patcher also lacks a native packed hook updater; those targets
are rejected before encoding, while static GGUF LoRAs remain available.

Guarded CPU tests cover native HookGroup/keyframe and CLIP/PhotoMaker/GLIGEN method
bodies, actual PhotoMaker, Qwen and GLIGEN callers, public API capture and metadata,
real Gradio controls, source leases and cleanup/cancel/retry. GLIGEN's public API
capture reaches its actual native box caller within the owned hook scope. No
real weights, rendered browser, image quality, inference or GPU
qualification is established here.
