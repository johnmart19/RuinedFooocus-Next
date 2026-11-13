# Native image stage companions

Stable Cascade Stage B is a companion for a Stage C prior. In Create, select
Stage C as the **Image model**, then select the bundled Stage B file in
**Stable Cascade · Stage B**. Stage B remains installed and available in that
selector. Models Hub presents its registered artifact as a component. It does
not offer Stage B as an independent Create checkpoint.

The same rule applies to `/v1/models`: Stage B is omitted from primary image
generation IDs. A saved Stage B primary choice is cleared from the active
Create selector, including after a metadata refresh. The saved preference is
preserved; choose a Stage C primary to continue. An API `default` resolving to
Stage B fails with the normal instruction to choose an image ID from
`/v1/models`. Requests still pass the existing explicit admission check that
requires Stage C primary and Stage B companion.

This role follows the pinned native model contract rather than filenames or
publisher captions. `models/stage_companions.py` uses the existing bounded
passive native detector and `BASE.matches` against `Stable_Cascade_B`. Both full
and lite headers qualify; unrelated and unknown headers remain selectable as
before. The current model metadata carries an ephemeral file-signature-bound
`image_stage_companion` role. Replaced files and changed native predicate source
invalidate the passive cache. Stale role fields from publisher/cache JSON are
discarded. The historic registry annotation remains unchanged; the catalogue's
current operation projection supplies the companion presentation.

At ComfyUI `fca60fe70e5b84c5c6b058263653d15cb3401e44`, native
`StableCascade_StageB_Conditioning.execute` attaches `stable_cascade_prior`.
`StableCascade_B.extra_conds` passes it as `effnet` to Stage B. Without that
input the native implementation constructs a zero prior. That fallback is not
evidence for a separately qualified text-to-image default, and this feature
does not promote it into one. The existing paired Cascade route supplies the
actual C prior and samples the matching native B grid before Stage A decoding.

Source pointers, inspected 2026-11-11:
[native stage detection](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy/model_detection.py),
[Stage B conditioning](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy_extras/nodes_stable_cascade.py),
and [Stage B runtime conditions](https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy/model_base.py).

Feature `native-cascade-companion-role` checks renamed/full/lite/unknown headers,
stale metadata and immutable registry annotations, the actual primary selector
and both refresh callers, the companion selector, actionable request admission,
public API listing/default resolution, and native prior/zero-fallback bodies.
A fresh-process import guard verifies passive role inspection does not import
Torch or Comfy. These are source and guarded CPU checks; no model weights are
executed, and rendering, device and image-quality acceptance remain separate.
