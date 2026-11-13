# Native YAML recipe contract

`features/models/yaml_recipes.py` parses explicit local YAML data for ordinary
native SD1/SD2/SDXL/SDXL Refiner checkpoints. Request capture, preparation, native
loading, conditioning caches, API, saved defaults and PNG/Library replay share
this contract. The contextual Image tools picker is a separate UI feature;
source/CPU caller evidence does not establish browser or real-model acceptance.

## Pinned native behavior

The inspected source is ComfyUI
[`fca60fe70e5b84c5c6b058263653d15cb3401e44`](https://github.com/comfyanonymous/ComfyUI/tree/fca60fe70e5b84c5c6b058263653d15cb3401e44),
read locally on 2026-11-09. Its deprecated but registered
[`CheckpointLoader`](https://github.com/comfyanonymous/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/nodes.py)
calls
[`comfy.sd.load_checkpoint`](https://github.com/comfyanonymous/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy/sd.py#L2250).
That function first guesses the checkpoint normally, then consumes only:

| YAML field                                        | Native behavior                                          | Normalized helper value            |
| ------------------------------------------------- | -------------------------------------------------------- | ---------------------------------- |
| `model.params.parameterization: v`                | Clone model and add discrete V-prediction sampling patch | `prediction="v_prediction"`        |
| `model.params.parameterization: eps`, or absent   | Leave detected model sampling unchanged                  | `prediction=None`; never force EPS |
| `model.params.cond_stage_config.params.layer_idx` | `clip.clip_layer(layer_idx)` when not null               | Preserve signed integer or null    |

Other prediction values and noninteger CLIP layers are refused. The parser
accepts other plain architecture data, including `target` strings, but never
imports or instantiates it. Those fields cannot select an architecture: the
normal native checkpoint detector remains authoritative. A recipe provides no
steps, CFG, sampler or scheduler and cannot qualify an otherwise unavailable
automatic sampling recipe.

The runtime reuses the native
[`ModelSamplingDiscrete.patch`](https://github.com/comfyanonymous/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy_extras/nodes_model_advanced.py)
operation with `sampling="v_prediction", zsnr=False`, plus native CLIP-layer
selection. It does not copy the deprecated loader or construct classes from YAML.
Caller tests execute the pinned native patch on tiny CPU sampling objects and
prove that V changes denoising while EPS leaves detected prediction unchanged.

## CLIP choice is explicit

Clip Skip needs explicit provenance: a numeric value of `1` does not distinguish
a deliberate user choice from a default. Recipe policy is captured independently
of named sampling presets; YAML never supplies their steps/CFG/sampler/scheduler.

The new helper stores `clip_layer_source`, accepting only:

- `request` (default): preserve the request's Clip Skip behavior, including
  explicit `1` and all legacy requests. The recipe CLIP layer is inactive.
- `recipe`: an explicit opt-in to the signed YAML layer. `recipe_clip_layer`
  exposes it without converting its sign or guessing from slider values.

The UI labels this choice explicitly. A user edit of Clip Skip switches the
source to `request`, including edits to `1`; selecting a PNG preset does the same.
New PNG/default replay retains its explicit recipe path, digest and source policy,
while legacy numeric-only metadata resets to `request` and clears any previous
recipe. API parsing checks field presence before defaults merge: explicit
`clip_skip` keeps request ownership, and simultaneous explicit recipe ownership
is rejected. An API selection of another checkpoint does not inherit the previous
checkpoint's recipe.

Do not persistently apply the YAML layer to the cached base CLIP and then expect
Clip Skip `1` to undo it. Layer selection uses a request CLIP clone; positive,
negative and switched-prompt cache keys include recipe identity, source policy
and effective signed layer. Recipe-to-request transitions retain the pristine
base CLIP, including after LoRA cloning.

## Data and file boundaries

The parser uses the installed PyYAML SafeLoader with additional bounds: 64 KiB
actual UTF-8 bytes, at most 2048 nodes and depth 32. It accepts only ordinary
mapping, sequence and scalar data tags. Anchors, aliases, merge keys, duplicate
keys, nontext mapping keys, custom/Python tags, binary/timestamp/object tags and
multiple documents are refused before an executable object can be constructed.
No new package, dynamic import, download or YAML-specified file access is added.

Capture requires an explicit `.yaml` or `.yml` file plus an existing checkpoint.
It never reads checkpoint tensor bytes or infers a sidecar from its basename.
Discovery keeps recipes separate from weight candidates and defaults to none.
It lists only configured checkpoint/diffusion-model roots, checks resolved
containment before directory traversal and does not parse YAML during filtering.
Merely refreshing files never attaches an adjacent sidecar.

`capture_recipe` freezes source/resolved paths and stat signatures of both files,
the bounded YAML SHA-256, normalized fields, family and CLIP source. Its identity
includes these values. `recipe_read_scope` reuses the existing preparation
service's local read leases, checks cancellation and verifies identity before and
after loading. Missing files, replacements, source-path retargets and changed
YAML bytes refuse a stale capture. No transfer or fallback file is attempted.
These leases coordinate RF-owned writes; external writers still require the
identity checks. The existing loading lane must be the sole owner of overlapping
file leases when integrating: do not nest this scope inside an already-held
checkpoint/component lease or create a competing preparation service.

Preflight accepts only native engine requests for `BaseModel`,
`SD 1.5`, `SD 2.0`, `SDXL`, `SDXL 1.0` and `SDXLRefiner`. This label check is
not runtime architecture proof. The loader additionally verifies the initialized
SD15/SD20/SDXL/SDXLRefiner configuration, discrete sampling type, non-inpaint
configuration and exact captured checkpoint path before applying an override.
Flow or unrelated models are rejected even when metadata was mislabeled.
GGUF, Diffusers bundles, image-edit special routes, video and optional engines
remain outside this bounded integration. Ordinary text-to-image and image-to-image
are accepted; inpaint, control, edit and utility modes require their own contracts.

## Evidence and remaining gates

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_yaml_recipes.py test_yaml_recipe_callers.py
```

The isolated suite has 11 tests covering native field extraction, EPS no-op,
signed CLIP layers, inert targets, hostile/malformed YAML, actual read limits,
file mutation/digest/path identity, cancellation, existing leases and no weight
reads/acquisition. Caller coverage adds the actual native sampling patch, API
field-presence policy, metadata/default replay, warm conditioning/model caches,
recursive component fallback and a checkpoint writer queued behind the real
outer loader lease. Recursive fallback reuses that outer lease rather than
acquiring a second reader that could deadlock behind the writer.

Browser picker/persistence and real-model execution remain separately assigned
gates. The parser/caller tests do not certify output quality, speed or GPU
stability. See [native model formats](native-model-formats.md) for container and
bundle limits; no arbitrary YAML-specified runtime or custom node execution is
introduced.
