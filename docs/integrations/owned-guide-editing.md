# Owned depth and normal guides for editing

The `image_editing/guide_handoffs.py` service separates the original image from
a previously computed control guide. Editor and typed API requests capture this
handoff before worker preparation. Source/CPU checks do not establish finished
controlled-edit quality or rendered browser qualification.

`describe_result_guide` recognizes current owned MoGe/Lotus depth or perspective
normal results from their saved execution provenance. `restore_result_guide`
atomically restores the retained original and returns an immutable
`ResultGuideHandoff`. It retains the result as a separate guide, preserving
original source pixels, alpha, history and generation provenance. An ordinary
image, stale or foreign handle, pending operation, changed file, mismatched
frame, unknown convention or precision working-copy mismatch is rejected.

The opaque `_editor_guide_handoff` stays in server state. Its public context uses
the existing session/revision/source handles. Capture independently rereads both
owned files, verifies signatures and SHA-256, and freezes separate original RGBA
and guide RGB bytes under `_editor_guide_request`. Display widgets cannot replace
either input. The initial working-image contract is exact-frame 8-bit sRGB;
precision sources/maps require an explicit working copy.

`capture_guide_inputs` requires a selected ControlNet admitted by the separate
`guide_control_compat` identity/header/semantic contract. No model is silently
substituted. The explicit `control_guide_denoise` is separate from ControlNet
strength; values including 0, 0.35 and 1 are retained. Capture preserves prompt,
negative prompt, requested dimensions and framing. Control model identity and
the frozen normalized control settings are checked again before compute.

## Effective plan identity

`effective_image_plan` binds the typed guide request's technical facts into both
the request fingerprint and exported native facts. The capture includes distinct
source/guide file and pixel hashes, file signatures, pixel dimensions/type,
producer convention and projection, ControlNet identity and settings, denoise,
and any pinned native scalar-conversion source digests. A changed packet requires
a fresh plan; a guide cannot silently change while retaining the old fingerprint.

Owned and explicit-upload packets share this contract while preserving their
different provenance. The exported projection contains no owner/session/revision,
artifact handles, frame UUIDs, source paths or opaque runtime packet. The complete
frozen execution controls also have a digest; their exported model names and
signatures are path-redacted. JSON, the Editor plan fingerprint and PNG metadata
use the same serialized technical facts. Runtime ownership validation remains with
the guide service. Ordinary requests without a guide retain their existing plan
identity and receive no guide facts from a previous capture.

For a guided Auto request with Preserve framing and no existing native size plan,
the shared `automatic_source_size` rule keeps source proportions within the
existing working-size budget. A source within that budget, such as an 832×1216
portrait, keeps its dimensions instead of model square defaults. Manual sizes and
existing native size plans retain precedence, without rewriting saved draft dimensions.
An oversized Auto source uses the maintained contain plan and common fit/pad
geometry to meet the SDXL 2048-side/4-megapixel working bound. The original stays
unchanged; very elongated sources satisfy the side bound as well as the area
bound. Explicit smaller targets retain precedence.

Guided editing must VAE-encode the original RGB image and apply the captured map
only as the ControlNet hint. Supplied maps bypass depth/normal inference.
`prepare_reference_pair` reuses existing image geometry for both images, checks
their plans agree, and retains source coverage separately. Default framing keeps
source dimensions within the existing budget; an explicit size override applies
one uniform resize/pad plan to both. `restore_source_alpha` restores that coverage
after native decoding. Ordinary control-only generation retains its existing
semantics when no typed guide request is present.

Normal-vector conventions remain explicit producer facts. Encoded OpenGL and
DirectX maps are not interchangeable merely because both are RGB images.
Compatibility must prove the selected model's expected convention or reject the
handoff. Panoramic radial depth is likewise distinct from perspective camera
depth. No new vector, depth-estimation or ControlNet algorithm is implemented.

For the specifically verified Xinsir base6 NormalBae consumer, the source-qualified
coordinate derivation complements encoded red for native OpenGL maps, or red and
green for native DirectX maps. The receipt freezes those channel choices. The
working guide applies pinned `InvertMask` to those scalar channels once before
shared geometry; there is no spatial flip, RGB/BGR reorder or re-estimation.
The original producer map and metadata remain unchanged. This is a source-derived
coordinate relation, not a claim that different learned normal estimators are
equivalent or that a finished normal-controlled edit has passed visual review.

Face-region results continue through the existing owned selection-mask handoff
and actual Library inpaint request builder. Their grayscale values become layer
alpha once; source alpha is independent. Existing `inpaint_denoise` and
`inpaint_mask_grow` values remain available. Mask handoff does not claim a general
boundary blur: native FeatherMask attenuates image-frame edges.

Focused CPU verification:

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_guide_handoffs.py test_guide_control_compat.py test_editor_session.py
```

The handoff tests use actual Editor artifacts/transactions, source restoration,
request capture, existing frame geometry and the actual inpaint request builder.
The compatibility suite separately owns real header/source identity admission.
Actual controlled edits, visual seam quality and rendered UI require their own
source-bound acceptance.

## Explicit API uploads

`POST /v1/images/generations` accepts the original as `input_image` and a separate
`control_guide` object with `image`, `kind`, `convention` and optional `denoise`.
Both image fields use the existing inline image or unexpired API media URL
transport. Arbitrary URLs, local paths and Editor handoff tickets are not upload
authority. Source and map must be exact-frame 8-bit sRGB images, at most 16
megapixels; inpaint masks and Editor contexts require their separate workflows.

For depth, declare `kind: "depth"` and `convention: "white-near"`. For normals,
declare `kind: "normal"` and `moge-opengl`, `moge-directx` or `normal-bae` as the
convention. These are caller declarations, never proof of an owned inference.
Already NormalBae-compatible normals pass through without axis inversion.
Perspective maps are supported; panorama conventions are rejected.

Use the existing `controlnet` object to select the reviewed installed control and
set its strength/start/stop. The server captures its current file identity. A
precomputed map disables further depth estimation; an explicitly requested
estimator conflicts with this route. Edit denoise defaults to 0.35 only when
omitted and retains an explicit zero. The original supplies VAE pixels and alpha,
while the guide supplies only the ControlNet hint through shared framing.
With omitted or `auto` size, guided Auto framing retains the original dimensions
instead of the model's recommended square. An explicit API size disables Auto for
that request; an explicit pad target applies one uniform plan to source and map.

The upload constructor freezes source/map bytes and request ownership in a
separate `UploadedGuideHandoff`. It projects into the common `GuideRequest` and
effective plan without inventing Editor session, revision or artifact handles.
Worker preparation revalidates ownership, content and control identity before
compute. API regressions live in `test_api_precomputed_guides.py`; actual finished
edits and their visual quality remain a separate acceptance gate.
