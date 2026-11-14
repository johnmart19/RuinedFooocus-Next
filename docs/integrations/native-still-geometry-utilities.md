# Still-image depth, landmarks, pose and albedo

In **Editor → More tools**, choose **Lotus depth**, **MoGe depth**, **Face
landmarks**, **Whole-body pose** or **Marigold albedo**. Add one source image and
run the image tool. Face landmarks and Whole-body pose select compatible local
components or their [registered first-use defaults](native-utility-defaults.md).
Advanced Lotus, MoGe and Marigold workflows require the components shown below.
These utilities use their own local models and a fixed native operation. The
generation checkpoint, prompt, text encoders and generation sampling settings
are not required. The existing **Depth map** DA3 tool remains a separate route.
An explicitly selected registered first-use component can be prepared before
compute through the existing download broker; browsing never starts a transfer.

| Tool / API `controlnet.type`        | Required installed components                                                                                                                                   | Native values and result                                                                                                                                                                                                                                                                                             |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Lotus depth / `lotus-depth`         | LotusD depth safe tensor in `geometry_estimation`, `diffusion_models` or `checkpoints`; compatible four-channel, eightfold SD VAE in `vae`                      | Existing native one-step Lotus workflow: null conditioning, disabled noise, Euler, normal one-step sigmas with first sigma 999, VAE decode and invert. One white-near relative depth image. SDPose heatmap checkpoints are excluded from this picker.                                                                |
| MoGe depth / `moge-depth`           | Native MoGe v1/v2/v3 safe tensor in `geometry_estimation`                                                                                                       | `LoadMoGeModel` → `MoGeInference` or `MoGePanoramaInference` → `MoGeRender`. Perspective depth or surface normals; equirectangular panorama depth. Detail 0–9 defaults to 2; MoGe 3 refinement 0–8 defaults to 3. Outputs remain 2D images.                                                                          |
| Face landmarks / `face-landmarks`   | Complete MediaPipe safe tensor bundle in `detection`, including both detectors, landmarks, blendshapes, canonical data and topology                             | `LoadMediaPipeFaceLandmarker` → `MediaPipeFaceLandmarker`, then an IMAGE overlay or native `MediaPipeFaceMask` → `MaskToImage` selection preview. Short/full/both detector range, 1–16 faces, confidence 0–1. No face leaves the overlay unchanged or returns an explicit empty region mask. No 3D mesh is exported. |
| Whole-body pose / `pose-map`        | SDPose LotusD safe tensor checkpoint with heatmap head and bundled SD VAE in `checkpoints` or `diffusion_models`                                                | Native checkpoint loader, `SDPoseKeypointExtractor` → `SDPoseDrawKeypoints`. One full-image person, body/head/hands/face/feet drawn on a black map; confidence 0–1 defaults to 0.5. The intermediate is `POSE_KEYPOINT`, not an image or mesh. Detector-assisted multi-person extraction is a separate operation.    |
| Marigold albedo / `marigold-albedo` | Compatible native Qwen Image Edit diffusion model; dedicated Marigold V2 albedo LoRA in `loras`, albedo VAE in `vae`, saved albedo conditioning in `embeddings` | Native LoRA strength 1, AuraFlow shift 1.73 with `img_to_img_velocity`, disabled noise, Euler with manual sigmas `[0.5, 0]`, VAE decode, `MarigoldV2PostProcess(prediction="albedo")`. One sRGB albedo image. No text encoder or prompt encoding.                                                                    |

The image is EXIF-oriented and frozen as RGBA bytes before queueing. The limit is
one source of at most 16 megapixels. Every result retains the oriented source
width, height and alpha. Estimators receive RGB. Lotus pads by replicating edges
to a multiple of eight; Marigold pads to sixteen for Qwen patchification. Both
crop that padding after the native operation. Pose and MoGe native nodes retain
the source canvas; any unexpected output size, batch count or non-finite value
fails rather than stretching the result. Default transport is an RGBA PNG;
the existing PNG/EXR precision controls can retain float output and source alpha.
Native still AVIF cannot preserve RGBA and is rejected at utility capture before
processing; alpha is never silently dropped.
The MoGe image is a normalized visualization, not exported metric geometry.

The source picker reads bounded safe tensor headers without loading networks or
hashing weights. Malformed or excessively nested header JSON is rejected as an
invalid candidate and skipped during passive picker refresh.

Marigold's backbone also needs positive source-bound **Edit2509** provenance;
the shared `QwenImage` architecture or a filename is insufficient. Its official
Marigold model tuple and reviewed registry Edit2509 encodings are accepted after
current signature/SHA/size verification, including alternate reviewed
quantizations and renamed copies. An existing preparation-service source receipt
can supply that proof without rehashing weights. Unknown same-class choices stay
visible as **Edit2509 source unverified** and fail submission with an actionable
provenance message; no default replacement is selected. Submission may hash an
explicitly selected candidate once through the existing signature-keyed cache.
Capture freezes the positive source record, and processing rechecks its current
registry/recipe binding and file signature before native loading or cache use.
Changed files or source evidence require a fresh selection/submission. The albedo
LoRA, VAE and saved conditioning retain their exact task-specific identities.

Lotus/SDPose use native architecture detection; SDPose additionally
requires its heatmap head and bundled planar four-channel VAE. MoGe requires its
native backbone and distinct geometry heads; a DA3 DINO backbone is insufficient.
MediaPipe requires the complete native bundle layout. The native loaders perform
the final weight/layout checks. Files may be renamed inside configured roots.
Absolute UI selections and relative API names must resolve below those roots;
traversal and paths outside the selected role's roots are rejected.

### MoGe surface normals and panorama depth

MoGe keeps perspective depth as its default. Select the normal output for a
perspective source and choose native OpenGL or DirectX convention. RF calls
`MoGeRender(output="normal_opengl"|"normal_directx")` without a separate normal
estimator: the native node uses predicted normals when the model supplies them,
or derives them from its point map otherwise. The native camera-space map is
encoded as `normal * 0.5 + 0.5`, including the native Y/Z sign conventions. A
point-derived normal map needs a source of at least 2 × 2 pixels. Output metadata
identifies the convention, predicted or point-derived source and camera frame.

Select panorama projection for depth from an equirectangular source. The pinned
`MoGePanoramaInference` uses twelve 90-degree perspective views and its own
multi-scale Poisson/gradient merge. RF does not copy its camera rig, image
sampling or SciPy solver. Per-view metric scale and predicted normals are
discarded by the native panorama node. Its radial distance becomes the same
native percentile-normalized, white-near disparity visualization used by the
depth renderer. Uncovered pixels use the native far-fill behavior; this is not
a metric depth export. Panorama normals, meshes and 3D exports are not supported.

`moge_output` accepts `depth` or `normal`, `moge_projection` accepts `perspective`
or `panorama`, and `moge_normal_convention` accepts `opengl` or `directx`. Defaults
are `depth`, `perspective` and `opengl`. Normal output with panorama projection is
rejected. Panorama controls are `moge_split_resolution` (256–1024, default 512),
`moge_merge_resolution` (256–8192, default 1920) and
`moge_panorama_batch_size` (1–12, default 1). Detail and refinement retain the
existing `moge_resolution_level` and `moge_refine_steps` fields. The native node
accepts any source aspect ratio; RF retains the oriented source canvas and does
not convert an ordinary perspective photograph into an equirectangular panorama.

Both routes retain the frozen source alpha and float output path. Native initial
placement, offload and dtype continue through ComfyUI model management, including
native memory/precision CPU choices and explicit CPU startup authority. Metadata
records actual inference placement and distinguishes intentional CPU placement
from accelerator-failure fallback. A genuine accelerator failure can retry once
with a fresh native model and CPU load target; the failed cached model is unloaded
first. The cache includes effective device identity. Native rendering, validation
and the final host transfer remain inside that recovery attempt.

CPU recovery is unavailable while an explicit GPU-only configuration keeps the
native construction/offload target on an accelerator. That combination fails
before a CPU-attempt model is constructed and publishes no result. Disable
GPU-only and use Auto, or restart with `--cpu`, before submitting again. Ordinary
Auto with native CPU offload supports the fresh-model CPU retry.

The source read lease covers inference, rendering and host transfer. Cancellation
is checked around native calls and between panorama inference batches. A running
native sparse solve completes before the following cancellation check; cancelled
work publishes no output provenance.

### Face-region selection masks

Face landmarks keeps its overlay output by default. Choose its region-mask
output to select native **All** regions or a custom union of the face oval,
lips, left eye, right eye and irises. The default custom choice is the face oval;
at least one custom region is required when producing a mask. Inactive custom
choices can remain saved while viewing an overlay.

RF passes the detected `FACE_LANDMARKS` packet, including the loaded model's
actual `connection_sets`, to pinned `MediaPipeFaceMask`, then uses native
`MaskToImage` for the preview. It contains no copied landmark indices or polygon
algorithm. Multiple detected faces are unioned by the native node. White selects
and black protects; the RGB preview channels are equal. Source dimensions,
orientation and alpha are preserved. Selection values remain independent of
image alpha when the explicit Editor mask action binds the result to its
retained original source and frame.

No detection produces an all-black mask with `face_count=0`, `mask_empty=true`
and `mask_empty_reason=no-face`; it never substitutes a full-image selection.
An empty mask cannot submit an inpaint operation. Overlay results do not carry
the selection-mask role. Mask provenance identifies the native output type and
SHA-256 of the frozen oriented RGBA source bytes. Cancellation or invalid native
mask dimensions/values publishes no result-mask provenance.

This pin bundles a pure-PyTorch MediaPipe Face Landmarker port; it does not use
the separately distributed `mediapipe` Python API. Its existing installed
Torch/NumPy/Pillow/SciPy dependencies and complete local face bundle are reused.
Changing the output choice does not install a package or select a different model.

The explicitly selected registered source is
`detection/mediapipe_face_fp32.safetensors`, published by
[Comfy-Org/mediapipe](https://huggingface.co/Comfy-Org/mediapipe/tree/f787efc1ad99492ec7d5b6564c0dd6c48d84ee49).
Revision `f787efc1ad99492ec7d5b6564c0dd6c48d84ee49` declares Apache-2.0 and one
5,423,900-byte bundle, SHA-256
`a98c4806081d40eba35102a0f6dc0000c2e1388b72cf24e691703d0605bd888a`.
The catalogue tuple and source-bound registry annotation use that immutable URL.
A missing registered selection is captured by `selected_image_artifact` and
prepared by the existing `capture_image_assets` / `prepare_image_assets` broker
before compute. Offline mode reports the missing face-landmarker component and
never starts a transfer. Installed renamed bundles retain local header admission;
unknown missing filenames or changed publisher tuples cannot request a download.

The shared worker policy preserves native `text_encoder_device()` placement,
including intentional CPU placement for low-memory or precision policies. Explicit
CPU remains authoritative; metadata separates the requested policy, actual native
inference device, fixed float32 inference and CPU polygon/preview rasterization.
A supported accelerator failure clears failed cached patchers, constructs a fresh
CPU model and retries once when native offloading permits CPU. Native inference,
mask/overlay rendering, validation and host transfer are all inside that recovery
boundary. Device identity is part of the native model cache key; native offload
placement and loaders remain unchanged.

Explicit GPU-only mode can also put the constructor's offload target on the GPU.
In that case CPU recovery is rejected before another constructor or fallback
success/warning is published. The error directs the user to disable GPU-only and
use Auto, or launch with `--cpu`. It does not silently retry construction on the
failed device. Ordinary Auto with CPU offloading and explicit CPU are the supported
recovery contracts.

Marigold task-specific components can share tensor shapes with depth or normal
variants. Albedo LoRA, VAE and conditioning are therefore bound to the exact
publisher byte sizes and SHA-256 values in
[`marigold-v2-albedo.json`](../../ruinedfooocus/resources/native_utility_recipes/marigold-v2-albedo.json).
The passive picker never hashes weights; submission verifies the three selected
task components with the existing signature-cached hash service. A compatible
alternative source-verified Edit2509 encoding is selected explicitly and does not need
the default UNET's exact digest. Source identity and reviewed automatic recipes
remain separate. This source-only packet does not enable a new generation Auto
recipe or register fabricated download tuples.

## Request, execution and replay

`depth_utilities.UtilityRequest` freezes operation, source bytes, dimensions,
file roles/signatures and validated scalar values. `ImageArtifact` entries carry
the exact local component paths into the existing preparation service. Native
loading and inference hold read leases. Replaced, deleted, shadowed or changed
components require a new submission. The existing serial image compute lane
owns admission and Stop/cancellation; no second queue or device manager is added.
Native patchers are cached by operation and all component identities, unloaded
when replaced or when leaving the utility pipeline, and cleaned up after a
partial loader failure. Cancel checks surround native calls. Inpainting,
reference images, generation recipes, mesh export, video and training
are not accepted modes of these routes.

Both JSON `POST /v1/images/generations` and multipart
`POST /v1/images/edits` accept the following example. Multipart uses the upload
field `image` and sends `controlnet` as JSON text.

```json
{
  "controlnet": { "type": "lotus-depth" },
  "input_image": "<base64 PNG or supported local image upload>",
  "image_utility_model": "lotus-depth-d-v1-1.safetensors",
  "image_utility_vae": "vae-ft-mse-840000-ema-pruned.safetensors",
  "response_format": "b64_json"
}
```

The four component fields are `image_utility_model`, `image_utility_vae`,
`image_utility_lora` and `image_utility_conditioning`; only the active operation's
required fields are captured. Scalar API fields are `moge_resolution_level`,
`moge_refine_steps`, `face_detector_variant`, `face_max_count`,
`face_min_confidence`, `face_output` (`overlay` or `region-mask`),
`face_mask_regions` (`all` or `custom`), five boolean `face_mask_*` fields
(`face_oval`, `lips`, `left_eye`, `right_eye`, `irises`) and `pose_score_threshold`.
Use `n: 1`; leave `size`
unset/`auto` or supply the exact oriented source size. Uploaded masks, canvas borders and
additional sources are rejected. JSON and multipart use the same validation.

PNG/Library metadata stores the operation, component role/basename/size, scalar
options, source size, alpha convention and native output type. It omits local
folder paths and fictitious generation settings. Library **Apply metadata** and
**Reuse setup** restore the named utility and scalar options without resolving a
generation checkpoint. They clear uploaded sources and component selections:
choose the original image and installed components again to estimate afresh.
Saved scalar startup choices are independent of request-owned image pixels.

## Source and acceptance

The owning adapter files are `features/image_processing/depth_utilities.py`,
`utility_{lotus,moge,face,pose,marigold}.py`, `depth_utility_pipeline.py` and
`ui_depth_utilities.py`. Lotus reuses `features/image_editing/lotus_depth.py`.
Shared callers are `image_preparation.py`, `pipelines.py`, `async_worker.py`,
`api_runtime.py`, the public OpenAI image schema and Library replay.

Native source is ComfyUI
`989d93d0d968e704c589ce0146bb18c77f6b50cc`, with `nodes_lotus.py`,
`nodes_moge.py`, `nodes_mediapipe.py`, `nodes_sdpose.py`, `nodes_marigold.py`,
`nodes_cond.py`, `nodes_model_advanced.py` and `nodes_custom_sampler.py`.
Installed official source recipes are in
`comfyui-workflow-templates-json==0.1.129`: `image_lotus_depth_v1_1.json`,
`utility_moge_depth_estimation.json`, `utility_moge3_geometry_estimation.json`,
`utility_sdpose_ood_image_to_pose.json` and
`image_marigold_v2_albedo_estimation.json`. The Marigold source metadata was read
on 2026-11-11 from the [publisher repository API](https://huggingface.co/api/models/Comfy-Org/marigold-v2-0?blobs=true),
revision `c304e4370a1bc677a98c906c3683ace49a6bdadd`, license Apache-2.0.
No weights were retrieved for this work.

Run the feature's declared CPU prerequisite closure and focused tests:

```text
python tools/check_cpu.py test_depth_utilities.py test_depth_utilities_ui.py test_lotus_depth.py test_native_vision.py test_api_image_edits.py test_image_metadata_boundary.py test_model_folders.py test_path_picker.py
python tools/check_cpu.py test_mediapipe_face_mask.py
python tools/check_cpu.py test_moge_geometry_outputs.py
```

The tests execute pinned native node bodies with tiny CPU network/device stand-ins,
real Gradio events, public JSON/multipart callers, the actual API request closure
and worker PNG/metadata path. They establish source, capture and transport
contracts. Installed weights, native/app output agreement, rendered browser
behavior, visual quality, memory fit, platform eligibility, performance and GPU
stability remain unqualified. MoGe mesh/3D export, MediaPipe 3D/animation, SDPose
multi-person/video and Marigold depth/normals remain separate scopes.

The MoGe output checks execute the actual pinned normal renderer, panorama node,
twelve-view split and SciPy merge on tiny CPU geometry fixtures. They establish
native conventions, radial-distance transport, frozen options/source alpha,
batching, read-lease release, cache cleanup, real host-copy failure recovery with
device stand-ins, explicit CPU authority and cancellation boundaries. Networks
are controlled stand-ins, so these checks do not qualify learned geometry,
panorama seam quality, real-model output agreement, memory fit, latency or GPU
stability. Use the existing local MoGe model only under the separately authorized
runtime gate; no model or package is downloaded by these routes.

The face-mask checks execute the actual pinned mask, ring ordering and
mask-to-image bodies with controlled detection/topology fixtures, plus the
installed pure-PyTorch port's import contract. They verify frozen scalar/source
capture, custom regions, face union, alpha, empty output, cancellation, explicit
CPU placement and native-device recovery, including a host-copy failure. Registry
and offline tests verify the immutable first-use tuple without network access.
The approved bundle was separately acquired through the actual app broker and
matched its publisher SHA-256, with no compute-runtime imports or model loads.
These fixtures and acquisition evidence do not establish real face-detection
accuracy or model execution; real-model, rendered UI and device qualification
remain separate.
