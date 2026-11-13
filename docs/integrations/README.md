# Integration contracts

See [Settings HTML transport](settings-html-transport.md) for client-side search
and profile state.

See [native Gradio completion](gradio-frontend-completion.md) for the pinned
frontend stream-state guard and launch-time hidden-progress scope normalization.

These guides describe backend boundaries, upstream contracts and qualification
limits. They are for developers and advanced users; ordinary controls belong in
the feature UI, with setup instructions in the [user guides](../README.md).

The [image and optional Chat scope](image-chat-scope.md) owns the product boundary
and retained shared consumers. [Dependency ownership](image-chat-dependencies.md)
keeps still-image codecs and optional Chat closure explicit.

The [tool contracts](tools.md) document CLI/API consumers, dependencies and side effects.

The [ComfyUI 0.45 adoption contract](comfy045-adoption.md) binds the exact core pin,
package requirements and new native families to their actual app dispositions.

The [MiniMax H3 Image adapter](native-minimax-image.md) connects its native
PDD text-to-image/reference-edit graph and keeps artifact/device qualification separate.

| Topic                                                                  | Guide                                                 |
| ---------------------------------------------------------------------- | ----------------------------------------------------- |
| ONNX package families, platform eligibility, registration and recovery | [ONNX providers](onnx-providers.md)                   |
| Wheel-owned HIP, native attention and optional AMD diagnostics         | [AMD runtime](amd-runtime.md)                         |
| Reuse core runtime APIs and compare upstream installation recipes      | [ComfyUI platforms](comfy-platforms.md)               |
| Native SD4x source conditioning, codec and transparency contracts      | [SD4x diffusion upscaling](native-sd4x-contract.md)   |
| Ordered Qwen Layered RGBA generation, source capture and replay        | [Qwen Layered](native-qwen-layered.md)                |
| Native Cascade source images and paired stage grids                    | [Cascade source](native-cascade-source.md)            |
| Native Cascade Stage B companion selection                             | [Stage companions](native-stage-companions.md)        |
| Native Cascade super-resolution and vision references                  | [Cascade guidance](native-cascade-stage-guidance.md)  |
| Native SDXL noisy-latent base to refiner handoff                       | [SDXL refiner](native-sdxl-refiner.md)                |
| Exact image targets, native pixel factors, padding and PNG replay      | [Exact image targets](exact-image-targets.md)         |
| Queued image capture, owned editor publication and local Chat tools    | [Shared image operations](shared-image-operations.md) |
| Local object detection and source-pixel point masks                    | [Native vision utilities](native-vision-utilities.md) |
| Ordered still grids, masks, alpha composition and recipe replay        | [Image composition](native-image-composition.md)      |
| Text-targeted detection and soft alpha matting                         | [WeVisOne image utilities](native-wevisone.md)        |
| Hardware-specific precision, atomics, visibility and CU controls       | [AMD controls](amd-controls.md)                       |
| Default Chat candidate behavior on both local engines                  | [Local LLM qualification](local-llm-qualification.md) |
| User-selected pinned compact model and native processor                | [AtomicChat default](atomic-lfm3-default.md)          |
| Public download inventory, reachability and revision drift             | [Download audit](../download-audit.md)                |
| Family resources, exact download identities and lazy qualification     | [Model registry schema](model-registry-schema.md)     |
| Model geometry/sampling sources, behavioral variants and overrides     | [Family defaults](model-defaults.md)                  |
| App source ownership and UI-to-backend routes                          | [Architecture](../architecture.md)                    |
| Cache identities, retention, migration and targeted Maintenance        | [Cache contract](cache-contract.md)                   |
| Bounded preparation, Library transactions and UI request ownership     | [Concurrency](concurrency.md)                         |
| Direct Library drafts, source capture and installed Image Tools        | [Library editing](library-editing.md)                 |
| Owned Face selection and separate depth/normal editing-guide actions   | [Map-guided editing](map-guided-editing.md)           |
| Named Anima/Qwen/Z-Image/USO patches, Union modes and AliMama masks    | [Native model patches](native-model-patches.md)       |
| Scheduled local LoRA strengths, native keyframes and request cleanup   | [Native LoRA hooks](native-lora-hooks.md)             |
| Browser publication, pending prompt results and protected undo         | [Prompt drafts](prompt-draft-publication.md)          |
| Local Chat variants, catalogue destinations and owned transfer actions | [Chat model catalogue](chat-model-catalogue.md)       |
| Validation, documentation and publication requirements                 | [Development](../development.md)                      |
| Qualified WSL Chat and image-utility workloads and provider limits     | [WSL model execution](wsl-model-acceptance.md)        |

The ONNX/AMD audit is dated **2026-11-09**, against RuinedFooocus
`3ffdc0a0c3448544dcde90620d7ee7b285464885` and its pinned ComfyUI
`1332fabfcc507aa7c548cd5152071ed494c207e8`. Research establishes published
contracts, not local installation, device execution or measured acceleration.
Sections explicitly distinguish current code from proposed integration.

See [SeedVR2 still restoration](native-seedvr2-restoration.md) for its matching
codec, complete source geometry, alpha and request transport.

See [native tensor generation files](native-tensor-generation-io.md) for explicit
latent/conditioning inputs, opt-in exports, original precision and replay context.

See [Bernini-R still editing](bernini-image.md) for the signed one-frame image
recipe, ordered references, native sequential stages and runtime qualification limits.

Recheck moving upstream documentation before changing a dependency. Retain
release/source and artifact identities with results; a newer revision does not
inherit qualification automatically. Context7's ONNX Python API ID is
`/websites/onnxruntime_ai_api_python`; query a specific missing API contract,
then confirm it against the installed runtime and primary documentation.

See [layout and attention guidance](native-layout-attention.md) for native
GLIGEN text boxes and request-local hypernetwork attention.

Do not copy whole API references into this repository. Document the APIs actually
used, why the adapter exists, its owner, inputs/outputs, failure behavior and how
to validate it. Update the relevant guide in the same feature change.

[Still geometry utilities](native-still-geometry-utilities.md) cover standalone
Lotus/MoGe depth, facial landmarks, SDPose maps and Marigold albedo.

[Native histograms](native-image-histogram.md) document source-only RGB and
BT.709 analysis, its typed data API and cancellation/color contract.

[Native image filters](native-image-filters.md) cover explicit blur, sharpen,
brightness and contrast with source-size/alpha preservation and no model recipe.

[Image transforms](native-image-geometry.md), [mask adjustments](native-image-mask-tools.md)
and [color finishing](native-image-color-tools.md) share the **Adjust image** entry
with Transform, Mask and Color groups. The [adjustment integration](native-image-adjustments.md)
documents strict API capture, original precision and the owned Editor selection handoff.
Geometry includes exact crop to selection; color finishing includes text overlay,
RGB morphology and explicit floating-point SDR-to-HDR export. The existing geometry
utilities also cover [MoGe surface normals and panorama depth](native-still-geometry-utilities.md#moge-surface-normals-and-panorama-depth)
and [face-region selection masks](native-still-geometry-utilities.md#face-region-selection-masks).

[Native image utility devices](native-image-utility-device-policy.md) documents
Auto, explicit CPU authority, native model offload and bounded CPU recovery,
including the explicit GPU-only limitation. [Auxiliary fallback reasons](auxiliary-device-fallback-reasons.md)
documents the existing DirectML CPU routes for captioning and Chat embeddings.

[Curves and named GLSL effects](native-image-curves-effects.md) connect native
curve LUTs to a bounded native shader consumer; actual GL acceptance is deferred.
