# Native image-prompt adapters

Ordered Flux.2/Klein, Qwen Edit Plus and LongCat editing use the separate
[native edit recipe contract](native-ordered-image-editing.md).

**Image tools → Image prompt adapter** uses the pinned native StyleModel and
PhotoMaker nodes. This is a local-file workflow, separate from image-to-image:
the reference upload does not select an editing tool or change the output canvas.
None is the default. No adapter, vision model or companion is downloaded.

| Choice       | Checkpoint contract                                            | Local files                                                                                  |
| ------------ | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| StyleAdapter | Ordinary native SD1, 768-wide text conditioning                | Native `style_embedding` adapter with 1024→768 projection; 1024-wide ViT-L/14 vision encoder |
| Redux        | Ordinary native Flux/Flux Schnell, 4096-wide text conditioning | Native Redux 1152→4096 projection; SigLIP 384 vision encoder                                 |
| PhotoMaker   | Ordinary dual-CLIP SDXL, 2048-wide conditioning                | Native PhotoMaker ID encoder; one reference image and one native 77-token prompt segment     |

The pickers filter safe tensor headers for the supported layouts. Local native
Torch containers are explicit selections for StyleAdapter, Redux, their vision
encoders and PhotoMaker. They are labeled **Compatibility unknown**; browsing
never deserializes them. The pinned native loaders use weights-only loading and
validate the selected state dictionary during admitted compute. Vision output
width is checked before applying reference conditioning. Suffixes, names and directory placement
do not certify model execution, artifact integrity or portrait quality.

Place StyleAdapter/Redux files in `models/style_models`, PhotoMaker in
`models/photomaker`, and vision encoders in `models/clip_vision`, or configure the
corresponding roots. Existing custom root priority is preserved. Missing saved
files remain visible and fail explicitly; Refresh does not choose replacements.
These are the built-in native adapters, not custom-node IPAdapter support.

For PhotoMaker, use a prompt such as `portrait of person photomaker`. The native
trigger is the standalone, lowercase `photomaker` token after the subject word.
It is not a universal face detector or a new reasoning/tool capability. PhotoMaker
uses its native ID-encoding behavior and has no strength control here. Native
StyleAdapter/Redux strength accepts zero through ten; None disables the adapter.
The native vision encoder applies its center crop to the reference only.

## Request and resource ownership

Submission captures a private image copy, the selected checkpoint signature,
resolved local component files/signatures, kind and strength before queueing.
Native model configuration and conditioning widths are checked again at use.
Prepared-asset read leases cover native loading/encoding, including cache hits;
file replacement invalidates the cache. A single pipeline-owned model cache
retains weights, while reference features and conditioning are recomputed for
each request. Switching the primary model discards that adapter cache.

The caller applies native `CLIPVisionEncode`/`StyleModelApply` or native
`PhotoMakerLoader`/`PhotoMakerEncode`. It does not implement another sampler,
loader, device manager or recognition network. Positive conditioning, including
scheduled prompt intervals, is request-local; cached text and negative
conditioning remain unchanged. PhotoMaker uses the same explicit Clip Skip/YAML
layer policy as ordinary encoding. Unsupported inpaint/control/edit combinations
fail before sampling rather than inheriting another route's behavior.

PNG/default replay retains the explicit kind and local file choices. Library
replay clears the upload and requires a new reference image; it never invents a
reference from a previous result. Output provenance records local file stat
signatures, reference dimensions and normalized RGB pixel hash, not image pixels.
The adapter panel hides during ControlNet, editing, inpainting and utility
operations. UI submissions omit inactive reference settings while retaining the
user's choices and upload for a return to None or Image to image. Explicit API
incompatible combinations still fail. Imported metadata follows the same context.
UI utility submissions omit inactive generation-only recipe/reference settings
from their captured request while preserving UI/profile values. Explicit API
utility combinations are rejected.

## API and evidence

Image requests accept `reference_adapter` (`none`, `style`, `redux`, `photomaker`),
`reference_adapter_model`, `reference_adapter_vision`,
`reference_adapter_strength` and `reference_adapter_image`. File names resolve
under the configured local roots. The image follows the existing base64/owned
API image-URL boundary, separate from `input_image`. Omitting the adapter never
reuses an image from the owner UI.

`test_reference_adapters.py` executes pinned native conditioning/PhotoMaker
methods on tiny CPU fixtures and tests source isolation, file mutation and
generation→utility→generation capture. `test_reference_adapters_ui.py` exercises
real Gradio/API and Library/default transport. Source/CPU evidence does not prove
actual weight execution, visual quality, performance or GPU stability. Those
remain separate acceptance gates.
