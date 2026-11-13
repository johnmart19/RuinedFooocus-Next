# Image and optional local Chat scope

RF provides local still-image generation, editing and image tools, plus optional
local Chat, image understanding, assistants and bounded document retrieval.
ComfyUI is the backend implementation. Its new nodes, models, formats and
blueprints do not automatically become RF controls or download recommendations.

The retained catalogue is reviewed against ComfyUI
`989d93d0d968e704c589ce0146bb18c77f6b50cc` and ComfyUI-GGUF
`da5f435cd9a9e54ad27eed417ed72b1f2543ef41`. It contains 62 image/utility
configuration rows, 23 used image folder roles, 28 logical image encoder types
and 26 image/utility encoder model types. These are source and resource
contracts, not real-model, quality, performance or device qualification.

Video/audio/3D output, their dedicated checkpoints, components, defaults and
download recommendations are outside RF's product scope. The cleanup removes
48 artifacts, 31 exclusive profiles, eight video presets and four image-labelled
recipes without an RF image caller. The former Capybara, Bernini-R, ChronoEdit
and MiniMax FL2VA advertisements had only removed video dispatch paths. Native
upstream source stays intact; locally stored models, settings and outputs stay
in place. The former full-native parity programme and its gate counts are
historical evidence, not a product acceptance target.

The explicit [Bernini-R still-image adapter](bernini-image.md) now connects the
pinned one-frame image-edit blueprint. Only its signed official model pair and
component recipe are admitted through that adapter; generic Wan and every
temporal output route remain retired. Its new image-only source identities do
not restore the former video catalogue entries.

## Shared components retained by their image consumers

| Component                                                   | Retained RF consumer                                                                                                                      |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Wan2.1 VAE and its existing conversion                      | `image_components._COMPONENT_KEYS`: CosmosPredict2 and JoyImage still-image generation/editing                                            |
| Official MiniMax H3 Qwen3-VL 32B encoders                   | `MiniMaxH3Image` uses `clip_minimax_h3_image`; `generation/minimax_image.py` validates the dedicated still-image graph and encoder layout |
| Hunyuan Image 2.1 and refiner VAEs                          | Separate `vae_hunyuan_image21` and `vae_hunyuan_refiner` roles in the image pipeline                                                      |
| Qwen, Qwen Layered and Ming codecs derived from WAN layouts | Distinct image, RGBA/layer and image-model contracts; channel/layout checks remain mandatory                                              |
| SeedVR2 model and exact codec                               | `image_processing/native_seedvr2.py` runs the one-frame restoration workflow exposed in Image tools                                       |
| PyAV and ImageIO                                            | Still-image decoding and the retained scikit-image closure; see [dependency ownership](image-chat-dependencies.md)                        |

Depth/albedo maps, pose/landmark images, RGBA layers, image batches, native reference guidance,
LoRAs/embeddings and 3D-render visual styles remain image features. Multimodal
Chat models keep their text and still-image use even when their publisher name
contains “Video”. Their upstream capabilities do not grant RF an audio/video
input or generation route.

Chat-origin image actions capture independent generation, image-prompt rewrite
and Create-prompt update permissions. Only the first grants image execution;
prompt-only Create proposals can run with generation Off. The separate explicit
Create, Editor and image API workflows retain their own request authorization.
See [Chat image tools](chat-image-tools.md) for immutable turn ownership,
revocation, supported document inputs and guarded prompt Apply/Undo.

## Resource verification and history

`model_metadata/index.json` is regenerated from the retained canonical registry.
The original migration hashes remain recorded with exact retired identity keys;
the validator checks a separately pinned retained projection. Retired weights,
metadata blobs and capability registries are not shipped as a hidden catalogue.
The original content remains available in Git history at
`6b1edc68e804c60abd9256f734456daddeebade8`. Old feature descriptors retain their
commit-bound tests for sequential prerequisite replay.

```sh
python tools/split_model_registry.py
python tools/check_cpu.py test_image_chat_resource_scope.py test_model_registry_resources.py test_model_default_resources.py test_native_catalogue_dispositions.py test_native_extended_catalogue.py test_model_registry.py test_registry_download_audit.py test_component_family_contracts.py
```

The scope guard checks the current data, all download annotations and companions,
retained image recipes, and explicit shared consumers. It fails if upstream
registrations silently repopulate non-image product resources. Runtime admission
and unsupported API input checks belong to their actual application callers.
