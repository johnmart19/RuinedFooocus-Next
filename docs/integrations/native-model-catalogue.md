# Native image catalogue dispositions

The active RF catalogue covers reviewed still-image generation, editing and image
utilities. ComfyUI is an implementation backend, not an automatic product list.
See the [image and optional Chat boundary](image-chat-scope.md) for the retained
consumer map. Upstream sources remain intact.

`load_registry().native_dispositions()` reads canonical indexed resources without
loading tensors, weights, model discovery or qualification history. The pinned
source identities remain ComfyUI `989d93d0d968e704c589ce0146bb18c77f6b50cc` and
ComfyUI-GGUF `da5f435cd9a9e54ad27eed417ed72b1f2543ef41`.

| Active resource group     | Reviewed RF scope                                                          |
| ------------------------- | -------------------------------------------------------------------------- |
| 62 configurations         | 56 image families and six image utilities                                  |
| 23 folder roles           | Retained image file readers and their GGUF discovery aliases               |
| 28 logical CLIP types     | Native image conditioning types, including distinct MiniMax Image          |
| 26 detected encoder types | Current image component roles, AuraFlow T5 XL and WeVisOne image utilities |

Each configuration keeps its exact native source contract, available artifact
identities, operation, RF route and separate source/installation/qualification
status. Catalogue presence never proves a local compatible installation, an Auto
sampling recipe, device fit or visual quality. Publisher access and ingestion gaps
remain explicit for retained image models.

The cleanup removes 44 other-media configurations and four image-labelled aliases
that had no RF image caller. These are not an active acceptance denominator or a
queue to implement missing capabilities. MiniMax H3 Image is a separate connected
24-channel image graph; SeedVR2 retains its connected one-frame restoration route.

## Component contracts

Component choices use exact resource roles and the selected native image family.
Bundled components keep precedence. A matching suffix, channel count, model name
or storage dtype is insufficient to establish tokenizer, codec or layout
compatibility. Qwen/Anima RGB, Qwen Layered RGBA, Qwen Image 2.1, Hunyuan Image 2.1,
its refiner, Ming and MiniMax Image retain their distinct contracts.

The shared Wan2.1 VAE stays for CosmosPredict2 and JoyImage. The official MiniMax
32B conditioning encoder stays for MiniMaxH3Image. Dedicated AV codecs/connectors
without a retained still-image consumer are removed. Typed image-utility payloads
and companion models remain distinct from ordinary generation checkpoints.

```sh
python tools/split_model_registry.py
python tools/check_cpu.py test_image_chat_resource_scope.py test_native_catalogue_dispositions.py test_component_family_contracts.py test_component_candidates.py test_component_overrides.py test_image_components.py
```

The guards check current product data and actual retained caller contracts. Full
native inventories remain optional source-audit output, not bundled product
catalogues or model-execution certification.
