# Extended native image and utility sources

The retained extended catalogue has 27 immutable tuples over 23 native classes:
20 image checkpoints, five image-utility payloads, one depth payload and the
SeedVR2 codec. Exact publisher revision, bytes, SHA-256 and bounded native header
classification stay attached to each tuple. No tensor values are inspected by
catalogue lookup.

The supported source contracts include SD 1.5/2/SD21 UnCLIP, InstructPix2Pix,
SDXL refiner/SSD1B/KOALA/Vega, SD3, AuraFlow, Cosmos Predict2, PixArt, HunyuanDiT,
HunyuanImage 2.1/refiner, Lotus depth, SDPose, RT-DETR, SAM3.1 and SeedVR2.
Capybara and ChronoEdit entries are retired because RF had no connected
still-image consumer. The [product scope](image-chat-scope.md) records why shared
components with real still-image callers remain.

SDPose requires RT-DETR. SeedVR2 requires its exact native codec and one-frame
restoration caller. HunyuanDiT and UnCLIP checkpoints retain their bundled
components. Qwen Layered keeps its separate RGBA image codec and channel/layout
checks; its shared WAN ancestry does not make it a video operation.

Recipes without a reviewed sampling contract retain
`auto_recipe_supported=false`. A local working size is not a publisher default,
and source metadata does not qualify real-model output or memory fit. Publisher
access gaps for retained SAM3/Flux3 and the local SenseNova sharded reader remain
separate from exact acquisition and device qualification.

The canonical full registry now has 218 profiles, 359 artifacts and 132 benchmark
records; the recipe index has 92 image recipes and 26 image preset projections.
[Migration verification](model-registry-schema.md) keeps the original historical
hashes while checking the exact retained projection after explicit media
retirements. Current counts are scoped resource facts, not native parity claims.

```sh
python tools/split_model_registry.py
python tools/check_cpu.py test_native_extended_catalogue.py test_native_catalogue_dispositions.py test_model_default_resources.py test_component_family_contracts.py test_image_chat_resource_scope.py
```
