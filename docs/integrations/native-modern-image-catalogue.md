# Modern native image sources and recipes

This expansion adds 35 immutable tuples across 17 native configurations: 33
primary image checkpoints, Ideogram 4's unconditional companion and Qwen Image
Layered's RGBA VAE. Sources are pinned in `model_catalogue/native_modern_*.json`;
the joined registry rows retain published bytes/SHA-256, bounded header digest,
native configuration, behavioral variant and workflow requirements.

| Source group             | Variant distinctions                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------- |
| Boogu, ErnieImage, Fibo2 | Edit/Base/Dev versus Turbo remain separate recipes                                          |
| ZImage                   | Base 25-step and Turbo 8-step contracts require a current variant declaration               |
| Flux, FluxInpaint        | Krea Dev, Kontext and Fill have distinct workflow controls                                  |
| Flux2                    | Klein Base 4B, distilled 4B, distilled 9B KV and Dev have separate configurations/contracts |
| LongCatImage, MOSSImage  | Base versus Edit/SFT retain distinct source identities                                      |
| Chroma, ChromaRadiance   | Latent versus pixel-space; Radiance needs custom beta parameters                            |
| Lumina2/NewBieImage      | Native configuration and architecture labels distinguish their encoder layouts              |
| PixelDiT, PiD            | Text-to-image versus source-latent upscale utility                                          |
| QwenImage                | Base, 2512, Edit, Edit 2509, Edit 2511 and Layered remain explicit                          |
| JoyImage                 | Dedicated edit encoder and source recipe                                                    |
| Ideogram4                | Conditional checkpoint requires its unconditional second UNET                               |

Thirty-four small resources add source/configuration recipe contracts. The original
34 legacy presets and their projection digest remain unchanged. Recipes can be
read by stable ID through `load_family_recipe`/`family_recipe`; native lookup uses
the index's `native_recipes` references and reads only that class's variants.
No benchmark history, weight tensors, publisher request or model hash is involved.

`compatible_native_recipe(path, metadata)` first accepts a current exact registered
source proof. Otherwise it requires an already initialized native header reader,
a matching native class/configuration/architecture, and explicit behavioral
declarations where variants differ. F16, BF16, FP8 and logical GGUF storage do not
change the recipe. A local filename, precision tag or family label cannot select
Turbo. Only unambiguous, documented configuration baselines can omit a variant.
Saved overrides and current explicit publisher settings retain precedence.

`artifact_qualified=False` explicitly separates this recommendation from exact
artifact/device/output qualification. `auto_recipe_supported` concerns the five
ordinary sampling controls, not end-to-end operation support. Specialized source
requirements remain explicit instead of substituting generic defaults:

- Flux2/Klein needs `Flux2Scheduler`; Ideogram 4 needs its dedicated scheduler and
  `DualModelGuider` with the second checkpoint.
- ChromaRadiance uses beta alpha/beta 0.4/0.4, not the generic beta schedule.
- Kontext/Fill retain their guidance and image-conditioning requirements. PiD
  remains a source-latent utility.
- Qwen Layered requires its four-channel image/16-channel latent VAE, layer count
  and ordered output handling. It cannot use the ordinary RGB Qwen/Anima codec.

The native specialized sampling adapter now consumes six of those contracts:
Flux2 Dev, Klein Base 4B, distilled 4B and 9B KV, ChromaRadiance, and Ideogram4's
conditional primary model. Their artifact, profile and recipe Auto projections
are enabled together. The KV source also records the native `FluxKVCache` patch
node. Five rows remain disabled: Kontext, Fill, PiD, Layered and Ideogram4's
standalone unconditional companion. Sampler/source CPU checks establish these
caller contracts; they do not qualify weights, output quality or GPU execution.

The current Qwen base/Edit template graphs also contain optional Lightning LoRAs.
Their 8/4-step controls are not applied to an unmodified base checkpoint. Those
rows retain the existing documented non-LoRA family recipes; Edit 2511 has its
own source-bound 40-step/CFG 3 contract.

The [extended catalogue](native-extended-image-catalogue.md) and
[image/Chat scope](image-chat-scope.md) own current availability and exclusions.
The earlier 65-class denominator and missing-consumer counts are historical
source receipts, not a requirement to implement native multimedia routes.
Retained modern image tuples, variants and their distinct codecs remain present.
This catalogue metadata certifies no GPU execution, memory fit or image quality.

The original migration hashes remain recorded by the
[registry schema](model-registry-schema.md); the current splitter verifies the
explicit retained projection after media retirement.

Local bounded metadata receipts and source-classification decisions are in
`cache/plan-execution-20261111/publisher-payload-review/wave-b-prepared.json` and
`native-classifications.json`. The maintained splitter verifies shipped data:

A fresh 25-sample comparison of one Qwen artifact/profile lookup against
`ad6d015` gives Python-cache-cold p50/p95 9.05/10.31 ms versus 19.02/20.46 ms;
warm lookup is 0.83/0.85 ms versus 0.67/0.68 ms. The larger image role shard now
contains the added tuples. OS cache is uncontrolled/warm; these are metadata-only
timings, not process startup, UI or generation measurements. The source/index
bound receipt is `cache/plan-execution-20261111/W02-registry-lookup-wave-b.json`.

```sh
python tools/split_model_registry.py
python tools/check_cpu.py test_native_modern_catalogue.py test_native_recipe_contracts.py test_model_default_resources.py test_model_registry_resources.py
```
