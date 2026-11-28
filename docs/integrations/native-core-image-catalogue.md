# Native image acquisition: first source-verified expansion

Ten immutable download tuples add eight native image configurations to the
catalogue. Their exact SHA-256, published byte count and revision are in
`resources/model_catalogue/native_core_*.json`; the registry annotations record
the bounded header digest, native class/configuration, behavioral variant,
component topology and applicable default consumer.

| Native family   | Payload selection                 | Component/default contract                                                             |
| --------------- | --------------------------------- | -------------------------------------------------------------------------------------- |
| Anima           | Official base 1.0, 4.18 GB        | Qwen3 0.6B and old-Qwen RGB VAE; existing Anima family recipe                          |
| OmniGen2        | FP16, 7.93 GB                     | Dedicated Qwen2.5 VL 3B and Flux VAE; existing qualified native configuration baseline |
| Kandinsky5Image | Lite, 12.04 GB                    | Qwen2.5 VL, CLIP-L and Flux VAE; existing native Lite baseline                         |
| FluxSchnell     | Bundled FP8, 17.24 GB             | Header-confirmed embedded encoders/VAE; existing Schnell recipe                        |
| HiDream I1      | Dev and Fast FP8, 17.11 GB each   | Five exact external encoder/VAE roles; distinct Dev/Fast recipes                       |
| HiDream O1      | Bundled Dev FP8, 8.07 GB          | Native pixel-space/embedded behavior; existing O1 Dev recipe                           |
| MageFlow        | Base and Turbo INT8, 4.16 GB each | Qwen3-VL and dedicated Mage codec; distinct Base/Turbo recipes                         |
| Krea2           | Turbo FP8, 13.14 GB               | Qwen3-VL and RGB Qwen VAE; smaller than the reviewed 13.49 GB INT8 payload             |

Numbers are decimal published file sizes, not peak memory or a promise that a
specific GPU can run the workflow. No complete weight was downloaded or loaded
for this expansion. Publisher manifest hashes and HTTP byte-range header reads
establish source identity/reachability; native detector/matcher bodies on CPU
meta tensors establish header classification. They do not establish tensor value
integrity, generated semantics, speed or sustained device stability.

The official workflow annotation determines the folder/consumer, then the actual
header and native class prefixes establish bundled versus external components.
This avoids downloading companions already embedded in Flux/O1. All external
roles join exact existing registered artifact IDs. The original 34 legacy presets
remain unchanged. Two small OmniGen2/Kandinsky default resources expose the
existing native baseline's 1024 geometry and source provenance to the size reader.

Native class alone is insufficient to infer all components. Ovis weights also
match the native FluxSchnell class but require their dedicated OVIS conditioning
contract; they are deliberately absent from the Flux tuple here. Image-labelled shared-media entries without a connected RF image consumer are
retired by the [product scope cleanup](image-chat-scope.md).

## Preserving the original migration

This guide records the original bounded core catalogue pass. Current scoped
counts and retirements are owned by [image/Chat scope](image-chat-scope.md) and
the [registry schema](model-registry-schema.md). Historical migration hashes stay
recorded in the index; the splitter now verifies its explicitly retained
projection after retired media identities are removed. No deleted resource is
shipped as a hidden catalogue to satisfy the old baseline.

The core tuples and their actual retained image consumers remain present.
Former full-native missing-tuple and missing-operation counts are historical
receipts, not an implementation queue or acceptance target.

Local source receipts are under
`cache/plan-execution-20261111/publisher-payload-review/`: `summary.json`,
`native-classifications.json` and `wave-a-prepared.json`. They retain immutable
publisher URLs/revisions and exact header digests without shipping complete
upstream responses. Re-run source inspection through bounded metadata reads or
inspect a downloaded file explicitly; rendering a model card does neither.

```sh
python tools/split_model_registry.py
python tools/check_cpu.py test_native_core_catalogue.py test_model_registry.py test_model_registry_resources.py test_model_default_resources.py test_native_catalogue_dispositions.py test_model_acquisition.py
```
