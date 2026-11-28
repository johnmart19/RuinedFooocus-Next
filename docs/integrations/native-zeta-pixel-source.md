# Native Zeta pixel-space sources

The pinned `ZImagePixelSpace` class consumes Zeta-Chroma's MLP pixel decoder.
The registered source is
[lodestones/Zeta-Chroma at 38ddc5f3e26fbbcb0ee4ff374b3f5cc384376d4b](https://huggingface.co/lodestones/Zeta-Chroma/tree/38ddc5f3e26fbbcb0ee4ff374b3f5cc384376d4b),
`zeta-chroma-base-x0-pixel-no-dino-1024.safetensors`, Apache-2.0. Its published
size is 12,997,743,288 bytes and SHA-256 is
`c652596db0e8b8211ae19b527295184ddfe99e60677fe1b10920ebc0d8572964`.
The 60,592-byte JSON header has SHA-256
`369a607fe9480b5523b89910618bf0221ebead4f5fdfe5f444e446cf3ca3e16e`.
Only the header was read; the full publisher digest was not recomputed locally.

Install the existing file under a configured `diffusion_models` or `checkpoints`
folder, then use **Create → Models → Refresh**. Its native family is
`ZImagePixelSpace`. The component panel uses the existing Qwen3-4B encoder slot
and pixel codec; no external VAE is required. Native header configuration gives
three RGB channels and a 32-pixel patch size. Header compatibility does not qualify
installed tensor values, memory fit or image quality.

Use Own settings or a complete saved checkpoint configuration. The local family
resource explicitly has `auto_recipe_supported=false`, no sampling values, no
recommended sizes and no default size. Its component contract is inspectable in
the normal model/default UI, while Auto remains unavailable until a recipe is
qualified. The `1024` publisher filename alone is not a sampling/geometry recipe.

## Why official L2P is a different source

The earlier catalogue linked
[zhen-nan/L2P at df14648875b75b880a639ba715e9b3ea2aaf2b4f](https://huggingface.co/zhen-nan/L2P/tree/df14648875b75b880a639ba715e9b3ea2aaf2b4f)
to the native pixel class. Primary-source review disproves a key-renaming adapter:

| Source                           | Decoder contract                                                                                                                                           |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Native Zeta / `ZImagePixelSpace` | `dec_net.cond_embed`, `input_embedder`, four AdaLN residual blocks and a linear output; the reviewed source predicts x0.                                   |
| Official L2P 1K                  | `local_decoder.enc*`, `pool*`, `bottleneck`, `up*`, `dec*`, `out_conv`; a convolutional U-Net consumes the full noisy image and spatial backbone features. |

[The official L2P decoder code](https://github.com/TencentYoutuResearch/T2I-L2P/blob/fea0d8f9852dfded83e39d0035f51733f5e58f6e/diffsynth/models/z_image_dit_L2P.py)
has SHA-256 `97f0ea4221936f58c6ca6c0dd59cac15346f712ef47d90a086c7e961526f839b`.
Its project-linked
[high-resolution source](https://github.com/HaojunChen663/PixVerve-95K/blob/91e352266128ff00af8ddf275b88c0826ff6f76f/L2P-ZImage-HR/diffsynth/models/z_image_dit_L2P.py)
also uses the U-Net decoder. The native pixel class originated in
[ComfyUI's Zeta change](https://github.com/Comfy-Org/ComfyUI/commit/9ebee0a2179b361a24c20838c1848d7988320636),
and its current implementation remains pinned at
`fca60fe70e5b84c5c6b058263653d15cb3401e44`.

`features/models/native_pixel_sources.py` recognizes the reviewed convolutional
topology from bounded tensor descriptors. Header classification no longer reports
these files as ordinary ZImage. Both native checkpoint and diffusion dispatch
refuse the known incompatible topology before model loading or cache reuse.
This guard uses tensor keys/shapes, so renaming the local file cannot bypass it.
Original weights are preserved. A real L2P implementation remains unsupported by
the pinned native engine; it is not represented as a completed conversion.

## Verification

`tests/fixtures/zeta-l2p-source-headers.json.gz` retains exact headers for three
publisher Zeta variants and official L2P 1K, without tensor data. Its SHA-256 is
`9e7afd274de0fce2c5c311c9c01ce13c568aac50ffb8cf15275bbfc882a2060f`.
Tests call the actual pinned classifier on all three Zeta headers and prove that
the pinned Diffusers converter alone incorrectly classifies official L2P as
ordinary ZImage. Production guards, native loader dispatch, cache admission,
encoder/pixel-codec inventory and the disabled Auto contract are checked on CPU.

```sh
python tools/check_cpu.py test_native_pixel_sources.py test_native_extended_catalogue.py test_native_catalogue_dispositions.py test_model_default_resources.py test_model_registry_resources.py test_native_recipe_contracts.py test_component_profiles.py test_image_components.py test_model_recommendations.py test_model_cache.py test_image_settings.py
```

These are source and CPU reader checks. No real-model inference, browser rendering,
visual output, speed, GPU stability or device-memory requirement is established.
