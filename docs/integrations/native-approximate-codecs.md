# Full paired approximate image codecs

In **Models → Model → Image model components → Component choices for this
checkpoint**, select its VAE role and an installed **Approximate codec**. Save
the choice to use it on the next load. **Automatic** keeps the normal bundled or
family VAE. The approximate choice supplies both encoding and decoding for that
request, including image-to-image, inpainting and output decoding.

Place both files in a configured `vae_approx` root with the exact native alias
prefix, for example `taesdxl_encoder.safetensors` and
`taesdxl_decoder.safetensors`. Native alias lookup uses root-level names;
nested names do not match it. Configured root priority and native sorted filename
priority determine the pair. The application does not download these files.
The sampling-preview setting remains independent: a preview decoder alone never
becomes a full codec.

| Native alias | VAE slot           | Effective latent channels | Spatial factor | Image channels |
| ------------ | ------------------ | ------------------------- | -------------- | -------------- |
| `taesd`      | `vae_sd`           | 4                         | 8              | RGB            |
| `taesdxl`    | `vae_sdxl`         | 4                         | 8              | RGB            |
| `taesd3`     | `vae_sd3`          | 16                        | 8              | RGB            |
| `taef1`      | `vae_flux`         | 16                        | 8              | RGB            |
| `taef2`      | `vae_flux2`        | 128, packed 2×2           | 16             | RGB            |
| `taeqi2_1`   | `vae_qwen_image21` | 64                        | 16             | RGBA           |

The slot includes the native normalization and layout contract. Equal channel
counts in Mage, Wan, Layered, Hunyuan or another codec do not qualify a pair for
that model. Pixel-space models have no VAE role. Local Diffusers and SenseNova
bundles keep their fixed component contracts. A compatible approximate pair can
replace a bundled SDXL VAE explicitly; its bundled text encoder stays available.

## Reader and request boundary

`features/models/approximate_codecs.py` discovers pairs without loading weights.
Safetensors and `.sft` admission validates every native encoder/decoder key,
shape, floating dtype and exact nonoverlapping byte range. Headers are limited
to 1 MiB and each source to 512 MiB. Duplicate keys, excessive JSON depth,
truncated payloads, incomplete sides and sources escaping the configured root
are refused. The shape receipt is compared to the actual pinned native TAESD
modules in CPU tests. Cached header inspection is keyed by current file identity;
browsing does not hash or read the tensor payload.

Native weights-only Torch candidates retain an **Unverified** status during
browsing. Choosing them in the editor requires the existing advanced option.
At native loading, `comfy.utils.load_torch_file(..., safe_load=True)` uses the
pinned `torch.load(weights_only=True)` reader, and both complete dictionaries
must match the same native key/shape/dtype contract before constructing a VAE.
The generic model extension policy is unchanged.

`image_component_overrides` carries the namespaced choice, for example:

```json
{
  "image_component_overrides": {
    "vae_sdxl": "approx:taesdxl"
  }
}
```

This mapping is accepted by the existing image generation/edit API and saved
checkpoint profiles, presets and Library setup replay. `/v1/capabilities` lists
installed full pairs per image model under `approximate_codecs`, including the
slot, choice, compatibility, pair identity, both filenames and their summed
local bytes. Replay captures the current local pair; saved image metadata does
not carry trusted private objects or old file leases.

Submission freezes both canonical source paths, size and timestamps, plus
safetensors header digests. Preparation accounts for two local-only artifacts.
Both read leases stay held through native loading, including recursive bundled
checkpoint fallback. A queued pair is verified rather than silently recaptured
after a source change. Native cache reuse and the VAE patcher reload factory bind
the same pair identity. Missing, replaced or retargeted sources require a fresh
selection; cancellation exits reader scopes without publishing a new model.

`backends/comfy/approximate_codecs.py` executes the pinned
`nodes.VAELoader.load_taesd` function with a private folder namespace, reusing its
prefix assembly and all scale/shift contracts without changing global native
folder registration. It then creates ordinary `comfy.sd.VAE`, so existing native
encode/decode consumers receive the selected full codec. Bundled checkpoints
use `output_vae=False` for an explicit pair and retain native bundled CLIP loading.
Automatic uses ordinary checkpoint loader behavior.

## Pinned source and validation limits

Inspected 2026-11-11 at ComfyUI
`fca60fe70e5b84c5c6b058263653d15cb3401e44`:

- [VAELoader paired aliases and normalization](https://github.com/comfyanonymous/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/nodes.py#L776).
- [Native TAESD encoder, decoder and Flux2 packing](https://github.com/comfyanonymous/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy/taesd/taesd.py).
- [Native VAE branch and consumers](https://github.com/comfyanonymous/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy/sd.py#L559).

At this pin, `EncoderFlux2`/`DecoderFlux2` pack and unpack 2×2 latent pixels, but
the generic TAESD VAE branch leaves its spatial properties at 8. The adapter sets
only `taef2`'s VAE downscale/upscale properties to 16. Actual native CPU body
execution on tiny synthetic tensors proves the 128-channel, 16× geometry; tests
also exercise native VAE crop, encode and decode wrappers with synthetic zero
weights. The adapter preserves native weights, channel packing and normalization.
Qwen's native branch already supplies its 16× RGBA geometry and alpha padding.

Focused source/CPU checks:

```sh
python tools/check_cpu.py test_approximate_codecs.py test_approximate_codec_routes.py test_model_cache.py test_component_profiles.py test_component_overrides.py test_image_preparation.py
```

These cover all six actual native alias readers and normalization contracts,
native encode/decode with synthetic weights, paired reload/cancellation/read
leases, bundled SDXL selection versus Automatic, the component editor, public
capabilities, pre-queue API capture, preparation and Library/preset replay. No
published codec weights, image sampling, rendered-browser acceptance, generated
quality or GPU/device qualification is established by these checks.
