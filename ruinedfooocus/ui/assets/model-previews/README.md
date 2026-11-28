# Category artwork

These sixteen conceptual category illustrations were created with the hosted
OpenAI image generator on 12 November 2026: one image per role, with no variants
or local model execution. They are default artwork, not publisher images or
examples produced by the selected model. The originals remain outside the source
tree; the packaged versions are 512 × 512 progressive JPEGs, optimized at quality 86. They contain no language text, people or model branding.

The shared direction is a charcoal studio with sculptural graphite, smoked glass,
porcelain and restrained warm-orange light. A strong central silhouette keeps
the category useful at thumbnail size.

| File                      | Role and motif                                               |
| ------------------------- | ------------------------------------------------------------ |
| `image-model.jpeg`        | Image model — creative aperture and folded image planes      |
| `chat-model.jpeg`         | Chat model — paired conversational ribbons                   |
| `vision-projector.jpeg`   | Vision/projectors — optical prism and projection plane       |
| `text-encoder.jpeg`       | Text encoder — unmarked token tiles converging on a core     |
| `vision-encoder.jpeg`     | Vision encoder — lens and sensor                             |
| `vae.jpeg`                | VAE — compression and reconstruction through a narrow waist  |
| `lora.jpeg`               | LoRA — adapter ring around a core                            |
| `embedding.jpeg`          | Embedding — connected vector constellation                   |
| `controlnet.jpeg`         | ControlNet — sculpture constrained by a wireframe cage       |
| `model-patch.jpeg`        | Model patch — fitted modular inlay                           |
| `upscaler.jpeg`           | Upscaler — coarse-to-fine crystal steps                      |
| `latent-upscaler.jpeg`    | Latent upscaler — expanding volumetric lattice               |
| `depth.jpeg`              | Depth — layered contour terraces                             |
| `background-removal.jpeg` | Background removal — object separated from its cutout        |
| `face-restoration.jpeg`   | Face restoration — abstract restoration lens, without a face |
| `model-component.jpeg`    | Other components — interlocking modular cube                 |

`features/models/resource_cards.py` maps the catalogue's actual category labels
and singular adapter names to these files. The Detection category (the face
detector) has no illustration of its own yet and uses the vision encoder image.
Unknown labels use the component illustration. Exact local preview sidecars and valid cached publisher artwork
retain priority. The UI supplies a localized **Default preview** label; no label
is baked into the images.

`default_preview(kind, cache_root)` publishes only into the existing allowed
`cache/default-previews` leaf. Its filename includes the bundled artwork's hash,
so an updated asset gets a fresh URL without overwriting old defaults or touching
publisher images. Corrupt current defaults are repaired through the existing
static-thumbnail writer. Lookups are local, bounded and independent of weights,
downloads and publisher requests.

Adapter callers pass `fallback_cache_dir=cache_root` to `resource_card` or
`resource_records`, separately from their per-kind metadata cache. The returned
`fallback` path supports browser decode failure; `preview_is_default` identifies
the initial fallback. Callers without that explicit root retain their existing
Pillow fallback compatibility. Models UI owns localized labels and rendering.
