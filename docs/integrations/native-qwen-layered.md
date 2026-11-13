# Qwen Layered RGBA generation

Select a currently identified Qwen Image Layered checkpoint in Create. Its
**Qwen Layered** panel sets **Output layers** and an optional **Layer source**.
Describe the overall scene; the prompt does not name or reorder individual
layers. **Image count** remains the number of independent generations. Two
layers with three generations produce six ordered RGBA files from three sampler
calls. Use PNG or EXR and turn automatic upscaling off. Layered currently keeps
the existing explicit sampling controls; this feature does not enable its Auto
recipe or qualify any GPU, quantization or visual result.

The panel also follows a checkpoint selected by **Saved settings (PNG)** model
replay. Changing the effective model or refreshing its metadata keeps the
uploaded layer-source draft. An inactive Layered panel does not apply that source
to another model; submission validates it when Layered becomes active again.

The API adds strict integer `qwen_layer_count` (1–16). `n` remains the repetition
count; `input_image` supplies the optional RGBA source through the existing
bounded image decoder. Other masks, additional references and image tools are
refused. Transparent still AVIF is refused because its current native writer
does not retain alpha. The output budget is 16 megapixels across every layer and
generation, including the aligned native canvas and final target. Repetitions
must be finite (1–100). Source images are also limited to 16 megapixels.

## Capture, components and cache

`generation/layered.py` freezes EXIF-oriented RGBA bytes and the checkpoint's
current signature before the existing preparation/compute queues. Later widget,
profile or upload changes cannot mutate the queued reference. Unknown or base
Qwen files do not gain Layered behavior from their filename.

The native detector's positive Layered configuration is retained through model
inspection: `use_additional_t_cond=True` and
`default_ref_method="negative_index"`. These flags come from the actual
additional timestep embedding and establish the Layered variant even when a
safetensors/SFT/GGUF file has no scalar variant tag. The Layered recipe requires
both flags; a scalar label alone cannot turn an ordinary Qwen checkpoint into
Layered, and a conflicting scalar variant is refused. This admits the matched
manual workflow while leaving `artifact_qualified` and Auto support false.

`models/native_component_layout.py` supplies the distinct
`vae_qwen_image_layered` role to component inventory, submission snapshots,
preparation and `image_pipeline.load_base_model`. The native Qwen text encoder
role remains `clip_qwen25`. The checkpoint and component signatures, chosen
roles and recipe identity participate in cache reuse. Loading validates the
native codec's 16 latent channels, four output channels and 8× encode/decode
scale. Ordinary Qwen RGB and Qwen Image 2.1's 64-channel RGBA codecs cannot fill
this slot. Existing preparation read leases cover external component loading;
this feature adds no device manager or model download mechanism.

## Native data flow

At ComfyUI `fca60fe70e5b84c5c6b058263653d15cb3401e44`, RF calls:

1. `EmptyQwenImageLayeredLatentImage.execute(width, height, layers, batch_size=1)`
   for `[1,16,layers+1,H/8,W/8]` target samples on a 16-pixel aligned canvas.
2. With a source, `VAEEncode.encode` once with four channels, followed by
   `ReferenceLatent.execute` on both text branches, including scheduled prompts.
   Source proportions and alpha are preserved by the shared framing contract.
3. `ModelSamplingAuraFlow.patch_aura(shift=1)` when an explicit sampling shift has
   not already been selected, then the existing native sampler.
4. `LatentCut.execute(sampled, "t", 1, layers)` removes reconstruction slot zero;
   `LatentCutToBatch.execute(..., "t", 1)` turns slots 1 through `layers` into
   ordered single-frame batches. The existing native VAE decoder preserves RGBA.
5. Source/alignment padding is removed. Each output is uniformly fitted to the
   captured target with transparent padding when necessary, then returned as
   `ImageOutputBatch`. The worker records layer index/count/latent slot separately
   from generation index and serializes each original output through the normal
   PNG/precision transport.

The installed official source recipe
`comfyui_workflow_templates_json==0.1.129/templates/image_qwen_image_layered.json`
was inspected on 2026-11-11. Its source and text-only subgraphs establish the
paired reference-conditioning, shift, latent-cut and decode order. RF implements
this named operation directly; workflow JSON is not executed as an API graph.
Exact native/template hashes and changed-source identities are retained in
`cache/plan-execution-20261111/W07-G3-layered-generation.json`.

## Metadata, replay and verification

Saved images include `qwen_layer_count`, safe `native_layered_conditioning`,
`native_component_layout` and the existing per-output group records. RGBA source
bytes and private capture tokens are never serialized. Text-only metadata can
replay the layer count directly. Source-dependent metadata requires reattaching
the exact original RGBA image; a hash cannot recover a missing upload. A changed
checkpoint or component must be resubmitted and revalidated.

Feature `native-qwen-layered-generation` declares guarded CPU checks for the
actual API/preparation, native bodies, complete weightless production caller,
loader/cache invalidation, real Gradio callback, worker repetitions, PNG alpha
and replay metadata. Use the project interpreter with
`tools/check_cpu.py test_layered_generation.py` and the descriptor's caller tests.
CPU/source checks do not certify real-model semantics, output quality, browser
rendering, speed or GPU stability. No generation, GPU probe, install or download
is part of this evidence.
