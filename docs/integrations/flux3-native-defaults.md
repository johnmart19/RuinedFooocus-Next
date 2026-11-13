# FLUX 3 native component and default selection

The pinned ComfyUI revision `989d93d0d968e704c589ce0146bb18c77f6b50cc`
recognizes FLUX 3 diffusion tensors. RuinedFooocus reuses its `Flux3` loader,
`CLIPType.FLUX3`, FLUX.2 latent creation, sampler, schedule and device management.
The cached Civitai label `Flux.3 Klein` maps to the existing `Flux3` component
slots: Qwen3-VL 4B and the FLUX.2 VAE. Passive inventory imports no native runtime.
This alias does not identify FLUX.2 Klein or certify an arbitrary weight file.

**Model Defaults** chooses the distilled recipe only after reading a local
safetensors header whose four-part file signature still matches its model
metadata. The FLUX 3 attention and text-modulation shapes identify the family;
`ref_time_in.in_layer.weight` identifies the flow-map variant. Filename words
such as `base` or `distilled` do not choose this recipe. The fallback is 15 steps,
CFG 1 (positive-only in ComfyUI), ER-SDE, Simple, Clip Skip 1, matching the current
official Comfy distilled workflow. Native sampling
retains the pinned backend's timestep shift and next-sigma conditioning.
Specific published recommendations and saved per-model overrides retain their
existing precedence. **Own settings** and explicitly selected presets remain
manual choices. Base, unknown, unsupported formats or changed/missing local
identities have no guessed Auto recipe; choose explicit settings or a complete
saved per-model override.

Source evidence, checked 2026-11-09:

- [Official Comfy distilled template at `369c1ed`](https://github.com/Comfy-Org/workflow_templates/blob/369c1ed12496587bbce50a7797ca342da170e5f7/templates/image_flux3_klein_distilled_t2i.json)
  sets its executable KSampler to 15/CFG 1/ER-SDE/Simple. Its SHA256 is
  `253fa4e783da734acd03b804a8731d0a7a8c564b73961837762800eb5cf85388`.
  The pinned native sampler registry includes ER-SDE; its implementation uses
  native flow-compatible sigma/SNR handling, callbacks and seeded noise. No
  separate solver or schedule implementation is added.
- [BFL reference pipeline at `62e90b9`](https://github.com/black-forest-labs/flux-3-image-klein/blob/62e90b9278b73ecc52927770c6965dc6b1966498/src/flux3/pipeline.py)
  selects 16 steps without CFG for distilled. Its
  [flow-map sampler](https://github.com/black-forest-labs/flux-3-image-klein/blob/62e90b9278b73ecc52927770c6965dc6b1966498/src/flux3/sampling.py)
  uses Euler updates with the next target time. This is a separate reference
  recipe; RF follows the current official native Comfy template and does not
  claim identical sigmas or pixels to BFL's Python implementation.
- [BFL's Comfy model card](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)
  identifies the components and distilled 16-step/no-guidance behavior. It
  recommends a complete scene caption. Write that caption or explicitly use an
  existing prompt enhancement workflow; this change acquires no upsampler and
  does not silently rewrite prompts.
- The downloaded model's exact cached Civitai version `3407437` has one example
  with 15 steps and custom sampler/scheduler names. One example and unavailable
  custom nodes do not establish a native fallback.
- Fresh [official Comfy template source at `369c1ed`](https://github.com/Comfy-Org/workflow_templates/blob/369c1ed12496587bbce50a7797ca342da170e5f7/templates/image_flux3_klein_distilled_t2i.json)
  currently uses 15/CFG 1/ER-SDE/Simple for both distilled and Base templates;
  that similarity is not proof the distilled template is defective. Older indexed
  templates show variant-specific settings. BFL's Base guidance
  also differs between the Comfy card and reference code. RF therefore does not
  promote those conflicting Base settings into Auto.

Focused CPU checks: `python tools/check_cpu.py test_image_components.py
test_model_recommendations.py test_generation_defaults.py test_model_resource_ui.py`.
They cover passive aliasing, header variant and replacement, request application,
published/saved/manual precedence and refusal of ambiguous Auto. They do not
qualify FLUX 3 inference, visual quality, native edits/layout or GPU stability.
