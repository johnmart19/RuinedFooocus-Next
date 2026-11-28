# Boogu native image editing

With a Boogu **Edit** checkpoint selected, use **Image tools → Native image edit**.
Automatic requires a current identified Edit variant. Boogu Turbo is a separate
text-to-image variant. Explicit **Boogu Edit** admits compatible uncurated Edit
weights without assigning publisher or hardware qualification.

Upload the primary image first, followed by up to fifteen extra references.
One reference is the native recommendation. The existing source framing, prompt,
negative, LoRA, size and sampling controls remain active. **Model Defaults** uses
25 steps, CFG 3.5, DPM++ 2M and Simple; the native model supplies shift 3.16.
**Image to image** stays available as an explicit denoise operation.

`ordered_native_references.py` calls the pinned `TextEncodeBooguEdit.execute`.
The node sends instruction vision tokens only to positive conditioning, while
the same ordered VAE reference latents attach to both signs. The native node owns
its 384-square vision and 1024-square reference pixel budgets. RF preserves the
extra images' aspect ratios, owns private input copies and reuses VAE encodings
only within the request and its scheduled positive prompts. The target is the
existing empty latent; native editing does not fall back to image-to-image.

The API accepts `native_edit_recipe="boogu-edit"`, an unmasked `input_image` and
up to fifteen `reference_images`. Automatic uses the current source identity.
The UI and API share native edit admission. PNG metadata records the applied
recipe, reference order/dimensions and pixel hashes. Full-setup Library replay
restores settings and clears uploaded references as before.

Source: ComfyUI `989d93d0d968e704c589ce0146bb18c77f6b50cc`,
`comfy_extras/nodes_boogu.py`, `comfy/supported_models.py`, and
`blueprints/Edit Image (Boogu Image Edit).json`, inspected locally 2026-11-12.
No loader, sampler or device manager is duplicated.

Focused checks: `python tools/check_cpu.py test_boogu_native_edit.py
test_ordered_native_references.py test_api_image_edits.py test_image_guidance.py`.
Fixtures execute the actual pinned conditioning body and production caller with
tiny CPU tensors, including negative isolation, scheduled references and source
mutation refusal. These establish source/caller behavior; real-model output,
rendered browser interaction, visual quality and GPU stability are unqualified.
