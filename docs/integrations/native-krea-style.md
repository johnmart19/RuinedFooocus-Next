# Krea2 style-reference images

Select Krea2, then **Image tools → Native image edit → Edit recipe**. Choose
**Krea2 style reference**, upload one to three ordered reference images, and select
the required **Krea2 style LoRA**. Automatic editing chooses this native recipe;
ordinary Krea2 text-to-image does not activate it. Its default style strength is
**1**. The output canvas uses the selected generation size independently of the
references, whose aspect ratios are preserved for native preprocessing.

The default adapter is `krea2_style_reference.safetensors`. Its registered
publisher tuple uses revision `588feaf955b08b9ac4723eeb751752ab045dba71`,
457,111,760 bytes and SHA-256
`f50df5a9e62e4be8aa926a63dd5bb1a64770c4004f763c1208007ae13daa82b8`.
Missing registered assets join the existing preparation/readiness service on
submission. Offline mode does not acquire files. Explicit compatible local
adapters remain unqualified and are validated by the native loader at use.

The native graph uses `TextEncodeQwenImageEditPlus` on the Krea2 Qwen3-VL 4B
encoder and Qwen RGB VAE, followed by the native `index_timestep_zero` reference
method. Empty negatives use native `ConditioningZeroOut`, matching the blueprint.
An explicitly supplied negative is encoded through the same native reference
method and retained in metadata. Its effect depends on the requested CFG; the
published CFG 1 recipe does not amplify the negative branch. **Own settings**
preserves explicit steps, CFG, sampler and scheduler.

**Model Defaults** uses eight Euler steps, CFG 1 and Simple. The request clones
its model and calls native `ModelSamplingFlux` with max shift 1.15, base shift
0.5 and the actual canvas. Native `LoraLoaderModelOnly` applies the required style
adapter without changing the text encoder. A prepared-file read lease covers the
whole operation. Success, failure and cancellation restore the exact original
model, so later ordinary images and replacement LoRAs cannot inherit this patch.

The API fields are `native_edit_recipe="krea2-style"`, `input_image`, up to two
additional `reference_images`, optional `krea_style_lora` and
`krea_style_strength` (-10 through 10). Omitted style fields use the registered
default and strength 1. **Save current defaults** captures both style settings;
startup validates the saved relative filename and finite strength before restoring
the controls. New profiles use the registered adapter name and strength 1 without
activating a style operation. Uploads and prepared runtime objects are never saved.
Full-setup Library replay
restores the recipe and style settings while requiring new reference uploads.
Inactive UI fields are omitted from other queued operations; incompatible
explicit API settings fail. No Chat model or automatic prompt rewrite is used.

Source: ComfyUI `989d93d0d968e704c589ce0146bb18c77f6b50cc`,
`blueprints/Image Style Reference (Krea-2 Turbo).json`, `nodes_qwen.py`,
`nodes_model_advanced.py` and `nodes.LoraLoaderModelOnly`. Official publisher
metadata was read 2026-11-12; model bytes were not downloaded or executed.

Focused checks: `python tools/check_cpu.py test_krea_style_reference.py
test_ordered_native_references.py test_api_image_edits.py test_image_guidance.py`.
They exercise pinned native conditioning/zeroing with tiny CPU tensors, model-only
scope restoration, capture, API transport and direct callers. Browser rendering,
actual adapter semantics, image quality, performance and GPU stability remain
separate qualification gates.
