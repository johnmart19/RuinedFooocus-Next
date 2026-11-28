# Auxiliary device fallback reasons

Captioning and Chat text embeddings retain their native Auto/GPU policy.
Caption device selection follows the native image device, so a CPU image
backend captions on CPU.

Chat captures its CPU policy with its request-local embedding preparation;
attachment capture and Settings preparation stay passive. Global CPU and an
explicitly selected embedding CPU policy are deliberate choices and do not
claim an accelerator failure.

There is no new Transformers Vulkan or Windows ML implementation. CUDA/ROCm and
other supported native Auto paths retain their existing selection and offload
behavior. Focused checks use mocked runtimes and tiny local metadata fixtures:
`python tools/check_cpu.py test_auxiliary_device_fallback_reasons.py
test_caption_runtime.py test_rag_preparation.py test_rag_settings.py`.
They do not load real models or qualify GPU behavior.
