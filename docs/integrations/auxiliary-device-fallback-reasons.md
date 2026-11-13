# Auxiliary device fallback reasons

Captioning and Chat text embeddings retain their native Auto/GPU policy. The
existing DirectML branches use CPU because Transformers execution on that Torch
backend is not qualified. Caption device selection now reports that precise
capability reason before execution, without attempting a DirectML model load.

Chat captures the DirectML reason with its request-local embedding preparation.
Only the actual txtai runtime-options boundary logs the reason; attachment
capture and Settings preparation stay passive. The reason neither changes
placement nor writes a setting. An explicitly selected embedding CPU policy is
recorded as a deliberate choice and does not claim an accelerator failure.

There is no new Transformers Vulkan or Windows ML implementation. CUDA/ROCm and
other supported native Auto paths retain their existing selection and offload
behavior. Focused checks use mocked runtimes and tiny local metadata fixtures:
`python tools/check_cpu.py test_auxiliary_device_fallback_reasons.py
test_caption_runtime.py test_rag_preparation.py test_rag_settings.py`.
They do not load real models or qualify GPU behavior.
