"""Keep LoRA-patched text encodings independent of what ran before.

With DynamicVRAM, the pinned ComfyUI applies LoRA patches lazily while a
module's weights are first cast to the device. That first forward pass after
the text encoder is (re)loaded produces slightly different numbers than later
passes, so the same prompt and seed could give a different image depending on
whether the encoder was still loaded. One empty encode settles the patched
weights before the real prompt is encoded.
"""

import logging


def _resident(patcher):
    import comfy.model_management as management

    return any(loaded.model is patcher for loaded in management.current_loaded_models)


def warm_patched_text_encoder(clip):
    """Run one empty encode when a LoRA-patched encoder is not loaded yet."""
    patcher = getattr(clip, "patcher", None)
    if patcher is None or not getattr(patcher, "patches", None):
        return False
    try:
        import comfy.memory_management as memory

        if not getattr(memory, "aimdo_enabled", False):
            return False
    except ImportError:
        return False
    if _resident(patcher):
        return False
    from nodes import CLIPTextEncode

    try:
        CLIPTextEncode().encode(clip=clip, text="")
    except Exception as error:  # The real encode reports genuine failures.
        logging.warning("Text encoder warm-up skipped: %s", error)
        return False
    return True
