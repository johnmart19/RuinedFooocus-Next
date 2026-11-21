"""ComfyUI's per-node cleanup, which the embedded application must run itself.

Upstream ComfyUI runs this after every node in `execution.py` when Dynamic VRAM
(comfy_aimdo) is active. RuinedFooocus calls the native nodes directly and never
reached it. Without it `comfy.model_management.LARGEST_AIMDO_CASTED_WEIGHT`
keeps a layer of the previous model, and through that layer's `_v` handle the
model's whole Dynamic VRAM pool stays on the GPU after the model is replaced.
Measured on an RX 7900 XTX (2026-11-21): each switch stranded one pool
(5.5 GB for Anima 2.9B); after four models VRAM was full and Krea 2 took 103-111 s
per image instead of 15 s. Watermark limits and prefetch queues were never
reset either.
"""


def end_native_step():
    """Run upstream's post-node cleanup; safe between jobs and before model changes."""
    try:
        import comfy.memory_management as memory
    except ImportError:
        return False
    if not getattr(memory, "aimdo_enabled", False):
        return False
    import comfy.model_management as management
    import comfy.model_prefetch as prefetch
    import comfy_aimdo.model_vbar as vbar

    prefetch.cleanup_prefetch_queues()
    management.reset_cast_buffers()
    vbar.vbars_reset_watermark_limits()
    return True
