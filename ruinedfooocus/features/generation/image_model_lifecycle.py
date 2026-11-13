"""Shared native model replacement boundary for image pipelines."""


def reset_image_models(owner):
    """Drop primary, patched and companion references before loading replacements."""
    for name in (
        "xl_base",
        "xl_base_hash",
        "xl_base_patched",
        "xl_base_patched_hash",
        "conditions",
        "model_info",
        "load_error",
        "cascade_stage_b_cache",
        "cascade_guidance_cache",
        "sdxl_refiner_cache",
        "native_negative_model_cache",
        "reference_adapter_models",
        "_bernini_plan",
        "_bernini_loras",
    ):
        setattr(owner, name, None)
    if getattr(owner, "_native_gligen", None) is not None:
        from ruinedfooocus.features.image_editing.gligen_layout import clear_cache

        clear_cache(owner)
    if getattr(owner, "_native_hypernetwork", None) is not None:
        from ruinedfooocus.features.image_editing.native_hypernetwork import clear_cache

        clear_cache(owner)


def collect_image_models(memory=None):
    """Use the existing native cleanup sequence after application references drop."""
    import gc

    if memory is None:
        import comfy.model_management as memory

    gc.collect(generation=2)
    memory.cleanup_models()
    memory.soft_empty_cache()
