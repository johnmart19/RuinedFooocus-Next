"""Image VAE decoding through native nodes and their memory estimates."""


def decode_image_vae(models, samples, *, tiled=False, tile_size=None, overlap=None):
    from comfy import model_management as memory
    from nodes import VAEDecode, VAEDecodeTiled

    vae = models.vae
    latent = samples["samples"]

    def decode():
        use_tiles = tiled
        budget = None
        image_latent = not latent.is_nested and (
            latent.ndim == 4 or (latent.ndim == 5 and latent.shape[2] == 1)
        )
        if vae.device.type != "cpu" and image_latent:
            budget = (
                memory.get_total_memory(vae.device)
                - memory.extra_reserved_memory()
                - vae.patcher.model_size()
            )
            use_tiles |= vae.memory_used_decode(latent.shape, vae.vae_dtype) > budget
        if not use_tiles:
            return VAEDecode().decode(vae=vae, samples=samples)[0]
        size = tile_size or 512
        if tile_size is None and budget is not None:
            compression = vae.spacial_compression_decode()
            for size in (512, 256, 128, 64):
                shape = vae._tile_bounded_shape(
                    latent.shape, size // compression, size // compression, 1
                )
                if vae.memory_used_decode(shape, vae.vae_dtype) <= budget:
                    break
        return VAEDecodeTiled().decode(
            vae=vae, samples=samples, tile_size=size, overlap=64 if overlap is None else overlap
        )[0]

    try:
        return decode()
    except Exception as error:
        memory.raise_non_oom(error)
    # Leave the exception scope before retrying, releasing failed decoder tensors.
    print("VAE decoding needs more VRAM; releasing diffusion weights and retrying.")
    memory.unload_model_and_clones(models.unet)
    memory.soft_empty_cache()
    return decode()
