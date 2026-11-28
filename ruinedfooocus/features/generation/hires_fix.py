"""Native ComfyUI two-pass Hires Fix (Latent or pixel-upscaler + second KSampler pass)."""

HIRES_UPSCALER_LATENT = "Latent"
HIRES_UPSCALE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "bislerp")
MAX_COMFY_SEED = (1 << 63) - 1


def hires_upscaler_choices():
    """Return 'Latent' followed by all available pixel upscalers (neural models and AMD FSR)."""
    from ruinedfooocus.features.generation import automatic_upscale

    return [HIRES_UPSCALER_LATENT] + [
        choice
        for choice in automatic_upscale.choices()
        if choice != "Off" and not choice.startswith("4× AMD FSR ")
    ]


def resolve_hires_upscaler(selection):
    """Return (name, native_scale) for a pixel hires upscaler, or (None, 1) for Latent."""
    if selection in (None, "", HIRES_UPSCALER_LATENT, "Off"):
        return None, 1
    from ruinedfooocus.features.generation import automatic_upscale

    return automatic_upscale.resolve(selection)


def _upscale_pixels_for_hires(
    decoded,
    *,
    upscaler_name,
    native_scale,
    target_w,
    target_h,
    owner=None,
    gen_data=None,
    check_cancelled=None,
):
    """Upscale decoded [B, H, W, C] pixels with AMD FSR or a native ComfyUI pixel upscaler."""
    import torch
    from ruinedfooocus.features.generation import automatic_upscale

    fsr_version = automatic_upscale.fsr_version_for(upscaler_name)
    if fsr_version is not None:
        from ruinedfooocus.features.image_processing import amd_fsr

        frames = [
            amd_fsr.upscale(
                fsr_version,
                decoded[index : index + 1],
                target_w,
                target_h,
                check_cancelled=check_cancelled,
            )
            for index in range(decoded.shape[0])
        ]
        result = torch.cat(frames, dim=0) if len(frames) > 1 else frames[0]
    else:
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
        from ruinedfooocus.features.image_processing.upscale_pipeline import (
            InterruptibleUpscaler,
            pipeline as create_upscale_pipeline,
        )

        upscale_pipe = (
            getattr(owner, "automatic_upscale_pipeline", None) if owner is not None else None
        )
        if upscale_pipe is None:
            upscale_pipe = create_upscale_pipeline()
            if owner is not None:
                owner.automatic_upscale_pipeline = upscale_pipe
        model = upscale_pipe.load_upscaler_model(upscaler_name, gen_data or {})
        if getattr(model, "scale", native_scale) != native_scale:
            raise ValueError(f"Unexpected scale for hires upscaler {upscaler_name}: {model.scale}")
        if getattr(model, "input_channels", 3) != 3 or getattr(model, "output_channels", 3) != 3:
            raise ValueError("Hires fix pixel upscaling requires a native RGB upscaler.")
        result = ImageUpscaleWithModel.execute(
            InterruptibleUpscaler(model, gen_data or {}), decoded
        )[0]

    if int(result.shape[1]) != target_h or int(result.shape[2]) != target_w:
        import comfy.utils

        result = (
            comfy.utils.common_upscale(
                result.movedim(-1, 1), target_w, target_h, "lanczos", "disabled"
            )
            .movedim(1, -1)
            .clamp(0.0, 1.0)
        )
    return result


def resolve_hires_seed(base_seed, hires_seed):
    """Return an explicit second-pass seed or derive a deterministic one from base_seed."""
    if hires_seed is not None and int(hires_seed) >= 0:
        return int(hires_seed)
    return (int(base_seed) + 1) & MAX_COMFY_SEED


def scale_reference_frame(plan, before_shape, after_shape):
    """Scale a pass-1 reference_frame geometry dict to match the upscaled latent canvas."""
    if plan is None:
        return None
    before_h, before_w = int(before_shape[-2]), int(before_shape[-1])
    after_h, after_w = int(after_shape[-2]), int(after_shape[-1])
    if before_h <= 0 or before_w <= 0 or (before_h == after_h and before_w == after_w):
        return plan
    scale_y = after_h / before_h
    scale_x = after_w / before_w
    canvas_w = round(plan["canvas_width"] * scale_x)
    canvas_h = round(plan["canvas_height"] * scale_y)
    left = round(plan["left"] * scale_x)
    top = round(plan["top"] * scale_y)
    width = min(canvas_w - left, max(1, round(plan["width"] * scale_x)))
    height = min(canvas_h - top, max(1, round(plan["height"] * scale_y)))
    updated = dict(plan)
    updated.update(
        canvas_width=canvas_w,
        canvas_height=canvas_h,
        left=left,
        top=top,
        width=width,
        height=height,
    )
    return updated


def apply_latent_hires_fix(
    sampled_latent,
    *,
    model,
    positive,
    negative,
    cfg,
    sampler_name,
    scheduler,
    base_seed,
    guidance,
    callback=None,
    disable_pbar=False,
    check_cancelled=None,
    vae=None,
    decode_fn=None,
    owner=None,
    gen_data=None,
):
    """Upscale Pass-1 output (latent or pixel upscaler) and refine with a second KSampler pass."""
    if not guidance.get("hires_enabled"):
        return sampled_latent
    if check_cancelled is not None:
        check_cancelled()

    import comfy.sample
    import nodes

    method = guidance["hires_upscale_method"]
    scale = float(guidance["hires_scale"])
    steps = int(guidance["hires_steps"])
    denoise = float(guidance["hires_denoise"])
    second_seed = resolve_hires_seed(base_seed, guidance.get("hires_seed"))
    upscaler_name, native_scale = resolve_hires_upscaler(guidance.get("hires_upscaler"))

    if upscaler_name is None:
        if gen_data is not None:
            gen_data.pop("hires_upscaler_model", None)
        upscaled = nodes.LatentUpscaleBy().upscale(
            samples=sampled_latent,
            upscale_method=method,
            scale_by=scale,
        )[0]
    else:
        if vae is None or decode_fn is None:
            raise ValueError("Pixel-upscaler hires fix requires an active VAE and decoder.")
        decoded = decode_fn(sampled_latent)
        if check_cancelled is not None:
            check_cancelled()
        orig_h, orig_w = int(decoded.shape[1]), int(decoded.shape[2])
        align = max(1, int(getattr(vae, "spacial_compression_decode", lambda: 8)()))
        target_w = max(align, round((orig_w * scale) / align) * align)
        target_h = max(align, round((orig_h * scale) / align) * align)
        upscaled_pixels = _upscale_pixels_for_hires(
            decoded,
            upscaler_name=upscaler_name,
            native_scale=native_scale,
            target_w=target_w,
            target_h=target_h,
            owner=owner,
            gen_data=gen_data,
            check_cancelled=check_cancelled,
        )
        if check_cancelled is not None:
            check_cancelled()
        if guidance.get("vae_tiled") and hasattr(nodes, "VAEEncodeTiled"):
            upscaled = nodes.VAEEncodeTiled().encode(
                vae=vae,
                pixels=upscaled_pixels,
                tile_size=int(guidance.get("vae_tile_size") or 512),
                overlap=int(guidance.get("vae_overlap") or 64),
            )[0]
        else:
            upscaled = nodes.VAEEncode().encode(vae=vae, pixels=upscaled_pixels)[0]
        if gen_data is not None:
            gen_data["hires_upscaler_model"] = upscaler_name
    if check_cancelled is not None:
        check_cancelled()

    latent_image = upscaled["samples"]
    batch_inds = upscaled.get("batch_index")
    noise = comfy.sample.prepare_noise(latent_image, second_seed, batch_inds)
    hires_scheduler = scheduler
    try:
        import comfy.samplers

        valid_schedulers = getattr(getattr(comfy.samplers, "KSampler", None), "SCHEDULERS", None)
        if valid_schedulers and hires_scheduler not in valid_schedulers:
            hires_scheduler = "simple"
    except Exception:
        pass
    refined = comfy.sample.sample(
        model=model,
        noise=noise,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=hires_scheduler,
        positive=positive,
        negative=negative,
        latent_image=latent_image,
        denoise=denoise,
        start_step=0,
        last_step=steps,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=second_seed,
    )
    result = upscaled.copy()
    result["samples"] = refined
    return result
