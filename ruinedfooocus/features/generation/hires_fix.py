"""Native ComfyUI two-pass Latent Hires Fix (LatentUpscaleBy + second KSampler pass)."""

HIRES_UPSCALE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "bislerp")
MAX_COMFY_SEED = (1 << 63) - 1


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
):
    """Upscale the latent with ComfyUI's LatentUpscaleBy and refine with a second KSampler pass."""
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

    upscaled = nodes.LatentUpscaleBy().upscale(
        samples=sampled_latent,
        upscale_method=method,
        scale_by=scale,
    )[0]
    if check_cancelled is not None:
        check_cancelled()

    latent_image = upscaled["samples"]
    batch_inds = upscaled.get("batch_index")
    noise = comfy.sample.prepare_noise(latent_image, second_seed, batch_inds)
    refined = comfy.sample.sample(
        model=model,
        noise=noise,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
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
