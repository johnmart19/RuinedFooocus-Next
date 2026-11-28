"""Native image sampling options; model recipes never replace explicit values."""

LCM_FIELDS = {
    "lcm_noise": "s_noise",
    "lcm_noise_end": "s_noise_end",
    "lcm_noise_clip": "noise_clip_std",
}


def hidream_o1_dev(checkpoint):
    """Share the sampling recipe's current source declaration with native patches."""
    import sys
    from ruinedfooocus.features.models.model_recommendations import ModelRecommendations
    from ruinedfooocus.features.models.performance import PerformanceSettings

    state = sys.modules.get("ruinedfooocus.app.state")
    performance = getattr(state, "performance_settings", None)
    if not isinstance(performance, PerformanceSettings):
        performance = PerformanceSettings.__new__(PerformanceSettings)
        performance.model_recommendations = ModelRecommendations()
    return performance._declared_variant(checkpoint, "HiDream O1") == "dev"


def lcm_options(values, family, checkpoint=None):
    clip = 2.5 if family in ("HiDreamO1", "HiDream O1") and hidream_o1_dev(checkpoint) else 0.0
    defaults = dict(s_noise=1.0, s_noise_end=1.0, noise_clip_std=clip)
    return {
        native: defaults[native] if values.get(key) is None else values[key]
        for key, native in LCM_FIELDS.items()
    }


def sample_lcm(
    *,
    model,
    noise,
    steps,
    cfg,
    scheduler,
    positive,
    negative,
    latent_image,
    denoise,
    noise_options,
    noise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    from comfy.sample import sample_custom
    from comfy_extras.nodes_custom_sampler import BasicScheduler
    from comfy_extras.nodes_advanced_samplers import SamplerLCM

    sampler = SamplerLCM.execute(**noise_options)[0]
    sigmas = BasicScheduler.execute(
        model=model, scheduler=scheduler, steps=steps, denoise=1.0 if denoise is None else denoise
    )[0]
    return sample_custom(
        model=model,
        noise=noise,
        cfg=cfg,
        sampler=sampler,
        sigmas=sigmas,
        positive=positive,
        negative=negative,
        latent_image=latent_image,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )


def sample_omnigen2(
    *,
    model,
    noise,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    denoise,
    noise_options=None,
    noise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    """Native OmniGen2 T2I dual guider; CFG controls cfg1 and native cfg2 stays 2."""
    from comfy import model_management
    from comfy_extras.nodes_custom_sampler import BasicScheduler, DualCFGGuider, KSamplerSelect

    if sampler_name == "lcm":
        from comfy_extras.nodes_advanced_samplers import SamplerLCM

        sampler = SamplerLCM.execute(**(noise_options or {}))[0]
    else:
        sampler = KSamplerSelect.execute(sampler_name=sampler_name)[0]
    sigmas = BasicScheduler.execute(
        model=model, scheduler=scheduler, steps=steps, denoise=1.0 if denoise is None else denoise
    )[0]
    # The official T2I graph shares the negative node between these two inputs.
    # Keep its native guider instead of simplifying away custom CFG/CFGNorm hooks.
    guider = DualCFGGuider.execute(
        model=model,
        cond1=positive,
        cond2=negative,
        negative=negative,
        cfg_conds=cfg,
        cfg_cond2_negative=2.0,
        style="regular",
    )[0]
    samples = guider.sample(
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    return samples.to(
        device=model_management.intermediate_device(), dtype=model_management.intermediate_dtype()
    )
