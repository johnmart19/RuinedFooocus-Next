"""Captured SDXL base/refiner pairing and native noisy-latent schedule handoff."""

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath

from ruinedfooocus.features.models.model_identity import file_signature

KEY = "_sdxl_refiner_plan"
FIELDS = ("sdxl_refiner_model", "sdxl_refiner_switch_step", "sdxl_refiner_cfg", "sdxl_refiner_clip")


@dataclass(frozen=True)
class RefinerPlan:
    base_name: str
    model_name: str
    switch_step: int
    steps: int
    cfg: float | None
    clip_name: str | None


def _name(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Select an installed SDXL refiner checkpoint.")
    name = PurePosixPath(value.replace("\\", "/"))
    if name.is_absolute() or ".." in name.parts or ":" in value:
        raise ValueError("Select the refiner from the configured model folders.")
    return name.as_posix()


def settings(data):
    # The worker derives runtime steps from the frozen custom_steps control.
    # Admission runs earlier; only normalized direct callers may lack that key.
    steps, split, cfg = (
        data.get("custom_steps", data.get("steps")),
        data.get("sdxl_refiner_switch_step"),
        data.get("sdxl_refiner_cfg"),
    )
    if type(steps) is not int or not 2 <= steps <= 200:
        raise ValueError("SDXL staging requires 2 through 200 total steps.")
    if type(split) is not int or not 1 <= split < steps:
        raise ValueError("Enter a refiner switch step between one and the total steps minus one.")
    if cfg is not None and (
        type(cfg) not in (int, float) or not math.isfinite(cfg) or not 0 <= cfg <= 100
    ):
        raise ValueError("Enter refiner CFG from zero through 100, or leave it blank.")
    return steps, split, None if cfg is None else float(cfg)


def capture_assets(data, manager, models, values=None):
    """Freeze both checkpoints and the refiner's actual encoder before compute."""
    from ruinedfooocus.core.preparation import PreparationRequest
    from ruinedfooocus.features.generation.image_preparation import ImageArtifact, _artifact
    from ruinedfooocus.features.models.component_candidates import validate_inventory
    from ruinedfooocus.features.models.component_profiles import component_inventory
    from ruinedfooocus.features.models.image_components import ALIASES

    result = dict(data)
    supplied = result.pop(KEY, None)
    result.pop("sdxl_refiner_source", None)
    controls = result.get("controlnet") or {}
    if controls.get("type", "").lower() in ("upscale", "rembg", "detect", "segment", "depth-map"):
        return result, ()
    if not result.get("sdxl_refiner_model"):
        return result, ()
    if str(result.get("image_engine", (values or {}).get("image_engine", "auto"))).startswith(
        "sdcpp-"
    ):
        raise ValueError("SDXL base/refiner staging requires the native image engine.")
    steps, split, cfg = settings(result)
    if models is None:
        raise ValueError("Capture the refiner with the installed image-model inventory.")
    base = _name(result.get("base_model_name"))
    name = _name(result["sdxl_refiner_model"])
    paths = [models.get_file("checkpoints", value) for value in (base, name)]
    if any(path is None for path in paths):
        raise FileNotFoundError("Install both selected SDXL checkpoints before submitting staging.")
    if Path(paths[0]).resolve() == Path(paths[1]).resolve():
        raise ValueError("SDXL staging requires separate base and refiner checkpoints.")
    families = [models.detect_checkpoint_base(path) for path in paths]
    if ALIASES.get(families[0], families[0]) != "SDXL" or families[1] != "SDXLRefiner":
        raise ValueError(
            "Pair an ordinary SDXL base checkpoint with a native SDXLRefiner checkpoint."
        )
    if any(Path(path).suffix.casefold() != ".safetensors" for path in paths):
        raise ValueError("SDXL staging currently requires native safetensors checkpoints.")
    overrides = {"clip_g": result.get("sdxl_refiner_clip") or None}
    inventory = component_inventory(paths[1], "SDXLRefiner", values or {}, manager, overrides)
    rows = [row for row in inventory["rows"] if row["key"] == "clip_g"]
    if len(rows) != 1:
        raise ValueError("The SDXL refiner requires its native single CLIP-G encoder.")
    validate_inventory(dict(rows=rows))
    row = rows[0]
    artifacts = [
        _artifact(manager, "sdxl-refiner-base", "checkpoints", base),
        _artifact(manager, "sdxl-refiner-model", "checkpoints", name),
    ]
    clip_name = None
    if row["status"] != "Embedded in checkpoint":
        location = Path(row["location"]).resolve()
        key = row.get("download_key")
        if not location.is_file() and not row.get("download_available"):
            raise FileNotFoundError("Install the refiner's selected CLIP-G encoder before staging.")
        request = (
            manager.preparation_request(key, destination=location)
            if key
            else PreparationRequest("", location)
        )
        clip_name = row["filename"]
        artifacts.append(
            ImageArtifact(
                "sdxl-refiner-clip",
                "clip_g",
                request,
                tuple(file_signature(location)) if location.is_file() else None,
            )
        )
    plan = RefinerPlan(base, name, split, steps, cfg, clip_name)
    if supplied is not None and supplied != plan:
        raise ValueError("The captured SDXL refiner selection changed. Submit the request again.")
    result.update({KEY: plan, "sdxl_refiner_model": name, "sdxl_refiner_clip": clip_name})
    return result, tuple(artifacts)


def validate_pair(base, refiner):
    """Native model/prediction/latent contracts decide compatibility, not filenames."""
    if type(base.model).__name__ != "SDXL" or type(refiner.model).__name__ != "SDXLRefiner":
        raise ValueError("SDXL staging requires native SDXL base and SDXLRefiner runtimes.")
    for model in (base.model, refiner.model):
        if model.model_config.unet_config.get("in_channels") != 4:
            raise ValueError("SDXL staging requires ordinary four-channel base/refiner models.")
    if base.model.model_type != refiner.model.model_type:
        raise ValueError("The SDXL base and refiner prediction types differ.")
    sampling = [model.get_model_object("model_sampling") for model in (base, refiner)]
    for key in ("calculate_input", "calculate_denoised", "noise_scaling", "inverse_noise_scaling"):
        methods = [getattr(value, key, None) for value in sampling]
        if any(not callable(method) for method in methods) or (
            getattr(methods[0], "__func__", methods[0])
            is not getattr(methods[1], "__func__", methods[1])
        ):
            raise ValueError("The active SDXL base and refiner prediction contracts differ.")
    if any(
        getattr(sampling[0], key, None) != getattr(sampling[1], key, None)
        for key in ("sigma_data", "noise_scale")
    ):
        raise ValueError("The active SDXL base and refiner noise scales differ.")
    left, right = base.model.latent_format, refiner.model.latent_format
    if type(left) is not type(right) or any(
        getattr(left, key, None) != getattr(right, key, None)
        for key in ("latent_channels", "scale_factor", "shift_factor")
    ):
        raise ValueError("The SDXL base and refiner latent formats differ.")


@contextmanager
def refiner_scope(owner, plan, data, manager, check_cancelled):
    from comfy.sd import load_checkpoint_guess_config, load_clip, CLIPType
    from ruinedfooocus.features.generation.image_preparation import prepared_asset
    from ruinedfooocus.backends.comfy.embedding_paths import register_embedding_paths
    from ruinedfooocus.backends.comfy.text_encoder_loader import has_gguf_encoder, load_mixed_clip

    with ExitStack() as stack:
        sources = {}
        roles = [
            ("sdxl-refiner-base", plan.base_name, "checkpoints"),
            ("sdxl-refiner-model", plan.model_name, "checkpoints"),
        ]
        if plan.clip_name:
            roles.append(("sdxl-refiner-clip", "clip_g", "clip"))
        for role, name, category in roles:
            check_cancelled()
            path = stack.enter_context(prepared_asset(data, role, name, manager, category))
            sources[role] = (path, tuple(file_signature(path)))
        base_signature = getattr(owner, "base_file_signature", None)
        if base_signature is None or tuple(base_signature) != sources["sdxl-refiner-base"][1]:
            raise ValueError("The loaded SDXL base differs from its captured source. Submit again.")
        signature = tuple(
            sources[role][1]
            for role in ("sdxl-refiner-model", "sdxl-refiner-clip")
            if role in sources
        )
        cached = getattr(owner, "sdxl_refiner_cache", None)
        if cached is None or cached[0] != signature:
            embeddings = register_embedding_paths(manager.model_paths["embedding_path"])
            refiner, clip, _, _ = load_checkpoint_guess_config(
                str(sources["sdxl-refiner-model"][0]),
                output_vae=False,
                output_clip=plan.clip_name is None,
                embedding_directory=embeddings,
            )
            if plan.clip_name is not None:
                paths = [str(sources["sdxl-refiner-clip"][0])]
                clip = (
                    load_mixed_clip(paths, "STABLE_DIFFUSION")
                    if has_gguf_encoder(paths)
                    else load_clip(
                        ckpt_paths=paths,
                        embedding_directory=embeddings,
                        clip_type=CLIPType.STABLE_DIFFUSION,
                    )
                )
            if clip is None:
                raise ValueError("The SDXL refiner's native CLIP-G encoder could not be loaded.")
            validate_pair(owner.xl_base_patched.unet, refiner)
            check_cancelled()
            owner.sdxl_refiner_cache = (signature, refiner, clip)
        _, refiner, clip = owner.sdxl_refiner_cache
        check_cancelled()
        yield (
            refiner.clone(),
            clip,
            {
                role: dict(path=str(path), signature=signature)
                for role, (path, signature) in sources.items()
            },
        )
        for path, signature in sources.values():
            if tuple(file_signature(path)) != signature:
                raise ValueError("An SDXL staging source changed while sampling. Submit again.")


def _conditioning(clip, data, model, sigmas, denoise, manager, prompts):
    from nodes import CLIPSetLastLayer
    from node_helpers import conditioning_set_values
    from ruinedfooocus.backends.comfy.text_encoder_warmup import warm_patched_text_encoder
    from ruinedfooocus.features.image_editing.native_conditioning import refiner_text_conditioning
    from ruinedfooocus.features.models.embedding_browser import native_embedding_text
    from ruinedfooocus.features.prompting.prompt_scheduling import (
        bind_prompt_steps,
        prompt_switch_per_step,
    )

    if data.get("clip_skip", 1) > 1:
        clip = CLIPSetLastLayer().set_last_layer(clip, -data["clip_skip"])[0]
    warm_patched_text_encoder(clip)

    def encode(text, negative=False):
        text = native_embedding_text(
            text,
            manager.model_paths["embedding_path"],
            manager.model_paths["cache_path"] / "embeddings",
        )
        encoded = refiner_text_conditioning(
            clip, text, data["width"], data["height"], negative=negative
        )
        if any(
            tuple(getattr(tensor, "shape", ()))[-1:] != (1280,)
            or tuple(getattr(options.get("pooled_output"), "shape", ()))[-1:] != (1280,)
            for tensor, options in encoded
        ):
            raise ValueError(
                "SDXL refiner conditioning requires native CLIP-G features and pooled output."
            )
        return encoded

    positive_prompt, negative_prompt = prompts
    positive = encode(positive_prompt)
    negative = encode(negative_prompt, True)
    schedule = prompt_switch_per_step(positive_prompt, data["steps"])
    if schedule:
        encoded, positive = {}, []
        for index, text in enumerate(schedule):
            if text not in encoded:
                encoded[text] = encode(text)
            positive += conditioning_set_values(encoded[text], {"_rf_prompt_step": index})
    return bind_prompt_steps(
        positive,
        model,
        data["steps"],
        data["sampler_name"],
        data["scheduler"],
        denoise,
        sigmas=sigmas,
    ), negative


def sample_handoff(
    owner,
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
    data,
    manager,
    check_cancelled,
    denoise=None,
    noise_mask=None,
    noise_options=None,
    prompts=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    """Use native advanced-sampler flags and the identical native sigma schedule."""
    import torch
    from comfy.sample import sample, prepare_empty_noise
    from comfy.samplers import KSampler

    plan = data.get(KEY)
    if not isinstance(plan, RefinerPlan) or (
        plan.base_name,
        plan.model_name,
        plan.steps,
        plan.switch_step,
        plan.cfg,
        plan.clip_name,
    ) != (
        data.get("base_model_name"),
        data.get("sdxl_refiner_model"),
        steps,
        data.get("sdxl_refiner_switch_step"),
        data.get("sdxl_refiner_cfg"),
        data.get("sdxl_refiner_clip"),
    ):
        raise ValueError("Capture this exact SDXL base/refiner selection before generation.")
    settings(data)
    denoise = 1.0 if denoise is None else denoise
    if type(denoise) not in (int, float) or not math.isfinite(denoise) or not 0 <= denoise <= 1:
        raise ValueError("SDXL staging denoise must be between zero and one.")
    if sampler_name not in KSampler.SAMPLERS or scheduler not in KSampler.SCHEDULERS:
        raise ValueError("SDXL staging requires a shared native sampler and scheduler.")
    check_cancelled()
    if denoise == 0:
        data["sdxl_refiner_source"] = dict(
            model=plan.model_name, switch_step=plan.switch_step, skipped_denoise_zero=True
        )
        return latent_image
    if denoise < steps / 10000:
        raise ValueError("The SDXL staging partial-denoise schedule would exceed 10000 steps.")
    with refiner_scope(owner, plan, data, manager, check_cancelled) as (refiner, clip, sources):
        validate_pair(model, refiner)
        schedule = KSampler(
            model, steps, device="cpu", sampler=sampler_name, scheduler=scheduler, denoise=denoise
        ).sigmas
        other = KSampler(
            refiner, steps, device="cpu", sampler=sampler_name, scheduler=scheduler, denoise=denoise
        ).sigmas
        if len(schedule) != steps + 1 or not torch.equal(schedule, other):
            raise ValueError("The SDXL base and refiner native sigma schedules differ.")
        refiner_positive, refiner_negative = _conditioning(
            clip,
            data,
            refiner,
            schedule,
            denoise,
            manager,
            prompts if prompts is not None else (data["positive_prompt"], data["negative_prompt"]),
        )

        def progress(offset):
            def report(step, x0, x, _total):
                check_cancelled()
                if callback is not None:
                    callback(step + offset, x0, x, steps)

            return report

        options = dict(
            steps=steps,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            sigmas=schedule,
            noise_mask=noise_mask,
            disable_pbar=disable_pbar,
            seed=seed,
        )

        def run(
            stage_model, stage_noise, stage_cfg, stage_positive, stage_negative, image, start, end
        ):
            if sampler_name == "lcm":
                from comfy.sample import sample_custom
                from comfy_extras.nodes_advanced_samplers import SamplerLCM

                if noise_options is None:
                    raise ValueError("Capture the selected LCM noise options before SDXL staging.")
                return sample_custom(
                    model=stage_model,
                    noise=stage_noise,
                    cfg=stage_cfg,
                    sampler=SamplerLCM.execute(**noise_options)[0],
                    sigmas=schedule[start : end + 1],
                    positive=stage_positive,
                    negative=stage_negative,
                    latent_image=image,
                    noise_mask=noise_mask,
                    callback=progress(start),
                    disable_pbar=disable_pbar,
                    seed=seed,
                )
            return sample(
                model=stage_model,
                noise=stage_noise,
                cfg=stage_cfg,
                positive=stage_positive,
                negative=stage_negative,
                latent_image=image,
                disable_noise=start > 0,
                start_step=start,
                last_step=end,
                force_full_denoise=end == steps,
                callback=progress(start),
                **options,
            )

        check_cancelled()
        tensor_handoff = (
            data.get("native_export_handoff") or data.get("_native_tensor_handoff") is not None
        )
        if tensor_handoff:
            import hashlib
            import json
            from ruinedfooocus.features.generation.native_tensor_io import sdxl_handoff_state

            mask_hash = None
            if noise_mask is not None:
                mask_hash = hashlib.sha256(
                    noise_mask.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
            handoff_payload = dict(
                switch_step=plan.switch_step,
                steps=steps,
                denoise=denoise,
                seed=seed,
                cfg=cfg,
                refiner_cfg=cfg if plan.cfg is None else plan.cfg,
                sampler=sampler_name,
                scheduler=scheduler,
                sigmas=schedule.tolist(),
                noise_options=noise_options,
                noise_mask_sha256=mask_hash,
                prompts_sha256=hashlib.sha256(
                    json.dumps(
                        prompts
                        if prompts is not None
                        else (data["positive_prompt"], data["negative_prompt"]),
                        ensure_ascii=False,
                    ).encode()
                ).hexdigest(),
                sources={
                    role: [
                        Path(row["signature"][0]).name,
                        row["signature"][1],
                        *(str(n) for n in row["signature"][2:]),
                    ]
                    for role, row in sources.items()
                },
            )
            resume = sdxl_handoff_state(
                owner, data, {"samples": latent_image}, payload=handoff_payload
            )
        else:
            resume = False
        noisy = (
            latent_image
            if resume
            else run(model, noise, cfg, positive, negative, latent_image, 0, plan.switch_step)
        )
        check_cancelled()
        if tensor_handoff and data.get("native_export_handoff"):
            saved_latent = {"samples": noisy}
            if noise_mask is not None:
                saved_latent["noise_mask"] = noise_mask
            sdxl_handoff_state(owner, data, saved_latent, payload=handoff_payload, export=True)
        refined = run(
            refiner,
            prepare_empty_noise(noisy),
            cfg if plan.cfg is None else plan.cfg,
            refiner_positive,
            refiner_negative,
            noisy,
            plan.switch_step,
            steps,
        )
        check_cancelled()
        data["sdxl_refiner_source"] = dict(
            model=plan.model_name,
            clip=plan.clip_name or "embedded",
            sources=sources,
            switch_step=plan.switch_step,
            steps=steps,
            cfg=cfg if plan.cfg is None else plan.cfg,
            sampler=sampler_name,
            scheduler=scheduler,
            seed=seed,
            denoise=denoise,
            sigmas=schedule.tolist(),
            latent_handoff="noisy-native-latent",
            base_return_with_leftover_noise=True,
            refiner_add_noise=False,
        )
        return refined
