"""Typed native still-image schedules and guiders, with request-owned assets."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
import math
from pathlib import Path, PurePosixPath

from ruinedfooocus.features.models.model_identity import file_signature

KEY = "_native_sampling_plan"
SCHEDULERS = ("Flux2Scheduler", "BetaSamplingScheduler", "Ideogram4Scheduler")
FLUX2 = ("Flux2", "Flux2Klein4B", "Flux2Klein9B")


@dataclass(frozen=True)
class NativeSampling:
    recipe_id: str
    family: str
    scheduler: str
    guider: str = "CFGGuider"
    parameters: tuple[tuple[str, float], ...] = ()
    distilled_guidance: float | None = None
    kv_cache: bool = False
    model_sampling_shift: float | None = None
    identity_basis: str = "explicit_compatible_recipe"

    def metadata(self):
        return dict(
            recipe_id=self.recipe_id,
            scheduler=self.scheduler,
            guider=self.guider,
            parameters=dict(self.parameters),
            distilled_guidance=self.distilled_guidance,
            kv_cache=self.kv_cache,
            model_sampling_default_shift=self.model_sampling_shift,
            identity_basis=self.identity_basis,
        )


def scheduler_choices(ordinary):
    """Expose real node IDs without changing Comfy's ordinary scheduler enum."""
    return sorted(set(ordinary) | set(SCHEDULERS))


def _control(recipe, name):
    rows = [
        tuple(row.get("values", ()))
        for row in recipe.get("source_workflow_controls", ())
        if row.get("node") == name and row.get("mode", 0) == 0
    ]
    if not rows or any(row != rows[0] for row in rows):
        raise ValueError(f"The native recipe needs one consistent {name} contract.")
    return rows[0]


def plan_from_recipe(recipe):
    """Interpret reviewed source controls; readiness and file identity stay separate."""
    scheduler = recipe.get("sampling", {}).get("scheduler")
    family = recipe.get("family")
    if scheduler not in SCHEDULERS:
        return None
    basis = recipe.get("identity_basis", "explicit_compatible_recipe")
    values = dict(recipe_id=recipe["id"], family=family, scheduler=scheduler, identity_basis=basis)
    if scheduler == "Flux2Scheduler" and family in FLUX2:
        _control(recipe, scheduler)
        if family == "Flux2":
            guidance = _control(recipe, "FluxGuidance")
            values.update(guider="BasicGuider", distilled_guidance=float(guidance[0]))
        values["kv_cache"] = recipe.get("variant") == "klein-distilled-9b-kv"
        if values["kv_cache"]:
            _control(recipe, "FluxKVCache")
    elif scheduler == "BetaSamplingScheduler" and family == "ChromaRadiance":
        _, alpha, beta = _control(recipe, scheduler)
        values["parameters"] = (("alpha", float(alpha)), ("beta", float(beta)))
        values["model_sampling_shift"] = float(_control(recipe, "ModelSamplingAuraFlow")[0])
    elif scheduler == "Ideogram4Scheduler" and family == "Ideogram4":
        _, _, _, mu, std = _control(recipe, scheduler)
        values.update(guider="DualModelGuider", parameters=(("mu", float(mu)), ("std", float(std))))
    else:
        raise ValueError("The native schedule does not match this recipe's model family.")
    return NativeSampling(**values)


def apply_options(plan, data):
    parameters = dict(plan.parameters)
    for key, bounds in (
        ("alpha", (0.01, 50)),
        ("beta", (0.01, 50)),
        ("mu", (-10, 10)),
        ("std", (0.1, 5)),
    ):
        value = data.get("native_schedule_" + key)
        if value is None:
            continue
        if key not in parameters:
            raise ValueError(f"{key} is not a parameter of the selected native schedule.")
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not bounds[0] <= value <= bounds[1]
        ):
            raise ValueError(f"Native schedule {key} must be between {bounds[0]} and {bounds[1]}.")
        parameters[key] = float(value)
    return replace(plan, parameters=tuple(parameters.items()))


def valid_scheduler(value, ordinary, plan=None):
    return value in ordinary or isinstance(plan, NativeSampling) and value == plan.scheduler


def capture_assets(data, manager, models):
    """Resolve the current recipe again and capture a secondary UNET before compute."""
    from ruinedfooocus.features.generation.image_preparation import _artifact
    from ruinedfooocus.features.models.native_recipes import current_native_recipe
    from ruinedfooocus.features.image_editing.ordered_native_references import current_source

    result = dict(data)
    supplied = result.pop(KEY, None)
    name = result.get("base_model_name")
    path = models.get_file("checkpoints", name) if models is not None and name else None
    recipe = {}
    if path is not None:
        metadata = models.get_models_by_path("checkpoints", name, fetch=False)
        recipe = current_native_recipe(path, manager, metadata)
    plan = plan_from_recipe(recipe)
    baseline_supplied = (
        replace(supplied, parameters=plan.parameters)
        if isinstance(supplied, NativeSampling) and plan is not None
        else supplied
    )
    if supplied is not None and baseline_supplied != plan:
        raise ValueError(
            "The native sampling recipe changed after selection. Select the model again."
        )
    if plan is None:
        if result.get("scheduler") in SCHEDULERS:
            raise ValueError(
                "Inspect this checkpoint to select its matching native sampling recipe."
            )
        if result.get("native_negative_model") or any(
            result.get("native_schedule_" + key) is not None
            for key in ("alpha", "beta", "mu", "std")
        ):
            raise ValueError(
                "Native sampling parameters require a matching native sampling checkpoint."
            )
        return result, ()
    plan = apply_options(plan, result)
    if (
        not valid_scheduler(result.get("scheduler"), (), plan)
        and result.get("scheduler") in SCHEDULERS
    ):
        raise ValueError("The selected native schedule belongs to a different model family.")
    result[KEY] = plan
    result["native_sampling"] = plan.metadata()
    artifacts = []
    if plan.guider == "DualModelGuider":
        source = current_source(name, models, manager)
        explicit = result.get("native_negative_model")
        key = None
        if not explicit:
            identity = (source or {}).get("component_contract", {}).get("secondary_model")
            from ruinedfooocus.features.models.model_registry import load_registry

            row = load_registry().artifact(identity) if identity else None
            if (
                row is None
                or row.get("role") != "image_diffusion_companion"
                or row.get("native_family") != "Ideogram4"
                or row.get("behavioral_variant") != "unconditional-companion"
            ):
                raise ValueError("Select Ideogram 4's unconditional checkpoint in Native sampling.")
            key = row["catalogue_key"]
            explicit = manager.DOWNLOADABLE_FILES[key]["filename"]
        if not isinstance(explicit, str) or not explicit:
            raise ValueError(
                "Select an unconditional checkpoint from the configured model folders."
            )
        relative = PurePosixPath(explicit.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or ":" in explicit:
            raise ValueError(
                "Select an unconditional checkpoint from the configured model folders."
            )
        explicit = relative.as_posix()
        result["native_negative_model"] = str(explicit)
        item = _artifact(manager, "native-negative-model", "checkpoints", explicit, key)
        if path is not None and item.request.destination.resolve() == Path(path).resolve():
            raise ValueError("Ideogram 4 needs its separate unconditional checkpoint.")
        if item.signature is not None:
            from ruinedfooocus.features.models.model_architecture import inspect_model_file

            if inspect_model_file(item.request.destination).get("native_class") != "Ideogram4":
                raise ValueError("Choose an Ideogram 4 unconditional diffusion checkpoint.")
        artifacts.append(item)
        result["_native_secondary_basis"] = (
            "captured_catalogue_companion" if key else "explicit_unconditional_selection"
        )
    return result, tuple(artifacts)


def negative_model_choices(models, primary):
    """Current bounded native headers, with no payload load or filename inference."""
    from ruinedfooocus.features.models.model_architecture import inspect_model_file
    from ruinedfooocus.features.models.model_identity import verified_native_source

    selected = models.get_file("checkpoints", primary) if primary else None
    selected = Path(selected).resolve() if selected is not None else None
    names = []
    for name in models.get_names("checkpoints"):
        try:
            path = models.get_file("checkpoints", name)
            if path is None or Path(path).resolve() == selected:
                continue
            metadata = models.get_models_by_path("checkpoints", name, fetch=False)
            if inspect_model_file(path).get("native_class") != "Ideogram4":
                continue
            source = verified_native_source(path, metadata)
            if source is not None and source.get("behavioral_variant") != "unconditional-companion":
                continue
            names.append(str(name))
        except (OSError, ValueError, TypeError, AttributeError, IndexError):
            continue
    return names


def require_model(plan, model):
    family = type(model.model).__name__
    if plan.family in FLUX2:
        valid = family == "Flux2"
        depth = model.model.model_config.unet_config.get("depth_single_blocks")
        if plan.family == "Flux2Klein4B":
            valid = valid and depth == 20
        elif plan.family == "Flux2Klein9B":
            valid = valid and depth == 24
        else:
            valid = valid and depth not in (20, 24)
    else:
        valid = family == plan.family
    if not valid:
        raise ValueError(
            "The captured native sampling recipe does not match the loaded checkpoint."
        )


def patch_model(plan, model, *, guidance=None):
    require_model(plan, model)
    if plan.model_sampling_shift is not None and (guidance or {}).get("sampling_shift") is None:
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow

        model = ModelSamplingAuraFlow().patch_aura(model, shift=plan.model_sampling_shift)[0]
    if plan.kv_cache:
        from comfy_extras.nodes_flux import FluxKVCache

        return FluxKVCache.execute(model)[0]
    return model


def condition(plan, positive):
    if plan.distilled_guidance is not None:
        from comfy_extras.nodes_flux import FluxGuidance

        return FluxGuidance.execute(positive, guidance=plan.distilled_guidance)[0]
    return positive


def validate_secondary(primary, secondary):
    if type(secondary.model).__name__ != "Ideogram4":
        raise ValueError("The unconditional checkpoint must be native Ideogram 4.")
    positive_config = primary.model.model_config.unet_config
    negative_config = secondary.model.model_config.unet_config
    if any(
        positive_config.get(key) != negative_config.get(key)
        for key in ("in_channels", "num_layers")
    ):
        raise ValueError("The conditional and unconditional Ideogram 4 configurations differ.")
    dtypes = []
    for model in (primary.model, secondary.model):
        get_dtype = getattr(model, "get_dtype_inference", None)
        dtype = get_dtype() if callable(get_dtype) else None
        allowed = getattr(model.model_config, "supported_inference_dtypes", ())
        if dtype is not None and allowed and dtype not in allowed:
            raise ValueError("The Ideogram 4 model uses an unsupported native inference dtype.")
        dtypes.append(str(dtype).removeprefix("torch.") if dtype is not None else None)
    return dtypes


def sigmas(plan, model, steps, width, height, denoise, scheduler):
    """Use native nodes and native partial-denoise slicing; never create KSampler."""
    from comfy_extras.nodes_custom_sampler import BasicScheduler

    denoise = 1.0 if denoise is None else denoise
    if (
        type(steps) is not int
        or not 1 <= steps <= 200
        or type(denoise) not in (int, float)
        or not math.isfinite(denoise)
        or not 0 <= denoise <= 1
    ):
        raise ValueError(
            "Native sampling needs 1 through 200 steps and denoise from zero through one."
        )
    if denoise == 0:
        import torch

        return torch.empty(0, dtype=torch.float32)
    if scheduler not in SCHEDULERS:
        return BasicScheduler.execute(model, scheduler, steps, denoise)[0]
    if scheduler != plan.scheduler:
        raise ValueError("The native schedule does not match the captured recipe.")
    if denoise < steps / 10000:
        raise ValueError("The native partial-denoise schedule would exceed 10000 steps.")
    total = steps if denoise == 1 else int(steps / denoise)
    options = dict(plan.parameters)
    if scheduler == "Flux2Scheduler":
        from comfy_extras.nodes_flux import Flux2Scheduler

        output = Flux2Scheduler.execute(total, width, height)[0]
    elif scheduler == "BetaSamplingScheduler":
        from comfy_extras.nodes_custom_sampler import BetaSamplingScheduler

        output = BetaSamplingScheduler.execute(model, total, **options)[0]
    else:
        from comfy_extras.nodes_ideogram4 import Ideogram4Scheduler

        output = Ideogram4Scheduler.execute(total, width, height, **options)[0]
    return output[-(steps + 1) :]


@contextmanager
def negative_model_scope(owner, plan, data, manager, check_cancelled):
    if plan.guider != "DualModelGuider":
        yield None
        return
    from ruinedfooocus.features.generation.image_preparation import prepared_asset
    from ruinedfooocus.features.generation.pipeline_utils import load_diffusion_model

    name = data.get("native_negative_model")
    if not name:
        raise ValueError("Select Ideogram 4's unconditional checkpoint before sampling.")
    with prepared_asset(data, "native-negative-model", name, manager, "checkpoints") as path:
        check_cancelled()
        signature = file_signature(path)
        cached = getattr(owner, "native_negative_model_cache", None)
        if cached is None or cached[0] != signature:
            negative = load_diffusion_model(str(path))
            validate_secondary(owner.xl_base_patched.unet, negative)
            owner.native_negative_model_cache = (signature, negative)
        negative = owner.native_negative_model_cache[1]
        dtypes = validate_secondary(owner.xl_base_patched.unet, negative)
        data.setdefault("native_sampling", plan.metadata())["secondary_model"] = dict(
            name=name,
            identity_basis=data.get("_native_secondary_basis", "explicit_unconditional_selection"),
            conditional_inference_dtype=dtypes[0],
            unconditional_inference_dtype=dtypes[1],
        )
        check_cancelled()
        yield negative
        if file_signature(path) != signature:
            raise ValueError("The unconditional checkpoint changed while sampling. Submit again.")


def sample(
    plan,
    *,
    model,
    negative_model,
    noise,
    cfg,
    sampler_name,
    sigmas,
    positive,
    negative,
    latent_image,
    noise_options=None,
    noise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    from comfy import model_management
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider,
        CFGGuider,
        DualModelGuider,
        KSamplerSelect,
    )

    require_model(plan, model)
    if sampler_name == "lcm":
        from comfy_extras.nodes_advanced_samplers import SamplerLCM

        sampler = SamplerLCM.execute(**(noise_options or {}))[0]
    else:
        sampler = KSamplerSelect.execute(sampler_name)[0]
    if plan.guider == "DualModelGuider":
        if negative_model is None:
            raise ValueError("Ideogram 4 requires the captured unconditional checkpoint.")
        guider = DualModelGuider.execute(model, positive, cfg, model_negative=negative_model)[0]
    elif plan.guider == "BasicGuider":
        if cfg != 1:
            raise ValueError("This native Flux.2 recipe uses distilled guidance and sampler CFG 1.")
        guider = BasicGuider.execute(model, positive)[0]
    else:
        guider = CFGGuider.execute(model, positive, negative, cfg)[0]
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
