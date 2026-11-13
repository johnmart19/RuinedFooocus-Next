"""Local Lotus depth through the official native one-step node workflow."""

from pathlib import Path
from contextlib import ExitStack

from ruinedfooocus.features.models.model_identity import file_signature


def validate_model(path):
    from ruinedfooocus.features.models.model_architecture import inspect_model_file

    if (
        Path(path).suffix.lower() not in (".safetensors", ".sft")
        or inspect_model_file(path).get("native_class") != "LotusD"
    ):
        raise ValueError("Select an installed native LotusD safetensors checkpoint.")


def validate_vae(path):
    from ruinedfooocus.features.models.component_candidates import inspect_local_component

    if (
        Path(path).suffix.lower() not in (".safetensors", ".sft")
        or inspect_local_component("vae_sd", path).get("compatibility") != "compatible"
    ):
        raise ValueError("Select a verified compatible SD VAE for Lotus depth.")


def _roots(manager, category):
    categories = (
        ("checkpoints", "diffusion_models", "geometry_estimation")
        if category == "model"
        else ("vae",)
    )
    return tuple(
        dict.fromkeys(
            Path(root).resolve() for kind in categories for root in manager.get_model_folders(kind)
        )
    )


def selected_file(manager, name, category):
    if (
        not isinstance(name, str)
        or not name
        or Path(name).is_absolute()
        or ".." in Path(name).parts
    ):
        raise ValueError("Choose a Lotus component from its configured local folders.")
    for root in _roots(manager, category):
        path = root / name
        if path.is_file():
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("The Lotus component moved outside its configured folder.")
            return resolved
    raise FileNotFoundError("The selected Lotus component is no longer installed.")


def _local_choices(manager, category, validate):
    choices, names = {}, set()
    for root in _roots(manager, category):
        pending, visited = [root], set()
        while pending:
            directory = pending.pop()
            try:
                resolved = directory.resolve()
                if not resolved.is_relative_to(root) or resolved in visited:
                    continue
                visited.add(resolved)
                children = sorted(directory.iterdir())
            except OSError:
                continue
            for path in children:
                if path.is_dir():
                    pending.append(path)
                elif path.is_file() and path.suffix.lower() in (".safetensors", ".sft"):
                    name = path.relative_to(root).as_posix()
                    if name in names:
                        continue
                    names.add(name)
                    try:
                        local = selected_file(manager, name, category)
                        validate(local)
                        choices[name] = str(local)
                    except (OSError, ValueError):
                        continue
    return choices


def local_models(manager):
    return _local_choices(manager, "model", validate_model)


def local_vaes(manager):
    return _local_choices(manager, "vae", validate_vae)


def prepare_lotus(owner, image, options, data, manager, *, check_cancelled=None):
    import torch
    import comfy.sd
    import comfy.utils
    from comfy.model_management import throw_exception_if_processing_interrupted
    from nodes import VAEEncode, VAEDecode, ImageInvert
    from comfy_extras.nodes_lotus import LotusConditioning
    from comfy_extras.nodes_custom_sampler import (
        BasicScheduler,
        SetFirstSigma,
        KSamplerSelect,
        BasicGuider,
        DisableNoise,
        SamplerCustomAdvanced,
    )
    from ruinedfooocus.features.generation.image_preparation import prepared_asset

    def check():
        throw_exception_if_processing_interrupted()
        if check_cancelled is not None:
            check_cancelled()

    if image.ndim != 4 or image.shape[-1] != 3 or any(size % 8 for size in image.shape[1:3]):
        raise ValueError("Lotus depth requires RGB input dimensions divisible by eight.")
    names = (options["depth_lotus_model"], options["depth_lotus_vae"])
    with (
        prepared_asset(data, "lotus-depth", names[0], manager, "checkpoints") as checkpoint,
        prepared_asset(data, "lotus-vae", names[1], manager, "vae") as codec,
    ):
        if checkpoint != selected_file(manager, names[0], "model") or codec != selected_file(
            manager, names[1], "vae"
        ):
            raise ValueError(
                "The configured Lotus component folders changed. Select the files again."
            )
        validate_model(checkpoint)
        validate_vae(codec)
        signature = (file_signature(checkpoint), file_signature(codec))
        check()
        cached = getattr(owner, "_lotus_depth", None)
        if cached is None or cached[0] != signature:
            from comfy.model_management import unload_model_and_clones

            if cached is not None:
                unload_model_and_clones(cached[1])
                unload_model_and_clones(cached[2].patcher)
                owner._lotus_depth = None
            with ExitStack() as cleanup:
                model = comfy.sd.load_diffusion_model(str(checkpoint))
                if model is not None:
                    cleanup.callback(unload_model_and_clones, model)
                check()
                if model is None or type(model.model.model_config).__name__ != "LotusD":
                    raise ValueError("The native loader did not initialize a LotusD model.")
                state, metadata = comfy.utils.load_torch_file(str(codec), return_metadata=True)
                check()
                vae = comfy.sd.VAE(sd=state, metadata=metadata)
                cleanup.callback(unload_model_and_clones, vae.patcher)
                vae.throw_exception_if_invalid()
                check()
                if vae.latent_channels != 4 or vae.downscale_ratio != 8:
                    raise ValueError(
                        "Lotus depth requires the native four-channel eightfold SD VAE."
                    )
                owner._lotus_depth = (signature, model, vae)
                cleanup.pop_all()
        _, model, vae = owner._lotus_depth
        with torch.inference_mode():
            latent = VAEEncode().encode(vae, image)[0]
            check()
            conditioning = LotusConditioning.execute()[0]
            check()
            guider = BasicGuider.execute(model, conditioning)[0]
            check()
            sigmas = BasicScheduler.execute(model, "normal", 1, 1.0)[0]
            check()
            sigmas = SetFirstSigma.execute(sigmas, 999.0)[0]
            check()
            noise = DisableNoise.execute()[0]
            check()
            sampler = KSamplerSelect.execute("euler")[0]
            check()
            output = SamplerCustomAdvanced.execute(
                noise,
                guider,
                sampler,
                sigmas,
                latent,
            )[0]
            check()
            decoded = VAEDecode().decode(vae, output)[0]
            check()
            depth = ImageInvert().invert(decoded)[0]
        check()
        if tuple(depth.shape) != tuple(image.shape) or not torch.isfinite(depth).all():
            raise ValueError("Lotus depth returned invalid image dimensions or values.")
        if signature != (file_signature(checkpoint), file_signature(codec)):
            raise ValueError("The Lotus depth components changed. Submit the image again.")
        owner.depth_model_name = names[0]
        return depth
