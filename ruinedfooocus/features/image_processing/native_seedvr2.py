"""Bounded one-frame SeedVR2 restoration using the pinned official image recipe."""

from contextlib import ExitStack
import math
from pathlib import Path

from ruinedfooocus.features.models.model_identity import file_signature

KIND = "restore-seedvr2"
FIELDS = ("seedvr2_model", "seedvr2_vae", "seedvr2_scale", "seedvr2_color")
DEFAULTS = {"seedvr2_scale": 4.0, "seedvr2_color": "none"}
COLORS = ("none", "lab", "wavelet", "adain")
MAX_PIXELS = 16_777_216
CODEC_KEY = "decoder.up_blocks.2.upsamplers.0.upscale_conv.weight"


def validate_model(path):
    from ruinedfooocus.features.models.utility_models import passive_utility_file

    if (
        Path(path).suffix.lower() not in (".safetensors", ".sft")
        or passive_utility_file(path) != "SeedVR2"
    ):
        raise ValueError("Choose an installed SeedVR2 restoration model.")


def validate_vae(path):
    from ruinedfooocus.features.models.model_identity import cached_safetensors_header

    if Path(path).suffix.lower() not in (".safetensors", ".sft"):
        raise ValueError("Choose the matching native SeedVR2 codec.")
    # This is the discriminator used by the pinned comfy.sd.VAE, before the
    # generic diffusers conversion. It is not an SD/Flux/video codec alias.
    entry = cached_safetensors_header(path).get(CODEC_KEY, {})
    shape = entry.get("shape")
    if (
        entry.get("dtype") not in ("F16", "BF16", "F32")
        or not isinstance(shape, list)
        or len(shape) != 5
        or any(type(size) is not int or size < 1 for size in shape)
    ):
        raise ValueError("Choose the matching native SeedVR2 codec.")


def _roots(manager, codec=False):
    categories = ("vae",) if codec else ("diffusion_models", "checkpoints")
    return tuple(
        dict.fromkeys(
            Path(root).resolve()
            for category in categories
            for root in manager.get_model_folders(category)
        )
    )


def _selected(manager, name, codec=False):
    if not isinstance(name, str) or not name or len(name) > 1024:
        raise ValueError("Choose an installed SeedVR2 model and its matching codec.")
    supplied, roots = Path(name), _roots(manager, codec)
    if ".." in supplied.parts:
        raise ValueError("Choose SeedVR2 files from configured model folders.")
    candidates = (
        (supplied.resolve(),)
        if supplied.is_absolute()
        else tuple((root / supplied).resolve() for root in roots)
    )
    path = next(
        (
            path
            for path in candidates
            if path.is_file() and any(path.is_relative_to(root) for root in roots)
        ),
        None,
    )
    if path is None:
        raise ValueError("Choose SeedVR2 files from configured model folders.")
    return path


def model_choices(manager, codec=False):
    """Header-only local choices; never download or follow a directory outside roots."""
    choices, seen = [], set()
    validator = validate_vae if codec else validate_model
    for root in _roots(manager, codec):
        pending, visited = [root], set()
        while pending:
            folder = pending.pop()
            resolved = folder.resolve()
            if not resolved.is_relative_to(root) or resolved in visited:
                continue
            visited.add(resolved)
            try:
                children = sorted(folder.iterdir())
            except OSError:
                continue
            for path in children:
                if path.is_dir():
                    pending.append(path)
                elif path.is_file() and path.resolve().is_relative_to(root):
                    value = str(path.resolve())
                    if value in seen:
                        continue
                    seen.add(value)
                    try:
                        validator(path)
                    except (OSError, ValueError, TypeError):
                        continue
                    choices.append((path.relative_to(root).as_posix(), value))
    return sorted(choices, key=lambda pair: pair[0].casefold())


def geometry(size, scale):
    """Keep the requested scaled dimensions; remove only artificial native padding."""
    if type(scale) not in (int, float) or not math.isfinite(scale) or not 1 <= scale <= 4:
        raise ValueError("SeedVR2 scale must be a finite number from 1 through 4.")
    width, height = (round(side * scale) for side in size)
    padded = tuple(((side + 15) // 16) * 16 for side in (width, height))
    if min(width, height) < 2 or padded[0] * padded[1] > MAX_PIXELS:
        raise ValueError(
            "SeedVR2 needs at least 2 pixels per side and at most 16 megapixels after padding."
        )
    return (width, height), (width, height), padded


def capture_assets(data, manager):
    from PIL import Image, ImageOps
    from ruinedfooocus.features.generation.image_preparation import selected_image_artifact

    from ruinedfooocus.features.image_processing.utility_defaults import default_components

    result = default_components(manager, KIND, data)
    image = result.get("input_image")
    if not isinstance(image, Image.Image) or image.width * image.height > MAX_PIXELS:
        raise ValueError("Add one still image of at most 16 megapixels for restoration.")
    if result.get("inpaint_toggle"):
        raise ValueError("Turn off inpainting before SeedVR2 restoration.")
    image = ImageOps.exif_transpose(image).convert("RGBA").copy()
    geometry(image.size, result.get("seedvr2_scale", DEFAULTS["seedvr2_scale"]))
    color = result.get("seedvr2_color", DEFAULTS["seedvr2_color"])
    if color not in COLORS:
        raise ValueError("Choose a supported SeedVR2 color correction method.")
    seed = result.get("seed", 0)
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("SeedVR2 seed must be an unsigned 64-bit integer.")
    result.update(
        input_image=image,
        seedvr2_scale=float(result.get("seedvr2_scale", DEFAULTS["seedvr2_scale"])),
        seedvr2_color=color,
        seed=seed,
    )
    artifacts = []
    for key, role, codec in (
        ("seedvr2_model", "seedvr2-model", False),
        ("seedvr2_vae", "seedvr2-vae", True),
    ):
        item = selected_image_artifact(
            manager,
            result.get(key),
            role,
            ("vae",) if codec else ("diffusion_models", "checkpoints"),
            identities=("SeedVR2 codec",) if codec else ("SeedVR2",),
            source_roles=("image_codec",) if codec else ("image_utility_model",),
        )
        path, before = item.request.destination, item.signature
        if before is not None:
            (validate_vae if codec else validate_model)(path)
        if before is not None and before != tuple(file_signature(path)):
            raise ValueError("A SeedVR2 component changed during capture. Choose it again.")
        result[key] = str(path)
        artifacts.append(item)
    return result, tuple(artifacts)


def validate_prepared(data):
    for key, validate in (("seedvr2_model", validate_model), ("seedvr2_vae", validate_vae)):
        path = Path(data[key])
        before = tuple(file_signature(path))
        validate(path)
        if before != tuple(file_signature(path)):
            raise ValueError("A SeedVR2 component changed during preparation. Submit it again.")
    return data


def worker_data(data):
    from ruinedfooocus.features.generation.image_preparation import omit_generation_conditioning

    result = omit_generation_conditioning(data)
    result.pop("_prompt_preparation", None)
    result.pop("_sdcpp_snapshot", None)
    result.pop("_sdcpp_prepared", None)
    _, output, _ = geometry(result["input_image"].size, result["seedvr2_scale"])
    result.update(
        prompt="",
        negative="",
        loras=[],
        style_selection=[],
        base_model_name=Path(result["seedvr2_model"]).name,
        base_model_hash="",
        custom_steps=1,
        cfg=1,
        sampler_name="euler",
        scheduler="simple",
        clip_skip=1,
        image_number=1,
        generate_forever=False,
        aspect_ratios_selection="Custom...",
        custom_width=output[0],
        custom_height=output[1],
        automatic_upscale="Off",
        _sampling_captured=False,
    )
    return result


def clear_model(owner):
    cached = getattr(owner, "_native_seedvr2", None)
    if cached is not None:
        from comfy.model_management import unload_model_and_clones

        owner._native_seedvr2 = None
        for patcher in (cached[1], cached[2].patcher):
            unload_model_and_clones(patcher)


def process(owner, data, manager, check_cancelled):
    """Execute exactly one native still restoration under both captured read leases."""
    import numpy as np
    import torch
    from PIL import Image
    import comfy.sd
    import comfy.utils
    from nodes import KSampler, VAEEncodeTiled, VAEDecodeTiled
    from comfy_extras.nodes_post_processing import ResizeImageMaskNode
    from comfy_extras.nodes_seedvr import (
        SeedVR2Conditioning,
        SeedVR2Preprocess,
        SeedVR2PostProcessing,
    )
    from ruinedfooocus.features.generation.image_preparation import prepared_asset
    from ruinedfooocus.features.image_processing.native_image_formats import advanced_output

    image = data["input_image"].convert("RGBA")
    resized_size, output_size, padded_size = geometry(image.size, data["seedvr2_scale"])
    with ExitStack() as stack:
        model_path = stack.enter_context(
            prepared_asset(
                data, "seedvr2-model", data["seedvr2_model"], manager, "diffusion_models"
            )
        )
        vae_path = stack.enter_context(
            prepared_asset(data, "seedvr2-vae", data["seedvr2_vae"], manager, "vae")
        )
        signature = (tuple(file_signature(model_path)), tuple(file_signature(vae_path)))
        validate_model(model_path)
        validate_vae(vae_path)
        check_cancelled()
        cached = getattr(owner, "_native_seedvr2", None)
        if cached is None or cached[0] != signature:
            clear_model(owner)
            model = comfy.sd.load_diffusion_model(str(model_path))
            if model is None or type(model.model.model_config).__name__ != "SeedVR2":
                if model is not None:
                    from comfy.model_management import unload_model_and_clones

                    unload_model_and_clones(model)
                raise ValueError("The native loader returned an incompatible restoration model.")
            # Record partial ownership so cancellation/codec failure unloads it.
            try:
                check_cancelled()
                state, metadata = comfy.utils.load_torch_file(
                    str(vae_path), safe_load=True, return_metadata=True
                )
                vae = comfy.sd.VAE(sd=state, metadata=metadata)
                vae.throw_exception_if_invalid()
                if (
                    vae.latent_channels != 16
                    or vae.latent_dim != 3
                    or type(vae.first_stage_model).__name__ != "VideoAutoencoderKLWrapper"
                ):
                    raise ValueError("SeedVR2 requires its matching native 16-channel codec.")
            except BaseException:
                from comfy.model_management import unload_model_and_clones

                unload_model_and_clones(model)
                if "vae" in locals():
                    unload_model_and_clones(vae.patcher)
                raise
            owner._native_seedvr2 = (signature, model, vae)
        _, model, vae = owner._native_seedvr2
        check_cancelled()
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255)[None]
        with torch.inference_mode():
            reference = ResizeImageMaskNode.execute(
                pixels,
                "lanczos",
                {"resize_type": "scale by multiplier", "multiplier": data["seedvr2_scale"]},
            )[0]
            if tuple(reference.shape) != (1, resized_size[1], resized_size[0], 4):
                raise ValueError("Native SeedVR2 resize returned invalid still geometry.")
            # Native postprocessing crops to even dimensions. Give it an even
            # reference made ONLY by padding, so no original edge can be lost.
            native_size = tuple((side + 1) // 2 * 2 for side in resized_size)
            reference = torch.nn.functional.pad(
                reference,
                (0, 0, 0, native_size[0] - resized_size[0], 0, native_size[1] - resized_size[1]),
            )
            padded = SeedVR2Preprocess.execute(reference)[0]
            if tuple(padded.shape) != (1, 1, padded_size[1], padded_size[0], 3):
                raise ValueError("Native SeedVR2 preprocessing returned more than one frame.")
            check_cancelled()
            latent = VAEEncodeTiled().encode(vae, padded, 512, 128, 4096, 8)[0]
            if tuple(latent["samples"].shape[:3]) != (1, 16, 1):
                raise ValueError("The SeedVR2 codec returned an invalid one-frame latent.")
            check_cancelled()
            positive, negative = SeedVR2Conditioning.execute(model, latent)
            output = KSampler().sample(
                model,
                data["seed"],
                1,
                1.0,
                "euler",
                "simple",
                positive,
                negative,
                latent,
                denoise=1.0,
            )[0]
            check_cancelled()
            decoded = VAEDecodeTiled().decode(vae, output, 512, 128, 4096, 8)[0]
            restored = SeedVR2PostProcessing.execute(decoded, reference, data["seedvr2_color"])[0]
            if (
                tuple(restored.shape) != (1, native_size[1], native_size[0], 4)
                or not torch.isfinite(restored).all()
            ):
                raise ValueError("Native SeedVR2 restoration returned invalid still pixels.")
            restored = restored[:, : output_size[1], : output_size[0], :]
        check_cancelled()
        if signature != (tuple(file_signature(model_path)), tuple(file_signature(vae_path))):
            raise ValueError("A SeedVR2 component changed. Submit the image again.")
        from ruinedfooocus.core.cache_contract import configured_cache_root
        from ruinedfooocus.features.models.model_identity import cached_file_sha256

        hashes = []
        for path in (model_path, vae_path):
            check_cancelled()
            hashes.append(cached_file_sha256(path, configured_cache_root(manager)))
        check_cancelled()
        if signature != (tuple(file_signature(model_path)), tuple(file_signature(vae_path))):
            raise ValueError("A SeedVR2 component changed while its identity was recorded.")
        data["native_seedvr2_source"] = {
            "native": "SeedVR2",
            "model_role": "image_utility_model",
            "model": Path(model_path).name,
            "codec": Path(vae_path).name,
            "model_bytes": signature[0][1],
            "codec_bytes": signature[1][1],
            "model_sha256": hashes[0],
            "codec_sha256": hashes[1],
            "source_size": list(image.size),
            "resized_size": list(resized_size),
            "native_reference_size": list(native_size),
            "padded_size": list(padded_size),
            "output_size": list(output_size),
            "alpha": "native resized source alpha; remove only artificial even/grid padding",
            "scale": data["seedvr2_scale"],
            "color": data["seedvr2_color"],
            "sampling": {"steps": 1, "cfg": 1, "sampler": "euler", "scheduler": "simple"},
        }
        array = restored[0].detach().cpu().float().numpy().copy()
        if advanced_output(data):
            return array
        return Image.fromarray((np.clip(array, 0, 1) * 255).round().astype(np.uint8))
