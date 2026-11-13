"""Real application pipeline adapters for ``validate_comfy_workflows.py``.

Example spec entry: ``{"entrypoint": "tools/comfy_feature_app.py:image_edit",
"kwargs": {"family": "mage", "diffusion_model": "models/checkpoints/model.safetensors",
"text_encoder": "models/clip/encoder.safetensors", "vae_model": "models/vae/vae.safetensors",
"source_image": "input.png", "output_png": "outputs/app.png"}}``.

Run the paired JSON spec with ``python tools/validate_comfy_workflows.py case.json --report outputs/comparison.json``.
Importing this module does not initialize Torch or load weights. These adapters
exercise load_base_model/load_loras/process, without the UI worker's transforms.
Sampling/decode wrappers observe the original calls; they never replace results.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


@contextmanager
def _values(mapping, updates):
    missing = object()
    old = {key: mapping.get(key, missing) for key in updates}
    mapping.update(updates)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is missing:
                mapping.pop(key, None)
            else:
                mapping[key] = value


def _local(filename):
    return Path(filename).resolve(strict=True)


@contextmanager
def _catalog(files, control_type=None, controlnet=None):
    """Temporarily expose local assets through the application's own catalogs."""
    import ruinedfooocus.app.state as shared

    with _values(
        shared.models.model_dirs,
        {
            category: list(shared.models.model_dirs[category]) + [path.parent]
            for category, path in files.items()
        },
    ):
        if controlnet is None:
            yield
        else:
            entry = shared.path_manager.DOWNLOADABLE_FILES["cn_" + control_type]
            if controlnet.name != entry["filename"]:
                raise ValueError(
                    "ControlNet filename must match its real cn_" + control_type + " catalog entry"
                )
            with _values(shared.path_manager.paths, {entry["path"]: str(controlnet.parent)}):
                yield


def _input(source_image, width, height):
    from PIL import Image, ImageOps

    with Image.open(source_image) as source:
        # Exact-size RGB input avoids an adapter-induced resize discrepancy:
        # process still performs its own real input preprocessing.
        return (
            ImageOps.exif_transpose(source)
            .convert("RGB")
            .resize((width, height), Image.Resampling.LANCZOS)
        )


def _gen(
    source_image,
    positive,
    negative,
    seed,
    steps,
    cfg,
    sampler,
    scheduler,
    width,
    height,
    model_name,
    controlnet,
):
    return {
        "input_image": _input(source_image, width, height),
        "positive_prompt": positive,
        "negative_prompt": negative,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler,
        "scheduler": scheduler,
        "clip_skip": 1,
        "width": width,
        "height": height,
        "base_model_name": model_name,
        "controlnet": controlnet,
        "task_id": "comfy-feature-validation",
        "automatic_upscale": "Off",
        "inpaint_toggle": False,
    }


def _run(owner, gen_data, output_png, control=False, inpaint=False):
    import ruinedfooocus.features.generation.image_pipeline as app
    from tools.comfy_feature_reference import _save

    observed = {}
    sample = app.comfy.sample.sample
    decode = owner.decode_latent

    def observe_sample(*args, **kwargs):
        observed["conditioning"] = {"positive": kwargs["positive"], "negative": kwargs["negative"]}
        if inpaint:
            observed["initial_latent"] = kwargs["latent_image"].clone()
            observed["noise_mask"] = kwargs["noise_mask"].clone()
        if control:
            net = kwargs["positive"][0][1]["control"]
            observed["control_image"] = net.cond_hint_original.movedim(1, -1)
        return sample(*args, **kwargs)

    def observe_decode(samples, **options):
        observed["latent"] = samples["samples"]
        images = decode(samples, **options)
        observed["output"] = images
        return images

    with (
        patch.object(app.comfy.sample, "sample", observe_sample),
        patch.object(owner, "decode_latent", observe_decode),
    ):
        owner.process(gen_data=gen_data, callback=None)
    if control:
        observed["conditioning"] = {
            side: [
                [
                    entry[0],
                    {
                        key: value
                        for key, value in entry[1].items()
                        if key not in ("control", "control_apply_to_uncond")
                    },
                ]
                for entry in entries
            ]
            for side, entries in observed["conditioning"].items()
        }
    _save(observed["output"], output_png)
    return observed


def sdxl_controlnet(
    checkpoint,
    controlnet,
    source_image,
    output_png,
    positive="a photo",
    negative="",
    seed=1,
    steps=20,
    cfg=7.0,
    sampler="euler",
    scheduler="normal",
    width=512,
    height=512,
    strength=1.0,
    start=0.0,
    stop=1.0,
    canny=True,
    low_threshold=0.4,
    high_threshold=0.8,
    lora=None,
    lora_strength=1.0,
    control_type=None,
):
    """Run the application's actual SDXL control route with explicit settings."""
    from ruinedfooocus.features.generation.image_pipeline import pipeline

    checkpoint, controlnet = _local(checkpoint), _local(controlnet)
    files = {"checkpoints": checkpoint}
    if lora:
        files["loras"] = _local(lora)
    if control_type is None:
        import ruinedfooocus.features.image_editing.controlnet as controls

        matches = [
            kind for kind, name in controls.controlnet_models.items() if name == controlnet.name
        ]
        if len(matches) != 1:
            raise ValueError("Specify a catalog ControlNet model and control_type")
        control_type = matches[0]
    if canny != (control_type == "canny"):
        raise ValueError("canny must agree with the selected application control_type")
    with _catalog(files, control_type, controlnet):
        owner = pipeline()
        owner.load_base_model(str(checkpoint))
        owner.load_loras([{"name": str(files["loras"]), "weight": lora_strength}] if lora else [])
        gen_data = _gen(
            source_image,
            positive,
            negative,
            seed,
            steps,
            cfg,
            sampler,
            scheduler,
            width,
            height,
            str(checkpoint),
            {
                "type": control_type,
                "strength": strength,
                "start": start,
                "stop": stop,
                "edge_low": low_threshold,
                "edge_high": high_threshold,
            },
        )
        artifacts = _run(owner, gen_data, output_png, control=True)
    artifacts["settings"] = {
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler": sampler,
        "scheduler": scheduler,
        "width": width,
        "height": height,
        "strength": strength,
        "start": start,
        "stop": stop,
        "canny": canny,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "lora_strength": lora_strength if lora else None,
    }
    return artifacts


def sdxl_inpaint(
    checkpoint,
    source_image,
    mask_image,
    output_png,
    positive="a red ceramic teapot",
    negative="",
    seed=1,
    steps=20,
    cfg=7.0,
    sampler="euler",
    scheduler="normal",
    width=1024,
    height=1024,
    grow_mask_by=20,
    denoise=1.0,
):
    """Exercise production layer-alpha inpainting with the real sampler/decode."""
    import numpy as np
    from PIL import Image
    from ruinedfooocus.features.generation.image_pipeline import pipeline

    checkpoint = _local(checkpoint)
    source_image = _local(source_image)
    with Image.open(source_image) as source:
        if source.mode != "RGB" or source.size != (width, height):
            raise ValueError(
                "Use an RGB source at the requested size for this explicit inpaint recipe"
            )
    with Image.open(mask_image) as mask:
        if mask.mode != "RGBA" or mask.size != (width, height):
            raise ValueError(
                "Inpaint mask must be an RGBA layer at the source size; alpha selects the edit"
            )
        layer = np.asarray(mask).copy()
    with _catalog({"checkpoints": checkpoint}):
        owner = pipeline()
        owner.load_base_model(str(checkpoint))
        owner.load_loras([])
        gen_data = _gen(
            str(source_image),
            positive,
            negative,
            seed,
            steps,
            cfg,
            sampler,
            scheduler,
            width,
            height,
            str(checkpoint),
            {},
        )
        gen_data.update(
            inpaint_toggle=True,
            main_view=str(source_image),
            inpaint_view={"layers": [layer]},
            inpaint_denoise=denoise,
            inpaint_mask_grow=grow_mask_by,
        )
        artifacts = _run(owner, gen_data, output_png, inpaint=True)
    artifacts["settings"] = {
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler": sampler,
        "scheduler": scheduler,
        "width": width,
        "height": height,
        "grow_mask_by": grow_mask_by,
        "denoise": denoise,
    }
    return artifacts


def image_edit(
    family,
    diffusion_model,
    text_encoder,
    vae_model,
    source_image,
    output_png,
    positive="change the object to red",
    negative="",
    seed=1,
    steps=20,
    cfg=4.0,
    sampler="euler",
    scheduler="normal",
    width=512,
    height=512,
    clip_type=None,
    weight_dtype="default",
    edit_format="ai-toolkit",
    image_strength=1.0,
    lora=None,
    lora_strength=1.0,
    sampling_shift=None,
    cfg_norm_strength=None,
    moss_slg=False,
    diagnostics=False,
):
    """Run real image-edit processing with a controlled component recipe.

    Optional shift, CFGNorm and MOSS controls use the production request fields.
    Omitted shifts retain the production default; MOSS normally uses shift 5.
    """
    import ruinedfooocus.app.state as shared

    families = {
        "qwen": ("QwenImage", "clip_qwen25", "vae_qwen_image", "qwen_image"),
        "mage": ("MageFlow", "clip_qwen3vl_4b", "vae_mage_flow", "mage"),
        "moss": ("MOSSImage", "clip_qwen35_9b", "vae_flux2", "moss_image"),
    }
    if family not in families:
        raise ValueError("family must be qwen, mage, or moss")
    architecture, clip_key, vae_key, native_clip_type = families[family]
    if clip_type not in (None, native_clip_type) or weight_dtype != "default":
        raise ValueError(
            "The application adapter requires the native family clip_type and default weight_dtype"
        )
    if type(moss_slg) is not bool or (moss_slg and family != "moss"):
        raise ValueError("moss_slg must be a boolean and is only available for MOSS")
    diffusion_model, text_encoder, vae_model = map(
        _local, (diffusion_model, text_encoder, vae_model)
    )
    files = {"checkpoints": diffusion_model}
    if lora:
        files["loras"] = _local(lora)
    updates = {
        clip_key: str(text_encoder),
        vae_key: str(vae_model),
        "moss_image_edit_format": edit_format if family == "moss" else "Disabled",
        "moss_image_edit_strength": image_strength,
        "moss_image_slg": moss_slg,
    }
    with _values(shared.settings.default_settings, updates), _catalog(files):
        from ruinedfooocus.features.generation.image_pipeline import pipeline

        owner = pipeline()
        owner.load_base_model(str(diffusion_model), unet_only=True)
        owner.load_loras([{"name": str(files["loras"]), "weight": lora_strength}] if lora else [])
        gen_data = _gen(
            source_image,
            positive,
            negative,
            seed,
            steps,
            cfg,
            sampler,
            scheduler,
            width,
            height,
            str(diffusion_model),
            {},
        )
        gen_data.update(
            moss_image_edit_format=edit_format if family == "moss" else "Disabled",
            moss_image_edit_strength=image_strength,
            moss_image_slg=moss_slg,
            sampling_shift=sampling_shift,
            cfg_norm_strength=cfg_norm_strength,
        )
        from tools.comfy_sampling_diagnostics import observe_sampling

        with observe_sampling(diagnostics) as sampling_diagnostics:
            artifacts = _run(owner, gen_data, output_png)
        if diagnostics:
            artifacts["diagnostics"] = sampling_diagnostics
    artifacts["settings"] = {
        "family": family,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler": sampler,
        "scheduler": scheduler,
        "width": width,
        "height": height,
        "edit_format": edit_format if family == "moss" else None,
        "image_strength": image_strength if family == "moss" else None,
        "sampling_shift": sampling_shift,
        "cfg_norm_strength": cfg_norm_strength,
        "moss_slg": moss_slg,
        "lora_strength": lora_strength if lora else None,
    }
    return artifacts
