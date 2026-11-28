"""Direct official-node reference adapters for validate_comfy_workflows.py.

These functions run only when explicitly invoked, with existing local weights.
``sdxl_controlnet`` kwargs: checkpoint, controlnet, source_image, output_png,
positive, negative, seed, steps, cfg, sampler, scheduler, width, height,
strength, start, stop, canny, low_threshold, high_threshold.
"""

from pathlib import Path


def _register(category, filename):
    import folder_paths

    path = Path(filename).resolve(strict=True)
    folder_paths.add_model_folder_path(category, str(path.parent), is_default=True)
    return path.name


def _image(filename, width, height):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    with Image.open(filename) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return torch.from_numpy(np.asarray(image).copy()).float().unsqueeze(0) / 255.0


def _save(images, filename):
    import numpy as np
    from PIL import Image

    target = Path(filename).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(images[0].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)).save(
        target
    )


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
):
    """Canonical SDXL ControlNet node chain, independent of app pipelines.

    Conditioning artifacts contain embedding tensors plus serializable metadata;
    Comfy ControlNet model objects are intentionally represented by explicit
    settings and the control image artifact. Final latent and decoded output are
    captured directly, allowing an app adapter to return matching named values.
    """
    from nodes import (
        CheckpointLoaderSimple,
        CLIPTextEncode,
        ControlNetLoader,
        ControlNetApplyAdvanced,
        EmptyLatentImage,
        KSampler,
        VAEDecode,
        LoraLoader,
    )

    model, clip, vae = CheckpointLoaderSimple().load_checkpoint(
        _register("checkpoints", checkpoint)
    )
    if lora:
        model, clip = LoraLoader().load_lora(
            model, clip, _register("loras", lora), lora_strength, lora_strength
        )
    pos = CLIPTextEncode().encode(clip, positive)[0]
    neg = CLIPTextEncode().encode(clip, negative)[0]
    image = _image(source_image, width, height)
    if canny:
        from comfy_extras.nodes_canny import Canny

        image = Canny.execute(image, low_threshold, high_threshold)[0]
    control = ControlNetLoader().load_controlnet(_register("controlnet", controlnet))[0]
    pos, neg = ControlNetApplyAdvanced().apply_controlnet(
        pos, neg, control, image, strength, start, stop, vae=vae
    )
    latent = EmptyLatentImage().generate(width, height, 1)[0]
    result = KSampler().sample(
        model, seed, steps, cfg, sampler, scheduler, pos, neg, latent, denoise=1.0
    )[0]
    images = VAEDecode().decode(vae, result)[0]
    _save(images, output_png)
    # Never return live model references to the generic fingerprint serializer.
    conditioning = {
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
        for side, entries in (("positive", pos), ("negative", neg))
    }
    return {
        "latent": result["samples"],
        "conditioning": conditioning,
        "control_image": image,
        "output": images,
        "settings": {
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
        },
    }


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
    """Canonical VAEEncodeForInpaint -> KSampler -> VAEDecode comparison.

    The mask is the RGBA editor layer: its alpha channel selects the edit area.
    Returns the actual encoded starting latent and expanded sampling mask too.
    """
    import numpy as np
    import torch
    from PIL import Image
    from nodes import (
        CheckpointLoaderSimple,
        CLIPTextEncode,
        VAEEncodeForInpaint,
        KSampler,
        VAEDecode,
    )

    with Image.open(source_image) as source:
        if source.mode != "RGB" or source.size != (width, height):
            raise ValueError(
                "Use an RGB source at the requested size for this explicit inpaint recipe"
            )
    with Image.open(mask_image) as layer:
        if layer.mode != "RGBA" or layer.size != (width, height):
            raise ValueError(
                "Inpaint mask must be an RGBA layer at the source size; alpha selects the edit"
            )
        mask = (
            torch.from_numpy(np.asarray(layer.getchannel("A")).copy()).float().unsqueeze(0) / 255.0
        )
    model, clip, vae = CheckpointLoaderSimple().load_checkpoint(
        _register("checkpoints", checkpoint)
    )
    pos = CLIPTextEncode().encode(clip, positive)[0]
    neg = CLIPTextEncode().encode(clip, negative)[0]
    pixels = _image(source_image, width, height)
    latent = VAEEncodeForInpaint().encode(vae, pixels, mask, grow_mask_by)[0]
    initial_latent, noise_mask = latent["samples"].clone(), latent["noise_mask"].clone()
    result = KSampler().sample(
        model, seed, steps, cfg, sampler, scheduler, pos, neg, latent, denoise=denoise
    )[0]
    images = VAEDecode().decode(vae, result)[0]
    _save(images, output_png)
    return {
        "latent": result["samples"],
        "initial_latent": initial_latent,
        "noise_mask": noise_mask,
        "conditioning": {"positive": pos, "negative": neg},
        "output": images,
        "settings": {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "width": width,
            "height": height,
            "grow_mask_by": grow_mask_by,
            "denoise": denoise,
        },
    }


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
    """Official Qwen/Mage/MOSS edit encoders with split local components.

    Supply clip_type explicitly when the model variant requires a different
    encoder. This reference intentionally applies no application preset sampling
    patches; a workflow needing one must add the matching official node.
    """
    from nodes import UNETLoader, CLIPLoader, VAELoader, LoraLoader, KSampler, VAEDecode
    from comfy_extras.nodes_sd3 import EmptySD3LatentImage

    types = {"qwen": "qwen_image", "mage": "mage", "moss": "moss_image"}
    if family not in types:
        raise ValueError("family must be qwen, mage, or moss")
    if type(moss_slg) is not bool or (moss_slg and family != "moss"):
        raise ValueError("moss_slg must be a boolean and is only available for MOSS")
    model = UNETLoader().load_unet(_register("diffusion_models", diffusion_model), weight_dtype)[0]
    clip = CLIPLoader().load_clip(
        _register("text_encoders", text_encoder), clip_type or types[family]
    )[0]
    vae = VAELoader().load_vae(_register("vae", vae_model))[0]
    if lora:
        model, clip = LoraLoader().load_lora(
            model, clip, _register("loras", lora), lora_strength, lora_strength
        )
    if moss_slg:
        from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple

        model = SkipLayerGuidanceDiTSimple.execute(
            model=model, double_layers="", single_layers="22", start_percent=0.0, end_percent=0.5
        )[0]
    if sampling_shift is not None:
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow

        model = ModelSamplingAuraFlow().patch_aura(model, sampling_shift)[0]
    if cfg_norm_strength is not None:
        from comfy_extras.nodes_cfg import CFGNorm

        model = CFGNorm.execute(model, cfg_norm_strength)[0]
    image = _image(source_image, width, height)
    if family == "mage":
        from comfy_extras.nodes_mage import TextEncodeMageFlowEdit

        pos, neg, latent = TextEncodeMageFlowEdit.execute(
            clip=clip,
            prompt=positive,
            negative_prompt=negative,
            vae=vae,
            images={"image_1": image},
            width=width,
            height=height,
        ).result
    else:
        if family == "qwen":
            from comfy_extras.nodes_qwen import TextEncodeQwenImageEdit

            pos = TextEncodeQwenImageEdit.execute(clip, positive, vae=vae, image=image)[0]
            neg = TextEncodeQwenImageEdit.execute(clip, negative, vae=vae, image=image)[0]
        else:
            from comfy_extras.nodes_moss_image import TextEncodeMOSSImageEdit

            pos = TextEncodeMOSSImageEdit.execute(
                clip,
                positive,
                edit_format,
                {"image_1": image},
                vae=vae,
                image_strength=image_strength,
            )[0]
            neg = TextEncodeMOSSImageEdit.execute(
                clip,
                negative,
                edit_format,
                {"image_1": image},
                vae=vae,
                image_strength=image_strength,
            )[0]
        if family == "moss":
            from comfy_extras.nodes_flux import EmptyFlux2LatentImage

            latent = EmptyFlux2LatentImage.execute(width, height, 1)[0]
        else:
            latent = EmptySD3LatentImage.execute(width, height, 1)[0]
    from tools.comfy_sampling_diagnostics import observe_sampling

    with observe_sampling(diagnostics) as sampling_diagnostics:
        result = KSampler().sample(
            model, seed, steps, cfg, sampler, scheduler, pos, neg, latent, denoise=1.0
        )[0]
    images = VAEDecode().decode(vae, result)[0]
    _save(images, output_png)
    artifacts = {
        "latent": result["samples"],
        "conditioning": {"positive": pos, "negative": neg},
        "output": images,
        "settings": {
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
        },
    }
    if diagnostics:
        artifacts["diagnostics"] = sampling_diagnostics
    return artifacts
