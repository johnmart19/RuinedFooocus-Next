"""Native bundled Stable Cascade C→B→StageA workflow on the existing worker."""

import math
import sys
from dataclasses import dataclass

from ruinedfooocus.features.models.model_identity import cached_safetensors_header, file_signature

STAGE_B_RECIPE = dict(
    cascade_b_steps=10,
    cascade_b_cfg=1.1,
    cascade_b_sampler="euler_ancestral",
    cascade_b_scheduler="simple",
)


@dataclass(frozen=True)
class CascadeLatents:
    """The paired native C/B grids; the B grid belongs to this source encode."""

    stage_c: dict
    stage_b: dict
    control_hint: object | None = None


def source_settings(data):
    compression = data.get("cascade_compression", 42)
    denoise = data.get("cascade_source_denoise", 0.75)
    if type(compression) is not int or not 4 <= compression <= 128:
        raise ValueError("Enter Cascade compression from 4 through 128.")
    if type(denoise) not in (int, float) or not math.isfinite(denoise) or not 0 <= denoise <= 1:
        raise ValueError("Enter Cascade source denoise from zero through one.")
    return compression, float(denoise)


def is_source_request(data):
    module = sys.modules.get("ruinedfooocus.features.image_editing.controlnet")
    controls = module.get_settings(data) if module is not None else data.get("controlnet") or {}
    control = controls.get("type", "")
    if data.get("inpaint_toggle"):
        raise ValueError("Cascade source conditioning requires an image without a mask.")
    if data.get("input_image") is not None:
        if control not in ("edit", "img2img"):
            raise ValueError(
                "Choose Native image edit or Image to image for Cascade's source image."
            )
        source_framing(data.get("image_resize_mode"))
        return True
    if control not in ("", "None", None):
        raise ValueError("Add a Cascade source image or clear the selected image tool.")
    return False


def source_framing(mode):
    mode = mode or "preserve"
    if mode not in ("preserve", "pad"):
        raise ValueError(
            "Cascade preserves the full source image. Choose Preserve proportions or Fit selected size (pad); cropping is unavailable."
        )
    return mode


def source_pixels(image, data, *, mode, to_tensor):
    """Honor an exact target by fitting the full source before native grid encoding."""
    from PIL import ImageOps
    from ruinedfooocus.features.generation.size_plan import KEY, SizePlan
    from ruinedfooocus.features.image_editing.image_geometry import prepare_reference

    mode = source_framing(mode)
    image = ImageOps.exif_transpose(image)
    plan = data.get(KEY)
    if isinstance(plan, SizePlan) or mode == "pad":
        target = plan.content if isinstance(plan, SizePlan) else (data["width"], data["height"])
        return prepare_reference(image, target, "pad", multiple=8)
    return to_tensor(image), None


def source_latents(vae, pixels, data, *, frame=None):
    """Use native EffNet encoding and retain both source-owned stage grids."""
    compression, denoise = source_settings(data)
    if pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.shape[-1] not in (3, 4):
        raise ValueError("Cascade requires one source image with RGB or RGBA channels.")
    height, width = map(int, pixels.shape[1:3])
    if min(width, height) < max(8, compression):
        raise ValueError("The Cascade source image is too small for this compression.")
    from comfy_extras.nodes_stable_cascade import StableCascade_StageC_VAEEncode

    stage_c, stage_b = StableCascade_StageC_VAEEncode.execute(pixels, vae, compression)
    if stage_c["samples"].ndim != 4 or stage_c["samples"].shape[1] != 16:
        raise ValueError("Cascade Stage C requires its native 16-channel EffNet codec.")
    if stage_b["samples"].ndim != 4 or stage_b["samples"].shape[1] != 4:
        raise ValueError("Cascade Stage B requires its native four-channel latent grid.")
    data["height"], data["width"] = (int(side) * 4 for side in stage_b["samples"].shape[-2:])
    data["native_cascade"] = dict(
        native_encoder="StableCascade_StageC_VAEEncode",
        source_width=frame["source_width"] if frame is not None else width,
        source_height=frame["source_height"] if frame is not None else height,
        native_input_width=width,
        native_input_height=height,
        source_frame=frame,
        output_width=data["width"],
        output_height=data["height"],
        compression=compression,
        source_denoise=denoise,
        stage_c_shape=list(stage_c["samples"].shape),
        stage_b_shape=list(stage_b["samples"].shape),
    )
    return CascadeLatents(stage_c, stage_b)


def stage_b_defaults(name, models):
    """Use a verified source or positive full native companion configuration."""
    from ruinedfooocus.features.models.model_identity import verified_native_source

    path = models.get_file("checkpoints", name) if name else None
    if path is None:
        return {}
    metadata = models.get_models_by_path("checkpoints", name, fetch=False)
    source = verified_native_source(path, metadata)
    if (
        source is not None
        and source.get("native_family") == "Stable_Cascade_B"
        and source.get("behavioral_variant") == "base"
    ):
        return STAGE_B_RECIPE.copy()
    # Menu/default lookup may reuse initialized native detection, never start it.
    if sys.modules.get("comfy.model_detection") is None:
        return {}
    from ruinedfooocus.features.models.model_architecture import inspect_model_file
    from ruinedfooocus.features.models.native_recipes import _VARIANT_KEYS

    try:
        signature = file_signature(path)
        if (
            metadata.get("_rf_stale_native_family")
            or metadata.get("_rf_file_signature", signature) != signature
        ):
            return {}
        contract = inspect_model_file(path)
        config = contract.get("native_config", {})
        if (
            contract.get("native_class") != "Stable_Cascade_B"
            or config.get("stable_cascade_stage") != "b"
            or config.get("c_hidden") != [320, 640, 1280, 1280]
        ):
            return {}
        declarations = [contract.get("metadata", {})]
        if metadata.get("_rf_file_signature") == signature:
            declarations.append(metadata)
        if any(
            isinstance(value := declaration.get(key), str)
            and value.strip()
            and value.strip().casefold() != "base"
            for declaration in declarations
            for key in _VARIANT_KEYS
        ):
            return {}
        require_bundle(path, "b")
        header = cached_safetensors_header(path)
        projections = (
            "text_encoder.clip_g.text_projection",
            "text_encoder.clip_g.transformer.text_projection.weight",
        )
        if not any(header.get(key, {}).get("shape") == [1280, 1280] for key in projections):
            return {}
        return STAGE_B_RECIPE.copy() if signature == file_signature(path) else {}
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return {}


def sampling_settings(data):
    from comfy.samplers import KSampler

    steps, cfg = data.get("cascade_b_steps"), data.get("cascade_b_cfg")
    sampler, scheduler = data.get("cascade_b_sampler"), data.get("cascade_b_scheduler")
    if type(steps) not in (int, float) or not 1 <= steps <= 200 or int(steps) != steps:
        raise ValueError("Enter Stage B steps from 1 through 200.")
    if type(cfg) not in (int, float) or not math.isfinite(cfg) or not 0 <= cfg <= 20:
        raise ValueError("Enter Stage B CFG from 0 through 20.")
    if sampler not in KSampler.SAMPLERS or scheduler not in KSampler.SCHEDULERS:
        raise ValueError("Choose Stage B's native sampler and scheduler.")
    if not data.get("cascade_stage_b"):
        raise ValueError("Select a bundled native Stable Cascade Stage B checkpoint.")
    source_settings(data)
    is_source_request(data)
    return int(steps), float(cfg), sampler, scheduler


def require_bundle(path, stage):
    """Check bounded native bundle markers before any model payload is read."""
    header = cached_safetensors_header(path)
    if not any(key.startswith("text_encoder.clip_g.") for key in header):
        raise ValueError("Stable Cascade requires its native bundled CLIP-G checkpoint.")
    marker = (
        "vae.vquantizer.codebook.weight"
        if stage == "b"
        else "vae.encoder.backbone.1.0.block.0.1.num_batches_tracked"
    )
    if marker not in header:
        raise ValueError(
            "Stable Cascade requires a bundled Stage A codec in B and EffNet codec in C."
        )

    if stage == "b" and header[marker].get("shape") != [8192, 4]:
        raise ValueError(
            "Stage B's bundled Stage A codebook must have the native 8192 by 4 layout."
        )


def decode_stage_b(
    owner, stage_c, data, manager, seed, *, stage_b_latent=None, check_cancelled=lambda: None
):
    from comfy.sd import load_checkpoint_guess_config
    from nodes import CLIPTextEncode, common_ksampler, VAEDecode
    from nodes import CLIPSetLastLayer
    from comfy_extras.nodes_stable_cascade import (
        StableCascade_EmptyLatentImage,
        StableCascade_StageB_Conditioning,
    )
    from ruinedfooocus.features.generation.image_preparation import prepared_asset
    from ruinedfooocus.backends.comfy.text_encoder_warmup import warm_patched_text_encoder
    from ruinedfooocus.backends.comfy.embedding_paths import register_embedding_paths
    from ruinedfooocus.features.prompting.prompt_scheduling import (
        bind_prompt_steps,
        prompt_switch_per_step,
    )
    from node_helpers import conditioning_set_values
    from ruinedfooocus.features.models.embedding_browser import native_embedding_text

    steps, cfg, sampler, scheduler = sampling_settings(data)
    name = data["cascade_stage_b"]
    with prepared_asset(data, "cascade_stage_b", name, manager, "checkpoints") as path:
        check_cancelled()
        require_bundle(path, "b")
        signature = file_signature(path)
        cached = getattr(owner, "cascade_stage_b_cache", None)
        if cached is None or cached[0] != signature:
            model, clip, vae, _ = load_checkpoint_guess_config(
                str(path),
                embedding_directory=register_embedding_paths(manager.model_paths["embedding_path"]),
            )
            if type(getattr(model, "model", None)).__name__ != "StableCascade_B":
                raise ValueError(
                    "The selected second checkpoint is not native Stable Cascade Stage B."
                )
            vae.throw_exception_if_invalid()
            if vae.spacial_compression_decode() != 4:
                raise ValueError("Stage B must contain its native fourfold Stage A codec.")
            owner.cascade_stage_b_cache = (signature, model, clip, vae)
        _, model, clip, vae = owner.cascade_stage_b_cache
        check_cancelled()
        if data.get("clip_skip", 1) > 1:
            clip = CLIPSetLastLayer().set_last_layer(clip, -data["clip_skip"])[0]
        warm_patched_text_encoder(clip)

        def encode(text):
            text = native_embedding_text(
                text,
                manager.model_paths["embedding_path"],
                manager.model_paths["cache_path"] / "embeddings",
            )
            return CLIPTextEncode().encode(clip, text)[0]

        positive = encode(data["positive_prompt"])
        negative = encode(data["negative_prompt"])
        schedule = prompt_switch_per_step(data["positive_prompt"], steps)
        if schedule:
            encoded, positive = {}, []
            for index, text in enumerate(schedule):
                if text not in encoded:
                    encoded[text] = encode(text)
                positive += conditioning_set_values(
                    encoded[text],
                    {
                        "start_percent": index / len(schedule),
                        "end_percent": (index + 1) / len(schedule),
                        "_rf_prompt_step": index,
                    },
                )
        positive = StableCascade_StageB_Conditioning.execute(positive, stage_c)[0]
        latent = stage_b_latent
        if latent is None:
            latent = StableCascade_EmptyLatentImage.execute(
                data["width"], data["height"], source_settings(data)[0], 1
            )[1]
        if tuple(latent["samples"].shape) != (1, 4, data["height"] // 4, data["width"] // 4):
            raise ValueError("The Cascade Stage B grid does not match the captured source size.")
        positive = bind_prompt_steps(positive, model, steps, sampler, scheduler, None)
        check_cancelled()
        samples = common_ksampler(
            model, seed, steps, cfg, sampler, scheduler, positive, negative, latent
        )[0]
        check_cancelled()
        pixels = VAEDecode().decode(vae, samples)[0]
        check_cancelled()
        if file_signature(path) != signature:
            raise ValueError("Stage B changed while loading. Submit the request again.")
        data["cascade_stage_b_source"] = dict(
            name=name,
            signature=signature,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
        )
        return pixels
